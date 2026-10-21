# ADR 0043: Whitespace text and padded integral batches

## Status

Accepted — user approval and main authority approval, 2026-10-20. Data and Text are implemented;
the final repository checkpoint is tracked in the task brief. This ADR explains the
[authoritative root](../../../ARCHITECTURE.md), not a separate authority.

## Context and decision drivers

Accepted Data0001/Data0002 provide a narrow Model-only extension and borrowed right-padded host
metadata. The next concrete consumer needs frozen token IDs and left/right host padding, not a
tokenizer training/persistence program. Reusable numeric preparation must stay outside NN and
must not create another execution or storage-lifetime owner.

The drivers are a minimal complete public workflow, explicit padding geometry, immutable ID
meaning, unchanged Embedding semantics, checked bounded allocation and enforceable dependencies.

## Options considered

- Retain right-only metadata and duplicate left padding in Text: leaves two padding owners and
  cannot represent the selected consumer with the shared Data value.
- Add a compatibility family beside the unpublished API: avoids source migration but leaves
  redundant types without a supported-release compatibility need.
- Start with interfaces, BPE/training/artifacts and a general input pipeline: addresses broader
  future needs but adds unselected lifecycle, format and policy decisions.
- Select a concrete frozen whitespace tokenizer plus one Data padding owner and an explicit
  unpublished source migration. This is the accepted option.

## Decision

Data owns `PaddingSide { LEFT, RIGHT }`, `PaddedSequenceBatch` and `LongSequenceBatcher` in its
existing flat package. Migrate `RightPaddedSequenceBatch` directly, without an alias. Existing
metadata geometry/borrowing behavior survives; lengths become position-independent counts, with
side selecting valid prefix or suffix. Axis order remains explicit and independent of side.

The integral batcher requires explicit padding value, side and order, pads long[] rows to batch
maximum without truncation, and creates one contiguous non-gradient INT64 heap Tensor through
public Model allocation. Check element geometry incrementally before batch-sized metadata/output;
the element carrier limit is Integer.MAX_VALUE and bytes use checked long arithmetic, not an int
byte cap. Fill the allocated destination directly, with no full-size staging/import recopy.
Callers coordinate mutation during synchronous batching; no transactional snapshot is promised.

Text gets exactly `api(project(":extensions:data"))`; Model is transitive, not a direct edge.
The flat concrete `WhitespaceTokenizer` copies ordered ordinary tokens, reserves PAD=0/UNK=1,
assigns IDs from 2 and splits code-point runs using Character.isWhitespace(int). Preserve case,
punctuation and exact non-normalized UTF-16. `TextSequenceBatcher` encodes and delegates padding
once to Data. Neither Text nor Data depends on NN/Engine/backends; NN gains no Data/Text edge.

No normalization, BOS/EOS, decoding, BPE, vocabulary training, artifact/fingerprint or persistence
is selected. Frozen ordinary vocabulary strings have no magic special-token spelling. Broader
float/sample Data and training/artifact Text plans remain deferred Draft.

## Rationale and scenario

For vocabulary `["Hello", "world"]`, encoding `"Hello world"` gives `[2,3]`, while `"missing"`
gives `[1]`. Left/batch-major adaptation yields values `[[2,3],[0,1]]`, counts `[2,1]`
and LEFT. Data owns this geometry; Text owns the ID meaning. Choosing RIGHT yields `[1,0]` for
the second row without changing counts. The [Text API example](../../api/text-api.md#complete-example)
composes these implemented APIs with unchanged Embedding and public Engine.

The external integration project composes unchanged Embedding and public
`engine.compute(embedding.forward(batch.values()))`. Its direct table-row oracle must cover
all side/order pairs and a distinct nonzero PAD row. Moving this numerical test into Text would
require forbidden NN/Engine edges; adding PAD row masking would change Embedding semantics.

## Consequences and migration

The unpublished source migration changes constructors/imports and requires explicit side in the
values Builder. The sole Data0003 worker migrates source/tests and the isolated architecture consumer.
Completed predecessor briefs stay historical. Direct metadata construction still inspects no
storage and admits long-extreme Shapes; allocation restrictions apply only to the batcher.

Text's new module and edge require dependency/namespace fixtures and a Text-only consumer proving
transitive Data/Model exposure. Integration receives Text only as a test dependency. Main owns
serialized shared-document integration, review coordination and the final root gate. The sole
Text0001 worker owns registration/enforcement and external numerical test plumbing; workers own
focused module validation, followed by independent targeted review and shared-doc finalization
after each worker stops.

LEFT metadata is not accepted by the current scan contract: scans remain right-valid-prefix only.
No implicit host-length runtime binding, BPTT extension, native lifetime or execution pipeline is
introduced. Frozen vocabulary is deliberately less capable than a trained/persisted tokenizer;
same-sized vocabularies need not have the same ID mapping, and no persistence compatibility claim
is made here.

## Related documentation

- [Owning Data contract](../../architecture/contracts/extensions-training.md#extensionsdata)
  and [owning Text contract](../../architecture/contracts/extensions-training.md#extensionstext).
- [Module boundaries](../../architecture/module-boundaries.md#extensionsdata).
- [Data0003](../../planning/extensions/data/tasks/0003-padded-integral-sequence-batching.md)
  and [Text0001](../../planning/extensions/text/tasks/0001-whitespace-tokenizer-and-data-batching.md).
- [ADR 0042](0042-data-extension-boundary.md) remains the historical Data-only foundation decision;
  this decision extends its selected capability and separately authorizes Text.
