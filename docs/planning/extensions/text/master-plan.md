# Text Extension Master Plan

## Goal

Plan text-specific input preparation outside `extensions/nn`: deterministic vocabulary training,
immutable tokenizer runtime state, durable tokenizer artifacts, reversible decoding where
supported, and conversion of token sequences into the future Data extension's right-padded
sequence batch.

Mental model:

```text
String values
  -> Tokenizer
  -> token-id sequences
  -> text sequence batcher
  -> Data-owned Tensor batch + valid sequence lengths
  -> typed NN Model input
```

This is future planning only. `extensions/text` is not yet authorized by `ARCHITECTURE.md`, present
in Gradle settings, or implemented.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Planning guide](../../planning-guide.md)
- [Implementation roadmap](../../roadmap.md)
- [Data master plan](../data/master-plan.md)
- [NN master plan](../nn/master-plan.md)

## Scope

- explicit tokenizer and immutable vocabulary contracts;
- deterministic dynamic vocabulary construction through a separate mutable trainer/builder that
  produces a frozen runtime vocabulary;
- versioned tokenizer-artifact read/write with bounded validation and a stable fingerprint;
- normalization and special-token policy owned by a concrete tokenizer;
- token encoding and supported decoding without Tensor or NN knowledge in the tokenizer core;
- text-specific right-padded batching that derives validity metadata automatically; and
- later application-level prediction convenience only after the core boundaries and Engine are
  stable.

## Out of scope

- embedding parameters or NN layers;
- model topology, gradients, optimizers, training sessions, compiler/runtime/backend behavior;
- a universal tokenizer algorithm, mandatory subword scheme, model download, remote vocabulary,
  implicit plugin registry, or catch-all corpus/file parser;
- automatic vocabulary growth after an Embedding or compatible checkpoint is created;
- padding inside `Model.forward`, inference of padding from token value, or a padding-row mutation
  policy for `Embedding`; and
- a core NN dependency on Text.

## Module invariants

- A tokenizer owns text normalization, vocabulary identity, token-to-ID mapping, supported
  decoding, and the meaning of special tokens such as unknown, beginning/end, and padding.
- Tokenizer output is ordinary host token data. It does not create NN parameters, choose an
  embedding size, run a model, or define sequence recurrence.
- The text batcher derives valid lengths from encoded sequences before padding. Users do not type,
  synchronize, or pass a raw length array to ordinary model code.
- Padding token identity and padding validity are separate facts. The padding token supplies a
  stored value; the Data-owned valid lengths say how many leading positions are valid. A token
  equal to the padding ID inside a valid prefix is governed by tokenizer policy and is never
  reinterpreted by scanning the Tensor after batching.
- `extensions/nn` never depends on Text. A typed `Model<I, O>` can accept a Data-owned batch or an
  application record without importing tokenizer behavior.
- Vocabulary training/building is separate from immutable runtime tokenization. Once built or
  loaded, token IDs and special-token IDs are frozen before Embedding construction.
- A tokenizer artifact persists the algorithm and schema versions, normalization policy, exact
  token-ID order, special-token IDs, and algorithm data such as byte-pair-encoding merge order. A
  token list alone is not sufficient.
- Artifact loading validates the complete candidate, computes or verifies its canonical
  fingerprint, and publishes no partially accepted tokenizer. Equal vocabulary sizes do not imply
  compatibility; a different token-to-ID mapping must produce a mismatch.
- Corpus-format readers supply text observations to a tokenizer trainer. They neither parse the
  tokenizer artifact format nor participate in `Model.forward`.

## Proposed dependencies

Subject to the shared architecture task:

```text
modules/model
  -> extensions/data
  -> extensions/text
```

Text may depend on Data for the sequence batch and valid-length metadata and on Model only where
its concrete batcher must create token Tensors. It must not depend on NN, Training, Compiler,
Runtime, Prepare, Engine, or concrete backends. A later predictor convenience may live in a
separate higher-level owner if it needs Engine; it must not broaden the tokenizer foundation's
dependency.

## Package structure

Proposed after architecture acceptance:

```text
io.github.pho001.synaptik.text/
  tokenization/  immutable tokenizer, vocabulary, special-token, encoding, and decoding contracts
  training/      deterministic vocabulary/tokenizer training and freeze boundaries
  artifact/      versioned bounded tokenizer artifact codec, fingerprint, and atomic file I/O
  batching/      text-to-token-sequence and right-padded Data-batch adaptation
  corpus/        later explicit corpus-format readers, separate from tokenizer artifact parsing
```

Do not add a generic `pipeline`, `processor`, `service`, `manager`, or model facade to the initial
module.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | Immutable tokenizer, vocabulary, and special-token foundation | Draft | Data 0001 architecture boundary | Define immutable text normalization, token vocabulary, special-token, encode/decode, and canonical fingerprint roles without Tensor, NN, network, corpus-format, or file-I/O coupling. |
| 0002 | Deterministic vocabulary and tokenizer training | Draft | 0001; one selected tokenizer algorithm | Add a separate mutable trainer/builder that observes caller-supplied text/token streams and freezes an immutable runtime artifact using fixed special IDs, frequency ordering, deterministic tie-breaking, size/frequency limits, and algorithm-specific state. |
| 0003 | Tokenizer artifact persistence and loading | Draft | 0001–0002; selected bounded codec and durable-publication policy | Persist and restore the complete tokenizer artifact with format/algorithm versions, normalization, special IDs, exact token-ID order, optional merge rules, checksums, resource limits, and a canonical fingerprint; validate fully before returning an immutable tokenizer. |
| 0004 | Explicit corpus-format readers | Draft | 0002; concrete line/CSV/JSON consumer | Add only selected bounded corpus readers that emit text observations; keep them separate from tokenizer-artifact parsing, batching, model definition, and an all-purpose dataset loader. |
| 0005 | Text right-padding batch adapter | Draft | 0001; Data 0002–0003 | Encode caller Strings, apply explicit truncation/special-token policy, invoke focused Data batching, and return a Data-owned token Tensor batch whose canonical valid lengths are derived automatically. |
| 0006 | Text model and tokenizer-checkpoint integration checkpoint | Draft | 0003, 0005; NN functional Model/lazy layers/genuine recurrent scan/runtime valid-length integration; Checkpoint model-input fingerprint boundary; required Engine execution capability for runnable claims | Validate build-or-load tokenizer flows, exact tokenizer fingerprint compatibility, and typed String-to-token-batch-to-Embedding/LSTM/classifier composition without specializing the graph to one batch's lengths. |
| 0007 | Explicit vocabulary extension and state migration | Draft | 0003, 0006; concrete extension consumer; Training checkpoint state | Preserve every existing token ID, append new IDs only, expand Embedding rows and every affected optimizer slot explicitly, produce a new fingerprint, and reject silent in-place tokenizer growth. |
| 0008 | High-level text predictor convenience | Draft | 0006; stable Engine/inference consumer | Reassess one optional application-level tokenizer + batcher + compiled-model convenience without moving text into NN or creating a broad pipeline facade. |

No detailed Text task exists. The first detailed task is created only after the architecture task
is accepted and becomes the ordered implementation frontier.

## Vocabulary and artifact lifecycle

The planned lifecycle keeps mutable observation state out of inference:

```text
corpus reader or caller Strings
  -> selected TokenizerTrainer / VocabularyBuilder
  -> deterministic freeze
  -> immutable TokenizerArtifact + fingerprint
  -> immutable runtime Tokenizer
```

Special tokens receive fixed declared IDs first. Other IDs are assigned by the selected
algorithm's documented deterministic order; a frequency-based vocabulary uses descending
frequency and a stable Unicode/code-point tie-break. Minimum frequency and maximum size are
explicit. The artifact records the exact resulting ID order rather than relying on rebuilding it
from counts.

Loading follows a distinct path:

```text
tokenizer artifact bytes
  -> bounded parse and schema/algorithm validation
  -> duplicate, contiguous-ID, special-token, and algorithm-state validation
  -> checksum/fingerprint verification
  -> immutable TokenizerArtifact
  -> immutable runtime Tokenizer
```

These flows are planned and non-runnable. Neither flow constructs an Embedding. An application
first freezes or loads the tokenizer, then uses its vocabulary size and fingerprint when it
defines the model. Automatic observation or ID growth after that point is forbidden. A future
extension operation must preserve old IDs and explicitly migrate Embedding and optimizer state.

Artifact writing uses a sibling temporary target, bounded deterministic encoding, checksums and a
manifest, durable flush behavior defined for the selected filesystem, then one validated rename.
The implementation task must state whether an unavailable atomic rename fails or uses a visibly
weaker mode; it must not silently promise crash atomicity.

## Planned end-to-end example

The following is conceptual planned API, not current runnable Synaptik code:

```java
Tokenizer tokenizer = ...;
TextSequenceBatcher batcher = TextSequenceBatcher.builder(tokenizer)
        .layout(SequenceLayout.TIME_MAJOR)
        .truncation(Truncation.limit(256))
        .build();

RightPaddedSequenceBatch hostBatch = batcher.batch(List.of(
        "Prvni veta",
        "Druha podstatne delsi veta"));

record TextOutput(Tensor logits) {}

var model = Model.define(topology -> {
    Embedding embedding = topology.addModule(
            "embedding",
            Embedding.initialized(
                    tokenizer.vocabulary().size(), embeddingSize, dataType, random));
    LstmSequence lstm = topology.addModule(
            "lstm",
            LstmSequence.lazy(hiddenSize, true, dataType, random));
    Linear classifier = topology.addModule(
            "classifier",
            Linear.lazy(classCount, true, dataType, random));

    return (RightPaddedSequenceInput input) -> {
        Tensor embedded = embedding.forward(input.values());
        LstmSequenceForwardResult encoded =
                lstm.forward(embedded, input.validLengths());
        return new TextOutput(classifier.forward(encoded.finalHidden()));
    };
});
```

Interpretation:

- tokenizer vocabulary size, embedding size, recurrent hidden size, and class count are genuine
  architecture or text-schema choices;
- batch size, maximum time, valid prefix per row, LSTM input width, and classifier input width are
  derived by batching or lazy binding;
- the batcher, not the model, inserts padding and records canonical valid lengths;
- `hostBatch.validLengths()` is planned immutable host metadata, while the intentionally unnamed
  future binding step supplies both components of conceptual `RightPaddedSequenceInput` as runtime
  Tensors; the example does not pass the host value directly to the recurrent scan;
- zero initial hidden/cell state is inferred by the planned recurrent convenience;
- backward is absent from the model definition because compiler/training owns it; and
- current Synaptik implements none of the shown Text/Data APIs, lazy constructors, runtime-length
  overload, recurrent scan, runtime input binding, or full execution pipeline. The example is not
  runnable until those lifecycle contracts exist. Current recurrent containers instead require
  fully static time-major Tensors, explicit initial states, and a Java `long[]`.

The tokenizer creation hidden by `...` is one of these two planned, non-runnable paths:

```java
TokenizerArtifact artifact = tokenizerTrainer.train(corpusReader.documents());
TokenizerArtifacts.write(tokenizerPath, artifact);
Tokenizer tokenizer = artifact.tokenizer();

TokenizerArtifact loaded = TokenizerArtifacts.read(tokenizerPath);
Tokenizer tokenizer = loaded.tokenizer();
```

The tokenizer artifact remains Text-owned. A model or training checkpoint records and validates
its fingerprint but does not parse tokenizer files or make Checkpoint depend on Text.

## Current status

Draft planning only. The user authorized planning this future program, not architecture edits,
Gradle project creation, or implementation ahead of the active CPU frontier.

## Open questions

- Select the first concrete tokenizer algorithm and the exact deterministic token/merge ordering.
- Select the bounded tokenizer artifact encoding and its maximum bytes, token count, token length,
  and algorithm-state limits.
- Decide whether the optional predictor belongs in Text, Engine, or another application extension
  once the compile/prepare/run facade exists.

## Decisions made

- Tokenizer and batching stay outside NN and Model forward computation.
- Tokenizer owns the padding token; Data owns canonical valid lengths and layout.
- Users do not manually construct sequence lengths for ordinary text batching.
- A dense Boolean mask is derived only for a concrete consumer and is never stored beside valid
  lengths as a second source of truth.
- A predictor is later convenience, not part of the core typed Model or tokenizer foundation.
- Vocabulary/tokenizer training is mutable and separate from immutable runtime use.
- The full tokenizer artifact, not only token Strings, is the persistence unit and owns a canonical
  fingerprint.
- A vocabulary is frozen before Embedding construction. Same-size/different-mapping artifacts are
  incompatible, and later growth requires an explicit ID-preserving model/optimizer migration.
- Tokenizer artifact parsing and corpus-format reading are distinct packages and tasks.

## Risks

- Combining tokenizer, batching, Model, Engine, and prediction in one facade would create a god
  object and reverse otherwise clean dependencies.
- Inferring validity by searching for a padding token would conflate stored token values with
  batch metadata and mishandle valid occurrences.
- Presenting the conceptual runtime `input.validLengths()` call as runnable before input binding
  and recurrent scan exist would hide the exact lifecycle gap that prevents graph reuse today.
- An underspecified tokenizer artifact could restore token Strings while silently changing
  normalization, special IDs, merge order, or token-to-ID mapping.
- A mutable runtime vocabulary could invalidate Embedding row meaning and optimizer slots without
  changing their Shapes enough for ordinary state validation to notice.
- An unbounded artifact or corpus parser could allocate excessive memory or accept corrupt IDs
  before validation.
- Conceptual planned examples can be mistaken for implemented API unless every occurrence stays
  labelled.

## Notes

This master plan is non-authoritative future coordination. The architecture contract wins.
