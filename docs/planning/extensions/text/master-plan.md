# Text Extension Master Plan

> **Editorial reading context — not a status update.** The application gives Text a frozen
> vocabulary and strings. Tokenization produces integer labels; Text's batch adapter calls
> Data to pad them and record valid counts. The application, not Text, then connects ordinary
> Tensor IDs to NN Embedding and Engine computation. PAD naming does not freeze a table row
> or mask all later operations. Today's [token identity rules](../../../reference/text/tokenization.md#frozen-token-ids)
> and [Text/Data delegation](../../../reference/text/tokenization.md#explicit-batch-configuration)
> explain these boundaries, including copied vocabulary, fresh outputs and the absence of
> a fluent integer-Embedding-scan execution chain.

## Goal

Add one [frozen whitespace tokenizer](../../../glossary.md#frozen-whitespace-tokenizer) and a fluent adapter that delegates integral [padding](../../../glossary.md#padding) to Data.
Text prepares input; existing [Embedding](../../../glossary.md#embedding) and [Engine](../../../glossary.md#term-engine) perform numerical computation externally.
The user approved this bounded Class C direction on 2026-10-20. Text0001 is Complete, accepted
2026-10-21 after independent code/documentation approvals and the final program join.

## Contracts and common base

- [Root layout, scope index, ownership, dependencies and testing](../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([layout glossary](../../../glossary.md#layout); [scope glossary](../../../glossary.md#term-scope); [ownership glossary](../../../glossary.md#term-ownership-meanings))</span>.
- [Sole owning Text contract](../../../architecture/contracts/extensions-training.md#extensionstext)
  and [shared Data contract](../../../architecture/contracts/extensions-training.md#extensionsdata).
- [ADR 0043](../../../design/decisions/0043-whitespace-text-and-padded-integral-batches.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#term-architecture-decision-record-adr))</span>.
- Common base: `ef35f2ec001b0c161d1abe825ff3be725b9be49e` plus accepted uncommitted
  Data0001/Data0002, enacted Data/Text authority and accepted Data0003 including the iteration fix
  and final documentation, plus accepted uncommitted Text0001; not merged predecessors.

## [Scope](../../../glossary.md#term-scope) and non-goals

Text0001 cohesively delivers `WhitespaceTokenizer` and `TextSequenceBatcher`, tests, [Javadoc](../../../glossary.md#term-javadoc) and
focused [API](../../../glossary.md#term-application-programming-interface-api) documentation. Vocabulary is an [immutable](../../../glossary.md#term-immutable) ordered caller-supplied list; [PAD](../../../glossary.md#frozen-whitespace-tokenizer)=0, [UNK](../../../glossary.md#frozen-whitespace-tokenizer)=1,
ordinary IDs begin at 2. No [normalization](../../../glossary.md#term-normalization), case folding, [BOS](../../../glossary.md#term-bos)/[EOS](../../../glossary.md#term-eos), decoding, [BPE](../../../glossary.md#term-byte-pair-encoding-bpe), vocabulary
training, [persistence](../../../glossary.md#term-persistence), [artifact](../../../glossary.md#term-artifact)/[fingerprint](../../../glossary.md#term-fingerprint) interface, corpus reader or predictor is selected.
Those larger proposals remain deferred Draft and are not prerequisites of this capability.
No [Embedding](../../../glossary.md#embedding) behavior, [scan](../../../glossary.md#term-cumulative-scan) semantics, [runtime](../../../glossary.md#term-runtime)-length [binding](../../../glossary.md#term-binding) or [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt) extension is included.
The [Complete NN0022A](../nn/tasks/0022a-padded-numeric-sequence-inference.md) delivers numeric [F32](../../../glossary.md#floating-point-precision-and-rounding)
inputs, not a complete fluent Text pipeline. Integer-input/Embedding builder support remains a
separate [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) capability decision; exporting Data lengths alone cannot supply it.
Its implemented and independently approved opt-in export adds no Text work or frontier;
main accepted NN0022A after the source-matched architecture/root [checkpoint](../../../glossary.md#term-checkpoint) on 2026-10-21.

## [Module](../../../glossary.md#term-module) invariants and dependencies

Physical `extensions/text`, [Gradle](../../../glossary.md#term-gradle) `:extensions:text`, flat package
`io.github.pho001.synaptik.text`. Exactly `api(project(":extensions:data"))` is permitted in
all configurations; public [Model](../../../glossary.md#term-model) APIs are available through Data's existing export, not a second
direct edge. No Text-to-[NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Engine](../../../glossary.md#term-engine)/[backend](../../../glossary.md#backend) edge or NN-to-Data/Text edge is permitted.
The external integration-test module gets Text only as `testImplementation` and composes its
existing NN/Engine dependencies.

Whitespace means `Character.isWhitespace(int)` over code points, not a regex shorthand.
Encoding preserves original case, punctuation and non-normalized [UTF-16](../../../glossary.md#term-utf-16). A [token ID](../../../glossary.md#frozen-whitespace-tokenizer) equal to [PAD](../../../glossary.md#frozen-whitespace-tokenizer)
is a stored value, not evidence of validity; valid counts and [padding](../../../glossary.md#padding) side belong to Data.
Frozen IDs do not change after [Embedding](../../../glossary.md#embedding) construction. Embedding row zero remains ordinary state.

## Package map

The selected production package contains exactly the two concrete public final types
`WhitespaceTokenizer` and `TextSequenceBatcher`, plus the latter's nested Builder. No [tokenizer](../../../glossary.md#frozen-whitespace-tokenizer)
interface, vocabulary object family, training/[artifact](../../../glossary.md#term-artifact)/batching subpackages or generic pipeline
is reserved without a concrete accepted need.

## [Dependency](../../../glossary.md#term-dependency) [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) and authorized frontiers

Data0001 Complete → Data0002 Complete → Data0003 Complete → Text0001 Complete.
Authorized Text frontier: None. Main accepted Text0001 on 2026-10-21 after independent code and
final documentation approvals and source-matched integration/root gates. The brief owns evidence
and historical launch verification. No automatic successor or parallel writers.

## Task list

| ID | Task | Status | Depends on | Intent/result |
|---|---|---|---|---|
| [0001](tasks/0001-whitespace-tokenizer-and-data-batching.md) | [Frozen whitespace tokenizer](../../../glossary.md#frozen-whitespace-tokenizer) and Data [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) adapter | Complete | Data0003 Complete/accepted; main verification recorded | Accepted [tokenizer](../../../glossary.md#frozen-whitespace-tokenizer)/adapter, registration and enforcement after independent code/docs approvals and numerical/root gates. |
| 0002 | Deterministic vocabulary/tokenizer training | Draft | 0001; separate selected algorithm and approval | Deferred; [mutable](../../../glossary.md#term-mutable) observation remains separate from frozen [runtime](../../../glossary.md#term-runtime) use. |
| 0003 | Tokenizer [artifact](../../../glossary.md#term-artifact) [persistence](../../../glossary.md#term-persistence)/loading | Draft | 0002; selected complete format and compatibility policy | Deferred; no artifact or [fingerprint](../../../glossary.md#term-fingerprint) in Text0001. |
| 0004 | Explicit corpus-format readers | Draft | 0002; concrete format consumer | Deferred; no universal [dataset](../../../glossary.md#term-dataset)/file loader. |
| 0005 | Additional text batching policies | Draft | 0001; concrete additional need | Basic whitespace adaptation is incorporated into 0001; only additional policy remains deferred. |
| 0006 | Broader [model](../../../glossary.md#term-model)/artifact integration | Draft | 0003; separately selected runtime/[checkpoint](../../../glossary.md#term-checkpoint) consumer | Text0001's direct [Engine](../../../glossary.md#term-engine)/[Embedding](../../../glossary.md#embedding) proof is not a recurrent or persisted-tokenizer workflow. |
| 0007 | Vocabulary extension/state migration | Draft | 0003, 0006; concrete extension consumer | Deferred explicit ID-preserving model/[optimizer](../../../glossary.md#term-optimizer) migration, never silent vocabulary growth. |
| 0008 | High-level predictor convenience | Draft | 0006; separately approved higher-level [owner](../../../glossary.md#term-owner) | No Engine edge is granted to Text. |

## Delivered example

Implemented [API](../../../glossary.md#term-application-programming-interface-api) usage (with `List`, `WhitespaceTokenizer` and `TextSequenceBatcher` imported):

```java
var tokenizer = WhitespaceTokenizer.of(List.of("Hello", "world"));
var batcher = TextSequenceBatcher.builder(tokenizer).leftPadded().batchMajor().build();
var batch = batcher.batch(List.of("Hello world", "missing"));
// INT64 values [[2,3],[0,1]], valid counts [2,1], LEFT, shape [2,2].
```

Unknown text becomes [UNK](../../../glossary.md#frozen-whitespace-tokenizer)=1; Data inserts [PAD](../../../glossary.md#frozen-whitespace-tokenizer)=0 on the left. Choosing `.rightPadded()` changes
the second row to `[1,0]`. A caller builds an existing [Embedding](../../../glossary.md#embedding) with four vocabulary rows and
executes `engine.compute(embedding.forward(batch.values()))`; row zero is gathered normally.
This is not a claim that LEFT [metadata](../../../glossary.md#term-metadata) can be passed to the current right-prefix [scan](../../../glossary.md#term-cumulative-scan).

## Integration [ownership](../../../glossary.md#term-ownership-meanings) and shared documents

Main [coordinates](../../../glossary.md#axis-rank-scalar-vector-and-matrix) shared authority/docs integration, independent review and the final external
integration/root [checkpoint](../../../glossary.md#term-checkpoint). The sole Text worker owns source/tests, [module](../../../glossary.md#term-module) [Javadoc](../../../glossary.md#term-javadoc), registration,
architecture enforcement and external integration plumbing. After that worker stops, independent
targeted review finalized the [API](../../../glossary.md#term-application-programming-interface-api) guide, shared glossary and indexes and
checked the actual diff without rerunning successful Java suites. Both selected briefs [carry](../../../glossary.md#term-carry) the
accepted dirty chain and all seven integration fields; Parallel group None. Main's final
source-matched root test and Javadoc gates passed at the Text program join after implementation,
independent review and external numerical integration; evidence is in Text0001 Result.

## Milestones and risks

Complete: registered Text [module](../../../glossary.md#term-module), concrete frozen [tokenizer](../../../glossary.md#frozen-whitespace-tokenizer), Data delegation and numerical program
join using accepted integral batching. Future training/[artifact](../../../glossary.md#term-artifact)/corpus/predictor work
requires its own concrete selection; its old speculative package/[lifecycle](../../../glossary.md#lifecycle) design is not authority.
Do not infer [normalization](../../../glossary.md#term-normalization) or special-[token](../../../glossary.md#term-token) spellings, reimplement Data [padding](../../../glossary.md#padding), grow IDs silently,
or present [host](../../../glossary.md#term-host) lengths as [runtime](../../../glossary.md#term-runtime) [graph](../../../glossary.md#graph) inputs. Same-size vocabularies can have different ID
meanings; Text0001 makes no [persistence](../../../glossary.md#term-persistence) compatibility claim.
