# Text Extension Master Plan

## Goal

Add one frozen whitespace tokenizer and a fluent adapter that delegates integral padding to Data.
Text prepares input; existing Embedding and Engine perform numerical computation externally.
The user approved this bounded Class C direction on 2026-10-20. Text0001 is Complete, accepted
2026-10-21 after independent code/documentation approvals and the final program join.

## Contracts and common base

- [Root layout, scope index, ownership, dependencies and testing](../../../../ARCHITECTURE.md).
- [Sole owning Text contract](../../../architecture/contracts/extensions-training.md#extensionstext)
  and [shared Data contract](../../../architecture/contracts/extensions-training.md#extensionsdata).
- [ADR 0043](../../../design/decisions/0043-whitespace-text-and-padded-integral-batches.md).
- Common base: `ef35f2ec001b0c161d1abe825ff3be725b9be49e` plus accepted uncommitted
  Data0001/Data0002, enacted Data/Text authority and accepted Data0003 including the iteration fix
  and final documentation, plus accepted uncommitted Text0001; not merged predecessors.

## Scope and non-goals

Text0001 cohesively delivers `WhitespaceTokenizer` and `TextSequenceBatcher`, tests, Javadoc and
focused API documentation. Vocabulary is an immutable ordered caller-supplied list; PAD=0, UNK=1,
ordinary IDs begin at 2. No normalization, case folding, BOS/EOS, decoding, BPE, vocabulary
training, persistence, artifact/fingerprint interface, corpus reader or predictor is selected.
Those larger proposals remain deferred Draft and are not prerequisites of this capability.
No Embedding behavior, scan semantics, runtime-length binding or BPTT extension is included.
The [Complete NN0022A](../nn/tasks/0022a-padded-numeric-sequence-inference.md) delivers numeric F32
inputs, not a complete fluent Text pipeline. Integer-input/Embedding builder support remains a
separate NN capability decision; exporting Data lengths alone cannot supply it.
Its implemented and independently approved opt-in export adds no Text work or frontier;
main accepted NN0022A after the source-matched architecture/root checkpoint on 2026-10-21.

## Module invariants and dependencies

Physical `extensions/text`, Gradle `:extensions:text`, flat package
`io.github.pho001.synaptik.text`. Exactly `api(project(":extensions:data"))` is permitted in
all configurations; public Model APIs are available through Data's existing export, not a second
direct edge. No Text-to-NN/Engine/backend edge or NN-to-Data/Text edge is permitted.
The external integration-test module gets Text only as `testImplementation` and composes its
existing NN/Engine dependencies.

Whitespace means `Character.isWhitespace(int)` over code points, not a regex shorthand.
Encoding preserves original case, punctuation and non-normalized UTF-16. A token ID equal to PAD
is a stored value, not evidence of validity; valid counts and padding side belong to Data.
Frozen IDs do not change after Embedding construction. Embedding row zero remains ordinary state.

## Package map

The selected production package contains exactly the two concrete public final types
`WhitespaceTokenizer` and `TextSequenceBatcher`, plus the latter's nested Builder. No tokenizer
interface, vocabulary object family, training/artifact/batching subpackages or generic pipeline
is reserved without a concrete accepted need.

## Dependency DAG and authorized frontiers

Data0001 Complete → Data0002 Complete → Data0003 Complete → Text0001 Complete.
Authorized Text frontier: None. Main accepted Text0001 on 2026-10-21 after independent code and
final documentation approvals and source-matched integration/root gates. The brief owns evidence
and historical launch verification. No automatic successor or parallel writers.

## Task list

| ID | Task | Status | Depends on | Intent/result |
|---|---|---|---|---|
| [0001](tasks/0001-whitespace-tokenizer-and-data-batching.md) | Frozen whitespace tokenizer and Data batch adapter | Complete | Data0003 Complete/accepted; main verification recorded | Accepted tokenizer/adapter, registration and enforcement after independent code/docs approvals and numerical/root gates. |
| 0002 | Deterministic vocabulary/tokenizer training | Draft | 0001; separate selected algorithm and approval | Deferred; mutable observation remains separate from frozen runtime use. |
| 0003 | Tokenizer artifact persistence/loading | Draft | 0002; selected complete format and compatibility policy | Deferred; no artifact or fingerprint in Text0001. |
| 0004 | Explicit corpus-format readers | Draft | 0002; concrete format consumer | Deferred; no universal dataset/file loader. |
| 0005 | Additional text batching policies | Draft | 0001; concrete additional need | Basic whitespace adaptation is incorporated into 0001; only additional policy remains deferred. |
| 0006 | Broader model/artifact integration | Draft | 0003; separately selected runtime/checkpoint consumer | Text0001's direct Engine/Embedding proof is not a recurrent or persisted-tokenizer workflow. |
| 0007 | Vocabulary extension/state migration | Draft | 0003, 0006; concrete extension consumer | Deferred explicit ID-preserving model/optimizer migration, never silent vocabulary growth. |
| 0008 | High-level predictor convenience | Draft | 0006; separately approved higher-level owner | No Engine edge is granted to Text. |

## Delivered example

Implemented API usage (with `List`, `WhitespaceTokenizer` and `TextSequenceBatcher` imported):

```java
var tokenizer = WhitespaceTokenizer.of(List.of("Hello", "world"));
var batcher = TextSequenceBatcher.builder(tokenizer).leftPadded().batchMajor().build();
var batch = batcher.batch(List.of("Hello world", "missing"));
// INT64 values [[2,3],[0,1]], valid counts [2,1], LEFT, shape [2,2].
```

Unknown text becomes UNK=1; Data inserts PAD=0 on the left. Choosing `.rightPadded()` changes
the second row to `[1,0]`. A caller builds an existing Embedding with four vocabulary rows and
executes `engine.compute(embedding.forward(batch.values()))`; row zero is gathered normally.
This is not a claim that LEFT metadata can be passed to the current right-prefix scan.

## Integration ownership and shared documents

Main coordinates shared authority/docs integration, independent review and the final external
integration/root checkpoint. The sole Text worker owns source/tests, module Javadoc, registration,
architecture enforcement and external integration plumbing. After that worker stops, independent
targeted review finalized the API guide, shared glossary and indexes and
checked the actual diff without rerunning successful Java suites. Both selected briefs carry the
accepted dirty chain and all seven integration fields; Parallel group None. Main's final
source-matched root test and Javadoc gates passed at the Text program join after implementation,
independent review and external numerical integration; evidence is in Text0001 Result.

## Milestones and risks

Complete: registered Text module, concrete frozen tokenizer, Data delegation and numerical program
join using accepted integral batching. Future training/artifact/corpus/predictor work
requires its own concrete selection; its old speculative package/lifecycle design is not authority.
Do not infer normalization or special-token spellings, reimplement Data padding, grow IDs silently,
or present host lengths as runtime graph inputs. Same-size vocabularies can have different ID
meanings; Text0001 makes no persistence compatibility claim.
