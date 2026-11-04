# Data Extension Master Plan

> **Editorial reading context — not a status update.** Data prepares numerical inputs; it
> does not call a model on the application's behalf. Wrapping an existing Tensor borrows
> its reference; host counts are copied when immutable length metadata is created. Batching
> long arrays instead allocates and fills new values. Logical time/batch order is a declaration,
> not a transpose. Padding counts describe valid positions without installing a global mask
> on later NN operations.
> The current [Data mental model](../../../reference/data/sequences.md#purpose-and-mental-model)
> and [explicit count export](../../../reference/data/sequences.md#explicit-runtime-scan-lengths)
> explain how the application passes ordinary values and lengths onward, without adding a Data-owned resource lifetime.

## Goal

[Prepare](../../../glossary.md#prepare) numeric [host](../../../glossary.md#term-host) sequences outside [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) computation. The delivered [slice](../../../glossary.md#slice) pads caller
`long[]` rows into one [INT64](../../../glossary.md#data-type--datatype) [Tensor](../../../glossary.md#tensor) with [immutable](../../../glossary.md#term-immutable) valid counts, explicit side and logical [axes](../../../glossary.md#term-axis).
Broader floating-sequence and equal-[shape](../../../glossary.md#shape) [sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) batching remain deferred, not cancelled.

Data0001–0003 are Complete, accepted locally on 2026-10-20. Current code supplies immutable valid
counts, explicit axes/side, [borrowed](../../../glossary.md#ownership-borrowing-and-lease) `PaddedSequenceBatch` construction and `LongSequenceBatcher`
allocation. Data0003's brief owns final review and [validation](../../../glossary.md#term-validation) evidence.

## Contracts and common base

- [Root scope index, layout, ownership, dependency and testing rules](../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../glossary.md#term-scope); [layout glossary](../../../glossary.md#layout); [ownership glossary](../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../glossary.md#term-dependency))</span>.
- [Sole owning Data contract](../../../architecture/contracts/extensions-training.md#extensionsdata).
- [Text contract](../../../architecture/contracts/extensions-training.md#extensionstext) governs the downstream adapter.
- [ADR 0043](../../../design/decisions/0043-whitespace-text-and-padded-integral-batches.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#term-architecture-decision-record-adr))</span>
  records user approval on 2026-10-20.
- Common base: `ef35f2ec001b0c161d1abe825ff3be725b9be49e` plus accepted uncommitted
  Data0001/Data0002, enacted Data/Text authority and accepted Data0003 including the iteration fix
  and final documentation, plus accepted uncommitted Text0001; no predecessor is claimed merged.

## [Scope](../../../glossary.md#term-scope) and non-goals

Selected: shared LEFT/RIGHT representation, unchanged [O(B)](../../../glossary.md#memory-management-atomic-counters-and-complexity) [host](../../../glossary.md#term-host)-count [API](../../../glossary.md#term-application-programming-interface-api), and a fluent
`LongSequenceBatcher` using public [Model](../../../glossary.md#term-model) [heap](../../../glossary.md#term-heap) allocation. There is no implicit [truncation](../../../glossary.md#term-truncation-toward-zero).
Data never tokenizes, owns [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) computation, runs [Engine](../../../glossary.md#term-engine), allocates native resources, or adds a
storage [lifetime](../../../glossary.md#term-lifetime), [runtime](../../../glossary.md#term-runtime)-length [binding](../../../glossary.md#term-binding), loader/sampler/[epoch](../../../glossary.md#term-epoch-meanings) [lifecycle](../../../glossary.md#lifecycle) or [persistence](../../../glossary.md#term-persistence).
Current scans still accept right-valid-prefix counts only; LEFT support is preparation [metadata](../../../glossary.md#term-metadata),
not a [scan](../../../glossary.md#term-cumulative-scan) capability change. The implemented NN0022A export only materializes counts for
RIGHT/TIME_MAJOR [batches](../../../glossary.md#term-batch-machine-learning-and-hyperparameters); binding remains owned by the application.

## [Module](../../../glossary.md#term-module) invariants and dependencies

Data retains exactly `api(project(":modules:model"))`, including tests. Text's separately approved
sole Data [API](../../../glossary.md#term-application-programming-interface-api) edge makes Text a downstream consumer; [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) gains no Data/Text edge.
Counts are independent of position: RIGHT means valid prefix, LEFT means valid suffix.
Stored values never determine validity; [padding](../../../glossary.md#padding) lengths and dense [masks](../../../glossary.md#term-boolean-mask) are not second sources
of truth. Generic [metadata](../../../glossary.md#term-metadata) construction borrows [Tensor](../../../glossary.md#tensor) references without storage inspection;
the integral batcher allocates fresh [contiguous](../../../glossary.md#term-contiguous) [heap](../../../glossary.md#term-heap) values. NN0022A additionally authorizes
explicit fresh heap count export, without changing metadata construction or accessors.

## Package map

All types stay in `io.github.pho001.synaptik.data`:

- Current `ValidSequenceLengths` and `SequenceAxisOrder` remain, with count terminology updated.
- Data0003 replaced `RightPaddedSequenceBatch` with `PaddedSequenceBatch` and its nested Builder.
- Data0003 added `PaddingSide` and `LongSequenceBatcher` with its nested Builder.
- No compatibility [alias](../../../glossary.md#aliasing-and-arena), reserved subpackage or generic utility/loader [facade](../../../glossary.md#term-facade).

## [Dependency](../../../glossary.md#term-dependency) [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) and authorized frontiers

Data0001 → Data0002 → Data0003 → Text0001.
Data0003 → deferred Data0005 (floating sequences) and Data0006 (equal-[shape](../../../glossary.md#shape) [samples](../../../glossary.md#term-sample-machine-learning-and-hyperparameters));
Data0003 + Data0005 + Data0006 → Data0004 (broader batching [checkpoint](../../../glossary.md#term-checkpoint)).
Data0001–0003 + NN0022/0022B/0028/0029 Complete →
[NN0022A numeric inference](../nn/tasks/0022a-padded-numeric-sequence-inference.md)<span data-glossary-navigation="true"> ([inference glossary](../../../glossary.md#term-inference))</span> Complete:
accepted 2026-10-21 after independent APPROVE and main's architecture/root checkpoint.
The approved Data-owned `PaddedSequenceBatch.toScanLengthsTensor()` export has no [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-to-Data edge.
The NN0022A cross-area frontier is closed; no separate Data successor is opened.
Its brief owns all seven integration fields, verified base `6eabea0c1ebf77b9bd8382855c7131eb89b52e04`,
serial conflicts and main's integration [ownership](../../../glossary.md#term-ownership-meanings). Shared cutover documents have one exclusive writer.

Authorized Data frontier: None. Data0003 is Complete after final independent APPROVE and local
acceptance on 2026-10-20. The [Text program join](../text/tasks/0001-whitespace-tokenizer-and-data-batching.md#result)
completed on 2026-10-21; Text0001 is Complete and the Text frontier is None. No automatic successor.

## Task list

| ID | Task | Status | Depends on | Intent/result |
|---|---|---|---|---|
| [0001](tasks/0001-data-extension-boundary.md) | Data-only architecture/build foundation | Complete | None | Accepted empty [module](../../../glossary.md#term-module), exact [Model](../../../glossary.md#term-model) [API](../../../glossary.md#term-application-programming-interface-api) edge and architecture enforcement; historical brief unchanged. |
| [0002](tasks/0002-valid-sequence-lengths-and-right-padded-batch.md) | [Host](../../../glossary.md#term-host) [metadata](../../../glossary.md#term-metadata) and [borrowed](../../../glossary.md#ownership-borrowing-and-lease) [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) builder | Complete | 0001 | Accepted [immutable](../../../glossary.md#term-immutable) host prefixes/logical [axes](../../../glossary.md#term-axis) and right-padded metadata; historical brief unchanged. |
| [0003](tasks/0003-padded-integral-sequence-batching.md) | Shared padded representation and integral batching | Complete | 0001/0002 Complete; main authority approved | Accepted after independent APPROVE and module/architecture-consumer gates; Text program root join complete. |
| 0005 | Floating sequence batching | Draft | 0003; selected concrete float consumer/API | Preserve broader numeric capability; no speculative [dtype](../../../glossary.md#data-type--datatype)/[truncation](../../../glossary.md#term-truncation-toward-zero) abstraction now. |
| 0006 | Equal-[shape](../../../glossary.md#shape) numeric [sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) batching | Draft | 0003; selected sample consumer/API | Preserve generic numeric stacking for later Vision use without image semantics. |
| 0004 | Broader Data batching [checkpoint](../../../glossary.md#term-checkpoint) | Draft | 0003, 0005, 0006 | Validate the broader sample/sequence capability after selection, not a prerequisite of Text0001. |

## Integration [ownership](../../../glossary.md#term-ownership-meanings) and shared documents

Main owns shared root/scoped authority, explanations, [API](../../../glossary.md#term-application-programming-interface-api) references, glossary, roadmap, master
plans and final integration plumbing. NN0022A's serial independent documentation/review pass
returned frozen source/docs to main; the source-matched architecture/root [checkpoint](../../../glossary.md#term-checkpoint) passed.
No Java executor remains active. The completed Data/Text program's ownership is recorded below.
All seven integration fields live in each selected brief; Parallel group is None.
The sole Data worker owns architecture consumer/[dependency](../../../glossary.md#term-dependency)-test migration; the sole Text worker
owns registration, Text enforcement and external [integration test](../../../glossary.md#term-integration-test) plumbing. Independent review
finalized shared glossary/index/public API docs after the worker stopped. Main completed the
source-matched architecture/Text–[Embedding](../../../glossary.md#embedding) integration and final root checkpoint at the Text program join,
reusing successful Data worker and focused architecture-consumer evidence.
This [planning](../../../glossary.md#term-planning) pass runs no
tests, builds, agents, [commits](../../../glossary.md#commit-release-distribution-log-and-seed-meanings) or pushes.

## Milestones

- Complete: foundation, migrated shared [padding](../../../glossary.md#padding)-side [metadata](../../../glossary.md#term-metadata) and [INT64](../../../glossary.md#data-type--datatype) batching.
- Complete in Text: frozen whitespace [tokenization](../../../glossary.md#frozen-whitespace-tokenizer), delegating Text adapter and program join.
- Deferred: floating sequences, numeric [samples](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) and broader batching [checkpoint](../../../glossary.md#term-checkpoint).

## Open decisions and risks

No architectural decision or local acceptance gap remains for Data0003; independent review and
main acceptance are complete. Broader float/[sample](../../../glossary.md#term-sample-machine-learning-and-hyperparameters) APIs, arbitrary [masks](../../../glossary.md#term-boolean-mask) and
[binding](../../../glossary.md#term-binding) protocols remain future decisions; the narrow opt-in count export is approved under
[ADR0044](../../../design/decisions/0044-opt-in-scan-length-tensor-export.md). Training's former Data-[lifecycle](../../../glossary.md#lifecycle) proposal
is not satisfied by batching. Early checked geometry must reject enormous virtual lists before
[batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-sized snapshots or output allocation; caller-coordinated mutation avoids a transactional
[snapshot](../../../glossary.md#term-snapshot) promise. Completed briefs remain historical evidence, not current architecture authority.
