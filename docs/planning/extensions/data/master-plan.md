# Data Extension Master Plan

## Goal

Prepare numeric host sequences outside NN computation. The delivered slice pads caller
`long[]` rows into one INT64 Tensor with immutable valid counts, explicit side and logical axes.
Broader floating-sequence and equal-shape sample batching remain deferred, not cancelled.

Data0001–0003 are Complete, accepted locally on 2026-10-20. Current code supplies immutable valid
counts, explicit axes/side, borrowed `PaddedSequenceBatch` construction and `LongSequenceBatcher`
allocation. Data0003's brief owns final review and validation evidence.

## Contracts and common base

- [Root scope index, layout, ownership, dependency and testing rules](../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
- [Sole owning Data contract](../../../architecture/contracts/extensions-training.md#extensionsdata).
- [Text contract](../../../architecture/contracts/extensions-training.md#extensionstext) governs the downstream adapter.
- [ADR 0043](../../../design/decisions/0043-whitespace-text-and-padded-integral-batches.md)
  records user approval on 2026-10-20.
- Common base: `ef35f2ec001b0c161d1abe825ff3be725b9be49e` plus accepted uncommitted
  Data0001/Data0002, enacted Data/Text authority and accepted Data0003 including the iteration fix
  and final documentation, plus accepted uncommitted Text0001; no predecessor is claimed merged.

## Scope and non-goals

Selected: shared LEFT/RIGHT representation, unchanged O(B) host-count API, and a fluent
`LongSequenceBatcher` using public Model heap allocation. There is no implicit truncation.
Data never tokenizes, owns NN computation, runs Engine, allocates native resources, or adds a
storage lifetime, runtime-length binding, loader/sampler/epoch lifecycle or persistence.
Current scans still accept right-valid-prefix counts only; LEFT support is preparation metadata,
not a scan capability change. The implemented NN0022A export only materializes counts for
RIGHT/TIME_MAJOR batches; binding remains owned by the application.

## Module invariants and dependencies

Data retains exactly `api(project(":modules:model"))`, including tests. Text's separately approved
sole Data API edge makes Text a downstream consumer; NN gains no Data/Text edge.
Counts are independent of position: RIGHT means valid prefix, LEFT means valid suffix.
Stored values never determine validity; padding lengths and dense masks are not second sources
of truth. Generic metadata construction borrows Tensor references without storage inspection;
the integral batcher allocates fresh contiguous heap values. NN0022A additionally authorizes
explicit fresh heap count export, without changing metadata construction or accessors.

## Package map

All types stay in `io.github.pho001.synaptik.data`:

- Current `ValidSequenceLengths` and `SequenceAxisOrder` remain, with count terminology updated.
- Data0003 replaced `RightPaddedSequenceBatch` with `PaddedSequenceBatch` and its nested Builder.
- Data0003 added `PaddingSide` and `LongSequenceBatcher` with its nested Builder.
- No compatibility alias, reserved subpackage or generic utility/loader facade.

## Dependency DAG and authorized frontiers

Data0001 → Data0002 → Data0003 → Text0001.
Data0003 → deferred Data0005 (floating sequences) and Data0006 (equal-shape samples);
Data0003 + Data0005 + Data0006 → Data0004 (broader batching checkpoint).
Data0001–0003 + NN0022/0022B/0028/0029 Complete →
[NN0022A numeric inference](../nn/tasks/0022a-padded-numeric-sequence-inference.md) Complete:
accepted 2026-10-21 after independent APPROVE and main's architecture/root checkpoint.
The approved Data-owned `PaddedSequenceBatch.toScanLengthsTensor()` export has no NN-to-Data edge.
The NN0022A cross-area frontier is closed; no separate Data successor is opened.
Its brief owns all seven integration fields, verified base `6eabea0c1ebf77b9bd8382855c7131eb89b52e04`,
serial conflicts and main's integration ownership. Shared cutover documents have one exclusive writer.

Authorized Data frontier: None. Data0003 is Complete after final independent APPROVE and local
acceptance on 2026-10-20. The [Text program join](../text/tasks/0001-whitespace-tokenizer-and-data-batching.md#result)
completed on 2026-10-21; Text0001 is Complete and the Text frontier is None. No automatic successor.

## Task list

| ID | Task | Status | Depends on | Intent/result |
|---|---|---|---|---|
| [0001](tasks/0001-data-extension-boundary.md) | Data-only architecture/build foundation | Complete | None | Accepted empty module, exact Model API edge and architecture enforcement; historical brief unchanged. |
| [0002](tasks/0002-valid-sequence-lengths-and-right-padded-batch.md) | Host metadata and borrowed batch builder | Complete | 0001 | Accepted immutable host prefixes/logical axes and right-padded metadata; historical brief unchanged. |
| [0003](tasks/0003-padded-integral-sequence-batching.md) | Shared padded representation and integral batching | Complete | 0001/0002 Complete; main authority approved | Accepted after independent APPROVE and module/architecture-consumer gates; Text program root join complete. |
| 0005 | Floating sequence batching | Draft | 0003; selected concrete float consumer/API | Preserve broader numeric capability; no speculative dtype/truncation abstraction now. |
| 0006 | Equal-shape numeric sample batching | Draft | 0003; selected sample consumer/API | Preserve generic numeric stacking for later Vision use without image semantics. |
| 0004 | Broader Data batching checkpoint | Draft | 0003, 0005, 0006 | Validate the broader sample/sequence capability after selection, not a prerequisite of Text0001. |

## Integration ownership and shared documents

Main owns shared root/scoped authority, explanations, API references, glossary, roadmap, master
plans and final integration plumbing. NN0022A's serial independent documentation/review pass
returned frozen source/docs to main; the source-matched architecture/root checkpoint passed.
No Java executor remains active. The completed Data/Text program's ownership is recorded below.
All seven integration fields live in each selected brief; Parallel group is None.
The sole Data worker owns architecture consumer/dependency-test migration; the sole Text worker
owns registration, Text enforcement and external integration test plumbing. Independent review
finalized shared glossary/index/public API docs after the worker stopped. Main completed the
source-matched architecture/Text–Embedding integration and final root checkpoint at the Text program join,
reusing successful Data worker and focused architecture-consumer evidence.
This planning pass runs no
tests, builds, agents, commits or pushes.

## Milestones

- Complete: foundation, migrated shared padding-side metadata and INT64 batching.
- Complete in Text: frozen whitespace tokenization, delegating Text adapter and program join.
- Deferred: floating sequences, numeric samples and broader batching checkpoint.

## Open decisions and risks

No architectural decision or local acceptance gap remains for Data0003; independent review and
main acceptance are complete. Broader float/sample APIs, arbitrary masks and
binding protocols remain future decisions; the narrow opt-in count export is approved under
[ADR0044](../../../design/decisions/0044-opt-in-scan-length-tensor-export.md). Training's former Data-lifecycle proposal
is not satisfied by batching. Early checked geometry must reject enormous virtual lists before
batch-sized snapshots or output allocation; caller-coordinated mutation avoids a transactional
snapshot promise. Completed briefs remain historical evidence, not current architecture authority.
