# Tuning Search

The search layer decides which candidates are worth measuring and in what order. It does not execute anything itself.

Its contract is:

- it receives `SearchContext`
- it returns `SearchResult`

Optionally it may support refinement over already measured candidates.

## Reading Guide

This document explains:

- what the minimum search-strategy contract is
- how exhaustive and tree strategies work
- how history-aware ordering works
- how the default strategy is selected according to candidate space

## Core Contracts

### `SearchStrategy`

Interface:

```java
interface SearchStrategy {
    SearchResult search(SearchContext context);
}
```

Optionally:

```java
boolean supportsRefinement();
SearchResult refine(
        SearchContext context,
        List<BenchmarkCandidateReport> evaluatedSoFar,
        int round,
        Set<String> seenFingerprints
);
```

That means:

- search can be single-round
- or iterative

### `SearchContext`

Contains:

- `AutotuneRequest`
- `CandidateSpace`

### `SearchResult`

Contains:

- `selectedCandidates`
- `preferredCandidate`

`preferredCandidate` is a hint, not an execute contract.

### `SearchPolicy`

Carries the budget:

- `maxCandidates`
- `beamWidth`
- `maxRounds`
- `allowPruning`

## Candidate Spaces

Search does not work directly with raw knobs. It works with candidate spaces.

Basic types:

- `CandidateSpace`
- `RefinableCandidateSpace`

`CandidateSpace`:

- can generate initial candidates

`RefinableCandidateSpace`:

- can generate neighbors around an already known candidate

That is what enables tree search without introducing a parallel execution model.

## Search Lifecycle In Autotune

`DefaultAutotuneSession` currently does:

1. call `search(context)` for the initial batch
2. validate and measure the candidates
3. if the strategy supports refinement:
   - call `refine(...)`
   - validate and measure the new batch again
4. choose the best finalist by median

So:

- the search layer never calls measurement directly
- the session uses it as a policy layer over the evaluation loop

## Simple Strategies

### `ExhaustiveSearchStrategy`

Use when:

- the candidate grid is small
- you want full coverage

Advantages:

- simplicity
- no heuristic error

Disadvantage:

- it does not scale

### `FirstKSearchStrategy`

Use when:

- you want a seed batch
- you need a budget guard

On its own it is usually not the final strategy. It often serves as a seed for tree strategies.

### `CompositeSearchStrategy`

Use when:

- you want to combine multiple ordering heuristics
- you want a deduplicated list of seed candidates

## Tree Strategies

Tree search only makes sense if the candidate space supports refinement or neighborhood relations.

### `TreeBeamSearchStrategy`

Idea:

1. select seed candidates
2. measure them
3. keep the best frontier according to `beamWidth`
4. expand their neighborhood
5. repeat

Use when:

- you want a reasonable tradeoff between breadth and cost
- the candidate space is refinable

### `BestFirstTreeSearchStrategy`

Idea:

- in each step, expand only the most promising frontier node

Use when:

- you trust the score model
- you want aggressive focus instead of breadth

### `BranchAndBoundSearchStrategy`

Idea:

1. keep the current best measured score
2. compute an optimistic bound for each frontier node
3. if the bound is worse than the best score, drop that branch
4. expand only the remaining branches

Use when:

- the workload family has a reasonable bound model
- the candidate space is larger

## Score And Bound Models

### Score Model

Relevant classes:

- [CandidateScoreModel.java](./search/CandidateScoreModel.java)
- [MedianSteadyStateScoreModel.java](./search/MedianSteadyStateScoreModel.java)

The current default score is:

- lower steady-state median = better

### Bound Models

Relevant classes:

- [CandidateBoundModel.java](./search/CandidateBoundModel.java)
- [ZeroBoundModel.java](./search/ZeroBoundModel.java)
- [ParentScoreBoundModel.java](./search/ParentScoreBoundModel.java)
- [WorkloadAwareBoundModel.java](./search/WorkloadAwareBoundModel.java)

`WorkloadAwareBoundModel` currently dispatches by `WorkloadKind`:

- `CONV2D`
- `MATMUL`
- `TRANSFORMER_HOT_PATH`
- otherwise generic fallback

That means:

- search heuristics can be workload-aware
- but they still return only ordering/pruning hints, not execute semantics

## History-Aware Search

`HistoryAwareSearchStrategy` is a wrapper around another strategy.

It does:

1. load the persisted best profile for the current hardware + workload
2. if the fingerprint matches, move it forward
3. load history entries for the same context
4. prefer historically good candidates
5. optionally skip historically invalid candidates if pruning is enabled

Important reality:

- it does not perform its own scoring
- it only reorders candidate space before delegating to the inner strategy

## Default Strategy Selection

Default strategy selection is handled by:

- [AutotuneDefaultStrategySelector.java](./session/AutotuneDefaultStrategySelector.java)

Current logic:

- non-refinable space
  - `Exhaustive`
- refinable space with sufficiently large candidate count
  - `BranchAndBound`
- refinable space of medium size
  - `TreeBeam`
- if persistence is enabled
  - wrap it in `HistoryAwareSearchStrategy`

So:

- default selection is not hardcoded inside the strategies
- it is a policy layer

## Example: Small Stage-Order Space

If you are tuning a small stage-order grid:

- the candidate space is small
- refinement usually does not make sense

Use:

- `ExhaustiveSearchStrategy`

## Example: Matmul Runtime Search

If you have a larger refinable matmul candidate space:

- tiles
- microkernels
- thresholds

a sensible default is:

- `BranchAndBoundSearchStrategy`

because:

- the space is larger
- `WorkloadAwareBoundModel` can provide matmul-specific hints

## Example: Repeated Tuning On Same Machine

If you already have persisted:

- best profile
- history JSONL

wrap the strategy with:

- `HistoryAwareSearchStrategy`

Why:

- you retest likely-good candidates earlier
- you can skip historically invalid variants

## Search Does Not Own Persistence

Search may read persistence as a prior, but it does not own its lifecycle.

Persistence lifecycle is handled by the session/store layers.

That matters because:

- history is auxiliary evidence
- search must not turn it into the execute source of truth

## Common Mistakes

- expecting a search strategy to measure candidates itself
- using branch-and-bound without a reasonable bound model
- forgetting candidate deduplication through fingerprint
- treating history-aware reordering as proof that the stored winner is still correct

## Related Docs

- architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)
- persistence: [PERSISTENCE.md](./PERSISTENCE.md)
- reporting: [REPORTING.md](./REPORTING.md)

## Concrete Search Example

Suppose graph autotune starts from a calibrated seed and a stage-order candidate space that can generate:

- `AR,CSE`
- `AR,CSE,MEM`
- `AR,CSE,FUSE`
- `AR,CSE,FUSE,MEM`

An exhaustive strategy simply returns all of them.
The session then:

1. validates each candidate
2. measures each candidate
3. ranks the valid ones by steady-state median

Search never decides the winner by itself.
It only decides what should be measured and in what order.

## Why History-Aware Reordering Helps

History-aware search is useful when:

- candidate spaces are stable across repeated runs
- the same workload and hardware appear again

It can:

- move historically good candidates earlier
- deprioritize or skip historically invalid candidates

It cannot:

- prove a historical winner is still correct after code changes
- replace fresh validation and measurement

## Good Search Strategy Selection

Use exhaustive search when:

- the space is small
- you care about full coverage

Use beam/best-first/branch-and-bound when:

- the space is refinable
- full enumeration is too expensive
- you have at least a somewhat meaningful bound or neighborhood model
