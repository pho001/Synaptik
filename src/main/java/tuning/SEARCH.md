# Tuning Search

## Contents

- [Purpose](#purpose)
- [Search Contracts](#search-contracts)
- [Candidate Generation and Refinement](#candidate-generation-and-refinement)
- [Simple Strategies](#simple-strategies)
- [Tree Strategies](#tree-strategies)
- [Score and Bound Models](#score-and-bound-models)
- [History-Aware Search](#history-aware-search)
- [Default Strategy Selection](#default-strategy-selection)
- [Choosing a Strategy in Practice](#choosing-a-strategy-in-practice)
- [Examples](#examples)

## Purpose

`tuning.search` decides which candidates are worth evaluating.

It does **not**:

- compile graphs
- execute kernels
- validate correctness
- persist results

It only selects candidate order and candidate subsets.

## Search Contracts

### `SearchStrategy`

Core contract:

```java
interface SearchStrategy {
    SearchResult search(SearchContext context);
}
```

Strategies may additionally support refinement:

```java
boolean supportsRefinement();
SearchResult refine(...);
```

This lets the autotuner do:

1. initial selection
2. measurement
3. refinement based on measured candidates

### `SearchResult`

Contains:

- selected candidates for this step
- one preferred candidate, if the strategy has one

### `SearchPolicy`

Global search budget:

- `maxCandidates`
- `beamWidth`
- `maxRounds`
- `allowPruning`

These values constrain strategy behavior but do not define strategy type.

## Candidate Generation and Refinement

Search depends on:

- `CandidateSpace`
- optionally `RefinableCandidateSpace`

`CandidateSpace`:

- generates candidates for the initial phase

`RefinableCandidateSpace`:

- can additionally generate neighbor candidates around an already known candidate

This is what makes tree search possible without introducing a second hidden execution model.

## Simple Strategies

### Exhaustive

- [ExhaustiveSearchStrategy.java](./search/ExhaustiveSearchStrategy.java)

Behavior:

- generate all candidates
- keep first `maxCandidates`

Best use:

- tiny candidate spaces

### FirstK

- [FirstKSearchStrategy.java](./search/FirstKSearchStrategy.java)

Behavior:

- take the first `K`

Best use:

- seed strategy
- budget guard

### Composite

- [CompositeSearchStrategy.java](./search/CompositeSearchStrategy.java)

Behavior:

- run multiple strategies
- merge outputs
- deduplicate by candidate name

Best use:

- combine seed heuristics

## Tree Strategies

Tree search works because:

- we fingerprint candidates
- we can generate neighbors through `RefinableCandidateSpace`
- we keep lineage in a search tree

Relevant classes:

- [SearchTreeNode.java](./search/SearchTreeNode.java)
- [SearchTreeSnapshot.java](./search/SearchTreeSnapshot.java)
- [SearchTreeReport.java](./search/SearchTreeReport.java)

### Tree Beam

- [TreeBeamSearchStrategy.java](./search/TreeBeamSearchStrategy.java)

Algorithm:

1. get seed candidates
2. store them as root nodes
3. after measurement, select best `beamWidth` frontier nodes
4. expand their neighbors
5. next frontier becomes those neighbors

This is the simplest useful tree strategy.

### Best-First Tree

- [BestFirstTreeSearchStrategy.java](./search/BestFirstTreeSearchStrategy.java)

Algorithm:

1. measure current frontier
2. pick the single best frontier node according to a score model
3. expand only its neighborhood

This is narrower than beam search and works well when:

- score model is already meaningful
- candidate space is large

### Branch-and-Bound Tree

- [BranchAndBoundSearchStrategy.java](./search/BranchAndBoundSearchStrategy.java)

Algorithm:

1. compute current best measured score
2. for every frontier node compute an optimistic bound
3. if bound is worse than current best, prune the branch
4. expand only remaining branches

This is the first strategy here that can suppress whole subtrees intentionally.

## Score and Bound Models

### Score model

Relevant classes:

- [CandidateScoreModel.java](./search/CandidateScoreModel.java)
- [MedianSteadyStateScoreModel.java](./search/MedianSteadyStateScoreModel.java)

Current default scoring:

- lower steady-state median time is better

### Bound model

Relevant classes:

- [CandidateBoundModel.java](./search/CandidateBoundModel.java)
- [ZeroBoundModel.java](./search/ZeroBoundModel.java)
- [ParentScoreBoundModel.java](./search/ParentScoreBoundModel.java)
- [WorkloadAwareBoundModel.java](./search/WorkloadAwareBoundModel.java)

#### Conv2d bound

- [Conv2dBoundModel.java](./search/Conv2dBoundModel.java)

Current heuristic:

- `HEURISTIC` lowering is treated as more promising
- `OFF` is treated as less promising

#### MatMul bound

- [MatMulBoundModel.java](./search/MatMulBoundModel.java)

Current heuristic:

- BLAS-enabled candidates are treated as more promising than `NONE`

#### Transformer hot path bound

- [TransformerHotPathBoundModel.java](./search/TransformerHotPathBoundModel.java)

Current heuristic:

- `attentionMatMul=AUTO` or `FORCE_ON` is usually more promising than `FORCE_OFF`
- BLAS-enabled variants are treated as more promising

These are heuristic bounds, not proofs.
That is acceptable as long as the bound role stays explicit.

## History-Aware Search

- [HistoryAwareSearchStrategy.java](./search/HistoryAwareSearchStrategy.java)

Purpose:

- reuse persisted best-profile and candidate-history knowledge

Algorithm:

1. load persisted best profile for current hardware + workload
2. move it to the front of candidate ordering if present
3. load history entries for the same fingerprint context
4. prefer historically good candidates
5. optionally prune historically invalid candidates if pruning is enabled

This is the first search layer that learns between runs.

## Default Strategy Selection

- [AutotuneDefaultStrategySelector.java](./session/AutotuneDefaultStrategySelector.java)

Current default logic:

- non-refinable space -> `Exhaustive`
- medium refinable space -> `TreeBeam`
- larger refinable space -> `BranchAndBound` with workload-aware bounds
- if persistence is enabled -> wrap with `HistoryAwareSearchStrategy`

This is intentionally policy, not hardwired inside `SearchStrategy`.

## Choosing a Strategy in Practice

Use:

- `Exhaustive`
  - when the candidate grid is small and you want a complete answer
- `TreeBeam`
  - when the space is refinable and you want breadth without full explosion
- `BestFirstTree`
  - when you already trust your score model and want aggressive focus
- `BranchAndBound`
  - when the workload family has useful optimistic bounds
- `HistoryAware`
  - when persistence exists and you want cross-run reuse

Typical examples:

- small matmul knob sweep:
  - `Exhaustive`
- medium conv2d lowering/runtime sweep:
  - `TreeBeam` or `BranchAndBound`
- repeated transformer tuning on the same machine:
  - `HistoryAware(BranchAndBound(...))`

## Examples

### Example 1: explicit exhaustive search

```java
SearchStrategy strategy = new ExhaustiveSearchStrategy();
```

Input:

- candidate space with `N` profiles

Output:

- up to `maxCandidates` selected candidates
- no refinement rounds

### Example 2: history-aware branch-and-bound

```java
SearchStrategy strategy = new HistoryAwareSearchStrategy(
        new BranchAndBoundSearchStrategy(
                new MedianSteadyStateScoreModel(),
                new WorkloadAwareBoundModel()
        ),
        resolver,
        historyStore,
        bestProfilePath,
        historyPath
);
```

Input:

- current workload context
- candidate history for the same hardware/workload fingerprint

Output:

- historically strong candidates move earlier
- obviously weak branches can be pruned before full evaluation

- first `maxCandidates` candidates in generation order

### Example 2: tree beam refinement

```java
SearchStrategy strategy = new TreeBeamSearchStrategy(
        new FirstKSearchStrategy(4),
        4,
        8
);
```

Meaning:

- seed with first 4 candidates
- after each round, keep best 4 frontier nodes
- expand up to 8 neighbors per node

### Example 3: branch-and-bound for transformer search

```java
SearchStrategy strategy = new BranchAndBoundSearchStrategy(
        new FirstKSearchStrategy(4),
        new MedianSteadyStateScoreModel(),
        new WorkloadAwareBoundModel(),
        4,
        8
);
```

Input:

- transformer workload
- refinable profile grid

Output:

- tree exploration biased by:
  - measured median score
  - transformer-specific optimistic bound heuristics
