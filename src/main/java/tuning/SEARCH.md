# Tuning Search

The search layer chooses which candidates should be measured and in what order.

It does not execute them directly.

## Core Contracts

### `SearchStrategy`

Main contract:

```java
SearchResult search(SearchContext context);
```

Some strategies also support refinement rounds:

```java
boolean supportsRefinement();
SearchResult refine(SearchContext context, List<BenchmarkCandidateReport> evaluatedSoFar, int round, Set<String> seenFingerprints);
```

### `SearchContext`

Contains:

- the autotune request
- the candidate space

### `SearchResult`

Contains:

- selected candidates
- optional preferred candidate hint

The preferred candidate is advisory, not an execution contract.

### `SearchPolicy`

Carries the budget:

- `maxCandidates`
- `beamWidth`
- `maxRounds`
- `allowPruning`

## Candidate Spaces

Search does not work on raw primitive knobs directly.
It works on candidate spaces.

Important types:

- `CandidateSpace`
- `RefinableCandidateSpace`

Meaning:

- `CandidateSpace`
  - can generate initial candidates
- `RefinableCandidateSpace`
  - can generate neighborhood candidates around promising already-measured ones

This is what makes tree or refinement search possible without inventing a second execution model.

## Current Strategies

Examples in the package:

- `ExhaustiveSearchStrategy`
- `FirstKSearchStrategy`
- `HistoryAwareSearchStrategy`
- `BestFirstTreeSearchStrategy`
- `TreeBeamSearchStrategy`
- `BranchAndBoundSearchStrategy`
- `CompositeSearchStrategy`

Support classes include:

- bound models
- score models
- tree snapshots/reports

## How Search Fits Into Autotune

The normal session shape is:

1. ask the strategy for an initial candidate batch
2. validate candidates
3. measure valid candidates
4. if refinement is supported:
   - ask for another batch
   - validate and measure again
5. choose the best finalist by measurement policy

So the search layer is a policy layer over candidate ordering.
It is not the measurement engine.

## Worked Examples

### Exhaustive search

Best when:

- the candidate grid is small
- full coverage is cheap enough

Example:

- 6 explicit stage-order candidates
- measure all 6
- choose the best median

### History-aware search

Best when:

- you already have useful prior history
- the grid is large enough that order matters

Example:

- a previous run suggests `AR,CSE,FUSE,MEM` and `AR,CSE,MEM` are historically strong
- history-aware ordering tries those first instead of random or lexical order

### Beam/tree search

Best when:

- candidate spaces are refinable
- local neighborhoods are meaningful
- exhaustive coverage would be too large

## Bound Models

The package includes bound models such as:

- `MatMulBoundModel`
- `Conv2dBoundModel`
- `TransformerHotPathBoundModel`
- `WorkloadAwareBoundModel`

These are heuristic pruning aids, not correctness proofs.

Their job is:

- estimate which unexplored regions are unlikely to beat current best candidates
- reduce wasted measurement work

## Search Reports

Tree-capable strategies can emit search tree reports through:

- [search/TextSearchTreeReportRenderer.java](./search/TextSearchTreeReportRenderer.java)
- [search/JsonSearchTreeReportRenderer.java](./search/JsonSearchTreeReportRenderer.java)

These are useful when debugging why a strategy picked or pruned certain branches.
