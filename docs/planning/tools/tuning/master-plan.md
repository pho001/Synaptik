# Tuning Master Plan

## Goal

[Coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix) explicit, bounded [model autotuning](../../../glossary.md#autotuning--model-autotuning) before [Runtime](../../../glossary.md#term-runtime): reuse compatible local workload
results, measure complete candidates, select deterministically, and manage explicit compact [cache](../../../glossary.md#term-cache)
[artifacts](../../../glossary.md#term-artifact) while keeping rich evidence separate.

## Authority and contracts

[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative. The exact applicable headings
are [`modules/runtime`](../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
[`modules/prepare`](../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
[Concrete backend modules](../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../glossary.md#concrete-backend); [modules glossary](../../../glossary.md#term-module))</span>, and
[Performance evidence and optimization tooling](../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
The [performance/tuning](../../../architecture/performance-evidence-and-tuning.md),
[module-boundary](../../../architecture/module-boundaries.md), and
[Runtime/Prepare/backend](../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../glossary.md#term-runtime); [Prepare glossary](../../../glossary.md#prepare); [backend glossary](../../../glossary.md#backend))</span> documents
explain those contracts.

## [Scope](../../../glossary.md#term-scope) and non-goals

Tuning owns [model](../../../glossary.md#term-model)-guided workload extraction, [canonical](../../../glossary.md#term-canonical-form) [signatures](../../../glossary.md#term-signature), exact-signature deduplication,
compatible [cache](../../../glossary.md#term-cache) reuse, miss-only local measurement, bounded end-to-end complete-plan comparison,
compact workload/model-plan [artifacts](../../../glossary.md#term-artifact), and separate rich evidence and inspection.

It does not own benchmark reporting, [graph](../../../glossary.md#graph) semantics or transformations, [Planning](../../../glossary.md#term-planning) [ownership](../../../glossary.md#term-ownership-meanings),
[backend](../../../glossary.md#backend) [lowering](../../../glossary.md#lowering) or private candidate vocabulary, [Runtime](../../../glossary.md#term-runtime) decisions, prepared-executable
[serialization](../../../glossary.md#term-serialization), hidden global state, Java serialization, or generated-code artifact caching.

## Stable invariants and dependencies

- Phase 1 loads the explicit workload cache before measurement, reuses authenticated compatible
  hits, and measures only misses. Phase 2 reuses the selected local decision and never repeats
  route-parameter search.<span data-glossary-navigation="true"> ([cache glossary](../../../glossary.md#term-cache); [parameter glossary](../../../glossary.md#term-parameter))</span>
- Tuning owns bounded warmup/timing, elapsed-sample validation, median selection, encounter-order
  ties, cache coordination, and evidence. Candidate owners supply complete valid opaque candidates,
  compatibility, identities, codecs, and fresh execution/preparation actions.<span data-glossary-navigation="true"> ([owners glossary](../../../glossary.md#term-owner); [identities glossary](../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([warmup glossary](../../../glossary.md#term-warmup); [sample glossary](../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [validation glossary](../../../glossary.md#term-validation); [median glossary](../../../glossary.md#term-median); [opaque glossary](../../../glossary.md#term-opaque))</span>
- Exact/default eligibility is resolved before measurement. Safe heuristics and cache-only
  preparation remain correct when tuning is absent or cannot produce a selected result.
- Runtime performs no tuning search, cache access, graph inspection, or execution-setting
  selection. Rich evidence is not reconstructed from compact cache state.<span data-glossary-navigation="true"> ([Runtime glossary](../../../glossary.md#term-runtime); [graph glossary](../../../glossary.md#graph))</span>
- The tool may use public Config, Compiler, Planning, Prepare, Engine, lifecycle, and trace
  contracts when composition requires them; it never imports private backend internals or uses a
  Runtime service lookup.<span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler); [Planning glossary](../../../glossary.md#term-planning); [Prepare glossary](../../../glossary.md#prepare); [Engine glossary](../../../glossary.md#term-engine); [lifecycle glossary](../../../glossary.md#lifecycle); [trace glossary](../../../glossary.md#trace); [backend glossary](../../../glossary.md#backend); [lookup glossary](../../../glossary.md#term-lookup))</span>

## Package map

```text
io.github.pho001.synaptik.tools.tuning/
  WorkloadTuning* / BackendWorkloadTuning / ColdCandidateMeasurement
  WorkloadCacheFile                 package-private compact Phase-1 persistence
  CompletePlanTuning* / BackendCompletePlanTuning / CompletePlanCorrectness
  CompleteCandidateMeasurement / ModelPlanCacheFile
  TuningInspection                 public read-only cache/evidence inspection
```

The root package is the deliberate small public surface. It contains no [backend](../../../glossary.md#backend), [Runtime](../../../glossary.md#term-runtime), Config,
[cache](../../../glossary.md#term-cache)-utility, [registry](../../../glossary.md#term-registry), or private-internals subpackage.

## Task list

| ID | Task | Status | Depends on | One-line result or intent |
|---|---|---|---|---|
| 0001 | [Exact/default model-guided workload tuning and reusable cache](tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)<span data-glossary-navigation="true"> ([model glossary](../../../glossary.md#term-model))</span> | Complete | [CPU](../../../glossary.md#term-cpu) 0010E; [Prepare](../../../glossary.md#prepare) 0004; caller-supplied stable [identity](../../../glossary.md#term-identity-meanings), typed [opaque](../../../glossary.md#term-opaque) collaboration, and operational cold measurement | Added [cache](../../../glossary.md#term-cache)-first deduplication, bounded miss-only measurement, deterministic selection, compact [persistence](../../../glossary.md#term-persistence), and separate evidence. |
| 0002 | [Bounded complete-plan tuning and model-plan cache](tasks/0002-bounded-complete-plan-tuning-and-model-plan-cache.md) | Complete | 0001; Complete CPU 0010J and [Engine](../../../glossary.md#term-engine) 0008A evidence | Added checked complete-plan correctness/timing, SESSION/PERSISTENT reuse, authenticated compact records, and rich evidence. |
| 0003 | [Read-only cache, plan, and evidence inspection](tasks/0003-read-only-cache-plan-and-evidence-inspection.md) | Complete | 0001–0002, stable [artifact](../../../glossary.md#term-artifact) [schemas](../../../glossary.md#term-schema) | Added bounded redacted inspection with structural/key mismatch reasons and no decoding, execution, or mutation. |
| 0004 | [Tuning package Javadoc status reconciliation](tasks/0004-package-javadoc-status-reconciliation.md) | Complete | 0001–0003; Engine 0009/0013 | Corrected public two-phase CPU Engine composition and remaining limitations in package [Javadoc](../../../glossary.md#term-javadoc). |
| 0005 | [Bound model-plan cache encoding before allocation](tasks/0005-model-plan-cache-encoding-bound.md) | Complete | 0002–0004; project review | [Preflight](../../../glossary.md#term-preflight) the existing aggregate artifact bound before encoding; preserve schema and [atomic](../../../glossary.md#resource-transaction-and-rollback) rejection. |

## Milestones and current frontier

[Canonical](../../../glossary.md#term-canonical-form) workload reuse, bounded complete-plan selection, and [artifact](../../../glossary.md#term-artifact)/evidence inspection are
`Complete` through 0003. Config 0006A–0006B and [Engine](../../../glossary.md#term-engine) 0009 provide the declarative request and
current [CPU](../../../glossary.md#term-cpu) public composition around the generic tool; CPU 0010J remains the first complete-plan
[producer](../../../glossary.md#term-producer). Documentation-only 0004 is `Complete`; its independent review confirmed the corrected
package status without changing authority or capability. Project-review repair 0005 is Complete;
it changes resource safety, not candidate semantics or the artifact format.

## [Artifact](../../../glossary.md#term-artifact), measurement, and [lifecycle](../../../glossary.md#lifecycle) [ownership](../../../glossary.md#term-ownership-meanings)

- Phase-1 candidate execution is complete and cold. Tuning owns warmups, positive odd sample
  counts, monotonic timing, integer-middle median comparison, and fail-before-measurement bounds.<span data-glossary-navigation="true"> ([sample glossary](../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [median glossary](../../../glossary.md#term-median))</span>
- For `N` Phase-2 [cache](../../../glossary.md#term-cache)-miss candidates, `W` warmups, and `S` samples, checked preflight bounds
  exactly <span data-glossary-navigation="true"> ([preflight glossary](../../../glossary.md#term-preflight))</span>`N * (1 + W + S)`. All exact correctness executions and cleanup succeed before timing.
  A compatible cache hit executes no correctness, warmup, or timed trial.<span data-glossary-navigation="true"> ([cache hit glossary](../../../glossary.md#cache-and-memoization); [warmup glossary](../../../glossary.md#term-warmup))</span>
- Compact workload and model-plan artifacts are bounded, versioned, checksummed, canonically
  ordered files with compatibility/schema identities and atomic replacement. Missing or
  incompatible entries are misses; corrupt, oversized, unsupported, duplicate, or invalid data
  fails safely without mutation.<span data-glossary-navigation="true"> ([identities glossary](../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([schema glossary](../../../glossary.md#term-schema); [atomic glossary](../../../glossary.md#resource-transaction-and-rollback))</span>
- `SESSION` performs no [model](../../../glossary.md#term-model)-plan-cache I/O. `PERSISTENT` reuse still requires the current
  producer/backend decoder to authenticate the decision and fresh preparation to build executable
  state. No prepared execution or native/JVM executable payload is serialized.<span data-glossary-navigation="true"> ([producer glossary](../../../glossary.md#term-producer); [backend glossary](../../../glossary.md#backend); [prepared execution glossary](../../../glossary.md#prepared-execution--preparedexecution); [JVM glossary](../../../glossary.md#term-java-virtual-machine-jvm); [payload glossary](../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Correctness references, publication bytes, Runtime resources, materialization, cleanup, selected
  preparation, and fallback execution remain with Engine/lifecycle owners. Tuning sees opaque
  actions and typed match/mismatch results.<span data-glossary-navigation="true"> ([owners glossary](../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([publication glossary](../../../glossary.md#term-publication); [Runtime glossary](../../../glossary.md#term-runtime); [materialization glossary](../../../glossary.md#materialization); [Engine glossary](../../../glossary.md#term-engine); [opaque glossary](../../../glossary.md#term-opaque))</span>

## Live risks and open decisions

- Prepared-executable serialization remains deliberately unresolved and is not implied by either
  cache format.<span data-glossary-navigation="true"> ([serialization glossary](../../../glossary.md#term-serialization); [cache glossary](../../../glossary.md#term-cache))</span>
- A future producer must expose complete valid typed candidates, compatibility/version identity,
  and file/lifecycle ownership without leaking private fields into shared orchestration.<span data-glossary-navigation="true"> ([producer glossary](../../../glossary.md#term-producer); [identity glossary](../../../glossary.md#term-identity-meanings); [lifecycle glossary](../../../glossary.md#lifecycle); [ownership glossary](../../../glossary.md#term-ownership-meanings))</span>
- Main risks are duplicate local search in Phase 2, family-wide cache keys, shared interpretation
  of backend knobs, implicit or stale caches, unbounded trials, and treating caller labels as
  canonical model/plan identity.<span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend); [canonical glossary](../../../glossary.md#term-canonical-form); [model glossary](../../../glossary.md#term-model))</span>

## History and update policy

The former “later Config extension” and “later [Engine](../../../glossary.md#term-engine) composition” notes are closed by Config
0006B and Engine 0009. They are not open tuning work. Detailed audits, context identifiers,
[validation](../../../glossary.md#term-validation) [logs](../../../glossary.md#commit-release-distribution-log-and-seed-meanings), and completion evidence remain in linked task files and Git history.

Update this map only for a selected task, [producer](../../../glossary.md#term-producer)/composition gate, [ownership](../../../glossary.md#term-ownership-meanings) or package change,
or live [artifact](../../../glossary.md#term-artifact)/measurement risk. Keep evidence in task briefs. If a [planning](../../../glossary.md#term-planning) change conflicts
with `ARCHITECTURE.md`, stop and use the architecture-decision process.
