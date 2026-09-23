# Tuning Master Plan

## Goal

Coordinate explicit, bounded model autotuning before Runtime: reuse compatible local workload
results, measure complete candidates, select deterministically, and manage explicit compact cache
artifacts while keeping rich evidence separate.

## Authority and contracts

[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative. The exact applicable headings
are [`modules/runtime`](../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
[`modules/prepare`](../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
[Concrete backend modules](../../../architecture/contracts/backend-execution.md#concrete-backend-modules), and
[Performance evidence and optimization tooling](../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
The [performance/tuning](../../../architecture/performance-evidence-and-tuning.md),
[module-boundary](../../../architecture/module-boundaries.md), and
[Runtime/Prepare/backend](../../../architecture/runtime-prepare-backend-boundary.md) documents
explain those contracts.

## Scope and non-goals

Tuning owns model-guided workload extraction, canonical signatures, exact-signature deduplication,
compatible cache reuse, miss-only local measurement, bounded end-to-end complete-plan comparison,
compact workload/model-plan artifacts, and separate rich evidence and inspection.

It does not own benchmark reporting, graph semantics or transformations, Planning ownership,
backend lowering or private candidate vocabulary, Runtime decisions, prepared-executable
serialization, hidden global state, Java serialization, or generated-code artifact caching.

## Stable invariants and dependencies

- Phase 1 loads the explicit workload cache before measurement, reuses authenticated compatible
  hits, and measures only misses. Phase 2 reuses the selected local decision and never repeats
  route-parameter search.
- Tuning owns bounded warmup/timing, elapsed-sample validation, median selection, encounter-order
  ties, cache coordination, and evidence. Candidate owners supply complete valid opaque candidates,
  compatibility, identities, codecs, and fresh execution/preparation actions.
- Exact/default eligibility is resolved before measurement. Safe heuristics and cache-only
  preparation remain correct when tuning is absent or cannot produce a selected result.
- Runtime performs no tuning search, cache access, graph inspection, or execution-setting
  selection. Rich evidence is not reconstructed from compact cache state.
- The tool may use public Config, Compiler, Planning, Prepare, Engine, lifecycle, and trace
  contracts when composition requires them; it never imports private backend internals or uses a
  Runtime service lookup.

## Package map

```text
io.github.pho001.synaptik.tools.tuning/
  WorkloadTuning* / BackendWorkloadTuning / ColdCandidateMeasurement
  WorkloadCacheFile                 package-private compact Phase-1 persistence
  CompletePlanTuning* / BackendCompletePlanTuning / CompletePlanCorrectness
  CompleteCandidateMeasurement / ModelPlanCacheFile
  TuningInspection                 public read-only cache/evidence inspection
```

The root package is the deliberate small public surface. It contains no backend, Runtime, Config,
cache-utility, registry, or private-internals subpackage.

## Task list

| ID | Task | Status | Depends on | One-line result or intent |
|---|---|---|---|---|
| 0001 | [Exact/default model-guided workload tuning and reusable cache](tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md) | Complete | CPU 0010E; Prepare 0004; caller-supplied stable identity, typed opaque collaboration, and operational cold measurement | Added cache-first deduplication, bounded miss-only measurement, deterministic selection, compact persistence, and separate evidence. |
| 0002 | [Bounded complete-plan tuning and model-plan cache](tasks/0002-bounded-complete-plan-tuning-and-model-plan-cache.md) | Complete | 0001; Complete CPU 0010J and Engine 0008A evidence | Added checked complete-plan correctness/timing, SESSION/PERSISTENT reuse, authenticated compact records, and rich evidence. |
| 0003 | [Read-only cache, plan, and evidence inspection](tasks/0003-read-only-cache-plan-and-evidence-inspection.md) | Complete | 0001–0002, stable artifact schemas | Added bounded redacted inspection with structural/key mismatch reasons and no decoding, execution, or mutation. |
| 0004 | [Tuning package Javadoc status reconciliation](tasks/0004-package-javadoc-status-reconciliation.md) | Complete | 0001–0003; Engine 0009/0013 | Corrected public two-phase CPU Engine composition and remaining limitations in package Javadoc. |

## Milestones and current frontier

Canonical workload reuse, bounded complete-plan selection, and artifact/evidence inspection are
`Complete` through 0003. Config 0006A–0006B and Engine 0009 provide the declarative request and
current CPU public composition around the generic tool; CPU 0010J remains the first complete-plan
producer. Documentation-only 0004 is `Complete`; its independent review confirmed the corrected
package status without changing authority or capability. No tuning task is `Ready`.

## Artifact, measurement, and lifecycle ownership

- Phase-1 candidate execution is complete and cold. Tuning owns warmups, positive odd sample
  counts, monotonic timing, integer-middle median comparison, and fail-before-measurement bounds.
- For `N` Phase-2 cache-miss candidates, `W` warmups, and `S` samples, checked preflight bounds
  exactly `N * (1 + W + S)`. All exact correctness executions and cleanup succeed before timing.
  A compatible cache hit executes no correctness, warmup, or timed trial.
- Compact workload and model-plan artifacts are bounded, versioned, checksummed, canonically
  ordered files with compatibility/schema identities and atomic replacement. Missing or
  incompatible entries are misses; corrupt, oversized, unsupported, duplicate, or invalid data
  fails safely without mutation.
- `SESSION` performs no model-plan-cache I/O. `PERSISTENT` reuse still requires the current
  producer/backend decoder to authenticate the decision and fresh preparation to build executable
  state. No prepared execution or native/JVM executable payload is serialized.
- Correctness references, publication bytes, Runtime resources, materialization, cleanup, selected
  preparation, and fallback execution remain with Engine/lifecycle owners. Tuning sees opaque
  actions and typed match/mismatch results.

## Live risks and open decisions

- Prepared-executable serialization remains deliberately unresolved and is not implied by either
  cache format.
- A future producer must expose complete valid typed candidates, compatibility/version identity,
  and file/lifecycle ownership without leaking private fields into shared orchestration.
- Main risks are duplicate local search in Phase 2, family-wide cache keys, shared interpretation
  of backend knobs, implicit or stale caches, unbounded trials, and treating caller labels as
  canonical model/plan identity.

## History and update policy

The former “later Config extension” and “later Engine composition” notes are closed by Config
0006B and Engine 0009. They are not open tuning work. Detailed audits, context identifiers,
validation logs, and completion evidence remain in linked task files and Git history.

Update this map only for a selected task, producer/composition gate, ownership or package change,
or live artifact/measurement risk. Keep evidence in task briefs. If a planning change conflicts
with `ARCHITECTURE.md`, stop and use the architecture-decision process.
