# Config Master Plan

## Goal

Keep compile, prepare, run, publication, planning-cost, and model-autotuning requests as immutable
declarative data. Config records caller intent; it never performs ownership, route, candidate, or
executable selection.

## Authority and contracts

[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative. The exact applicable headings
are [`modules/config`](../../../architecture/contracts/foundational-modules.md#modulesconfig),
[`modules/prepare`](../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
[Concrete backend modules](../../../architecture/contracts/backend-execution.md#concrete-backend-modules), and
[Performance evidence and optimization tooling](../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
The [module-boundary](../../../architecture/module-boundaries.md),
[dependency](../../../architecture/dependency-rules.md), and
[partition-scoring](../../../architecture/partition-scoring.md) documents explain those contracts.

## Scope and non-goals

Config owns compile modes, backend intent, optimization and scoring inputs; the current graph-wide
numerical-profile identity; later backend-neutral prepare/run/publication values; planning-cost
inputs only after a cost-bearing consumer and units stabilize; and immutable model-autotuning
request policy and explicit cache locations.

It does not own live services, concrete backends or kernels, Runtime state or executable units,
candidate generation, measurement, selection, cache I/O or schemas, discovery, or mutable
evidence. An explicit cache path requests a location; it does not make Config a cache owner or
promise persistence.

## Stable invariants and dependencies

- Hard eligibility, soft preference, planning cost, tuning evidence, and route selection remain
  separate. An absent hard requirement is not a fallback promise.
- Config requests never select an owner, route, candidate, prepared plan, or executable.
- Numerical and determinism compatibility filter candidates before performance comparison.
  Hardware, availability, workload size, objectives, caches, and evidence never grant relaxed
  mathematics.
- Completed 0006 owns only the two-value graph numerical-profile identity. Complete Engine 0018
  owns coordinated propagation, strict default selection, and backend plan/cache identity.
  Complete CPU 0017 and Metal 0015 separately own their first backend realizations.
- Config may depend on the JDK and explicitly justified declarative contracts. Its public backend
  identity/requirement surface uses `modules/backend-contract`; it has no concrete-backend or
  Runtime dependency.

## Package map

```text
io.github.pho001.synaptik.config/
  compile/  compile mode, numerical profile, hard backend intent, optimization, and scoring inputs
  prepare/  planned backend-neutral determinism and other prepare permissions
  run/      planned invocation and publication inputs
  profile/  planned planning-cost inputs after a stable cost-bearing consumer
  tuning/   implemented immutable model-autotuning request policy and cache paths
```

The root is not a catch-all facade. Each package owns declarative values for one lifecycle concern.
The `tuning` package does not depend on `tools/tuning`.

## Task list

| ID | Task | Status | Depends on | One-line result or intent |
|---|---|---|---|---|
| 0001 | [Backend intent foundation](tasks/0001-backend-intent-foundation.md) | Complete | Completed backend-contract 0001–0004 and trace foundation | Added one immutable optional hard backend requirement and the required public backend-contract edge. |
| 0002 | [Compile modes and graph optimization configuration](tasks/0002-compile-modes-and-graph-optimization-configuration.md) | Complete | 0001 | Added the three graph-scope modes and one optional-optimization permission without exposing passes. |
| 0003 | [Partition scoring configuration](tasks/0003-partition-scoring-configuration.md) | Complete | 0001–0002, planning 0001 | Added an optional `DeviceClass` preference for ranking already eligible owners without choosing one. |
| 0004 | Planning cost-profile contract | Draft | 0001–0003, planning 0001–0003, stable backend-neutral cost classification | Define only immutable facts required by a concrete cost-bearing Planning consumer. |
| 0005 | Compile configuration aggregate | Draft | 0001–0004 | Compose justified compile leaves without compiler orchestration or invented defaults. |
| 0006 | [Explicit graph numerical-profile identity](tasks/0006-explicit-graph-numerical-profile-identity.md) | Complete | Model 0027 | Added the public two-constant identity in `config.compile`, exact shape/identity tests, and package/public documentation with no semantic interpretation, default, propagation, or backend logic. |
| 0006A | [Model-autotuning request configuration](tasks/0006a-model-autotuning-request-configuration.md) | Complete | 0001–0003; tools/tuning 0001; explicit staged ordering exception around Draft 0004–0006 | Added Phase-1 objective, budget, profile identity, fallback policy, and workload-cache path. |
| 0006B | [Complete-plan autotuning request configuration](tasks/0006b-complete-plan-autotuning-request-configuration.md) | Complete | 0006A; tools/tuning 0002–0003; CPU 0010J; Engine 0008A; second staged ordering exception around Draft 0004–0006 | Added independent Phase-2 bounds and model-plan-cache path without changing Phase-1 meanings. |
| 0007 | Run and publication configuration | Draft | 0005 | Define immutable invocation and publication options without execution state. |
| 0008 | Configuration contract closure | Draft | 0001–0007, including 0006A–0006B | Audit validation, API/package cohesion, documentation, and dependency boundaries. |

## Milestones and current frontier

The ordered milestones are compile configuration, prepare/run configuration, then profiles and
closure. The area is deliberately interleaved: 0001–0003 and the independently staged 0006A–0006B
are `Complete`; 0004–0005 and 0007–0008 remain `Draft`. The two completed exceptions did not
advance or reorder those Draft rows.

Model 0027 is Complete at implementation commit `ff86a302` after independent validation passed
1,114 Model tests and nine architecture tests with zero failures or skips and mandatory Class C
review returned `APPROVE` with zero findings. Config 0006 is Complete at implementation `314e049`
plus remediation `37e9e9db`; its fresh test/Javadoc run passed 33 tests with zero failures, errors,
or skips, generated enum/package pages and baseline-aware Markdown evidence passed inspection, and
independent Class B review returned final `APPROVE` with zero findings. Engine 0018 is Complete at
implementation `ce7a7dfa` plus remediation `07a01b9c`; CPU 0017 and Metal 0015 are Complete after
independent Class C approval and the combined serial checkpoint. Metal 0016 is Blocked by its
unchanged unary gate; exact both-profile Metal 0019 is the next Ready backend successor. Config
0004 still waits for a concrete cost-bearing Planning consumer; 0005
follows 0004. Config 0007 still follows 0005, and 0008 closes the full ledger.

## Live gates, risks, and open decisions

- The Phase-1 and Phase-2 budgets are independent. No later composition may reinterpret the
  Phase-1 miss/candidate/timing fields as plan or total-execution bounds.
- Config 0006A/0006B own request vocabulary only. Tuning owns measurement, deterministic
  selection, cache behavior and artifacts, and rich evidence; backend producers own candidates
  and codecs; Engine owns lifecycle composition and fallback execution.
- Safe heuristic fallback is not relaxed numerical permission. `STRICT_IEEE` remains the future
  Engine default; neither Config identity nor tuning evidence enables `ACCELERATOR` implicitly.
- Compile, prepare, run, and publication aggregate shapes and defaults remain with their Draft
  rows. Current consumers may compose completed leaves directly without manufacturing an
  aggregate prerequisite.
- Main risks are embedding services or implementations in data, mixing eligibility with ranking
  or evidence, and reviving a backend-wide average as both a Planning cost model and tuning input.

## History and update policy

The retired fixed-plus-linear Config 0004 proposal is not an implementation contract: it mixed
Planning cost with backend tuning and averaged unrelated workloads. Detailed completion evidence,
prior frontier stories, and ordering audits remain in linked task files and Git history.

Update this map only for task order/status/result, dependencies, package direction, a live gate or
risk, or an explicit ordering exception. Keep evidence and execution logs in task briefs. If a
planning change conflicts with `ARCHITECTURE.md`, stop and use the architecture-decision process.
