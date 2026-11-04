# Config Master Plan

## Goal

Keep [compile](../../../glossary.md#compile), [prepare](../../../glossary.md#prepare), [run](../../../glossary.md#term-run), [publication](../../../glossary.md#term-publication), [planning](../../../glossary.md#term-planning)-cost, and [model](../../../glossary.md#term-model)-[autotuning](../../../glossary.md#autotuning--model-autotuning) requests as [immutable](../../../glossary.md#term-immutable)
declarative data. Config records caller intent; it never performs [ownership](../../../glossary.md#term-ownership-meanings), route, candidate, or
executable selection.

## Authority and contracts

[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative. The exact applicable headings
are [`modules/config`](../../../architecture/contracts/foundational-modules.md#modulesconfig),
[`modules/prepare`](../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
[Concrete backend modules](../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../glossary.md#concrete-backend); [modules glossary](../../../glossary.md#term-module))</span>, and
[Performance evidence and optimization tooling](../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
The [module-boundary](../../../architecture/module-boundaries.md),
[dependency](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([dependency glossary](../../../glossary.md#term-dependency))</span>, and
[partition-scoring](../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([partition glossary](../../../glossary.md#partition))</span> documents explain those contracts.

## [Scope](../../../glossary.md#term-scope) and non-goals

Config owns [compile](../../../glossary.md#compile) modes, [backend intent](../../../glossary.md#backend-intent--backendintent), optimization and scoring inputs; later
[backend](../../../glossary.md#backend)-neutral [prepare](../../../glossary.md#prepare)/[run](../../../glossary.md#term-run)/[publication](../../../glossary.md#term-publication) values; [planning](../../../glossary.md#term-planning)-cost
inputs only after a cost-bearing consumer and units stabilize; and [immutable](../../../glossary.md#term-immutable) [model-autotuning
request](../../../glossary.md#model-autotuning-request--preparation-result) policy and explicit [cache](../../../glossary.md#term-cache) locations.

It does not own live services, concrete backends or [kernels](../../../glossary.md#kernel), [Runtime](../../../glossary.md#term-runtime) state or executable units,
candidate generation, measurement, selection, cache I/O or [schemas](../../../glossary.md#term-schema), discovery, or [mutable](../../../glossary.md#term-mutable)
evidence. An explicit cache path requests a location; it does not make Config a cache [owner](../../../glossary.md#term-owner) or
promise [persistence](../../../glossary.md#term-persistence).

## Stable invariants and dependencies

- Hard eligibility, soft preference, planning cost, tuning evidence, and route selection remain
  separate. An absent hard requirement is not a fallback promise.<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>
- Config requests never select an [owner](../../../glossary.md#term-owner), route, candidate, prepared plan, or executable.
- Model's profile-free operation semantics constrain candidate eligibility before performance
  comparison. Hardware, availability, workload size, objectives, caches, and evidence never
  redefine those semantics.<span data-glossary-navigation="true"> ([Model glossary](../../../glossary.md#term-model); [operation glossary](../../../glossary.md#operation))</span>
- Model 0032 removed the numerical-profile identity and propagation introduced by historical
  Config 0006 and Engine 0018. Representative tuning-profile identity is unrelated and remains
  part of 0006A/0006B.<span data-glossary-navigation="true"> ([identity glossary](../../../glossary.md#term-identity-meanings); [Engine glossary](../../../glossary.md#term-engine))</span>
- Config may depend on the JDK and explicitly justified declarative contracts. Its public backend
  identity/requirement surface uses <span data-glossary-navigation="true"> ([JDK glossary](../../../glossary.md#term-java-development-kit-jdk); [backend identity glossary](../../../glossary.md#backend-identity--backendid))</span>`modules/backend-contract`; it has no concrete-backend or
  Runtime dependency.<span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend); [Runtime glossary](../../../glossary.md#term-runtime); [dependency glossary](../../../glossary.md#term-dependency))</span>

## Package map

```text
io.github.pho001.synaptik.config/
  compile/  compile mode, hard backend intent, optimization, and scoring inputs
  prepare/  reserved for concrete backend-neutral prepare requests; no current API
  run/      planned invocation and publication inputs
  profile/  planned planning-cost inputs after a stable cost-bearing consumer
  tuning/   implemented immutable model-autotuning request policy and cache paths
```

The root is not a catch-all [facade](../../../glossary.md#term-facade). Each package owns declarative values for one [lifecycle](../../../glossary.md#lifecycle) concern.
The `tuning` package does not depend on `tools/tuning`.

## Task list

| ID | Task | Status | Depends on | One-line result or intent |
|---|---|---|---|---|
| 0001 | [Backend intent foundation](tasks/0001-backend-intent-foundation.md)<span data-glossary-navigation="true"> ([Backend intent glossary](../../../glossary.md#backend-intent--backendintent))</span> | Complete | Completed [backend](../../../glossary.md#backend)-contract 0001–0004 and [trace](../../../glossary.md#trace) foundation | Added one [immutable](../../../glossary.md#term-immutable) optional hard [backend requirement](../../../glossary.md#backend-requirement--backendrequirement) and the required public backend-contract edge. |
| 0002 | [Compile modes and graph optimization configuration](tasks/0002-compile-modes-and-graph-optimization-configuration.md)<span data-glossary-navigation="true"> ([graph optimization configuration glossary](../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig))</span> | Complete | 0001 | Added the three [graph](../../../glossary.md#graph)-[scope](../../../glossary.md#term-scope) modes and one optional-optimization permission without exposing passes. |
| 0003 | [Partition scoring configuration](tasks/0003-partition-scoring-configuration.md)<span data-glossary-navigation="true"> ([Partition scoring configuration glossary](../../../glossary.md#partition-scoring-configuration--partitionscoringconfig))</span> | Complete | 0001–0002, [planning](../../../glossary.md#term-planning) 0001 | Added an optional [`DeviceClass`](../../../glossary.md#device-class--deviceclass) preference for [ranking](../../../glossary.md#term-ranking) already eligible [owners](../../../glossary.md#term-owner) without choosing one. |
| 0004 | Planning cost-profile contract | Draft | 0001–0003, planning 0001–0003, stable backend-neutral cost classification | Define only immutable facts required by a concrete cost-bearing Planning consumer. |
| 0005 | [Compile](../../../glossary.md#compile) configuration aggregate | Draft | 0001–0004 | Compose justified compile leaves without [compiler](../../../glossary.md#term-compiler) orchestration or invented defaults. |
| 0006 | [Explicit graph numerical-profile identity](tasks/0006-explicit-graph-numerical-profile-identity.md) | Complete | [Model](../../../glossary.md#term-model) 0027 | Historical implementation; its [identity](../../../glossary.md#term-identity-meanings) was removed by Model 0032. Not current [API](../../../glossary.md#term-application-programming-interface-api). |
| 0006A | [Model-autotuning request configuration](tasks/0006a-model-autotuning-request-configuration.md)<span data-glossary-navigation="true"> ([Model-autotuning request glossary](../../../glossary.md#model-autotuning-request--preparation-result))</span> | Complete | 0001–0003; tools/tuning 0001; explicit staged ordering exception around Draft 0004–0006 | Added Phase-1 [objective](../../../glossary.md#term-objective), budget, profile identity, fallback policy, and workload-[cache](../../../glossary.md#term-cache) path. |
| 0006B | [Complete-plan autotuning request configuration](tasks/0006b-complete-plan-autotuning-request-configuration.md)<span data-glossary-navigation="true"> ([autotuning glossary](../../../glossary.md#autotuning--model-autotuning))</span> | Complete | 0006A; tools/tuning 0002–0003; [CPU](../../../glossary.md#term-cpu) 0010J; [Engine](../../../glossary.md#term-engine) 0008A; second staged ordering exception around Draft 0004–0006 | Added independent Phase-2 bounds and model-plan-cache path without changing Phase-1 meanings. |
| 0007 | [Run](../../../glossary.md#term-run) and [publication](../../../glossary.md#term-publication) configuration | Draft | 0005 | Define immutable invocation and publication options without execution state. |
| 0008 | Configuration contract closure | Draft | 0001–0007, including 0006A–0006B | Audit [validation](../../../glossary.md#term-validation), API/package cohesion, documentation, and [dependency](../../../glossary.md#term-dependency) boundaries. |

## Milestones and current frontier

The ordered milestones are [compile](../../../glossary.md#compile) configuration, justified [prepare](../../../glossary.md#prepare)/[run](../../../glossary.md#term-run) configuration, then
closure. The area is deliberately interleaved: 0001–0003 and the independently staged 0006A–0006B
are `Complete`; 0004–0005 and 0007–0008 remain `Draft`. The two completed exceptions did not
advance or reorder those Draft rows.

No Config task is [Ready](../../../glossary.md#term-ready). Config 0004 still waits for a concrete cost-bearing [Planning](../../../glossary.md#term-planning) consumer;
0005 follows 0004, 0007 follows 0005, and 0008 closes the ledger. The profile-free [Model](../../../glossary.md#term-model) 0032
cutover is Complete and is not a pending prerequisite. Consult the roadmap for other workstreams
rather than interpreting this area's status as a repository-wide prohibition.

## Live gates, risks, and open decisions

- The Phase-1 and Phase-2 budgets are independent. No later composition may reinterpret the
  Phase-1 miss/candidate/timing fields as plan or total-execution bounds.
- Config 0006A/0006B own request vocabulary only. Tuning owns measurement, deterministic
  selection, cache behavior and artifacts, and rich evidence; backend producers own candidates
  and codecs; Engine owns lifecycle composition and fallback execution.<span data-glossary-navigation="true"> ([producers glossary](../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([cache glossary](../../../glossary.md#term-cache); [artifacts glossary](../../../glossary.md#term-artifact); [backend glossary](../../../glossary.md#backend); [Engine glossary](../../../glossary.md#term-engine); [lifecycle glossary](../../../glossary.md#lifecycle))</span>
- Safe heuristic fallback does not relax Model semantics. There is no current numerical-profile
  switch, strict/accelerator default, or tuning permission to bypass operation contracts.<span data-glossary-navigation="true"> ([heuristic glossary](../../../glossary.md#term-heuristic); [Model glossary](../../../glossary.md#term-model); [accelerator glossary](../../../glossary.md#term-accelerator); [operation glossary](../../../glossary.md#operation))</span>
- Compile, prepare, run, and publication aggregate shapes and defaults remain with their Draft
  rows. Current consumers may compose completed leaves directly without manufacturing an
  aggregate prerequisite.<span data-glossary-navigation="true"> ([Compile glossary](../../../glossary.md#compile); [prepare glossary](../../../glossary.md#prepare); [run glossary](../../../glossary.md#term-run); [publication glossary](../../../glossary.md#term-publication); [shapes glossary](../../../glossary.md#shape))</span>
- Main risks are embedding services or implementations in data, mixing eligibility with ranking
  or evidence, and reviving a backend-wide average as both a Planning cost model and tuning input.<span data-glossary-navigation="true"> ([ranking glossary](../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([embedding glossary](../../../glossary.md#embedding); [Planning cost model glossary](../../../glossary.md#term-planning-cost-model))</span>

## History and update policy

The retired fixed-plus-[linear](../../../glossary.md#term-linear-meanings) Config 0004 proposal is not an implementation contract: it mixed
[Planning](../../../glossary.md#term-planning) cost with [backend](../../../glossary.md#backend) tuning and averaged unrelated workloads. Detailed completion evidence,
prior frontier stories, and ordering audits remain in linked task files and Git history.

Update this map only for task order/status/result, dependencies, package direction, a live gate or
risk, or an explicit ordering exception. Keep evidence and execution [logs](../../../glossary.md#commit-release-distribution-log-and-seed-meanings) in task briefs. If a
planning change conflicts with `ARCHITECTURE.md`, stop and use the architecture-decision process.
