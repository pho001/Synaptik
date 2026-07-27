# Config Master Plan

## Goal

Define immutable, declarative configuration for compile, prepare, run, publication, planning
costs, and model-autotuning inputs.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Partition scoring](../../../architecture/partition-scoring.md)

## Scope

- compile modes and optimization configuration
- backend intent and partition scoring configuration
- prepare and run configuration
- planning cost profiles after their consumer and cost classification are stable
- model-autotuning objective, budget, constraint, and explicit-cache inputs after their consumers
  are stable

## Out of scope

- live services
- kernel types
- runtime state
- backend implementation logic
- benchmark runners, model-autotuning search, cache mutation, live discovery, and mutable
  measurement evidence

## Module invariants

- Configuration is declarative data.
- Concrete backends interpret backend-specific prepare settings.
- Configuration does not select executable implementations.

## Allowed dependencies

- JDK standard library and explicitly justified declarative contract types.
- `modules/backend-contract` for public backend requirement and identity values exposed by config.

## Forbidden dependencies

- concrete backend implementations
- runtime executable units and mutable runtime state

## Package structure

```text
io.github.pho001.synaptik.config/
  compile/  public compile mode, backend intent, optimization, and scoring configuration
  prepare/  later public backend-neutral and backend-class prepare configuration
  run/      later public invocation and publication configuration
  profile/  later immutable planning-cost inputs with a stable consumer
```

The module root is not a catch-all facade. Each package owns immutable declarative values for one
lifecycle concern. This map is progressive: task 0001 opens only `compile` with hard-requirement
intent, and later rows may refine their package contents before becoming Ready. Package placement
for model-autotuning request inputs waits for the stable prepare/tuning consumer rather than
inventing a public surface now.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Backend intent foundation](tasks/0001-backend-intent-foundation.md) | Complete | Completed backend-contract 0001–0004 and trace foundation | Replaced the placeholder with one immutable owner for an optional hard backend requirement, added the public backend-contract dependency, and preserved preference, scoring, profile, and evaluation work for later tasks. |
| 0002 | [Compile modes and graph optimization configuration](tasks/0002-compile-modes-and-graph-optimization-configuration.md) | Complete | 0001 | Added the exact three architecture-defined graph-scope modes and one stable optional-optimization permission without exposing graph passes or compiler behavior. |
| 0003 | [Partition scoring configuration](tasks/0003-partition-scoring-configuration.md) | Complete | 0001–0002, planning 0001 | Added one optional coarse `DeviceClass` preference as soft input for later comparison of already eligible ownership candidates, without evaluating candidates or choosing ownership. |
| 0004 | Planning cost-profile contract | Draft | 0001–0003, planning 0001–0003, stable backend-neutral cost classification | Define only immutable backend-neutral estimates required by the concrete ownership-scoring consumer; do not encode backend route or model-autotuning values. |
| 0005 | Compile configuration aggregate | Draft | 0001–0004 | Compose compile mode, backend intent, optimization, scoring, and any justified planning-cost inputs without compiler orchestration. |
| 0006 | Prepare and model-autotuning request configuration | Draft | 0005, stable prepare/tuning consumers | Define only the immutable objective, budget, constraints, representative profiles, fallback policy, and explicit-cache inputs required by stable consumers; do not expose backend candidate fields or own search/persistence. |
| 0007 | Run and publication configuration | Draft | 0005 | Define immutable invocation and publication options without runtime state or execution. |
| 0008 | Configuration contract closure | Draft | 0001–0007 | Audit validation, package/API cohesion, documentation, and dependency boundaries before planning begins. |


## Milestones

- Compile configuration
- Prepare and run configuration
- Profiles and validation

## Current status

In progress after the terminology and ownership reset. Tasks 0001–0003 remain Complete. The
unimplemented Config 0004 fixed-plus-linear platform/backend/tuning profile specification was
retired because it conflated planning cost with backend tuning and averaged unrelated workloads.
Config 0004 is again a Draft row without a detailed specification. It follows a stable planning
cost consumer and backend-neutral cost classification. Planning task 0003 is Complete, and its
exact baseline intentionally consumes no cost value or classification. Tasks 0005–0008 remain
Draft. Planning task 0004 is Complete with maximal consecutive same-owner grouping over completed
per-occurrence `BackendId` ownership and the current immutable model graph. It creates no cost-
bearing consumer. Planning task 0005 is Complete with logical materialization and memory
requirements derived without numeric cost, element/byte estimates, or profile input. Config 0004
therefore remains Draft without a detailed specification and no config task is Ready. Planning
0006 is Complete with a `CLOSED` documentation-only audit verdict. That closure does not define
the cost-bearing consumer or make Config 0004 Ready. Compiler task 0005 now consumes all four
completed standalone leaves through its package-private complete artifact entry: compile mode and
optimization configure graph work, while backend intent and scoring preference are passed to
Planning once per final node. It adds no config type, aggregate, default, profile, or public
compile entry, so Config 0004 remains Draft and no config status advances.

## Open questions

- Planning 0003 has stabilized the cost-free baseline ownership consumer. The first later concrete
  cost-bearing planning consumer must still establish the exact backend-neutral classification
  and units before Config 0004 can become Ready.
- Model-autotuning request inputs wait for stable prepare and tuning consumers. Workload and plan
  cache schemas, measurement evidence, and persistence remain with their lifecycle/tooling owners.
- Exact composition and defaults for compile, prepare, run, and publication aggregates remain for
  their owning tasks.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Backend intent owns optionality for one hard `BackendRequirement`; current Planning evaluates it
  for Compiler-supplied operation occurrences.
- Hard eligibility, ranking preference, planning cost, model autotuning, benchmarking, and runtime
  profiling are separate concepts.
- `tools/benchmarks` later produces observational reports from fixed workloads and never selects
  production settings. `tools/tuning` later coordinates one explicit model-autotuning workflow.
  Config stores only immutable request inputs after their consumers exist; it owns no cache or
  backend candidate vocabulary.
- A public config signature exposing a backend-contract type uses a public Gradle `api` edge and a
  focused architecture test.
- Compile mode uses exactly the architecture-defined `FORWARD_ONLY`, `FORWARD_AND_BACKWARD`, and
  `TRAINING_STEP` vocabulary without putting autograd or optimizer behavior in config.
- Graph optimization configuration exposes only whether optional semantics-preserving compiler
  work is enabled. It does not expose compiler pass identities or order, and mandatory correctness
  work remains outside that switch.
- Compiler task 0003 is the first current consumer of `GraphOptimizationConfig`. Its package-private
  transformation boundary uses the boolean only to enable or skip optional forward optimization;
  mandatory canonicalization and validation remain outside the permission. This consumer adds no
  config API, dependency, aggregate, mode interpretation, intent interpretation, or scoring work.
- Compiler task 0005 is the first current consumer of `CompileMode`, `BackendIntent`, and
  `PartitionScoringConfig` together with `GraphOptimizationConfig`. Its package-private direct
  entry validates them in declaration order, preserves each leaf's semantics, and adds no
  `CompileConfig`, profile, config-to-Planning dependency, public compiler facade, or default.
- Planning task 0001 intentionally interleaves before config task 0003. Capability answers the
  hard semantic question first; scoring configuration later describes how eligible ownership
  choices are compared and must not redefine capability.
- Task 0003 uses one `Optional<DeviceClass>` preference with `neutral()` and
  `preferring(DeviceClass)` factories. It is soft input after hard eligibility; it neither filters
  candidates nor guarantees selection.
- Task 0003 adds no `PartitionScoringPolicy`, numeric weights, preferred backend list, callback,
  candidate model, scoring evaluation, or profile data because no stable consumer or formula
  justifies those surfaces yet.
- Planning task 0002 evaluates this module's existing `BackendIntent` only through a package-
  private Planning entry point. It does not modify config, interpret the task-0003 soft
  preference, expose config in a public Planning signature, or make profile/scoring work current.
- The discarded Config 0004 records are not an implementation contract. A future planning cost
  profile contains only backend-neutral facts required by Planning; concrete routes, vectors,
  threads, chunks, tiles, and kernels belong to backend-owned candidate configurations.
- No stable shared production `OperationFamily` or workload-bucket contract exists. Config does
  not invent one, and it owns no runner, search algorithm, live discovery, or mutable evidence.
- Complete Planning task 0003 consumes `PartitionScoringConfig` internally after hard eligibility.
  Its preferred-class-first/provider-order baseline needs no candidate record, numeric score,
  cost input, or new config type, so Config 0004 remains Draft.

## Risks

- Embedding service objects or concrete implementation choices in configuration.
- Treating absence of a hard requirement as a fallback promise or a sentinel requirement.
- Mixing hard eligibility, preference, scoring, and tuning measurements into one broad intent
  object.
- Letting profile contracts own benchmarking, tuning algorithms, live platform discovery, or
  mutable measurement state.
- Treating one backend-wide fixed-plus-linear average as both a planning cost model and a tuning
  profile.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).

Task 0001 added only `BackendIntent`, compile-package documentation, the public backend-contract
edge, and focused config/architecture-test coverage. The final combined affected-suite command
passed seven tests across three suites with no failures, errors, or skips. Its independent
documentation pass finalized the affected Javadocs and explanatory/status documentation without
changing executable Java or repeating those tests. The single repository dependency checkpoint
passed 1,061 tests across 137 suites with no failures, errors, or skips.

Task 0002 deliberately keeps public optimization policy coarser than the compiler's internal pass
pipeline. `disabled` suppresses only optional optimization; `standard` permits a compiler-owned
semantics-preserving pipeline. The later `CompileConfig` aggregate, not these leaves, chooses
defaults.

Task 0002 passed its final 12-test/three-suite config module run with no failures, errors, or
skips. Its separate documentation pass finalized the two new type Javadocs, package Javadoc,
current-status explanations, glossary, and planning status; config Javadoc, repository Markdown,
exact fifteen-path, and whitespace validation passed without rerunning Java tests or changing
executable behavior. Repository-wide validation remains deferred to the config capability
checkpoint or continuous integration.

The next-frontier reassessment selected only planning task 0001 before config task 0003. This is an
ordering interleave, not a new config Gradle/module dependency or architecture change: config
tasks 0001–0002 remain Complete, and planning 0001 changed no config Java or build file. Planning
0001 is now Complete after its single final root suite passed. The following planning step made
only config task 0003 Ready with one detailed specification; that task is now Complete. Config
tasks 0004–0008 remain Draft without detailed specifications. Reassess the frontier rather than
assuming planning scoring can proceed while profile data remains undefined or assuming config
task 0004 is automatically next.

Task 0003 passed its final 17-test/four-suite config module run with no failures, errors, or skips.
Its separate documentation pass finalized the new type and package Javadocs, current-status
architecture/API/user-guide text, glossary terminology, and planning records. Config Javadoc,
repository Markdown, exact fourteen-path, status, later-spec, dependency, generated-page, and
whitespace validation passed without changing executable Java or rerunning the successful Java
suite. A later reassessment made only Planning task 0002 Ready for capability/availability/hard-
intent intersection. Config 0004 remains Draft without a detailed specification. After Planning
0002, Config 0004 profile contracts are the likely next area before Planning 0003 scoring, but
that selection requires a separate reassessment. Planning task 0002 is now Complete. A subsequent
reassessment drafted Config 0004, but the terminology/ownership reset rejected that unimplemented
design and removed its detailed specification. Planning 0003 is now Complete with a cost-free
baseline selector. Config 0004 remains Draft because no concrete cost-bearing consumer has
stabilized its classification or units. Planning 0004 is now Complete for owner-transition
grouping; it consumes no cost input and therefore does not advance Config 0004. A separate
reassessment made Planning 0005 Ready with a descriptor- and relationship-retaining logical plan.
That task is now Complete and likewise adds no cost quantity or configuration input, so Config
0004 remains Draft. No next task was made Ready as part of its implementation.
