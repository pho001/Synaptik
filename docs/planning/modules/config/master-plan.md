# Config Master Plan

## Goal

Define immutable, declarative configuration for compile, prepare, run, publication, platforms, and tuning.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Partition scoring](../../../architecture/partition-scoring.md)

## Scope

- compile modes and optimization configuration
- backend intent and partition scoring configuration
- prepare and run configuration
- platform, backend, and tuning profiles

## Out of scope

- live services
- kernel types
- runtime state
- backend implementation logic

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
  profile/  later immutable platform, backend, and tuning profile data
```

The module root is not a catch-all facade. Each package owns immutable declarative values for one
lifecycle concern. This map is progressive: task 0001 opens only `compile` with hard-requirement
intent, and later rows may refine their package contents before becoming Ready.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Backend intent foundation](tasks/0001-backend-intent-foundation.md) | Complete | Completed backend-contract 0001–0004 and trace foundation | Replaced the placeholder with one immutable owner for an optional hard backend requirement, added the public backend-contract dependency, and preserved preference, scoring, profile, and evaluation work for later tasks. |
| 0002 | [Compile modes and graph optimization configuration](tasks/0002-compile-modes-and-graph-optimization-configuration.md) | Complete | 0001 | Added the exact three architecture-defined graph-scope modes and one stable optional-optimization permission without exposing graph passes or compiler behavior. |
| 0003 | [Partition scoring configuration](tasks/0003-partition-scoring-configuration.md) | Complete | 0001–0002, planning 0001 | Added one optional coarse `DeviceClass` preference as soft input for later comparison of already eligible ownership candidates, without evaluating candidates or choosing ownership. |
| 0004 | Immutable platform, backend, and tuning profiles | Draft | 0001 | Define versioned, validated profile data consumed by later scoring and preparation; later tuning tooling produces it from repeatable benchmark evidence. |
| 0005 | Compile configuration aggregate | Draft | 0001–0004 | Compose compile mode, backend intent, optimization, scoring, and selected immutable profile inputs without compiler orchestration. |
| 0006 | Prepare configuration | Draft | 0005 | Define backend-neutral plus CPU/accelerator-class prepare data without concrete backend implementation behavior. |
| 0007 | Run and publication configuration | Draft | 0005 | Define immutable invocation and publication options without runtime state or execution. |
| 0008 | Configuration contract closure | Draft | 0001–0007 | Audit validation, package/API cohesion, documentation, and dependency boundaries before planning begins. |


## Milestones

- Compile configuration
- Prepare and run configuration
- Profiles and validation

## Current status

In progress after completing
[task 0003](tasks/0003-partition-scoring-configuration.md). Tasks 0001–0003 are Complete. Task 0003
adds only the optional soft coarse `DeviceClass` preference and no candidate evaluation, scoring
formula, owner selection, profile data, or separate policy type. Tasks 0004–0008 remain ordered
Draft work without detailed specifications. Planning task 0002 is Complete; its internal
consumption of hard intent changed no config Java or dependency surface. No config or global task
is Ready until a separate frontier reassessment.

## Open questions

- Profile identity, units, versioning, portability, measurement provenance, and persistence remain
  for task 0004; task 0001 adds no calibration field.
- Exact composition and defaults for compile, prepare, run, and publication aggregates remain for
  their owning tasks.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Backend intent owns optionality for one hard `BackendRequirement`; planning later evaluates it.
- Hard eligibility, ranking preference, and calibrated profile data are separate concepts.
- `tools/benchmarks` later produces repeatable measurements and reports; `tools/tuning` later turns
  selected evidence into validated immutable profile values owned by config.
- A public config signature exposing a backend-contract type uses a public Gradle `api` edge and a
  focused architecture test.
- Compile mode uses exactly the architecture-defined `FORWARD_ONLY`, `FORWARD_AND_BACKWARD`, and
  `TRAINING_STEP` vocabulary without putting autograd or optimizer behavior in config.
- Graph optimization configuration exposes only whether optional semantics-preserving compiler
  work is enabled. It does not expose compiler pass identities or order, and mandatory correctness
  work remains outside that switch.
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

## Risks

- Embedding service objects or concrete implementation choices in configuration.
- Treating absence of a hard requirement as a fallback promise or a sentinel requirement.
- Mixing hard eligibility, preference, scoring, and calibrated measurements into one broad intent
  object.
- Letting profile contracts own benchmarking, tuning algorithms, live platform discovery, or
  mutable measurement state.

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
that selection requires a separate reassessment. Planning task 0002 is now Complete, and that
reassessment has not made Config 0004 or another task Ready.
