# Task 0006: Prepared Execution Aggregate

## Status

Complete

## Goal

Implement the smallest immutable Runtime-owned root for the exact reusable prepared state that
exists today. One `PreparedExecution` retains one exact `PreparedMemoryPlan` and one exact
`PreparedSchedule`, and proves by reference identity that the schedule belongs to that plan.

The exact public surface is:

```java
package io.github.pho001.synaptik.runtime.execution;

public record PreparedExecution(
        PreparedMemoryPlan memoryPlan,
        PreparedSchedule schedule) {}
```

This is an aggregate and consistency contract only. It has no run method, lifecycle callback,
resource acquisition, ownership transfer, or configuration input. All invocation mutation and
run-owned resource cleanup remain in a distinct `RunState` for each active logical run.

## Mental model

```text
exact PreparedMemoryPlan + exact same-plan PreparedSchedule
                         -> PreparedExecution

one reusable PreparedExecution
  -> later run A uses distinct RunState A
  -> later run B uses distinct RunState B
```

The aggregate closes the current prepared-recipe root without making that root a runner or a
resource owner. The two current components are sufficient because the schedule already retains
every current executable occurrence and each executable already retains the same exact plan.

## Scope

- Add the exact two-component public record above in `runtime.execution`.
- Require `memoryPlan` and `schedule` to be non-null, in component order.
- Require `schedule.memoryPlan() == memoryPlan`; structural equality is insufficient.
- Retain and return the exact two supplied immutable references without copying or wrapping them.
- Preserve ordinary record equality, hashing, and diagnostic text over the two components.
- Document the aggregate as immutable, reusable, thread-safe prepared recipe state that owns no
  current closeable or per-run resource.
- Add one focused test suite covering exact surface, validation order and messages, reference
  identity, retention, record semantics, action absence, reuse, and forbidden mechanisms.
- Update `runtime.execution` package Javadoc for the current prepared root while preserving the
  existing executable and bound-invocation contracts.
- Finalize Runtime/Public API status, the focused Runtime/Prepare/backend implementation status,
  glossary impact, and planning evidence in a separate clean documentation-focused context.

## Out of scope

- `run(...)`, a runner, schedule consumption, cold binding, invocation creation, or execution
- `RunOptions`, run/publication configuration, `PrepareConfig`, or any Config consumption
- `RunState` creation or mutation, input binding, allocation, representation creation, physical
  access, validity, residency, transfer, materialization, publication, or `RunResult`
- `AutoCloseable`, `close()`, an empty lifecycle method, cleanup orchestration, or failure rollback
- immutable persistent prepared resources, native handles, their ownership, or their lifecycle
- `PreparedPartition`, Prepare source associations, public Prepare orchestration, or Engine
  composition
- a separate executable list, `PreparedUnit`, generic payload, resource collection, metadata map,
  builder, factory, manager, service, registry, or facade
- concrete backends, lowering, route/kernel choice, backend discovery, fallback, tuning/cache
  behavior, profiling, or tracing emission
- graph, Compiler, Planning, Model, Prepare, Engine, concrete-backend, or upstream identifier types
  in Runtime production
- changes to existing Runtime behavior, Config, Prepare Java, other module Java, Gradle,
  dependencies, architecture contracts, ADRs, architecture tests, backend conformance, or
  integration tests
- Runtime 0007+, Prepare 0003, or another detailed task specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core lifecycle and invariants
  - `modules/runtime`
  - Prepare lifecycle
  - Run lifecycle
  - Dependency rules
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Runtime API](../../../../api/runtime-api.md)
- [Runtime master plan](../master-plan.md)
- [Runtime 0005](0005-prepared-schedule-contract.md)

## Architecture constraints

- `PreparedExecution`, its current plan/schedule/executable recipes, and any future immutable
  persistent prepared resources are immutable and reusable across runs.
- Every active complete logical run has exactly one distinct mutable `RunState`; no run binding,
  validity, residency, ownership transition, or cleanup state may enter this aggregate.
- The record owns only its own two references. It does not own or close its plan, schedule,
  executables, future run states, representations, invocations, or resources.
- The exact plan reference is the consistency identity already used by `PreparedExecutable`,
  `RunState`, and `PreparedSchedule`; this task must not introduce a second equality rule.
- Runtime remains independent of Prepare, Planning, Compiler, Model, Engine, and concrete
  backends. Existing Runtime Gradle dependencies remain unchanged.
- Runtime executes prepared schedules only, but this task does not implement that execution.
- Config 0007 is not a dependency. Its future run/publication values belong to runner or
  publication consumption and do not affect the identity of reusable prepared state.
- If implementation requires a lifecycle owner, persistent prepared resource, configuration,
  another aggregate component, module edge, or architecture change, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.runtime.memory` — supplies the exact immutable plan component.
- `io.github.pho001.synaptik.runtime.schedule` — supplies the exact immutable schedule component.

Package changed:

- `io.github.pho001.synaptik.runtime.execution` — owns reusable prepared execution recipes and
  their per-run bound invocation boundary; the complete current prepared root belongs beside
  those contracts rather than in the module root or the schedule package.

Type placement:

- `io.github.pho001.synaptik.runtime.execution.PreparedExecution` — Runtime owns the immutable
  runtime-ready aggregate, while Prepare will later construct it and Engine will later expose it.

Tests mirror `runtime.execution`. No package is added, moved, or removed.

## Exact API and consistency contract

`PreparedExecution` is a public final record with exactly these components, in order:

1. `PreparedMemoryPlan memoryPlan`
2. `PreparedSchedule schedule`

Its compact canonical constructor validates in this exact order:

1. require `memoryPlan` non-null;
2. require `schedule` non-null; and
3. require `schedule.memoryPlan() == memoryPlan`.

Exact failures are:

- `NullPointerException("memoryPlan")`;
- `NullPointerException("schedule")`; and
- `IllegalArgumentException("schedule memory plan does not match prepared execution memory plan")`.

The constructor reads `schedule.memoryPlan()` only after both null checks. It does not
compare with `equals`, traverse steps, revalidate executables, copy either component, create a
`RunState`, or perform any physical or lifecycle action. Successful construction retains the
exact references. Add no factory, builder, overload, custom object method, run method, lifecycle
method, nested type, interface, or package-private helper.

## Reference, ownership, immutability, and thread safety

- `memoryPlan()` and `schedule()` return the exact non-null references supplied at construction.
- The record is immutable because both retained current components are immutable contracts and
  the record exposes no mutation. It performs no defensive copy because there is no mutable
  container component at this boundary.
- One instance may be shared across threads to prepare distinct later runs. This does not make a
  `RunState` or `BoundInvocation` thread-safe and does not permit sharing run-owned mutable state.
- The aggregate transfers no ownership, acquires no resource, is not `AutoCloseable`, and has no
  failure cleanup obligation.
- A future persistent prepared-resource capability requires its own explicit ownership and
  partial-construction failure contract; it must not be anticipated with an empty lifecycle here.
- Ordinary record equality and hashing are structural over the retained components. Exact plan
  reference identity remains the constructor's association rule; equality is not a substitute
  for that validation. Record text is diagnostic, not serialization.

## Performance semantics

- Construction and successful access are constant-time.
- Construction allocates only the ordinary record object and performs two null checks, one
  schedule-plan accessor call, and one reference comparison.
- No list/array snapshot, traversal, map lookup, reflection, string dispatch, graph inspection,
  backend discovery, resource lookup, synchronization, or identifier allocation occurs.
- The record adds no hot-path method. A later runner must cold-consume the schedule and preserve
  the established direct-reference execution path.

## Affected files

Expected Runtime production/test paths:

- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java`
- `modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/package-info.java`
- `modules/runtime/src/test/java/io/github/pho001/synaptik/runtime/execution/PreparedExecutionTest.java`

Expected explanatory documentation paths:

- `docs/api/runtime-api.md`
- `docs/api/public-api.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` — implementation-status clarification
  only; no architecture rule change
- `docs/glossary.md`

Expected planning paths:

- this task
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/modules/prepare/master-plan.md` — retain the already-selected dependency and
  current-frontier correction; do not create or advance Prepare 0003
- `docs/planning/roadmap.md`

Review only unless a concrete contradiction is found: `AGENTS.md`, `ARCHITECTURE.md`, ADRs
0010/0011, lifecycle/module/dependency architecture docs, Runtime 0001–0005, Prepare 0001–0002,
the Prepare and Config master plans, current Runtime/Prepare source/tests/Javadocs, Runtime and
Prepare Gradle files, Config 0001–0003 source/tests, user/backend guides, architecture tests,
backend conformance, and integration tests.

## Maximum scope

At most exactly 11 paths:

- 3 Runtime production/test paths;
- 4 explanatory documentation paths; and
- 4 Runtime/Prepare/global planning paths.

No Java/test path outside Runtime, Prepare source, Config source, Gradle path, dependency,
architecture contract, ADR, architecture-test, backend-conformance, or integration path may
change. Stop if another type, component, behavior owner, package, module edge, or path is needed.
Do not create a later detailed task specification.

## Test requirements

`PreparedExecutionTest` must cover:

- exact public final record shape, component order/types, one public constructor, generated
  accessors, ordinary object methods, and absence of interfaces, nested types, and extra members;
- null validation in component order with exact messages;
- mismatched equal-but-distinct plans, exact reference comparison, and exact failure message;
- exact plan and schedule reference retention;
- ordinary equality, hashing, and diagnostic record text without treating text as serialization;
- valid empty and non-empty schedules;
- reuse by concurrent readers without mutation or per-run state; and
- absence of closing, binding, execution, allocation, resource action, ownership transfer,
  configuration, upstream identity, and forbidden implementation mechanisms.

Tests may use the smallest local `PreparedExecutable` subclass needed to construct a non-empty
schedule. They must not implement a runner, allocate representations, bind an invocation, or add
deferred semantics.

## Acceptance criteria

- The exact one-record surface exists in `runtime.execution` with no additional production type.
- Validation follows the exact order, reference comparison, exception types, and messages above.
- The record retains and returns the exact plan and schedule references.
- The aggregate is immutable, reusable, constant-time, and contains no per-run mutation or
  resource ownership.
- It does not implement `AutoCloseable`, `run`, prepare orchestration, schedule consumption,
  persistent prepared resources, Config consumption, or another later capability.
- Runtime production imports only the current Runtime plan and schedule contracts plus
  `java.util.Objects`; no forbidden upstream or concrete-backend type is added.
- Existing Runtime executable, schedule, memory, and run-state behavior remains unchanged.
- Every public member has complete Javadoc for purpose, inputs, results, nullability, exact
  reference retention, consistency, immutability, ownership, reuse, thread safety, performance,
  failures, and deliberate lifecycle exclusions.
- Runtime/Public API, focused boundary status, and glossary make the two-component aggregate
  current without claiming a runnable public lifecycle or revising architecture.
- A separate clean documentation-focused pass finalizes affected Javadocs, explanatory docs,
  glossary impact, links, terminology, and planning evidence in the same overall change.
- Runtime 0001–0005 and Prepare 0001–0002 remain Complete; Runtime 0006 becomes Complete only
  after all gates; Runtime 0007–0011 and Prepare 0003 remain Draft without detailed specs.
- Exactly the authorized 11 paths change, final newlines and whitespace are valid, and
  `git diff --check` passes.

## Tests / validation

Implementation-focused validation:

```bash
./gradlew :modules:runtime:test \
  --tests io.github.pho001.synaptik.runtime.execution.PreparedExecutionTest
./gradlew :modules:runtime:test
```

Run the focused command while developing and one final module command after executable Java
stabilizes.

Documentation-focused pass after final Javadocs and documentation:

```bash
./gradlew :modules:runtime:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/api/runtime-api.md \
  docs/api/public-api.md \
  docs/architecture/runtime-prepare-backend-boundary.md \
  docs/glossary.md \
  docs/planning/modules/runtime/tasks/0006-prepared-execution-aggregate.md \
  docs/planning/modules/runtime/master-plan.md \
  docs/planning/modules/prepare/master-plan.md \
  docs/planning/roadmap.md
git diff --check
```

If the temporary Markdown validator is absent, create an equivalent validator outside the
repository. Validate local targets and heading anchors, unique effective anchors, balanced
backtick and tilde fences, final newlines, and trailing whitespace.

Required source/scope/status checks:

- exact public record/component/constructor/accessor surface and package placement;
- exact validation order, messages, reference identity, and exact component retention;
- production import/mechanism scan for no maps, reflection, raw/unchecked access, string
  dispatch, registry/service locator, upstream/concrete types, lifecycle, resource, config, or
  run behavior;
- unchanged existing Runtime executable/schedule/run behavior and unchanged Gradle files;
- Java 26 root toolchain/release remains unchanged with no Runtime override;
- exact 11-path ceiling and no Java/test outside Runtime;
- task/master/roadmap status synchronization and Prepare master wording consistency;
- Runtime 0007–0011 and Prepare 0003 remain Draft, exactly Runtime 0001–0006 and Prepare
  0001–0002 have detailed specifications after this planning step, and no later spec exists; and
- final newlines, trailing whitespace, and `git diff --check`.

Repository-wide and architecture tests are deferred to the Runtime prepared-contract capability
checkpoint or continuous integration. This task changes one module without a dependency, build,
architecture, concrete-backend, or end-to-end behavior change. Backend-conformance and
integration tests are not applicable.

The documentation context reuses successful Runtime Java-test evidence unless it changes
executable behavior or records a concrete reason to repeat it.

## Dependencies

- Runtime 0002 `PreparedMemoryPlan` — Complete.
- Runtime 0003 `RunState` separation — Complete and preserved, but not a constructor component.
- Runtime 0004 prepared executable/cold-binding contracts — Complete and represented transitively
  through the schedule.
- Runtime 0005 `PreparedSchedule` — Complete and the direct aggregate dependency.
- Prepare 0002 finalization handoff — Complete and proves current executable/plan consistency;
  Runtime does not import it.
- ADR 0011 immutable-prepared/per-run ownership split — Accepted.

Config 0007, Prepare 0003, Runtime 0007+, concrete backends, Engine, run/publication options,
persistent prepared resources, and execution are not dependencies of this bounded aggregate.

## Follow-up tasks

- Runtime 0007 remains Draft for representation creation plus validity/residency foundations.
- Runtime 0008 remains Draft for transfer/materialization schedule steps after Runtime 0007.
- Runtime 0009 remains Draft for publication/result associations and schedule work.
- Runtime 0010 remains Draft for cold schedule consumption and dynamic execution.
- Runtime 0011 remains Draft for Runtime contract closure.
- Prepare 0003 remains Draft for public preparation orchestration and complete prepared-result
  validation using the stable Runtime aggregate.

Do not create any follow-up detailed specification in this task.

## Javadocs and documentation impact

- `PreparedExecution` Javadoc must explain the aggregate's role, exact-reference consistency,
  immutability, reuse, thread safety, ownership, constant-time construction, and the strict
  separation from `RunState` and later runner behavior.
- Its canonical constructor Javadoc must document both non-null exact references and all three
  caller-visible failures. Explicit record accessors must document exact retained-reference and
  non-null result semantics without adding another member.
- `runtime.execution` package Javadoc must make the prepared root current while preserving the
  cold-binding/hot-invocation distinction and labeling runner/publication/resource creation later.
- Runtime and Public API references must make the aggregate current without presenting
  `PreparedExecution.run`, `RunOptions`, `RunResult`, public Prepare orchestration, or Engine as
  implemented.
- The focused Runtime/Prepare/backend page receives implementation-status wording only. It must
  not restate or change architecture rules.
- The glossary's existing `PreparedExecution` entry becomes current and must distinguish the
  two-component implemented root from possible later persistent resources and run behavior.
- Lifecycle, module-boundary, dependency, other API/user/backend guides, completed task history,
  source outside the three Runtime paths, Gradle, and architecture/conformance/integration tests
  remain review-only because this task changes no architecture, dependency, public runnable
  workflow, physical resource, backend, or end-to-end behavior. Record reasoned no-change
  conclusions unless a concrete contradiction is found.

## Architecture impact

Expected impact: None.

The architecture already names `PreparedExecution` as immutable reusable runtime-ready state and
separates every run's mutation into `RunState`. This task implements only the smallest aggregate
over the exact current plan and schedule contracts. Stop if implementation needs a new ownership
rule, lifecycle, component, dependency, or architecture update.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, ADRs 0010/0011, the focused
Runtime/Prepare/backend lifecycle, module, dependency, and boundary documents, documentation
rules and General/API-Javadoc/Architecture/Planning profiles, the Runtime and Prepare master
plans, Runtime 0001–0005, Prepare 0001–0002, current Runtime and relevant Prepare source/tests,
Runtime/Public APIs, glossary, build files, and
docs/planning/modules/runtime/tasks/0006-prepared-execution-aggregate.md.

Implement Runtime 0006 exactly within its one-record surface and 11-path ceiling. Add only the
immutable exact-plan/exact-schedule PreparedExecution aggregate, focused tests/Javadocs, and the
specified current-status documentation. Do not add AutoCloseable/close, persistent resources,
RunOptions or Config consumption, RunState creation, allocation, representation creation,
residency, transfer, materialization, publication/result, binding/execution/runner behavior,
Prepare or Engine orchestration, generic payloads, another module type/edge, Gradle/architecture
changes, or later task specs. Stop on any architecture, package, API, validation, or scope
conflict.

Run the focused tests, one final Runtime module test, and all source/scope/status checks. Then
hand the actual diff and exact Java evidence to a separate clean documentation-focused context.
That pass must follow documentation-rules.md, independently finalize affected Javadocs/docs/
glossary/planning evidence, and not repeat Java tests unless executable behavior changes or a
concrete risk is recorded. Mark Complete only after every implementation and documentation gate
passes. Return both context IDs, exact paths, commands/results/counts, unresolved issues,
follow-up, and the repository completion status format.
```

## Local decisions

- Select exactly `memoryPlan` and `schedule` as components. The schedule already contains every
  current executable occurrence, so separate partition/executable collections would duplicate
  another owner's association or current schedule state.
- Place the aggregate in `runtime.execution`. That package owns reusable prepared execution
  recipes and the per-run invocation boundary; the schedule package continues to own only ordered
  step recipes, and the module root does not become a facade.
- Use exact plan reference identity, matching `PreparedExecutable`, `RunState`, and
  `PreparedSchedule`. A value-equal plan created separately is not the same prepared context.
- Keep the aggregate non-closeable. No current finalization contract may acquire a persistent
  closeable prepared resource, so an empty lifecycle would create a misleading obligation.
- Do not consume stable run configuration as a prerequisite. Configuration affects a later run
  or publication consumer, not the identity or consistency of current reusable prepared state.
- Keep `PreparedPartition` in Prepare. Runtime needs no compile/prepare partition association to
  represent or validate its current executable schedule.

## Known limitations

- The aggregate cannot be run, create a `RunState`, allocate or bind representations, transfer or
  materialize values, publish results, or expose a public lifecycle.
- It contains no persistent prepared resource collection or lifecycle. A later concrete need must
  define ownership and partial-failure cleanup explicitly.
- Current schedules contain executable occurrences only; later step variants require Runtime
  0007–0009 foundations.
- No current public Prepare orchestrator constructs the aggregate; Prepare 0003 remains Draft.

## Notes

- This specification is non-authoritative planning. `ARCHITECTURE.md` remains authoritative.
- Preserve all completed Runtime 0001–0005 and Prepare 0001–0002 history and evidence. Add Runtime
  0006 evidence without rewriting predecessor completion records.
- The Prepare master-plan edit in this overall change only removes stale Runtime 0005/0006
  dependency wording; it does not specify or advance Prepare 0003.

## Validation evidence

- Implementation context `019fbc26-0444-7a80-8d74-3f1a86670a57` ran the focused
  `./gradlew :modules:runtime:test --tests
  io.github.pho001.synaptik.runtime.execution.PreparedExecutionTest` command. Its final result
  passed one suite and 8 tests with zero failures, errors, or skips. An initial test-only
  expectation incorrectly treated separately constructed record values as unequal; the test was
  corrected to recognize ordinary structural record equality, and production remained unchanged.
- The same implementation context ran the single final `./gradlew :modules:runtime:test` after
  executable Java stabilized. It passed 10 suites and 82 tests with zero failures, errors, or
  skips. Its exact final `javap` inspection showed one public final record with two private final
  fields, the sole public `(PreparedMemoryPlan, PreparedSchedule)` constructor, both accessors,
  and ordinary record object methods. Import/mechanism scans, Java 26, and `git diff --check`
  passed before handoff.
- Documentation context `019fbc2b-b23e-7792-9c4a-b4b4c81ebc23` applied the General,
  API/Javadoc, Architecture, and Planning profiles. It inspected the final implementation diff,
  source, focused test, current plan/schedule/executable/run contracts, relevant Prepare
  contracts, architecture and ADR boundaries, Runtime/Public APIs, glossary, and planning state.
  It changed no executable Java and therefore reused the successful Java-test evidence without
  repeating either suite.
- The documentation context independently finalized the `PreparedExecution` and
  `runtime.execution` package Javadocs. They document exact reference retention and identity
  validation, immutability, concurrent reuse, ownership, constant-time construction/access,
  ordinary structural record semantics, failures, and the absence of run/resource behavior.
  `./gradlew :modules:runtime:javadoc` passed with `BUILD SUCCESSFUL`; five tasks were actionable,
  one executed and four were up-to-date.
- Runtime API, Public API, focused boundary status, and glossary now describe the exact current
  two-component root while keeping persistent resources, run configuration/state creation,
  representation/residency/transfer/materialization/publication, schedule consumption, public
  Prepare orchestration, Engine, results, and runner behavior planned.
- `python3 /tmp/validate_synaptik_markdown.py docs/api/runtime-api.md docs/api/public-api.md
  docs/architecture/runtime-prepare-backend-boundary.md docs/glossary.md
  docs/planning/modules/runtime/tasks/0006-prepared-execution-aggregate.md
  docs/planning/modules/runtime/master-plan.md docs/planning/modules/prepare/master-plan.md
  docs/planning/roadmap.md` passed with `validated 8 Markdown files`. It checked local targets and
  heading anchors, unique effective anchors, balanced backtick/tilde fences, final newlines, and
  trailing whitespace.
- The final source/mechanism audit found only Runtime plan, schedule, and `Objects` imports in
  `PreparedExecution`; no map, list, reflection, service loader, raw/unchecked payload, upstream
  or concrete-backend type, lifecycle, configuration, resource, run, binding, or execution
  mechanism is present. The exact constructor order, messages, identity comparison, accessors,
  and retention match the task and focused tests.
- `modules/runtime/build.gradle.kts` remains unchanged with only Config, Backend Contract, and
  Trace dependencies. Root `build.gradle.kts` alone retains Java toolchain and release 26; Runtime
  has no override. Existing executable, schedule, memory, run, and Prepare Java behavior is
  unchanged.
- The final scope inventory contains exactly the authorized 11 paths: three Runtime
  production/test paths, four explanatory documentation paths, and four planning paths. No other
  Java/test, Gradle, architecture contract/ADR/test, backend-conformance, integration, or module
  path changed.
- Status/specification checks confirm Runtime 0001–0006 and Prepare 0001–0002 are Complete;
  Runtime 0007–0011 and Prepare 0003 remain Draft; detailed specifications exist exactly for
  Runtime 0001–0006 and Prepare 0001–0002; and no later specification exists.
- Final `git diff --check` passed with no output.
- No-change conclusions:
  - `ARCHITECTURE.md`, ADRs 0010/0011, lifecycle, module-boundary, and dependency pages remain
    accurate because the task implements an already-defined immutable prepared root without a
    new rule, component owner, module edge, or runnable lifecycle. Only the focused boundary page
    needed current implementation-status wording.
  - Existing Runtime memory, resource, run, executable, invocation, and schedule Javadocs remain
    accurate because their contracts and behavior did not change. Only the new type and affected
    execution-package overview required finalization.
  - Prepare source/Javadocs/tests require no change because Prepare 0003, the future public
    constructor/validator of this aggregate, remains Draft; the Prepare master plan alone needed
    dependency/frontier synchronization.
  - Config, other API/user/backend guides, architecture tests, backend conformance, and
    integration tests need no update because the aggregate consumes no configuration, adds no
    public runnable workflow, changes no dependency, and implements no physical, backend, or
    end-to-end behavior.

## Implementation notes

- Added the exact `PreparedExecution(PreparedMemoryPlan, PreparedSchedule)` public record in the
  existing `runtime.execution` package. Its canonical constructor performs two ordered null
  checks followed by one schedule-plan reference comparison and retains both exact references.
- Added the focused eight-test contract suite and expanded execution-package documentation. No
  factory, builder, nested type, interface, custom object method, run/close/bind/execute member,
  resource collection, configuration, persistent resource, or helper was introduced.
- Synchronized Runtime/Public API, focused architecture status, glossary, Runtime/Prepare master
  plans, and roadmap without changing architecture authority or creating the next task spec.

## Completion summary

- Completed changes: implemented and documented the exact immutable two-reference
  `PreparedExecution` aggregate with exact-plan schedule consistency.
- Files changed or created: exactly the authorized 11 paths—three Runtime production/test paths,
  four explanatory documentation paths, and four planning paths.
- Tests and validation: reused the final focused 8-test and Runtime 82-test implementation
  evidence; Runtime Javadoc, eight-file Markdown, exact source/surface/import/mechanism/build/
  Java-26/scope/status/specification, final-newline/fence/whitespace, and `git diff --check` gates
  passed.
- Documentation-agent review: completed in clean context
  `019fbc2b-b23e-7792-9c4a-b4b4c81ebc23` without executable Java changes or repeated Java tests.
- Documentation impact: current Runtime/Public API, focused boundary status, glossary, and
  planning records now expose the two-component reusable root while keeping all run, resource,
  orchestration, and later schedule behavior planned.
- Javadoc review: finalized `PreparedExecution` and `runtime.execution` package contracts;
  reviewed the unchanged adjacent Runtime/Prepare Javadocs and found them accurate.
- Glossary impact: replaced the aspirational `PreparedExecution` entry with the implemented exact
  component, identity, ownership, concurrency, and lifecycle boundaries.
- Unresolved issues: None.
- Follow-up required: None for this task. Runtime 0007–0011 and Prepare 0003 remain Draft and
  require separate future planning steps.

Status: Complete
