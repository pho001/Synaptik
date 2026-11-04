# Task 0001: [Backend Intent](../../../../glossary.md#backend-intent--backendintent) Foundation

## Status

Complete

## Goal

Open `modules/config` with one [immutable](../../../../glossary.md#term-immutable), declarative [owner](../../../../glossary.md#term-owner) for the optional hard [backend
requirement](../../../../glossary.md#backend-requirement--backendrequirement) completed by [backend](../../../../glossary.md#backend)-contract task 0004.

[Mental model](../../../../glossary.md#term-mental-model):

```text
BackendIntent.unconstrained()
  -> no hard eligibility target

BackendIntent.requiring(requirement)
  -> exactly one retained hard eligibility target
```

[`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) records only whether a hard [target](../../../../glossary.md#term-target) is present. It does not evaluate that target,
[rank](../../../../glossary.md#term-ranking) candidates, express preference, contain calibrated platform data, or select [ownership](../../../../glossary.md#term-ownership-meanings) or an
implementation.

## [Scope](../../../../glossary.md#term-scope)

- Replace the config placeholder with public record [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) in
  `io.github.pho001.synaptik.config.compile`.
  - The record contains exactly one component:
    `Optional<BackendRequirement> hardRequirement`.
  - The [canonical](../../../../glossary.md#term-canonical-form) constructor rejects a null `Optional` with `NullPointerException` and exact
    message `hardRequirement`.
  - It accepts empty and present optionals, retains the exact supplied `Optional` reference, and
    retains the exact [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement) reference contained by a present value.
  - It exposes an explicitly documented `hardRequirement()` accessor returning the exact retained
    `Optional` reference.
  - It preserves ordinary record equality, hashing, and diagnostic `toString()` behavior.
- Add exactly two public static factories:
  - `BackendIntent unconstrained()` returns a new equal intent whose hard requirement is empty.
  - `BackendIntent requiring(BackendRequirement requirement)` rejects null with
    `NullPointerException` and exact message `requirement`, then returns a new intent containing
    the exact supplied requirement reference.
- `unconstrained` means only that no hard eligibility target constrains later planning. It does
  not promise a default backend, automatic discovery, fallback, preference, availability,
  capability, or successful ownership selection.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Add no preference to `BackendIntent`. Later [`PartitionScoringConfig`](../../../../glossary.md#partition-scoring-configuration--partitionscoringconfig) work owns ranking policy
  and preference after eligible candidates are known.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span>
- Add no calibrated platform, backend, or tuning values. Later profile tasks own immutable profile
  schemas; <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [schemas glossary](../../../../glossary.md#term-schema))</span>`tools/tuning` will eventually produce validated profiles from measurement evidence,
  while `tools/benchmarks` owns repeatable measurements and reports.
- Delete the `ConfigModule` placeholder and add package Javadoc for the current compile-
  configuration package and its planned boundaries.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [compile glossary](../../../../glossary.md#compile))</span>
- Add an `api(project(":modules:backend-contract"))` [dependency](../../../../glossary.md#term-dependency) to `modules/config` because the
  public record component exposes `BackendRequirement` to config consumers.
- Add one focused config test covering exact record/API shape, validation and messages, direct and
  factory construction, reference retention, equality, freshness, and absence of evaluation,
  preference, profile, service, or nested surfaces.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Add one focused architecture test covering the public config-to-backend-contract dependency and
  absence of config dependencies on concrete backends.<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- Finalize Javadocs and update Public API, Compile API, capability-provider guide,
  backend-selection guide, compile workflow guide, glossary, config master plan, completed
  backend-contract status, trace interleave status, and roadmap in the same overall change.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>
- Run a repository checkpoint after the implementation and documentation passes because the task
  adds a public inter-module dependency.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [module glossary](../../../../glossary.md#term-module))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- backend preference, preference targets, ordering, priority, score, weight, bonus, penalty,
  avoidance, fallback, retry, or policy<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [targets glossary](../../../../glossary.md#term-target); [weight glossary](../../../../glossary.md#term-weight))</span>
- [`PartitionScoringConfig`](../../../../glossary.md#partition-scoring-configuration--partitionscoringconfig), `PartitionScoringPolicy`, candidate evaluation, matching, filtering,
  no-match failure, ownership selection, partitioning, or logical memory planning<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [planning glossary](../../../../glossary.md#term-planning))</span>
- `CompileConfig`, [`CompileMode`](../../../../glossary.md#compile-mode--compilemode), graph optimization, training mode, publication policy, compile
  orchestration, graph capture, or compiler behavior<span data-glossary-navigation="true"> ([graph optimization glossary](../../../../glossary.md#forward-graph-optimization); [publication glossary](../../../../glossary.md#term-publication); [compile glossary](../../../../glossary.md#compile); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- `PrepareConfig`, CPU or accelerator prepare config, run options, runtime configuration,
  publication execution, or lifecycle behavior<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [accelerator glossary](../../../../glossary.md#term-accelerator); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [runtime glossary](../../../../glossary.md#term-runtime); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `PlatformProfile`, `BackendProfile`, `TuningProfile`, calibration values, platform fingerprint,
  profile versioning, persistence, serialization, file formats, measurement ingestion, benchmark
  execution, or tuning generation<span data-glossary-navigation="true"> ([fingerprint glossary](../../../../glossary.md#term-fingerprint); [persistence glossary](../../../../glossary.md#term-persistence); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- changing [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement) or any backend-contract declaration, validation, behavior, tests,
  permits list, or requirement semantics<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- availability or capability evaluation, backend registration/discovery/refresh, concrete backend
  lookup, live services, service locators, <span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [lookup glossary](../../../../glossary.md#term-lookup))</span>`ServiceLoader`, or reflection-based discovery
- concrete backend classes, kernel classes, routes, lowering, preparation, executable units,
  storage, physical memory, runtime state, or execution<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [lowering glossary](../../../../glossary.md#lowering))</span>
- [trace](../../../../glossary.md#trace) DTOs, payloads, attributes, translation, or emission
- dependency changes other than the exact public config-to-backend-contract edge and its focused
  architecture test<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- architecture contract, focused architecture explanations, ADR, Java version, root build,
  concrete backend, conformance, integration, another config capability, or detailed later task
  specifications<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Partition scoring](../../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition))</span>
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [No runtime service locator ADR](../../../../design/decisions/0006-no-runtime-service-locator.md)<span data-glossary-navigation="true"> ([service locator glossary](../../../../glossary.md#term-service-locator); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Backend guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Config master plan](../master-plan.md)
- [Backend-contract master plan](../../backend-contract/master-plan.md)
- [Trace master plan](../../trace/master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>
- [Backend-contract task 0004](../../backend-contract/tasks/0004-declarative-backend-requirements.md)
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Compile workflow guide](../../../../user-guide/compiling-graphs.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/config` owns immutable declarative configuration and may expose backend-contract values
  needed by its public contracts.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) owns only optionality for one hard requirement. Backend-contract owns the target
  vocabulary and planning later owns its evaluation.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [planning glossary](../../../../glossary.md#term-planning); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Ranking preference and calibrated profiles are separate configuration concepts with different
  origins and lifecycles; neither is embedded in the hard-requirement holder.<span data-glossary-navigation="true"> ([Ranking glossary](../../../../glossary.md#term-ranking))</span>
- Config must not retain a concrete backend, provider, service, executable, runtime state, kernel
  class, physical resource, or mutable measurement source.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [runtime glossary](../../../../glossary.md#term-runtime); [kernel glossary](../../../../glossary.md#kernel); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- The public record component requires an `api` [dependency](../../../../glossary.md#term-dependency) so downstream users compiling against
  `BackendIntent` can see [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement).
- No reverse [backend](../../../../glossary.md#backend)-contract-to-config dependency is added.
- Stop if implementation needs a second config production type beyond package documentation, a
  second dependency, a preference/profile decision, consumer behavior, another module API, or an
  architecture change.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Package impact

Package added:

- `io.github.pho001.synaptik.config.compile` — public immutable compile-time configuration values;
  this task adds only hard-requirement intent<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [compile glossary](../../../../glossary.md#compile))</span>

Type placement:

- `io.github.pho001.synaptik.config.compile.BackendIntent` — compile configuration owns whether a
  completed backend-contract hard target is present<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [target glossary](../../../../glossary.md#term-target))</span>

The root placeholder `io.github.pho001.synaptik.config.ConfigModule` is deleted. No generic
`util`, service, provider, [registry](../../../../glossary.md#term-registry), profile, [planning](../../../../glossary.md#term-planning), or implementation package is added. Future
[prepare](../../../../glossary.md#prepare), [run](../../../../glossary.md#term-run), and profile contracts will use cohesive packages recorded by the master plan rather
than accumulating unrelated types in the [module](../../../../glossary.md#term-module) root.

## Affected files

Production and build — exactly four paths:

- delete `modules/config/src/main/java/io/github/pho001/synaptik/config/ConfigModule.java`
- add `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/BackendIntent.java`
- add `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/package-info.java`
- update `modules/config/build.gradle.kts` only with
  `api(project(":modules:backend-contract"))`

Tests — exactly two paths:

- add `modules/config/src/test/java/io/github/pho001/synaptik/config/compile/BackendIntentTest.java`
- add
  `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/ConfigDependencyContractTest.java`

Documentation — exactly six paths:

- `docs/api/public-api.md`
- `docs/api/compile-api.md`
- `docs/backend-guide/capability-provider.md`
- `docs/user-guide/backend-selection.md`
- `docs/user-guide/compiling-graphs.md`
- `docs/glossary.md`

[Planning](../../../../glossary.md#term-planning) — exactly five paths:

- add and finalize this task
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/modules/backend-contract/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement) and its three variants/tests/Javadocs,
[`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot), all completed [backend](../../../../glossary.md#backend)-contract tasks, [trace](../../../../glossary.md#trace) Java/tests, config
consumers and placeholders outside this [module](../../../../glossary.md#term-module), `AGENTS.md`, `ARCHITECTURE.md`, focused
architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) documents, root build and settings, other module build files, concrete backends,
backend [conformance](../../../../glossary.md#term-conformance), integration tests, legacy evidence, and all later config tasks.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact seventeen paths above. Stop if implementation requires another production
type, test, document, [dependency](../../../../glossary.md#term-dependency), package, [module](../../../../glossary.md#term-module) [API](../../../../glossary.md#term-application-programming-interface-api), architecture change, consumer
implementation, or detailed follow-up specification. The user's standing instruction permits a
necessary [Javadoc](../../../../glossary.md#term-javadoc)-only path-count increase, but any expansion must be recorded and must not alter
a completed declaration or behavior.

## Acceptance criteria

- [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) is a public record with exactly one
  `Optional<BackendRequirement> hardRequirement` component, the canonical constructor, explicitly
  documented accessor, exactly <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>`unconstrained()` and `requiring(BackendRequirement)` as added
  public factories, ordinary record object methods, and no other project field, method, nested
  type, interface, builder, or factory.
- Direct construction rejects a null optional with exact message `hardRequirement`, accepts empty
  and present optionals, and returns the exact supplied optional and contained requirement
  references.
- `unconstrained()` produces fresh equal intents containing `Optional.empty()`.
- `requiring(...)` rejects null with exact message `requirement` and produces fresh equal intents
  whose optionals contain the exact supplied requirement reference.
- Neither construction path evaluates availability/capability, allocates IDs, discovers a
  backend, accesses a service, scores candidates, chooses ownership, or changes the supplied
  requirement.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- The API contains no preference, fallback, scoring, profile, calibration, benchmark, tuning,
  compile mode, optimization, prepare, run, publication, serialization, or execution surface.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [compile mode glossary](../../../../glossary.md#compile-mode--compilemode); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- The placeholder is removed and package Javadoc clearly identifies current hard-requirement
  optionality versus planned scoring, profiles, consumers, and lifecycle configuration.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `modules/config` declares exactly the new public `api` edge to `modules/backend-contract`; no
  concrete backend or other new dependency is added.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- The focused architecture test protects that dependency visibility and config independence from
  concrete backend projects without redefining architecture.<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- Existing backend-contract source, bytecode shape, behavior, tests, Javadocs, build, and
  dependency direction remain unchanged.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [shape glossary](../../../../glossary.md#shape); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- Public API, Compile API, capability-provider, backend-selection, compile workflow, and glossary
  identify <span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>`BackendIntent` as current while leaving `CompileConfig`, preference/scoring, profiles,
  planning evaluation, compiler, prepare, runtime, engine, and concrete backends planned.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [evaluation glossary](../../../../glossary.md#term-evaluation); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine))</span>
- Config master plan records the progressive package map, task 0001 as Complete, no Ready task,
  and ordered Draft follow-ups. The immutable platform/backend/tuning profile row explicitly
  records that tuning later produces profiles from measurement evidence; it is not specified in
  detail.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Backend-contract master plan, trace master plan, and roadmap record the config interleave without
  reopening completed backend-contract work or prematurely resuming trace payload work.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [trace payload glossary](../../../../glossary.md#trace-payload--tracepayload))</span>
- A separate clean-context documentation-focused pass finalizes all new Javadocs and the eleven
  documentation/planning paths after Java validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Final config and architecture-test validation, config Javadoc, exact scope, Markdown, final
  newlines, trailing whitespace, <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`, and the single repository [checkpoint](../../../../glossary.md#term-checkpoint) pass.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused config and architecture tests while developing. After executable Java and the build
edge stabilize, run exactly one final affected-suite command:

```bash
./gradlew :modules:config:test :testing:architecture-tests:test
```

Record suite and test counts from XML reports. Then hand the actual diff and exact Java evidence
to a separate clean-context documentation-focused agent in the same overall change. That pass
independently inspects the final source/tests, completed [backend requirement](../../../../glossary.md#backend-requirement--backendrequirement) contracts, build
edge, and [architecture test](../../../../glossary.md#term-architecture-test); applies General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend) guide, User guide, [Planning](../../../../glossary.md#term-planning),
and Example profiles as relevant; finalizes Javadocs and eleven documentation/planning paths;
records reasoned no-change conclusions; and runs:

```bash
./gradlew :modules:config:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun. Inspect the generated package and [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) pages,
confirm exactly seventeen paths, synchronize task 0001 to Complete with no [Ready](../../../../glossary.md#term-ready) row after the
[checkpoint](../../../../glossary.md#term-checkpoint) passes, and confirm no later task specification exists.

After both passes and final documentation stabilization, run the [dependency](../../../../glossary.md#term-dependency)-change checkpoint
exactly once:

```bash
./gradlew test
```

Record repository test/task results. Do not repeat the checkpoint in the documentation context.

## Dependencies

- Completed [model](../../../../glossary.md#term-model) milestone.
- Completed [trace](../../../../glossary.md#trace) tasks 0001–0002 as the stable diagnostic foundation.
- Completed [backend](../../../../glossary.md#backend)-contract tasks 0001–0004, especially the [sealed](../../../../glossary.md#term-sealed) [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement) family.

## Follow-up tasks

- [Compile mode](../../../../glossary.md#compile-mode--compilemode) and [graph](../../../../glossary.md#graph)-optimization configuration remain Draft.
- [Backend](../../../../glossary.md#backend)-neutral [partition](../../../../glossary.md#partition)-scoring policy and [ranking](../../../../glossary.md#term-ranking) preference remain Draft and separate from
  [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent).
- [Immutable](../../../../glossary.md#term-immutable) platform, backend, and tuning profile [schemas](../../../../glossary.md#term-schema) remain Draft. Later `tools/tuning`
  produces validated profiles from repeatable measurement evidence; `tools/benchmarks` owns
  measurements and reporting.
- Compile-config aggregation, prepare configuration, run/publication configuration, and config
  closure remain Draft without detailed specifications.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication))</span>
- Planning evaluation, compiler consumption, trace translation, preparation, runtime, engine,
  and concrete backend behavior remain in their owning later project areas.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [evaluation glossary](../../../../glossary.md#term-evaluation); [compiler glossary](../../../../glossary.md#term-compiler); [trace glossary](../../../../glossary.md#trace); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>

## Architecture impact

Expected impact: None.

The task realizes the existing allowed config-to-[backend](../../../../glossary.md#backend)-contract [dependency](../../../../glossary.md#term-dependency) and adds a focused
test for it. It changes no [ownership](../../../../glossary.md#term-ownership-meanings) rule, forbidden edge, [lifecycle](../../../../glossary.md#lifecycle) responsibility, backend
behavior, or architecture document. If implementation reveals that the public dependency or
hard-intent ownership conflicts with the [architecture contract](../../../../glossary.md#architecture-contract), stop before editing it.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, focused module/dependency/partition-scoring/no-service-locator
architecture documents, documentation/planning rules and profiles, roadmap, config,
backend-contract, and trace master plans, completed backend-contract task 0004, config task 0001,
current config placeholder/build, completed backend requirement source/tests/Javadocs, existing
architecture dependency tests, Public/Compile APIs, capability-provider/backend-selection/compile
workflow guides, glossary, and Java 26 root configuration.

Implement docs/planning/modules/config/tasks/0001-backend-intent-foundation.md exactly inside its
seventeen authorized paths. Replace only the config placeholder with exact BackendIntent and
package Javadoc, add only the public api dependency on backend-contract and its focused
architecture test, and preserve hard-requirement optionality/reference/value/no-evaluation
semantics. Add no preference, scoring, profile, calibration, compile aggregate, service,
consumer behavior, second config type, dependency, or later task. Stop on architecture,
dependency visibility, intent semantics, package, affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final combined config/architecture-test command
after executable Java stabilizes. Then hand the actual diff and Java evidence to a separate clean-
context documentation agent in the same overall change. That pass finalizes Javadocs, Public and
Compile APIs, capability-provider/backend-selection/compile guides, glossary, task/master/backend-
contract/trace/roadmap status, config Javadoc, and documentation/scope checks while reusing Java
evidence unless executable behavior changes. After both passes, run exactly one root test
dependency checkpoint. Mark config task 0001 Complete only after all criteria pass. Leave every
later config and trace task Draft without a detailed specification.
```

## Local decisions

- Represent optionality directly as `Optional<BackendRequirement>` in one record rather than a
  nullable component, absence sentinel, sealed automatic/required hierarchy, or mutable builder.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Name the component `hardRequirement` so callers cannot confuse eligibility with ranking
  preference.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span>
- Provide `unconstrained()` and `requiring(...)` as the only conveniences. `unconstrained` avoids
  the misleading implication that an `AUTO` requirement exists; it promises no selection result.
- Keep preference out of this record. Later scoring configuration can represent ranking without
  weakening or overloading the meaning of backend-contract requirements.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Keep calibrated profile data out of backend intent. Profiles have machine/platform identity,
  versioning, units, provenance, and lifecycle questions that deserve a cohesive later task.<span data-glossary-navigation="true"> ([backend intent glossary](../../../../glossary.md#backend-intent--backendintent); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Place [compile](../../../../glossary.md#compile) configuration under `config.compile` rather than growing the module root into a
  broad facade. Delete the root placeholder without replacing it with a module marker.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [facade glossary](../../../../glossary.md#term-facade))</span>
- Use [Gradle](../../../../glossary.md#term-gradle) `api`, not `implementation`, because a public config signature exposes a
  backend-contract type.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Add an architecture test in the same change because the task materializes a module dependency,
  even though the dependency direction was already authorized.<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test); [dependency glossary](../../../../glossary.md#term-dependency); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>

## Known limitations

- [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) cannot express preference, avoidance, ordered fallback, scoring weights, or
  multiple hard requirements.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight))</span>
- It does not evaluate whether its target is registered, available, capable, or preparable and
  defines no no-match exception.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- It is not yet accepted by `CompileConfig` because that aggregate remains planned.
- Platform/backend/tuning profiles, calibration generation, storage, versioning, and selection
  remain undefined.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- No [serialization](../../../../glossary.md#term-serialization) or external compatibility guarantee is established.

## [Validation](../../../../glossary.md#term-validation) evidence

- Planning inspected the authoritative config ownership and dependency rules, partition-scoring
  boundary, completed backend requirement contract, current config placeholder/build, current
  architecture dependency tests, current API/guide/glossary language, and Java 26 baseline.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency); [partition glossary](../../../../glossary.md#partition); [backend requirement glossary](../../../../glossary.md#backend-requirement--backendrequirement))</span>
- The public component makes `api(project(":modules:backend-contract"))` necessary; duplicating
  requirement targets or hiding them behind <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>`Object` would violate the completed contract and
  type-safety goal.
- The planned change contains four production/build, two test, six documentation, and five
  planning paths: exactly seventeen total.
- Planning validation passed across 220 Markdown files, 3,823 local links, 224 local anchors,
  2,762 fence markers, final newlines, and trailing whitespace.
- `git diff --check` passed. The planning diff contains exactly this task plus the config,
  backend-contract, and trace master plans and the roadmap; it changes no Java, test, Gradle,
  architecture, API, guide, or glossary file before implementation.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Repository [planning](../../../../glossary.md#term-planning) contains exactly one [`Ready`](../../../../glossary.md#term-ready) master-plan row: config task 0001. No config
  task 0002 or later detailed specification and no trace task 0003 specification exists.
- The implementation context `/root/implement_config_0001` added the exact record/package/build
  and focused-test changes. Its development-focused [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) and
  `ConfigDependencyContract` runs passed. After executable Java stabilized, exactly one final
  `./gradlew :modules:config:test :testing:architecture-tests:test` passed with eight actionable
  tasks (two executed and six up-to-date). XML reports contain seven tests across three suites
  with zero failures, errors, or skips: `BackendIntentTest` has five,
  `ConfigDependencyContractTest` has one, and `NnTrainingDependencyContractTest` has one.
- Final `javap` inspection confirms one public record component and field of type
  `Optional<BackendRequirement>`, the public [canonical](../../../../glossary.md#term-canonical-form) constructor, exactly the explicit
  `hardRequirement` accessor and `unconstrained`/`requiring` factories beyond ordinary record
  methods, and no added nested, service, preference, scoring, profile, or lifecycle surface.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The separate clean documentation context
  `/root/implement_config_0001/config_0001_docs` applied General style with API/Javadoc as the
  primary profile for <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`BackendIntent`, its package, Public [API](../../../../glossary.md#term-application-programming-interface-api), and [Compile](../../../../glossary.md#compile) API; [Backend](../../../../glossary.md#backend) guide for
  `capability-provider.md`; User guide for backend selection and compiling graphs; Planning for
  the task, master plans, and roadmap; and Example format for current construction examples. It
  independently inspected the architecture boundaries, documentation/planning rules, final
  source/tests/build, completed requirement family and availability snapshot, public/compile
  APIs, focused guides, glossary, relevant plans, architecture tests, and Java 26 configuration.<span data-glossary-navigation="true"> ([graphs glossary](../../../../glossary.md#graph); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- The documentation context changed only Javadoc and Markdown after the successful combined Java
  run. It did not change executable Java, tests, or Gradle behavior, so it reused the supplied
  seven-test evidence rather than rerunning a Java test suite.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `./gradlew :modules:config:javadoc` reported `BUILD SUCCESSFUL` in one second with three
  actionable tasks: config compile and Javadoc executed, and backend-contract compile was
  up-to-date. Inspection of generated `package-summary.html` and `BackendIntent.html` confirmed
  the rendered optional-target mental model, non-null optional rule, exact-reference semantics,
  both factory results and null failures, ordinary record semantics, unconstrained limitations,
  and current-versus-planned boundaries.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed 220 Markdown files, 3,825 local links, 226
  local anchors, 2,770 fence markers, final newlines, and trailing whitespace. `git diff --check`
  passed.
- The scope and status commands confirmed exactly seventeen authorized paths: four
  production/build paths including the deleted placeholder, two focused-test paths, six
  explanatory-documentation paths, and five planning paths. Config tasks 0002–0008 and trace
  tasks 0003–0008 remain Draft, no master-plan task is Ready, and no later config or trace task
  specification exists.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- No change was needed in `ARCHITECTURE.md`, focused architecture explanations, ADR 0006, or other
  architecture tests because this task realizes the already-authorized declarative config-to-
  backend-contract edge and does not change ownership or dependency rules. No change was needed
  in backend-contract Java, tests, Javadocs, build, or dependency direction because<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
  `BackendIntent` only consumes the completed requirement vocabulary. No trace Java or test
  change was needed because intent adds no diagnostic DTO or producer schema.<span data-glossary-navigation="true"> ([DTO glossary](../../../../glossary.md#data-transfer-object--dto); [producer glossary](../../../../glossary.md#term-producer); [schema glossary](../../../../glossary.md#term-schema))</span>
- No change was needed in config consumers outside the [module](../../../../glossary.md#term-module), `CompileConfig`, compiler,
  planning, prepare, runtime, engine, lifecycle implementations, concrete backends, backend
  conformance, integration, or legacy code because no current consumer accepts the standalone
  intent and all evaluation/execution behavior remains planned. Root build/settings and Java 26
  configuration required no change because the module inherits the established Java 26 build and
  adds only its permitted project dependency.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- After the documentation diff stabilized, the coordinator ran the required repository dependency
  checkpoint exactly once. <span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`./gradlew test` reported `BUILD SUCCESSFUL` in 883 milliseconds with
  44 actionable tasks: one executed and 43 up-to-date; the configuration cache was stored.
  Aggregated XML reports contain 1,061 tests across 137 suites with zero failures, errors, or
  skips.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Final status synchronization marks only config task 0001 Complete, leaves config tasks
  0002–0008 and trace tasks 0003–0008 Draft, and leaves no Ready task or detailed later config or
  trace task specification.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Implementation notes

Implementation, focused [validation](../../../../glossary.md#term-validation), independent documentation review, documentation validation,
and the single repository [dependency](../../../../glossary.md#term-dependency) [checkpoint](../../../../glossary.md#term-checkpoint) are complete. No executable Java changed after
the supplied combined test [run](../../../../glossary.md#term-run).

## Completion summary

Implemented and documented the exact standalone [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) contract, public [backend](../../../../glossary.md#backend)-contract
edge, and focused [dependency](../../../../glossary.md#term-dependency) enforcement within all seventeen authorized paths. The affected
suites, config [Javadoc](../../../../glossary.md#term-javadoc), generated-page inspection, Markdown, [scope](../../../../glossary.md#term-scope), whitespace, and single
repository dependency [checkpoint](../../../../glossary.md#term-checkpoint) passed. Existing architecture, backend-contract behavior, [trace](../../../../glossary.md#trace)
Java, consumers, [lifecycle](../../../../glossary.md#lifecycle) implementations, concrete backends, and other [modules](../../../../glossary.md#term-module) remain
unchanged. Config tasks 0002–0008 and trace tasks 0003–0008 remain Draft with no detailed later
specification.

Status: Complete
