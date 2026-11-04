# Task 0003: [Partition Scoring Configuration](../../../../glossary.md#partition-scoring-configuration--partitionscoringconfig)

## Status

Complete

## Goal

Add the smallest [immutable](../../../../glossary.md#term-immutable) [compile](../../../../glossary.md#compile)-configuration value that lets later [planning](../../../../glossary.md#term-planning) observe an
optional soft preference between already eligible [ownership](../../../../glossary.md#term-ownership-meanings) candidates.

[Mental model](../../../../glossary.md#term-mental-model):

```text
hard requirement + capability + supplied availability
  -> later planning determines eligible candidates
  -> PartitionScoringConfig supplies an optional DeviceClass preference
  -> later planning compares candidates and selects backend ownership
```

The configuration records an input to later comparison. It does not determine eligibility,
enumerate candidates, calculate a score, or choose an [owner](../../../../glossary.md#term-owner), [device](../../../../glossary.md#term-device), [partition](../../../../glossary.md#partition), route, or [kernel](../../../../glossary.md#kernel).

## [Scope](../../../../glossary.md#term-scope)

- Add public record [`PartitionScoringConfig`](../../../../glossary.md#partition-scoring-configuration--partitionscoringconfig) in
  `io.github.pho001.synaptik.config.compile` with exactly one component:
  `Optional<DeviceClass> preferredDeviceClass`.
- Implement the [canonical](../../../../glossary.md#term-canonical-form)-constructor [validation](../../../../glossary.md#term-validation) and retention sequence exactly:
  1. reject a null `preferredDeviceClass` optional with `NullPointerException` and exact message
     `preferredDeviceClass`; and
  2. retain the exact validated `Optional` reference without copying, normalization, candidate
     lookup, or evaluation.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [lookup glossary](../../../../glossary.md#term-lookup); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- An empty optional means only that no explicit coarse device-class preference is supplied. It
  does not mean equal candidate scores, CPU default, accelerator default, automatic discovery,
  fallback, successful ownership selection, or absence of other later planning inputs.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [CPU glossary](../../../../glossary.md#term-cpu); [accelerator glossary](../../../../glossary.md#term-accelerator); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [planning glossary](../../../../glossary.md#term-planning))</span>
- A present optional requests a soft preference for the exact retained [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) value after
  hard eligibility has been established. It must not make another candidate ineligible, weaken a
  hard [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement), or guarantee that a candidate of the preferred class is selected.
- Preserve ordinary record equality, hashing, and diagnostic `toString()` behavior over the
  optional preference.
- Expose an explicitly declared and documented `preferredDeviceClass()` accessor that returns the
  exact retained optional reference.
- Add exactly two public static factories:
  - `PartitionScoringConfig neutral()` returns a new value with
    `Optional.empty()`; this is an explicit neutral configuration, not the selected default of the
    later `CompileConfig` aggregate.
  - `PartitionScoringConfig preferring(DeviceClass deviceClass)` rejects null with
    `NullPointerException` and exact message `deviceClass`, then returns a new value containing the
    exact supplied enum reference.
- Add no separate `PartitionScoringPolicy` type. The only currently justified policy input is the
  optional typed coarse-class preference. A policy hierarchy or enum with formula-level modes
  would either duplicate this value or speculate beyond a stable planning consumer.
- Add no numeric weights, generic score map, string key, backend identifier preference list,
  candidate list, callback, strategy implementation, registry, builder, service locator, or
  backend-specific field.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight); [backend glossary](../../../../glossary.md#backend); [callback glossary](../../../../glossary.md#term-callback); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- Keep calibrated platform, backend, and tuning measurements out of this record. Config task 0004
  owns those immutable profile schemas after their identity, units, provenance, and versioning
  decisions are ready.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [schemas glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [ready glossary](../../../../glossary.md#term-ready))</span>
- Update compile-package Javadoc to describe the four current standalone configuration values and
  distinguish hard eligibility from soft ranking input and later profile data.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Add one focused test covering exact record/API shape, constructor validation and message,
  direct reference retention, both factories, value behavior, and absence of eligibility,
  scoring, candidate, service, backend-route, profile, and lifecycle surfaces.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Finalize the new Javadoc and the focused current-status/API/user-guide/glossary/planning text in
  the same overall change through the required separate clean-context documentation pass.

## Out of [scope](../../../../glossary.md#term-scope)

- changing [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement), [`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot), [`BackendId`](../../../../glossary.md#backend-identity--backendid),
  [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), [`DeviceClass`](../../../../glossary.md#device-class--deviceclass), or their semantics, tests, Javadocs, build, or dependencies
- evaluating capability, supplied availability, a hard requirement, registration, or
  preparability; constructing a capability matrix; filtering candidates; or defining no-match
  failure<span data-glossary-navigation="true"> ([capability matrix glossary](../../../../glossary.md#review-tables-and-support-matrices))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span>
- weakening a hard requirement because its eligible result conflicts with the preferred device
  class<span data-glossary-navigation="true"> ([device class glossary](../../../../glossary.md#device-class--deviceclass))</span>
- enumerating candidate [backends](../../../../glossary.md#backend) or devices, accepting a preferred `BackendId` list, defining
  fallback order, or selecting a device<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- score calculation, score result types, score comparison, tie breaking, deterministic candidate
  order, owner selection, node or segment selection, partitioning, transfer planning,
  materialization planning, or logical memory planning<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [node glossary](../../../../glossary.md#node); [planning glossary](../../../../glossary.md#term-planning); [materialization glossary](../../../../glossary.md#materialization))</span>
- numeric transfer, boundary, materialization, accelerator, small-region, or other weights,
  bonuses, penalties, thresholds, formulas, normalization, units, or calibration values<span data-glossary-navigation="true"> ([accelerator glossary](../../../../glossary.md#term-accelerator); [weights glossary](../../../../glossary.md#term-weight); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- `PartitionScoringPolicy`, policy implementations, callbacks, strategies, registries, plugins,
  maps, string keys, generic weights, builders, loaders, serialization, or compatibility schemas<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [schemas glossary](../../../../glossary.md#term-schema))</span>
- `PlatformProfile`, `BackendProfile`, `TuningProfile`, benchmark evidence, platform fingerprint,
  persistence, measurement ingestion, tuning generation, or live platform discovery<span data-glossary-navigation="true"> ([fingerprint glossary](../../../../glossary.md#term-fingerprint); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- changing [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery), [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider), or adding a planning consumer,
  provider implementation, candidate model, ownership result, compiler orchestration, or trace
  schema<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [trace glossary](../../../../glossary.md#trace))</span>
- `CompileConfig`, aggregate defaults, compiler interpretation, graph behavior, prepare, run,
  publication, runtime, engine, backend lowering, route or kernel selection, executable state,
  storage, physical memory, or execution<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel))</span>
- dependencies, Gradle, Java version, architecture contract, architecture rules, ADRs,
  architecture tests, backend-conformance tests, integration tests, another module's executable
  behavior, or detailed later task specifications<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [architecture contract glossary](../../../../glossary.md#architecture-contract); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module))</span>
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Partition scoring](../../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Architecture style](../../../../developer-guide/documentation/architecture-style.md)
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Config master plan](../master-plan.md)
- [Planning master plan](../../planning/master-plan.md)
- [Trace master plan](../../trace/master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>
- [Config task 0001](0001-backend-intent-foundation.md)
- [Config task 0002](0002-compile-modes-and-graph-optimization-configuration.md)
- [Planning task 0001](../../planning/tasks/0001-operation-capability-query-foundation.md)
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Backend-selection guide](../../../../user-guide/backend-selection.md)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>
- [Compile workflow guide](../../../../user-guide/compiling-graphs.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/config` owns [immutable](../../../../glossary.md#term-immutable) declarative scoring inputs. `modules/planning` owns capability
  and hard-eligibility evaluation, candidate construction, comparison, score calculation, and
  ownership decisions.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Hard eligibility and soft [ranking](../../../../glossary.md#term-ranking) remain separate. [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) may remove candidates through
  a hard requirement; [`PartitionScoringConfig`](../../../../glossary.md#partition-scoring-configuration--partitionscoringconfig) must never remove a candidate or weaken that
  requirement.
- [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) is the existing backend-neutral coarse CPU/accelerator classification. Enum order
  has no preference meaning; only a present explicit configuration value supplies a class
  preference.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu); [accelerator glossary](../../../../glossary.md#term-accelerator))</span>
- Config may retain the classification value because its existing public backend-contract
  dependency already exposes declarative backend vocabulary. No dependency or visibility change
  is required.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Planning scoring uses compile-time facts only and selects backend ownership, not a device,
  executable implementation, route, or kernel.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [compile glossary](../../../../glossary.md#compile); [backend ownership glossary](../../../../glossary.md#backend-ownership); [device glossary](../../../../glossary.md#term-device); [kernel glossary](../../../../glossary.md#kernel))</span>
- Immutable calibrated profile data remains a distinct config concern because its identity,
  measurement units, provenance, portability, and versioning are not properties of a preference.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- Config must not retain a capability provider, live service, candidate object, planning callback,
  runtime or prepare state, concrete backend class, executable, kernel class, physical resource,
  or mutable measurement source.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [concrete backend glossary](../../../../glossary.md#concrete-backend); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Stop if implementation needs another production type, a planning consumer, a dependency or
  build edit, an exact scoring formula, a calibrated value, candidate semantics, ownership
  behavior, or an architecture decision.

## Package impact

Existing package extended:

- `io.github.pho001.synaptik.config.compile` — immutable public graph-compilation configuration
  leaves; task 0003 adds only a soft coarse device-class preference for later ownership scoring<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [device glossary](../../../../glossary.md#term-device); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

Type added:

- `io.github.pho001.synaptik.config.compile.PartitionScoringConfig` — configuration owns the
  optional declarative preference while planning later owns every evaluation and decision<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>

Existing [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), [`CompileMode`](../../../../glossary.md#compile-mode--compilemode), and [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig) remain unchanged. Package
[Javadoc](../../../../glossary.md#term-javadoc) is updated to describe the fourth current standalone value. No `PartitionScoringPolicy`,
subpackage, helper, builder, service, [registry](../../../../glossary.md#term-registry), candidate, score, profile, [planning](../../../../glossary.md#term-planning), [runtime](../../../../glossary.md#term-runtime), or
[backend](../../../../glossary.md#backend)-specific type is added.

## Affected files

The intended overall implementation [scope](../../../../glossary.md#term-scope) is exactly fourteen paths.

Production — exactly two paths:

- add
  `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/PartitionScoringConfig.java`
- [Javadoc](../../../../glossary.md#term-javadoc)-only update to
  `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/package-info.java`

Tests — exactly one path:

- add
  `modules/config/src/test/java/io/github/pho001/synaptik/config/compile/PartitionScoringConfigTest.java`

Architecture-status and explanatory documentation — exactly six paths:

- current-versus-planned wording only in `docs/architecture/partition-scoring.md`, without
  changing an architecture rule or defining the later scoring formula
- `docs/api/public-api.md`
- `docs/api/compile-api.md`
- `docs/user-guide/backend-selection.md`
- `docs/user-guide/compiling-graphs.md`
- `docs/glossary.md`

[Planning](../../../../glossary.md#term-planning) — exactly five paths:

- add and finalize this task
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/modules/planning/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: all current [backend](../../../../glossary.md#backend)-contract source/tests/Javadocs/build,
[`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), [`CompileMode`](../../../../glossary.md#compile-mode--compilemode), [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig), their tests and generated Javadocs,
config build, existing config architecture [dependency](../../../../glossary.md#term-dependency) test, planning capability source/tests/
Javadocs/build, [trace](../../../../glossary.md#trace) contracts, [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[prepare](../../../../glossary.md#prepare)/[engine](../../../../glossary.md#term-engine) placeholders, concrete backends,
`AGENTS.md`, `ARCHITECTURE.md`, focused architecture documents beyond the authorized status
sentence, ADRs, root build/settings, other [module](../../../../glossary.md#term-module) builds, backend-[conformance](../../../../glossary.md#term-conformance) tests, integration
tests, and all later task rows.

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly the fourteen paths listed above. Stop if implementation requires another production
type, test, document, [dependency](../../../../glossary.md#term-dependency), package, [module](../../../../glossary.md#term-module) [API](../../../../glossary.md#term-application-programming-interface-api), architecture change, scoring consumer,
profile [schema](../../../../glossary.md#term-schema), candidate or [ownership](../../../../glossary.md#term-ownership-meanings) behavior, or detailed follow-up specification. Do not use
a later task to hide an incomplete acceptance criterion.

## Required [Javadoc](../../../../glossary.md#term-javadoc) contracts

- The type Javadoc must define the value as an immutable soft ranking input applied only after
  hard eligibility, distinguish absence from a default/fallback/equal-score promise, state that a
  present preference does not make other candidates ineligible or guarantee selection, and list
  the evaluation, profile, ownership, route, prepare, runtime, and execution boundaries.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [evaluation glossary](../../../../glossary.md#term-evaluation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- The record-component and canonical-constructor Javadocs must state that the optional is
  non-null, the exact optional reference is retained, and the exact enum reference is retained
  when present.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Constructor Javadoc must document the exact `preferredDeviceClass` null failure and message and
  state that no snapshot, normalization, lookup, matching, or evaluation occurs.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [normalization glossary](../../../../glossary.md#term-normalization); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- `neutral()` Javadoc must state that it returns a new non-null value with no explicit device-class
  preference and does not select an aggregate default, owner, fallback, or successful result.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [owner glossary](../../../../glossary.md#term-owner))</span>
- `preferring(DeviceClass)` Javadoc must document the non-null input, exact reference retention,
  new non-null result, soft-only meaning, and exact `deviceClass` null failure and message.
- `preferredDeviceClass()` Javadoc must document the exact non-null retained optional reference
  and exact contained enum reference when present.
- Package Javadoc must identify all four current standalone compile values and keep current config
  construction separate from planned <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>`CompileConfig`, profiles, planning evaluation, compiler
  orchestration, preparation, runtime, engine, and backend implementation.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [compiler glossary](../../../../glossary.md#term-compiler); [engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend))</span>

## Acceptance criteria

- [`PartitionScoringConfig`](../../../../glossary.md#partition-scoring-configuration--partitionscoringconfig) is a public final record with exactly one
  `Optional<DeviceClass> preferredDeviceClass` component, one public canonical constructor, one
  explicitly documented accessor, exactly <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>`neutral()` and `preferring(DeviceClass)` as added
  public factories, ordinary record object methods, and no other project field, method,
  constructor, interface, nested type, builder, or factory.
- Direct construction rejects a null optional with exact message `preferredDeviceClass`, accepts
  empty and present optionals, and returns the exact supplied optional and contained enum
  references.
- Constructor validation occurs before retention, and valid construction performs no collection
  snapshot, normalization, lookup, matching, score calculation, or other evaluation.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [normalization glossary](../../../../glossary.md#term-normalization); [lookup glossary](../../../../glossary.md#term-lookup); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- `neutral()` produces fresh equal values with empty optionals and promises only absence of an
  explicit device-class preference.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- `preferring(...)` rejects null with exact message `deviceClass` and produces fresh equal values
  containing the exact supplied enum reference.
- A present preference remains soft: it neither filters an eligible candidate nor weakens a hard
  requirement, and it guarantees no selected backend, device, score, route, or execution result.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- No candidate enumeration, backend-ID preference list, fallback order, tie rule, numeric factor,
  profile/calibration value, score result, policy type, callback, map, string key, service,
  registry, planner, lifecycle, route, kernel, or executable surface is added.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [registry glossary](../../../../glossary.md#term-registry); [lifecycle glossary](../../../../glossary.md#lifecycle); [kernel glossary](../../../../glossary.md#kernel))</span>
- Existing [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), [`CompileMode`](../../../../glossary.md#compile-mode--compilemode), [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig), backend-contract contracts,
  and planning capability contracts retain their exact source/API/behavior/dependency boundaries.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Config build and all module dependencies remain unchanged; the existing config-to-backend-
  contract API edge is sufficient and no architecture test changes.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- Focused automated tests lock the exact generic record shape, validation/message/reference rules,
  factory behavior, record value behavior, and forbidden added surfaces. No recurring API-shape
  invariant is left to manual reflection or <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [invariant glossary](../../../../glossary.md#term-invariant))</span>`javap` checks.
- Package Javadoc, partition-scoring implementation status, Public API, Compile API,
  backend-selection guide, compile workflow guide, and glossary distinguish the current soft
  input from planned profiles, eligibility/scoring evaluation, ownership, compiler consumption,
  preparation, runtime, and execution.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [partition glossary](../../../../glossary.md#partition); [Compile glossary](../../../../glossary.md#compile); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Config master plan marks only task 0003 Ready before implementation. Config tasks 0001–0002 and
  planning task 0001 remain Complete; config 0004+, planning 0002+, and trace 0003+ remain Draft
  without detailed specifications.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [trace glossary](../../../../glossary.md#trace))</span>
- A separate clean-context documentation-focused pass finalizes the new type and package Javadocs
  plus the eleven explanatory/planning paths in the same overall change, reviews glossary impact,
  and reuses successful Java evidence unless executable Java changes.
- Exactly one final config module suite, config Javadoc, repository Markdown, generated-page
  inspection, exact fourteen-path scope, dependency no-change, status, later-spec absence, final
  newlines, trailing whitespace, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` [validation](../../../../glossary.md#term-validation) pass.
- No repository-wide test is run merely by habit: this task changes one module's declarative API
  without changing a dependency, build rule, architecture boundary, or cross-module executable
  behavior. The config capability checkpoint or continuous integration owns that tier.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development commands are allowed while Java changes. After executable Java stabilizes,
[run](../../../../glossary.md#term-run) exactly one final config [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:config:test
```

Record suite and test counts from XML reports. Then hand the actual diff and exact Java evidence
to the separate documentation-focused context described below. That context reuses the successful
Java evidence and does not rerun the suite unless it changes executable Java behavior or records a
concrete reason. After final [Javadoc](../../../../glossary.md#term-javadoc) and Markdown edits, it runs:

```bash
./gradlew :modules:config:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

Inspect generated `package-summary.html` and `PartitionScoringConfig.html`. Confirm exactly the
fourteen authorized paths; unchanged [Gradle](../../../../glossary.md#term-gradle) dependencies; task 0003 Complete only after all
implementation criteria pass; no global [Ready](../../../../glossary.md#term-ready) task after completion until a separate frontier
reassessment; no detailed config 0004, [planning](../../../../glossary.md#term-planning) 0002, or [trace](../../../../glossary.md#trace) 0003 specification; balanced
fences; final newlines; and no trailing whitespace.

Do not run repository-wide tests in this task. Defer them to the config capability [checkpoint](../../../../glossary.md#term-checkpoint) or
[continuous integration](../../../../glossary.md#term-continuous-integration-ci) because no [dependency](../../../../glossary.md#term-dependency), shared build rule, architecture boundary, or
cross-module executable behavior changes.

## Documentation [handoff](../../../../glossary.md#term-handoff)

After the implementation context records the final config test result, hand this task, the actual
diff, and exact test evidence to a separate clean-context documentation-focused agent or thread.
The handoff must identify [`PartitionScoringConfig`](../../../../glossary.md#partition-scoring-configuration--partitionscoringconfig) as the only new [API](../../../../glossary.md#term-application-programming-interface-api), its soft-after-eligibility
semantics, the exact fourteen-path limit, the unchanged dependencies and architecture rules, the
expected six explanatory documents and five [planning](../../../../glossary.md#term-planning) files, and the [validation](../../../../glossary.md#term-validation) commands above.

That pass must read the [architecture contract](../../../../glossary.md#architecture-contract), documentation rules, General, API/[Javadoc](../../../../glossary.md#term-javadoc),
Architecture, User guide, Planning, and Example profiles, final source/test changes, current
[backend](../../../../glossary.md#backend)-contract/config/planning contracts, and directly affected documents. It must independently
finalize the type, constructor, component, accessor, factory, and package Javadocs; current-versus-
planned wording; examples; links; terminology; and glossary impact. It must record reasoned
no-change conclusions for [Gradle](../../../../glossary.md#term-gradle) and dependencies, architecture rules and ADRs, architecture
tests, backend-[conformance](../../../../glossary.md#term-conformance) and integration tests, backend-contract/planning/[trace](../../../../glossary.md#trace) Java, concrete
backends, and every other [module](../../../../glossary.md#term-module). It reuses successful Java evidence unless executable behavior
changes.

## Dependencies

- Complete config task 0001 and its stable separation of optional hard requirements from later
  preference and evaluation.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Complete config task 0002 and the current `config.compile` package foundation.
- Complete planning task 0001 and its stable semantic operation-capability question/provider
  boundary.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [operation glossary](../../../../glossary.md#operation))</span>
- Complete [backend](../../../../glossary.md#backend)-contract tasks 0001–0004, especially stable [`DeviceClass`](../../../../glossary.md#device-class--deviceclass), supplied
  availability, and hard-requirement vocabulary.
- The existing architecture distinction between hard eligibility, backend-neutral scoring input,
  backend ownership, and backend prepare route selection.<span data-glossary-navigation="true"> ([backend ownership glossary](../../../../glossary.md#backend-ownership); [prepare glossary](../../../../glossary.md#prepare))</span>

## Follow-up tasks

- Config task 0004 retains immutable platform, backend, and tuning profile schemas as Draft. It
  must resolve identity, units, provenance, portability, versioning, and persistence before those
  calibrated inputs become Ready.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [backend glossary](../../../../glossary.md#backend); [schemas glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [persistence glossary](../../../../glossary.md#term-persistence); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Config task 0005 later composes current compile leaves and selected profile inputs; it remains
  Draft without a detailed specification.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- Planning task 0002 retains capability-matrix construction and hard-eligibility evaluation as
  Draft. Planning task 0003 retains candidate comparison and ownership scoring as Draft and
  depends on task 0002 plus stable scoring configuration and profiles.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [evaluation glossary](../../../../glossary.md#term-evaluation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Config tasks 0006–0008, planning tasks 0004–0006, and trace tasks 0003–0008 remain Draft without
  detailed specifications.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>
- After task 0003 completes, reassess the global frontier as a separate planning step. Do not
  assume planning scoring can advance while config task 0004 profile inputs remain undefined, and
  do not assume config task 0004 is automatically next if another prerequisite frontier is more
  coherent.

## Architecture impact

Expected impact: None.

The task realizes the existing config [ownership](../../../../glossary.md#term-ownership-meanings) of [immutable](../../../../glossary.md#term-immutable) scoring-policy input by reusing the
existing coarse [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) vocabulary and existing public [backend](../../../../glossary.md#backend)-contract [dependency](../../../../glossary.md#term-dependency). It
changes no ownership rule, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), hard-eligibility semantics, [planning](../../../../glossary.md#term-planning) behavior,
backend behavior, or architecture document rule. The authorized [partition](../../../../glossary.md#partition)-scoring edit updates
implementation status only. If implementation reveals a need for a scoring formula, candidate
[model](../../../../glossary.md#term-model), policy hierarchy, [module](../../../../glossary.md#term-module) dependency, or architecture change, stop before editing
architecture documentation.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md, the directly relevant documentation profiles,
docs/planning/roadmap.md, the config/planning/trace master plans, and
docs/planning/modules/config/tasks/0003-partition-scoring-configuration.md in full.

Implement task 0003 exactly inside its fourteen authorized paths. Add only the exact optional-
DeviceClass PartitionScoringConfig contract, focused automated tests, required Javadocs, current-
status documentation, and synchronized planning records. Do not add scoring evaluation, weights,
profiles, candidates, ownership, policy types, callbacks, services, dependencies, build changes,
or later task specifications. Stop and report any architecture or scope conflict instead of
inventing a broader contract.

After executable Java and the final config module validation, hand the actual diff and recorded
evidence to a separate clean-context documentation-focused agent or thread as required by this
task. Reuse successful Java evidence unless executable behavior changes. Do not run a root test.
Mark task 0003 Complete only after every criterion and the documentation pass succeed; then leave
the next global frontier for a separate reassessment.
```

## Local decisions

- Use one `Optional<DeviceClass>` record component because the existing CPU/accelerator
  classification is the only stable backend-neutral preference vocabulary and does not enumerate
  candidate identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [accelerator glossary](../../../../glossary.md#term-accelerator); [backend glossary](../../../../glossary.md#backend))</span>
- Keep the preference soft and downstream of hard eligibility. This preserves
  [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement) meaning even when the preferred class conflicts with the eligible set.
- Use `neutral()` and `preferring(DeviceClass)` as the only conveniences. The later aggregate
  chooses its default; this task does not silently make neutral, CPU, or accelerator the system
  default.
- Retain the exact optional and enum references to match current immutable config value
  conventions and avoid an unnecessary snapshot or normalization rule.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Do not add `PartitionScoringPolicy`. One optional typed preference already expresses the only
  concrete policy input; additional modes would require scoring formulas or consumers that are
  not stable.
- Do not expose weights for transfer, boundary, materialization, acceleration, or region size.
  Their units and combination depend on profile and planning work that remains Draft.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight); [materialization glossary](../../../../glossary.md#materialization); [planning glossary](../../../../glossary.md#term-planning))</span>
- Do not expose preferred backend IDs or an ordered backend list. That would mix user-provided
  identities with candidate enumeration/fallback semantics before candidate contracts exist.
- Keep the existing [Gradle](../../../../glossary.md#term-gradle) [API](../../../../glossary.md#term-application-programming-interface-api) edge unchanged because [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) is already part of the
  backend-contract dependency exposed publicly by config task 0001.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Known limitations

- The value cannot prefer one [`BackendId`](../../../../glossary.md#backend-identity--backendid), one [device](../../../../glossary.md#term-device), or an ordered list of candidates.
- It defines no preference magnitude, score, formula, tie rule, transfer estimate, boundary cost,
  small-region threshold, or deterministic owner order.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- It does not define how a later candidate represents devices or how planning combines a class
  preference with profile-derived estimates; those are planning/profile decisions.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- No current `CompileConfig`, [compiler](../../../../glossary.md#term-compiler), or planner consumes the value.
- Platform, backend, and tuning profiles remain undefined, so calibrated scoring cannot become
  current in this task.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- No [serialization](../../../../glossary.md#term-serialization) or external compatibility guarantee is established.

## [Validation](../../../../glossary.md#term-validation) evidence

- [Planning](../../../../glossary.md#term-planning) context `/root/plan_config_0003` read the architecture contract, focused architecture
  explanations, documentation/planning rules and selected General/Planning profiles, roadmap,
  config/planning/trace master plans, completed config tasks 0001–0002, completed planning task
  0001, current config/planning source/tests/build, and stable backend identity, classification,
  availability, hard-requirement, and capability-query/provider contracts before selecting the
  exact API.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [trace glossary](../../../../glossary.md#trace); [backend identity glossary](../../../../glossary.md#backend-identity--backendid); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Planning selected one optional coarse-class preference because it is type-safe, backend-neutral,
  and directly consumable by later candidate comparison without evaluating eligibility or
  committing to a scoring formula. No architecture ambiguity requires a contract change.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed for 223 Markdown files, 3,914 local links,
  235 local anchors, 2,808 fence markers, final newlines, and trailing whitespace.
- `git diff --check` passed with no output. The combined tracked/untracked scope audit returned
  exactly the five authorized planning paths, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git status --short` showed no path outside that
  set.
- Manual status and task-directory checks confirmed one global Ready task represented by the
  config master-plan row and this specification; config 0001–0002 and planning 0001 remain
  Complete; config 0004+, planning 0002+, and trace 0003+ remain Draft; and no detailed config
  0004, planning 0002, or trace 0003 specification exists.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Final planning-diff inspection confirmed balanced current-versus-planned wording, the exact
  proposed API and validation messages, all canonical task sections, unchanged architecture/API/
  guide/Java/test/Gradle files, and the required post-0003 frontier reassessment.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Implementation context `/root/implement_config_0003` added only the exact one-component public
  record, its package Javadoc, and one focused test. Its focused<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
  `PartitionScoringConfigTest` run passed five tests. After executable Java stabilized, its single
  final <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`./gradlew :modules:config:test` passed 17 tests across four XML suites
  (`BackendIntentTest` 5, `CompileModeTest` 2, `GraphOptimizationConfigTest` 5, and
  `PartitionScoringConfigTest` 5), with zero failures, errors, or skips.
- Documentation context `/root/implement_config_0003/doc_config_0003` independently read the
  architecture contract, current architecture index, documentation rules, General, API/Javadoc,
  Architecture, User guide, Planning, and Example profiles, planning guide and roadmap, the three
  affected master plans, this task, final source/test/package Javadoc, current config leaves,
  backend requirement/availability/classification contracts, planning capability contracts, and
  all directly affected documents. It finalized the type/package Javadocs and the six explanatory
  plus five planning/status paths without changing executable Java.<span data-glossary-navigation="true"> ([backend requirement glossary](../../../../glossary.md#backend-requirement--backendrequirement))</span>
- The documentation context reused the successful Java evidence because no executable source or
  test changed after that run; it did not rerun the config suite and did not run a repository-wide
  test. Repository-wide validation remains deferred to the config capability checkpoint or
  continuous integration because no dependency, build rule, architecture boundary, or
  cross-module executable behavior changed.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci); [module glossary](../../../../glossary.md#term-module))</span>
- `./gradlew :modules:config:javadoc` passed with `BUILD SUCCESSFUL`; generated
  `package-summary.html` showed all four current standalone config values and their planned
  boundaries, and generated `PartitionScoringConfig.html` showed the component, canonical
  constructor, both factories, explicit accessor, null failures/messages, reference retention,
  soft-after-eligibility meaning, and excluded lifecycle behavior.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed for 223 Markdown files, 3,916 local links,
  238 local anchors, 2,812 fence markers, final newlines, and trailing whitespace.
- `git diff --check` passed with no output. The combined tracked/untracked path audit returned
  exactly the fourteen authorized paths, and `git status --short` showed no other path.
- Manual checks confirmed unchanged config and repository Gradle dependencies, no detailed config
  0004, planning 0002, or trace 0003 task specification, balanced fences, final newlines, and
  synchronized status: config 0001–0003 and planning 0001 are Complete; config 0004+, planning
  0002+, and trace 0003+ are Draft; no global task is Ready.
- No architecture rule or [dependency](../../../../glossary.md#term-dependency) changed, so `ARCHITECTURE.md`, ADRs, and architecture tests
  correctly remain unchanged. No backend behavior or end-to-end execution changed, so backend-
  conformance and integration tests correctly remain unchanged. Backend-contract, planning, and
  trace Java; concrete backends; compiler, prepare, runtime, and engine; other modules; and shared
  Gradle configuration correctly remain unchanged because this task adds only declarative config
  input and documentation of current status.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine))</span>

## Implementation notes

- Added the exact `Optional<DeviceClass>` record with direct null validation, exact reference
  retention, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`neutral()`, `preferring(DeviceClass)`, and an explicitly declared accessor.
- Added five focused tests that lock the complete public record shape, failure messages, factory
  behavior, record value behavior, and absence of broader scoring or lifecycle surfaces.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Finalized package/API/architecture/user-guide/glossary documentation so the current preference
  input is distinct from planned eligibility evaluation, score calculation, profiles, ownership,
  routes, preparation, runtime, and execution.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [evaluation glossary](../../../../glossary.md#term-evaluation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Updated task, master-plan, and roadmap status without selecting or specifying a later frontier.

## Completion summary

- Completed changes: added the smallest immutable soft device-class preference config and its
  focused contract test; finalized Javadocs, explanatory documentation, glossary, and planning
  status.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [device glossary](../../../../glossary.md#term-device))</span>
- Files changed or created: exactly two production paths, one test path, six explanatory/status
  documentation paths, and five planning paths listed under Affected files.
- Tests and validation: reused the successful five-test focused result and final 17-test/four-suite
  config result; config Javadoc, generated-page inspection, repository Markdown, exact scope,
  dependency/status/later-spec checks, and whitespace validation passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Documentation-agent review: `/root/implement_config_0003/doc_config_0003` completed the required
  independent targeted pass without changing executable Java or repeating Java tests.
- Documentation impact: current [`PartitionScoringConfig`](../../../../glossary.md#partition-scoring-configuration--partitionscoringconfig) use and all planned consumer/evaluator
  boundaries are documented consistently.
- Javadoc review: type, record component, canonical constructor, factories, accessor, and package
  contracts are complete and rendered successfully.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Glossary impact: added the reusable partition-scoring-configuration term and synchronized
  backend intent, device class, and implementation-status distinctions.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [backend intent glossary](../../../../glossary.md#backend-intent--backendintent); [device class glossary](../../../../glossary.md#device-class--deviceclass))</span>
- Unresolved issues: None.
- Follow-up required: None. A separate [planning](../../../../glossary.md#term-planning) reassessment must select any next frontier.

Status: Complete
