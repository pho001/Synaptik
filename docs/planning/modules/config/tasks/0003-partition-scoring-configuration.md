# Task 0003: Partition Scoring Configuration

## Status

Complete

## Goal

Add the smallest immutable compile-configuration value that lets later planning observe an
optional soft preference between already eligible ownership candidates.

Mental model:

```text
hard requirement + capability + supplied availability
  -> later planning determines eligible candidates
  -> PartitionScoringConfig supplies an optional DeviceClass preference
  -> later planning compares candidates and selects backend ownership
```

The configuration records an input to later comparison. It does not determine eligibility,
enumerate candidates, calculate a score, or choose an owner, device, partition, route, or kernel.

## Scope

- Add public record `PartitionScoringConfig` in
  `io.github.pho001.synaptik.config.compile` with exactly one component:
  `Optional<DeviceClass> preferredDeviceClass`.
- Implement the canonical-constructor validation and retention sequence exactly:
  1. reject a null `preferredDeviceClass` optional with `NullPointerException` and exact message
     `preferredDeviceClass`; and
  2. retain the exact validated `Optional` reference without copying, normalization, candidate
     lookup, or evaluation.
- An empty optional means only that no explicit coarse device-class preference is supplied. It
  does not mean equal candidate scores, CPU default, accelerator default, automatic discovery,
  fallback, successful ownership selection, or absence of other later planning inputs.
- A present optional requests a soft preference for the exact retained `DeviceClass` value after
  hard eligibility has been established. It must not make another candidate ineligible, weaken a
  hard `BackendRequirement`, or guarantee that a candidate of the preferred class is selected.
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
  backend-specific field.
- Keep calibrated platform, backend, and tuning measurements out of this record. Config task 0004
  owns those immutable profile schemas after their identity, units, provenance, and versioning
  decisions are ready.
- Update compile-package Javadoc to describe the four current standalone configuration values and
  distinguish hard eligibility from soft ranking input and later profile data.
- Add one focused test covering exact record/API shape, constructor validation and message,
  direct reference retention, both factories, value behavior, and absence of eligibility,
  scoring, candidate, service, backend-route, profile, and lifecycle surfaces.
- Finalize the new Javadoc and the focused current-status/API/user-guide/glossary/planning text in
  the same overall change through the required separate clean-context documentation pass.

## Out of scope

- changing `BackendIntent`, `BackendRequirement`, `BackendAvailabilitySnapshot`, `BackendId`,
  `BackendDeviceId`, `DeviceClass`, or their semantics, tests, Javadocs, build, or dependencies
- evaluating capability, supplied availability, a hard requirement, registration, or
  preparability; constructing a capability matrix; filtering candidates; or defining no-match
  failure
- weakening a hard requirement because its eligible result conflicts with the preferred device
  class
- enumerating candidate backends or devices, accepting a preferred `BackendId` list, defining
  fallback order, or selecting a device
- score calculation, score result types, score comparison, tie breaking, deterministic candidate
  order, owner selection, node or segment selection, partitioning, transfer planning,
  materialization planning, or logical memory planning
- numeric transfer, boundary, materialization, accelerator, small-region, or other weights,
  bonuses, penalties, thresholds, formulas, normalization, units, or calibration values
- `PartitionScoringPolicy`, policy implementations, callbacks, strategies, registries, plugins,
  maps, string keys, generic weights, builders, loaders, serialization, or compatibility schemas
- `PlatformProfile`, `BackendProfile`, `TuningProfile`, benchmark evidence, platform fingerprint,
  persistence, measurement ingestion, tuning generation, or live platform discovery
- changing `OperationCapabilityQuery`, `BackendCapabilityProvider`, or adding a planning consumer,
  provider implementation, candidate model, ownership result, compiler orchestration, or trace
  schema
- `CompileConfig`, aggregate defaults, compiler interpretation, graph behavior, prepare, run,
  publication, runtime, engine, backend lowering, route or kernel selection, executable state,
  storage, physical memory, or execution
- dependencies, Gradle, Java version, architecture contract, architecture rules, ADRs,
  architecture tests, backend-conformance tests, integration tests, another module's executable
  behavior, or detailed later task specifications
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Architecture style](../../../../developer-guide/documentation/architecture-style.md)
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Config master plan](../master-plan.md)
- [Planning master plan](../../planning/master-plan.md)
- [Trace master plan](../../trace/master-plan.md)
- [Config task 0001](0001-backend-intent-foundation.md)
- [Config task 0002](0002-compile-modes-and-graph-optimization-configuration.md)
- [Planning task 0001](../../planning/tasks/0001-operation-capability-query-foundation.md)
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Compile workflow guide](../../../../user-guide/compiling-graphs.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/config` owns immutable declarative scoring inputs. `modules/planning` owns capability
  and hard-eligibility evaluation, candidate construction, comparison, score calculation, and
  ownership decisions.
- Hard eligibility and soft ranking remain separate. `BackendIntent` may remove candidates through
  a hard requirement; `PartitionScoringConfig` must never remove a candidate or weaken that
  requirement.
- `DeviceClass` is the existing backend-neutral coarse CPU/accelerator classification. Enum order
  has no preference meaning; only a present explicit configuration value supplies a class
  preference.
- Config may retain the classification value because its existing public backend-contract
  dependency already exposes declarative backend vocabulary. No dependency or visibility change
  is required.
- Planning scoring uses compile-time facts only and selects backend ownership, not a device,
  executable implementation, route, or kernel.
- Immutable calibrated profile data remains a distinct config concern because its identity,
  measurement units, provenance, portability, and versioning are not properties of a preference.
- Config must not retain a capability provider, live service, candidate object, planning callback,
  runtime or prepare state, concrete backend class, executable, kernel class, physical resource,
  or mutable measurement source.
- Stop if implementation needs another production type, a planning consumer, a dependency or
  build edit, an exact scoring formula, a calibrated value, candidate semantics, ownership
  behavior, or an architecture decision.

## Package impact

Existing package extended:

- `io.github.pho001.synaptik.config.compile` — immutable public graph-compilation configuration
  leaves; task 0003 adds only a soft coarse device-class preference for later ownership scoring

Type added:

- `io.github.pho001.synaptik.config.compile.PartitionScoringConfig` — configuration owns the
  optional declarative preference while planning later owns every evaluation and decision

Existing `BackendIntent`, `CompileMode`, and `GraphOptimizationConfig` remain unchanged. Package
Javadoc is updated to describe the fourth current standalone value. No `PartitionScoringPolicy`,
subpackage, helper, builder, service, registry, candidate, score, profile, planning, runtime, or
backend-specific type is added.

## Affected files

The intended overall implementation scope is exactly fourteen paths.

Production — exactly two paths:

- add
  `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/PartitionScoringConfig.java`
- Javadoc-only update to
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

Planning — exactly five paths:

- add and finalize this task
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/modules/planning/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: all current backend-contract source/tests/Javadocs/build,
`BackendIntent`, `CompileMode`, `GraphOptimizationConfig`, their tests and generated Javadocs,
config build, existing config architecture dependency test, planning capability source/tests/
Javadocs/build, trace contracts, compiler/runtime/prepare/engine placeholders, concrete backends,
`AGENTS.md`, `ARCHITECTURE.md`, focused architecture documents beyond the authorized status
sentence, ADRs, root build/settings, other module builds, backend-conformance tests, integration
tests, and all later task rows.

## Maximum scope

Exactly the fourteen paths listed above. Stop if implementation requires another production
type, test, document, dependency, package, module API, architecture change, scoring consumer,
profile schema, candidate or ownership behavior, or detailed follow-up specification. Do not use
a later task to hide an incomplete acceptance criterion.

## Required Javadoc contracts

- The type Javadoc must define the value as an immutable soft ranking input applied only after
  hard eligibility, distinguish absence from a default/fallback/equal-score promise, state that a
  present preference does not make other candidates ineligible or guarantee selection, and list
  the evaluation, profile, ownership, route, prepare, runtime, and execution boundaries.
- The record-component and canonical-constructor Javadocs must state that the optional is
  non-null, the exact optional reference is retained, and the exact enum reference is retained
  when present.
- Constructor Javadoc must document the exact `preferredDeviceClass` null failure and message and
  state that no snapshot, normalization, lookup, matching, or evaluation occurs.
- `neutral()` Javadoc must state that it returns a new non-null value with no explicit device-class
  preference and does not select an aggregate default, owner, fallback, or successful result.
- `preferring(DeviceClass)` Javadoc must document the non-null input, exact reference retention,
  new non-null result, soft-only meaning, and exact `deviceClass` null failure and message.
- `preferredDeviceClass()` Javadoc must document the exact non-null retained optional reference
  and exact contained enum reference when present.
- Package Javadoc must identify all four current standalone compile values and keep current config
  construction separate from planned `CompileConfig`, profiles, planning evaluation, compiler
  orchestration, preparation, runtime, engine, and backend implementation.

## Acceptance criteria

- `PartitionScoringConfig` is a public final record with exactly one
  `Optional<DeviceClass> preferredDeviceClass` component, one public canonical constructor, one
  explicitly documented accessor, exactly `neutral()` and `preferring(DeviceClass)` as added
  public factories, ordinary record object methods, and no other project field, method,
  constructor, interface, nested type, builder, or factory.
- Direct construction rejects a null optional with exact message `preferredDeviceClass`, accepts
  empty and present optionals, and returns the exact supplied optional and contained enum
  references.
- Constructor validation occurs before retention, and valid construction performs no collection
  snapshot, normalization, lookup, matching, score calculation, or other evaluation.
- `neutral()` produces fresh equal values with empty optionals and promises only absence of an
  explicit device-class preference.
- `preferring(...)` rejects null with exact message `deviceClass` and produces fresh equal values
  containing the exact supplied enum reference.
- A present preference remains soft: it neither filters an eligible candidate nor weakens a hard
  requirement, and it guarantees no selected backend, device, score, route, or execution result.
- No candidate enumeration, backend-ID preference list, fallback order, tie rule, numeric factor,
  profile/calibration value, score result, policy type, callback, map, string key, service,
  registry, planner, lifecycle, route, kernel, or executable surface is added.
- Existing `BackendIntent`, `CompileMode`, `GraphOptimizationConfig`, backend-contract contracts,
  and planning capability contracts retain their exact source/API/behavior/dependency boundaries.
- Config build and all module dependencies remain unchanged; the existing config-to-backend-
  contract API edge is sufficient and no architecture test changes.
- Focused automated tests lock the exact generic record shape, validation/message/reference rules,
  factory behavior, record value behavior, and forbidden added surfaces. No recurring API-shape
  invariant is left to manual reflection or `javap` checks.
- Package Javadoc, partition-scoring implementation status, Public API, Compile API,
  backend-selection guide, compile workflow guide, and glossary distinguish the current soft
  input from planned profiles, eligibility/scoring evaluation, ownership, compiler consumption,
  preparation, runtime, and execution.
- Config master plan marks only task 0003 Ready before implementation. Config tasks 0001–0002 and
  planning task 0001 remain Complete; config 0004+, planning 0002+, and trace 0003+ remain Draft
  without detailed specifications.
- A separate clean-context documentation-focused pass finalizes the new type and package Javadocs
  plus the eleven explanatory/planning paths in the same overall change, reviews glossary impact,
  and reuses successful Java evidence unless executable Java changes.
- Exactly one final config module suite, config Javadoc, repository Markdown, generated-page
  inspection, exact fourteen-path scope, dependency no-change, status, later-spec absence, final
  newlines, trailing whitespace, and `git diff --check` validation pass.
- No repository-wide test is run merely by habit: this task changes one module's declarative API
  without changing a dependency, build rule, architecture boundary, or cross-module executable
  behavior. The config capability checkpoint or continuous integration owns that tier.

## Tests / validation

Focused development commands are allowed while Java changes. After executable Java stabilizes,
run exactly one final config module suite:

```bash
./gradlew :modules:config:test
```

Record suite and test counts from XML reports. Then hand the actual diff and exact Java evidence
to the separate documentation-focused context described below. That context reuses the successful
Java evidence and does not rerun the suite unless it changes executable Java behavior or records a
concrete reason. After final Javadoc and Markdown edits, it runs:

```bash
./gradlew :modules:config:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

Inspect generated `package-summary.html` and `PartitionScoringConfig.html`. Confirm exactly the
fourteen authorized paths; unchanged Gradle dependencies; task 0003 Complete only after all
implementation criteria pass; no global Ready task after completion until a separate frontier
reassessment; no detailed config 0004, planning 0002, or trace 0003 specification; balanced
fences; final newlines; and no trailing whitespace.

Do not run repository-wide tests in this task. Defer them to the config capability checkpoint or
continuous integration because no dependency, shared build rule, architecture boundary, or
cross-module executable behavior changes.

## Documentation handoff

After the implementation context records the final config test result, hand this task, the actual
diff, and exact test evidence to a separate clean-context documentation-focused agent or thread.
The handoff must identify `PartitionScoringConfig` as the only new API, its soft-after-eligibility
semantics, the exact fourteen-path limit, the unchanged dependencies and architecture rules, the
expected six explanatory documents and five planning files, and the validation commands above.

That pass must read the architecture contract, documentation rules, General, API/Javadoc,
Architecture, User guide, Planning, and Example profiles, final source/test changes, current
backend-contract/config/planning contracts, and directly affected documents. It must independently
finalize the type, constructor, component, accessor, factory, and package Javadocs; current-versus-
planned wording; examples; links; terminology; and glossary impact. It must record reasoned
no-change conclusions for Gradle and dependencies, architecture rules and ADRs, architecture
tests, backend-conformance and integration tests, backend-contract/planning/trace Java, concrete
backends, and every other module. It reuses successful Java evidence unless executable behavior
changes.

## Dependencies

- Complete config task 0001 and its stable separation of optional hard requirements from later
  preference and evaluation.
- Complete config task 0002 and the current `config.compile` package foundation.
- Complete planning task 0001 and its stable semantic operation-capability question/provider
  boundary.
- Complete backend-contract tasks 0001–0004, especially stable `DeviceClass`, supplied
  availability, and hard-requirement vocabulary.
- The existing architecture distinction between hard eligibility, backend-neutral scoring input,
  backend ownership, and backend prepare route selection.

## Follow-up tasks

- Config task 0004 retains immutable platform, backend, and tuning profile schemas as Draft. It
  must resolve identity, units, provenance, portability, versioning, and persistence before those
  calibrated inputs become Ready.
- Config task 0005 later composes current compile leaves and selected profile inputs; it remains
  Draft without a detailed specification.
- Planning task 0002 retains capability-matrix construction and hard-eligibility evaluation as
  Draft. Planning task 0003 retains candidate comparison and ownership scoring as Draft and
  depends on task 0002 plus stable scoring configuration and profiles.
- Config tasks 0006–0008, planning tasks 0004–0006, and trace tasks 0003–0008 remain Draft without
  detailed specifications.
- After task 0003 completes, reassess the global frontier as a separate planning step. Do not
  assume planning scoring can advance while config task 0004 profile inputs remain undefined, and
  do not assume config task 0004 is automatically next if another prerequisite frontier is more
  coherent.

## Architecture impact

Expected impact: None.

The task realizes the existing config ownership of immutable scoring-policy input by reusing the
existing coarse `DeviceClass` vocabulary and existing public backend-contract dependency. It
changes no ownership rule, dependency direction, hard-eligibility semantics, planning behavior,
backend behavior, or architecture document rule. The authorized partition-scoring edit updates
implementation status only. If implementation reveals a need for a scoring formula, candidate
model, policy hierarchy, module dependency, or architecture change, stop before editing
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
  candidate identities.
- Keep the preference soft and downstream of hard eligibility. This preserves
  `BackendRequirement` meaning even when the preferred class conflicts with the eligible set.
- Use `neutral()` and `preferring(DeviceClass)` as the only conveniences. The later aggregate
  chooses its default; this task does not silently make neutral, CPU, or accelerator the system
  default.
- Retain the exact optional and enum references to match current immutable config value
  conventions and avoid an unnecessary snapshot or normalization rule.
- Do not add `PartitionScoringPolicy`. One optional typed preference already expresses the only
  concrete policy input; additional modes would require scoring formulas or consumers that are
  not stable.
- Do not expose weights for transfer, boundary, materialization, acceleration, or region size.
  Their units and combination depend on profile and planning work that remains Draft.
- Do not expose preferred backend IDs or an ordered backend list. That would mix user-provided
  identities with candidate enumeration/fallback semantics before candidate contracts exist.
- Keep the existing Gradle API edge unchanged because `DeviceClass` is already part of the
  backend-contract dependency exposed publicly by config task 0001.

## Known limitations

- The value cannot prefer one `BackendId`, one device, or an ordered list of candidates.
- It defines no preference magnitude, score, formula, tie rule, transfer estimate, boundary cost,
  small-region threshold, or deterministic owner order.
- It does not define how a later candidate represents devices or how planning combines a class
  preference with profile-derived estimates; those are planning/profile decisions.
- No current `CompileConfig`, compiler, or planner consumes the value.
- Platform, backend, and tuning profiles remain undefined, so calibrated scoring cannot become
  current in this task.
- No serialization or external compatibility guarantee is established.

## Validation evidence

- Planning context `/root/plan_config_0003` read the architecture contract, focused architecture
  explanations, documentation/planning rules and selected General/Planning profiles, roadmap,
  config/planning/trace master plans, completed config tasks 0001–0002, completed planning task
  0001, current config/planning source/tests/build, and stable backend identity, classification,
  availability, hard-requirement, and capability-query/provider contracts before selecting the
  exact API.
- Planning selected one optional coarse-class preference because it is type-safe, backend-neutral,
  and directly consumable by later candidate comparison without evaluating eligibility or
  committing to a scoring formula. No architecture ambiguity requires a contract change.
- `python3 /tmp/validate_synaptik_markdown.py` passed for 223 Markdown files, 3,914 local links,
  235 local anchors, 2,808 fence markers, final newlines, and trailing whitespace.
- `git diff --check` passed with no output. The combined tracked/untracked scope audit returned
  exactly the five authorized planning paths, and `git status --short` showed no path outside that
  set.
- Manual status and task-directory checks confirmed one global Ready task represented by the
  config master-plan row and this specification; config 0001–0002 and planning 0001 remain
  Complete; config 0004+, planning 0002+, and trace 0003+ remain Draft; and no detailed config
  0004, planning 0002, or trace 0003 specification exists.
- Final planning-diff inspection confirmed balanced current-versus-planned wording, the exact
  proposed API and validation messages, all canonical task sections, unchanged architecture/API/
  guide/Java/test/Gradle files, and the required post-0003 frontier reassessment.
- Implementation context `/root/implement_config_0003` added only the exact one-component public
  record, its package Javadoc, and one focused test. Its focused
  `PartitionScoringConfigTest` run passed five tests. After executable Java stabilized, its single
  final `./gradlew :modules:config:test` passed 17 tests across four XML suites
  (`BackendIntentTest` 5, `CompileModeTest` 2, `GraphOptimizationConfigTest` 5, and
  `PartitionScoringConfigTest` 5), with zero failures, errors, or skips.
- Documentation context `/root/implement_config_0003/doc_config_0003` independently read the
  architecture contract, current architecture index, documentation rules, General, API/Javadoc,
  Architecture, User guide, Planning, and Example profiles, planning guide and roadmap, the three
  affected master plans, this task, final source/test/package Javadoc, current config leaves,
  backend requirement/availability/classification contracts, planning capability contracts, and
  all directly affected documents. It finalized the type/package Javadocs and the six explanatory
  plus five planning/status paths without changing executable Java.
- The documentation context reused the successful Java evidence because no executable source or
  test changed after that run; it did not rerun the config suite and did not run a repository-wide
  test. Repository-wide validation remains deferred to the config capability checkpoint or
  continuous integration because no dependency, build rule, architecture boundary, or
  cross-module executable behavior changed.
- `./gradlew :modules:config:javadoc` passed with `BUILD SUCCESSFUL`; generated
  `package-summary.html` showed all four current standalone config values and their planned
  boundaries, and generated `PartitionScoringConfig.html` showed the component, canonical
  constructor, both factories, explicit accessor, null failures/messages, reference retention,
  soft-after-eligibility meaning, and excluded lifecycle behavior.
- `python3 /tmp/validate_synaptik_markdown.py` passed for 223 Markdown files, 3,916 local links,
  238 local anchors, 2,812 fence markers, final newlines, and trailing whitespace.
- `git diff --check` passed with no output. The combined tracked/untracked path audit returned
  exactly the fourteen authorized paths, and `git status --short` showed no other path.
- Manual checks confirmed unchanged config and repository Gradle dependencies, no detailed config
  0004, planning 0002, or trace 0003 task specification, balanced fences, final newlines, and
  synchronized status: config 0001–0003 and planning 0001 are Complete; config 0004+, planning
  0002+, and trace 0003+ are Draft; no global task is Ready.
- No architecture rule or dependency changed, so `ARCHITECTURE.md`, ADRs, and architecture tests
  correctly remain unchanged. No backend behavior or end-to-end execution changed, so backend-
  conformance and integration tests correctly remain unchanged. Backend-contract, planning, and
  trace Java; concrete backends; compiler, prepare, runtime, and engine; other modules; and shared
  Gradle configuration correctly remain unchanged because this task adds only declarative config
  input and documentation of current status.

## Implementation notes

- Added the exact `Optional<DeviceClass>` record with direct null validation, exact reference
  retention, `neutral()`, `preferring(DeviceClass)`, and an explicitly declared accessor.
- Added five focused tests that lock the complete public record shape, failure messages, factory
  behavior, record value behavior, and absence of broader scoring or lifecycle surfaces.
- Finalized package/API/architecture/user-guide/glossary documentation so the current preference
  input is distinct from planned eligibility evaluation, score calculation, profiles, ownership,
  routes, preparation, runtime, and execution.
- Updated task, master-plan, and roadmap status without selecting or specifying a later frontier.

## Completion summary

- Completed changes: added the smallest immutable soft device-class preference config and its
  focused contract test; finalized Javadocs, explanatory documentation, glossary, and planning
  status.
- Files changed or created: exactly two production paths, one test path, six explanatory/status
  documentation paths, and five planning paths listed under Affected files.
- Tests and validation: reused the successful five-test focused result and final 17-test/four-suite
  config result; config Javadoc, generated-page inspection, repository Markdown, exact scope,
  dependency/status/later-spec checks, and whitespace validation passed.
- Documentation-agent review: `/root/implement_config_0003/doc_config_0003` completed the required
  independent targeted pass without changing executable Java or repeating Java tests.
- Documentation impact: current `PartitionScoringConfig` use and all planned consumer/evaluator
  boundaries are documented consistently.
- Javadoc review: type, record component, canonical constructor, factories, accessor, and package
  contracts are complete and rendered successfully.
- Glossary impact: added the reusable partition-scoring-configuration term and synchronized
  backend intent, device class, and implementation-status distinctions.
- Unresolved issues: None.
- Follow-up required: None. A separate planning reassessment must select any next frontier.

Status: Complete
