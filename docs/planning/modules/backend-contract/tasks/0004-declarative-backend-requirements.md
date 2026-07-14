# Task 0004: Declarative Backend Requirements

## Status

Complete

## Goal

Complete the minimal backend-contract milestone with a closed, type-safe family of hard
eligibility requirements targeting an exact backend identity, an exact backend-scoped device
identity, or a coarse device class.

Mental model:

```text
BackendRequirement
  ├── BackendIdRequirement(BackendId("cuda"))
  ├── BackendDeviceIdRequirement(BackendDeviceId("cuda", "0"))
  └── DeviceClassRequirement(DeviceClass.ACCELERATOR)
```

The family records requested eligibility only. It neither evaluates a snapshot nor expresses
preference, scoring, fallback, discovery, capability support, ownership selection, or execution.

## Scope

- Add public sealed interface `BackendRequirement` in
  `io.github.pho001.synaptik.backend.contract`.
  - It permits exactly `BackendIdRequirement`, `BackendDeviceIdRequirement`, and
    `DeviceClassRequirement`.
  - It declares no field, method, nested type, constant, factory, default behavior, or metadata.
  - Its only role is nominal exhaustiveness for later config and planning consumers.
- Add public record `BackendIdRequirement` implementing `BackendRequirement` with exactly one
  `BackendId backendId` component.
  - It expresses a hard request that later eligible ownership use that exact backend identity.
  - Reject null with `NullPointerException` and exact message `backendId`.
  - Retain and return the exact supplied reference through an explicitly documented accessor.
- Add public record `BackendDeviceIdRequirement` implementing `BackendRequirement` with exactly
  one `BackendDeviceId deviceId` component.
  - It expresses a hard request for that exact backend-scoped device identity and thereby implies
    its owning backend.
  - Reject null with `NullPointerException` and exact message `deviceId`.
  - Retain and return the exact supplied reference through an explicitly documented accessor.
- Add public record `DeviceClassRequirement` implementing `BackendRequirement` with exactly one
  `DeviceClass deviceClass` component.
  - It expresses a hard request for any later eligible device of that coarse class.
  - Reject null with `NullPointerException` and exact message `deviceClass`.
  - Retain and return the exact supplied enum reference through an explicitly documented accessor.
- Each record must add no other component, field, constructor, public method, nested type, or
  implemented interface beyond `BackendRequirement`, and must preserve ordinary record equality,
  hashing, and diagnostic `toString()` behavior.
- Add no `ANY`, `AUTO`, `NONE`, `DEFAULT`, `PREFER`, `AVOID`, `REQUIRE`, `CPU`, `GPU`, or another
  requirement variant. Absence of a hard requirement belongs to the later owning config field,
  not to a sentinel `BackendRequirement` instance.
- Add no requirement combination, list, conjunction, disjunction, negation, predicate,
  `matches`, `test`, `satisfiedBy`, evaluation result, explanation, priority, or score.
- Define the family as a hard eligibility vocabulary. Later config determines where an optional
  requirement is supplied; later planning combines it with availability and capabilities. If no
  eligible target remains, the later owning layer fails rather than silently weakening the
  requirement.
- Update package Javadoc to include the current requirement family while preserving identity,
  classification, supplied-snapshot, and no-live-service boundaries.
- Add one focused test class covering exact sealed hierarchy/API shape, record components,
  validation/messages, exact reference retention, ordinary value behavior, exhaustive distinct
  variants, and absence of factories, sentinels, evaluators, extra interfaces, and nested types.
- Finalize Javadocs and update Public API status, capability-provider guide, backend-selection
  guide, glossary, backend-contract master plan, trace interleave status, and roadmap in the same
  overall change.
- After the implementation and documentation passes, run the backend-contract capability
  checkpoint once with the full repository test suite.

## Out of scope

- preference, avoidance, fallback, priority, score, bonus, penalty, mode, policy, profile, or
  automatic-selection semantics
- requirement composition, collections, boolean algebra, predicates, visitors, matching methods,
  evaluation results, rejection reasons, diagnostics, or validation against a snapshot
- operation support, operation kind, data type, shape, layout, precision, memory, capacity,
  topology, property, vendor, route, kernel, executable, or performance requirements
- creating or modifying config `BackendIntent`, compile/prepare configs, platform/backend/tuning
  profiles, capability-provider, capability matrix, planning candidates, scoring, partitions, or
  ownership decisions
- changing `BackendId`, `BackendDeviceId`, `DeviceClass`, `BackendAvailabilitySnapshot`, or their
  tests and behavior
- availability discovery, refresh, liveness, health, registration, service locator,
  `ServiceLoader`, classpath scanning, concrete backend lookup, or engine composition
- preparation, lowering, kernel selection, storage, transfer, runtime residency, publication, or
  execution
- trace IDs, payloads, attributes, translation, emission, or changes to trace Java/tests
- serialization, external schema, persistence, networking, dependencies, Gradle, Java version,
  architecture contract, ADR, architecture tests, another module, concrete backend, backend
  conformance, or integration implementation changes
- a detailed config, planning, trace, or later backend-contract task specification
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [No runtime service locator ADR](../../../../design/decisions/0006-no-runtime-service-locator.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Backend guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Backend-contract master plan](../master-plan.md)
- [Trace master plan](../../trace/master-plan.md)
- [Task 0001](0001-backend-and-device-identifiers.md)
- [Task 0002](0002-device-classification.md)
- [Task 0003](0003-backend-availability-snapshot.md)
- [Public API status](../../../../api/public-api.md)
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/backend-contract` owns minimal declarative backend requirements and identity vocabulary.
- Requirements contain immutable requested facts only. Config owns user-facing intent and planning
  owns eligibility enforcement and ownership selection.
- Requirements must not retain a concrete backend, provider, service, executable, device handle,
  runtime state, or capability implementation.
- Compile-time plans retain selected `BackendId` values, not requirement objects as live services
  or concrete backend instances.
- `modules/backend-contract` remains JDK-only and independent of model, trace, config, planning,
  compiler, runtime, prepare, engine, and concrete backend modules.
- Stop if implementation needs a fourth variant, combination/evaluation API, consumer-layer type,
  another module, dependency edit, or architecture decision.

## Package impact

Existing package retained:

- `io.github.pho001.synaptik.backend.contract` — minimal backend-neutral identities,
  classification, supplied availability, and declarative requirement vocabulary

Types added:

- `io.github.pho001.synaptik.backend.contract.BackendRequirement` — sealed method-free hard
  eligibility marker
- `io.github.pho001.synaptik.backend.contract.BackendIdRequirement` — exact backend identity target
- `io.github.pho001.synaptik.backend.contract.BackendDeviceIdRequirement` — exact backend-scoped
  device identity target
- `io.github.pho001.synaptik.backend.contract.DeviceClassRequirement` — coarse device-class target

Existing package Javadoc is updated to explain the complete current vocabulary. Completed
identity, classification, and snapshot declarations and behavior remain unchanged. The focused
test mirrors the production package. No subpackage or helper type is added.

## Affected files

Production — exactly five paths:

- add `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/BackendRequirement.java`
- add `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/BackendIdRequirement.java`
- add `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/BackendDeviceIdRequirement.java`
- add `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/DeviceClassRequirement.java`
- Javadoc-only update to
  `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/package-info.java`

Tests — exactly one path:

- add `modules/backend-contract/src/test/java/io/github/pho001/synaptik/backend/contract/BackendRequirementTest.java`

Documentation and planning — exactly eight paths:

- `docs/api/public-api.md`
- `docs/backend-guide/capability-provider.md`
- `docs/user-guide/backend-selection.md`
- `docs/glossary.md`
- add and finalize this task
- `docs/planning/modules/backend-contract/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: completed task-0001–0003 source/tests/Javadocs other than package
documentation, config/planning placeholders and master plans, `AGENTS.md`, `ARCHITECTURE.md`,
focused architecture/ADR documents and architecture tests, Gradle, other modules, concrete
backends, backend conformance, integration tests, and legacy evidence.

## Maximum scope

At most the exact fourteen paths above. Stop if implementation requires another production type,
test, document, package, dependency, module, architecture change, build edit, consumer
implementation, or detailed follow-up specification. The user's standing instruction permits a
necessary Javadoc-only path-count increase, but any such expansion must be recorded and must not
change a completed declaration or behavior.

## Acceptance criteria

- `BackendRequirement` is a public sealed interface permitting exactly the three specified public
  record variants and declares no field, method, nested type, constant, factory, or behavior.
- `BackendIdRequirement`, `BackendDeviceIdRequirement`, and `DeviceClassRequirement` each contain
  exactly their one specified component, implement only `BackendRequirement`, validate null with
  the exact component-name message, retain the exact reference, expose an explicitly documented
  accessor, and preserve ordinary record value semantics.
- No sentinel, preference, mode, combination, evaluation, capability, scoring, fallback,
  registration, discovery, service, execution, or serialization API is added.
- The family represents hard eligibility without evaluating it. Later consumer documentation
  clearly owns optionality, matching against availability/capabilities, and no-match failure.
- Existing `BackendId`, `BackendDeviceId`, `DeviceClass`, and `BackendAvailabilitySnapshot`
  declarations, behavior, and focused tests remain unchanged.
- Production imports are limited to same-package types; the preferred implementation needs no
  import declaration.
- Focused tests fail on permits/component/API drift and cover all validation, identity, equality,
  hierarchy, and forbidden-surface contracts.
- Package Javadoc, Public API status, capability-provider guide, backend-selection guide, and
  glossary mark only the immutable requirement vocabulary current. Config, capability providers,
  planning interpretation, prepare, runtime, engine, and concrete backends remain planned.
- Backend-contract master plan marks tasks 0001–0004 and its selected milestone Complete after
  every validation passes. Trace master plan and roadmap record the completed producer foundation
  and leave trace 0003+ Draft pending broader producer contracts.
- A separate clean-context documentation-focused pass finalizes all new/package Javadocs and eight
  documentation/planning paths after Java tests pass.
- Exactly one final backend-contract module test run occurs after executable Java stabilizes. The
  single repository-wide capability checkpoint occurs only after implementation and documentation
  are stable.
- Exact fourteen-path scope, Markdown, final newlines, trailing whitespace, `git diff --check`, and
  the repository checkpoint pass; no completed contract behavior, architecture, dependency,
  Gradle, trace Java/test, config/planning implementation, concrete backend, conformance, or
  integration path changes.

## Tests / validation

Run focused tests while developing. After executable Java stabilizes, run exactly one final
backend-contract module suite:

```bash
./gradlew :modules:backend-contract:test
```

Record test and suite counts from XML reports. Then hand the actual diff and exact Java evidence
to a separate clean-context documentation-focused agent in the same overall change. That pass
independently inspects final source/tests and completed identity/classification/snapshot contracts,
applies General style with API/Javadoc, Backend guide, User guide, Planning, and Example profiles
where relevant, finalizes all Javadocs and eight documentation/planning paths, records reasoned
no-change conclusions, and runs:

```bash
./gradlew :modules:backend-contract:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun. Inspect generated package, interface, and three record pages;
rely on focused automated sealed-hierarchy/API-shape checks; confirm exact task scope; and confirm
backend-contract 0001–0004 Complete/no Ready row/no detailed follow-up task.

After both passes and final documentation stabilization, run the capability checkpoint exactly
once:

```bash
./gradlew test
```

Record repository test/task results. Do not repeat the checkpoint in the documentation context.

## Dependencies

- Completed backend-contract task 0001 identities.
- Completed backend-contract task 0002 device classification.
- Completed backend-contract task 0003 availability snapshot.
- The selected model milestone and trace tasks 0001–0002 are Complete as roadmap sequencing
  context only; this task adds no model or trace dependency.

## Follow-up tasks

- After the capability checkpoint, reassess the ordered frontier. No detailed config, planning,
  trace, or additional backend-contract task is created by this task.
- Config is the default next project-area candidate because backend identity, class, availability,
  and requirement vocabulary will be stable. Trace remains interleaved until additional producer
  contracts make concrete payload schemas timely.

## Architecture impact

Expected impact: None.

The task implements the existing `BackendRequirement` ownership rule and completes the selected
backend-contract vocabulary. It changes no module direction, discovery/registration mechanism,
lifecycle decision, planning algorithm, or backend behavior. Documentation changes
implementation-status wording only.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, focused module/dependency/partition/prepare-backend/no-service-
locator architecture documents, documentation/planning rules and profiles, roadmap,
backend-contract and trace master plans, completed backend-contract tasks 0001–0003, task 0004,
current backend-contract source/tests/Javadocs, Public API, capability-provider and
backend-selection guides, glossary, relevant config/planning master plans, and Java 26 root
configuration.

Implement docs/planning/modules/backend-contract/tasks/0004-declarative-backend-requirements.md
exactly inside its fourteen authorized paths. Add only the sealed method-free BackendRequirement
family with exact BackendIdRequirement, BackendDeviceIdRequirement, and DeviceClassRequirement
records, one focused test, package-Javadoc update, and permitted documentation/status changes.
Preserve exact validation/messages/reference/value semantics and hard-eligibility/no-evaluation
boundaries. Add no sentinel, preference, combination, matcher, provider, consumer implementation,
dependency, or later task. Stop on architecture, requirement semantics, dependency, package,
affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final :modules:backend-contract:test after
executable Java stabilizes. Then hand the actual diff and Java evidence to a separate clean-context
documentation agent in the same overall change. That pass finalizes Javadocs, Public API, backend
and user guides, glossary, task/master/trace/roadmap status, runs backend-contract Javadoc and
documentation/scope checks, and reuses successful Java evidence unless executable behavior
changes. After both passes, run exactly one final root test capability checkpoint. Mark 0004 and
the backend-contract milestone Complete only after all criteria pass. Leave every later frontier
Draft without a detailed specification.
```

## Local decisions

- Use a sealed interface plus three one-component public records so invalid target combinations
  are unrepresentable and later consumers can exhaustively pattern-match without a registry or
  discriminator string.
- Name variants after their exact target types. This keeps identity, device, and class constraints
  explicit and avoids ambiguous generic `Target` or nullable/optional multi-field records.
- Add no absence sentinel. Later config can use `Optional<BackendRequirement>` or another explicit
  owning contract; absence is not a kind of hard requirement.
- Add no preference or avoidance mode. Those are user intent and scoring policy, not eligibility
  target identity.
- Add no matcher. Requirement evaluation needs availability, capabilities, config semantics, and
  failure ownership that belong to later consumer tasks.
- Treat exact-device requirement as implicitly scoped by its existing `BackendDeviceId.backendId`
  component rather than duplicating a second backend component.
- Complete the backend-contract milestone here and defer any new requirement dimension until a
  concrete config/planning use case proves it necessary.

## Known limitations

- Requirements cannot be combined, negated, preferred, avoided, weighted, or ordered.
- The family cannot express operation/data-type/shape/layout support or device properties beyond
  the selected identity/class targets.
- A requirement does not prove that a matching backend/device is registered, available, capable,
  or preparable and provides no evaluator.
- The no-match failure type/message and optional requirement placement remain undefined until
  their config/planning owner is implemented.
- No external serialization or compatibility guarantee is established.

## Validation evidence

- Planning inspected authoritative backend-contract ownership, dependency, partition-scoring,
  prepare/backend, and no-service-locator boundaries; completed identity/classification/snapshot
  source and tests; config/planning master plans; Public API, capability and backend-selection
  guides, glossary, roadmap; documentation/planning profiles; and the Java 26 module baseline.
- Exact backend, exact device, and coarse class are the three current concrete hard-eligibility
  targets. A sealed marker plus three records represents them without invalid combinations or
  consumer-layer behavior.
- The planned change contains five production, one test, and eight documentation/planning paths:
  exactly fourteen total.
- Planning validation passed across 219 Markdown files, 3,792 local links, 224 local anchors,
  2,744 fence markers, final newlines, and trailing whitespace.
- `git diff --check` passed. The planning diff contains exactly this task, the backend-contract
  master plan, the trace master plan, and the roadmap; it changes no Java, Gradle, architecture,
  API, guide, or glossary file before implementation.
- Repository planning contains exactly one `Ready` master-plan row: backend-contract task 0004.
  No backend-contract task 0005, config task specification, or trace task 0003 specification
  exists.
- The implementation context `/root/implement_backend_contract_0004` initially ran the focused
  `BackendRequirementTest`; executable behavior passed, but the test incorrectly expected five
  declared methods on a one-component record. The expectation was corrected to the ordinary four
  record methods, and the next focused run passed all six selected tests.
- After executable Java stabilized, the implementation context ran exactly one final
  `./gradlew :modules:backend-contract:test`; Gradle reported `BUILD SUCCESSFUL` in 783 ms. XML
  reports contain 22 tests across four suites with zero failures, errors, or skips:
  `BackendAvailabilitySnapshotTest` has 7, `BackendIdentityTest` has 6,
  `BackendRequirementTest` has 6, and `DeviceClassTest` has 3. Executable Java and tests have not
  changed since that final module run.
- The separate clean documentation context
  `/root/implement_backend_contract_0004/backend_contract_0004_docs` applied General style with
  API/Javadoc as the primary profile for the marker, records, package, and Public API; Backend
  guide for `capability-provider.md`; User guide for `backend-selection.md`; Planning for the
  task, master plans, and roadmap; and Example format for the current construction examples. It
  independently inspected the authoritative architecture and no-service-locator boundary,
  documentation/planning rules, completed tasks 0001–0003, final source/tests, public and backend
  guides, glossary, relevant module plans, roadmap, and Java 26 root/module configuration.
- The documentation context changed only Javadoc and Markdown after the successful final module
  suite. It did not change executable Java or tests and therefore reused the implementation
  context's Java-test evidence rather than rerunning a Java test suite.
- `./gradlew :modules:backend-contract:javadoc` reported `BUILD SUCCESSFUL` in 1 s with two
  executed tasks. Inspection of generated `package-summary.html`, `BackendRequirement.html`,
  `BackendIdRequirement.html`, `BackendDeviceIdRequirement.html`, and
  `DeviceClassRequirement.html` confirmed the rendered closed-family, equal-identity,
  exact-device owning-backend, class-wide eligibility, exact-reference, null-failure,
  accessor/result, and current-versus-planned boundaries.
- `python3 /tmp/validate_synaptik_markdown.py` passed 219 Markdown files, 3,793 local links, 224
  local anchors, 2,750 fence markers, final newlines, and trailing whitespace. Final
  `git diff --check` passed.
- The final scope command
  `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u` and
  `git status --short` confirmed exactly fourteen authorized paths: five production/Javadoc
  paths, one focused test, and eight documentation/planning paths. No unrelated path is present.
- Source and focused-test inspection confirms the exact sealed method-free interface; exact three
  permitted records; exact one-component public record shapes; exact null messages and reference
  retention; ordinary record equality, hashing, and diagnostic text; and no extra interface,
  nested type, sentinel, factory, matcher, evaluator, preference, combination, capability,
  scoring, provider, service, or serialization surface.
- Final-status inspection confirms backend-contract tasks 0001–0004 and the selected project
  area are Complete; no task is Ready. Trace 0003–0008 and every later project frontier remain
  Draft, and no backend-contract task 0005, trace task 0003, config task, or planning task
  specification exists.
- No change was needed in `ARCHITECTURE.md`, focused architecture explanations, ADR 0006, or
  architecture tests because the family implements the existing declarative backend-requirement
  ownership rule without changing module direction, composition, discovery, or dependency rules.
  No backend conformance or integration test changed because the values add no backend behavior
  or end-to-end execution.
- No change was needed in Gradle, dependencies, or the Java 26 configuration because all new
  production declarations are JDK-only and remain in the existing leaf module. Completed
  `BackendId`, `BackendDeviceId`, `DeviceClass`, `BackendAvailabilitySnapshot`, and their tests
  remain unchanged because requirements compose those values without changing their contracts.
- Config and planning Java remain unchanged because optionality, intent, eligibility evaluation,
  capability intersection, and no-match failure belong to later consumer tasks. Trace Java/tests
  remain unchanged because requirements are producer-domain vocabulary and no trace payload,
  correlation, translation, or emission was added. Other modules and concrete backends remain
  unchanged because the family adds no registration, discovery, provider, preparation, runtime,
  storage, route, kernel, or execution behavior.
- After the first documentation stabilization, the parent implementation context ran exactly one
  final `./gradlew test` repository capability checkpoint. Gradle reported `BUILD SUCCESSFUL in
  763ms`, `42 actionable tasks: 3 executed, 39 up-to-date`, and configuration-cache reuse. A
  read-only aggregation of the current JUnit XML reports found 1,055 tests across 135 suites with
  zero failures, errors, or skips. No file or executable Java changed during or after the
  checkpoint.

## Implementation notes

- Added the sealed method-free marker and the exact three one-component public records. Their
  canonical constructors preserve the specified null messages and exact component references
  without adding a helper, factory, sentinel, evaluator, preference, or consumer behavior.
- Added one focused test class locking the hierarchy, record/API shape, validation, reference
  retention, ordinary value behavior, and forbidden surface. The development-only five-method
  expectation was corrected to the four methods declared by an ordinary one-component record.
- The clean documentation context finalized the four type Javadocs, package Javadoc, Public API,
  capability-provider guide, backend-selection guide, glossary, and final planning status. It
  changed no executable Java behavior.

## Completion summary

- Completed changes: added the sealed hard-eligibility family, focused exact-contract tests, and
  finalized every authorized Javadoc, explanatory-documentation, glossary, and planning path.
- Files changed or created: exactly the five production/Javadoc, one test, and eight
  documentation/planning paths listed under Affected files.
- Tests and validation: the implementation context passed the focused six-test run and one final
  22-test/four-suite backend-contract module run; the documentation context reused that evidence
  because executable Java remained unchanged. The parent then passed the single final root
  checkpoint covering 1,055 tests across 135 suites with no failures, errors, or skips.
- Documentation-agent review: clean context
  `/root/implement_backend_contract_0004/backend_contract_0004_docs` independently reviewed and
  finalized all affected Javadocs and documentation.
- Documentation impact: only immutable hard-target vocabulary is current. Config owns later
  optionality and intent; planning owns later eligibility evaluation and no-match failure.
- Javadoc review: the marker, three variants, and package documentation distinguish equal-identity
  targets, exact reference retention, exact-device backend implication, class-wide eligibility,
  null failures, ordinary record behavior, and unsupported consumer/lifecycle behavior.
- Glossary impact: added the current backend-requirement family and distinguished hard
  eligibility from availability, capability, preference, and evaluation.
- Unresolved issues: None.
- Follow-up required: None. Any next project-area selection is a separate roadmap reassessment;
  this task creates no detailed follow-up specification.

Status: Complete
