# Task 0003: Backend Availability Snapshot

## Status

Complete

## Goal

Add one immutable, caller-supplied snapshot that reports which devices a particular backend
currently presents as available and the coarse class of each reported device.

Mental model:

```text
BackendAvailabilitySnapshot
  ├── backendId = BackendId("cuda")
  └── devices
      ├── BackendDeviceId("cuda", "0") -> DeviceClass.ACCELERATOR
      └── BackendDeviceId("cuda", "1") -> DeviceClass.ACCELERATOR
```

The snapshot is declarative point-in-time data. It neither discovers devices nor retains a live
backend. An empty device map reports that the supplying context currently exposes no available
device for that backend identity.

## Scope

- Add public record `BackendAvailabilitySnapshot` in
  `io.github.pho001.synaptik.backend.contract` with exactly these components in order:
  1. `BackendId backendId`
  2. `Map<BackendDeviceId, DeviceClass> devices`
- Define one snapshot as scoped to exactly one backend identity. The `backendId` remains meaningful
  when `devices` is empty and provides the namespace against which every reported device is
  validated.
- Validate in this exact order:
  1. reject null `backendId` with `NullPointerException` and exact message `backendId`;
  2. reject null `devices` with `NullPointerException` and exact message `devices`;
  3. while traversing source entries, reject a null key with `NullPointerException` and exact
     message `devices contains null deviceId`;
  4. reject a null value with `NullPointerException` and exact message
     `devices contains null deviceClass`;
  5. reject a device whose `BackendDeviceId.backendId()` is not equal to snapshot `backendId` with
     `IllegalArgumentException` and exact message
     `device backendId must match snapshot backendId`.
- Validate each source entry in its source-map iteration order, checking its key, value, then
  backend match before moving to the next entry. Do not promise deterministic choice among
  multiple invalid entries supplied by an unordered map.
- After validation, snapshot the map with `Map.copyOf` and expose only that immutable structural
  snapshot. Retain the exact `BackendId`, `BackendDeviceId`, and `DeviceClass` component references;
  do not retain the caller's mutable map reference.
- Accept an empty map. Its meaning is exactly no device currently reported available for this
  backend in this snapshot. Add no separate or derived `available` component or method.
- Accept one or more distinct device IDs for the backend. Ordinary map-key equality prevents
  duplicate equal device identities. Add no ordering, default-device, preference, priority, or
  fallback meaning; `Map.copyOf` iteration order is unspecified.
- Add no other record component, field, constructor, public method, nested type, or implemented
  interface. Preserve ordinary record equality, hashing, and diagnostic `toString()` behavior.
- Update package Javadoc to include the current snapshot and explicitly distinguish supplied
  availability facts from discovery, registration, capability, ownership, and liveness.
- Correct only the stale temporal sentence in `DeviceClass` Javadoc so it points to the now-current
  `BackendAvailabilitySnapshot`; do not change the enum declaration or behavior.
- Add one focused test class covering exact generic record/API shape, validation order/messages,
  empty and populated snapshots, backend consistency, structural immutability, source-map
  isolation, exact element-reference retention, equality, and forbidden API additions.
- Finalize Javadocs and update Public API status, the capability-provider guide, glossary,
  backend-contract master plan, and roadmap in the same overall change.

## Out of scope

- a separate device descriptor, availability-entry record, backend registration record, global
  snapshot, snapshot collection, registry, provider, service, manager, builder, or factory
- timestamps, monotonic clocks, generation/version counters, expiration, refresh, caching,
  polling, events, subscriptions, listeners, health, liveness, readiness, error reasons, or status
  messages
- device ordering, default selection, priority, preference, scoring, fallback, load, utilization,
  free/total memory, capacity, topology, properties, vendor, architecture, or native handles
- capability support, operation kinds, data types, shapes, layouts, precision, routes, kernels,
  executables, preparation, storage, workspaces, transfers, runtime residency, or execution
- backend/device discovery, environment probing, native calls, explicit engine registration,
  service locator, `ServiceLoader`, classpath scanning, or reflective plugin discovery
- modifying `BackendId`, `BackendDeviceId`, the executable declaration or behavior of
  `DeviceClass`, or any of their focused tests
- `BackendRequirement`, config intent/profile, planning candidate/scoring contracts, or compiler,
  prepare, runtime, engine, and concrete backend behavior
- trace IDs, trace payloads, attributes, translation, emission, or changes to trace Java/tests
- serialization, external schema, persistence, networking, dependencies, Gradle, Java version,
  architecture contract, ADR, architecture tests, another module, concrete backend, backend
  conformance, or integration changes
- a detailed backend-contract task 0004 or later specification
- returning to or implementing trace task 0003 or another trace task
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
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Backend-contract master plan](../master-plan.md)
- [Task 0001](0001-backend-and-device-identifiers.md)
- [Task 0002](0002-device-classification.md)
- [Public API status](../../../../api/public-api.md)
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/backend-contract` owns immutable backend availability snapshots as declarative facts.
- Snapshot creation consumes already-known facts. It must not perform discovery, registration,
  native access, capability evaluation, planning, preparation, or execution.
- Compile-time consumers may use supplied availability when selecting valid ownership candidates,
  but this DTO contains no score, operation support, or live provider object.
- `modules/backend-contract` remains JDK-only and independent of model, trace, config, planning,
  compiler, runtime, prepare, engine, and concrete backend modules.
- Stop if implementation needs another availability type, timestamp/status field, provider or
  registry interface, another module, dependency edit, or architecture decision.

## Package impact

Existing package retained:

- `io.github.pho001.synaptik.backend.contract` — minimal backend-neutral identities,
  classification, availability, and later declarative requirement vocabulary

Type added:

- `io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot` — immutable
  single-backend map of currently reported device identities to device classes

Existing package Javadoc is updated to explain the new current type. `DeviceClass` Javadoc receives
the necessary temporal association correction; its declaration and behavior remain unchanged.
`BackendId` and `BackendDeviceId` declarations and behavior remain unchanged. The focused test
mirrors the production package. No subpackage or helper type is added.

## Affected files

Production — exactly three paths:

- add `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/BackendAvailabilitySnapshot.java`
- Javadoc-only update to
  `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/package-info.java`
- Javadoc-only temporal-association correction to
  `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/DeviceClass.java`

Tests — exactly one path:

- add `modules/backend-contract/src/test/java/io/github/pho001/synaptik/backend/contract/BackendAvailabilitySnapshotTest.java`

Documentation and planning — exactly six paths:

- `docs/api/public-api.md`
- `docs/backend-guide/capability-provider.md`
- `docs/glossary.md`
- add and finalize this task
- `docs/planning/modules/backend-contract/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: completed task-0001/task-0002 source/tests/Javadocs, trace
source/tests/master plan, `AGENTS.md`, `ARCHITECTURE.md`, focused architecture and ADR
documents/tests, Gradle, config/planning placeholders and master plans, concrete backends, backend
conformance, integration tests, and legacy evidence.

## Maximum scope

At most the exact ten paths above. Stop if implementation requires another production type,
test, document, package, dependency, module, architecture change, build edit, or detailed
follow-up specification.

## Acceptance criteria

- `BackendAvailabilitySnapshot` is a public record in the exact backend-contract package with
  exactly `BackendId backendId` and `Map<BackendDeviceId, DeviceClass> devices` in that order.
- The constructor follows the exact top-level and per-entry validation order, exception types, and
  messages and accepts empty and populated same-backend maps.
- Every device key belongs by `BackendId.equals` to the snapshot backend. Reference identity is
  not required for that comparison, and unequal backend identities are rejected.
- `Map.copyOf` provides an immutable structural snapshot isolated from later source-map mutations.
  Exact keys and enum values are retained; the source map reference and its iteration order are
  not part of the stored contract.
- The empty map means no currently reported available device for this backend. No boolean status,
  default device, ordering, timestamp, reason, liveness, or capability promise is added.
- The record adds no extra public API, interface, nested type, serialization, discovery, provider,
  registry, or live behavior and preserves ordinary record value semantics.
- `BackendId`, `BackendDeviceId`, and all completed focused tests remain source- and
  behavior-unchanged. `DeviceClass` executable declaration and behavior remain unchanged; its
  Javadoc identifies the current snapshot association without assigning availability meaning to
  the category itself.
- Production imports are limited to JDK `Map`; no other Synaptik module or producer object appears.
- Focused tests fail on component/generic/API drift and cover every validation, immutability,
  backend-scope, empty-map, equality, and retention contract.
- Package Javadoc, Public API status, capability-provider guide, and glossary mark the supplied
  availability snapshot current while keeping capability providers, discovery, requirements,
  planning, prepare, runtime, engine, and concrete backends planned.
- Backend-contract master plan and roadmap identify 0003 as Complete. Task 0004 and trace 0003+
  remain Draft without detailed specifications, and no stale Ready frontier remains.
- A separate clean-context documentation-focused pass finalizes snapshot/package/`DeviceClass`
  Javadocs and six documentation/planning paths after Java tests pass.
- Exact ten-path scope, Markdown, final newlines, trailing whitespace, and `git diff --check`
  pass; no completed contract, architecture, dependency, Gradle, trace, config/planning Java,
  concrete backend, or cross-module implementation path changes.

## Tests / validation

Run focused tests while developing. After executable Java stabilizes, run exactly one final
backend-contract module suite:

```bash
./gradlew :modules:backend-contract:test
```

Record test and suite counts from XML reports. Do not run repository-wide tests: this remains a
small JDK-only additive DTO task with no dependency or architecture-rule change. Task 0004 will
own the backend-contract capability checkpoint after the complete minimal contract set exists.

Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That pass independently inspects final
source/tests and completed identity/classification contracts, applies General style with
API/Javadoc, Backend guide, Planning, and Example profiles where relevant, finalizes record/package
Javadocs and six documentation/planning paths, records reasoned no-change conclusions, and runs:

```bash
./gradlew :modules:backend-contract:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun. Inspect generated package and snapshot pages; rely on focused
automated record/generic/API-shape and validation checks; confirm exactly ten task paths; and
confirm backend-contract 0003 Complete/0004 Draft/no detailed 0004 task before completion.

## Dependencies

- Completed backend-contract task 0001 identities.
- Completed backend-contract task 0002 device classification.
- The selected model milestone and trace tasks 0001–0002 are Complete as roadmap sequencing
  context only; this task adds no model or trace dependency.

## Follow-up tasks

- Task 0004 declarative backend requirements remains Draft without a detailed specification. It
  may later constrain exact backend/device identities or coarse classes without retaining live
  services.
- After task 0004, run the backend-contract capability checkpoint and reassess whether config or a
  producer-backed trace slice is the next coherent frontier.
- Trace tasks remain Draft until their relevant producer-owned contracts are concrete.

## Architecture impact

Expected impact: None.

The task implements the existing availability-snapshot ownership rule and changes no module
direction, discovery/registration mechanism, lifecycle decision, or backend behavior.
Documentation changes implementation-status wording only.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, focused module/dependency/partition/prepare-backend/no-service-
locator architecture documents, documentation/planning rules and profiles, roadmap,
backend-contract and trace master plans, completed backend-contract tasks 0001–0002, task 0003,
current identity/classification/package source/tests, Public API, capability-provider guide,
glossary, relevant config/planning master plans, and Java 26 root configuration.

Implement docs/planning/modules/backend-contract/tasks/0003-backend-availability-snapshot.md
exactly inside its ten authorized paths. Add only BackendAvailabilitySnapshot, one focused test,
the permitted snapshot/package/DeviceClass-Javadoc updates, and status documentation updates.
Preserve exact component and
entry validation order/messages, same-backend device scope, Map.copyOf snapshot semantics, empty
snapshot meaning, and no-discovery/no-registration/no-capability/no-service boundaries. Add no
entry type, status/timestamp/reason, provider, registry, requirement, dependency, or later task.
Stop on architecture, availability semantics, dependency, package, affected-file, or
maximum-scope conflict.

Run focused tests while developing and exactly one final :modules:backend-contract:test after
executable Java stabilizes. Then hand the actual diff and Java evidence to a separate clean-context
documentation agent in the same overall change. That pass finalizes Javadocs, Public API,
capability-provider guide, glossary, task/master/roadmap status, runs backend-contract Javadoc and
documentation/scope checks, and reuses successful Java evidence unless executable behavior
changes. Mark 0003 Complete only after both passes succeed. Leave 0004 and trace 0003+ Draft
without detailed specifications.
```

## Local decisions

- Use one snapshot per backend. The explicit `backendId` preserves scope even when no device is
  available and allows every device key to be checked against one ownership domain.
- Use `Map<BackendDeviceId, DeviceClass>` instead of another public entry record. The map directly
  expresses unique device identity to class association with no extra abstraction.
- Let an empty map represent no currently available target for that backend. A separate boolean
  could disagree with the device facts and has no current consumer requiring that distinction.
- Use `Map.copyOf` for structural immutability and deliberately promise no device order. Default
  selection and preference belong to later config/planning policy, not availability facts.
- Validate keys, values, and backend scope before copying so callers receive task-owned failure
  messages. Multiple-invalid-entry choice follows source-map iteration and is not globally ordered.
- Add no timestamp or refresh semantics. The producer decides when to create and replace a
  snapshot; the DTO contains only the supplied point-in-time fact.
- Keep trace translation deferred. A later trace producer may translate availability facts into
  trace-owned DTOs without making trace depend on backend-contract.

## Known limitations

- The snapshot reports availability only at creation time and provides no freshness, expiry,
  liveness, health, refresh, or notification mechanism.
- It cannot explain why a device is absent or unavailable.
- It contains no device ordering, default, performance, capacity, topology, property, or
  capability information.
- It does not prove engine registration or guarantee that preparation will succeed.
- No external serialization or compatibility guarantee is established.

## Validation evidence

- Planning inspected authoritative backend-contract ownership, dependency, partition-scoring,
  prepare/backend, and no-service-locator boundaries; completed identity/classification source and
  tests; config/planning master plans; Public API, capability guide, glossary, roadmap;
  documentation/planning profiles; and the Java 26 module baseline.
- A single-backend immutable map is the smallest current fact shape that associates exact device
  identities with coarse classes, represents no available devices, and avoids another public DTO.
- The original planned change contained two production, one test, and six documentation/planning
  paths: exactly nine total. The clean documentation review found that `DeviceClass` Javadoc still
  described the association as future after the snapshot became current. The coordinator
  therefore authorized that one Javadoc-only correction as a tenth path; no executable enum or
  test change was authorized or made.
- Repository Markdown validation passed for 218 Markdown files, 3,760 local links, 220 local
  anchors, 2,732 fence markers, final newlines, and trailing whitespace.
- `git diff --check` passed. Planning contains exactly one Ready row, for backend-contract task
  0003. No detailed backend-contract task-0004 or trace task-0003 specification exists.
- The implementation context `/root/implement_backend_contract_0003` ran the focused
  `BackendAvailabilitySnapshotTest` twice during development; all seven selected tests passed in
  each run. After executable Java stabilized, that context ran exactly one final
  `./gradlew :modules:backend-contract:test`; Gradle reported `BUILD SUCCESSFUL` in 722 ms. XML
  reports contain 16 tests across 3 suites: `BackendAvailabilitySnapshotTest` has 7 tests,
  `BackendIdentityTest` has 6, and `DeviceClassTest` has 3, with zero failures, errors, or skips.
- The separate clean documentation context
  `/root/implement_backend_contract_0003/backend_contract_0003_docs` applied General style with
  API/Javadoc, Backend guide, Planning, and Example profiles as applicable. It independently
  inspected the authoritative architecture and no-service-locator boundary, documentation and
  planning rules, completed identity/classification contracts, final source/tests, Public API,
  capability guide, glossary, relevant module plans, roadmap, and Java 26 build configuration.
- Documentation review exposed one stale temporal sentence in `DeviceClass` Javadoc: it still
  described a later availability association after `BackendAvailabilitySnapshot` became current.
  The coordinator authorized that Javadoc-only correction as the tenth path. The enum declaration,
  executable behavior, and tests remained unchanged.
- The documentation context changed only Javadoc and Markdown after the successful final module
  suite. Executable Java and tests remained unchanged, so it reused the implementation context's
  Java-test evidence rather than rerunning it.
- `./gradlew :modules:backend-contract:javadoc` reported `BUILD SUCCESSFUL` in 987 ms with two
  executed tasks. Inspection of generated `BackendAvailabilitySnapshot.html`, `DeviceClass.html`,
  and `package-summary.html` confirmed the rendered backend scope, empty-map meaning, exact failure
  messages, structural-copy/reference-retention semantics, unspecified iteration order, current
  category association, and unsupported-behavior boundaries.
- `python3 /tmp/validate_synaptik_markdown.py` passed 218 Markdown files, 3,764 local links, 224
  local anchors, 2,734 fence markers, final newlines, and trailing whitespace. Final
  `git diff --check` passed.
- The final scope command
  `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u` and
  `git status --short` confirmed exactly ten authorized task paths: three production paths, one
  focused test, and six documentation/planning paths, with no unrelated working-tree path.
- Source and focused-test inspection confirms the exact two-component generic public record shape;
  top-level and per-entry validation order/messages; same-backend equality scope; empty and
  populated map behavior; `Map.copyOf` isolation; exact element-reference retention; ordinary
  record equality, hashing, and diagnostic text; and no extra component, field, public method,
  interface, nested type, serialization, provider, registry, discovery, or live behavior.
- Status and task-directory inspection confirms backend-contract 0003 and its master-plan row are
  Complete; task 0004 remains Draft without a detailed specification. Trace 0003–0008 remain
  Draft, and no detailed trace task-0003 specification exists.
- `BackendId`, `BackendDeviceId`, their Javadocs/source, and all completed focused tests remain
  unchanged because the snapshot composes the existing identity/category values without changing
  them. `DeviceClass` changed only in its authorized temporal Javadoc sentence; its exact enum
  declaration and classification behavior remain unchanged.
- No change was needed in `ARCHITECTURE.md`, focused architecture explanations, ADR 0006, or
  architecture tests because the snapshot implements the existing backend-contract DTO ownership
  and changes no module, dependency, composition, discovery, or service-locator rule. Gradle,
  dependencies, and the Java 26 baseline remain unchanged because the record uses only JDK `Map`.
- Trace Java/tests/master plan remain unchanged because this task adds no trace identity, payload,
  attribute, emission, or producer-to-trace translation. Config and planning Java/master plans
  remain unchanged because the snapshot adds no requirement, intent, capability query, scoring,
  ownership-selection, or policy behavior.
- Concrete backends, backend-conformance tests, and integration tests remain unchanged because a
  caller-supplied fact adds no discovery, registration, backend implementation, preparation,
  execution, or end-to-end behavior. Repository-wide Java validation remains deferred to task
  0004's recorded backend-contract capability checkpoint or CI, as specified for this additive
  leaf-module task.

## Implementation notes

- Added the exact two-component public record and one focused test class. The constructor preserves
  the planned top-level and per-entry validation order and messages, checks backend scope by value
  equality, then uses `Map.copyOf` without adding another public availability type or behavior.
- The clean documentation context finalized only the authorized snapshot, package, and
  `DeviceClass` Javadocs plus six Markdown/planning paths. It added the current snapshot to Public
  API, backend capability guidance, and the glossary, then synchronized task, master-plan, and
  roadmap status without changing executable Java.

## Completion summary

- Completed changes: added the immutable single-backend availability snapshot, focused exact-
  contract tests, finalized snapshot/package/`DeviceClass` Javadocs, and synchronized current
  public, glossary, guide, task, master-plan, and roadmap documentation.
- Files changed or created: exactly the three production, one test, and six documentation/planning
  paths listed under Affected files.
- Tests and validation: the implementation context passed two focused seven-test runs and one
  final 16-test/three-suite backend-contract module run; the documentation context reused that
  evidence because executable Java and tests remained unchanged.
- Documentation-agent review: clean context
  `/root/implement_backend_contract_0003/backend_contract_0003_docs` independently reviewed and
  finalized all affected Javadocs and documentation.
- Documentation impact: the caller-supplied point-in-time snapshot is current; discovery,
  refresh/liveness, requirements, capabilities, registration, planning, preparation, execution,
  concrete backend behavior, and trace translation remain planned or separately owned.
- Javadoc review: the record and package documentation define backend scope, empty-map meaning,
  exact validation order and failures, immutable structural copying, exact element-reference
  retention, unspecified map iteration order, and unsupported lifecycle behavior. `DeviceClass`
  now names the current association while remaining category vocabulary only.
- Glossary impact: added the current `BackendAvailabilitySnapshot` definition and updated the
  backend-contract foundation and `DeviceClass` association boundary.
- Unresolved issues: None.
- Follow-up required: None. Task 0004 and trace tasks 0003–0008 remain Draft without detailed
  specifications.

Status: Complete
