# Task 0003: [Backend Availability Snapshot](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot)

> **Editorial reading key — not part of the historical record.**
>
> Here the supplying context passes a device-to-class map to BackendAvailabilitySnapshot. The
> constructor takes an immutable structural snapshot of that map while retaining its identity and
> class elements; it does not copy device memory. Planning consumes the report as supplied
> availability, not as discovery or a continuing liveness guarantee. The context and backend
> retain responsibility for any actual devices or services outside this data value.
>
> Current reading: [backend identities, supplied availability and requirements](../../../../reference/backend-identities.md); [capability queries and hard-eligible owner selection](../../../../reference/compiler/planning.md#current-operation-capability-contracts).

## Status

Complete

## Goal

Add one [immutable](../../../../glossary.md#term-immutable), caller-supplied [snapshot](../../../../glossary.md#term-snapshot) that reports which devices a particular [backend](../../../../glossary.md#backend)
currently presents as available and the coarse class of each reported [device](../../../../glossary.md#term-device).

[Mental model](../../../../glossary.md#term-mental-model):

```text
BackendAvailabilitySnapshot
  ├── backendId = BackendId("cuda")
  └── devices
      ├── BackendDeviceId("cuda", "0") -> DeviceClass.ACCELERATOR
      └── BackendDeviceId("cuda", "1") -> DeviceClass.ACCELERATOR
```

The snapshot is declarative point-in-time data. It neither discovers devices nor retains a live
backend. An empty device map reports that the supplying context currently exposes no available
device for that [backend identity](../../../../glossary.md#backend-identity--backendid).

## [Scope](../../../../glossary.md#term-scope)

- Add public record [`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot) in
  `io.github.pho001.synaptik.backend.contract` with exactly these components in order:
  1. `BackendId backendId`
  2. `Map<BackendDeviceId, DeviceClass> devices`
- Define one [snapshot](../../../../glossary.md#term-snapshot) as scoped to exactly one [backend identity](../../../../glossary.md#backend-identity--backendid). The `backendId` remains meaningful
  when `devices` is empty and provides the namespace against which every reported device is
  validated.
- Validate in this exact order:
  1. reject null `backendId` with `NullPointerException` and exact message `backendId`;
  2. reject null `devices` with `NullPointerException` and exact message `devices`;
  3. while traversing source entries, reject a null key with `NullPointerException` and exact
     message `devices contains null deviceId`;
  4. reject a null value with `NullPointerException` and exact message
     `devices contains null deviceClass`;
  5. reject a [device](../../../../glossary.md#term-device) whose `BackendDeviceId.backendId()` is not equal to snapshot `backendId` with
     `IllegalArgumentException` and exact message
     `device backendId must match snapshot backendId`.
- Validate each source entry in its source-map iteration order, checking its key, value, then
  backend match before moving to the next entry. Do not promise deterministic choice among
  multiple invalid entries supplied by an unordered map.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- After [validation](../../../../glossary.md#term-validation), snapshot the map with `Map.copyOf` and expose only that immutable structural
  snapshot. Retain the exact <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>[`BackendId`](../../../../glossary.md#backend-identity--backendid), [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), and [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) component references;
  do not retain the caller's mutable map reference.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>
- Accept an empty map. Its meaning is exactly no device currently reported available for this
  backend in this snapshot. Add no separate or derived `available` component or method.
- Accept one or more distinct device IDs for the backend. Ordinary map-key equality prevents
  duplicate equal device identities. Add no ordering, default-device, preference, priority, or
  fallback meaning; <span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>`Map.copyOf` iteration order is unspecified.
- Add no other record component, field, constructor, public method, nested type, or implemented
  interface. Preserve ordinary record equality, hashing, and diagnostic `toString()` behavior.
- Update package Javadoc to include the current snapshot and explicitly distinguish supplied
  availability facts from discovery, registration, capability, ownership, and liveness.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- Correct only the stale temporal sentence in `DeviceClass` [Javadoc](../../../../glossary.md#term-javadoc) so it points to the now-current
  `BackendAvailabilitySnapshot`; do not change the enum declaration or behavior.
- Add one focused test class covering exact generic record/API shape, validation order/messages,
  empty and populated snapshots, backend consistency, structural immutability, source-map
  isolation, exact element-reference retention, equality, and forbidden API additions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [element glossary](../../../../glossary.md#term-element))</span>
- Finalize Javadocs and update Public API status, the capability-provider guide, glossary,
  backend-contract master plan, and roadmap in the same overall change.

## Out of [scope](../../../../glossary.md#term-scope)

- a separate device descriptor, availability-entry record, backend registration record, global
  snapshot, snapshot collection, registry, provider, service, manager, builder, or factory<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [descriptor glossary](../../../../glossary.md#term-descriptor); [snapshot glossary](../../../../glossary.md#term-snapshot); [registry glossary](../../../../glossary.md#term-registry))</span>
- timestamps, monotonic clocks, generation/version counters, expiration, refresh, caching,
  polling, events, subscriptions, listeners, health, liveness, readiness, error reasons, or status
  messages<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness))</span>
- device ordering, default selection, priority, preference, scoring, fallback, load, utilization,
  free/total memory, capacity, topology, properties, vendor, architecture, or native handles<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>
- capability support, operation kinds, data types, shapes, layouts, precision, routes, kernels,
  executables, preparation, storage, workspaces, transfers, runtime residency, or execution<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [precision glossary](../../../../glossary.md#term-precision); [kernels glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency))</span>
- backend/device discovery, environment probing, native calls, explicit engine registration,
  service locator, <span data-glossary-navigation="true"> ([engine glossary](../../../../glossary.md#term-engine); [service locator glossary](../../../../glossary.md#term-service-locator))</span>`ServiceLoader`, [classpath](../../../../glossary.md#term-classpath) scanning, or reflective [plugin](../../../../glossary.md#term-plugin) discovery
- modifying [`BackendId`](../../../../glossary.md#backend-identity--backendid), [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), the executable declaration or behavior of
  [`DeviceClass`](../../../../glossary.md#device-class--deviceclass), or any of their focused tests
- [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement), config intent/profile, planning candidate/scoring contracts, or compiler,
  prepare, runtime, engine, and concrete backend behavior<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- [trace](../../../../glossary.md#trace) IDs, trace payloads, attributes, translation, emission, or changes to trace Java/tests
- serialization, external schema, persistence, networking, dependencies, Gradle, Java version,
  architecture contract, ADR, architecture tests, another module, concrete backend, backend
  conformance, or integration changes<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [schema glossary](../../../../glossary.md#term-schema); [persistence glossary](../../../../glossary.md#term-persistence); [Gradle glossary](../../../../glossary.md#term-gradle); [architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- a detailed [backend](../../../../glossary.md#backend)-contract task 0004 or later specification
- returning to or implementing trace task 0003 or another trace task
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
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
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

- `modules/backend-contract` owns [immutable](../../../../glossary.md#term-immutable) [backend](../../../../glossary.md#backend) availability snapshots as declarative facts.
- Snapshot creation consumes already-known facts. It must not perform discovery, registration,
  native access, capability evaluation, planning, preparation, or execution.<span data-glossary-navigation="true"> ([Snapshot glossary](../../../../glossary.md#term-snapshot); [evaluation glossary](../../../../glossary.md#term-evaluation); [planning glossary](../../../../glossary.md#term-planning))</span>
- Compile-time consumers may use supplied availability when selecting valid ownership candidates,
  but this DTO contains no score, operation support, or live provider object.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [DTO glossary](../../../../glossary.md#data-transfer-object--dto); [operation glossary](../../../../glossary.md#operation))</span>
- `modules/backend-contract` remains JDK-only and independent of model, trace, config, planning,
  compiler, runtime, prepare, engine, and concrete backend modules.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [model glossary](../../../../glossary.md#term-model); [trace glossary](../../../../glossary.md#trace); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>
- Stop if implementation needs another availability type, timestamp/status field, provider or
  registry interface, another module, dependency edit, or architecture decision.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package retained:

- `io.github.pho001.synaptik.backend.contract` — minimal backend-neutral identities,
  classification, availability, and later declarative requirement vocabulary<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

Type added:

- `io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot` — immutable
  single-backend map of currently reported device identities to device classes<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [device glossary](../../../../glossary.md#term-device))</span>

Existing package [Javadoc](../../../../glossary.md#term-javadoc) is updated to explain the new current type. [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) Javadoc receives
the necessary temporal association correction; its declaration and behavior remain unchanged.
[`BackendId`](../../../../glossary.md#backend-identity--backendid) and [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid) declarations and behavior remain unchanged. The focused test
mirrors the production package. No subpackage or helper type is added.

## Affected files

Production — exactly three paths:

- add `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/BackendAvailabilitySnapshot.java`
- [Javadoc](../../../../glossary.md#term-javadoc)-only update to
  `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/package-info.java`
- Javadoc-only temporal-association correction to
  `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/DeviceClass.java`

Tests — exactly one path:

- add `modules/backend-contract/src/test/java/io/github/pho001/synaptik/backend/contract/BackendAvailabilitySnapshotTest.java`

Documentation and [planning](../../../../glossary.md#term-planning) — exactly six paths:

- `docs/api/public-api.md`
- `docs/backend-guide/capability-provider.md`
- `docs/glossary.md`
- add and finalize this task
- `docs/planning/modules/backend-contract/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: completed task-0001/task-0002 source/tests/Javadocs, [trace](../../../../glossary.md#trace)
source/tests/master plan, `AGENTS.md`, `ARCHITECTURE.md`, focused architecture and [ADR](../../../../glossary.md#term-architecture-decision-record-adr)
documents/tests, [Gradle](../../../../glossary.md#term-gradle), config/planning placeholders and master plans, concrete [backends](../../../../glossary.md#backend), backend
[conformance](../../../../glossary.md#term-conformance), integration tests, and legacy evidence.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact ten paths above. Stop if implementation requires another production type,
test, document, package, [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module), architecture change, build edit, or detailed
follow-up specification.

## Acceptance criteria

- [`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot) is a public record in the exact backend-contract package with
  exactly <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>`BackendId backendId` and `Map<BackendDeviceId, DeviceClass> devices` in that order.
- The constructor follows the exact top-level and per-entry validation order, exception types, and
  messages and accepts empty and populated same-backend maps.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Every [device](../../../../glossary.md#term-device) key belongs by `BackendId.equals` to the snapshot backend. Reference identity is
  not required for that comparison, and unequal backend identities are rejected.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Reference identity glossary](../../../../glossary.md#term-reference-identity))</span>
- `Map.copyOf` provides an immutable structural snapshot isolated from later source-map mutations.
  Exact keys and enum values are retained; the source map reference and its iteration order are
  not part of the stored contract.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- The empty map means no currently reported available device for this backend. No boolean status,
  default device, ordering, timestamp, reason, liveness, or capability promise is added.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- The record adds no extra public API, interface, nested type, serialization, discovery, provider,
  registry, or live behavior and preserves ordinary record value semantics.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [serialization glossary](../../../../glossary.md#term-serialization); [registry glossary](../../../../glossary.md#term-registry))</span>
- [`BackendId`](../../../../glossary.md#backend-identity--backendid), [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), and all completed focused tests remain source- and
  behavior-unchanged. [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) executable declaration and behavior remain unchanged; its
  Javadoc identifies the current snapshot association without assigning availability meaning to
  the category itself.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Production imports are limited to [JDK](../../../../glossary.md#term-java-development-kit-jdk) `Map`; no other Synaptik [module](../../../../glossary.md#term-module) or [producer](../../../../glossary.md#term-producer) object appears.
- Focused tests fail on component/generic/API drift and cover every validation, immutability,
  backend-scope, empty-map, equality, and retention contract.
- Package Javadoc, Public API status, capability-provider guide, and glossary mark the supplied
  availability snapshot current while keeping capability providers, discovery, requirements,
  planning, prepare, runtime, engine, and concrete backends planned.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine))</span>
- Backend-contract master plan and roadmap identify 0003 as Complete. Task 0004 and trace 0003+
  remain Draft without detailed specifications, and no stale Ready frontier remains.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [Ready glossary](../../../../glossary.md#term-ready))</span>
- A separate clean-context documentation-focused pass finalizes [snapshot](../../../../glossary.md#term-snapshot)/package/`DeviceClass`
  Javadocs and six documentation/planning paths after Java tests pass.
- Exact ten-path [scope](../../../../glossary.md#term-scope), Markdown, final newlines, trailing whitespace, and `git diff --check`
  pass; no completed contract, architecture, dependency, Gradle, trace, config/planning Java,
  concrete backend, or cross-module implementation path changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests while developing. After executable Java stabilizes, run exactly one final
[backend](../../../../glossary.md#backend)-contract [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:backend-contract:test
```

Record test and suite counts from XML reports. Do not run repository-wide tests: this remains a
small [JDK](../../../../glossary.md#term-java-development-kit-jdk)-only additive [DTO](../../../../glossary.md#data-transfer-object--dto) task with no [dependency](../../../../glossary.md#term-dependency) or architecture-rule change. Task 0004 will
own the backend-contract capability [checkpoint](../../../../glossary.md#term-checkpoint) after the complete minimal contract set exists.

Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That pass independently inspects final
source/tests and completed [identity](../../../../glossary.md#term-identity-meanings)/classification contracts, applies General style with
[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), Backend guide, [Planning](../../../../glossary.md#term-planning), and Example profiles where relevant, finalizes record/package
Javadocs and six documentation/planning paths, records reasoned no-change conclusions, and runs:

```bash
./gradlew :modules:backend-contract:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun. Inspect generated package and [snapshot](../../../../glossary.md#term-snapshot) pages; rely on focused
automated record/generic/API-[shape](../../../../glossary.md#shape) and validation checks; confirm exactly ten task paths; and
confirm backend-contract 0003 Complete/0004 Draft/no detailed 0004 task before completion.

## Dependencies

- Completed [backend](../../../../glossary.md#backend)-contract task 0001 [identities](../../../../glossary.md#term-identity-meanings).
- Completed backend-contract task 0002 [device](../../../../glossary.md#term-device) classification.
- The selected model milestone and trace tasks 0001–0002 are Complete as roadmap sequencing
  context only; this task adds no model or trace dependency.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [trace glossary](../../../../glossary.md#trace); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Follow-up tasks

- Task 0004 declarative backend requirements remains Draft without a detailed specification. It
  may later constrain exact backend/device identities or coarse classes without retaining live
  services.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [device glossary](../../../../glossary.md#term-device))</span>
- After task 0004, run the backend-contract capability checkpoint and reassess whether config or a
  producer-backed trace slice is the next coherent frontier.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [slice glossary](../../../../glossary.md#slice))</span>
- [Trace](../../../../glossary.md#trace) tasks remain Draft until their relevant [producer](../../../../glossary.md#term-producer)-owned contracts are concrete.

## Architecture impact

Expected impact: None.

The task implements the existing availability-[snapshot](../../../../glossary.md#term-snapshot) [ownership](../../../../glossary.md#term-ownership-meanings) rule and changes no [module](../../../../glossary.md#term-module)
direction, discovery/registration mechanism, [lifecycle](../../../../glossary.md#lifecycle) decision, or [backend](../../../../glossary.md#backend) behavior.
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

- Use one [snapshot](../../../../glossary.md#term-snapshot) per [backend](../../../../glossary.md#backend). The explicit `backendId` preserves scope even when no device is
  available and allows every device key to be checked against one ownership domain.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [device glossary](../../../../glossary.md#term-device); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Use `Map<BackendDeviceId, DeviceClass>` instead of another public entry record. The map directly
  expresses unique device identity to class association with no extra abstraction.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Let an empty map represent no currently available target for that backend. A separate boolean
  could disagree with the device facts and has no current consumer requiring that distinction.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- Use `Map.copyOf` for structural immutability and deliberately promise no device order. Default
  selection and preference belong to later config/planning policy, not availability facts.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Validate keys, values, and backend scope before copying so callers receive task-owned failure
  messages. Multiple-invalid-entry choice follows source-map iteration and is not globally ordered.
- Add no timestamp or refresh semantics. The producer decides when to create and replace a
  snapshot; the DTO contains only the supplied point-in-time fact.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [DTO glossary](../../../../glossary.md#data-transfer-object--dto))</span>
- Keep trace translation deferred. A later trace producer may translate availability facts into
  trace-owned DTOs without making trace depend on backend-contract.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>

## Known limitations

- The snapshot reports availability only at creation time and provides no freshness, expiry,
  liveness, health, refresh, or notification mechanism.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- It cannot explain why a [device](../../../../glossary.md#term-device) is absent or unavailable.
- It contains no device ordering, default, performance, capacity, topology, property, or
  capability information.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>
- It does not prove [engine](../../../../glossary.md#term-engine) registration or guarantee that preparation will succeed.
- No external [serialization](../../../../glossary.md#term-serialization) or compatibility guarantee is established.

## [Validation](../../../../glossary.md#term-validation) evidence

- Planning inspected authoritative backend-contract ownership, dependency, partition-scoring,
  prepare/backend, and no-service-locator boundaries; completed identity/classification source and
  tests; config/planning master plans; Public API, capability guide, glossary, roadmap;
  documentation/planning profiles; and the Java 26 module baseline.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency); [partition glossary](../../../../glossary.md#partition); [prepare glossary](../../../../glossary.md#prepare); [identity glossary](../../../../glossary.md#term-identity-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>
- A single-backend immutable map is the smallest current fact shape that associates exact device
  identities with coarse classes, represents no available devices, and avoids another public DTO.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [shape glossary](../../../../glossary.md#shape); [device glossary](../../../../glossary.md#term-device); [DTO glossary](../../../../glossary.md#data-transfer-object--dto))</span>
- The original planned change contained two production, one test, and six documentation/planning
  paths: exactly nine total. The clean documentation review found that [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) Javadoc still
  described the association as future after the snapshot became current. The coordinator
  therefore authorized that one Javadoc-only correction as a tenth path; no executable enum or
  test change was authorized or made.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Repository Markdown validation passed for 218 Markdown files, 3,760 local links, 220 local
  anchors, 2,732 fence markers, final newlines, and trailing whitespace.
- `git diff --check` passed. Planning contains exactly one Ready row, for backend-contract task
  0003. No detailed backend-contract task-0004 or trace task-0003 specification exists.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [trace glossary](../../../../glossary.md#trace))</span>
- The implementation context `/root/implement_backend_contract_0003` ran the focused
  `BackendAvailabilitySnapshotTest` twice during development; all seven selected tests passed in
  each run. After executable Java stabilized, that context ran exactly one final<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
  `./gradlew :modules:backend-contract:test`; [Gradle](../../../../glossary.md#term-gradle) reported `BUILD SUCCESSFUL` in 722 ms. XML
  reports contain 16 tests across 3 suites: `BackendAvailabilitySnapshotTest` has 7 tests,
  `BackendIdentityTest` has 6, and `DeviceClassTest` has 3, with zero failures, errors, or skips.
- The separate clean documentation context
  `/root/implement_backend_contract_0003/backend_contract_0003_docs` applied General style with
  API/Javadoc, Backend guide, Planning, and Example profiles as applicable. It independently
  inspected the authoritative architecture and no-service-locator boundary, documentation and
  planning rules, completed identity/classification contracts, final source/tests, Public API,
  capability guide, glossary, relevant module plans, roadmap, and Java 26 build configuration.
- Documentation review exposed one stale temporal sentence in `DeviceClass` Javadoc: it still
  described a later availability association after [`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot) became current.
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
- The final [scope](../../../../glossary.md#term-scope) command
  `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u` and
  `git status --short` confirmed exactly ten authorized task paths: three production paths, one
  focused test, and six documentation/planning paths, with no unrelated working-tree path.
- Source and focused-test inspection confirms the exact two-component generic public record shape;
  top-level and per-entry validation order/messages; same-backend equality scope; empty and
  populated map behavior; `Map.copyOf` isolation; exact element-reference retention; ordinary
  record equality, hashing, and diagnostic text; and no extra component, field, public method,
  interface, nested type, serialization, provider, registry, discovery, or live behavior.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [serialization glossary](../../../../glossary.md#term-serialization); [registry glossary](../../../../glossary.md#term-registry))</span>
- Status and task-directory inspection confirms backend-contract 0003 and its master-plan row are
  Complete; task 0004 remains Draft without a detailed specification. Trace 0003–0008 remain
  Draft, and no detailed trace task-0003 specification exists.
- [`BackendId`](../../../../glossary.md#backend-identity--backendid), [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), their Javadocs/source, and all completed focused tests remain
  unchanged because the snapshot composes the existing identity/category values without changing
  them. `DeviceClass` changed only in its authorized temporal Javadoc sentence; its exact enum
  declaration and classification behavior remain unchanged.
- No change was needed in `ARCHITECTURE.md`, focused architecture explanations, ADR 0006, or
  architecture tests because the snapshot implements the existing backend-contract DTO ownership
  and changes no module, dependency, composition, discovery, or service-locator rule. Gradle,
  dependencies, and the Java 26 baseline remain unchanged because the record uses only JDK <span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`Map`.
- Trace Java/tests/master plan remain unchanged because this task adds no trace identity, payload,
  attribute, emission, or producer-to-trace translation. Config and planning Java/master plans
  remain unchanged because the snapshot adds no requirement, intent, capability query, scoring,
  ownership-selection, or policy behavior.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [producer glossary](../../../../glossary.md#term-producer))</span>
- Concrete backends, backend-conformance tests, and integration tests remain unchanged because a
  caller-supplied fact adds no discovery, registration, backend implementation, preparation,
  execution, or end-to-end behavior. Repository-wide Java validation remains deferred to task
  0004's recorded backend-contract capability checkpoint or CI, as specified for this additive
  leaf-module task.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci); [leaf glossary](../../../../glossary.md#term-leaf))</span>

## Implementation notes

- Added the exact two-component public record and one focused test class. The constructor preserves
  the planned top-level and per-entry validation order and messages, checks backend scope by value
  equality, then uses <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend); [scope glossary](../../../../glossary.md#term-scope))</span>`Map.copyOf` without adding another public availability type or behavior.
- The clean documentation context finalized only the authorized [snapshot](../../../../glossary.md#term-snapshot), package, and
  [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) Javadocs plus six Markdown/planning paths. It added the current snapshot to Public
  API, backend capability guidance, and the glossary, then synchronized task, master-plan, and
  roadmap status without changing executable Java.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend capability glossary](../../../../glossary.md#backend-capability))</span>

## Completion summary

- Completed changes: added the immutable single-backend availability snapshot, focused exact-
  contract tests, finalized snapshot/package/<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [backend availability snapshot glossary](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>[`DeviceClass`](../../../../glossary.md#device-class--deviceclass) Javadocs, and synchronized current
  public, glossary, guide, task, master-plan, and roadmap documentation.
- Files changed or created: exactly the three production, one test, and six documentation/planning
  paths listed under Affected files.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: the implementation context passed two focused seven-test runs and one
  final 16-test/three-suite backend-contract module run; the documentation context reused that
  evidence because executable Java and tests remained unchanged.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend); [module glossary](../../../../glossary.md#term-module); [run glossary](../../../../glossary.md#term-run))</span>
- Documentation-agent review: clean context
  `/root/implement_backend_contract_0003/backend_contract_0003_docs` independently reviewed and
  finalized all affected Javadocs and documentation.
- Documentation impact: the caller-supplied point-in-time snapshot is current; discovery,
  refresh/liveness, requirements, capabilities, registration, planning, preparation, execution,
  concrete backend behavior, and trace translation remain planned or separately owned.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [concrete backend glossary](../../../../glossary.md#concrete-backend); [trace glossary](../../../../glossary.md#trace))</span>
- Javadoc review: the record and package documentation define backend scope, empty-map meaning,
  exact validation order and failures, immutable structural copying, exact element-reference
  retention, unspecified map iteration order, and unsupported lifecycle behavior. <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [element glossary](../../../../glossary.md#term-element); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>`DeviceClass`
  now names the current association while remaining category vocabulary only.
- Glossary impact: added the current [`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot) definition and updated the
  backend-contract foundation and `DeviceClass` association boundary.
- Unresolved issues: None.
- Follow-up required: None. Task 0004 and trace tasks 0003–0008 remain Draft without detailed
  specifications.

Status: Complete
