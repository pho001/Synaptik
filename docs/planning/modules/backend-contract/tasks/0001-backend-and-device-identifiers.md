# Task 0001: [Backend](../../../../glossary.md#backend) and [Device](../../../../glossary.md#term-device) Identifiers

> **Editorial reading key — not part of the historical record.**
>
> A component constructing BackendId or BackendDeviceId supplies names, not live implementation
> objects. The device token is interpreted only inside its backend namespace: the same token under
> two backend names denotes different identities. The records retain the supplied immutable
> component references; constructing either record neither copies tensor values nor acquires a
> device. Later Planning can use these names without receiving resource handles.
>
> Current reading: [backend identities, supplied availability and requirements](../../../../reference/backend-identities.md).

## Status

Complete

## Goal

Replace the `modules/backend-contract` placeholder with the two minimal public [identity](../../../../glossary.md#term-identity-meanings) values
required to name a [backend](../../../../glossary.md#backend) [owner](../../../../glossary.md#term-owner) and one [device](../../../../glossary.md#term-device) within that backend's [identity domain](../../../../glossary.md#term-identity-domain).

[Mental model](../../../../glossary.md#term-mental-model):

```text
BackendId("cpu")
  └── BackendDeviceId(backendId = "cpu", value = "default")

BackendId("cuda")
  └── BackendDeviceId(backendId = "cuda", value = "0")
```

[`BackendId`](../../../../glossary.md#backend-identity--backendid) names the backend that may own planned work. [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid) combines that [backend
identity](../../../../glossary.md#backend-identity--backendid) with an [opaque](../../../../glossary.md#term-opaque) backend-defined device [token](../../../../glossary.md#term-token), so a token such as `"0"` is never interpreted
without its owning backend. This task defines [immutable](../../../../glossary.md#term-immutable) vocabulary only. It performs no discovery,
registration, availability query, capability [evaluation](../../../../glossary.md#term-evaluation), [ownership](../../../../glossary.md#term-ownership-meanings) selection, or device access.

## [Scope](../../../../glossary.md#term-scope)

- Delete the temporary `BackendContractModule` marker.
- Add package documentation for `io.github.pho001.synaptik.backend.contract` that distinguishes
  declarative shared contracts from concrete backend implementations and live services.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- Add public record [`BackendId`](../../../../glossary.md#backend-identity--backendid) with exactly one `String value` component.
  - The value names a [backend ownership](../../../../glossary.md#backend-ownership) domain such as `"cpu"`, `"metal"`, or `"cuda"` without
    reserving those examples as constants or a closed set.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>
  - Reject `null` with `NullPointerException` and exact message `value`.
  - Reject `String.isBlank()` values with `IllegalArgumentException` and exact message
    `value must not be blank`.
  - Accept every other `String`, retain the exact supplied reference and content, and perform no
    trimming, case folding, Unicode normalization, syntax validation, interning, or aliasing.<span data-glossary-navigation="true"> ([Unicode normalization glossary](../../../../glossary.md#term-unicode-normalization); [validation glossary](../../../../glossary.md#term-validation); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
  - Expose an explicitly documented `value()` accessor.
- Add public record [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid) with exactly `BackendId backendId` and `String value`, in
  that order.
  - The first component establishes the device-token namespace and prevents cross-backend token
    collisions.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token); [backend glossary](../../../../glossary.md#backend))</span>
  - Validate `backendId`, then `value`, in component order.
  - Reject a null [backend identity](../../../../glossary.md#backend-identity--backendid) with `NullPointerException` and exact message `backendId`.
  - Reject a null [device](../../../../glossary.md#term-device) value with `NullPointerException` and exact message `value`.
  - Reject `String.isBlank()` device values with `IllegalArgumentException` and exact message
    `value must not be blank`.
  - Accept every other device value, retain both exact references, and perform no [normalization](../../../../glossary.md#term-normalization).
  - Expose explicitly documented `backendId()` and `value()` accessors.
- Both records must:
  - remain final through record semantics and add no other component, field, constructor, public
    method, nested type, or implemented interface;
  - preserve ordinary record equality, hashing, and diagnostic `toString()` behavior;
  - define [identity](../../../../glossary.md#term-identity-meanings) by their exact stored components without object [identity](../../../../glossary.md#term-reference-identity) or hidden state; and
  - add no [serialization](../../../../glossary.md#term-serialization), allocation, [registry](../../../../glossary.md#term-registry), discovery, availability, or [lifecycle](../../../../glossary.md#lifecycle) behavior.
- Add one focused test class covering exact package/API shape, component order and types,
  validation order and messages, exact reference/content retention, open backend names,
  backend-scoped device equality, ordinary record behavior, and forbidden interfaces/nested types.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape))</span>
- Finalize Javadocs and update Public API status, the capability-provider guide, glossary,
  backend-contract master plan, trace interleave record, and roadmap in the same overall change.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [`DeviceClass`](../../../../glossary.md#device-class--deviceclass), [`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot), [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement), capability-provider,
  prepare, executable, storage, buffer, memory, route, kernel, or service contracts<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [buffer glossary](../../../../glossary.md#term-buffer); [kernel glossary](../../../../glossary.md#kernel))</span>
- predefined [CPU](../../../../glossary.md#term-cpu), [Metal](../../../../glossary.md#term-metal), [CUDA](../../../../glossary.md#term-cuda), [accelerator](../../../../glossary.md#term-accelerator), [host](../../../../glossary.md#term-host), default-[device](../../../../glossary.md#term-device), or route [constants](../../../../glossary.md#term-constant)
- enums or a closed [backend](../../../../glossary.md#backend)/device vocabulary
- numeric device ordinals, device handles, native IDs, addresses, UUID allocation, parsing,
  formatting, comparison ordering, aliases, or canonicalization<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- backend registration, [lookup](../../../../glossary.md#term-lookup), discovery, `ServiceLoader`, classpath scanning, a service locator,
  singleton, registry, cache, allocator, global counter, environment probing, or native calls<span data-glossary-navigation="true"> ([classpath glossary](../../../../glossary.md#term-classpath); [service locator glossary](../../../../glossary.md#term-service-locator); [singleton glossary](../../../../glossary.md#term-singleton); [registry glossary](../../../../glossary.md#term-registry); [cache glossary](../../../../glossary.md#term-cache))</span>
- availability, capability support, ownership selection, scoring, partitioning, preparation,
  execution, runtime residency, transfer, publication, or tracing behavior<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [publication glossary](../../../../glossary.md#term-publication))</span>
- [`TraceBackendId`](../../../../glossary.md#trace-backend-correlation-identity--tracebackendid), [`TraceDeviceId`](../../../../glossary.md#trace-device-correlation-identity--tracedeviceid), a [trace](../../../../glossary.md#trace) mapping [API](../../../../glossary.md#term-application-programming-interface-api), or changes to `modules/trace` Java/tests
- dependencies, Gradle, Java version, architecture contract, ADR, architecture tests, another
  module, backend conformance, integration tests, or concrete backend implementations<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- a detailed task 0002 or later backend-contract specification
- implementing trace task 0003 or any later [trace payload](../../../../glossary.md#trace-payload--tracepayload)/attribute task
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle explanation](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Partition scoring](../../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition))</span>
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
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
- [Trace master plan](../../trace/master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>
- [Public API status](../../../../api/public-api.md)
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/backend-contract` owns minimal [backend](../../../../glossary.md#backend) [identities](../../../../glossary.md#term-identity-meanings) and declarative requirements.
- [Compile](../../../../glossary.md#compile)-time [ownership](../../../../glossary.md#term-ownership-meanings) refers to [`BackendId`](../../../../glossary.md#backend-identity--backendid), never a [concrete backend](../../../../glossary.md#concrete-backend) object or live service.
- [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid) is descriptive identity only. It must not expose a device handle, imply
  availability, or make a resource usable.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [device glossary](../../../../glossary.md#term-device))</span>
- The module remains JDK-only and independent of model, trace, config, planning, compiler,
  runtime, prepare, engine, and concrete backend modules.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [model glossary](../../../../glossary.md#term-model); [trace glossary](../../../../glossary.md#trace); [planning glossary](../../../../glossary.md#term-planning); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine))</span>
- Concrete backends may create these values later, but the shared identity types contain no
  backend-specific implementation behavior.
- Stop if implementation needs registration, discovery, another module/type, a dependency edit,
  an architecture change, or a decision about availability or capabilities.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package retained and documented:

- `io.github.pho001.synaptik.backend.contract` — minimal backend-neutral identity and later
  declarative contract vocabulary

Types added:

- `io.github.pho001.synaptik.backend.contract.BackendId` — [backend ownership](../../../../glossary.md#backend-ownership) [identity](../../../../glossary.md#term-identity-meanings)
- `io.github.pho001.synaptik.backend.contract.BackendDeviceId` — [backend](../../../../glossary.md#backend)-scoped [device](../../../../glossary.md#term-device) identity

Temporary type removed:

- `io.github.pho001.synaptik.backend.contract.BackendContractModule`

The focused test mirrors the production package. No subpackage or helper type is introduced.

## Affected files

Production — exactly four paths:

- delete `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/BackendContractModule.java`
- add `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/package-info.java`
- add `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/BackendId.java`
- add `modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/BackendDeviceId.java`

Tests — exactly one path:

- add `modules/backend-contract/src/test/java/io/github/pho001/synaptik/backend/contract/BackendIdentityTest.java`

Documentation and [planning](../../../../glossary.md#term-planning) — exactly seven paths:

- `docs/api/public-api.md`
- `docs/backend-guide/capability-provider.md`
- `docs/glossary.md`
- add and finalize this task
- `docs/planning/modules/backend-contract/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: `AGENTS.md`, `ARCHITECTURE.md`, focused architecture documents and
ADRs, completed [trace](../../../../glossary.md#trace) source/tests/tasks, other placeholder [modules](../../../../glossary.md#term-module), [Gradle](../../../../glossary.md#term-gradle), concrete [backends](../../../../glossary.md#backend),
architecture tests, backend [conformance](../../../../glossary.md#term-conformance), integration tests, and legacy evidence.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact twelve paths above. Stop if implementation requires another production type,
test, document, package, [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module), architecture change, build edit, or detailed follow-up
specification.

## Acceptance criteria

- The placeholder is gone and the package contains exactly the package documentation plus
  [`BackendId`](../../../../glossary.md#backend-identity--backendid) and [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid) as its public production types.
- `BackendId` has exactly one `String value` component and the specified null/blank validation,
  messages, preservation, accessor, and record-value behavior.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `BackendDeviceId` has exactly `BackendId backendId` and `String value` in that order and follows
  the exact component-order validation, messages, preservation, accessors, and record behavior.
- Equal [device](../../../../glossary.md#term-device) [tokens](../../../../glossary.md#term-token) under unequal [backend](../../../../glossary.md#backend) IDs produce unequal `BackendDeviceId` values; equal
  components produce equal values. No global device-token uniqueness is implied.
- Nonblank backend and device values are open vocabulary. No case, syntax, whitespace trimming,
  predefined constant, or numeric-ordinal policy is introduced.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Public-shape tests prevent extra fields, constructors, methods, interfaces, nested types, or
  component drift and confirm neither record implements Java serialization.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Production imports remain JDK-only; no other Synaptik module or producer object appears in the
  signatures or implementation.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [module glossary](../../../../glossary.md#term-module); [producer glossary](../../../../glossary.md#term-producer); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Javadocs explain identity scope, exact preservation, equality, validation, ownership meaning,
  and the absence of registration, discovery, availability, resource, and execution behavior.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Public API status marks only the two identity values current. Capability-provider and backend
  integration contracts remain clearly planned.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The glossary distinguishes backend ownership identity, backend-scoped device identity, concrete
  backend implementations, and trace-local backend/device correlations.<span data-glossary-navigation="true"> ([backend ownership glossary](../../../../glossary.md#backend-ownership); [concrete backend glossary](../../../../glossary.md#concrete-backend); [trace glossary](../../../../glossary.md#trace))</span>
- Backend-contract master plan and roadmap identify 0001 as Complete. Backend-contract 0002–0004
  and trace 0003+ remain Draft without detailed next-task specifications, and no stale Ready
  frontier remains.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- A separate clean-context documentation-focused pass finalizes all affected Javadocs and the
  seven documentation/planning paths after Java tests pass.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Exact twelve-path [scope](../../../../glossary.md#term-scope), Markdown, final newlines, trailing whitespace, and `git diff --check`
  pass; no architecture, dependency, Gradle, trace Java/test, concrete backend, or cross-module
  implementation path changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests while developing. After executable Java stabilizes, run exactly one final
[backend](../../../../glossary.md#backend)-contract [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:backend-contract:test
```

Record test and suite counts from XML reports. Do not run repository-wide tests: this is a small
[JDK](../../../../glossary.md#term-java-development-kit-jdk)-only [leaf](../../../../glossary.md#term-leaf)-module [identity](../../../../glossary.md#term-identity-meanings) task with no [dependency](../../../../glossary.md#term-dependency) or architecture-rule change. A later
backend-contract capability [checkpoint](../../../../glossary.md#term-checkpoint) owns repository-wide validation.

Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That pass independently inspects final
source/tests and completed [trace](../../../../glossary.md#trace) identity contracts, applies General style with [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc),
Backend guide, [Planning](../../../../glossary.md#term-planning), and Example profiles where relevant, finalizes the Javadocs and seven
documentation/planning paths, records reasoned no-change conclusions, and runs:

```bash
./gradlew :modules:backend-contract:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun. Inspect generated package and record pages; rely on focused
automated API-[shape](../../../../glossary.md#shape), validation-order, import, interface, and nested-type checks; confirm exactly
twelve task paths; and confirm backend-contract 0001 Complete/0002 Draft/no detailed 0002 task
before completion.

## Dependencies

- The selected model milestone and trace tasks 0001–0002 are Complete. Trace 0002 is a roadmap
  sequencing prerequisite only; backend-contract adds no trace or model dependency.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [trace glossary](../../../../glossary.md#trace))</span>
- The authoritative [backend](../../../../glossary.md#backend)-contract [ownership](../../../../glossary.md#term-ownership-meanings) and [dependency](../../../../glossary.md#term-dependency) boundaries are already stable.

## Follow-up tasks

- Task 0002 [device](../../../../glossary.md#term-device) classification remains Draft without a detailed specification.
- Later focused tasks add availability snapshots and declarative backend requirements after their
  dependencies are concrete.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Trace returns to backend/device correlation and typed backend payload/attribute work only after
  these producer-owned backend contracts make the required diagnostic facts concrete.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [producer glossary](../../../../glossary.md#term-producer))</span>

## Architecture impact

Expected impact: None.

The task implements the existing `modules/backend-contract` [ownership](../../../../glossary.md#term-ownership-meanings) rule and changes no [module](../../../../glossary.md#term-module)
direction, [lifecycle](../../../../glossary.md#lifecycle) decision, registration mechanism, or [backend](../../../../glossary.md#backend) behavior. Focused documentation
changes implementation-status wording only.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, focused module/dependency/lifecycle/partition/prepare-backend
architecture documents, documentation/planning rules and profiles, roadmap, backend-contract and
trace master plans, completed trace tasks 0001–0002, backend-contract task 0001, the current
backend-contract placeholder/build, Public API, capability-provider guide, glossary, and Java 26
root configuration.

Implement docs/planning/modules/backend-contract/tasks/0001-backend-and-device-identifiers.md
exactly inside its twelve authorized paths. Replace only the backend-contract placeholder with
package documentation, BackendId, BackendDeviceId, and one focused test. Preserve exact open
string identity, validation order/messages, reference retention, backend-scoped device equality,
and no-registration/no-discovery/no-availability/no-service boundaries. Add no constants, device
class, snapshot, requirement, trace ID, dependency, or later task. Stop on architecture,
dependency, package, affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final :modules:backend-contract:test after
executable Java stabilizes. Then hand the actual diff and Java evidence to a separate clean-context
documentation agent in the same overall change. That pass finalizes Javadocs, Public API,
capability-provider guide, glossary, task/master/roadmap status, runs backend-contract Javadoc and
documentation/scope checks, and reuses successful Java evidence unless executable behavior
changes. Mark 0001 Complete only after both passes succeed. Leave 0002+ Draft without detailed
specifications and trace 0003+ Draft.
```

## Local decisions

- Use an open `String` backend value instead of an enum so third-party and later concrete backends
  do not require editing the shared contract.
- Reject null and blank values but otherwise retain exact text. Normalizing case, surrounding
  whitespace, Unicode, or aliases would create a naming policy without a current requirement.<span data-glossary-navigation="true"> ([Unicode glossary](../../../../glossary.md#term-unicode); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Make [device](../../../../glossary.md#term-device) [identity](../../../../glossary.md#term-identity-meanings) composite. [Backend](../../../../glossary.md#backend)-owned [tokens](../../../../glossary.md#term-token) such as `"0"` or `"default"` are meaningful
  only inside the backend that issued them and must not collide across backend domains.
- Store the exact [`BackendId`](../../../../glossary.md#backend-identity--backendid) reference instead of copying its string into [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid); this
  preserves nominal ownership and ordinary composite record semantics.<span data-glossary-navigation="true"> ([nominal glossary](../../../../glossary.md#term-nominal); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Add no allocator or well-known constants. Explicit engine composition and later concrete
  backends own creation and registration; identity DTOs own neither.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [engine glossary](../../../../glossary.md#term-engine))</span>
- Interleave backend-contract after the completed trace foundation because further trace schemas
  need real producer-owned backend vocabulary.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [schemas glossary](../../../../glossary.md#term-schema); [producer glossary](../../../../glossary.md#term-producer))</span>

## Known limitations

- The records do not prove that a backend is registered, installed, available, or able to execute
  any operation.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation))</span>
- A syntactically valid [device](../../../../glossary.md#term-device) [identity](../../../../glossary.md#term-identity-meanings) may name no currently available device.
- No canonical backend-name registry, case policy, alias policy, persistence format, or external
  compatibility guarantee exists yet.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [registry glossary](../../../../glossary.md#term-registry); [alias glossary](../../../../glossary.md#aliasing-and-arena); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Device class, capability, availability, requirements, discovery, configuration, and trace
  translation remain unavailable.<span data-glossary-navigation="true"> ([Device class glossary](../../../../glossary.md#device-class--deviceclass); [trace glossary](../../../../glossary.md#trace))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation context `/root/implement_backend_contract_0001` ran
  `./gradlew :modules:backend-contract:test --tests
  io.github.pho001.synaptik.backend.contract.BackendIdentityTest`; all six focused tests passed.
  After executable Java stabilized, that context ran exactly one final
  `./gradlew :modules:backend-contract:test`; [Gradle](../../../../glossary.md#term-gradle) reported `BUILD SUCCESSFUL` in 825 ms and XML
  `TEST-io.github.pho001.synaptik.backend.contract.BackendIdentityTest.xml` reports six tests in
  one suite with zero failures, errors, or skips.
- The separate clean documentation context
  `/root/implement_backend_contract_0001/backend_contract_0001_docs` applied General style with
  API/Javadoc as the primary profile, Backend guide for <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Backend glossary](../../../../glossary.md#backend))</span>`capability-provider.md`, Planning for the
  task/master plans/roadmap, and Example format for the current identity and conceptual capability
  examples. It inspected <span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`AGENTS.md`, the authoritative architecture contract, focused module,
  dependency, lifecycle, partition, runtime/prepare/backend and tracing explanations, relevant
  ADRs, documentation/planning rules, completed trace tasks and identities, the final
  backend-contract source/tests, Public API, capability guide, glossary, and Java 26 root/module
  build configuration.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [partition glossary](../../../../glossary.md#partition); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [trace glossary](../../../../glossary.md#trace))</span>
- The documentation context changed only Javadoc and Markdown after the successful final Java
  suite, so it did not rerun executable Java tests. `./gradlew
  :modules:backend-contract:javadoc` reported `BUILD SUCCESSFUL` in 1 s with two executed tasks.
  Generated output contains the package page and [`BackendId`](../../../../glossary.md#backend-identity--backendid) and [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid) pages with the
  rendered open-vocabulary, exact-reference, composite identity, validation-order/message,
  accessor, equality, ownership, and unsupported-behavior contracts.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed 216 Markdown files, 3,713 local links, 218
  local anchors, 2,714 fence markers, final newlines, and trailing whitespace. `git diff --check`
  passed.
- The final [scope](../../../../glossary.md#term-scope) command
  `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u` and
  `git status --short` confirmed exactly twelve task paths: four production paths, one focused
  test, and seven documentation/planning paths. The placeholder is the only deletion, and no
  unrelated working-tree path is present.
- Source and focused-test inspection confirms exactly one `String value` component for
  `BackendId`; exactly `BackendId backendId` then `String value` for `BackendDeviceId`; the sole
  public constructors; explicitly documented accessors; exact null/blank validation order and
  messages; exact component-reference retention; open names; backend-scoped equality; ordinary
  record hashing and diagnostic text; and no extra fields, methods, interfaces, nested types, or
  Java serialization. Production source has no import declaration and no Synaptik producer type.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [producer glossary](../../../../glossary.md#term-producer))</span>
- Status inspection confirms backend-contract task 0001 and its master-plan row are Complete;
  backend-contract 0002–0004 and trace 0003–0008 remain Draft; no row is Ready; and neither tasks
  directory contains a detailed next-task specification.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- No changes were needed in `ARCHITECTURE.md`, focused architecture explanations, ADRs, or
  architecture tests because this task implements the existing backend-contract identity leaf and
  changes no ownership or dependency rule. Gradle, dependencies, and the Java 26 baseline remain
  unchanged because both records are JDK-only. Other modules, trace Java/tests, concrete backend
  Java, backend-conformance tests, and integration tests remain unchanged because identity values
  add no producer, backend, trace-emission, capability, preparation, execution, or end-to-end
  behavior. Repository-wide Java validation remains deferred to the recorded backend-contract
  capability checkpoint or CI, as specified for this single-module leaf task.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [concrete backend glossary](../../../../glossary.md#concrete-backend); [conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Implementation notes

- Deleted the placeholder and added package documentation plus two independent public identity
  records. Their canonical constructors preserve the planned component order, exact failure
  messages, and exact references without introducing a helper, constant, registry, or service.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [canonical glossary](../../../../glossary.md#term-canonical-form); [constant glossary](../../../../glossary.md#term-constant); [registry glossary](../../../../glossary.md#term-registry))</span>
- Added one focused test class that locks the complete public record shape, validation contract,
  preservation, equality scoping, and forbidden interfaces/nested types.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- The documentation pass finalized the package/type/member Javadocs, added current identity
  guidance and glossary definitions, and synchronized backend-contract/trace/roadmap status. It
  changed no executable Java behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [trace glossary](../../../../glossary.md#trace))</span>

## Completion summary

- Completed changes: replaced the [backend](../../../../glossary.md#backend)-contract placeholder with package documentation,
  [`BackendId`](../../../../glossary.md#backend-identity--backendid), [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), and one focused exact-contract test; finalized all affected
  explanatory and planning documentation.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly the four production, one test, and seven
  documentation/planning paths listed under Affected files; no other path changed.
- Tests and validation: the implementation context passed the focused six-test command and one
  final six-test/one-suite backend-contract run; the documentation context passed final Javadoc,
  generated-page inspection, repository Markdown, exact-scope, status, and whitespace checks
  without rerunning Java tests.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context
  `/root/implement_backend_contract_0001/backend_contract_0001_docs` independently reviewed and
  finalized every affected Javadoc plus Public API, capability-provider, glossary, task, both
  master plans, and roadmap content.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation impact: only backend and backend-scoped device identities are current;
  availability, requirements, capabilities, registration, concrete backends, and trace-local
  backend/device correlations remain explicitly planned.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [trace glossary](../../../../glossary.md#trace))</span>
- Javadoc review: package and both public records document purpose, exact reference retention,
  open string vocabulary, composite/backend-scoped equality, inputs, results, failure order and
  messages, ownership meaning, and unsupported lifecycle behavior.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Glossary impact: added current backend identity and backend device identity definitions and
  distinguished them from concrete backends and planned trace-local correlations.<span data-glossary-navigation="true"> ([backend identity glossary](../../../../glossary.md#backend-identity--backendid); [backend device identity glossary](../../../../glossary.md#backend-device-identity--backenddeviceid))</span>
- Unresolved issues: None.
- Follow-up required: None. Backend-contract tasks 0002–0004 and trace tasks 0003–0008 remain
  Draft without detailed next-task specifications.

Status: Complete
