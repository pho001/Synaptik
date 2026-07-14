# Task 0001: Backend and Device Identifiers

## Status

Complete

## Goal

Replace the `modules/backend-contract` placeholder with the two minimal public identity values
required to name a backend owner and one device within that backend's identity domain.

Mental model:

```text
BackendId("cpu")
  └── BackendDeviceId(backendId = "cpu", value = "default")

BackendId("cuda")
  └── BackendDeviceId(backendId = "cuda", value = "0")
```

`BackendId` names the backend that may own planned work. `BackendDeviceId` combines that backend
identity with an opaque backend-defined device token, so a token such as `"0"` is never interpreted
without its owning backend. This task defines immutable vocabulary only. It performs no discovery,
registration, availability query, capability evaluation, ownership selection, or device access.

## Scope

- Delete the temporary `BackendContractModule` marker.
- Add package documentation for `io.github.pho001.synaptik.backend.contract` that distinguishes
  declarative shared contracts from concrete backend implementations and live services.
- Add public record `BackendId` with exactly one `String value` component.
  - The value names a backend ownership domain such as `"cpu"`, `"metal"`, or `"cuda"` without
    reserving those examples as constants or a closed set.
  - Reject `null` with `NullPointerException` and exact message `value`.
  - Reject `String.isBlank()` values with `IllegalArgumentException` and exact message
    `value must not be blank`.
  - Accept every other `String`, retain the exact supplied reference and content, and perform no
    trimming, case folding, Unicode normalization, syntax validation, interning, or aliasing.
  - Expose an explicitly documented `value()` accessor.
- Add public record `BackendDeviceId` with exactly `BackendId backendId` and `String value`, in
  that order.
  - The first component establishes the device-token namespace and prevents cross-backend token
    collisions.
  - Validate `backendId`, then `value`, in component order.
  - Reject a null backend identity with `NullPointerException` and exact message `backendId`.
  - Reject a null device value with `NullPointerException` and exact message `value`.
  - Reject `String.isBlank()` device values with `IllegalArgumentException` and exact message
    `value must not be blank`.
  - Accept every other device value, retain both exact references, and perform no normalization.
  - Expose explicitly documented `backendId()` and `value()` accessors.
- Both records must:
  - remain final through record semantics and add no other component, field, constructor, public
    method, nested type, or implemented interface;
  - preserve ordinary record equality, hashing, and diagnostic `toString()` behavior;
  - define identity by their exact stored components without object identity or hidden state; and
  - add no serialization, allocation, registry, discovery, availability, or lifecycle behavior.
- Add one focused test class covering exact package/API shape, component order and types,
  validation order and messages, exact reference/content retention, open backend names,
  backend-scoped device equality, ordinary record behavior, and forbidden interfaces/nested types.
- Finalize Javadocs and update Public API status, the capability-provider guide, glossary,
  backend-contract master plan, trace interleave record, and roadmap in the same overall change.

## Out of scope

- `DeviceClass`, `BackendAvailabilitySnapshot`, `BackendRequirement`, capability-provider,
  prepare, executable, storage, buffer, memory, route, kernel, or service contracts
- predefined CPU, Metal, CUDA, accelerator, host, default-device, or route constants
- enums or a closed backend/device vocabulary
- numeric device ordinals, device handles, native IDs, addresses, UUID allocation, parsing,
  formatting, comparison ordering, aliases, or canonicalization
- backend registration, lookup, discovery, `ServiceLoader`, classpath scanning, a service locator,
  singleton, registry, cache, allocator, global counter, environment probing, or native calls
- availability, capability support, ownership selection, scoring, partitioning, preparation,
  execution, runtime residency, transfer, publication, or tracing behavior
- `TraceBackendId`, `TraceDeviceId`, a trace mapping API, or changes to `modules/trace` Java/tests
- dependencies, Gradle, Java version, architecture contract, ADR, architecture tests, another
  module, backend conformance, integration tests, or concrete backend implementations
- a detailed task 0002 or later backend-contract specification
- implementing trace task 0003 or any later trace payload/attribute task
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle explanation](../../../../architecture/lifecycle.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
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
- [Trace master plan](../../trace/master-plan.md)
- [Public API status](../../../../api/public-api.md)
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/backend-contract` owns minimal backend identities and declarative requirements.
- Compile-time ownership refers to `BackendId`, never a concrete backend object or live service.
- `BackendDeviceId` is descriptive identity only. It must not expose a device handle, imply
  availability, or make a resource usable.
- The module remains JDK-only and independent of model, trace, config, planning, compiler,
  runtime, prepare, engine, and concrete backend modules.
- Concrete backends may create these values later, but the shared identity types contain no
  backend-specific implementation behavior.
- Stop if implementation needs registration, discovery, another module/type, a dependency edit,
  an architecture change, or a decision about availability or capabilities.

## Package impact

Existing package retained and documented:

- `io.github.pho001.synaptik.backend.contract` — minimal backend-neutral identity and later
  declarative contract vocabulary

Types added:

- `io.github.pho001.synaptik.backend.contract.BackendId` — backend ownership identity
- `io.github.pho001.synaptik.backend.contract.BackendDeviceId` — backend-scoped device identity

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

Documentation and planning — exactly seven paths:

- `docs/api/public-api.md`
- `docs/backend-guide/capability-provider.md`
- `docs/glossary.md`
- add and finalize this task
- `docs/planning/modules/backend-contract/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: `AGENTS.md`, `ARCHITECTURE.md`, focused architecture documents and
ADRs, completed trace source/tests/tasks, other placeholder modules, Gradle, concrete backends,
architecture tests, backend conformance, integration tests, and legacy evidence.

## Maximum scope

At most the exact twelve paths above. Stop if implementation requires another production type,
test, document, package, dependency, module, architecture change, build edit, or detailed follow-up
specification.

## Acceptance criteria

- The placeholder is gone and the package contains exactly the package documentation plus
  `BackendId` and `BackendDeviceId` as its public production types.
- `BackendId` has exactly one `String value` component and the specified null/blank validation,
  messages, preservation, accessor, and record-value behavior.
- `BackendDeviceId` has exactly `BackendId backendId` and `String value` in that order and follows
  the exact component-order validation, messages, preservation, accessors, and record behavior.
- Equal device tokens under unequal backend IDs produce unequal `BackendDeviceId` values; equal
  components produce equal values. No global device-token uniqueness is implied.
- Nonblank backend and device values are open vocabulary. No case, syntax, whitespace trimming,
  predefined constant, or numeric-ordinal policy is introduced.
- Public-shape tests prevent extra fields, constructors, methods, interfaces, nested types, or
  component drift and confirm neither record implements Java serialization.
- Production imports remain JDK-only; no other Synaptik module or producer object appears in the
  signatures or implementation.
- Javadocs explain identity scope, exact preservation, equality, validation, ownership meaning,
  and the absence of registration, discovery, availability, resource, and execution behavior.
- Public API status marks only the two identity values current. Capability-provider and backend
  integration contracts remain clearly planned.
- The glossary distinguishes backend ownership identity, backend-scoped device identity, concrete
  backend implementations, and trace-local backend/device correlations.
- Backend-contract master plan and roadmap identify 0001 as Complete. Backend-contract 0002–0004
  and trace 0003+ remain Draft without detailed next-task specifications, and no stale Ready
  frontier remains.
- A separate clean-context documentation-focused pass finalizes all affected Javadocs and the
  seven documentation/planning paths after Java tests pass.
- Exact twelve-path scope, Markdown, final newlines, trailing whitespace, and `git diff --check`
  pass; no architecture, dependency, Gradle, trace Java/test, concrete backend, or cross-module
  implementation path changes.

## Tests / validation

Run focused tests while developing. After executable Java stabilizes, run exactly one final
backend-contract module suite:

```bash
./gradlew :modules:backend-contract:test
```

Record test and suite counts from XML reports. Do not run repository-wide tests: this is a small
JDK-only leaf-module identity task with no dependency or architecture-rule change. A later
backend-contract capability checkpoint owns repository-wide validation.

Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That pass independently inspects final
source/tests and completed trace identity contracts, applies General style with API/Javadoc,
Backend guide, Planning, and Example profiles where relevant, finalizes the Javadocs and seven
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
automated API-shape, validation-order, import, interface, and nested-type checks; confirm exactly
twelve task paths; and confirm backend-contract 0001 Complete/0002 Draft/no detailed 0002 task
before completion.

## Dependencies

- The selected model milestone and trace tasks 0001–0002 are Complete. Trace 0002 is a roadmap
  sequencing prerequisite only; backend-contract adds no trace or model dependency.
- The authoritative backend-contract ownership and dependency boundaries are already stable.

## Follow-up tasks

- Task 0002 device classification remains Draft without a detailed specification.
- Later focused tasks add availability snapshots and declarative backend requirements after their
  dependencies are concrete.
- Trace returns to backend/device correlation and typed backend payload/attribute work only after
  these producer-owned backend contracts make the required diagnostic facts concrete.

## Architecture impact

Expected impact: None.

The task implements the existing `modules/backend-contract` ownership rule and changes no module
direction, lifecycle decision, registration mechanism, or backend behavior. Focused documentation
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
  whitespace, Unicode, or aliases would create a naming policy without a current requirement.
- Make device identity composite. Backend-owned tokens such as `"0"` or `"default"` are meaningful
  only inside the backend that issued them and must not collide across backend domains.
- Store the exact `BackendId` reference instead of copying its string into `BackendDeviceId`; this
  preserves nominal ownership and ordinary composite record semantics.
- Add no allocator or well-known constants. Explicit engine composition and later concrete
  backends own creation and registration; identity DTOs own neither.
- Interleave backend-contract after the completed trace foundation because further trace schemas
  need real producer-owned backend vocabulary.

## Known limitations

- The records do not prove that a backend is registered, installed, available, or able to execute
  any operation.
- A syntactically valid device identity may name no currently available device.
- No canonical backend-name registry, case policy, alias policy, persistence format, or external
  compatibility guarantee exists yet.
- Device class, capability, availability, requirements, discovery, configuration, and trace
  translation remain unavailable.

## Validation evidence

- The implementation context `/root/implement_backend_contract_0001` ran
  `./gradlew :modules:backend-contract:test --tests
  io.github.pho001.synaptik.backend.contract.BackendIdentityTest`; all six focused tests passed.
  After executable Java stabilized, that context ran exactly one final
  `./gradlew :modules:backend-contract:test`; Gradle reported `BUILD SUCCESSFUL` in 825 ms and XML
  `TEST-io.github.pho001.synaptik.backend.contract.BackendIdentityTest.xml` reports six tests in
  one suite with zero failures, errors, or skips.
- The separate clean documentation context
  `/root/implement_backend_contract_0001/backend_contract_0001_docs` applied General style with
  API/Javadoc as the primary profile, Backend guide for `capability-provider.md`, Planning for the
  task/master plans/roadmap, and Example format for the current identity and conceptual capability
  examples. It inspected `AGENTS.md`, the authoritative architecture contract, focused module,
  dependency, lifecycle, partition, runtime/prepare/backend and tracing explanations, relevant
  ADRs, documentation/planning rules, completed trace tasks and identities, the final
  backend-contract source/tests, Public API, capability guide, glossary, and Java 26 root/module
  build configuration.
- The documentation context changed only Javadoc and Markdown after the successful final Java
  suite, so it did not rerun executable Java tests. `./gradlew
  :modules:backend-contract:javadoc` reported `BUILD SUCCESSFUL` in 1 s with two executed tasks.
  Generated output contains the package page and `BackendId` and `BackendDeviceId` pages with the
  rendered open-vocabulary, exact-reference, composite identity, validation-order/message,
  accessor, equality, ownership, and unsupported-behavior contracts.
- `python3 /tmp/validate_synaptik_markdown.py` passed 216 Markdown files, 3,713 local links, 218
  local anchors, 2,714 fence markers, final newlines, and trailing whitespace. `git diff --check`
  passed.
- The final scope command
  `{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u` and
  `git status --short` confirmed exactly twelve task paths: four production paths, one focused
  test, and seven documentation/planning paths. The placeholder is the only deletion, and no
  unrelated working-tree path is present.
- Source and focused-test inspection confirms exactly one `String value` component for
  `BackendId`; exactly `BackendId backendId` then `String value` for `BackendDeviceId`; the sole
  public constructors; explicitly documented accessors; exact null/blank validation order and
  messages; exact component-reference retention; open names; backend-scoped equality; ordinary
  record hashing and diagnostic text; and no extra fields, methods, interfaces, nested types, or
  Java serialization. Production source has no import declaration and no Synaptik producer type.
- Status inspection confirms backend-contract task 0001 and its master-plan row are Complete;
  backend-contract 0002–0004 and trace 0003–0008 remain Draft; no row is Ready; and neither tasks
  directory contains a detailed next-task specification.
- No changes were needed in `ARCHITECTURE.md`, focused architecture explanations, ADRs, or
  architecture tests because this task implements the existing backend-contract identity leaf and
  changes no ownership or dependency rule. Gradle, dependencies, and the Java 26 baseline remain
  unchanged because both records are JDK-only. Other modules, trace Java/tests, concrete backend
  Java, backend-conformance tests, and integration tests remain unchanged because identity values
  add no producer, backend, trace-emission, capability, preparation, execution, or end-to-end
  behavior. Repository-wide Java validation remains deferred to the recorded backend-contract
  capability checkpoint or CI, as specified for this single-module leaf task.

## Implementation notes

- Deleted the placeholder and added package documentation plus two independent public identity
  records. Their canonical constructors preserve the planned component order, exact failure
  messages, and exact references without introducing a helper, constant, registry, or service.
- Added one focused test class that locks the complete public record shape, validation contract,
  preservation, equality scoping, and forbidden interfaces/nested types.
- The documentation pass finalized the package/type/member Javadocs, added current identity
  guidance and glossary definitions, and synchronized backend-contract/trace/roadmap status. It
  changed no executable Java behavior.

## Completion summary

- Completed changes: replaced the backend-contract placeholder with package documentation,
  `BackendId`, `BackendDeviceId`, and one focused exact-contract test; finalized all affected
  explanatory and planning documentation.
- Files changed or created: exactly the four production, one test, and seven
  documentation/planning paths listed under Affected files; no other path changed.
- Tests and validation: the implementation context passed the focused six-test command and one
  final six-test/one-suite backend-contract run; the documentation context passed final Javadoc,
  generated-page inspection, repository Markdown, exact-scope, status, and whitespace checks
  without rerunning Java tests.
- Documentation-agent review: clean context
  `/root/implement_backend_contract_0001/backend_contract_0001_docs` independently reviewed and
  finalized every affected Javadoc plus Public API, capability-provider, glossary, task, both
  master plans, and roadmap content.
- Documentation impact: only backend and backend-scoped device identities are current;
  availability, requirements, capabilities, registration, concrete backends, and trace-local
  backend/device correlations remain explicitly planned.
- Javadoc review: package and both public records document purpose, exact reference retention,
  open string vocabulary, composite/backend-scoped equality, inputs, results, failure order and
  messages, ownership meaning, and unsupported lifecycle behavior.
- Glossary impact: added current backend identity and backend device identity definitions and
  distinguished them from concrete backends and planned trace-local correlations.
- Unresolved issues: None.
- Follow-up required: None. Backend-contract tasks 0002–0004 and trace tasks 0003–0008 remain
  Draft without detailed next-task specifications.

Status: Complete
