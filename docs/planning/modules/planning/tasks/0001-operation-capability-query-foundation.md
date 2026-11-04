# Task 0001: [Operation](../../../../glossary.md#operation) Capability-Query Foundation

## Status

Complete

## Goal

Open `modules/planning` with the smallest [immutable](../../../../glossary.md#term-immutable) question that [planning](../../../../glossary.md#term-planning) can ask a named [backend](../../../../glossary.md#backend)
about one [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence) before configuration exposes [partition](../../../../glossary.md#partition)-scoring [weights](../../../../glossary.md#term-weight) or policy.

[Mental model](../../../../glossary.md#term-mental-model):

```text
Operation + ordered input descriptors + ordered output descriptors
  -> OperationCapabilityQuery
  -> BackendCapabilityProvider.supports(query)
  -> semantic ownership eligibility for that provider's BackendId
```

The query describes one structurally valid occurrence. The provider returns only a deterministic
[boolean](../../../../glossary.md#data-type--datatype) capability answer. Neither contract discovers a backend, evaluates availability or hard
requirements, scores candidates, selects a route, or creates executable state.

## [Scope](../../../../glossary.md#term-scope)

- Delete the [planning](../../../../glossary.md#term-planning) placeholder and add public record [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) in
  `io.github.pho001.synaptik.planning.capability` with exactly these ordered components:
  `Operation operation`, `List<TensorDescriptor> inputs`, and
  `List<TensorDescriptor> outputs`.
- Implement the [canonical](../../../../glossary.md#term-canonical-form)-constructor [validation](../../../../glossary.md#term-validation) and [snapshot](../../../../glossary.md#term-snapshot) sequence exactly:
  1. reject a null [`operation`](../../../../glossary.md#operation) with `NullPointerException` message `operation`;
  2. reject a null `inputs` list with `NullPointerException` message `inputs`;
  3. reject a null `outputs` list with `NullPointerException` message `outputs`;
  4. [scan](../../../../glossary.md#term-cumulative-scan) input [elements](../../../../glossary.md#term-element) in encounter order and reject the first null with
     `NullPointerException` message `inputs[index]`;
  5. snapshot the input list with `List.copyOf`;
  6. scan output elements in encounter order and reject the first null with
     `NullPointerException` message `outputs[index]`;
  7. snapshot the output list with `List.copyOf`; and
  8. finally call
     `operation.signature().validateOccurrence(inputs.size(), outputs.size())` against the
     snapshots.
- Retain the exact `Operation` reference and the exact `TensorDescriptor` element references in
  ordered immutable list snapshots. Preserve ordinary record equality, hashing, and diagnostic
  `toString()` semantics.
- Accept empty or repeated inputs when the operation signature permits them, repeated output
  descriptor references, and valid multi-output occurrences. The existing signature contract
  requires every valid occurrence to have a positive output count.<span data-glossary-navigation="true"> ([operation signature glossary](../../../../glossary.md#operation-signature); [descriptor glossary](../../../../glossary.md#term-descriptor); [signature glossary](../../../../glossary.md#term-signature); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Add public interface [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider) in the same package with exactly these two
  abstract public methods:
  - `BackendId backendId()` returns a stable, non-null [backend identity](../../../../glossary.md#backend-identity--backendid); and
  - `boolean supports(OperationCapabilityQuery query)` answers only whether that named backend can
    semantically own the immutable operation occurrence.
- Specify that every provider implementation rejects a null query with `NullPointerException`
  message `query`. For an [immutable](../../../../glossary.md#term-immutable) query and immutable provider configuration, `supports` is
  deterministic.
- Treat the provider as an explicitly supplied compile-time collaboration. It is not a service
  locator and performs no registry or discovery lookup.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [service locator glossary](../../../../glossary.md#term-service-locator); [registry glossary](../../../../glossary.md#term-registry); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Add no production provider implementation. Concrete backends may later implement this
  inward-facing planning contract through their architecture-approved dependency on planning.
- Add package Javadoc defining capability as semantic compile-time ownership eligibility and
  distinguishing it from availability, requirement evaluation, scoring, preparation, routing,
  and execution.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Require meaningful type, canonical-constructor, method, record-component, and package Javadocs.
  All record-component accessors must be explicitly declared and documented; the interface
  methods must fully document stability, nullability, determinism, and their narrow result
  semantics.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability))</span>
- Change only [model](../../../../glossary.md#term-model) and [backend](../../../../glossary.md#backend)-contract [dependency](../../../../glossary.md#term-dependency) visibility in
  `modules/planning/build.gradle.kts` from `implementation` to `api`, because public planning
  signatures expose `Operation`, `TensorDescriptor`, and [`BackendId`](../../../../glossary.md#backend-identity--backendid). Keep config and [trace](../../../../glossary.md#trace) as
  `implementation` dependencies.
- Add focused contract tests for exact API shape, exact validation order and messages, immutable
  ordered snapshots, reference retention, value semantics, accepted cardinalities, signature
  failure propagation, provider identity/support semantics, and forbidden added surfaces.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add one focused [architecture test](../../../../glossary.md#term-architecture-test) for the exact four planning project dependencies, exact
  `api` visibility of model and backend-contract, exact `implementation` visibility of config and
  trace, and absence of concrete-backend, runtime, prepare, and engine edges.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine))</span>
- Finalize the affected API, backend, user, glossary, architecture-status, and planning documents
  in the same overall change.

## Out of [scope](../../../../glossary.md#term-scope)

- capability matrix, candidate collection, candidate stored type, hard-eligibility result, or
  ownership candidate model<span data-glossary-navigation="true"> ([capability matrix glossary](../../../../glossary.md#review-tables-and-support-matrices))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [model glossary](../../../../glossary.md#term-model))</span>
- result enum or record, rejection reason, structured rejection diagnostics, exception taxonomy,
  or explanation payload<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- [device](../../../../glossary.md#term-device)-level capability query or [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid) selection
- [`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot) [evaluation](../../../../glossary.md#term-evaluation), [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) evaluation, or
  [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement) matching
- preference, ranking, scoring, profiles, weights, bonuses, penalties, estimates, or calibrated
  data<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight))</span>
- node or segment ownership, partition construction, same-owner grouping, transfer planning,
  materialization planning, or logical memory planning<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition); [owner glossary](../../../../glossary.md#term-owner); [planning glossary](../../../../glossary.md#term-planning); [materialization glossary](../../../../glossary.md#materialization))</span>
- provider [registry](../../../../glossary.md#term-registry), discovery, [classpath](../../../../glossary.md#term-classpath) scanning, reflection, `ServiceLoader`, service locator,
  factory, or engine registration<span data-glossary-navigation="true"> ([service locator glossary](../../../../glossary.md#term-service-locator); [engine glossary](../../../../glossary.md#term-engine))</span>
- a concrete backend, provider implementation, fallback provider, route, kernel, lowering, or
  backend conformance behavior<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [kernel glossary](../../../../glossary.md#kernel); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- compiler orchestration, graph closure, graph/node/value identifiers, graph phase, prepare,
  runtime, engine, execution, storage, physical buffers, schedules, or mutable state<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [graph phase glossary](../../../../glossary.md#graph-phase); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [buffers glossary](../../../../glossary.md#term-buffer); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- semantic data-type, shape, layout, gradient, operand-role, descriptor-compatibility, or graph-
  closure validation beyond the existing occurrence-count signature check<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [occurrence glossary](../../../../glossary.md#term-occurrence); [signature glossary](../../../../glossary.md#term-signature))</span>
- [trace](../../../../glossary.md#trace) DTOs, attributes, payloads, emission, or [serialization](../../../../glossary.md#term-serialization)
- serialization or external [schema](../../../../glossary.md#term-schema) for either new contract
- dependencies or build changes beyond the two visibility changes in the planning build and their
  focused architecture test<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- architecture-rule changes, ADR changes, Java-version changes, root build changes, another
  module's executable behavior, unrelated refactoring, or a detailed specification for any later
  task<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Backend guide style](../../../../developer-guide/documentation/backend-guide-style.md)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Planning master plan](../master-plan.md)
- [Config master plan](../../config/master-plan.md)
- [Trace master plan](../../trace/master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Planning owns backend-neutral capability query contracts and asks where an operation occurrence
  may run. It must not select a concrete kernel, executable, BLAS route, MPSGraph route, CUDA
  implementation, or other backend route.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [run glossary](../../../../glossary.md#term-run); [kernel glossary](../../../../glossary.md#kernel); [BLAS glossary](../../../../glossary.md#term-blas); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [CUDA glossary](../../../../glossary.md#term-cuda); [backend route glossary](../../../../glossary.md#backend-route))</span>
- [`Operation`](../../../../glossary.md#operation) continues to own backend-independent semantics and exposes no backend support.
  Backend capability is expressed through the planning-owned provider contract.<span data-glossary-navigation="true"> ([Backend capability glossary](../../../../glossary.md#backend-capability))</span>
- [Compile](../../../../glossary.md#compile)-time plans retain [`BackendId`](../../../../glossary.md#backend-identity--backendid), not provider objects or live backend services. This task
  creates no plan or retained provider association.
- A concrete backend may later depend inward on planning and implement the provider interface.
  Planning must not depend outward on a concrete backend.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- Capability is distinct from supplied availability, hard-requirement eligibility, candidate
  scoring, registration, preparation, and execution.
- Public [planning](../../../../glossary.md#term-planning) [signatures](../../../../glossary.md#term-signature) exposing [model](../../../../glossary.md#term-model) and [backend](../../../../glossary.md#backend)-contract types require `api` visibility.
  Existing config and trace edges remain internal implementation dependencies.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>
- Planning must remain independent of runtime, prepare, engine, and concrete backends. The focused
  architecture test enforces the current exact dependency surface without changing the
  architecture contract.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [architecture test glossary](../../../../glossary.md#term-architecture-test); [dependency glossary](../../../../glossary.md#term-dependency); [architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- Stop if implementation needs a capability result model, device identity, registry, consumer,
  concrete implementation, extra dependency, additional production type, architecture rule, or
  public API beyond the exact two contracts and package documentation.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [identity glossary](../../../../glossary.md#term-identity-meanings); [registry glossary](../../../../glossary.md#term-registry); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Package impact

Package added:

- `io.github.pho001.synaptik.planning.capability` — public backend-neutral compile-time contracts
  for asking whether one named backend can semantically own one immutable operation occurrence<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [immutable glossary](../../../../glossary.md#term-immutable); [operation glossary](../../../../glossary.md#operation); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>

Type placement:

- `io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery` — planning owns the
  immutable operation-occurrence question used during capability analysis<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- `io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider` — planning owns the
  inward-facing collaboration that concrete backends may later implement

The root placeholder `io.github.pho001.synaptik.planning.PlanningModule` is deleted. No generic
root [facade](../../../../glossary.md#term-facade), `util`, candidate, [matrix](../../../../glossary.md#term-matrix-meanings), scoring, [registry](../../../../glossary.md#term-registry), service, discovery, provider-
implementation, [runtime](../../../../glossary.md#term-runtime), or [backend](../../../../glossary.md#backend)-specific package is added.

## Affected files

The intended implementation [scope](../../../../glossary.md#term-scope) is exactly eighteen paths.

Production and build — exactly five paths:

- delete
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/PlanningModule.java`
- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/OperationCapabilityQuery.java`
- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/BackendCapabilityProvider.java`
- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/package-info.java`
- update `modules/planning/build.gradle.kts`

Tests — exactly two paths:

- add
  `modules/planning/src/test/java/io/github/pho001/synaptik/planning/capability/BackendCapabilityContractTest.java`
- add
  `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/PlanningDependencyContractTest.java`

Architecture-status and explanatory documentation — exactly six paths:

- current-versus-planned wording only in `docs/architecture/partition-scoring.md`, without changing
  an architecture rule
- `docs/api/public-api.md`
- `docs/api/compile-api.md`
- `docs/backend-guide/capability-provider.md`
- `docs/user-guide/backend-selection.md`
- `docs/glossary.md`

[Planning](../../../../glossary.md#term-planning) — exactly five paths:

- add and finalize this task
- `docs/planning/modules/planning/master-plan.md`
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: [`Operation`](../../../../glossary.md#operation), `OperationSignature`, `TensorDescriptor`, [`BackendId`](../../../../glossary.md#backend-identity--backendid),
their tests and Javadocs, current config and [trace](../../../../glossary.md#trace) Java contracts, [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[prepare](../../../../glossary.md#prepare)/[engine](../../../../glossary.md#term-engine)
placeholders, all concrete [backends](../../../../glossary.md#backend), [`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot), [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), backend
requirements, `AGENTS.md`, `ARCHITECTURE.md`, other architecture explanations, ADRs, root build
and settings, other [module](../../../../glossary.md#term-module) build files, backend-[conformance](../../../../glossary.md#term-conformance) tests, integration tests, and all
later task rows.

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly the eighteen paths listed above. Stop if implementation requires another production
type, test, document, [dependency](../../../../glossary.md#term-dependency), package, [module](../../../../glossary.md#term-module) [API](../../../../glossary.md#term-application-programming-interface-api), architecture change, provider
implementation, consumer behavior, or detailed follow-up specification. Do not use a later task
to hide an incomplete acceptance criterion from this task.

## Acceptance criteria

- [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) is a public record with exactly the ordered components
  `Operation operation`, `List<TensorDescriptor> inputs`, and `List<TensorDescriptor> outputs`, a
  public canonical constructor, explicitly documented component accessors, ordinary record object
  methods, and no other project field, method, constructor, interface, factory, builder, or nested
  type.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- The constructor performs the exact null, encounter-order element scan, snapshot, and final
  occurrence-count validation sequence specified in Scope. Every null failure has the exact
  required message, and an input failure precedes any output-element scan or signature call.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [scan glossary](../../../../glossary.md#term-cumulative-scan); [snapshot glossary](../../../../glossary.md#term-snapshot); [occurrence glossary](../../../../glossary.md#term-occurrence); [validation glossary](../../../../glossary.md#term-validation); [Scope glossary](../../../../glossary.md#term-scope); [signature glossary](../../../../glossary.md#term-signature))</span>
- Construction retains the exact operation and descriptor references while isolating ordered list
  membership from later caller mutation. Returned lists are immutable snapshots.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Empty and repeated inputs are accepted when permitted by the signature. Repeated output
  descriptor references and signature-valid multi-output occurrences are accepted. Zero outputs
  fail through `OperationSignature.validateOccurrence` because valid signatures require positive
  output counts.
- The query performs no data-type, shape, layout, gradient, operand-role, graph, identifier,
  phase, backend, device, availability, config, estimate, capability, score, route, prepare,
  runtime, or execution validation.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [device glossary](../../../../glossary.md#term-device); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider) is public and declares exactly `backendId()` and `supports(query)` as
  abstract public methods. It has no fields, constants, default/static/private methods, nested
  types, factory, registry, discovery, lookup, route, kernel, prepare, runtime, or execution
  surface.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [registry glossary](../../../../glossary.md#term-registry); [kernel glossary](../../../../glossary.md#kernel))</span>
- `backendId()` is documented and tested as stable and non-null. `supports` answers only semantic
  ownership capability for that provider's backend, rejects null with exact message <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`query` in
  implementations, and is deterministic for an immutable query and immutable provider
  configuration.
- `supports` does not report backend registration or availability, evaluate a hard requirement,
  score candidates, select a kernel or route, inspect prepare/runtime state, or promise successful
  execution.
- The provider is supplied explicitly to later compile-time planning; it is not obtained through
  a service locator, registry, discovery scan, or <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [service locator glossary](../../../../glossary.md#term-service-locator))</span>`ServiceLoader` [lookup](../../../../glossary.md#term-lookup).
- No production provider implementation is added. Documentation explains that concrete backends
  may later implement this inward-facing planning contract.
- The planning placeholder is removed and package Javadoc distinguishes the two current contracts
  from all deferred planning and backend behavior.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The [planning](../../../../glossary.md#term-planning) build declares [model](../../../../glossary.md#term-model) and [backend](../../../../glossary.md#backend)-contract with `api`, config and [trace](../../../../glossary.md#trace) with
  `implementation`, and no other project dependency. The focused architecture test locks exact
  visibility and forbids concrete-backend, runtime, prepare, and engine edges.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [architecture test glossary](../../../../glossary.md#term-architecture-test); [engine glossary](../../../../glossary.md#term-engine))</span>
- Existing model, backend-contract, config, and trace declarations and behavior remain unchanged.
- The six explanatory documents mark the query/provider contract current only after
  implementation while leaving matrices, eligibility, scoring, ownership, partitioning,
  provider implementations, compiler/prepare/runtime/engine behavior, diagnostics, and device-
  level capability planned.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-matrix-meanings); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Planning master plan records task 0001 as the sole Ready frontier before implementation and
  tasks 0002–0006 as ordered Draft rows without detailed specifications. Config tasks 0001–0002
  and trace tasks 0001–0002 remain Complete; their later rows remain Draft without detailed
  specifications.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- A separate clean-context documentation-focused pass finalizes every affected Javadoc and all
  eleven documentation/planning paths in the same overall change without repeating successful
  Java tests unless executable Java changes or a concrete cross-check risk is recorded.
- The final combined planning/architecture-test suite, planning Javadoc, repository Markdown,
  exact eighteen-path scope, synchronized statuses, absence of later detailed specs, final
  newlines, trailing whitespace, `git diff --check`, and single final root suite pass.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development commands are allowed while Java is changing. After executable Java and the
build visibility changes stabilize, [run](../../../../glossary.md#term-run) exactly one final combined affected-suite command:

```bash
./gradlew :modules:planning:test :testing:architecture-tests:test
```

Record suite and test counts from XML reports. Then hand the actual combined diff and exact Java
evidence to the separate documentation-focused context described below. That context reuses the
successful Java evidence and does not rerun either Java suite unless it changes executable Java
behavior or records a concrete reason. After its final [Javadoc](../../../../glossary.md#term-javadoc) and Markdown edits, it runs:

```bash
./gradlew :modules:planning:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

Inspect the generated package and both type pages. Confirm exactly the eighteen authorized paths,
the exact [dependency](../../../../glossary.md#term-dependency) declarations, one global [`Ready`](../../../../glossary.md#term-ready) task during implementation, no detailed
config 0003, [trace](../../../../glossary.md#trace) 0003, or [planning](../../../../glossary.md#term-planning) 0002 specification, synchronized current-versus-planned
wording, balanced fences, final newlines, and no trailing whitespace.

Because public dependency visibility and architecture-test coverage change, after the complete
implementation/documentation diff stabilizes run exactly one final repository suite:

```bash
./gradlew test
```

Record its task, suite, and test result. Do not repeat a successful root run in the documentation
or coordination context.

## Documentation [handoff](../../../../glossary.md#term-handoff)

After the implementation context records the final combined [planning](../../../../glossary.md#term-planning)/architecture-test result,
hand this task, the actual diff, and exact test evidence to a separate clean-context
documentation-focused agent or thread. It must read the [architecture contract](../../../../glossary.md#architecture-contract), documentation
rules, General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend) guide, User guide, Planning, and Example profiles, final
source/tests/build changes, and the directly affected documents.

That pass must independently finalize both new type Javadocs, explicit record accessors, package
Javadoc, and all affected explanatory and planning text. It must review glossary impact and record
reasoned no-change conclusions for [Compile](../../../../glossary.md#compile) API consumers beyond the documented surface, current
config/[trace](../../../../glossary.md#trace) Java, architecture rules and tests beyond the new focused [dependency](../../../../glossary.md#term-dependency) test,
backend-[conformance](../../../../glossary.md#term-conformance) and integration tests, [Gradle](../../../../glossary.md#term-gradle) outside planning, and all other [modules](../../../../glossary.md#term-module). It
reuses successful Java evidence unless it changes executable Java behavior.

## Dependencies

- Completed [model](../../../../glossary.md#term-model) capability milestone, including stable [`Operation`](../../../../glossary.md#operation), `OperationSignature`, and
  `TensorDescriptor` contracts.
- Completed [backend](../../../../glossary.md#backend)-contract tasks 0001–0004, including stable [`BackendId`](../../../../glossary.md#backend-identity--backendid).
- Completed [trace](../../../../glossary.md#trace) tasks 0001–0002 as the diagnostic foundation; this task adds no trace contract.
- Completed config tasks 0001–0002. Planning task 0001 intentionally interleaves before config
  task 0003 so the typed capability question is stable before scoring weights and policy are
  exposed.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [weights glossary](../../../../glossary.md#term-weight))</span>

## Follow-up tasks

- Planning task 0002 will define capability-matrix construction and hard eligibility after this
  contract is complete; it remains Draft without a detailed specification.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Planning task 0003 will define ownership candidates and backend-neutral scoring after its
  config prerequisites are stable; it remains Draft without a detailed specification.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend))</span>
- Planning tasks 0004–0006 retain same-owner partitioning, logical materialization/memory
  requirements, and planning closure as Draft rows without detailed specifications.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [materialization glossary](../../../../glossary.md#materialization))</span>
- After this task completes, the likely active frontier returns to config task 0003 for
  partition-scoring configuration before planning scoring proceeds.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
- Boolean rejection diagnostics and device-level capability queries are deliberately deferred
  limitations, not incomplete acceptance criteria for this foundation.<span data-glossary-navigation="true"> ([Boolean glossary](../../../../glossary.md#data-type--datatype); [device glossary](../../../../glossary.md#term-device))</span>

## Architecture impact

Expected impact: None.

The task realizes existing [planning](../../../../glossary.md#term-planning) capability [ownership](../../../../glossary.md#term-ownership-meanings) and existing concrete-[backend](../../../../glossary.md#backend) inward
[dependency](../../../../glossary.md#term-dependency) permission. Changing [model](../../../../glossary.md#term-model) and backend-contract to public dependency visibility is a
build/[API](../../../../glossary.md#term-application-programming-interface-api) realization of the existing direction, not a new architecture rule. The focused
[architecture test](../../../../glossary.md#term-architecture-test) protects that direction. If implementation reveals a need to change [module](../../../../glossary.md#term-module)
ownership, allowed dependencies, provider direction, [lifecycle](../../../../glossary.md#lifecycle) rules, or service-discovery
policy, stop before editing architecture documentation.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md, the directly relevant documentation profiles,
docs/planning/roadmap.md, the planning/config/trace master plans, and
docs/planning/modules/planning/tasks/0001-operation-capability-query-foundation.md in full.

Implement task 0001 exactly as specified, within its exact eighteen-path scope. Do not implement
out-of-scope or later-task work. Stop and report any architecture conflict or required scope
expansion instead of inventing a new contract.

After executable Java and the final combined affected-suite validation, hand the actual diff and
recorded evidence to a separate clean-context documentation-focused agent or thread as required
by the task. Do not repeat successful Java tests there unless executable behavior changes or a
concrete reason is recorded. Run the single final root suite only after the combined diff is
stable. Update this task with decisions, evidence, implementation notes, canonical completion
summary, and final status only when every acceptance criterion passes.
```

## Local decisions

- The first capability result is boolean because the next planning need is hard semantic
  inclusion/exclusion. Structured diagnostics are useful but would introduce a second result
  contract before its consumers and trace vocabulary are stable.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype); [planning glossary](../../../../glossary.md#term-planning); [trace glossary](../../../../glossary.md#trace))</span>
- The query represents an occurrence by immutable semantic and descriptor facts rather than node
  or value identity. Capability does not need graph identity, graph closure, or live graph state.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [immutable glossary](../../../../glossary.md#term-immutable); [descriptor glossary](../../../../glossary.md#term-descriptor); [node glossary](../../../../glossary.md#node); [identity glossary](../../../../glossary.md#term-identity-meanings); [graph glossary](../../../../glossary.md#graph))</span>
- Output descriptors are explicit and may repeat by reference because capability can depend on
  every logical output of a multi-output occurrence; the query does not infer outputs.
- Model and backend-contract are public dependencies because they appear in public signatures;
  config and trace remain internal dependencies until a public planning signature actually
  exposes them.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend); [signatures glossary](../../../../glossary.md#term-signature))</span>
- The config interleave pauses before scoring configuration only long enough to stabilize the
  typed capability question. It does not authorize planning scoring before config task 0003.

## Known limitations

- A boolean rejection carries no reason or structured diagnostic. A later task may add a separate
  result contract only when a concrete consumer and trace boundary are ready.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype); [trace glossary](../../../../glossary.md#trace); [ready glossary](../../../../glossary.md#term-ready))</span>
- Capability is [backend](../../../../glossary.md#backend)-level only. [Device](../../../../glossary.md#term-device)-specific support and device selection remain deferred.
- The provider contract does not prove availability, registration, requirement satisfaction,
  preparation success, route existence, resource access, or executable success.
- This task supplies no provider implementation, capability matrix, eligible-candidate model, or
  planning consumer.<span data-glossary-navigation="true"> ([capability matrix glossary](../../../../glossary.md#review-tables-and-support-matrices))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix); [model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/implement_planning_0001` ran
  `./gradlew :modules:planning:test :testing:architecture-tests:test` after executable Java and
  dependency visibility stabilized: <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>`BUILD SUCCESSFUL`; 13 tests across four suites, with 0
  failures, 0 errors, and 0 skips. `BackendCapabilityContractTest` contributed 10 tests;
  `ConfigDependencyContractTest`, `NnTrainingDependencyContractTest`, and
  `PlanningDependencyContractTest` contributed one test each.
- Documentation context `/root/implement_planning_0001/planning_0001_docs` independently read the
  architecture contract, documentation rules and selected Architecture, API/Javadoc, Backend
  guide, User guide, Planning, and Example profiles, planning guide and roadmap, three affected
  master plans, this task, final source/tests/build, directly affected documentation, and the
  relevant model/backend-contract contracts. It changed no executable Java behavior and reused
  the successful 13-test evidence without rerunning Java tests.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The documentation context ran `./gradlew :modules:planning:javadoc`: `BUILD SUCCESSFUL`; six
  actionable tasks, two executed and four up-to-date. It inspected generated
  `package-summary.html`, `OperationCapabilityQuery.html`, and
  `BackendCapabilityProvider.html`, including the package boundary, record signature, constructor
  failures, immutable-list accessors, stable provider identity, narrow boolean result, and null-
  query obligation.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- The documentation context ran `python3 /tmp/validate_synaptik_markdown.py`: passed for 222
  Markdown files, 3,885 local links, 235 local anchors, 2,800 fence markers, and final-newline/
  trailing-whitespace validation.
- The documentation context ran `git diff --check`: passed with no output. The combined tracked
  and untracked path audit returned exactly the eighteen authorized paths, and `git status
  --short` showed no path outside that set.
- Manual checks confirmed the exact [planning](../../../../glossary.md#term-planning) dependencies (`api` [model](../../../../glossary.md#term-model) and [backend](../../../../glossary.md#backend)-contract;
  `implementation` config and trace), task 0001 and the master plan at Complete, config 0003
  as the likely next Draft frontier, later planning/trace rows still Draft, no detailed config
  0003, trace 0003, or planning 0002 task specification, balanced fences, final newlines, and no
  trailing whitespace.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>
- Implementation coordinator `/root/implement_planning_0001` ran the single final
  `./gradlew test` from the repository root after the combined implementation/documentation diff
  stabilized and without any intervening executable change: `BUILD SUCCESSFUL in 628ms`; 46
  actionable tasks, one executed and 45 up-to-date. Aggregated XML reports contain 1,079 tests
  across 141 suites, with 0 skipped, 0 failures, and 0 errors. This was the only final root-suite
  run.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>

## Implementation notes

- [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) is the exact three-component public record. Its compact constructor
  validates top-level references, scans and snapshots inputs, scans and snapshots outputs, then
  delegates count validation to the retained operation signature. The exact operation and
  descriptor references are retained; only list membership is copied.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [operation signature glossary](../../../../glossary.md#operation-signature); [operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider) declares only `backendId()` and `supports(query)`. Its Javadoc makes
  stable non-null identity, deterministic semantic ownership support, exact null-query failure,
  explicit supply, and the absence of discovery, availability, eligibility, scoring, route,
  prepare, and execution meanings explicit. No provider implementation was added.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepare glossary](../../../../glossary.md#prepare))</span>
- The planning placeholder was deleted. Model and backend-contract became public Gradle API
  dependencies because their types occur in public signatures; config and trace remain internal
  implementation dependencies. The focused architecture test locks that exact surface and the
  forbidden outward edges.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend); [Gradle glossary](../../../../glossary.md#term-gradle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [signatures glossary](../../../../glossary.md#term-signature); [trace glossary](../../../../glossary.md#trace); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- The documentation pass marked only the query/provider contracts current. Capability matrices,
  hard eligibility, scoring, ownership, partitioning, provider implementations, compiler/prepare/
  runtime/engine behavior, diagnostics, and device-level capability remain planned.<span data-glossary-navigation="true"> ([matrices glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [device glossary](../../../../glossary.md#term-device))</span>
- No in-scope adjustment was required. The documentation pass removed one draft Javadoc phrase
  that prohibited retaining the immutable query because neither the interface nor tested contract
  establishes that additional ownership restriction.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Current config and trace Java required no change because neither consumes or exposes the new
  contracts. `ARCHITECTURE.md`, ADRs, and architecture explanations beyond the authorized
  current-status page required no change because the implementation realizes existing ownership
  and dependency rules. Architecture tests beyond the focused dependency test, backend-
  conformance tests, and integration tests required no change because no architecture rule,
  concrete backend behavior, preparation, or end-to-end behavior changed. Gradle outside planning
  and all other modules required no change because the only public-signature visibility adjustment
  is local to <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance); [concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>`modules/planning`. Compile API consumers beyond the documented current surface
  required no change because no compiler or planning consumer exists.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>

## Completion summary

- Completed changes: Implemented and documented the immutable operation-capability query and
  explicit provider collaboration, removed the placeholder, and exposed only the two public
  signature dependencies.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature))</span>
- Files changed or created: Exactly eighteen authorized paths: five production/build paths, two
  test paths, six explanatory documentation paths, and five planning paths.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: Reused the implementation context's successful 13-test/four-suite combined
  run; planning Javadoc, generated-page inspection, repository Markdown, exact-scope, dependency,
  status, later-spec absence, final-newline, trailing-whitespace, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [dependency glossary](../../../../glossary.md#term-dependency))</span>`git diff --check` validation
  passed in the documentation context. The implementation coordinator's single final root suite
  passed 1,079 tests across 141 suites with no skips, failures, or errors.
- Documentation-agent review: Clean context
  `/root/implement_planning_0001/planning_0001_docs` completed the required independent pass
  without changing executable behavior or repeating Java tests.
- Documentation impact: Finalized architecture-status, public/compile API, backend guide,
  backend-selection guide, glossary, task, three master plans, and roadmap wording while preserving
  current-versus-planned boundaries.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend))</span>
- Javadoc review: Finalized and generated the package and both public type pages; every parameter,
  non-void return, caller-visible failure, nullability, snapshot/reference-retention rule,
  deterministic result, and deliberate boundary is documented.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Glossary impact: Added current operation-capability query and provider definitions and refined
  backend capability without changing architecture authority.<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability))</span>
- Unresolved issues: None.
- Follow-up required: None. Config task 0003 is the likely next frontier but remains Draft without
  a detailed specification; making it Ready is a separate planning step.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

Status: Complete
