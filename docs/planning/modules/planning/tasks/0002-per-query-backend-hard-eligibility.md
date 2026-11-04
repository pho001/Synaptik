# Task 0002: Per-Query [Backend Hard Eligibility](../../../../glossary.md#backend-hard-eligibility)

> **Editorial historical reading key.**
>
> This milestone introduces an internal evaluator, not a public backend-selection workflow. Its caller
> supplies one query, BackendIntent, ordered providers, and matching immutable availability snapshots.
> Planning validates the complete association, filters by available-device membership and exact hard
> requirements, then calls each surviving provider once. Providers remain authoritative for
> backend-level semantic support; snapshots establish matching availability, not device-specific
> operation support. The result retains the query and exact provider-returned BackendIds, not
> providers, snapshots, selected devices, or storage. Evaluation acquires no closeable resources and
> transfers no cleanup duty.
>
> See the [current capability and eligibility
> contract](../../../../reference/compiler/planning.md#current-operation-capability-contracts) for the
> subsequent public BackendOwnerPlanning composition. The [current Compiler
> integration](../../../../reference/compiler/artifacts.md#current-complete-artifact-compilation-and-integration-port)
> distinguishes this internal empty-list fact from Compiler’s contextual terminal failure. Historical
> claims that no external consumer exists belong to this task’s completion point.

## Status

Complete

## Goal

Add the smallest [planning](../../../../glossary.md#term-planning)-owned [evaluation](../../../../glossary.md#term-evaluation) that combines one [immutable](../../../../glossary.md#term-immutable) [operation](../../../../glossary.md#operation)-capability
question, explicitly supplied [backend capability](../../../../glossary.md#backend-capability) providers, caller-supplied availability
snapshots, and one [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) into the ordered [backend](../../../../glossary.md#backend) [identities](../../../../glossary.md#term-identity-meanings) that remain hard-eligible
before any scoring.

[Mental model](../../../../glossary.md#term-mental-model):

```text
one OperationCapabilityQuery
  + ordered BackendCapabilityProvider values
  + one matching BackendAvailabilitySnapshot per provider
  + BackendIntent
  -> validate the complete supplied backend set
  -> require current device availability and exact hard-target matching
  -> ask each still-possible provider once for backend-level support
  -> immutable ordered BackendId list
```

The result is per operation [occurrence](../../../../glossary.md#term-occurrence) and retains only [backend ownership](../../../../glossary.md#backend-ownership) identities. It does not
retain providers, snapshots, requirements, devices, or a selected [device](../../../../glossary.md#term-device). It represents no match
with an empty list; later planning orchestration must stop before scoring that empty result and
must not weaken a hard requirement or invent a fallback.

## [Scope](../../../../glossary.md#term-scope)

- Add package-private record `BackendEligibility` in
  `io.github.pho001.synaptik.planning.capability` with exactly these ordered components:
  `OperationCapabilityQuery query` and `List<BackendId> eligibleBackendIds`.
- Keep the record package-private. No current compiler, planner facade, or other external consumer
  exists, and a public entry point that exposed <span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [facade glossary](../../../../glossary.md#term-facade))</span>[`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) would unnecessarily change the
  current config dependency from <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>`implementation` to `api`. Later ownership/scoring work must
  reassess the public facade from its concrete consumer rather than speculating here.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Give the record one package-private canonical constructor with this exact validation and
  snapshot sequence:<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation))</span>
  1. reject null `query` with `NullPointerException` message `query`;
  2. reject null `eligibleBackendIds` with `NullPointerException` message
     `eligibleBackendIds`;
  3. [scan](../../../../glossary.md#term-cumulative-scan) eligible [backend](../../../../glossary.md#backend) [identities](../../../../glossary.md#term-identity-meanings) in encounter order and reject the first null with
     `NullPointerException` message `eligibleBackendIds[index]`;
  4. reject the second [occurrence](../../../../glossary.md#term-occurrence) of an equal [backend identity](../../../../glossary.md#backend-identity--backendid) with
     `IllegalArgumentException` message
     `duplicate eligible backendId: <backendId.value()>`; and
  5. [snapshot](../../../../glossary.md#term-snapshot) list membership with `List.copyOf`, retaining the exact [`BackendId`](../../../../glossary.md#backend-identity--backendid) element
     references.
- Preserve ordinary record equality, hashing, and diagnostic `toString()` behavior over the
  exact query reference's value and ordered immutable eligible-identity snapshot.
- Explicitly declare and document the public `query()` and `eligibleBackendIds()` record component
  accessors. Java requires a record component accessor to be public; because the enclosing top-
  level record remains package-private, these methods do not create an externally accessible
  planning API. They return the exact retained query reference and immutable ordered list snapshot,
  respectively.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Add exactly one package-private static factory on the record:

  ```java
  static BackendEligibility evaluate(
          OperationCapabilityQuery query,
          BackendIntent intent,
          List<BackendCapabilityProvider> providers,
          List<BackendAvailabilitySnapshot> availabilitySnapshots)
  ```

- The factory validates top-level references in this exact order before reading any list [element](../../../../glossary.md#term-element):
  1. null `query` -> `NullPointerException("query")`;
  2. null `intent` -> `NullPointerException("intent")`;
  3. null `providers` -> `NullPointerException("providers")`; and
  4. null `availabilitySnapshots` ->
     `NullPointerException("availabilitySnapshots")`.
- Scan providers completely in caller encounter order before scanning any availability snapshot:
  1. reject a null provider at index `i` with `NullPointerException` message `providers[i]`;
  2. call that provider's `backendId()` exactly once;
  3. reject a null returned [identity](../../../../glossary.md#term-identity-meanings) with `NullPointerException` message
     `providers[i].backendId()`;
  4. retain that exact provider and returned `BackendId` reference in temporary evaluation state;
     and
  5. reject the second provider whose returned identity is equal to an earlier one with
     `IllegalArgumentException` message
     `duplicate provider backendId: <backendId.value()>`.
- After the provider scan succeeds, scan availability snapshots completely in caller encounter
  order:
  1. reject a null snapshot at index `i` with `NullPointerException` message
     `availabilitySnapshots[i]`;
  2. use the snapshot's non-null `backendId()` as its identity;
  3. retain the exact snapshot reference in temporary [evaluation](../../../../glossary.md#term-evaluation) state; and
  4. reject the second snapshot whose backend identity is equal to an earlier one with
     `IllegalArgumentException` message
     `duplicate availability snapshot backendId: <backendId.value()>`.
- After both scans succeed, validate the supplied identity sets before calling any provider's
  `supports` method:
  1. walk provider identities in provider encounter order and reject the first identity with no
     equal snapshot identity using `IllegalArgumentException` message
     `missing availability snapshot for backendId: <backendId.value()>`; then
  2. walk snapshot identities in snapshot encounter order and reject the first identity with no
     equal provider identity using `IllegalArgumentException` message
     `missing capability provider for backendId: <backendId.value()>`.
- Equal provider and snapshot `BackendId` values match even when they are different object
  references. Snapshot order does not reorder providers or final eligible identities.
- Only after all structural and identity validation succeeds, evaluate backends in provider
  encounter order. For each provider:
  1. skip it without calling `supports` when its matching snapshot reports no currently available
     device;<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
  2. skip it without calling `supports` when its identity/snapshot does not satisfy the optional
     hard requirement exactly as defined below;
  3. otherwise call `supports(query)` exactly once with the exact query reference; and
  4. include the exact retained provider `BackendId` reference only when that call returns
     `true`.
- This availability/requirement-before-capability order avoids querying a backend that cannot
  possibly survive hard eligibility. Provider call order remains the subsequence of provider
  encounter order selected by those immutable facts; snapshot-list order never affects it.
- Propagate a queried provider's runtime failure unchanged. Stop immediately and do not call later
  eligible-to-query providers after such a failure. Do not translate it into unsupported work or
  a no-match result. A provider skipped for availability or requirement mismatch is never called
  and cannot fail the evaluation.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- Match every current [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement) subtype exhaustively:
  - absent requirement from `BackendIntent.unconstrained()` — every supported backend with at
    least one reported available device satisfies the intent;
  - `BackendIdRequirement` — the provider's retained `BackendId` must be equal to the required
    `backendId`; a matching backend still needs support and a non-empty snapshot;
  - `BackendDeviceIdRequirement` — the provider identity must be equal to the required device's
    owning `BackendId`, and the matching snapshot's device map must contain an equal exact device
    identity key;
  - `DeviceClassRequirement` — the matching snapshot must contain at least one available device
    whose [`DeviceClass`](../../../../glossary.md#device-class--deviceclass) is equal to the required class.
- Treat exact-device and device-class matching only as proof that the snapshot's backend currently
  reports at least one matching available device. The existing boolean provider answer remains
  backend-level. Do not infer or claim operation support for that particular device, do not choose
  among matching devices, and do not retain a device identity or class in the result.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype); [operation glossary](../../../../glossary.md#operation))</span>
- A snapshot with an empty device map makes its backend unavailable for this evaluation even when
  its provider supports the operation and an exact-backend requirement matches.<span data-glossary-navigation="true"> ([backend requirement glossary](../../../../glossary.md#backend-requirement--backendrequirement))</span>
- An unsupported backend is excluded even when its snapshot and hard requirement match. A hard
  requirement mismatch is excluded even when capability and availability match. Never reinterpret
  either exclusion as fallback eligibility.
- Accept two empty supplied lists together and return an empty result. Reject a non-empty list on
  only one side through the specified missing-snapshot or missing-provider failure.
- Represent every valid no-match outcome, constrained or unconstrained, with an [immutable](../../../../glossary.md#term-immutable) empty
  `eligibleBackendIds` list. Evaluation itself does not throw solely because no backend remains:
  there is no current compiler-facing failure contract. Planning owns the no-match fact, and the
  later planning orchestration that first consumes this result must fail before scoring or owner
  selection. The public compile exception type/message remains with that later consumer.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [compile glossary](../../../../glossary.md#compile))</span>
- Add no reusable public capability-matrix row or map. The direct hard filters plus one backend-
  level boolean call for each still-possible backend are sufficient for the only current need:
  one query plus one intent produces one hard-eligible backend list. A reusable matrix would query
  and expose unsupported/unavailable rows and snapshot facts without a current consumer.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Update capability package Javadoc to describe the current internal per-query hard-eligibility
  step and its strict backend/device boundary while keeping public provider/query contracts clear.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Require meaningful record, component, canonical-constructor, factory, and package Javadocs.
  Document every input, retained reference, immutable snapshot, result, failure, ordering rule,
  hard-matching rule, no-match meaning, and deliberate boundary.
- Add one focused test class covering exact package-private API/record shape, constructor and
  factory validation, complete validation/evaluation order, identity equality/reference rules,
  all requirement variants, availability and support filtering, deterministic order, no-match,
  provider failure propagation, immutable snapshots, and forbidden added surfaces.
- Finalize affected Javadocs and explanatory/planning documentation through the required separate
  clean-context documentation pass in the same overall implementation change.

## Out of [scope](../../../../glossary.md#term-scope)

- a public capability matrix, public evaluator, public eligibility result, matrix row, candidate
  record, candidate map, backend registry, provider collection object, or planning facade<span data-glossary-navigation="true"> ([capability matrix glossary](../../../../glossary.md#review-tables-and-support-matrices))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade))</span>
- changing [planning](../../../../glossary.md#term-planning)'s config [dependency](../../../../glossary.md#term-dependency) from `implementation` to `api`, adding a dependency, or
  changing any Gradle file or architecture test<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- reusable evaluation across multiple queries or intents, batch graph evaluation, node/value
  identity, graph phase, graph traversal, or compiler orchestration<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [identity glossary](../../../../glossary.md#term-identity-meanings); [graph phase glossary](../../../../glossary.md#graph-phase); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- score calculation, [`PartitionScoringConfig`](../../../../glossary.md#partition-scoring-configuration--partitionscoringconfig) interpretation, calibrated profiles, weights,
  bonuses, penalties, estimates, tie breaking, owner choice, or fallback order<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight); [owner glossary](../../../../glossary.md#term-owner))</span>
- ownership-candidate public contracts, selected ownership, node/segment assignment, graph
  partitioning, same-owner grouping, transfer/materialization planning, or logical memory planning<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [materialization glossary](../../../../glossary.md#materialization))</span>
- a selected [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), device candidate, device-level operation capability, per-device
  provider call, device ranking, default device, route, kernel, executable, or backend prepare<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [kernel glossary](../../../../glossary.md#kernel); [prepare glossary](../../../../glossary.md#prepare))</span>
- weakening an exact [backend](../../../../glossary.md#backend), exact [device](../../../../glossary.md#term-device), or device-class requirement because no match remains
- a no-match exception type, public compilation exception, rejection-reason enum, diagnostics
  taxonomy, structured rejection payload, trace event, attribute, or serialization schema<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [trace glossary](../../../../glossary.md#trace); [serialization glossary](../../../../glossary.md#term-serialization); [schema glossary](../../../../glossary.md#term-schema))</span>
- provider discovery, registration, refresh, [liveness](../../../../glossary.md#term-liveness) monitoring, [service locator](../../../../glossary.md#term-service-locator),
  `ServiceLoader`, classpath scanning, reflection-based discovery, or concrete provider
  implementation<span data-glossary-navigation="true"> ([classpath glossary](../../../../glossary.md#term-classpath))</span>
- changing [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery), [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider), [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent),
  `PartitionScoringConfig`, any backend-contract type, or their existing tests and executable
  behavior
- compiler, prepare, runtime, engine, concrete backend, backend conformance, integration,
  execution, storage, buffer, workspace, schedule, publication, or mutable runtime work<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend); [conformance glossary](../../../../glossary.md#term-conformance); [buffer glossary](../../../../glossary.md#term-buffer); [publication glossary](../../../../glossary.md#term-publication); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- [architecture contract](../../../../glossary.md#architecture-contract), [ADR](../../../../glossary.md#term-architecture-decision-record-adr), Java-version, root-build, or shared-build changes
- unrelated refactoring, documentation cleanup, or a detailed task specification for Planning
  0003+, Config 0004+, or Trace 0003+

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Partition scoring](../../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition))</span>
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Architecture style](../../../../developer-guide/documentation/architecture-style.md)
- [Backend guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Planning master plan](../master-plan.md)
- [Config master plan](../../config/master-plan.md)
- [Backend-contract master plan](../../backend-contract/master-plan.md)
- [Trace master plan](../../trace/master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>
- [Planning task 0001](0001-operation-capability-query-foundation.md)
- [Config task 0001](../../config/tasks/0001-backend-intent-foundation.md)
- [Config task 0003](../../config/tasks/0003-partition-scoring-configuration.md)
- [Backend-contract task 0003](../../backend-contract/tasks/0003-backend-availability-snapshot.md)
- [Backend-contract task 0004](../../backend-contract/tasks/0004-declarative-backend-requirements.md)
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Planning owns backend-neutral capability analysis and hard eligibility before scoring. This task
  asks where one operation occurrence may run and must not answer which device, route, kernel, or
  executable implementation will run it.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [occurrence glossary](../../../../glossary.md#term-occurrence); [run glossary](../../../../glossary.md#term-run); [device glossary](../../../../glossary.md#term-device); [kernel glossary](../../../../glossary.md#kernel))</span>
- [`Operation`](../../../../glossary.md#operation) remains backend-independent. Capability continues to come only from the explicitly
  supplied planning-owned [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider) collaboration.
- [Compile](../../../../glossary.md#compile)-time [ownership](../../../../glossary.md#term-ownership-meanings) is [`BackendId`](../../../../glossary.md#backend-identity--backendid)-level. The result retains only `BackendId` values, never a
  provider, concrete backend, snapshot, requirement, device identity, or live service.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [snapshot glossary](../../../../glossary.md#term-snapshot); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- The supplied snapshot is point-in-time availability data, not discovery or device capability.
  Exact-device and class requirements may use it only to establish the existence of a matching
  currently available device under the candidate backend.
- The current provider answer is backend-level. A matching device or class does not strengthen it
  into device-level operation support. Absence of device-level capability is an explicit accepted
  limitation.
- Hard requirements are filters, not preferences. An empty result remains empty and cannot be
  relaxed by later preference, scoring, profile, fallback, or route logic.
- The package-private contract uses planning's existing implementation dependency on config and
  its current public dependencies on model and backend-contract. No dependency visibility or
  architecture boundary changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [model glossary](../../../../glossary.md#term-model))</span>
- Planning remains independent of runtime, prepare, engine, and concrete backends. This task adds
  no public signature, build edge, or architecture rule, so no architecture-test change is
  justified.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [signature glossary](../../../../glossary.md#term-signature))</span>
- Stop if implementation needs a public evaluator/result, `api` visibility for config, another
  production type, device-level capability, chosen device, score/candidate/owner contract,
  diagnostics taxonomy, dependency edit, or architecture decision.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

## Package impact

Existing package extended:

- `io.github.pho001.synaptik.planning.capability` — current public operation query/provider
  contracts plus one package-private per-query hard-eligibility value for later in-module planning<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [module glossary](../../../../glossary.md#term-module))</span>

Type placement:

- `io.github.pho001.synaptik.planning.capability.BackendEligibility` — package-private immutable
  result and concrete static evaluation entry point because capability and hard-requirement
  intersection occur together before ownership scoring<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>

No public type is added. No [matrix](../../../../glossary.md#term-matrix-meanings), row, candidate, [registry](../../../../glossary.md#term-registry), service, evaluator abstraction,
[ownership](../../../../glossary.md#term-ownership-meanings), [partition](../../../../glossary.md#partition), memory, [device](../../../../glossary.md#term-device), [runtime](../../../../glossary.md#term-runtime), or [backend](../../../../glossary.md#backend)-specific package is added. [Planning](../../../../glossary.md#term-planning) task
0003 must reassess how its later `ownership` package receives eligible facts and whether a public
planner [facade](../../../../glossary.md#term-facade) is then justified.

## Affected files

The intended overall implementation [scope](../../../../glossary.md#term-scope) is exactly fourteen paths.

Production — exactly two paths:

- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/BackendEligibility.java`
- update
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/package-info.java`
  for the current internal eligibility boundary

Tests — exactly one path:

- add
  `modules/planning/src/test/java/io/github/pho001/synaptik/planning/capability/BackendEligibilityTest.java`

Architecture-status and explanatory documentation — exactly six paths:

- current-versus-planned wording only in `docs/architecture/partition-scoring.md`, without
  changing an architecture rule
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

Review without modification: all completed planning/config/[backend](../../../../glossary.md#backend)-contract production and tests
outside the two authorized planning production paths and one new test; every build file; current
architecture tests; [compile](../../../../glossary.md#compile) workflow guide; other architecture explanations and ADRs; [compiler](../../../../glossary.md#term-compiler),
[runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), [engine](../../../../glossary.md#term-engine), and [concrete backend](../../../../glossary.md#concrete-backend) placeholders/implementations; backend-[conformance](../../../../glossary.md#term-conformance)
and integration tests; Public [Tensor](../../../../glossary.md#tensor) APIs; [trace](../../../../glossary.md#trace) Java/tests; and every later task row/specification.

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly the fourteen paths listed above. Stop if implementation requires another production
type, test, document, package, public [API](../../../../glossary.md#term-application-programming-interface-api), [dependency](../../../../glossary.md#term-dependency)/build edit, [architecture test](../../../../glossary.md#term-architecture-test), architecture
change, candidate/score/[owner](../../../../glossary.md#term-owner)/[device](../../../../glossary.md#term-device)/diagnostic contract, provider implementation, [lifecycle](../../../../glossary.md#lifecycle)
behavior, or detailed later task specification. Do not use a follow-up task to hide an incomplete
acceptance criterion.

## Required [Javadoc](../../../../glossary.md#term-javadoc) contracts

- Record Javadoc must define this value as one package-private per-query result, explain the exact
  provider-order eligible [`BackendId`](../../../../glossary.md#backend-identity--backendid) list, and state that it retains no provider, snapshot,
  requirement, device, score, owner choice, or executable state.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [device glossary](../../../../glossary.md#term-device); [owner glossary](../../../../glossary.md#term-owner))</span>
- Record-component, canonical-constructor, and explicitly declared public component-accessor
  Javadocs must document
  non-null inputs/results, encounter-order null/duplicate validation, exact reference retention,
  immutable membership snapshot, ordinary record semantics, and every exact failure message.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Factory Javadoc must document all four inputs, their ownership and retention behavior, the full
  structural-validation-before-provider-call sequence, availability/requirement short-circuiting,
  exactly-once calls for still-possible providers, matching by <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`BackendId.equals`, provider order,
  snapshot-order irrelevance, every requirement subtype, empty/no-match meaning, and propagated
  queried-provider failures.
- Factory `@return` must state that the result is new and non-null, retains the exact query,
  snapshots the exact eligible provider identity references in provider order, and may contain an
  empty list.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Package Javadoc must distinguish the two public query/provider contracts from the package-
  private eligibility step and keep scoring, ownership, public orchestration, partitioning,
  device-level capability/selection, preparation, runtime, and execution planned.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>

## Acceptance criteria

- `BackendEligibility` is a package-private final record in the exact capability package with
  exactly `OperationCapabilityQuery query` and `List<BackendId> eligibleBackendIds` components in
  that order, one package-private canonical constructor, explicitly declared public component
  accessors on the package-private record, ordinary record object methods, and one added package-private static<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
  `evaluate` factory with the exact [signature](../../../../glossary.md#term-signature) in [Scope](../../../../glossary.md#term-scope).
- It has no public constructor, field, nested type, interface, builder, registry, service,
  callback, public matrix, or externally constructible/evaluable surface. Its only declared public
  methods are the two Java-required component accessors and ordinary record object methods; the
  package-private enclosing type keeps them outside the externally accessible planning API.
  Existing public planning API remains exactly <span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [callback glossary](../../../../glossary.md#term-callback); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>[`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) and
  [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider) plus their ordinary methods.
- Direct result construction follows the exact query/list/element/duplicate/snapshot validation
  sequence and messages, retains the exact query and identity references, and exposes an
  immutable ordered membership snapshot.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [identity glossary](../../../../glossary.md#term-identity-meanings); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Factory top-level, provider, snapshot, duplicate, and missing-counterpart validation follows the
  exact order and messages in Scope. Every structural/identity failure occurs before the first
  `supports` call.
- Each provider's `backendId()` is called exactly once in provider order. Equal identities match
  snapshots by value, duplicate equal provider or snapshot identities fail, and the exact
  provider-returned identity reference becomes the result element when eligible.
- After full validation, providers with empty snapshots or hard-requirement mismatches are not
  called. Every remaining provider is called exactly once with the exact query, in provider order.
  Snapshot input order does not affect call or result order.
- Queried-provider failures propagate unchanged and stop later eligible-to-query provider calls.
  A `false` result excludes that [backend](../../../../glossary.md#backend) without a rejection reason.
- An unconstrained intent accepts exactly supported backends with non-empty snapshots.
  `BackendIdRequirement`, `BackendDeviceIdRequirement`, and `DeviceClassRequirement` follow the
  exact matching semantics in Scope.
- Exact-device and class matching reads only immutable snapshot membership. It neither claims
  device-level operation support nor selects, retains, or returns a matching device.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [snapshot membership glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [operation glossary](../../../../glossary.md#operation))</span>
- Empty snapshots exclude their backend. Missing snapshot/provider entries are invalid supplied
  composition and fail; they are not treated as ordinary unavailable/unsupported candidates.
- Valid zero-match evaluations return an immutable empty list. Hard requirements are never
  weakened, and no default/fallback backend becomes eligible.
- Result state contains only the query and eligible backend IDs. No provider, [snapshot](../../../../glossary.md#term-snapshot),
  [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement), [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid), [`DeviceClass`](../../../../glossary.md#device-class--deviceclass), capability boolean,
  score, candidate, owner, route, or runtime object is retained.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype); [owner glossary](../../../../glossary.md#term-owner); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Focused automated tests lock the exact package-private generic record/factory shape, complete
  validation and evaluation order, exact failures, identity/reference/list semantics, all
  requirement cases, no-match representation, and forbidden surfaces. No recurring reflection,
  bytecode, or import invariant is left as a manual implementation check.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [evaluation glossary](../../../../glossary.md#term-evaluation); [bytecode glossary](../../../../glossary.md#term-bytecode); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- [Planning](../../../../glossary.md#term-planning) build dependencies and visibility remain unchanged: [model](../../../../glossary.md#term-model) and backend-contract stay
  `api`; config and [trace](../../../../glossary.md#trace) stay `implementation`; no architecture-test or root-build file changes.
- Package Javadoc and the six explanatory documents mark per-query hard eligibility current only
  after implementation, explicitly state that it is internal, and leave public planner
  orchestration, reusable matrix APIs, scoring, profiles, ownership, partitioning, device-level
  capability/selection, compiler integration, preparation, runtime, and execution planned.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Planning master plan marks only task 0002 Ready before implementation. Planning 0001 and Config
  0001–0003 remain Complete; Planning 0003+, Config 0004+, and Trace 0003+ remain Draft without
  detailed specifications.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- A separate clean-context documentation-focused pass finalizes the new/package Javadocs and all
  eleven documentation/planning paths in the same overall change. It reuses successful Java
  evidence unless executable Java changes or a concrete reason is recorded.
- Exactly one final planning module suite, planning Javadoc, repository Markdown, generated-page
  inspection for the remaining public package/types, exact fourteen-path scope, unchanged
  dependency/status/later-spec checks, final newlines, trailing whitespace, and<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>
  `git diff --check` [validation](../../../../glossary.md#term-validation) pass.
- No architecture test or repository-wide suite is run by habit: the task adds one internal
  planning evaluation with no dependency, build, public API, architecture boundary, concrete
  backend behavior, or cross-module executable change. Continuous integration or the next
  recorded capability checkpoint owns that tier.<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test); [run glossary](../../../../glossary.md#term-run); [concrete backend glossary](../../../../glossary.md#concrete-backend); [Continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development commands are allowed while executable Java changes. After executable Java
stabilizes, [run](../../../../glossary.md#term-run) exactly one final [planning](../../../../glossary.md#term-planning) [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:planning:test
```

Record suite and test counts from XML reports. Then hand this task, the actual diff, and exact Java
evidence to the separate clean-context documentation-focused pass. That pass must not rerun the
successful Java suite unless it changes executable behavior or records a concrete cross-check
risk. After final [Javadoc](../../../../glossary.md#term-javadoc) and Markdown edits, run:

```bash
./gradlew :modules:planning:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

Inspect generated `package-summary.html`, `OperationCapabilityQuery.html`, and
`BackendCapabilityProvider.html` for the public/current-versus-internal boundary; the package-
private record is intentionally not a new public generated page. Confirm exactly fourteen
authorized paths, unchanged planning dependencies, exactly one global [Ready](../../../../glossary.md#term-ready) task during
implementation, no detailed Planning 0003, Config 0004, or [Trace](../../../../glossary.md#trace) 0003 specification, balanced
fences, valid links/anchors, final newlines, no trailing whitespace, and the final diff.

Do not run architecture tests or the repository-wide suite unless implementation actually changes
a [dependency](../../../../glossary.md#term-dependency)/public surface or reveals another concrete repository-wide risk. Such a change is
outside this task and requires stopping rather than silently expanding validation and [scope](../../../../glossary.md#term-scope).

## Documentation [handoff](../../../../glossary.md#term-handoff)

After the implementation context records the final [planning](../../../../glossary.md#term-planning) test result, hand this specification,
the actual fourteen-path diff, and exact Java evidence to a separate clean-context documentation-
focused agent or thread. The handoff must identify the package-private result/factory, exact
matching and ordering rules, [BackendId](../../../../glossary.md#backend-identity--backendid)-only output, unchanged public/[dependency](../../../../glossary.md#term-dependency) surface, the six
expected explanatory documents, and the [validation](../../../../glossary.md#term-validation) commands above.

That pass must read the [architecture contract](../../../../glossary.md#architecture-contract), documentation rules, General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc),
Architecture, [Backend](../../../../glossary.md#backend) guide, User guide, Planning, and Example profiles, final source/tests,
current backend-contract/config/planning contracts, and directly affected documents. It must
independently finalize record/component/constructor/factory/package Javadocs, examples,
current-versus-planned wording, links, terminology, and glossary impact.

It must record reasoned no-change conclusions for planning [Gradle](../../../../glossary.md#term-gradle)/dependencies, architecture
rules/ADRs/tests, [compile](../../../../glossary.md#compile) workflow beyond the focused status pages, backend-[conformance](../../../../glossary.md#term-conformance) and
integration tests, backend-contract/config/[trace](../../../../glossary.md#trace) Java, [compiler](../../../../glossary.md#term-compiler)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[engine](../../../../glossary.md#term-engine), concrete
backends, and every other [module](../../../../glossary.md#term-module). It reuses the implementation context's successful Java evidence
unless executable behavior changes.

## Dependencies

- Complete [Planning](../../../../glossary.md#term-planning) task 0001 with stable [immutable](../../../../glossary.md#term-immutable) [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) and explicitly
  supplied backend-level [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider) contracts.
- Complete [Backend](../../../../glossary.md#backend)-contract tasks 0001–0004 with stable [`BackendId`](../../../../glossary.md#backend-identity--backendid), [`BackendDeviceId`](../../../../glossary.md#backend-device-identity--backenddeviceid),
  [`DeviceClass`](../../../../glossary.md#device-class--deviceclass), [`BackendAvailabilitySnapshot`](../../../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot), and exhaustive hard-requirement vocabulary.
- Complete Config task 0001 with stable [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) optionality and exact hard-requirement
  retention.
- Complete Config task 0003 as sequencing context: its optional device-class preference is soft
  input for later scoring and is deliberately not consumed here.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- Complete [Trace](../../../../glossary.md#trace) tasks 0001–0002 as diagnostic foundation only; this task adds no trace [schema](../../../../glossary.md#term-schema).

## Follow-up tasks

- After task 0002 completes, separately reassess Config task 0004 profile contracts as the likely
  next area before Planning task 0003 scoring. Do not make either task Ready automatically.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Planning task 0003 remains Draft and will define the smallest ownership-candidate/scoring
  contract only after hard eligibility and required profile inputs are stable. It must reassess
  the cross-package/public facade for consuming <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [facade glossary](../../../../glossary.md#term-facade))</span>`BackendEligibility` rather than assuming this
  task's internal entry point becomes public.
- Planning tasks 0004–0006 retain same-owner partitioning, logical materialization/memory
  requirements, and planning closure as Draft rows without detailed specifications.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [materialization glossary](../../../../glossary.md#materialization))</span>
- Config tasks 0004–0008 and [Trace](../../../../glossary.md#trace) tasks 0003–0008 remain Draft without detailed specifications.
- Device-level operation capability, chosen-device planning, structured rejection diagnostics,
  and public compiler failure contracts remain explicit limitations, not incomplete criteria.<span data-glossary-navigation="true"> ([Device glossary](../../../../glossary.md#term-device); [operation glossary](../../../../glossary.md#operation); [compiler glossary](../../../../glossary.md#term-compiler))</span>

## Architecture impact

Expected impact: None.

The task realizes existing [planning](../../../../glossary.md#term-planning) [ownership](../../../../glossary.md#term-ownership-meanings) of capability analysis and hard eligibility while
preserving [BackendId](../../../../glossary.md#backend-identity--backendid)-level ownership, current dependencies, and the provider/[snapshot](../../../../glossary.md#term-snapshot)/config
boundaries. It adds no public [API](../../../../glossary.md#term-application-programming-interface-api) or architecture-test need. If implementation reveals that a
public config-exposing entry point, [dependency](../../../../glossary.md#term-dependency) visibility change, [device](../../../../glossary.md#term-device)-level capability, chosen
device, or another architecture rule is required, stop before editing source or architecture
documentation.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md, the directly relevant documentation profiles,
docs/planning/roadmap.md, the planning/config/backend-contract/trace master plans, and
docs/planning/modules/planning/tasks/0002-per-query-backend-hard-eligibility.md in full.

Implement task 0002 exactly inside its fourteen authorized paths. Add only the package-private
BackendEligibility record/static evaluation, focused automated tests, required Javadocs,
current-status documentation, and synchronized planning records. Preserve exact validation,
provider-order, snapshot, requirement, BackendId-only, empty-result, and no-device-selection
semantics. Add no public matrix/evaluator, score/candidate/owner/device/diagnostic contract,
dependency/build change, provider implementation, lifecycle behavior, or later task spec. Stop
and report any architecture or scope conflict.

After executable Java and the single final planning module suite, hand the actual diff and exact
evidence to a separate clean-context documentation-focused agent or thread. Reuse successful Java
evidence unless executable behavior changes. Do not run architecture or root tests without the
task's concrete trigger. Mark task 0002 Complete only after every criterion and documentation
validation pass.
```

## Local decisions

- Use one per-query result instead of a reusable public capability matrix. The next concrete need
  is the ordered hard-eligible backend identities for one occurrence; no consumer needs retained
  unsupported rows, reusable intent application, or public snapshot facts.<span data-glossary-navigation="true"> ([capability matrix glossary](../../../../glossary.md#review-tables-and-support-matrices))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix); [backend glossary](../../../../glossary.md#backend); [occurrence glossary](../../../../glossary.md#term-occurrence); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Keep evaluation and result in one package-private record. This avoids a generic evaluator
  abstraction and avoids exposing config through a public Planning signature before a public
  planner consumer exists.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [Planning glossary](../../../../glossary.md#term-planning); [signature glossary](../../../../glossary.md#term-signature))</span>
- Retain only the exact provider-returned [`BackendId`](../../../../glossary.md#backend-identity--backendid) references. These are the capability and
  ownership domain; equal snapshot identities validate supplied composition but do not replace
  the provider identity reference in the result.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Treat the provider list as the canonical deterministic backend order. Snapshot maps explicitly
  promise no iteration order, and snapshot-list order is only validation order for snapshot-side
  failures.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation))</span>
- Validate the complete supplied composition before any capability call. Missing or duplicate
  identities are caller composition errors, not unsupported/unavailable candidates, and provider
  side effects must not begin before those errors are known.
- Apply availability and hard intent before calling capability providers. These immutable checks
  are cheaper, avoid irrelevant provider failures and per-occurrence work, and make a reusable
  all-backend matrix unnecessary. Every provider that can still qualify is called exactly once in
  provider order.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Treat a backend as currently available only when its snapshot contains at least one device.
  The snapshot has no separate backend status, so an empty device map cannot support ownership.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- Exact-device and class requirements prove only matching availability under one backend. They do
  not narrow the existing backend-level capability answer or select the matching device.
- Represent no match as an empty immutable list. A dedicated exception would create a public or
  internal failure contract before compiler/planning orchestration exists. Later orchestration
  must turn emptiness into failure before scoring and may not weaken the requirement.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- Keep config and [trace](../../../../glossary.md#trace) dependencies at `implementation`. The new type is package-private and no
  public signature exposes [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) or trace vocabulary.
- Declare the two record component accessors `public`. The earlier package-private-accessor
  wording was not implementable: Java requires a record component accessor to be public, and the
  compiler rejected both narrower declarations with `invalid accessor method in record
  BackendEligibility (accessor method must be public)`. The enclosing record remains package-
  private, so its effective external API boundary is unchanged.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Known limitations

- Capability remains backend-level. The result cannot prove that any specific available device
  supports the operation.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [device glossary](../../../../glossary.md#term-device); [operation glossary](../../../../glossary.md#operation))</span>
- Exact-device and device-class requirements do not identify or retain which matching device
  made the backend eligible.
- The result contains no rejection reason and cannot distinguish unsupported, unavailable, or
  hard-requirement-mismatched backends after evaluation.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- The result is internal to the capability package. No current compiler or public planning facade
  invokes it; Planning task 0003 must reassess the public/cross-package handoff.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [facade glossary](../../../../glossary.md#term-facade); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- An empty result has no dedicated exception yet. Later orchestration owns the public compile
  failure type/message but must treat emptiness as terminal before scoring.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- Availability is only as current as the caller-supplied snapshots and has no refresh, timestamp,
  liveness, or registration guarantee.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness))</span>
- No [serialization](../../../../glossary.md#term-serialization) or external compatibility guarantee is established.

## [Validation](../../../../glossary.md#term-validation) evidence

- [Planning](../../../../glossary.md#term-planning) context `/root/plan_planning_0002` read the architecture contract; focused lifecycle,
  module, dependency, partition-scoring, and prepare/backend explanations; documentation rules
  and General/Planning profiles; planning guide and roadmap; planning/config/backend-contract/
  trace master plans; completed Planning 0001, Config 0001–0003, and Backend-contract 0001–0004
  tasks; and current production, focused tests, builds, package Javadocs, API/user/backend guides,
  and glossary sections before selecting this contract.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [partition glossary](../../../../glossary.md#partition); [prepare glossary](../../../../glossary.md#prepare); [trace glossary](../../../../glossary.md#trace); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The current consumers justify only a per-query internal eligibility result. A public reusable
  matrix would expose unsupported rows and supplied snapshots without a consumer, while a public
  evaluator accepting <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>[`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) would require an otherwise unnecessary config `api`
  visibility change. The selected package-private record preserves the architecture and current
  dependency surface.
- The exact [BackendId](../../../../glossary.md#backend-identity--backendid)/[device](../../../../glossary.md#term-device)/class boundary is resolved: every final candidate is a provider-owned
  `BackendId`; exact-device/class requirements consult only the matching immutable snapshot for
  existence and never infer device-level capability or retain a chosen device.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed for 224 Markdown files, 3,950 local links,
  238 local anchors, 2,822 fence markers, final newlines, and trailing whitespace.
- Planning-time `git diff --check` passed with no output. The combined tracked/untracked scope is
  exactly the five authorized planning paths: this task, the planning/config/trace master plans,
  and the roadmap. No Java, test, Gradle, architecture, API, guide, or glossary file changed
  during planning.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Status and task-directory checks found exactly one global Ready master-plan row and exactly one
  Ready detailed task: Planning 0002. Planning 0001 and Config 0001–0003 remain Complete;
  Planning 0003+, Config 0004+, and Trace 0003+ remain Draft. No detailed Planning 0003, Config
  0004, or Trace 0003 specification exists.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- The original package-private accessor requirement was corrected before final validation. Java
  record component accessors must be public; attempting the narrower declarations produced
  `invalid accessor method in record BackendEligibility (accessor method must be public)` for both
  accessors. Explicit public accessors on a package-private top-level record preserve the intended
  external API boundary.
- Implementation context `/root/implement_planning_0002` ran
  `./gradlew :modules:planning:test --tests io.github.pho001.synaptik.planning.capability.BackendEligibilityTest`;
  it passed 15 tests with no failures, errors, or skips after the accessor correction. An earlier
  focused compile failed only because the original specification required impossible package-
  private record accessors; no behavioral test failure was involved.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- The same implementation context then ran the single final
  `./gradlew :modules:planning:test`; it passed 25 tests across two suites with no failures,
  errors, or skips. Documentation context
  `/root/implement_planning_0002/planning_0002_docs` reused this evidence because it changed no
  executable Java behavior and identified no concrete cross-check risk requiring duplicate tests.
- Documentation context `/root/implement_planning_0002/planning_0002_docs` applied the General,
  API/Javadoc, Architecture, Backend guide, User guide, Planning, and Example profiles. It read
  the authoritative architecture and planning contracts, final source and tests, directly related
  backend/config/planning contracts, and every affected document before finalizing record,
  component, constructor, factory, accessor, package, explanatory, glossary, and planning text.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `./gradlew :modules:planning:javadoc` passed both before and after the final factory exception-
  message Javadoc refinement. Inspection of the final generated `package-summary.html`,
  `OperationCapabilityQuery.html`, and
  `BackendCapabilityProvider.html` confirmed that the public surface remains the query/provider
  pair, the internal hard-eligibility boundary is explicit, and no public `BackendEligibility`
  page or all-classes entry exists.
- Repeated `python3 /tmp/validate_synaptik_markdown.py` runs after the explanatory and completion-
  record refinements all passed for 224 Markdown files, 3,955 local links, 243 local anchors,
  2,822 fence markers, final newlines, and trailing whitespace.
- The combined tracked/untracked path audit contains exactly the fourteen authorized paths: two
  planning production paths, the implementation-owned focused test, six explanatory documents,
  and five planning documents. The test remains implementation-owned and was not edited by the
  documentation context.
- Planning [dependency](../../../../glossary.md#term-dependency) declarations remain exactly `api` for [model](../../../../glossary.md#term-model) and [backend](../../../../glossary.md#backend)-contract and
  `implementation` for config and trace. No Gradle file, dependency visibility, root build,
  Java-version setting, architecture rule, ADR, or architecture test changed because the new
  result/factory is package-private and adds no module edge or public config-exposing signature.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [signature glossary](../../../../glossary.md#term-signature))</span>
- Compile workflow documentation beyond the authorized focused Compile API status page, compiler,
  prepare, runtime, engine, concrete backends, backend-contract/config/trace Java and tests, and
  every other module remain unchanged because this task adds only an internal compile-time
  eligibility fact and no lifecycle consumer, provider implementation, backend behavior, or
  execution path. Backend-conformance and integration tests therefore need no change.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Final status checks found zero global Ready master-plan rows or detailed tasks. Planning tasks
  0001 and 0002 and Config tasks 0001–0003 are Complete; Planning 0003+, Config 0004+, and Trace
  0003+ remain Draft. No detailed Planning 0003, Config 0004, or Trace 0003 specification exists.
- Final exact-scope, status, later-spec, dependency, generated-page, newline, trailing-whitespace,
  fence, link, anchor, and `git diff --check` validation passed. Architecture and repository-wide
  tests were not run because no dependency, public surface, architecture boundary, concrete
  backend behavior, or cross-module executable contract changed.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>

## Implementation notes

- Added the package-private `BackendEligibility` record and factory with the exact two-component
  state, validation order, provider/snapshot equality association, availability and hard-intent
  short-circuiting, provider-order evaluation, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [snapshot glossary](../../../../glossary.md#term-snapshot); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>[`BackendId`](../../../../glossary.md#backend-identity--backendid)-only result, and immutable empty
  no-match representation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Explicitly declared `query()` and `eligibleBackendIds()` public because Java record component
  accessors cannot reduce visibility. The initial narrower declarations failed compilation with<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
  `invalid accessor method in record BackendEligibility (accessor method must be public)`; the
  top-level record, canonical constructor, and static factory remain package-private.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>

## Completion summary

- Completed changes: added internal per-query backend hard eligibility with complete structural
  validation, exact availability/requirement matching, provider-order backend-level capability
  calls, and immutable <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend); [immutable glossary](../../../../glossary.md#term-immutable))</span>[`BackendId`](../../../../glossary.md#backend-identity--backendid)-only results; finalized the public/internal/planned boundary.
- Files changed or created: `BackendEligibility.java`, capability `package-info.java`,
  `BackendEligibilityTest.java`, the six authorized explanatory documents, this task, the
  planning/config/trace master plans, and the roadmap; exactly fourteen paths.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [trace glossary](../../../../glossary.md#trace))</span>
- Tests and validation: reused the passing 15-test focused result and 25-test/two-suite final
  planning result; planning Javadoc, generated-page inspection, repository Markdown validation,
  exact-scope/dependency/status/later-spec checks, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [dependency glossary](../../../../glossary.md#term-dependency))</span>`git diff --check` passed.
- Documentation-agent review: clean context
  `/root/implement_planning_0002/planning_0002_docs` finalized all affected Javadocs,
  explanations, glossary terminology, and planning evidence without changing executable behavior
  or rerunning Java tests.
- Documentation impact: the current internal eligibility step and its BackendId/device boundary
  are documented; reusable/public matrices, orchestration, scoring/profiles/ownership/partitioning,
  compiler integration, device-level capability/selection, preparation, runtime, and execution
  remain explicitly planned.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [matrices glossary](../../../../glossary.md#term-matrix-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: record, components, package-private canonical constructor/factory, Java-required
  public component accessors on the package-private record, and package boundaries are complete.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Glossary impact: added the reusable [`Backend hard eligibility`](../../../../glossary.md#backend-hard-eligibility) distinction and updated capability,
  intent, requirement, and implementation-status boundaries.
- Unresolved issues: None.
- Follow-up required: None; a separate frontier reassessment may select Config 0004, but no next
  task was made Ready.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

Status: Complete
