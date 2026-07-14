# Task 0002: Per-Query Backend Hard Eligibility

## Status

Complete

## Goal

Add the smallest planning-owned evaluation that combines one immutable operation-capability
question, explicitly supplied backend capability providers, caller-supplied availability
snapshots, and one `BackendIntent` into the ordered backend identities that remain hard-eligible
before any scoring.

Mental model:

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

The result is per operation occurrence and retains only backend ownership identities. It does not
retain providers, snapshots, requirements, devices, or a selected device. It represents no match
with an empty list; later planning orchestration must stop before scoring that empty result and
must not weaken a hard requirement or invent a fallback.

## Scope

- Add package-private record `BackendEligibility` in
  `io.github.pho001.synaptik.planning.capability` with exactly these ordered components:
  `OperationCapabilityQuery query` and `List<BackendId> eligibleBackendIds`.
- Keep the record package-private. No current compiler, planner facade, or other external consumer
  exists, and a public entry point that exposed `BackendIntent` would unnecessarily change the
  current config dependency from `implementation` to `api`. Later ownership/scoring work must
  reassess the public facade from its concrete consumer rather than speculating here.
- Give the record one package-private canonical constructor with this exact validation and
  snapshot sequence:
  1. reject null `query` with `NullPointerException` message `query`;
  2. reject null `eligibleBackendIds` with `NullPointerException` message
     `eligibleBackendIds`;
  3. scan eligible backend identities in encounter order and reject the first null with
     `NullPointerException` message `eligibleBackendIds[index]`;
  4. reject the second occurrence of an equal backend identity with
     `IllegalArgumentException` message
     `duplicate eligible backendId: <backendId.value()>`; and
  5. snapshot list membership with `List.copyOf`, retaining the exact `BackendId` element
     references.
- Preserve ordinary record equality, hashing, and diagnostic `toString()` behavior over the
  exact query reference's value and ordered immutable eligible-identity snapshot.
- Explicitly declare and document the public `query()` and `eligibleBackendIds()` record component
  accessors. Java requires a record component accessor to be public; because the enclosing top-
  level record remains package-private, these methods do not create an externally accessible
  planning API. They return the exact retained query reference and immutable ordered list snapshot,
  respectively.
- Add exactly one package-private static factory on the record:

  ```java
  static BackendEligibility evaluate(
          OperationCapabilityQuery query,
          BackendIntent intent,
          List<BackendCapabilityProvider> providers,
          List<BackendAvailabilitySnapshot> availabilitySnapshots)
  ```

- The factory validates top-level references in this exact order before reading any list element:
  1. null `query` -> `NullPointerException("query")`;
  2. null `intent` -> `NullPointerException("intent")`;
  3. null `providers` -> `NullPointerException("providers")`; and
  4. null `availabilitySnapshots` ->
     `NullPointerException("availabilitySnapshots")`.
- Scan providers completely in caller encounter order before scanning any availability snapshot:
  1. reject a null provider at index `i` with `NullPointerException` message `providers[i]`;
  2. call that provider's `backendId()` exactly once;
  3. reject a null returned identity with `NullPointerException` message
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
  3. retain the exact snapshot reference in temporary evaluation state; and
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
     device;
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
  and cannot fail the evaluation.
- Match every current `BackendRequirement` subtype exhaustively:
  - absent requirement from `BackendIntent.unconstrained()` — every supported backend with at
    least one reported available device satisfies the intent;
  - `BackendIdRequirement` — the provider's retained `BackendId` must be equal to the required
    `backendId`; a matching backend still needs support and a non-empty snapshot;
  - `BackendDeviceIdRequirement` — the provider identity must be equal to the required device's
    owning `BackendId`, and the matching snapshot's device map must contain an equal exact device
    identity key;
  - `DeviceClassRequirement` — the matching snapshot must contain at least one available device
    whose `DeviceClass` is equal to the required class.
- Treat exact-device and device-class matching only as proof that the snapshot's backend currently
  reports at least one matching available device. The existing boolean provider answer remains
  backend-level. Do not infer or claim operation support for that particular device, do not choose
  among matching devices, and do not retain a device identity or class in the result.
- A snapshot with an empty device map makes its backend unavailable for this evaluation even when
  its provider supports the operation and an exact-backend requirement matches.
- An unsupported backend is excluded even when its snapshot and hard requirement match. A hard
  requirement mismatch is excluded even when capability and availability match. Never reinterpret
  either exclusion as fallback eligibility.
- Accept two empty supplied lists together and return an empty result. Reject a non-empty list on
  only one side through the specified missing-snapshot or missing-provider failure.
- Represent every valid no-match outcome, constrained or unconstrained, with an immutable empty
  `eligibleBackendIds` list. Evaluation itself does not throw solely because no backend remains:
  there is no current compiler-facing failure contract. Planning owns the no-match fact, and the
  later planning orchestration that first consumes this result must fail before scoring or owner
  selection. The public compile exception type/message remains with that later consumer.
- Add no reusable public capability-matrix row or map. The direct hard filters plus one backend-
  level boolean call for each still-possible backend are sufficient for the only current need:
  one query plus one intent produces one hard-eligible backend list. A reusable matrix would query
  and expose unsupported/unavailable rows and snapshot facts without a current consumer.
- Update capability package Javadoc to describe the current internal per-query hard-eligibility
  step and its strict backend/device boundary while keeping public provider/query contracts clear.
- Require meaningful record, component, canonical-constructor, factory, and package Javadocs.
  Document every input, retained reference, immutable snapshot, result, failure, ordering rule,
  hard-matching rule, no-match meaning, and deliberate boundary.
- Add one focused test class covering exact package-private API/record shape, constructor and
  factory validation, complete validation/evaluation order, identity equality/reference rules,
  all requirement variants, availability and support filtering, deterministic order, no-match,
  provider failure propagation, immutable snapshots, and forbidden added surfaces.
- Finalize affected Javadocs and explanatory/planning documentation through the required separate
  clean-context documentation pass in the same overall implementation change.

## Out of scope

- a public capability matrix, public evaluator, public eligibility result, matrix row, candidate
  record, candidate map, backend registry, provider collection object, or planning facade
- changing planning's config dependency from `implementation` to `api`, adding a dependency, or
  changing any Gradle file or architecture test
- reusable evaluation across multiple queries or intents, batch graph evaluation, node/value
  identity, graph phase, graph traversal, or compiler orchestration
- score calculation, `PartitionScoringConfig` interpretation, calibrated profiles, weights,
  bonuses, penalties, estimates, tie breaking, owner choice, or fallback order
- ownership-candidate public contracts, selected ownership, node/segment assignment, graph
  partitioning, same-owner grouping, transfer/materialization planning, or logical memory planning
- a selected `BackendDeviceId`, device candidate, device-level operation capability, per-device
  provider call, device ranking, default device, route, kernel, executable, or backend prepare
- weakening an exact backend, exact device, or device-class requirement because no match remains
- a no-match exception type, public compilation exception, rejection-reason enum, diagnostics
  taxonomy, structured rejection payload, trace event, attribute, or serialization schema
- provider discovery, registration, refresh, liveness monitoring, service locator,
  `ServiceLoader`, classpath scanning, reflection-based discovery, or concrete provider
  implementation
- changing `OperationCapabilityQuery`, `BackendCapabilityProvider`, `BackendIntent`,
  `PartitionScoringConfig`, any backend-contract type, or their existing tests and executable
  behavior
- compiler, prepare, runtime, engine, concrete backend, backend conformance, integration,
  execution, storage, buffer, workspace, schedule, publication, or mutable runtime work
- architecture contract, ADR, Java-version, root-build, or shared-build changes
- unrelated refactoring, documentation cleanup, or a detailed task specification for Planning
  0003+, Config 0004+, or Trace 0003+

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Architecture style](../../../../developer-guide/documentation/architecture-style.md)
- [Backend guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Planning master plan](../master-plan.md)
- [Config master plan](../../config/master-plan.md)
- [Backend-contract master plan](../../backend-contract/master-plan.md)
- [Trace master plan](../../trace/master-plan.md)
- [Planning task 0001](0001-operation-capability-query-foundation.md)
- [Config task 0001](../../config/tasks/0001-backend-intent-foundation.md)
- [Config task 0003](../../config/tasks/0003-partition-scoring-configuration.md)
- [Backend-contract task 0003](../../backend-contract/tasks/0003-backend-availability-snapshot.md)
- [Backend-contract task 0004](../../backend-contract/tasks/0004-declarative-backend-requirements.md)
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Planning owns backend-neutral capability analysis and hard eligibility before scoring. This task
  asks where one operation occurrence may run and must not answer which device, route, kernel, or
  executable implementation will run it.
- `Operation` remains backend-independent. Capability continues to come only from the explicitly
  supplied planning-owned `BackendCapabilityProvider` collaboration.
- Compile-time ownership is `BackendId`-level. The result retains only `BackendId` values, never a
  provider, concrete backend, snapshot, requirement, device identity, or live service.
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
  architecture boundary changes.
- Planning remains independent of runtime, prepare, engine, and concrete backends. This task adds
  no public signature, build edge, or architecture rule, so no architecture-test change is
  justified.
- Stop if implementation needs a public evaluator/result, `api` visibility for config, another
  production type, device-level capability, chosen device, score/candidate/owner contract,
  diagnostics taxonomy, dependency edit, or architecture decision.

## Package impact

Existing package extended:

- `io.github.pho001.synaptik.planning.capability` — current public operation query/provider
  contracts plus one package-private per-query hard-eligibility value for later in-module planning

Type placement:

- `io.github.pho001.synaptik.planning.capability.BackendEligibility` — package-private immutable
  result and concrete static evaluation entry point because capability and hard-requirement
  intersection occur together before ownership scoring

No public type is added. No matrix, row, candidate, registry, service, evaluator abstraction,
ownership, partition, memory, device, runtime, or backend-specific package is added. Planning task
0003 must reassess how its later `ownership` package receives eligible facts and whether a public
planner facade is then justified.

## Affected files

The intended overall implementation scope is exactly fourteen paths.

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

Planning — exactly five paths:

- add and finalize this task
- `docs/planning/modules/planning/master-plan.md`
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: all completed planning/config/backend-contract production and tests
outside the two authorized planning production paths and one new test; every build file; current
architecture tests; compile workflow guide; other architecture explanations and ADRs; compiler,
runtime, prepare, engine, and concrete backend placeholders/implementations; backend-conformance
and integration tests; Public Tensor APIs; trace Java/tests; and every later task row/specification.

## Maximum scope

Exactly the fourteen paths listed above. Stop if implementation requires another production
type, test, document, package, public API, dependency/build edit, architecture test, architecture
change, candidate/score/owner/device/diagnostic contract, provider implementation, lifecycle
behavior, or detailed later task specification. Do not use a follow-up task to hide an incomplete
acceptance criterion.

## Required Javadoc contracts

- Record Javadoc must define this value as one package-private per-query result, explain the exact
  provider-order eligible `BackendId` list, and state that it retains no provider, snapshot,
  requirement, device, score, owner choice, or executable state.
- Record-component, canonical-constructor, and explicitly declared public component-accessor
  Javadocs must document
  non-null inputs/results, encounter-order null/duplicate validation, exact reference retention,
  immutable membership snapshot, ordinary record semantics, and every exact failure message.
- Factory Javadoc must document all four inputs, their ownership and retention behavior, the full
  structural-validation-before-provider-call sequence, availability/requirement short-circuiting,
  exactly-once calls for still-possible providers, matching by `BackendId.equals`, provider order,
  snapshot-order irrelevance, every requirement subtype, empty/no-match meaning, and propagated
  queried-provider failures.
- Factory `@return` must state that the result is new and non-null, retains the exact query,
  snapshots the exact eligible provider identity references in provider order, and may contain an
  empty list.
- Package Javadoc must distinguish the two public query/provider contracts from the package-
  private eligibility step and keep scoring, ownership, public orchestration, partitioning,
  device-level capability/selection, preparation, runtime, and execution planned.

## Acceptance criteria

- `BackendEligibility` is a package-private final record in the exact capability package with
  exactly `OperationCapabilityQuery query` and `List<BackendId> eligibleBackendIds` components in
  that order, one package-private canonical constructor, explicitly declared public component
  accessors on the package-private record, ordinary record object methods, and one added package-private static
  `evaluate` factory with the exact signature in Scope.
- It has no public constructor, field, nested type, interface, builder, registry, service,
  callback, public matrix, or externally constructible/evaluable surface. Its only declared public
  methods are the two Java-required component accessors and ordinary record object methods; the
  package-private enclosing type keeps them outside the externally accessible planning API.
  Existing public planning API remains exactly `OperationCapabilityQuery` and
  `BackendCapabilityProvider` plus their ordinary methods.
- Direct result construction follows the exact query/list/element/duplicate/snapshot validation
  sequence and messages, retains the exact query and identity references, and exposes an
  immutable ordered membership snapshot.
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
  A `false` result excludes that backend without a rejection reason.
- An unconstrained intent accepts exactly supported backends with non-empty snapshots.
  `BackendIdRequirement`, `BackendDeviceIdRequirement`, and `DeviceClassRequirement` follow the
  exact matching semantics in Scope.
- Exact-device and class matching reads only immutable snapshot membership. It neither claims
  device-level operation support nor selects, retains, or returns a matching device.
- Empty snapshots exclude their backend. Missing snapshot/provider entries are invalid supplied
  composition and fail; they are not treated as ordinary unavailable/unsupported candidates.
- Valid zero-match evaluations return an immutable empty list. Hard requirements are never
  weakened, and no default/fallback backend becomes eligible.
- Result state contains only the query and eligible backend IDs. No provider, snapshot,
  `BackendIntent`, `BackendRequirement`, `BackendDeviceId`, `DeviceClass`, capability boolean,
  score, candidate, owner, route, or runtime object is retained.
- Focused automated tests lock the exact package-private generic record/factory shape, complete
  validation and evaluation order, exact failures, identity/reference/list semantics, all
  requirement cases, no-match representation, and forbidden surfaces. No recurring reflection,
  bytecode, or import invariant is left as a manual implementation check.
- Planning build dependencies and visibility remain unchanged: model and backend-contract stay
  `api`; config and trace stay `implementation`; no architecture-test or root-build file changes.
- Package Javadoc and the six explanatory documents mark per-query hard eligibility current only
  after implementation, explicitly state that it is internal, and leave public planner
  orchestration, reusable matrix APIs, scoring, profiles, ownership, partitioning, device-level
  capability/selection, compiler integration, preparation, runtime, and execution planned.
- Planning master plan marks only task 0002 Ready before implementation. Planning 0001 and Config
  0001–0003 remain Complete; Planning 0003+, Config 0004+, and Trace 0003+ remain Draft without
  detailed specifications.
- A separate clean-context documentation-focused pass finalizes the new/package Javadocs and all
  eleven documentation/planning paths in the same overall change. It reuses successful Java
  evidence unless executable Java changes or a concrete reason is recorded.
- Exactly one final planning module suite, planning Javadoc, repository Markdown, generated-page
  inspection for the remaining public package/types, exact fourteen-path scope, unchanged
  dependency/status/later-spec checks, final newlines, trailing whitespace, and
  `git diff --check` validation pass.
- No architecture test or repository-wide suite is run by habit: the task adds one internal
  planning evaluation with no dependency, build, public API, architecture boundary, concrete
  backend behavior, or cross-module executable change. Continuous integration or the next
  recorded capability checkpoint owns that tier.

## Tests / validation

Focused development commands are allowed while executable Java changes. After executable Java
stabilizes, run exactly one final planning module suite:

```bash
./gradlew :modules:planning:test
```

Record suite and test counts from XML reports. Then hand this task, the actual diff, and exact Java
evidence to the separate clean-context documentation-focused pass. That pass must not rerun the
successful Java suite unless it changes executable behavior or records a concrete cross-check
risk. After final Javadoc and Markdown edits, run:

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
authorized paths, unchanged planning dependencies, exactly one global Ready task during
implementation, no detailed Planning 0003, Config 0004, or Trace 0003 specification, balanced
fences, valid links/anchors, final newlines, no trailing whitespace, and the final diff.

Do not run architecture tests or the repository-wide suite unless implementation actually changes
a dependency/public surface or reveals another concrete repository-wide risk. Such a change is
outside this task and requires stopping rather than silently expanding validation and scope.

## Documentation handoff

After the implementation context records the final planning test result, hand this specification,
the actual fourteen-path diff, and exact Java evidence to a separate clean-context documentation-
focused agent or thread. The handoff must identify the package-private result/factory, exact
matching and ordering rules, BackendId-only output, unchanged public/dependency surface, the six
expected explanatory documents, and the validation commands above.

That pass must read the architecture contract, documentation rules, General, API/Javadoc,
Architecture, Backend guide, User guide, Planning, and Example profiles, final source/tests,
current backend-contract/config/planning contracts, and directly affected documents. It must
independently finalize record/component/constructor/factory/package Javadocs, examples,
current-versus-planned wording, links, terminology, and glossary impact.

It must record reasoned no-change conclusions for planning Gradle/dependencies, architecture
rules/ADRs/tests, compile workflow beyond the focused status pages, backend-conformance and
integration tests, backend-contract/config/trace Java, compiler/prepare/runtime/engine, concrete
backends, and every other module. It reuses the implementation context's successful Java evidence
unless executable behavior changes.

## Dependencies

- Complete Planning task 0001 with stable immutable `OperationCapabilityQuery` and explicitly
  supplied backend-level `BackendCapabilityProvider` contracts.
- Complete Backend-contract tasks 0001–0004 with stable `BackendId`, `BackendDeviceId`,
  `DeviceClass`, `BackendAvailabilitySnapshot`, and exhaustive hard-requirement vocabulary.
- Complete Config task 0001 with stable `BackendIntent` optionality and exact hard-requirement
  retention.
- Complete Config task 0003 as sequencing context: its optional device-class preference is soft
  input for later scoring and is deliberately not consumed here.
- Complete Trace tasks 0001–0002 as diagnostic foundation only; this task adds no trace schema.

## Follow-up tasks

- After task 0002 completes, separately reassess Config task 0004 profile contracts as the likely
  next area before Planning task 0003 scoring. Do not make either task Ready automatically.
- Planning task 0003 remains Draft and will define the smallest ownership-candidate/scoring
  contract only after hard eligibility and required profile inputs are stable. It must reassess
  the cross-package/public facade for consuming `BackendEligibility` rather than assuming this
  task's internal entry point becomes public.
- Planning tasks 0004–0006 retain same-owner partitioning, logical materialization/memory
  requirements, and planning closure as Draft rows without detailed specifications.
- Config tasks 0004–0008 and Trace tasks 0003–0008 remain Draft without detailed specifications.
- Device-level operation capability, chosen-device planning, structured rejection diagnostics,
  and public compiler failure contracts remain explicit limitations, not incomplete criteria.

## Architecture impact

Expected impact: None.

The task realizes existing planning ownership of capability analysis and hard eligibility while
preserving BackendId-level ownership, current dependencies, and the provider/snapshot/config
boundaries. It adds no public API or architecture-test need. If implementation reveals that a
public config-exposing entry point, dependency visibility change, device-level capability, chosen
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
  unsupported rows, reusable intent application, or public snapshot facts.
- Keep evaluation and result in one package-private record. This avoids a generic evaluator
  abstraction and avoids exposing config through a public Planning signature before a public
  planner consumer exists.
- Retain only the exact provider-returned `BackendId` references. These are the capability and
  ownership domain; equal snapshot identities validate supplied composition but do not replace
  the provider identity reference in the result.
- Treat the provider list as the canonical deterministic backend order. Snapshot maps explicitly
  promise no iteration order, and snapshot-list order is only validation order for snapshot-side
  failures.
- Validate the complete supplied composition before any capability call. Missing or duplicate
  identities are caller composition errors, not unsupported/unavailable candidates, and provider
  side effects must not begin before those errors are known.
- Apply availability and hard intent before calling capability providers. These immutable checks
  are cheaper, avoid irrelevant provider failures and per-occurrence work, and make a reusable
  all-backend matrix unnecessary. Every provider that can still qualify is called exactly once in
  provider order.
- Treat a backend as currently available only when its snapshot contains at least one device.
  The snapshot has no separate backend status, so an empty device map cannot support ownership.
- Exact-device and class requirements prove only matching availability under one backend. They do
  not narrow the existing backend-level capability answer or select the matching device.
- Represent no match as an empty immutable list. A dedicated exception would create a public or
  internal failure contract before compiler/planning orchestration exists. Later orchestration
  must turn emptiness into failure before scoring and may not weaken the requirement.
- Keep config and trace dependencies at `implementation`. The new type is package-private and no
  public signature exposes `BackendIntent` or trace vocabulary.
- Declare the two record component accessors `public`. The earlier package-private-accessor
  wording was not implementable: Java requires a record component accessor to be public, and the
  compiler rejected both narrower declarations with `invalid accessor method in record
  BackendEligibility (accessor method must be public)`. The enclosing record remains package-
  private, so its effective external API boundary is unchanged.

## Known limitations

- Capability remains backend-level. The result cannot prove that any specific available device
  supports the operation.
- Exact-device and device-class requirements do not identify or retain which matching device
  made the backend eligible.
- The result contains no rejection reason and cannot distinguish unsupported, unavailable, or
  hard-requirement-mismatched backends after evaluation.
- The result is internal to the capability package. No current compiler or public planning facade
  invokes it; Planning task 0003 must reassess the public/cross-package handoff.
- An empty result has no dedicated exception yet. Later orchestration owns the public compile
  failure type/message but must treat emptiness as terminal before scoring.
- Availability is only as current as the caller-supplied snapshots and has no refresh, timestamp,
  liveness, or registration guarantee.
- No serialization or external compatibility guarantee is established.

## Validation evidence

- Planning context `/root/plan_planning_0002` read the architecture contract; focused lifecycle,
  module, dependency, partition-scoring, and prepare/backend explanations; documentation rules
  and General/Planning profiles; planning guide and roadmap; planning/config/backend-contract/
  trace master plans; completed Planning 0001, Config 0001–0003, and Backend-contract 0001–0004
  tasks; and current production, focused tests, builds, package Javadocs, API/user/backend guides,
  and glossary sections before selecting this contract.
- The current consumers justify only a per-query internal eligibility result. A public reusable
  matrix would expose unsupported rows and supplied snapshots without a consumer, while a public
  evaluator accepting `BackendIntent` would require an otherwise unnecessary config `api`
  visibility change. The selected package-private record preserves the architecture and current
  dependency surface.
- The exact BackendId/device/class boundary is resolved: every final candidate is a provider-owned
  `BackendId`; exact-device/class requirements consult only the matching immutable snapshot for
  existence and never infer device-level capability or retain a chosen device.
- `python3 /tmp/validate_synaptik_markdown.py` passed for 224 Markdown files, 3,950 local links,
  238 local anchors, 2,822 fence markers, final newlines, and trailing whitespace.
- Planning-time `git diff --check` passed with no output. The combined tracked/untracked scope is
  exactly the five authorized planning paths: this task, the planning/config/trace master plans,
  and the roadmap. No Java, test, Gradle, architecture, API, guide, or glossary file changed
  during planning.
- Status and task-directory checks found exactly one global Ready master-plan row and exactly one
  Ready detailed task: Planning 0002. Planning 0001 and Config 0001–0003 remain Complete;
  Planning 0003+, Config 0004+, and Trace 0003+ remain Draft. No detailed Planning 0003, Config
  0004, or Trace 0003 specification exists.
- The original package-private accessor requirement was corrected before final validation. Java
  record component accessors must be public; attempting the narrower declarations produced
  `invalid accessor method in record BackendEligibility (accessor method must be public)` for both
  accessors. Explicit public accessors on a package-private top-level record preserve the intended
  external API boundary.
- Implementation context `/root/implement_planning_0002` ran
  `./gradlew :modules:planning:test --tests io.github.pho001.synaptik.planning.capability.BackendEligibilityTest`;
  it passed 15 tests with no failures, errors, or skips after the accessor correction. An earlier
  focused compile failed only because the original specification required impossible package-
  private record accessors; no behavioral test failure was involved.
- The same implementation context then ran the single final
  `./gradlew :modules:planning:test`; it passed 25 tests across two suites with no failures,
  errors, or skips. Documentation context
  `/root/implement_planning_0002/planning_0002_docs` reused this evidence because it changed no
  executable Java behavior and identified no concrete cross-check risk requiring duplicate tests.
- Documentation context `/root/implement_planning_0002/planning_0002_docs` applied the General,
  API/Javadoc, Architecture, Backend guide, User guide, Planning, and Example profiles. It read
  the authoritative architecture and planning contracts, final source and tests, directly related
  backend/config/planning contracts, and every affected document before finalizing record,
  component, constructor, factory, accessor, package, explanatory, glossary, and planning text.
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
- Planning dependency declarations remain exactly `api` for model and backend-contract and
  `implementation` for config and trace. No Gradle file, dependency visibility, root build,
  Java-version setting, architecture rule, ADR, or architecture test changed because the new
  result/factory is package-private and adds no module edge or public config-exposing signature.
- Compile workflow documentation beyond the authorized focused Compile API status page, compiler,
  prepare, runtime, engine, concrete backends, backend-contract/config/trace Java and tests, and
  every other module remain unchanged because this task adds only an internal compile-time
  eligibility fact and no lifecycle consumer, provider implementation, backend behavior, or
  execution path. Backend-conformance and integration tests therefore need no change.
- Final status checks found zero global Ready master-plan rows or detailed tasks. Planning tasks
  0001 and 0002 and Config tasks 0001–0003 are Complete; Planning 0003+, Config 0004+, and Trace
  0003+ remain Draft. No detailed Planning 0003, Config 0004, or Trace 0003 specification exists.
- Final exact-scope, status, later-spec, dependency, generated-page, newline, trailing-whitespace,
  fence, link, anchor, and `git diff --check` validation passed. Architecture and repository-wide
  tests were not run because no dependency, public surface, architecture boundary, concrete
  backend behavior, or cross-module executable contract changed.

## Implementation notes

- Added the package-private `BackendEligibility` record and factory with the exact two-component
  state, validation order, provider/snapshot equality association, availability and hard-intent
  short-circuiting, provider-order evaluation, `BackendId`-only result, and immutable empty
  no-match representation.
- Explicitly declared `query()` and `eligibleBackendIds()` public because Java record component
  accessors cannot reduce visibility. The initial narrower declarations failed compilation with
  `invalid accessor method in record BackendEligibility (accessor method must be public)`; the
  top-level record, canonical constructor, and static factory remain package-private.

## Completion summary

- Completed changes: added internal per-query backend hard eligibility with complete structural
  validation, exact availability/requirement matching, provider-order backend-level capability
  calls, and immutable `BackendId`-only results; finalized the public/internal/planned boundary.
- Files changed or created: `BackendEligibility.java`, capability `package-info.java`,
  `BackendEligibilityTest.java`, the six authorized explanatory documents, this task, the
  planning/config/trace master plans, and the roadmap; exactly fourteen paths.
- Tests and validation: reused the passing 15-test focused result and 25-test/two-suite final
  planning result; planning Javadoc, generated-page inspection, repository Markdown validation,
  exact-scope/dependency/status/later-spec checks, and `git diff --check` passed.
- Documentation-agent review: clean context
  `/root/implement_planning_0002/planning_0002_docs` finalized all affected Javadocs,
  explanations, glossary terminology, and planning evidence without changing executable behavior
  or rerunning Java tests.
- Documentation impact: the current internal eligibility step and its BackendId/device boundary
  are documented; reusable/public matrices, orchestration, scoring/profiles/ownership/partitioning,
  compiler integration, device-level capability/selection, preparation, runtime, and execution
  remain explicitly planned.
- Javadoc review: record, components, package-private canonical constructor/factory, Java-required
  public component accessors on the package-private record, and package boundaries are complete.
- Glossary impact: added the reusable `Backend hard eligibility` distinction and updated capability,
  intent, requirement, and implementation-status boundaries.
- Unresolved issues: None.
- Follow-up required: None; a separate frontier reassessment may select Config 0004, but no next
  task was made Ready.

Status: Complete
