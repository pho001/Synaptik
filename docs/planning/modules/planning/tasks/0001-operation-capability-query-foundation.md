# Task 0001: Operation Capability-Query Foundation

## Status

Complete

## Goal

Open `modules/planning` with the smallest immutable question that planning can ask a named backend
about one operation occurrence before configuration exposes partition-scoring weights or policy.

Mental model:

```text
Operation + ordered input descriptors + ordered output descriptors
  -> OperationCapabilityQuery
  -> BackendCapabilityProvider.supports(query)
  -> semantic ownership eligibility for that provider's BackendId
```

The query describes one structurally valid occurrence. The provider returns only a deterministic
boolean capability answer. Neither contract discovers a backend, evaluates availability or hard
requirements, scores candidates, selects a route, or creates executable state.

## Scope

- Delete the planning placeholder and add public record `OperationCapabilityQuery` in
  `io.github.pho001.synaptik.planning.capability` with exactly these ordered components:
  `Operation operation`, `List<TensorDescriptor> inputs`, and
  `List<TensorDescriptor> outputs`.
- Implement the canonical-constructor validation and snapshot sequence exactly:
  1. reject a null `operation` with `NullPointerException` message `operation`;
  2. reject a null `inputs` list with `NullPointerException` message `inputs`;
  3. reject a null `outputs` list with `NullPointerException` message `outputs`;
  4. scan input elements in encounter order and reject the first null with
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
  requires every valid occurrence to have a positive output count.
- Add public interface `BackendCapabilityProvider` in the same package with exactly these two
  abstract public methods:
  - `BackendId backendId()` returns a stable, non-null backend identity; and
  - `boolean supports(OperationCapabilityQuery query)` answers only whether that named backend can
    semantically own the immutable operation occurrence.
- Specify that every provider implementation rejects a null query with `NullPointerException`
  message `query`. For an immutable query and immutable provider configuration, `supports` is
  deterministic.
- Treat the provider as an explicitly supplied compile-time collaboration. It is not a service
  locator and performs no registry or discovery lookup.
- Add no production provider implementation. Concrete backends may later implement this
  inward-facing planning contract through their architecture-approved dependency on planning.
- Add package Javadoc defining capability as semantic compile-time ownership eligibility and
  distinguishing it from availability, requirement evaluation, scoring, preparation, routing,
  and execution.
- Require meaningful type, canonical-constructor, method, record-component, and package Javadocs.
  All record-component accessors must be explicitly declared and documented; the interface
  methods must fully document stability, nullability, determinism, and their narrow result
  semantics.
- Change only model and backend-contract dependency visibility in
  `modules/planning/build.gradle.kts` from `implementation` to `api`, because public planning
  signatures expose `Operation`, `TensorDescriptor`, and `BackendId`. Keep config and trace as
  `implementation` dependencies.
- Add focused contract tests for exact API shape, exact validation order and messages, immutable
  ordered snapshots, reference retention, value semantics, accepted cardinalities, signature
  failure propagation, provider identity/support semantics, and forbidden added surfaces.
- Add one focused architecture test for the exact four planning project dependencies, exact
  `api` visibility of model and backend-contract, exact `implementation` visibility of config and
  trace, and absence of concrete-backend, runtime, prepare, and engine edges.
- Finalize the affected API, backend, user, glossary, architecture-status, and planning documents
  in the same overall change.

## Out of scope

- capability matrix, candidate collection, candidate stored type, hard-eligibility result, or
  ownership candidate model
- result enum or record, rejection reason, structured rejection diagnostics, exception taxonomy,
  or explanation payload
- device-level capability query or `BackendDeviceId` selection
- `BackendAvailabilitySnapshot` evaluation, `BackendIntent` evaluation, or
  `BackendRequirement` matching
- preference, ranking, scoring, profiles, weights, bonuses, penalties, estimates, or calibrated
  data
- node or segment ownership, partition construction, same-owner grouping, transfer planning,
  materialization planning, or logical memory planning
- provider registry, discovery, classpath scanning, reflection, `ServiceLoader`, service locator,
  factory, or engine registration
- a concrete backend, provider implementation, fallback provider, route, kernel, lowering, or
  backend conformance behavior
- compiler orchestration, graph closure, graph/node/value identifiers, graph phase, prepare,
  runtime, engine, execution, storage, physical buffers, schedules, or mutable state
- semantic data-type, shape, layout, gradient, operand-role, descriptor-compatibility, or graph-
  closure validation beyond the existing occurrence-count signature check
- trace DTOs, attributes, payloads, emission, or serialization
- serialization or external schema for either new contract
- dependencies or build changes beyond the two visibility changes in the planning build and their
  focused architecture test
- architecture-rule changes, ADR changes, Java-version changes, root build changes, another
  module's executable behavior, unrelated refactoring, or a detailed specification for any later
  task

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
- [Backend guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Planning master plan](../master-plan.md)
- [Config master plan](../../config/master-plan.md)
- [Trace master plan](../../trace/master-plan.md)
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Planning owns backend-neutral capability query contracts and asks where an operation occurrence
  may run. It must not select a concrete kernel, executable, BLAS route, MPSGraph route, CUDA
  implementation, or other backend route.
- `Operation` continues to own backend-independent semantics and exposes no backend support.
  Backend capability is expressed through the planning-owned provider contract.
- Compile-time plans retain `BackendId`, not provider objects or live backend services. This task
  creates no plan or retained provider association.
- A concrete backend may later depend inward on planning and implement the provider interface.
  Planning must not depend outward on a concrete backend.
- Capability is distinct from supplied availability, hard-requirement eligibility, candidate
  scoring, registration, preparation, and execution.
- Public planning signatures exposing model and backend-contract types require `api` visibility.
  Existing config and trace edges remain internal implementation dependencies.
- Planning must remain independent of runtime, prepare, engine, and concrete backends. The focused
  architecture test enforces the current exact dependency surface without changing the
  architecture contract.
- Stop if implementation needs a capability result model, device identity, registry, consumer,
  concrete implementation, extra dependency, additional production type, architecture rule, or
  public API beyond the exact two contracts and package documentation.

## Package impact

Package added:

- `io.github.pho001.synaptik.planning.capability` — public backend-neutral compile-time contracts
  for asking whether one named backend can semantically own one immutable operation occurrence

Type placement:

- `io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery` — planning owns the
  immutable operation-occurrence question used during capability analysis
- `io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider` — planning owns the
  inward-facing collaboration that concrete backends may later implement

The root placeholder `io.github.pho001.synaptik.planning.PlanningModule` is deleted. No generic
root facade, `util`, candidate, matrix, scoring, registry, service, discovery, provider-
implementation, runtime, or backend-specific package is added.

## Affected files

The intended implementation scope is exactly eighteen paths.

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

Planning — exactly five paths:

- add and finalize this task
- `docs/planning/modules/planning/master-plan.md`
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: `Operation`, `OperationSignature`, `TensorDescriptor`, `BackendId`,
their tests and Javadocs, current config and trace Java contracts, compiler/runtime/prepare/engine
placeholders, all concrete backends, `BackendAvailabilitySnapshot`, `BackendIntent`, backend
requirements, `AGENTS.md`, `ARCHITECTURE.md`, other architecture explanations, ADRs, root build
and settings, other module build files, backend-conformance tests, integration tests, and all
later task rows.

## Maximum scope

Exactly the eighteen paths listed above. Stop if implementation requires another production
type, test, document, dependency, package, module API, architecture change, provider
implementation, consumer behavior, or detailed follow-up specification. Do not use a later task
to hide an incomplete acceptance criterion from this task.

## Acceptance criteria

- `OperationCapabilityQuery` is a public record with exactly the ordered components
  `Operation operation`, `List<TensorDescriptor> inputs`, and `List<TensorDescriptor> outputs`, a
  public canonical constructor, explicitly documented component accessors, ordinary record object
  methods, and no other project field, method, constructor, interface, factory, builder, or nested
  type.
- The constructor performs the exact null, encounter-order element scan, snapshot, and final
  occurrence-count validation sequence specified in Scope. Every null failure has the exact
  required message, and an input failure precedes any output-element scan or signature call.
- Construction retains the exact operation and descriptor references while isolating ordered list
  membership from later caller mutation. Returned lists are immutable snapshots.
- Empty and repeated inputs are accepted when permitted by the signature. Repeated output
  descriptor references and signature-valid multi-output occurrences are accepted. Zero outputs
  fail through `OperationSignature.validateOccurrence` because valid signatures require positive
  output counts.
- The query performs no data-type, shape, layout, gradient, operand-role, graph, identifier,
  phase, backend, device, availability, config, estimate, capability, score, route, prepare,
  runtime, or execution validation.
- `BackendCapabilityProvider` is public and declares exactly `backendId()` and `supports(query)` as
  abstract public methods. It has no fields, constants, default/static/private methods, nested
  types, factory, registry, discovery, lookup, route, kernel, prepare, runtime, or execution
  surface.
- `backendId()` is documented and tested as stable and non-null. `supports` answers only semantic
  ownership capability for that provider's backend, rejects null with exact message `query` in
  implementations, and is deterministic for an immutable query and immutable provider
  configuration.
- `supports` does not report backend registration or availability, evaluate a hard requirement,
  score candidates, select a kernel or route, inspect prepare/runtime state, or promise successful
  execution.
- The provider is supplied explicitly to later compile-time planning; it is not obtained through
  a service locator, registry, discovery scan, or `ServiceLoader` lookup.
- No production provider implementation is added. Documentation explains that concrete backends
  may later implement this inward-facing planning contract.
- The planning placeholder is removed and package Javadoc distinguishes the two current contracts
  from all deferred planning and backend behavior.
- The planning build declares model and backend-contract with `api`, config and trace with
  `implementation`, and no other project dependency. The focused architecture test locks exact
  visibility and forbids concrete-backend, runtime, prepare, and engine edges.
- Existing model, backend-contract, config, and trace declarations and behavior remain unchanged.
- The six explanatory documents mark the query/provider contract current only after
  implementation while leaving matrices, eligibility, scoring, ownership, partitioning,
  provider implementations, compiler/prepare/runtime/engine behavior, diagnostics, and device-
  level capability planned.
- Planning master plan records task 0001 as the sole Ready frontier before implementation and
  tasks 0002–0006 as ordered Draft rows without detailed specifications. Config tasks 0001–0002
  and trace tasks 0001–0002 remain Complete; their later rows remain Draft without detailed
  specifications.
- A separate clean-context documentation-focused pass finalizes every affected Javadoc and all
  eleven documentation/planning paths in the same overall change without repeating successful
  Java tests unless executable Java changes or a concrete cross-check risk is recorded.
- The final combined planning/architecture-test suite, planning Javadoc, repository Markdown,
  exact eighteen-path scope, synchronized statuses, absence of later detailed specs, final
  newlines, trailing whitespace, `git diff --check`, and single final root suite pass.

## Tests / validation

Focused development commands are allowed while Java is changing. After executable Java and the
build visibility changes stabilize, run exactly one final combined affected-suite command:

```bash
./gradlew :modules:planning:test :testing:architecture-tests:test
```

Record suite and test counts from XML reports. Then hand the actual combined diff and exact Java
evidence to the separate documentation-focused context described below. That context reuses the
successful Java evidence and does not rerun either Java suite unless it changes executable Java
behavior or records a concrete reason. After its final Javadoc and Markdown edits, it runs:

```bash
./gradlew :modules:planning:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

Inspect the generated package and both type pages. Confirm exactly the eighteen authorized paths,
the exact dependency declarations, one global `Ready` task during implementation, no detailed
config 0003, trace 0003, or planning 0002 specification, synchronized current-versus-planned
wording, balanced fences, final newlines, and no trailing whitespace.

Because public dependency visibility and architecture-test coverage change, after the complete
implementation/documentation diff stabilizes run exactly one final repository suite:

```bash
./gradlew test
```

Record its task, suite, and test result. Do not repeat a successful root run in the documentation
or coordination context.

## Documentation handoff

After the implementation context records the final combined planning/architecture-test result,
hand this task, the actual diff, and exact test evidence to a separate clean-context
documentation-focused agent or thread. It must read the architecture contract, documentation
rules, General, API/Javadoc, Backend guide, User guide, Planning, and Example profiles, final
source/tests/build changes, and the directly affected documents.

That pass must independently finalize both new type Javadocs, explicit record accessors, package
Javadoc, and all affected explanatory and planning text. It must review glossary impact and record
reasoned no-change conclusions for Compile API consumers beyond the documented surface, current
config/trace Java, architecture rules and tests beyond the new focused dependency test,
backend-conformance and integration tests, Gradle outside planning, and all other modules. It
reuses successful Java evidence unless it changes executable Java behavior.

## Dependencies

- Completed model capability milestone, including stable `Operation`, `OperationSignature`, and
  `TensorDescriptor` contracts.
- Completed backend-contract tasks 0001–0004, including stable `BackendId`.
- Completed trace tasks 0001–0002 as the diagnostic foundation; this task adds no trace contract.
- Completed config tasks 0001–0002. Planning task 0001 intentionally interleaves before config
  task 0003 so the typed capability question is stable before scoring weights and policy are
  exposed.

## Follow-up tasks

- Planning task 0002 will define capability-matrix construction and hard eligibility after this
  contract is complete; it remains Draft without a detailed specification.
- Planning task 0003 will define ownership candidates and backend-neutral scoring after its
  config prerequisites are stable; it remains Draft without a detailed specification.
- Planning tasks 0004–0006 retain same-owner partitioning, logical materialization/memory
  requirements, and planning closure as Draft rows without detailed specifications.
- After this task completes, the likely active frontier returns to config task 0003 for
  partition-scoring configuration before planning scoring proceeds.
- Boolean rejection diagnostics and device-level capability queries are deliberately deferred
  limitations, not incomplete acceptance criteria for this foundation.

## Architecture impact

Expected impact: None.

The task realizes existing planning capability ownership and existing concrete-backend inward
dependency permission. Changing model and backend-contract to public dependency visibility is a
build/API realization of the existing direction, not a new architecture rule. The focused
architecture test protects that direction. If implementation reveals a need to change module
ownership, allowed dependencies, provider direction, lifecycle rules, or service-discovery
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
  contract before its consumers and trace vocabulary are stable.
- The query represents an occurrence by immutable semantic and descriptor facts rather than node
  or value identity. Capability does not need graph identity, graph closure, or live graph state.
- Output descriptors are explicit and may repeat by reference because capability can depend on
  every logical output of a multi-output occurrence; the query does not infer outputs.
- Model and backend-contract are public dependencies because they appear in public signatures;
  config and trace remain internal dependencies until a public planning signature actually
  exposes them.
- The config interleave pauses before scoring configuration only long enough to stabilize the
  typed capability question. It does not authorize planning scoring before config task 0003.

## Known limitations

- A boolean rejection carries no reason or structured diagnostic. A later task may add a separate
  result contract only when a concrete consumer and trace boundary are ready.
- Capability is backend-level only. Device-specific support and device selection remain deferred.
- The provider contract does not prove availability, registration, requirement satisfaction,
  preparation success, route existence, resource access, or executable success.
- This task supplies no provider implementation, capability matrix, eligible-candidate model, or
  planning consumer.

## Validation evidence

- Implementation context `/root/implement_planning_0001` ran
  `./gradlew :modules:planning:test :testing:architecture-tests:test` after executable Java and
  dependency visibility stabilized: `BUILD SUCCESSFUL`; 13 tests across four suites, with 0
  failures, 0 errors, and 0 skips. `BackendCapabilityContractTest` contributed 10 tests;
  `ConfigDependencyContractTest`, `NnTrainingDependencyContractTest`, and
  `PlanningDependencyContractTest` contributed one test each.
- Documentation context `/root/implement_planning_0001/planning_0001_docs` independently read the
  architecture contract, documentation rules and selected Architecture, API/Javadoc, Backend
  guide, User guide, Planning, and Example profiles, planning guide and roadmap, three affected
  master plans, this task, final source/tests/build, directly affected documentation, and the
  relevant model/backend-contract contracts. It changed no executable Java behavior and reused
  the successful 13-test evidence without rerunning Java tests.
- The documentation context ran `./gradlew :modules:planning:javadoc`: `BUILD SUCCESSFUL`; six
  actionable tasks, two executed and four up-to-date. It inspected generated
  `package-summary.html`, `OperationCapabilityQuery.html`, and
  `BackendCapabilityProvider.html`, including the package boundary, record signature, constructor
  failures, immutable-list accessors, stable provider identity, narrow boolean result, and null-
  query obligation.
- The documentation context ran `python3 /tmp/validate_synaptik_markdown.py`: passed for 222
  Markdown files, 3,885 local links, 235 local anchors, 2,800 fence markers, and final-newline/
  trailing-whitespace validation.
- The documentation context ran `git diff --check`: passed with no output. The combined tracked
  and untracked path audit returned exactly the eighteen authorized paths, and `git status
  --short` showed no path outside that set.
- Manual checks confirmed the exact planning dependencies (`api` model and backend-contract;
  `implementation` config and trace), task 0001 and the master plan at Complete, config 0003
  as the likely next Draft frontier, later planning/trace rows still Draft, no detailed config
  0003, trace 0003, or planning 0002 task specification, balanced fences, final newlines, and no
  trailing whitespace.
- Implementation coordinator `/root/implement_planning_0001` ran the single final
  `./gradlew test` from the repository root after the combined implementation/documentation diff
  stabilized and without any intervening executable change: `BUILD SUCCESSFUL in 628ms`; 46
  actionable tasks, one executed and 45 up-to-date. Aggregated XML reports contain 1,079 tests
  across 141 suites, with 0 skipped, 0 failures, and 0 errors. This was the only final root-suite
  run.

## Implementation notes

- `OperationCapabilityQuery` is the exact three-component public record. Its compact constructor
  validates top-level references, scans and snapshots inputs, scans and snapshots outputs, then
  delegates count validation to the retained operation signature. The exact operation and
  descriptor references are retained; only list membership is copied.
- `BackendCapabilityProvider` declares only `backendId()` and `supports(query)`. Its Javadoc makes
  stable non-null identity, deterministic semantic ownership support, exact null-query failure,
  explicit supply, and the absence of discovery, availability, eligibility, scoring, route,
  prepare, and execution meanings explicit. No provider implementation was added.
- The planning placeholder was deleted. Model and backend-contract became public Gradle API
  dependencies because their types occur in public signatures; config and trace remain internal
  implementation dependencies. The focused architecture test locks that exact surface and the
  forbidden outward edges.
- The documentation pass marked only the query/provider contracts current. Capability matrices,
  hard eligibility, scoring, ownership, partitioning, provider implementations, compiler/prepare/
  runtime/engine behavior, diagnostics, and device-level capability remain planned.
- No in-scope adjustment was required. The documentation pass removed one draft Javadoc phrase
  that prohibited retaining the immutable query because neither the interface nor tested contract
  establishes that additional ownership restriction.
- Current config and trace Java required no change because neither consumes or exposes the new
  contracts. `ARCHITECTURE.md`, ADRs, and architecture explanations beyond the authorized
  current-status page required no change because the implementation realizes existing ownership
  and dependency rules. Architecture tests beyond the focused dependency test, backend-
  conformance tests, and integration tests required no change because no architecture rule,
  concrete backend behavior, preparation, or end-to-end behavior changed. Gradle outside planning
  and all other modules required no change because the only public-signature visibility adjustment
  is local to `modules/planning`. Compile API consumers beyond the documented current surface
  required no change because no compiler or planning consumer exists.

## Completion summary

- Completed changes: Implemented and documented the immutable operation-capability query and
  explicit provider collaboration, removed the placeholder, and exposed only the two public
  signature dependencies.
- Files changed or created: Exactly eighteen authorized paths: five production/build paths, two
  test paths, six explanatory documentation paths, and five planning paths.
- Tests and validation: Reused the implementation context's successful 13-test/four-suite combined
  run; planning Javadoc, generated-page inspection, repository Markdown, exact-scope, dependency,
  status, later-spec absence, final-newline, trailing-whitespace, and `git diff --check` validation
  passed in the documentation context. The implementation coordinator's single final root suite
  passed 1,079 tests across 141 suites with no skips, failures, or errors.
- Documentation-agent review: Clean context
  `/root/implement_planning_0001/planning_0001_docs` completed the required independent pass
  without changing executable behavior or repeating Java tests.
- Documentation impact: Finalized architecture-status, public/compile API, backend guide,
  backend-selection guide, glossary, task, three master plans, and roadmap wording while preserving
  current-versus-planned boundaries.
- Javadoc review: Finalized and generated the package and both public type pages; every parameter,
  non-void return, caller-visible failure, nullability, snapshot/reference-retention rule,
  deterministic result, and deliberate boundary is documented.
- Glossary impact: Added current operation-capability query and provider definitions and refined
  backend capability without changing architecture authority.
- Unresolved issues: None.
- Follow-up required: None. Config task 0003 is the likely next frontier but remains Draft without
  a detailed specification; making it Ready is a separate planning step.

Status: Complete
