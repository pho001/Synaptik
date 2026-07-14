# Task 0003: Ownership Candidates and Baseline Scoring

## Status

Complete

## Goal

Add the smallest planning-owned consumer of current per-query hard eligibility that compares the
remaining backend ownership alternatives with the current optional device-class preference and
deterministically returns one `BackendId` owner.

Mental model:

```text
BackendEligibility.eligibleBackendIds() in provider encounter order
  + the associated immutable availability snapshots
  + PartitionScoringConfig preferred DeviceClass, when present
  -> validate the selection inputs and eligible-to-snapshot associations
  -> compare only hard-eligible BackendId values
  -> first preferred-class match, otherwise first eligible BackendId
  -> one exact retained BackendId reference
```

The hard-eligible identities are already the complete candidate set. This baseline needs no
candidate record, numeric score, operation-family classification, workload bucket, cost profile,
or public planner facade. It selects backend ownership only; it does not select a device, route,
kernel, executable, partition, or runtime state.

## Motivation

Planning task 0002 deliberately represents a valid no-match outcome with an empty immutable list
and defers terminal ownership-selection failure to its first consumer. Config task 0003 supplies
one optional coarse `DeviceClass` preference but deliberately performs no evaluation. This task
connects those two current facts with one deterministic backend-neutral baseline before maximal
same-owner partitioning.

The discarded Config 0004 design attempted to define broad fixed and linear costs before a stable
consumer or workload classification existed. The current baseline consumes no cost quantity. It
therefore establishes the exact current selection seam without inventing placeholder operation
families, workload buckets, profile units, or backend tuning vocabulary.

## Scope

- Add package-private final class `BackendOwnerSelection` in
  `io.github.pho001.synaptik.planning.capability`.
- Give the class exactly one private no-argument constructor and no instance state, interface,
  nested type, enum, record component, builder, registry, service, or callback.
- Add exactly one package-private static method:

  ```java
  static BackendId select(
          BackendEligibility eligibility,
          PartitionScoringConfig scoringConfig,
          List<BackendAvailabilitySnapshot> availabilitySnapshots)
  ```

- Keep the selector in `planning.capability` for this task. `BackendEligibility` is deliberately
  package-private, and the current consumer is also internal. Putting the selector in the later
  `ownership` package would require widening the completed eligibility result or adding a public
  bridge with no external compiler consumer. The focused colocated handoff is smaller and keeps
  Planning's public surface unchanged. A later public compiler-planning orchestration task must
  reassess the external facade from its concrete aggregate consumer.
- Treat `eligibility.eligibleBackendIds()` as the complete ordered ownership-candidate set. Do not
  copy candidates into a production record, row, matrix, map, score result, or public collection.
- Use associated `BackendAvailabilitySnapshot` values only to determine whether an already
  eligible backend reports at least one device whose `DeviceClass` equals the optional preferred
  class. Do not re-evaluate support, availability, or the hard requirement and do not remove or
  restore any candidate.
- Validate top-level inputs in this exact order before reading an availability-snapshot element:
  1. null `eligibility` -> `NullPointerException("eligibility")`;
  2. null `scoringConfig` -> `NullPointerException("scoringConfig")`; and
  3. null `availabilitySnapshots` ->
     `NullPointerException("availabilitySnapshots")`.
- After top-level validation, treat an empty eligible list as terminal before reading any
  availability-snapshot element. Throw `IllegalStateException` with exact message:

  ```text
  no hard-eligible backend is available for ownership selection
  ```

  This is the exact current internal failure. It is not a public compiler exception and must not
  be translated into a default or fallback backend in this task.
- For a non-empty eligible list, scan the complete availability-snapshot list in caller encounter
  order:
  1. reject the first null snapshot at index `i` with `NullPointerException` message
     `availabilitySnapshots[i]`;
  2. use the snapshot's already non-null `backendId()` as its association key; and
  3. reject the second snapshot with an equal backend identity using
     `IllegalArgumentException` message
     `duplicate availability snapshot backendId: <backendId.value()>`.
- After the complete snapshot scan succeeds, walk eligible identities in their existing provider
  order and reject the first identity with no equal snapshot identity using
  `IllegalArgumentException` message
  `missing availability snapshot for backendId: <backendId.value()>`.
- Match eligible and snapshot identities through `BackendId.equals`, never by list position or
  object identity. A snapshot may hold an equal but non-identical `BackendId` reference.
- Permit extra unique snapshots for backends not present in the eligible list. Task 0002 receives
  one snapshot for every provider before filtering, so snapshots for unavailable, mismatched, or
  unsupported providers are expected input and must not become candidates.
- A matching snapshot with an empty device map is a valid nonmatch for the soft preference. The
  hard-eligibility result remains authoritative: selection must not remove that backend, and the
  backend may still win through the ordinary first-eligible fallback. Under the intended handoff,
  task 0002 cannot produce such a pairing from the same snapshots; defining it here prevents the
  selector from silently reapplying hard eligibility when directly constructed package-private
  test values are used.
- Apply the baseline comparison exactly:
  - when `preferredDeviceClass()` is empty, return the first eligible backend identity;
  - when a preferred class is present, return the first eligible identity whose associated
    snapshot contains at least one device of that exact enum class;
  - when no eligible backend matches the preferred class, return the first eligible identity; and
  - when multiple eligible backends match, the first in existing provider order wins.
- Treat preference matching as a two-bucket lexicographic comparison, not a numeric bonus,
  penalty, weight, or score. Matching candidates precede nonmatching candidates only for this
  baseline; provider order resolves every equality within a bucket.
- Preserve soft preference semantics: every hard-eligible nonmatch remains a candidate, and a
  preferred-class miss never weakens a hard requirement or restores an ineligible backend.
- Return the exact `BackendId` reference stored in `eligibility.eligibleBackendIds()`. Do not
  return the snapshot's equal identity reference and do not allocate or normalize an identity.
- Read only immutable compile-time facts. Do not call capability providers, discover backends,
  inspect mutable runtime state, allocate IDs, mutate any supplied value or list, emit a trace,
  or retain selection state.
- Add no production workload or cost classification. The current baseline requires only the
  existing `DeviceClass` preference. Shared `OperationFamily`, workload-bucket, cost-profile, and
  fixed/per-element/per-byte classifications remain deferred until a concrete backend-neutral
  cost consumer requires their exact distinctions.
- Update capability-package Javadoc to describe the current internal eligibility-to-baseline-
  owner step while preserving the public query/provider boundary.
- Add one focused test class covering exact selector shape, validation order/messages, empty
  terminal failure, association rules, reference and ordering rules, neutral/preferred/no-match/
  tie behavior, hard-eligibility preservation, absence of side effects, and forbidden surfaces.
- Finalize Javadocs and affected explanatory/planning documentation through the required separate
  clean-context documentation pass in the same overall implementation change.

## Out of scope

- a public planner facade, public ownership selector, public eligibility result/evaluator,
  public candidate or score type, public matrix, public ownership decision record, or compiler
  entry point
- changing `BackendEligibility`, `OperationCapabilityQuery`, `BackendCapabilityProvider`,
  `PartitionScoringConfig`, `BackendIntent`, `BackendAvailabilitySnapshot`, `DeviceClass`, or
  `BackendId` signatures, value behavior, visibility, or existing validation
- re-evaluating capability, availability, or hard requirements; weakening hard eligibility;
  restoring an ineligible backend; provider discovery, registration, refresh, or service lookup
- a selected `BackendDeviceId`, device candidate, default device, device-level capability,
  device ranking, device ownership, or retention of a device or class in the returned result
- numeric score values, score maps, callbacks, policy hierarchies, weights, bonuses, penalties,
  thresholds, transfer estimates, boundary estimates, materialization estimates, region-size
  estimates, or calibrated units
- a production `OperationFamily`, workload bucket, cost class, cost profile, platform profile,
  backend profile, tuning profile, measurement evidence, workload cache, model-autotuning value,
  or serialization schema
- concrete backend, route, Vector API species or lane, thread, chunk, tile, OpenBLAS, MPSGraph,
  CUDA, kernel, fusion, specialization, lowering, prepare, executable, schedule, buffer,
  workspace, residency, or run-state logic
- graph/node/value identity, graph phase, producer/consumer ownership, segment comparison,
  transfer boundaries, maximal same-owner partition construction, physical transfers, logical
  materialization, logical memory planning, compiler orchestration, or compile artifacts
- generic candidate registries, broad managers/facades, service locators, string dispatch,
  reflection-based dispatch, `ServiceLoader`, plugin discovery, or diagnostics matrices
- trace event or payload schema, rejection-reason taxonomy, public compile exception, backend
  conformance behavior, integration behavior, or end-to-end execution
- dependency, Gradle, Java-version, root-build, architecture-contract, ADR, architecture-test,
  backend-conformance-test, or integration-test changes
- Config 0004 or Planning 0004 detailed specifications, unrelated refactoring, or documentation
  cleanup outside the exact paths below

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [ADR 0008](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
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
- [Planning task 0002](0002-per-query-backend-hard-eligibility.md)
- [Config task 0001](../../config/tasks/0001-backend-intent-foundation.md)
- [Config task 0002](../../config/tasks/0002-compile-modes-and-graph-optimization-configuration.md)
- [Config task 0003](../../config/tasks/0003-partition-scoring-configuration.md)
- [Backend-contract task 0003](../../backend-contract/tasks/0003-backend-availability-snapshot.md)
- [Backend-contract task 0004](../../backend-contract/tasks/0004-declarative-backend-requirements.md)
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Capability-provider guide](../../../../backend-guide/capability-provider.md)
- [Backend-selection guide](../../../../user-guide/backend-selection.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Planning owns backend-neutral compile-time capability analysis, scoring, and `BackendId`
  ownership. This task answers only where one operation occurrence should run.
- Hard eligibility is complete before this task. Selection may compare only the identities in
  `BackendEligibility.eligibleBackendIds()` and may neither restore an excluded backend nor remove
  an eligible nonpreferred backend.
- Compile-time ownership uses `BackendId`, not a provider, concrete backend object, device,
  preparer, executable, route, kernel, or live service.
- `PartitionScoringConfig` supplies one soft coarse-class preference. A class match may influence
  ranking but cannot become device selection or device-level capability.
- Baseline comparison uses current immutable compile-time facts only. Runtime residency, buffers,
  prepared state, backend-local caches, model-autotuning values, and measurement evidence remain
  outside planning selection.
- Planning selects backend ownership, not implementation routes. ADR 0008's compiler/planning/
  prepare/backend/tuning/runtime ownership boundaries remain unchanged.
- The package-private selector preserves planning's existing `implementation` dependency on
  config. No public signature exposes `PartitionScoringConfig`, so no dependency-visibility or
  architecture-test change is justified.
- A workload/cost classification is not an architecture prerequisite for this exact baseline.
  Adding one without consumed cost data would be speculative and would incorrectly precede its
  consumer.
- Planning remains independent of runtime, prepare, engine, and concrete backends.
- Stop if implementation requires a public selector/result, another production type, numeric
  score, workload/cost classification, dependency edit, graph/partition model, device selection,
  diagnostics schema, or architecture decision.

## Current contract inventory

| Contract | Current meaning consumed by this task | Deliberate boundary |
|---|---|---|
| `OperationCapabilityQuery` | Immutable one-occurrence semantic question retained inside `BackendEligibility` | No graph identity, availability, score, route, or execution fact |
| `BackendCapabilityProvider` | Source of backend-level semantic support already consumed by task 0002 | Never called by baseline selection |
| `BackendEligibility` | Package-private immutable query plus unique provider-ordered hard-eligible `BackendId` list | Empty is valid before selection; no snapshot, device, score, or rejection reason retained |
| `BackendAvailabilitySnapshot` | Immutable association used transiently to test preferred `DeviceClass` presence | Not discovery, device capability, selected device, or retained selection state |
| `PartitionScoringConfig` | Optional soft preferred `DeviceClass` | No formula, weight, default, cost data, or ownership result |
| `BackendId` | Compile-time ownership identity and exact return type | Not a live backend service or concrete implementation route |

## Package impact

Existing package extended:

- `io.github.pho001.synaptik.planning.capability` — public operation query/provider contracts,
  package-private hard eligibility, and the smallest package-private baseline owner selection
  that can consume that eligibility without widening it

Type placement:

- `io.github.pho001.synaptik.planning.capability.BackendOwnerSelection` — package-private
  stateless selection operation colocated with the package-private eligibility result it consumes

No `ownership` package is opened in this task. That package remains reserved for later public or
cross-package ownership/candidate contracts justified by compiler orchestration, maximal
same-owner partitioning, or another concrete consumer. No workload, cost, profile, diagnostics,
registry, service, runtime, prepare, or backend-specific package is added.

## Proposed production surface

```java
package io.github.pho001.synaptik.planning.capability;

final class BackendOwnerSelection {
    private BackendOwnerSelection() {}

    static BackendId select(
            BackendEligibility eligibility,
            PartitionScoringConfig scoringConfig,
            List<BackendAvailabilitySnapshot> availabilitySnapshots) {
        // Exact behavior is specified above; implementation strategy is not a contract.
    }
}
```

This conceptual shape is exact for names, visibility, parameters, generic arguments, return type,
constructor count, and absence of state. It is not authorization to add a generic utility or a
second method.

## Behavioral contract

| Eligible IDs | Preference | Matching eligible snapshots | Result |
|---|---|---|---|
| empty | any valid config | not read | exact terminal `IllegalStateException` |
| `[cpu, metal]` | empty | irrelevant after association validation | exact `cpu` eligibility reference |
| `[cpu, metal]` | `ACCELERATOR` | only `metal` | exact `metal` eligibility reference |
| `[cpu, metal, cuda]` | `ACCELERATOR` | `metal`, `cuda` | exact `metal` eligibility reference by provider order |
| `[cpu, metal]` | `ACCELERATOR` | none | exact `cpu` eligibility reference; neither candidate was filtered |
| `[hybrid, metal]` | `ACCELERATOR` | both; hybrid snapshot also contains CPU | exact `hybrid` eligibility reference by provider order |

The table describes a stable lexicographic comparison: preferred-class match first when requested,
then existing provider order. It defines no numeric score or future cost combination rule.

## Validation and failure contract

| Stage | Condition | Failure |
|---|---|---|
| 1 | null `eligibility` | `NullPointerException("eligibility")` |
| 2 | null `scoringConfig` | `NullPointerException("scoringConfig")` |
| 3 | null `availabilitySnapshots` | `NullPointerException("availabilitySnapshots")` |
| 4 | no eligible backend | `IllegalStateException("no hard-eligible backend is available for ownership selection")` |
| 5 | first null snapshot at index `i` | `NullPointerException("availabilitySnapshots[i]")` |
| 5 | second snapshot with equal backend ID | `IllegalArgumentException("duplicate availability snapshot backendId: <value>")` |
| 6 | first eligible ID without an equal snapshot | `IllegalArgumentException("missing availability snapshot for backendId: <value>")` |

Top-level validation always completes before stage 4. Stage 4 occurs before any list-element read.
For a non-empty candidate set, the complete snapshot scan and duplicate validation finish before
the first missing-association check or preference comparison. Preference comparison begins only
after all eligible identities have a matching snapshot.

## Ordering, identity, immutability, and side effects

- Eligible/provider encounter order is the sole stable tie order.
- Snapshot list order never reorders candidates or changes a successful result.
- Equal `BackendId` values associate a candidate and snapshot; the selected return reference is
  always the eligibility element, never the snapshot component.
- The selector retains no list, map, snapshot, class, score, candidate, or chosen owner after the
  call and exposes no mutable state.
- It does not mutate supplied lists or snapshots, call a provider, allocate an ID, read a device
  handle, discover a backend, emit diagnostics, or perform preparation/execution work.
- It creates no ID or graph/value/node identity. There is therefore no allocation order,
  uniqueness, overflow, or rollback behavior to specify.

## Affected files

The intended overall implementation scope is exactly fourteen paths.

Production — exactly two paths:

- add
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/BackendOwnerSelection.java`
- update
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/package-info.java`

Tests — exactly one path:

- add
  `modules/planning/src/test/java/io/github/pho001/synaptik/planning/capability/BackendOwnerSelectionTest.java`

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

Review without modification: `BackendEligibility.java` and its test; all other planning/config/
backend-contract/trace source and tests; planning/config/trace Gradle files; architecture contract,
ADRs, and architecture tests; compiler/prepare/runtime/engine master plans and source; concrete
backends; backend-conformance and integration tests; Public Tensor APIs; every unrelated guide;
and every later task row/specification.

## Maximum scope

Exactly the fourteen paths above. Stop if implementation requires modifying
`BackendEligibility`, another production type or test, a public API, the `ownership` package, a
workload/cost/profile type, a dependency/build file, an architecture test or rule, a diagnostics
schema, a graph/partition contract, a concrete backend, or a detailed Planning 0004 or Config 0004
specification. Do not use a follow-up task to hide an incomplete acceptance criterion.

## Required Javadoc contracts

- Class Javadoc must define one stateless package-private selection operation, the existing
  hard-eligible identity list as its candidate set, and its backend-ownership-only boundary.
- Private-constructor Javadoc must explain that instances are forbidden because the class retains
  no selection state.
- Method Javadoc must document all three non-null inputs, the empty terminal failure, complete
  snapshot validation, equal-ID association, extra-snapshot allowance, empty-snapshot preference
  nonmatch, exact reference return, provider-order tie behavior, neutral/preferred/no-match
  behavior, no eligibility re-evaluation, and no device/route/kernel selection.
- Method `@return` must state that the result is the exact non-null `BackendId` reference from the
  eligible list.
- Method `@throws` entries must state every exact failure condition/message from the validation
  table.
- Package Javadoc must distinguish the public query/provider contracts from current internal hard
  eligibility and baseline owner selection. Public planning orchestration, numeric/cost scoring,
  operation-family/workload classification, profiles, partitioning, logical memory, compiler
  integration, device/route/kernel selection, prepare, runtime, and execution remain planned.

## Documentation impact and no-change expectations

- Update the six authorized explanatory documents from “ownership selection planned” to the exact
  current internal baseline: provider-ordered hard-eligible IDs, optional preferred-class match,
  deterministic fallback/ties, and internal no-eligible failure.
- Keep the public API pages explicit that no external planner workflow or public owner selector
  exists.
- Keep future cost scoring conceptual. Do not present the user guide's illustrative transfer or
  boundary numbers as current behavior.
- Add or revise one glossary entry only where needed to distinguish internal baseline ownership
  selection from hard eligibility, general partition scoring, and backend prepare routing.
- No architecture contract or ADR changes are expected because this implements the already
  assigned planning responsibility and preserves ADR 0008 boundaries.
- No Gradle/dependency, architecture-test, backend-conformance, integration, compile-workflow,
  config/backend-contract/trace Java, compiler/prepare/runtime/engine, concrete-backend, or other-
  module documentation change is expected. The documentation pass must record a reasoned
  no-change conclusion for each group rather than `N/A`.

## Acceptance criteria

- `BackendOwnerSelection` has the exact package-private final stateless class shape and sole
  package-private static `select` method specified above.
- The selector consumes `BackendEligibility` directly; no copied production candidate, public
  bridge, public matrix, or public planner facade is introduced.
- Top-level, empty-result, snapshot-element, duplicate-association, and missing-association
  validation follows the exact order, types, and messages in this specification.
- Empty hard eligibility always fails terminally before scoring or snapshot-element reads and
  never selects a default or fallback backend.
- Equal backend identities associate candidates and snapshots. Extra unique noncandidate
  snapshots are accepted; duplicate snapshot identities and missing candidate snapshots fail.
- Neutral selection returns the first eligible identity. A present preference returns the first
  eligible preferred-class match, or the first eligible identity when none matches.
- Multiple preferred matches and all other equal comparisons resolve by preserved provider
  encounter order. Snapshot input order and `DeviceClass` enum ordinal never define priority.
- Nonpreferred candidates remain hard eligible. Empty matching snapshots are preference
  nonmatches and are not silently removed. No ineligible backend can be added or restored.
- The method returns the exact selected eligibility identity reference and never the equal
  snapshot identity reference.
- No provider call, mutable input change, ID allocation, device selection/retention, route/kernel
  selection, trace emission, preparation, runtime, execution, or other side effect occurs.
- No numeric score, generic score map, classification type, `OperationFamily`, workload bucket,
  planning cost profile, platform/backend/tuning profile, or model-autotuning value is added.
- Focused automated tests lock selector shape and every recurring validation, ordering, identity,
  immutability, side-effect, and forbidden-surface invariant. Reflection is used in the focused
  test for the concrete API-shape risk; no recurring shape check remains manual.
- Existing planning dependencies remain exactly `api` for model/backend-contract and
  `implementation` for config/trace. No Gradle or architecture-test file changes.
- The two production paths, one focused test, six explanatory documents, and five planning paths
  are the exact final implementation scope.
- A separate clean-context documentation-focused pass finalizes Javadocs, explanations, glossary
  impact, planning evidence, and status in the same overall change after executable Java and the
  final module suite stabilize.
- Planning 0003 becomes Complete only after every implementation, test, documentation, Javadoc,
  scope, status, and whitespace criterion passes. Planning 0004 and Config 0004 remain Draft
  without detailed specifications after completion pending a separate frontier reassessment.

## Tests / validation

Focused development validation while executable Java is changing:

```bash
./gradlew :modules:planning:test --tests io.github.pho001.synaptik.planning.capability.BackendOwnerSelectionTest
```

After executable Java stabilizes, run exactly one final affected-module suite:

```bash
./gradlew :modules:planning:test
```

Record XML suite/test/failure/error/skip counts. Then hand the task, actual diff, and exact Java
evidence to the separate clean-context documentation pass. That pass reuses the successful Java
evidence unless it changes executable Java behavior or records a concrete cross-check risk.

After final Javadoc and Markdown edits, the documentation pass runs:

```bash
./gradlew :modules:planning:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

Inspect generated `package-summary.html`, `OperationCapabilityQuery.html`, and
`BackendCapabilityProvider.html` for the public/internal boundary. The package-private selector
must not appear as a new public generated page or all-classes entry.

Manual final checks are limited to risks not better locked by tests:

- exact fourteen-path tracked/untracked scope and no Java/test/Gradle/architecture/API/glossary
  changes outside those authorized for implementation;
- unchanged planning Gradle dependency declarations;
- one Ready row during implementation, then synchronized Complete/Draft status;
- no detailed Planning 0004 or Config 0004 specification;
- no production operation-family/workload/cost/profile type;
- generated public Javadoc boundary, balanced Markdown fences, links/anchors, final newlines, and
  trailing whitespace.

No manual `javap`, separate bytecode inspection, or source-import scan is required: ordinary
compilation plus the focused reflection/API-shape and forbidden-surface tests cover those risks.
No architecture test or repository-wide suite is run by habit because this task adds one internal
planning selector with no dependency, public API, architecture boundary, shared build, concrete
backend, or cross-module executable change. Continuous integration or the Planning 0006 closure
checkpoint owns the repository tier unless implementation reveals a concrete trigger; such a
trigger requires stopping before scope expansion.

## Documentation handoff

After the implementation context records the final planning-module result, hand this
specification, the actual fourteen-path diff, and the exact focused/final Java evidence to a
separate clean-context documentation-focused agent or thread.

The handoff must identify:

- the package-private `BackendOwnerSelection` shape and exact method signature;
- `BackendEligibility` as the unchanged candidate handoff;
- the exact null, empty, duplicate, and missing-association failures;
- neutral, preferred-match, preferred-miss, and tie behavior;
- exact eligibility-reference return and unchanged public/dependency surface;
- the decision to add no workload/cost classification or Config 0004 contract; and
- the six explanatory documents, five planning paths, and final validation commands.

The documentation pass must read the architecture contract, documentation rules, General,
API/Javadoc, Architecture, Backend guide, User guide, Planning, and Example profiles, final source
and tests, current backend-contract/config/planning contracts, ADR 0008 boundaries, and all
directly affected documents. It independently finalizes class/constructor/method/package
Javadocs, current-versus-planned wording, examples, links, terminology, glossary impact, evidence,
and completion status.

It must record reasoned no-change conclusions for Gradle/dependencies, architecture contract and
ADRs, architecture tests, backend-conformance and integration tests, compile workflow beyond the
authorized API/status pages, config/backend-contract/trace Java, compiler/prepare/runtime/engine,
concrete backends, and every other module. It does not rerun successful Java tests unless
executable behavior changes.

## Dependencies

- Complete Planning task 0001 with stable public `OperationCapabilityQuery` and
  `BackendCapabilityProvider` contracts.
- Complete Planning task 0002 with package-private immutable `BackendEligibility`, provider-order
  eligible identities, exact provider/snapshot association, and empty no-match representation.
- Complete Config task 0003 with stable optional soft `DeviceClass` preference.
- Complete Backend-contract tasks 0001–0004 with stable identity, class, snapshot, and hard-
  requirement vocabulary.
- Accepted ADR 0008 separation of planning cost, model autotuning, backend route configuration,
  benchmarking, and runtime observation.

## Follow-up tasks

- Planning task 0004 remains Draft for maximal same-owner partitioning after this one-occurrence
  owner result is stable. Do not create its detailed specification in this task.
- Config task 0004 remains Draft without a detailed specification. This baseline consumes no
  cost quantity and therefore justifies no operation-family, workload-bucket, or cost-profile
  schema. A later concrete backend-neutral cost extension must first identify the exact
  classification and units it consumes.
- Planning tasks 0005–0006 retain logical materialization/memory and closure work as Draft rows.
- Public compiler-planning orchestration and any public ownership failure remain with their later
  concrete consumer. That task may translate the current internal empty-eligibility failure but
  must not weaken hard eligibility.
- Device-level capability/selection, structured rejection diagnostics, tuning candidates and
  caches, concrete backend routing, preparation, runtime, and execution remain with their owning
  future tasks.

## Architecture impact

Expected impact: None.

The task realizes the existing planning responsibility for backend-neutral ownership selection,
uses only current immutable compile-time facts, returns `BackendId`, and preserves the hard-
eligibility and prepare/backend/tuning/runtime boundaries. If implementation reveals a need for a
public API, dependency visibility change, new cost/workload classification, device selection,
route vocabulary, graph/partition contract, or another architecture decision, stop before editing
source or architecture documentation and report the conflict.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md, the directly relevant documentation profiles,
docs/planning/roadmap.md, the planning/config/backend-contract/trace master plans, and
docs/planning/modules/planning/tasks/0003-ownership-candidates-and-baseline-scoring.md in full.

Implement task 0003 exactly inside its fourteen authorized paths. Add only the package-private
stateless BackendOwnerSelection, its focused automated test, required Javadocs, current-status
documentation, and synchronized planning evidence. Preserve BackendEligibility as the complete
candidate handoff, exact validation/failure order and messages, provider-order ties, exact
BackendId reference return, soft preference, and no-side-effect boundaries. Add no public API,
candidate record, numeric score/map, workload/cost/profile type, dependency/build change,
partitioning, device/route/kernel selection, diagnostics schema, or later task specification.
Stop and report any architecture or scope conflict.

After executable Java and the single final planning module suite, hand the actual diff and exact
evidence to a separate clean-context documentation-focused agent or thread. Reuse successful Java
evidence unless executable behavior changes. Do not run architecture or repository-wide tests
without this task's concrete trigger. Mark task 0003 Complete only after every criterion and the
documentation pass succeed.
```

## Local decisions

- Use the existing hard-eligible `BackendId` list directly as the candidate set. A separate
  candidate record would duplicate identity and add no fact required by the baseline.
- Keep the selector package-private and colocated with `BackendEligibility`. This is the smallest
  implementable cross-contract handoff and avoids exposing config through a new public planning
  signature before a compiler aggregate exists.
- Use a stateless focused selector class rather than adding selection behavior to the eligibility
  value. Hard eligibility remains a value/factory; owner comparison is a separate operation.
- Interpret the optional preferred class as one lexicographic match bucket. No numeric magnitude
  exists, so provider order remains the deterministic tie and fallback rule.
- Treat extra snapshots as expected because task 0002 validates the complete provider set before
  filtering. Only eligible identities become candidates.
- Treat an empty matching snapshot as a soft-preference nonmatch, not a renewed hard-eligibility
  filter. The eligibility result is authoritative, and this prevents scoring from weakening it.
- Return the exact eligibility identity reference so equal snapshot identities validate
  association without replacing the ownership identity selected from the candidate set.
- Define the empty result as an internal `IllegalStateException` now. A public compiler exception
  remains deferred until public orchestration exists.
- Add no workload or cost classification. The current consumer has no cost input, and a
  placeholder family/bucket type would not stabilize Config 0004.

## Known limitations

- Baseline selection considers only the current optional coarse device-class preference and
  provider order. It has no transfer, boundary, materialization, region-size, or execution-cost
  estimate.
- A backend reporting both CPU and accelerator devices matches either preferred class, but no
  device is selected and capability remains backend-level.
- The selector is internal. No current compiler, public planner, or `CompileConfig` aggregate
  invokes it externally.
- The internal no-eligible exception is not a public compile failure contract and carries no
  structured rejection reason.
- Availability is only as current as the immutable caller-supplied snapshots; selection performs
  no discovery, refresh, or liveness check.
- No serialization or external compatibility guarantee is established.

## Validation evidence

- Planning context `/root/plan_planning_0003` read the repository instructions, authoritative
  architecture contract, focused module/dependency/lifecycle/partition-scoring/performance
  explanations, ADR 0008, documentation rules and applicable profiles, planning guide and
  roadmap, planning/config/backend-contract/prepare/compiler/engine master plans, completed
  Planning 0001–0002 and Config 0001–0003 tasks, current production/tests/builds, Compile API,
  backend/user guides, and glossary before selecting this contract.
- The current consumer needs no production candidate or workload/cost classification: the
  provider-ordered eligible IDs are complete alternatives, and snapshot class presence is the
  only current ranking fact. The selected package-private handoff preserves current dependencies
  and creates no architecture uncertainty.
- Planning-stage `python3 /tmp/validate_synaptik_markdown.py` passed for 227 Markdown files, 4,012
  local links, 244 local anchors, 2,846 fence markers, final newlines, and trailing whitespace.
- `git diff --check` passed with no output. The combined tracked/untracked scope is exactly this
  task, the planning/config/trace master plans, and the roadmap: five planning-only paths. No
  Java, test, Gradle, architecture, API, guide, or glossary file changed during planning.
- Status and task-directory checks found exactly one global Ready master-plan row and exactly one
  Ready detailed task: Planning 0003. Planning 0001–0002 and Config 0001–0003 remain Complete;
  Planning 0004+, Config 0004+, and Trace 0003+ remain Draft. No detailed Planning 0004 or Config
  0004 specification exists.
- Manual planning review confirmed the exact proposed package-private shape, validation messages,
  provider-order tie/fallback semantics, unchanged public/dependency surface, cost-classification
  deferral, canonical sections and implementation prompt, and preserved completed-task history.
- Implementation context `/root/implement_planning_0003` added the package-private final stateless
  `BackendOwnerSelection` and its sole package-private static `select` method. The selector consumes
  `BackendEligibility` directly, validates inputs and equal-identity snapshot associations in the
  specified order, applies the optional preferred class, preserves provider-order ties/fallback,
  returns the exact eligible identity reference, and introduces no candidate, score, device,
  route, kernel, provider call, trace, preparation, runtime, or execution behavior.
- Focused validation
  `./gradlew :modules:planning:test --tests io.github.pho001.synaptik.planning.capability.BackendOwnerSelectionTest`
  passed with `BUILD SUCCESSFUL`; its XML reports 13 tests with zero failures, errors, or skips.
  The one final `./gradlew :modules:planning:test` suite then passed with `BUILD SUCCESSFUL`; its
  three XML suites report 38 tests with zero failures, errors, or skips. The documentation context
  reused this stabilized Java evidence and changed no executable Java or test behavior.
- Documentation context `/root/implement_planning_0003/docs_planning_0003` independently read the
  final source and focused tests, architecture and planning contracts, ADR 0008, documentation
  rules and applicable profiles, current public/config/backend-contract/planning contracts, and
  all affected explanatory, API, guide, glossary, and planning pages. It finalized the class,
  constructor, method, and package Javadocs plus current-versus-planned wording.
- `./gradlew :modules:planning:javadoc` passed with `BUILD SUCCESSFUL`; six actionable tasks were
  reported, two executed and four up-to-date. Generated `package-summary.html`,
  `OperationCapabilityQuery.html`, and `BackendCapabilityProvider.html` preserve the public query/
  provider versus internal eligibility/selection boundary. The package-private selector has no
  generated public page or all-classes entry.
- The final Markdown validation passed for 227 files, 4,014 local links, 246 local anchors, 2,844
  fence markers, final newlines, and trailing whitespace. `git diff --check` passed with no output.
  The two-marker reduction from the planning-stage count is expected: finalization replaced the
  fenced completion-summary placeholder with the actual unfenced summary below.
- Final status checks found no Ready task, Planning 0004–0006, Config 0004–0008, and Trace 0003+
  remain Draft, and no detailed Planning 0004 or Config 0004 specification exists. Source checks
  found no production operation-family, workload-bucket, cost-profile, platform-profile, or
  tuning-profile type.
- The final combined tracked/untracked scope is exactly the two production paths, one focused-test
  path, six explanatory-document paths, and five planning paths authorized below; there are no
  extra changed paths.

## Implementation notes

The implementation uses the existing immutable provider-ordered eligibility identities as the
complete candidate set and builds one temporary equal-identity snapshot lookup. It does not widen
the package-private handoff or add a numeric comparison. The focused tests lock class/method shape,
validation precedence and messages, neutral/preferred/miss/tie behavior, equality association,
reference identity, immutable inputs, and forbidden side effects/surfaces.

The documentation pass changed no executable Java, tests, build files, or dependencies. The
planning Gradle declarations remain exactly `api` for model/backend-contract and `implementation`
for config/trace. No architecture contract or ADR changed because the selector implements the
existing planning responsibility and ADR 0008 boundary. No architecture test was needed because
no dependency, module, or public boundary changed. No backend-conformance or integration test was
needed because no concrete backend or end-to-end behavior changed. No compile workflow beyond the
authorized API/status explanations changed because no compiler or public planning workflow exists.
Config, backend-contract, and trace Java remained unchanged because the selector only consumes
their current immutable contracts. Compiler, prepare, runtime, engine, concrete backends, and all
other modules remained unchanged because the selector performs no work in those layers.

## Completion summary

- Completed changes: Added the exact internal preferred-class/provider-order baseline owner
  selector over the complete hard-eligible identity list, with deterministic validation, fallback,
  tie, and exact-reference semantics.
- Files changed or created: Exactly 14 paths: production
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/BackendOwnerSelection.java`
  and `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/package-info.java`;
  test
  `modules/planning/src/test/java/io/github/pho001/synaptik/planning/capability/BackendOwnerSelectionTest.java`;
  explanatory documentation `docs/architecture/partition-scoring.md`, `docs/api/public-api.md`,
  `docs/api/compile-api.md`, `docs/backend-guide/capability-provider.md`,
  `docs/user-guide/backend-selection.md`, and `docs/glossary.md`; planning records this task,
  `docs/planning/modules/planning/master-plan.md`, `docs/planning/modules/config/master-plan.md`,
  `docs/planning/modules/trace/master-plan.md`, and `docs/planning/roadmap.md`.
- Tests and validation: Focused selector tests passed 13/13; the final planning suite passed 38/38
  across three suites; planning Javadoc, 227-file Markdown validation, exact scope/status/source
  checks, and `git diff --check` passed.
- Documentation-agent review: Clean context `/root/implement_planning_0003/docs_planning_0003`
  completed the independent documentation pass without changing executable Java behavior.
- Documentation impact: Six explanatory pages and five planning records now describe the current
  internal baseline and preserve the planned public orchestration, cost scoring, partitioning,
  device/route/kernel, prepare, runtime, and execution boundaries; the reasoned no-change groups
  are recorded above.
- Javadoc review: Finalized selector and package contracts; generated public Javadocs preserve the
  public/internal surface and exclude the package-private selector from public indexes.
- Glossary impact: Updated backend ownership and directly dependent intent/requirement status text
  to distinguish internal baseline selection from hard eligibility, future cost scoring, and
  backend prepare routing.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
