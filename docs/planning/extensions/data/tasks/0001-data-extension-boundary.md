# Task Data0001: Data extension boundary and build foundation

## Status

Complete — accepted 2026-10-20 after Franklin's binding final APPROVE (zero findings) and
main's source-matched cached root gate. Authorized Data/NN/CPU/Text/Vision frontiers: None.
Data0002–0004 and NN0022A/0023/0024 remain Draft; no successor launch is authorized.

## Change class

Class C — introduces a module, dependency edge, namespace and architecture ownership.
The approved authority is enacted before a clean implementation and independent targeted review.

## Goal

Establish the smallest Data-only foundation for host numeric sample/sequence batching:
Model supplies Tensor semantics; Data will prepare host inputs. Register and enforce the boundary
without implementing batching or introducing a public API or production type.

## Scope and approved decision

- Add Gradle `:extensions:data`, physical `extensions/data`, and Java namespace
  `io.github.pho001.synaptik.data`, approved 2026-10-20.
- Declare exactly `api(project(":modules:model"))`. The planned batch boundary exposes Model
  `Tensor` values, so consumers must see Model without a second dependency. An `implementation`
  edge would hide those signature types. NN and relocated Checkpoint already export Model for
  the same reason; Checkpoint's private Compiler edge is not a Data precedent.
- Use root `build.gradle.kts` conventions: `java-library`, Java/release 26, UTF-8 and JUnit.
  Add no new plugin, external library, runtime flag or convention layer.
- Data owns only future host numeric sample/sequence preparation. No inward module or existing
  extension gains an edge to Data. Text/Vision remain separate future approval gates.
- Add no production type, marker class, speculative interface or reserved subpackage.
  Add `package-info.java` only if an approved package contract genuinely needs it; the default is
  an empty production module with its namespace enforced by architecture tests.

## Non-goals

No Data0002/0003/0004 or NN0022A implementation, API naming, batching redesign, NN/Training/optimizer
behavior, execution, persistence, file loading, DataLoader, sampler, epoch, token or image API.
No Compiler, Runtime, Prepare, Engine, Checkpoint, backend, tuning, NN, Training, Text or Vision
dependency, including test-only edges. No Text/Vision registration, decoder adapter or blanket
extension permission. No legacy source copying or new Tensor/storage ownership lifecycle.

Preserve the staged representation in the [Data master](../master-plan.md#valid-length-representation-decision):
Data0002's immutable host valid lengths describe valid prefixes, not runtime graph inputs.
Later runtime rank-one non-gradient `INT64` Tensor binding belongs to its own lifecycle decision.
Padding lengths and dense masks remain derived, with no duplicate source of truth.

## Contracts

Read the root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), then:

- Root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  and [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates):
  one coordinated root/owning-contract change is required; plans cannot grant permission.
- Root [Java namespace](../../../../../ARCHITECTURE.md#java-namespace),
  [Repository layout](../../../../../ARCHITECTURE.md#repository-layout),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Existing `extensions-training.md` [Scope](../../../../architecture/contracts/extensions-training.md#scope),
  [`extensions/data`](../../../../architecture/contracts/extensions-training.md#extensionsdata) and
  [`extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn):
  Data-only ownership/exact Model API edge; NN's existing edges and behavior remain unchanged.
- `foundational-modules.md` [`modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel):
  Tensor/host storage remain Model-owned; Data adds no execution ownership.
- `recurrent-scan.md` [Static Shape and runtime-value boundary](../../../../architecture/contracts/recurrent-scan.md#static-shape-and-runtime-value-boundary)
  and [Valid lengths, traversal, and outputs](../../../../architecture/contracts/recurrent-scan.md#valid-lengths-traversal-and-outputs):
  existing runtime lengths do not make host batch metadata a runtime input.

Pre-code authority is enacted: root namespace/layout, scope index, ownership routing, dependency
rules and testing requirements, plus the existing owning contract's Scope and `extensions/data`.
[ADR 0042](../../../../design/decisions/0042-data-extension-boundary.md) and focused module/dependency/layout
explanations record the approved choice. No seventh contract or unrelated semantic amendment.
Stop if the implementation needs a different boundary.

## Dependencies and integration

- Depends on: None (no capability predecessor requiring NN); approval and readiness verified below.
- Conflicts with: writers of root/owning extension authority, settings/build conventions,
  architecture tests, module/dependency/layout explanations, ADR allocation and shared plans.
- Parallel group: None.
- Common base revision: `ef35f2ec001b0c161d1abe825ff3be725b9be49e` (HEAD=main=origin/main verified 2026-10-20),
  plus the accepted dirty planning packet and this approved serial authority patch.
- Integration order: serial approved pre-code contracts → clean implementation → independent
  review → one coordinated root gate for the dependency/build change.
- Integration validation: `./gradlew test javadoc` and `git diff --check` after final integration.
- Shared-document integration owner: main.

Launch verification (2026-10-20): explicit Data-only user approval (“pust se di toho”), enacted
root/owning authority and no capability predecessor satisfied readiness. HEAD/main/origin/main
matched the recorded base; six accepted planning modifications were preserved. Main verified the
final authority/Ready diff before launching the clean executor. Data0001 alone was authorized;
NN/CPU/Text/Vision frontiers were None and successors Draft. Serial ownership made conflicts
inactive; baseline/JDK26/native hash and the isolated consumer pattern were checked read-only.

## Files and symbols

The complete change comprises 17 task paths, grouped below; no code was superseded or removed.

- `ARCHITECTURE.md` and `docs/architecture/contracts/extensions-training.md` — coordinated
  Data-only permission and exact boundary described above.
- `docs/architecture/module-boundaries.md` (`Extensions`),
  `docs/architecture/dependency-rules.md` (`Intended direction`, `Concrete forbidden dependencies`,
  `Architecture-test enforcement`) and `docs/developer-guide/repository-layout.md` — explain Data.
- `docs/design/decisions/0042-data-extension-boundary.md` — accepted Data-only decision;
  API export rationale and rejected joint gate/speculative types.
- `settings.gradle.kts` — include only `:extensions:data`, using its default physical path.
  New `extensions/data/build.gradle.kts` — exactly the Model API project edge.
  Read root `build.gradle.kts`, `extensions/nn/build.gradle.kts` and actual relocated
  `modules/checkpoint/build.gradle.kts`; do not modify their working conventions/edges.
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/DataDependencyContractTest.java`
  (new test-only type) — registration/path, exact edges in all configurations, reverse-edge
  rejection, package/import rules and no Text/Vision inclusion. Follow existing
  `CheckpointDependencyContractTest` and `NnTrainingDependencyContractTest` file-based conventions;
  retain `EngineCompositionContractTest` and `BackendLayoutContractTest` protections.
- `testing/architecture-tests/build.gradle.kts` — add Data source inputs with relative path
  sensitivity so new/changed sources invalidate tests; wire isolated `dataApiConsumer` compilation
  into `test`, following `nnApiConsumer`/`checkpointApiConsumer` without inheriting their classpaths.
- `testing/architecture-tests/src/dataApiConsumer/java/io/github/pho001/synaptik/testing/architecture/DataApiConsumer.java`
  (new compile-only test witness) — depend only on Data, import/use Model Tensor declarations;
  no direct Model dependency, Data production placeholder, Engine or runtime/backend execution.
- Six planning paths: this `docs/planning/extensions/data/tasks/0001-data-extension-boundary.md`,
  `docs/planning/extensions/{data,nn,text,vision}/master-plan.md` and `docs/planning/roadmap.md` —
  synchronized foundation status and separate unchanged successor/Text/Vision gates.
  Model `TensorFactory.fromFlatArray`, `Tensor.stack` and `TensorFactoryFlatImportTest` remain
  reading references, not implementation targets.

## Acceptance criteria

1. Explicit user approval and coordinated root/single owning contract/ADR/explanations precede code.
2. Only the Data project is added; its sole project edge is Model API. Existing module identities,
   edges, conventions and completed NN behavior remain intact. No production type or public API.
3. Architecture tests positively require the included project, physical build and exact Model API
   export even with no production sources; reject other project edges, including test-only ones,
   reverse inward/existing-extension edges, and unauthorized Text/Vision registration.
4. Namespace/import checks cover all Data Java sources when present. With an empty module, use
   test-local source/build fixtures to prove allowed namespace/Model imports pass and wrong
   namespace, forbidden imports and forbidden configuration edges fail. Do not rely on a vacuous
   scan or add production classes just to exercise tests.
5. The isolated consumer compiles Model Tensor declarations through Data alone. Record that this
   proves dependency export, not a Data API, batch semantics or execution capability.
6. Required independent Class C code/documentation review and one main-owned root gate pass.
   No batching, runtime length binding, NN integration or Text/Vision authority is delivered.

## Implementation validation

Recorded worker command, then main's single repository checkpoint after final review:

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home ./gradlew :extensions:data:test :extensions:data:javadoc :testing:architecture-tests:test
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc
git diff --check
```

Architecture `test` depends on `compileDataApiConsumerJava`. Independent review and planning
closure reuse successful executable evidence; no duplicate suite or native/performance campaign.

## Contract validation and review impact

Independent Class C code/documentation review covers the final implementation, coordinated
authority, explanations and ADR. General/planning profiles govern this closure; validate local
links/anchors, brief limits, new-file whitespace and final diff without rerunning Gradle.
Targeted Tensor/batching/padding/valid-length glossary review introduces no new term. New
test/helper/consumer Javadoc is accurate; Model Javadoc is unchanged and remains accurate.
No `package-info.java` is needed: the empty foundation defines no production package contract.

## Result

Delivered the 17 grouped paths: Data-only authority/ADR/explanations, empty Gradle foundation,
relative source inputs, seven fixture-backed contract tests and isolated Model-export consumer.
No production type/API, package-info, batching, runtime binding or NN/Text/Vision integration.
Recorded worker command passed: 25 tasks, 3 executed/22 UP-TO-DATE; 20 architecture tests in
10 suites, zero failures/errors/skips, including seven Data tests. `compileDataApiConsumerJava`
succeeded through Data alone: dependency export only, not batching/execution. Data test/Javadoc: NO-SOURCE.
Franklin's independent binding final code/docs verdict: APPROVE, zero findings; review closed.
Main's recorded root command passed, exit 0, BUILD SUCCESSFUL in 1s: 108 actionable tasks,
all 108 UP-TO-DATE. It reused valid prior results including worker architecture evidence;
no fresh repository tests or native device runs executed. Existing cached integration assumption
skip `eligibleCpuAlternativesCompleteBothPublicTuningPhases` reflects a host fixture with no eligible
local workload alternatives, unrelated to Data; it was not newly run/skipped by this root gate.
Main verified five executable hashes unchanged from worker through final review/root gate, both
protected hashes unchanged, and nine NN Java hashes matching the previous frozen baseline.
Configured native fixture SHA-256 unchanged: `100d194c2273a657ad0d5d03bfe2280a0acea4f5d9f8260cee79e7cfa44562f3`.
Documentation/Javadoc/glossary/package-info review above is complete; review links: 325 across
12 documents passed. Closure planning links/anchors, limits, new-file whitespace and `git diff --check` passed.
All acceptance satisfied; no unresolved issue or required follow-up. Protected unrelated changes
preserved; no code superseded/removed. All named successors Draft and frontiers None; no next launch.
Status: Complete
