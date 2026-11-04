# Task Data0001: Data extension boundary and build foundation

> **Editorial reading context — not a status update.** This foundation exposes Model types
> to Data consumers through a Gradle API dependency; it implements no batcher. Dependency
> visibility means an application can name Tensor in public signatures, not that Data owns
> Tensor semantics, Engine execution or storage cleanup. Host sequence preparation and NN
> expression construction remain separate calls made by the application.
> Today's [Data purpose and boundary](../../../../reference/data/sequences.md#purpose-and-mental-model)
> illustrates the later consumer, while [metadata borrowing](../../../../reference/data/sequences.md#validation-failures-and-borrowing)
> explains why a batch wrapper is not a new Tensor lifetime owner.

## Status

Complete — accepted 2026-10-20 after Franklin's [binding](../../../../glossary.md#term-binding) final APPROVE (zero findings) and
main's source-matched cached root gate. Authorized Data/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[CPU](../../../../glossary.md#term-cpu)/Text/Vision frontiers: None.
Data0002–0004 and NN0022A/0023/0024 remain Draft; no successor launch is authorized.

## Change class

Class C — introduces a [module](../../../../glossary.md#term-module), [dependency edge](../../../../glossary.md#term-dependency-edge), namespace and architecture [ownership](../../../../glossary.md#term-ownership-meanings).
The approved authority is enacted before a clean implementation and independent targeted review.

## Goal

Establish the smallest Data-only foundation for [host](../../../../glossary.md#term-host) numeric [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)/sequence batching:
[Model](../../../../glossary.md#term-model) supplies [Tensor](../../../../glossary.md#tensor) semantics; Data will prepare host inputs. Register and enforce the boundary
without implementing batching or introducing a public [API](../../../../glossary.md#term-application-programming-interface-api) or production type.

## [Scope](../../../../glossary.md#term-scope) and approved decision

- Add [Gradle](../../../../glossary.md#term-gradle) `:extensions:data`, physical `extensions/data`, and Java namespace
  `io.github.pho001.synaptik.data`, approved 2026-10-20.
- Declare exactly `api(project(":modules:model"))`. The planned [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) boundary exposes [Model](../../../../glossary.md#term-model)
  [`Tensor`](../../../../glossary.md#tensor) values, so consumers must see Model without a second [dependency](../../../../glossary.md#term-dependency). An `implementation`
  edge would hide those signature types. NN and relocated Checkpoint already export Model for
  the same reason; Checkpoint's private Compiler edge is not a Data precedent.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Use root `build.gradle.kts` conventions: `java-library`, Java/release 26, UTF-8 and JUnit.
  Add no new plugin, external library, runtime flag or convention layer.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release); [UTF-8 glossary](../../../../glossary.md#term-utf-8); [JUnit glossary](../../../../glossary.md#term-junit); [plugin glossary](../../../../glossary.md#term-plugin); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Data owns only future host numeric sample/sequence preparation. No inward module or existing
  extension gains an edge to Data. Text/Vision remain separate future approval gates.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [module glossary](../../../../glossary.md#term-module))</span>
- Add no production type, marker class, speculative interface or reserved subpackage.
  Add `package-info.java` only if an approved package contract genuinely needs it; the default is
  an empty production module with its namespace enforced by architecture tests.

## Non-goals

No Data0002/0003/0004 or NN0022A implementation, [API](../../../../glossary.md#term-application-programming-interface-api) naming, batching redesign, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training/[optimizer](../../../../glossary.md#term-optimizer)
behavior, execution, [persistence](../../../../glossary.md#term-persistence), file loading, DataLoader, sampler, [epoch](../../../../glossary.md#term-epoch-meanings), [token](../../../../glossary.md#term-token) or image API.
No [Compiler](../../../../glossary.md#term-compiler), [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), [Engine](../../../../glossary.md#term-engine), [Checkpoint](../../../../glossary.md#term-checkpoint), [backend](../../../../glossary.md#backend), tuning, NN, Training, Text or Vision
[dependency](../../../../glossary.md#term-dependency), including test-only edges. No Text/Vision registration, decoder adapter or blanket
extension permission. No legacy source copying or new [Tensor](../../../../glossary.md#tensor)/storage [ownership](../../../../glossary.md#term-ownership-meanings) [lifecycle](../../../../glossary.md#lifecycle).

Preserve the staged representation in the [Data master](../master-plan.md#module-invariants-and-dependencies):
Data0002's [immutable](../../../../glossary.md#term-immutable) [host](../../../../glossary.md#term-host) valid lengths describe valid prefixes, not runtime [graph](../../../../glossary.md#graph) inputs.
Later runtime [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) non-[gradient](../../../../glossary.md#term-gradient) [`INT64`](../../../../glossary.md#data-type--datatype) Tensor [binding](../../../../glossary.md#term-binding) belongs to its own lifecycle decision.
[Padding](../../../../glossary.md#padding) lengths and dense [masks](../../../../glossary.md#term-boolean-mask) remain derived, with no duplicate source of truth.

## Contracts

Read the root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>, then:

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
  Data-only ownership/exact Model API edge; NN's existing edges and behavior remain unchanged.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- `foundational-modules.md` [`modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel):
  Tensor/host storage remain Model-owned; Data adds no execution ownership.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [host storage glossary](../../../../glossary.md#host-storage))</span>
- `recurrent-scan.md` [Static Shape and runtime-value boundary](../../../../architecture/contracts/recurrent-scan.md#static-shape-and-runtime-value-boundary)<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [runtime glossary](../../../../glossary.md#term-runtime))</span>
  and [Valid lengths, traversal, and outputs](../../../../architecture/contracts/recurrent-scan.md#valid-lengths-traversal-and-outputs):
  existing runtime lengths do not make host batch metadata a runtime input.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [metadata glossary](../../../../glossary.md#term-metadata))</span>

Pre-code authority is enacted: root namespace/[layout](../../../../glossary.md#layout), [scope](../../../../glossary.md#term-scope) index, [ownership](../../../../glossary.md#term-ownership-meanings) routing, [dependency](../../../../glossary.md#term-dependency)
rules and testing requirements, plus the existing owning contract's Scope and `extensions/data`.
[ADR 0042](../../../../design/decisions/0042-data-extension-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span> and focused [module](../../../../glossary.md#term-module)/dependency/layout
explanations record the approved choice. No seventh contract or unrelated semantic amendment.
Stop if the implementation needs a different boundary.

## Dependencies and integration

- Depends on: None (no capability predecessor requiring [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)); approval and readiness verified below.
- Conflicts with: writers of root/owning extension authority, settings/build conventions,
  architecture tests, module/dependency/layout explanations, ADR allocation and shared plans.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [layout glossary](../../../../glossary.md#layout); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Parallel group: None.
- Common base revision: `ef35f2ec001b0c161d1abe825ff3be725b9be49e` (HEAD=main=origin/main verified 2026-10-20),
  plus the accepted dirty planning packet and this approved serial authority patch.
- Integration order: serial approved pre-code contracts → clean implementation → independent
  review → one coordinated root gate for the dependency/build change.
- Integration [validation](../../../../glossary.md#term-validation): `./gradlew test javadoc` and `git diff --check` after final integration.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main.

Launch verification (2026-10-20): explicit Data-only user approval (“pust se di toho”), enacted
root/owning authority and no capability predecessor satisfied readiness. HEAD/main/origin/main
matched the recorded base; six accepted [planning](../../../../glossary.md#term-planning) modifications were preserved. Main verified the
final authority/[Ready](../../../../glossary.md#term-ready) diff before launching the clean executor. Data0001 alone was authorized;
NN/[CPU](../../../../glossary.md#term-cpu)/Text/Vision frontiers were None and successors Draft. Serial [ownership](../../../../glossary.md#term-ownership-meanings) made conflicts
inactive; baseline/JDK26/native [hash](../../../../glossary.md#term-hash) and the isolated consumer pattern were checked read-only.

## Files and symbols

The complete change comprises 17 task paths, grouped below; no code was superseded or removed.

- `ARCHITECTURE.md` and `docs/architecture/contracts/extensions-training.md` — coordinated
  Data-only permission and exact boundary described above.
- `docs/architecture/module-boundaries.md` (`Extensions`),
  `docs/architecture/dependency-rules.md` (`Intended direction`, `Concrete forbidden dependencies`,
  `Architecture-test enforcement`) and `docs/developer-guide/repository-layout.md` — explain Data.
- `docs/design/decisions/0042-data-extension-boundary.md` — accepted Data-only decision;
  API export rationale and rejected joint gate/speculative types.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `settings.gradle.kts` — include only `:extensions:data`, using its default physical path.
  New `extensions/data/build.gradle.kts` — exactly the Model API project edge.
  Read root <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>`build.gradle.kts`, `extensions/nn/build.gradle.kts` and actual relocated
  `modules/checkpoint/build.gradle.kts`; do not modify their working conventions/edges.
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/DataDependencyContractTest.java`
  (new test-only type) — registration/path, exact edges in all configurations, reverse-edge
  rejection, package/import rules and no Text/Vision inclusion. Follow existing
  `CheckpointDependencyContractTest` and `NnTrainingDependencyContractTest` file-based conventions;
  retain `EngineCompositionContractTest` and `BackendLayoutContractTest` protections.
- `testing/architecture-tests/build.gradle.kts` — add Data source inputs with relative path
  sensitivity so new/changed sources invalidate tests; wire isolated <span data-glossary-navigation="true"> ([sensitivity glossary](../../../../glossary.md#term-sensitivity))</span>`dataApiConsumer` compilation
  into <span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>`test`, following `nnApiConsumer`/`checkpointApiConsumer` without inheriting their classpaths.
- `testing/architecture-tests/src/dataApiConsumer/java/io/github/pho001/synaptik/testing/architecture/DataApiConsumer.java`
  (new compile-only test witness) — depend only on Data, import/use Model Tensor declarations;
  no direct Model dependency, Data production placeholder, Engine or runtime/backend execution.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [dependency glossary](../../../../glossary.md#term-dependency); [Engine glossary](../../../../glossary.md#term-engine); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- Six [planning](../../../../glossary.md#term-planning) paths: this `docs/planning/extensions/data/tasks/0001-data-extension-boundary.md`,
  `docs/planning/extensions/{data,nn,text,vision}/master-plan.md` and `docs/planning/roadmap.md` —
  synchronized foundation status and separate unchanged successor/Text/Vision gates.
  Model `TensorFactory.fromFlatArray`, `Tensor.stack` and `TensorFactoryFlatImportTest` remain
  reading references, not implementation targets.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>

## Acceptance criteria

1. Explicit user approval and coordinated root/single owning contract/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/explanations precede code.
2. Only the Data project is added; its sole project edge is Model API. Existing module identities,
   edges, conventions and completed NN behavior remain intact. No production type or public API.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
3. Architecture tests positively require the included project, physical build and exact Model API
   export even with no production sources; reject other project edges, including test-only ones,
   reverse inward/existing-extension edges, and unauthorized Text/Vision registration.
4. Namespace/import checks cover all Data Java sources when present. With an empty module, use
   test-local source/build fixtures to prove allowed namespace/Model imports pass and wrong
   namespace, forbidden imports and forbidden configuration edges fail. Do not rely on a vacuous
   scan or add production classes just to exercise tests.
5. The isolated consumer compiles Model Tensor declarations through Data alone. Record that this
   proves dependency export, not a Data API, batch semantics or execution capability.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [dependency glossary](../../../../glossary.md#term-dependency); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
6. Required independent Class C code/documentation review and one main-owned root gate pass.
   No batching, runtime length binding, NN integration or Text/Vision authority is delivered.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding))</span>

## Implementation [validation](../../../../glossary.md#term-validation)

Recorded worker command, then main's single repository [checkpoint](../../../../glossary.md#term-checkpoint) after final review:

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home ./gradlew :extensions:data:test :extensions:data:javadoc :testing:architecture-tests:test
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc
git diff --check
```

Architecture `test` depends on `compileDataApiConsumerJava`. Independent review and [planning](../../../../glossary.md#term-planning)
closure reuse successful executable evidence; no duplicate suite or native/performance campaign.

## Contract [validation](../../../../glossary.md#term-validation) and review impact

Independent Class C code/documentation review covers the final implementation, coordinated
authority, explanations and [ADR](../../../../glossary.md#term-architecture-decision-record-adr). General/[planning](../../../../glossary.md#term-planning) profiles govern this closure; validate local
links/anchors, brief limits, new-file whitespace and final diff without rerunning [Gradle](../../../../glossary.md#term-gradle).
Targeted [Tensor](../../../../glossary.md#tensor)/batching/[padding](../../../../glossary.md#padding)/valid-length glossary review introduces no new term. New
test/helper/consumer [Javadoc](../../../../glossary.md#term-javadoc) is accurate; [Model](../../../../glossary.md#term-model) Javadoc is unchanged and remains accurate.
No `package-info.java` is needed: the empty foundation defines no production package contract.

## Result

Delivered the 17 grouped paths: Data-only authority/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/explanations, empty [Gradle](../../../../glossary.md#term-gradle) foundation,
relative source inputs, seven [fixture](../../../../glossary.md#term-fixture)-backed contract tests and isolated [Model](../../../../glossary.md#term-model)-export consumer.
No production type/[API](../../../../glossary.md#term-application-programming-interface-api), package-info, batching, [runtime](../../../../glossary.md#term-runtime) [binding](../../../../glossary.md#term-binding) or [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Text/Vision integration.
Recorded worker command passed: 25 tasks, 3 executed/22 UP-TO-DATE; 20 architecture tests in
10 suites, zero failures/errors/skips, including seven Data tests. `compileDataApiConsumerJava`
succeeded through Data alone: [dependency](../../../../glossary.md#term-dependency) export only, not batching/execution. Data test/[Javadoc](../../../../glossary.md#term-javadoc): NO-SOURCE.
Franklin's independent binding final code/docs verdict: APPROVE, zero findings; review closed.
Main's recorded root command passed, exit 0, BUILD SUCCESSFUL in 1s: 108 actionable tasks,
all 108 UP-TO-DATE. It reused valid prior results including worker architecture evidence;
no fresh repository tests or native [device](../../../../glossary.md#term-device) runs executed. Existing cached integration assumption
skip `eligibleCpuAlternativesCompleteBothPublicTuningPhases` reflects a [host](../../../../glossary.md#term-host) fixture with no eligible
local workload alternatives, unrelated to Data; it was not newly [run](../../../../glossary.md#term-run)/skipped by this root gate.
Main verified five executable [hashes](../../../../glossary.md#term-hash) unchanged from worker through final review/root gate, both
protected hashes unchanged, and nine NN Java hashes matching the previous frozen baseline.
Configured native fixture [SHA-256](../../../../glossary.md#term-sha-256) unchanged: `100d194c2273a657ad0d5d03bfe2280a0acea4f5d9f8260cee79e7cfa44562f3`.
Documentation/Javadoc/glossary/package-info review above is complete; review links: 325 across
12 documents passed. Closure [planning](../../../../glossary.md#term-planning) links/anchors, limits, new-file whitespace and `git diff --check` passed.
All acceptance satisfied; no unresolved issue or required follow-up. Protected unrelated changes
preserved; no code superseded/removed. All named successors Draft and frontiers None; no next launch.
Status: Complete
