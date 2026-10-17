# Task Engine0021: Prepared execution and fresh-value binding

## Status

Complete — implemented in a clean context after user authorization and verified Ready launch.
Independent final Class C approval and the coordinator's native-enabled integrated gate passed
on 2026-10-16. Original base/dependencies remain recorded below; downstream tasks stay Draft.

## Change class

Class C — public API removal, prepared ownership/admission and borrowed-input lifetime boundary,
with atomic mechanical consumer migration, including Training; not a Training redesign.

## Goal

Keep Engine `PreparedExecution` as the sole ordinary public prepared owner. Remove the thin
`InferenceSession` and `Engine.session`; add `Engine.runValues(PreparedExecution, List<Tensor>)`
returning existing `RunResult` for fresh positional values. Keep exact-identity `Engine.run`.

## Scope and non-goals

One capability includes API/delegate removal, positional binding, all real consumer migrations,
focused tests and coordinated documentation. No replacement session/executor/runner/compatibility
facade or new public type. Retain Runtime PreparedExecution, advanced lifecycle, TrainingSession,
RunResult and detached HostTensorValue. Engine's outward ownership/adapters versus Runtime's
unique prepared-resource lease authority is intentional, not duplication to remove.

Training changes only its internal Engine owner/calls; generic relocation/admission (Training0007)
and fresh non-parameter inputs (Training0008) remain separate downstream Draft capabilities.
No Model convenience, Checkpoint move/codec, shape specialization, Runtime lease rewrite,
backend/kernel/build change, discovery, tuning, zero-node support or performance/proof campaign.

## Contracts

Start with [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read root `Core lifecycle`, prepared/run ownership bullets in `Core invariants`, `Module-ownership
routing`, `Dependency rules`, `Explicit non-goals`, and `Testing requirements`.

- [Runtime/Prepare/Engine](../../../../architecture/contracts/runtime-prepare-engine.md):
  `modules/runtime`, `modules/engine`, `Public reusable prepared execution`,
  `Construction failure and closure`, and `Run lifecycle`.
- [Extensions/training](../../../../architecture/contracts/extensions-training.md):
  `extensions/nn`, `extensions/training`, and `Optimizer/training lifecycle` (owner, admission,
  parameter validation, transaction and close rules; no optimizer arithmetic change).

Implemented authority cutover: root `Core lifecycle`; scoped Engine `modules/engine`, prepared-owner
rule (legacy session anchor retained), `Construction failure and closure`, `Run lifecycle`; Training
`Optimizer/training lifecycle` owner/open/run text. Dependency/ownership/testing edges remain
unchanged. Runtime retains unique lease authority. This plan is not authority.

## Dependencies and integration

- Depends on: Engine0017/0019/0020 Complete; CP0005D Complete; reviewed P2 five-choice packet.
- Conflicts with: None active; only this isolated planning writer and subsequent read-only reviewer.
  Serialize future Engine binding/close and Training consumer writers and shared-contract edits.
- Parallel group: None.
- Common base revision: recorded current-main snapshot, HEAD
  9e7e8aa15e0833a1bc91263a7add99a80d88cd92 plus completed integrated uncommitted P1 and reviewed
  P2 documents. Coordinator confirmed all nine approved planning files matched at launch;
  inherited dirty files are not this amendment. Reverify snapshot/conflicts before any launch.
- Integration order: coordinated authority → Engine0021 and mechanical consumers/review → gate;
  then Training0007 → Compiler0008 → Checkpoint0008 → Training0008 → NN0028 shared-document order.
- Integration validation: coordinator-owned `Engine0021 prepared-owner cross-module gate` below.
- Shared-document integration owner: model-execution coordinator.

## Files, symbols and mechanical migration

Paths abbreviated with `...` use the existing `io/github/pho001/synaptik` prefix.

- `modules/engine/src/main/java/.../engine/InferenceSession.java`: remove the entire type, including
  compiledGraph/run/isClosed/close. Remove Engine.session and package-private Engine.runSession;
  AdvancedEngine.openInferenceSession/runSessionOrdinary/unregister(InferenceSession) and
  session-only registration/delegate branches. Retain advanced/tuning paths and prepared registry.
- Engine/AdvancedEngine: prepare, run, runOrdinaryOpen, validateStorage, finishPreparedHandle,
  finishOrdinaryRun and close; share validated storage/adapter execution, not another runner.
  PreparedExecution keeps compiledGraph/isClosed/close and captured adapters; CompiledGraph,
  RunResult, HostTensorValue and package-info Javadocs reflect the resulting lifecycle.
- Every `engine.session(graph)` becomes `engine.prepare(graph)`; every `session.run(inputs)`
  becomes `engine.run(prepared, inputs)`, not runValues. Metadata/isClosed/close use `prepared`.
  Fresh callers explicitly opt into `engine.runValues(prepared, compiledOrderValues)`.
- `extensions/training/src/main/java/.../training/TrainingSession.java`: field/constructor type
  InferenceSession → Engine PreparedExecution; open uses borrowedEngine.prepare(graph);
  executeAndDecode uses engine.run(prepared, inputs); isClosed/beginOperation/close use prepared.
  Preserve seed-arena order and failed-open cleanup. TrainingSession remains because it owns
  optimizer/accumulation/counter state, busy admission and atomic update transactions.
- `tools/benchmarks/src/main/java/.../tools/benchmarks/CpuLifecycleBenchmark.java`: migrate owner
  and timedSample/runBatch/checksumRun arguments to Engine + PreparedExecution; no benchmark run.
- `modules/engine/src/test/java/.../engine/EngineTypedLifecycleTest.java`,
  `api/EngineTypedPublicShapeTest.java` and AdvancedEnginePublicShapeTest.java: migrate cases,
  assert removals/new method and
  ordinary signatures' continued exclusion of Runtime/Prepare/backend internals.
- `testing/integration-tests/src/test/java/.../engine/EngineMixedOwnerTestAccess.java` (reflective hook),
  `.../testing/integration/EngineExplicitCompositionMetalIntegrationTest.java`, and every actual
  session factory/run consumer in that integration package (CPU, Metal, dtype/layout, numerical,
  RNG fixtures): mechanical prepare/run migration retaining existing assertions. Do not replace
  Training fixture session() accessors returning the retained TrainingSession.
  Add fresh-value coverage to existing EngineTypedLifecycleIntegrationTest; do not create a campaign.
- `extensions/training/src/test/java/.../training/TrainingSessionTest.java`,
  TrainingSessionFailureContractTest and existing trainingApiConsumer: validate preserved public
  Training API/native capture/rollback; no new Training inputs or dependency edge.
- `testing/architecture-tests/src/test/java/.../architecture/EngineCompositionContractTest.java`
  and NnTrainingDependencyContractTest: preserve exact current role-aware edges; extend focused
  boundary assertions if needed, no blanket directory exemption. Consumer scan must leave no
  executable reference to the removed API, including inferred `var` factories and test hooks.

InferenceSession/Engine.session removal intentionally breaks source/binary compatibility; migrate
and recompile without a shim. Model/declarative APIs, old readers and Runtime handles remain.

## Acceptance criteria

1. Prepare publishes exactly one outward prepared registration, with rollback on inward prepare,
   wrapper construction or publication failure. Engine rejects new operations, waits for admitted
   work, then closes results, ordinary/advanced preparations and integrations in their existing
   reverse publication/registration order; attempt-all, suppression and failure replay remain.
2. Both ordinary run methods admit Engine first, validate non-null/exact-owner prepared handle,
   retrieve/check its exact inward delegate, then inspect inputs. Deliberate precedence change:
   today direct run can validate inputs/borrow before discovering prepared closure; after cutover
   closed prepared wins over null/malformed/dead inputs with `prepared execution is closed` and
   no borrow. Prepared null/foreign-owner validation also precedes input-list validation. Closed
   Engine still wins all arguments. This adopts former session-strength admission, not identical
   old direct-run failure precedence. Test/document combinations explicitly.
3. Runtime remains the unique nonwaiting run-lease authority after that delegate check. Close/run
   races neither add a second lease nor hold an outward lock through execution. Already leased
   runs may finish; cleanup is deferred to their lease release. Returned RunResult survives
   prepared close while Engine is open; borrowed inputs last until result close; detached values
   outlive result/prepared/Engine/input storage. Metadata stays readable after close.
4. run retains exact TensorId membership once each, arbitrary input order, full descriptor checks
   and open-valid-handle diagnostics. runValues uses compiledGraph.inputs order and exact count,
   with no ID requirement; duplicate fresh Tensor references may serve matching distinct slots.
   Require fully static shape and exact full descriptor, including dtype/layout/requiresGrad.
   [32,784] rejects [64,784]; absent versus resolved layout is not interchangeable.
5. Before any borrow, snapshot each association once and validate all values/storage: nulls,
   count, descriptor, storage dtype, presence, liveness, current-thread accessibility and capacity.
   Resolved layout uses checked referenced span; unresolved layout uses checked canonical static
   element/byte geometry, not today's layout-present-only check. Reject unknown/overflow geometry.
   Keep this added unresolved-layout validation scoped to runValues; do not widen run semantics.
6. Reuse prepared metadata and direct adapters; precompute static positional geometry where needed.
   No prototype mutation, synthetic Tensor wrapper, recapture/recompile/reprepare, provider/registry
   lookup, extra per-element work or second authoritative binding map. Reuse ordinary per-run
   storage/borrow collections. Concurrent fresh calls have isolated mutable state/results and
   no cross-call input leak; no throughput/device-overlap promise.
7. Deterministic tests cover both close-race outcomes, shutdown order, prepare/publication rollback,
   later borrow/run/result-publication failure, reverse cleanup and suppressed failures. Retain
   exact occurrence authentication and detached materialization. No sleeps as concurrency evidence.
8. All consumers migrate atomically. Training still compiles/prepares once, runs exact-ID inputs,
   closes result before prepared/seed, preserves busy/close admission, native capture, live
   precommit checks, counters/slots and rollback. No algorithms or generic admission move.
9. Real CPU integration: two fresh same-descriptor batches yield known different bytes with one
   preparation, unchanged prototypes, mismatch rejection; old run rejects fresh IDs. Preserve
   generic/declarative Model behavior, file readers, advanced lifecycle and one-shot semantics.

## Validation

```bash
./gradlew :testing:integration-tests:test --tests '*EngineTypedLifecycleIntegrationTest' --tests '*TrainingSessionCpuIntegrationTest' --max-workers=2 --console=plain
./gradlew :modules:engine:test :modules:engine:javadoc :extensions:training:test :extensions:training:javadoc :tools:benchmarks:compileJava --max-workers=2 --console=plain
./gradlew :testing:architecture-tests:compileTrainingApiConsumerJava --max-workers=2 --console=plain
./gradlew :modules:engine:javadoc --max-workers=2 --console=plain
./gradlew :modules:engine:test --max-workers=2 --console=plain
```

Coordinator runs `Engine0021 prepared-owner cross-module gate` once after integration/review:
`./gradlew test javadoc --max-workers=2 --console=plain`, including architecture and compiled
benchmark/consumer coverage; no benchmark execution or new native/performance build campaign.
Independent review reuses successful evidence unless behavior changed or a concrete risk warrants
rerun. Validate actual-delta Markdown links/anchors, brief size and whitespace; no builds in planning.

## Documentation and review impact

Mandatory independent Class C API/lifecycle/documentation pass; coordinator integrates authority.
Implemented cutover: root/scoped headings above; docs/api/{public-api,runtime-api,training-api,tensor-api}.md;
docs/user-guide/{preparing-execution,running-models,training}.md; docs/architecture/{lifecycle,
module-boundaries,training-graph,current-architecture-plan,runtime-prepare-backend-boundary}.md;
docs/developer-guide/benchmarking.md;
targeted glossary inference/prepared/Training session, borrow and result entries. Replace current
session examples with explicit prepare/run and a separately labeled positional fresh-value example.
Update affected Engine/PreparedExecution/TrainingSession and consumer Javadocs with all params,
returns/failures, ownership and precedence; inspect unchanged RunResult/HostTensorValue contracts.
ADR0039 records the approved target. At implementation annotate ADR0017 status with a narrow
supersession link to ADR0039, preserving its historical rationale; reconcile ADR0018's Training
owner reference without changing optimizer decisions. Neither historical ADR is edited in planning.

## Result

Delivered prepared-only ownership, runValues/static geometry, admission precedence and publication
rollback; migrated Training/benchmark/test consumers and updated authority, APIs/guides/glossary/Javadocs.
No dependency/backend/build/downstream change. Protected user files and completed histories preserved.
Worker Engine 136/136 and Training 72/72 passed; focused integration had 15 passes/two Metal-config skips.
Both affected Javadocs and benchmark/Training consumer compile passed. Final Class C review: APPROVE.
Coordinator gate passed in 10s: 106 tasks, 21 executed, 85 up-to-date; no native rebuild or benchmark:
`SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain`
Integrated cases: 131, zero failures/errors, one host-CPU tuning-eligibility skip. Both previously
skipped real-Metal Training cases passed. Aggregate JUnit reports: 4132 cases, zero failures/errors,
41 skips; unchanged task outputs were reused. XML roots: `modules/engine/build/test-results/test/`,
`extensions/training/build/test-results/test/`, `testing/integration-tests/build/test-results/test/`.
Post-delegate/pre-lease race covers both APIs, reverse cleanup, suppression and released admission.
Links/legacy anchors/whitespace pass with no new errors; HostTensorValue Javadoc reviewed unchanged.
Status: Complete
