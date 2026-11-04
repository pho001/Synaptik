# Task 0017: Reusable [Inference](../../../../glossary.md#term-inference) Session [API](../../../../glossary.md#term-application-programming-interface-api)

## Status

Complete

## Change class

Class C — adds a public [Engine](../../../../glossary.md#term-engine) [ownership](../../../../glossary.md#term-ownership-meanings) and [concurrency](../../../../glossary.md#term-concurrency) [lifecycle](../../../../glossary.md#lifecycle) over reusable [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution),
including [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) mixed-[owner](../../../../glossary.md#term-owner) and hot-path invariants.

## Goal

Add the smallest public user-facing session that fixes one [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel), prepares it once, and
supports repeated or concurrent runs through the existing [Engine](../../../../glossary.md#term-engine), [Runtime](../../../../glossary.md#term-runtime), and result contracts.

## [Scope](../../../../glossary.md#term-scope)

- Add `Engine.session(CompiledGraph)` and public closeable [`InferenceSession`](../../../../glossary.md#former-inference-session--inferencesession).
- Make the session own exactly one existing ordinary [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) while [borrowing](../../../../glossary.md#term-borrowing) its [Engine](../../../../glossary.md#term-engine).
- Delegate runs to the existing `Engine.run` path with unchanged Tensor identity, descriptor,
  storage, publication, materialization, and result-lifetime semantics.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Register the session once in Engine cleanup ownership without registering its hidden prepared
  handle a second time.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Preserve direct prepared input/publication adapters and per-run isolated Runtime state for CPU,
  Metal, and supported mixed-owner schedules.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Add public [shape](../../../../glossary.md#shape), [lifecycle](../../../../glossary.md#lifecycle), [concurrency](../../../../glossary.md#term-concurrency), standard [CPU](../../../../glossary.md#term-cpu), real [Metal](../../../../glossary.md#term-metal), and mixed-[owner](../../../../glossary.md#term-owner) evidence.
- Update the authoritative contract, API and user guides, Javadocs, glossary, ADR, master plan, and
  roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Non-goals

- A second [compiler](../../../../glossary.md#term-compiler), preparer, scheduler, runner, result, [cache](../../../../glossary.md#term-cache), or [backend composition](../../../../glossary.md#backend-composition) layer.
- Raw-storage maps, named bindings, dynamic descriptors, asynchronous execution, implicit
  materialization, device-result handles, persistence, or session tuning controls.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [descriptors glossary](../../../../glossary.md#term-descriptor); [asynchronous glossary](../../../../glossary.md#term-asynchronous); [materialization glossary](../../../../glossary.md#materialization); [device glossary](../../../../glossary.md#term-device); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Changing `compute(...)`, `backward(...)`, standalone `prepare(...)`, or tuned-preparation
  semantics.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span> — Engine owns
  orchestration and Runtime owns isolated per-run state.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- [Runtime, Prepare, and Engine contract — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
  and [public reusable inference session](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-inference-session)<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
  — public ownership, close races, direct adapters, and no per-run lookup.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [run glossary](../../../../glossary.md#term-run); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [ADR 0017](../../../../design/decisions/0017-reusable-inference-session-facade.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span> — selected thin
  facade and rejected duplicate binding/execution alternatives.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [binding glossary](../../../../glossary.md#term-binding))</span>

If an applicable contract is missing or ambiguous, stop and report it.

## Dependencies and integration

- Depends on: [Engine](../../../../glossary.md#term-engine) 0016
- Conflicts with: Engine public [lifecycle](../../../../glossary.md#lifecycle), [ownership](../../../../glossary.md#term-ownership-meanings) [registry](../../../../glossary.md#term-registry), and shared lifecycle documentation
- Parallel group: None
- Common base revision: `4fc4fd3d1e44613cdbdc44051e0bc637992d0de4`
- Integration order: Any after 0016
- Integration validation: affected Engine/integration suites, real native Metal smoke, Javadocs,
  architecture/documentation checks, then repository <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Metal glossary](../../../../glossary.md#term-metal); [smoke glossary](../../../../glossary.md#term-smoke-test))</span>`build`
- Shared-document integration [owner](../../../../glossary.md#term-owner): `inference-session-api`

## Files and symbols

- `modules/engine/.../Engine.java`, `InferenceSession.java`, `AdvancedEngine.java` — facade,
  construction admission, cleanup registration, and existing run delegation.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [admission glossary](../../../../glossary.md#term-admission); [run glossary](../../../../glossary.md#term-run))</span>
- `modules/engine/.../CompiledGraph.java`, `RunResult.java`, `package-info.java` — affected public
  Javadocs.
- Engine and integration tests — public shape, one-time prepare, validation, order, concurrency,
  close races, lifetime, CPU, Metal, and mixed-owner behavior.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [shape glossary](../../../../glossary.md#shape); [prepare glossary](../../../../glossary.md#prepare); [validation glossary](../../../../glossary.md#term-validation); [concurrency glossary](../../../../glossary.md#term-concurrency); [lifetime glossary](../../../../glossary.md#term-lifetime); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [owner glossary](../../../../glossary.md#term-owner))</span>
- Authoritative/focused architecture, API/user documentation, glossary, ADR, plan, and roadmap —
  current contract and workflow.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Acceptance criteria

- Session construction validates exact [Engine](../../../../glossary.md#term-engine) [ownership](../../../../glossary.md#term-ownership-meanings) and prepares exactly once.
- Repeated and concurrent runs reuse that exact preparation and direct captured adapters; each run
  has isolated Runtime state and adds no work beyond the existing run contract.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `CompiledGraph.inputs()` remains the stable input occurrence authority; runs keep arbitrary list
  order and validate identity, descriptor, and storage before borrow.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Existing forward-then-gradient publication order, alias occurrences, exact-publication
  materialization, result lifetime, and detached-value lifetime remain unchanged.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [alias glossary](../../../../glossary.md#aliasing-and-arena); [materialization glossary](../../../../glossary.md#materialization); [lifetime glossary](../../../../glossary.md#term-lifetime); [detached glossary](../../../../glossary.md#detached-model-results))</span>
- Session close is idempotent, rejects new runs, does not wait for already leased runs, closes the
  hidden preparation once, and does not close returned results.<span data-glossary-navigation="true"> ([idempotent glossary](../../../../glossary.md#term-idempotent))</span>
- Engine remains [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) and closes results before sessions/preparations before integrations.
- Standard CPU, explicit real Metal, and bounded mixed-owner public sessions reuse prepared state;
  post-prepare registry poison does not affect them.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [owner glossary](../../../../glossary.md#term-owner); [prepare glossary](../../../../glossary.md#prepare); [registry glossary](../../../../glossary.md#term-registry))</span>
- The independent Class C review approves the uncommitted review-[ready](../../../../glossary.md#term-ready) diff before [commit](../../../../glossary.md#term-commit).

## [Validation](../../../../glossary.md#term-validation)

Worker validation completed:

```bash
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.EngineTypedLifecycleTest \
  --tests io.github.pho001.synaptik.engine.api.EngineTypedPublicShapeTest \
  --tests io.github.pho001.synaptik.engine.api.AdvancedEnginePublicShapeTest
./gradlew :modules:engine:test
./gradlew :testing:integration-tests:test
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :testing:integration-tests:test \
  --tests io.github.pho001.synaptik.testing.integration.EngineExplicitCompositionMetalIntegrationTest \
  --rerun-tasks
./gradlew :modules:engine:javadoc
./gradlew :testing:architecture-tests:test --rerun-tasks
./gradlew build
git diff --check
```

All commands succeeded after the first Class C review fixes. The forced native [Metal](../../../../glossary.md#term-metal) [run](../../../../glossary.md#term-run) executed
one test with zero skipped, failures, or errors. A targeted local-link validator checked all 24
changed Markdown files and reported zero missing local files or anchors.

## Documentation and review impact

- Public API, ownership, workflow, concurrency, terminology, authoritative contract, explanatory
  architecture, API reference, user guides, Javadocs, glossary, and planning all change.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency); [planning glossary](../../../../glossary.md#term-planning))</span>
- Mandatory independent Class C review completed against the final uncommitted diff, including
  lifecycle races, hot-path preservation, mixed-owner direct adapters, documentation, and
  validation evidence.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [owner glossary](../../../../glossary.md#term-owner); [validation glossary](../../../../glossary.md#term-validation))</span>
- The initial independent Class C review returned `BLOCK` with one closed-session precedence
  finding and two documentation/concurrency-test findings. The implementation now checks the
  hidden prepared delegate before input inspection under the Engine admission, the regression
  matrix proves closed-session and closed-Engine precedence without borrowing, the recording list
  and bounded close-race test are concurrency-safe, and the affected guidance is session-first.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [admission glossary](../../../../glossary.md#term-admission); [regression glossary](../../../../glossary.md#term-regression); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [borrowing glossary](../../../../glossary.md#term-borrowing); [race glossary](../../../../glossary.md#term-race))</span>
- Class C re-review approved the code and returned one close-order documentation blocker. The
  Engine and AdvancedEngine Javadocs, lifecycle/API guides, and authoritative contract were
  reconciled to retained results first, sessions and standalone prepared handles in shared reverse
  preparation-publication order second, and integrations last, including non-waiting in-flight
  Runtime lease behavior.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [Runtime glossary](../../../../glossary.md#term-runtime); [lease glossary](../../../../glossary.md#term-lease))</span>
- Narrow documentation review returned `BLOCK` for two current residuals in the
  Runtime/Prepare/backend boundary and Engine master-plan invariants. Both were reconciled to that
  same session-aware close order and lease rule; intentionally advanced-only runtime/package
  wording remains scoped to <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>`AdvancedEngine`.
- Final Class C and narrow documentation review returned `APPROVE` with no remaining blocker.
- No changelog exists; the [release](../../../../glossary.md#term-release) process explicitly leaves release notes undefined.

## Review trail

1. Initial Class C review: `BLOCK` — closed-session precedence, session-first guidance, and
   concurrency-test safety.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency))</span>
2. Class C re-review: code `APPROVE`; documentation `BLOCK` — close-order consistency.
3. Narrow documentation review: `BLOCK` — two current residuals in the
   Runtime/Prepare/backend boundary and Engine master plan.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine))</span>
4. Final Class C and narrow review: `APPROVE`.

## Result

- Implemented the thin public session [facade](../../../../glossary.md#term-facade) and [Engine](../../../../glossary.md#term-engine)-owned construction/cleanup registration.
- Added focused and real public lifecycle evidence for reuse, input/storage snapshots, occurrence
  order and aliases, result materialization/lifetime, invalid/closed states, concurrency, in-flight
  close, CPU, Metal, mixed ownership, and registry-poison direct adapters.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [occurrence glossary](../../../../glossary.md#term-occurrence); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [materialization glossary](../../../../glossary.md#materialization); [lifetime glossary](../../../../glossary.md#term-lifetime); [concurrency glossary](../../../../glossary.md#term-concurrency); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [registry glossary](../../../../glossary.md#term-registry))</span>
- Updated all scoped contract, [API](../../../../glossary.md#term-application-programming-interface-api), user, [Javadoc](../../../../glossary.md#term-javadoc), glossary, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), and [planning](../../../../glossary.md#term-planning) surfaces named above.
- Engine, integration, forced real Metal, Javadoc, architecture, full-build, Markdown-link, and
  whitespace validation completed successfully.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Addressed all three initial Class C review findings and reran focused, full, native, Javadoc,
  architecture, documentation-link, and repository validation.
- After the documentation consistency fixes, Engine Javadocs, all 24 changed Markdown links and
  anchors, targeted stale-wording searches, and `git diff --check` pass.
- No changelog update: the repository has no changelog/[release](../../../../glossary.md#term-release)-note convention.

Status: Complete
