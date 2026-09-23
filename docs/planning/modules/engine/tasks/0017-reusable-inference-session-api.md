# Task 0017: Reusable Inference Session API

## Status

Complete

## Change class

Class C — adds a public Engine ownership and concurrency lifecycle over reusable prepared execution,
including CPU/Metal mixed-owner and hot-path invariants.

## Goal

Add the smallest public user-facing session that fixes one compiled graph, prepares it once, and
supports repeated or concurrent runs through the existing Engine, Runtime, and result contracts.

## Scope

- Add `Engine.session(CompiledGraph)` and public closeable `InferenceSession`.
- Make the session own exactly one existing ordinary `PreparedExecution` while borrowing its Engine.
- Delegate runs to the existing `Engine.run` path with unchanged Tensor identity, descriptor,
  storage, publication, materialization, and result-lifetime semantics.
- Register the session once in Engine cleanup ownership without registering its hidden prepared
  handle a second time.
- Preserve direct prepared input/publication adapters and per-run isolated Runtime state for CPU,
  Metal, and supported mixed-owner schedules.
- Add public shape, lifecycle, concurrency, standard CPU, real Metal, and mixed-owner evidence.
- Update the authoritative contract, API and user guides, Javadocs, glossary, ADR, master plan, and
  roadmap.

## Non-goals

- A second compiler, preparer, scheduler, runner, result, cache, or backend composition layer.
- Raw-storage maps, named bindings, dynamic descriptors, asynchronous execution, implicit
  materialization, device-result handles, persistence, or session tuning controls.
- Changing `compute(...)`, `backward(...)`, standalone `prepare(...)`, or tuned-preparation
  semantics.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle) — Engine owns
  orchestration and Runtime owns isolated per-run state.
- [Runtime, Prepare, and Engine contract — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  and [public reusable inference session](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-inference-session)
  — public ownership, close races, direct adapters, and no per-run lookup.
- [ADR 0017](../../../../design/decisions/0017-reusable-inference-session-facade.md) — selected thin
  facade and rejected duplicate binding/execution alternatives.

If an applicable contract is missing or ambiguous, stop and report it.

## Dependencies and integration

- Depends on: Engine 0016
- Conflicts with: Engine public lifecycle, ownership registry, and shared lifecycle documentation
- Parallel group: None
- Common base revision: `4fc4fd3d1e44613cdbdc44051e0bc637992d0de4`
- Integration order: Any after 0016
- Integration validation: affected Engine/integration suites, real native Metal smoke, Javadocs,
  architecture/documentation checks, then repository `build`
- Shared-document integration owner: `inference-session-api`

## Files and symbols

- `modules/engine/.../Engine.java`, `InferenceSession.java`, `AdvancedEngine.java` — facade,
  construction admission, cleanup registration, and existing run delegation.
- `modules/engine/.../CompiledGraph.java`, `RunResult.java`, `package-info.java` — affected public
  Javadocs.
- Engine and integration tests — public shape, one-time prepare, validation, order, concurrency,
  close races, lifetime, CPU, Metal, and mixed-owner behavior.
- Authoritative/focused architecture, API/user documentation, glossary, ADR, plan, and roadmap —
  current contract and workflow.

## Acceptance criteria

- Session construction validates exact Engine ownership and prepares exactly once.
- Repeated and concurrent runs reuse that exact preparation and direct captured adapters; each run
  has isolated Runtime state and adds no work beyond the existing run contract.
- `CompiledGraph.inputs()` remains the stable input occurrence authority; runs keep arbitrary list
  order and validate identity, descriptor, and storage before borrow.
- Existing forward-then-gradient publication order, alias occurrences, exact-publication
  materialization, result lifetime, and detached-value lifetime remain unchanged.
- Session close is idempotent, rejects new runs, does not wait for already leased runs, closes the
  hidden preparation once, and does not close returned results.
- Engine remains borrowed and closes results before sessions/preparations before integrations.
- Standard CPU, explicit real Metal, and bounded mixed-owner public sessions reuse prepared state;
  post-prepare registry poison does not affect them.
- The independent Class C review approves the uncommitted review-ready diff before commit.

## Validation

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

All commands succeeded after the first Class C review fixes. The forced native Metal run executed
one test with zero skipped, failures, or errors. A targeted local-link validator checked all 24
changed Markdown files and reported zero missing local files or anchors.

## Documentation and review impact

- Public API, ownership, workflow, concurrency, terminology, authoritative contract, explanatory
  architecture, API reference, user guides, Javadocs, glossary, and planning all change.
- Mandatory independent Class C review completed against the final uncommitted diff, including
  lifecycle races, hot-path preservation, mixed-owner direct adapters, documentation, and
  validation evidence.
- The initial independent Class C review returned `BLOCK` with one closed-session precedence
  finding and two documentation/concurrency-test findings. The implementation now checks the
  hidden prepared delegate before input inspection under the Engine admission, the regression
  matrix proves closed-session and closed-Engine precedence without borrowing, the recording list
  and bounded close-race test are concurrency-safe, and the affected guidance is session-first.
- Class C re-review approved the code and returned one close-order documentation blocker. The
  Engine and AdvancedEngine Javadocs, lifecycle/API guides, and authoritative contract were
  reconciled to retained results first, sessions and standalone prepared handles in shared reverse
  preparation-publication order second, and integrations last, including non-waiting in-flight
  Runtime lease behavior.
- Narrow documentation review returned `BLOCK` for two current residuals in the
  Runtime/Prepare/backend boundary and Engine master-plan invariants. Both were reconciled to that
  same session-aware close order and lease rule; intentionally advanced-only runtime/package
  wording remains scoped to `AdvancedEngine`.
- Final Class C and narrow documentation review returned `APPROVE` with no remaining blocker.
- No changelog exists; the release process explicitly leaves release notes undefined.

## Review trail

1. Initial Class C review: `BLOCK` — closed-session precedence, session-first guidance, and
   concurrency-test safety.
2. Class C re-review: code `APPROVE`; documentation `BLOCK` — close-order consistency.
3. Narrow documentation review: `BLOCK` — two current residuals in the
   Runtime/Prepare/backend boundary and Engine master plan.
4. Final Class C and narrow review: `APPROVE`.

## Result

- Implemented the thin public session facade and Engine-owned construction/cleanup registration.
- Added focused and real public lifecycle evidence for reuse, input/storage snapshots, occurrence
  order and aliases, result materialization/lifetime, invalid/closed states, concurrency, in-flight
  close, CPU, Metal, mixed ownership, and registry-poison direct adapters.
- Updated all scoped contract, API, user, Javadoc, glossary, ADR, and planning surfaces named above.
- Engine, integration, forced real Metal, Javadoc, architecture, full-build, Markdown-link, and
  whitespace validation completed successfully.
- Addressed all three initial Class C review findings and reran focused, full, native, Javadoc,
  architecture, documentation-link, and repository validation.
- After the documentation consistency fixes, Engine Javadocs, all 24 changed Markdown links and
  anchors, targeted stale-wording searches, and `git diff --check` pass.
- No changelog update: the repository has no changelog/release-note convention.

Status: Complete
