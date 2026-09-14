# Task 0002: Standard Built-In Composition

## Status

Complete

## Goal

Provide the ordinary deterministic Engine construction path without requiring an application to
name, construct, or supply `CpuBackendIntegration` or another backend implementation. The current
truthful standard inventory is exactly one CPU lifecycle adapter. Each standard construction owns
one fresh independent composition and closes it through the lifecycle established by task 0001.

The public construction shape is:

```java
try (Engine engine = Engine.standard()) {
    // Ordinary typed compile, prepare, run, and result APIs arrive in task 0003.
}
```

This task deliberately creates a distinct ordinary `Engine` type rather than adding a factory to
`AdvancedEngine`. Returning `AdvancedEngine` would make its `FunctionalGradientRequest`,
`BufferRepresentation`, advanced handles, and representation-level run seam the apparent ordinary
lifecycle. A distinct type gives task 0003 an Engine-owned user surface on which inward Compiler,
Prepare, Runtime, and CPU integration contracts never need to appear. It also preserves
`AdvancedEngine.takeOwnership(CpuBackendIntegration)` as the explicit advanced path for tests,
deployment policy, and caller-managed adapter choice.

## Scope

- Add public final `io.github.pho001.synaptik.engine.Engine`, implementing `AutoCloseable`, as the
  ordinary standard-composition owner.
- Add exactly one public construction method, `Engine.standard()`. Each call opens one new
  `CpuBackendIntegration`, transfers its ownership into one new `AdvancedEngine`, and retains that
  advanced owner privately for lifecycle delegation.
- Fix the standard built-in backend inventory and order for this task to exactly CPU. The CPU
  adapter is constructed directly by Engine code; it is not discovered, looked up, selected at
  Runtime, or supplied by the caller.
- Keep the current CPU adapter's bounded automatic OpenBLAS route setup unchanged. OpenBLAS is an
  internal CPU route, not another Engine backend or backend-discovery mechanism, and portable CPU
  execution remains its existing fallback.
- Expose only `standard()`, `isClosed()`, and `close()` on the ordinary type. Task 0003 will add the
  ordinary typed lifecycle after its logical binding and result contracts are specified.
- Delegate `isClosed()` and `close()` to the exact privately owned `AdvancedEngine`; do not expose
  that delegate or any conversion to the advanced surface.
- Preserve `AdvancedEngine.takeOwnership(CpuBackendIntegration)` and the complete task-0001
  advanced API and behavior unchanged.
- Add focused API-shape, lifecycle, fresh-instance, cleanup-failure, and real standard-factory
  integration coverage.
- Update the Engine package Javadoc and the public, compile, and runtime API explanations so they
  distinguish the current ordinary construction-only surface from both the current advanced
  lifecycle and task 0003's planned typed lifecycle.
- Require a separate clean documentation-focused pass after implementation to finalize Javadocs,
  explanatory documentation, glossary impact, and planning evidence in the same overall change.

## Out of scope

- Any ordinary `compile`, `prepare`, `borrow`, `run`, input-binding, publication, result-access,
  or host-materialization method. Typed logical binding and published-result access are task 0003;
  host materialization is task 0004.
- Returning, accepting, or exposing `AdvancedEngine`, `AdvancedCompiledGraph`,
  `AdvancedPreparedExecution`, `AdvancedRunResult`, `CpuBackendIntegration`,
  `FunctionalGradientRequest`, `CompileArtifacts`, `PreparedExecution`, `RunResult`, `RunState`,
  `BufferRepresentation`, or another inward integration contract from any public or protected
  `Engine` signature.
- A factory on `AdvancedEngine`, a static utility namespace returning `AdvancedEngine`, or an
  ordinary-to-advanced escape hatch.
- More than one built-in backend, caller-selected backend sets, a backend builder, registration
  API, borrowed backend ownership, runtime backend selection, or successful mixed-backend or
  multi-partition schedule composition.
- Metal or CUDA lifecycle construction or support claims. Their modules still have no supported
  lifecycle adapter.
- Backend discovery, reflection, classpath or annotation scanning, `ServiceLoader`, a service
  locator, registry singleton, mutable process-global Engine instance, or hidden global backend
  state.
- `CompileConfig`, `PrepareConfig`, `RunOptions`, another configuration aggregate, or tuning
  integration. Existing Config 0006A and tools/tuning 0001 remain Engine 0007 work.
- Successful zero-node/pass-through preparation or any change to the CPU adapter's existing
  one-non-empty-maximal-CPU-partition domain.
- One-shot forward or backward convenience, Tensor execution methods, `Tensor.backward`, mutable
  Tensor gradient state, or compiler/backend/runtime behavior.
- Changes to Gradle declarations, module dependencies, architecture rules, ADRs, inward Java
  modules, concrete backend implementation, generated code, backend conformance, or performance
  policy.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core lifecycle, Core invariants,
  Engine, concrete backends, dependency rules, and the runtime service-locator and reflective-
  discovery prohibitions.
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Planning guide](../../../planning-guide.md)
- [Engine master plan](../master-plan.md)
- [Engine 0001 advanced lifecycle foundation](0001-advanced-composition-and-representation-level-lifecycle-foundation.md)
- [CPU 0010F supported lifecycle adapter](../../../backends/cpu/tasks/0010f-supported-cpu-lifecycle-integration-adapter.md)

## Architecture constraints

- Engine remains the explicit outer composition root and may directly construct known built-in
  adapters in a fixed order. This is explicit composition code, not discovery.
- The current built-in inventory is exactly CPU because `CpuBackendIntegration` is the only
  supported complete lifecycle adapter. Existing Metal and CUDA project dependencies and marker
  types establish no executable lifecycle capability.
- Compiler and Planning select eligible ownership before Runtime execution; CPU Prepare selects
  CPU-internal routes. Standard construction must not perform runtime backend or route selection.
- Runtime, Prepare, Compiler, Model, Planning, Config, and concrete backends retain their current
  ownership. No inward module may depend on Engine, and Engine must import no concrete backend
  `.internal` package.
- Ordinary public signatures must remain Engine-owned and must not expose CPU or cross-module
  lifecycle SPI. The current task therefore stops at construction and lifetime; task 0003 owns the
  first ordinary compile/prepare/run-related types and signatures.
- `AdvancedEngine.takeOwnership(...)` remains the explicit caller-supplied advanced composition
  path. Standard construction neither replaces it nor changes its ownership, failure, concurrency,
  close, or representation-level semantics.
- The existing package-private `CpuEngineBackendComposition` remains the sole production
  realization of `EngineBackendComposition`. Task 0002 adds no generic backend hierarchy,
  multi-assembler abstraction, registry, or schedule-contribution contract.
- If implementation needs a second adapter, a shared mixed-schedule contract, a new project
  dependency, a public inward type, an advanced-surface change, or an architecture rule change,
  stop and report the exact prerequisite rather than expanding this task.

## Package impact

Existing package used and changed:

- `io.github.pho001.synaptik.engine` remains the deliberate public facade package and gains one
  distinct ordinary composition owner beside the advanced task-0001 surface.

Packages added or changed:

- No Java package is added.

Type placement:

- `io.github.pho001.synaptik.engine.Engine` — the root package owns the ordinary standard
  composition and its lifetime because Engine is the architecture's public composition root.

No existing type moves. No public `spi`, `standard`, `backend`, `config`, or `internal` package is
created.

## Affected files

Expected production paths:

- add `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java`
- update `modules/engine/src/main/java/io/github/pho001/synaptik/engine/package-info.java`

Expected focused test paths:

- add `modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineStandardCompositionTest.java`
- update `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/AdvancedEnginePublicShapeTest.java`
- add `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineStandardCompositionIntegrationTest.java`

Expected explanatory documentation paths:

- `docs/api/public-api.md`
- `docs/api/compile-api.md`
- `docs/api/runtime-api.md`

Expected planning paths:

- `docs/planning/modules/engine/tasks/0002-standard-built-in-composition.md`
- `docs/planning/modules/engine/master-plan.md`
- `docs/planning/roadmap.md`

No build file, architecture contract, ADR, other architecture explanation, glossary, backend
guide, inward Java source, backend-conformance test, or later Engine task specification is expected
to change. The documentation pass must record reasoned no-change conclusions for those areas.

## Maximum scope

This task may create or modify at most 11 paths:

- two Engine production/Javadoc paths;
- two Engine test paths;
- one integration-test path;
- three explanatory API documentation paths; and
- exactly three Engine task/master/roadmap planning paths.

If a build change, another production type, another test owner, another documentation path, a
twelfth path, or any inward-module change is required, stop and propose the smallest prerequisite
or follow-up task.

## Exact API shape

`Engine` is public, final, and implements `AutoCloseable`. It has no public or protected
constructor and exactly these public declarations:

```java
public static Engine standard();
public boolean isClosed();
@Override public void close();
```

The class retains exactly one private final `AdvancedEngine` delegate. It may have one
package-private constructor with this exact shape for standard construction and same-package
lifecycle tests:

```java
Engine(AdvancedEngine delegate);
```

Successful package-private construction transfers cleanup ownership of the non-null delegate to
the new `Engine`. A null delegate throws `NullPointerException("delegate")` and transfers nothing.
No accessor exposes the delegate. No additional top-level type is authorized.

`Engine.standard()` directly calls `CpuBackendIntegration.open()` and transfers the returned exact
adapter once through `AdvancedEngine.takeOwnership(...)`. It returns a new `Engine` around that
owner. No public or protected declaration on `Engine` may name an advanced Engine type, CPU type,
Compiler type, Prepare type, Runtime type, Config type, backend-contract type, Planning type, or
Trace type.

Task 0001's four public advanced types and all their public declarations remain byte-for-byte
source compatible; this task adds no overload or factory to `AdvancedEngine`.

## Lifecycle, concurrency, and failure semantics

Standard construction:

1. Every `standard()` invocation attempts one fresh `CpuBackendIntegration.open()` call. It does
   not cache, reuse, publish, or mutate a process-global Engine or adapter.
2. Once CPU opening succeeds, ownership transfers exactly once into the private advanced owner and
   then into the returned ordinary Engine. The caller never receives or owns the adapter or the
   advanced delegate.
3. If CPU opening fails, its exact unchecked failure propagates and no `Engine` is returned.
4. If any unchecked failure occurs after CPU opening but before the ordinary Engine is returned,
   construction attempts to close the current owner exactly once. The original failure instance
   remains primary; a distinct cleanup failure is suppressed on it. No partially constructed
   Engine or adapter escapes.
5. Each successful call returns a distinct Engine with an independent owned CPU composition.

Ordinary lifetime:

- `isClosed()` delegates to the exact advanced owner and has the same point-in-time, thread-safe
  observation: it becomes true when closure begins.
- `close()` delegates once per caller invocation to the advanced owner's already-proved
  concurrent/idempotent close protocol. The underlying CPU integration is physically closed at
  most once; concurrent and repeated callers wait as task 0001 specifies and observe the same
  retained cleanup failure instance when cleanup failed.
- A direct close failure or `Error` preserves its exact identity and suppression tree. `Engine`
  adds no exception translation in this task.
- This wrapper adds no new lock, worker, background thread, shutdown hook, mutable global state,
  or Runtime hot-path work.
- Because task 0002 exposes no ordinary operation method, post-close operation precedence is not
  extended here. Task 0003 must define it when it adds the ordinary lifecycle.

## Acceptance criteria

- The exact `Engine` modifiers, constructor visibility, field ownership, and three-method public
  surface match this specification.
- A caller can construct and close a standard Engine while importing only
  `io.github.pho001.synaptik.engine.Engine`; it does not name or supply a CPU adapter.
- Each standard call owns a fresh independent CPU-only composition, and closing one standard
  Engine does not close or change another.
- Successful standard construction transfers and hides the exact CPU integration and advanced
  owner; no public or protected signature, method, field, constructor, generic bound, thrown type,
  or Javadoc link exposes either one or any inward collaboration contract.
- Construction rollback, close delegation, concurrent/repeated close, exact failure identity, and
  suppression behavior follow the lifecycle semantics above.
- The complete task-0001 `AdvancedEngine.takeOwnership(...)` path, its four public types, and its
  representation-level behavior remain unchanged.
- The fixed standard inventory is exactly CPU. Metal/CUDA markers, a second fake adapter, mixed or
  multiple partitions, backend selection controls, discovery, reflection, `ServiceLoader`, a
  service locator, registry/global singleton, and Runtime backend selection are absent.
- `Engine` has no compile, prepare, borrow, run, input, output, result, materialize, execute,
  backward, tuning, or configuration method. Task 0003 remains the sole next owner of ordinary
  typed logical input and publication access.
- No Tensor API, Config aggregate, inward public contract, concrete backend implementation,
  Gradle declaration, dependency edge, architecture contract, ADR, generated code, or performance
  behavior changes.
- Public/package Javadocs explain the construction-only status, CPU-only inventory, ownership,
  independent-instance, close, concurrency, and failure contracts without promising task 0003 or
  fictional Metal/CUDA behavior.
- Public, compile, and runtime API pages distinguish current ordinary construction from the
  existing advanced compile/prepare/run seam and still label ordinary typed lifecycle work as
  planned.
- A separate documentation-focused clean context finalizes affected Javadocs, explanatory API
  documentation, glossary no-change reasoning, and planning evidence before completion.

## Tests / validation

Implementation-focused checks:

```bash
./gradlew :modules:engine:test --tests '*Engine*'
./gradlew :testing:integration-tests:test --tests '*EngineStandardCompositionIntegrationTest*'
./gradlew :testing:architecture-tests:test --tests '*EngineCompositionContractTest*'
./gradlew :modules:engine:test
```

Focused Engine tests must lock the exact ordinary and unchanged advanced public shapes from a
distinct package. Same-package tests must cover constructor ownership, null-before-transfer,
delegated lifecycle status, independent owners, concurrent/repeated close, exact cleanup-failure
identity, and absence of an advanced escape hatch. The integration test must call
`Engine.standard()` and close it using only the public ordinary Engine type; it must not import or
construct `CpuBackendIntegration` or an advanced Engine type.

Documentation-focused pass:

```bash
./gradlew :modules:engine:javadoc
git diff --check
```

Manual and structural validation:

- compile a distinct-package Java fixture that imports only `Engine`, calls `standard()`, observes
  lifecycle state, and closes it;
- inspect `javap -public` for `Engine` and the four task-0001 advanced types, proving the exact new
  surface and unchanged advanced surface;
- scan public/protected `Engine` declarations and generated Javadoc for advanced Engine, CPU,
  Compiler, Prepare, Runtime, Config, Planning, backend-contract, Trace, and `.internal` type
  leakage;
- scan Engine production for reflection, classpath/annotation scanning, `ServiceLoader`, service
  location, registry/global singleton state, Runtime backend selection, and Metal/CUDA lifecycle
  claims;
- verify all Gradle files and the exact Engine dependency inventory are unchanged;
- validate changed Markdown headings, local links and anchors, fences, terminology, trailing
  whitespace, and final newlines;
- verify the exact 11-path allowlist and ceiling;
- confirm task 0001 remains `Complete`, task 0002 is synchronized across this specification,
  master plan, and roadmap, tasks 0003–0008 remain `Draft`, and no task 0003 specification exists;
  and
- run `git diff --check` on the final combined change.

Repository-wide validation is deferred to Engine 0008's lifecycle capability checkpoint and CI.
This task adds one outer-facade production type without changing dependencies, shared contracts,
or executable backend behavior. Backend conformance is unchanged, and no benchmark is required
because construction and close are cold constant-count orchestration outside Runtime execution.

## Dependencies

- Engine 0001 advanced composition and representation-level lifecycle foundation — Complete.
- Compiler 0006B3, Prepare 0003–0004, Runtime through 0014, and CPU 0010F — Complete as recorded by
  task 0001 and its repository checkpoint.
- `CpuBackendIntegration.open()` — current supported CPU lifecycle construction with portable
  fallback and bounded optional OpenBLAS route setup.
- `CpuEngineBackendComposition` and the `AdvancedEngine` ownership/lifecycle gate — current
  package-private Engine composition and public advanced owner used internally without change.
- Metal and CUDA lifecycle adapters — absent and intentionally not prerequisites because this
  task's truthful built-in inventory is CPU-only.

All prerequisites required for the bounded CPU-only standard path were demonstrably complete, and
the implemented and documented result satisfies this task.

## Follow-up tasks

- Engine 0003: add the ordinary typed logical input binding and published-result access surface on
  `Engine` without exposing advanced handles, Runtime representations, or Compiler/Prepare SPI.
- Engine 0004: add explicit host materialization after typed publication access exists.
- Engine 0005–0006: add Engine-owned one-shot forward and scalar-objective backward convenience.
- Engine 0007: optionally compose Config 0006A and tools/tuning 0001 before preparation.
- Engine 0008: run the lifecycle capability checkpoint.
- A future standard-inventory expansion requires another supported lifecycle adapter and a
  separately justified complete mixed-schedule contribution contract. It must not be inferred
  from current Metal/CUDA placeholder projects.

## Architecture impact

Expected impact: None.

This task realizes the existing rule that Engine is the explicit composition root. Direct
construction of one fixed known CPU adapter is explicit registration in Engine-owned code; it is
not reflective discovery, service location, or Runtime selection. It uses only the existing
Engine-to-CPU dependency and existing task-0001 ownership path. `ARCHITECTURE.md`, focused
architecture explanations, ADRs, dependencies, and architecture tests require no semantic
change. If implementation reveals otherwise, stop and report the conflicting rule and required
decision.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in /Users/phujka/IdeaProjects/Synaptik on Engine task 0002.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/modules/engine/master-plan.md, and
docs/planning/modules/engine/tasks/0002-standard-built-in-composition.md in full. Read the focused
Engine lifecycle/module/dependency documentation and inspect the final task-0001 Engine source,
tests, CPU integration adapter, current entry surfaces, and Gradle declarations named by the task.

Implement task 0002 exactly as specified. Add only the distinct ordinary CPU-only standard
construction-and-close surface. Preserve the advanced API and do not implement task 0003 binding
or results, host materialization, one-shot or backward convenience, tuning, Config aggregates,
mixed backends, discovery, Runtime selection, Tensor execution methods, build changes, or
architecture changes. Stop and report if the exact task cannot fit its architecture or 11-path
ceiling. Do not commit or push.

Run the focused Engine, integration, architecture, module, public-shape, forbidden-mechanism,
scope, status, and whitespace validation specified by the task. After executable Java stabilizes,
hand the actual diff and exact test evidence to a separate clean documentation-focused
agent/thread in the same overall change. That pass must follow
docs/developer-guide/documentation-rules.md, independently finalize affected Javadocs, API
documentation, glossary impact, and planning evidence, and reuse successful Java tests unless
executable behavior changes.

Update this task with local decisions, known limitations, implementation notes, exact validation
evidence, documentation-context identity/result, completion summary, and final status. Mark
Complete only after every acceptance criterion and the documentation pass succeed.
```

## Local decisions

- A distinct ordinary `Engine` is selected over an `AdvancedEngine` factory because the advanced
  type's existing public lifecycle necessarily exposes Compiler and Runtime integration contracts.
- Task 0002 exposes construction and lifetime only. Pulling compile/prepare/run methods forward
  would either leak those contracts or prematurely decide task 0003's typed binding/result API.
- The standard inventory is exactly CPU and is constructed afresh per Engine. Current Metal/CUDA
  placeholders provide no honest second entry.
- The existing advanced owner is reused privately so task 0002 does not duplicate lifecycle,
  concurrency, cleanup, or backend-composition logic.

## Known limitations

- The ordinary Engine is construction-only until task 0003 adds typed lifecycle methods. It cannot
  yet compile, prepare, run, bind inputs, or expose results.
- The standard composition is CPU-only and inherits the advanced CPU path's current rejection of
  zero-node, mixed-owner, and multiple-partition preparation once later ordinary lifecycle methods
  delegate to it.
- Standard construction applies the CPU integration's current exact/default policy and optional
  internal OpenBLAS setup. It accepts no caller backend or tuning policy.

## Validation evidence

- Implementation context `01a0a004-1539-7ff2-a423-3fe8e4c11bc5` supplied the stabilized
  executable evidence. The focused and full Engine test commands completed with 12/12 tests; the
  public standard-factory integration command completed with 1/1 test; and a fresh
  `./gradlew :testing:architecture-tests:test --tests '*EngineCompositionContractTest*' --rerun-tasks`
  completed with 1/1 test. Its distinct-package fixture importing only `Engine` compiled and ran.
  Public/private/bytecode `javap` inspection, forbidden-mechanism scans, source-compatibility,
  exact scope/status, Gradle-unchanged, and diff checks passed. This documentation pass reused
  that Java evidence because it changed no executable Java tokens.
- Documentation context `01a0a00b-e021-76a3-8272-fcef47e1b026` independently reviewed the final
  Engine source, tests, generated API surface, API explanations, planning records, architecture
  boundaries, documentation profiles, and complete uncommitted diff. It finalized only the two
  permitted Javadocs and the three API pages, then synchronized this task, the Engine master plan,
  and the roadmap.
- `./gradlew :modules:engine:javadoc` passed with 14 actionable tasks, 2 executed and 12
  up-to-date. Generated `Engine.html` exposes only the ordinary construction/lifetime contract;
  the package page distinguishes it from the current advanced representation-level lifecycle.
  Generated-page and source scans found no advanced type, CPU type, Compiler, Prepare, Runtime,
  Config, Planning, Backend Contract, Trace, or `.internal` link/signature leakage from the
  ordinary public surface.
- Fresh `javap -public` inspection showed exactly `standard()`, `isClosed()`, and `close()` on
  `Engine`; the four advanced types retained their task-0001 public shapes. `javap -private` and
  `javap -c -private` confirmed one private final `AdvancedEngine` field, one package-private
  constructor, direct fresh CPU opening, ownership transfer, rollback, and delegated lifecycle.
- Comment-stripped executable-token SHA-256 values were identical before and after documentation
  edits: `Engine.java` is
  `28d32f2dc3e1e058eeb8b30b977afc6c7451cc5a82aa33af165c3c25c85c4af4`, and
  `package-info.java` is
  `a42d004e704c25b4e5fe86e3d878f8c96a6e37fb08e54d00b77e69ba7ff9425e`.
- Targeted Markdown validation passed for headings, local links and anchors, fences, terminology,
  trailing whitespace, and final newlines in the six changed documentation/planning files. The
  exact final worktree contains all and only the 11 allowlisted paths. Task 0001 remains
  `Complete`; task 0002 is `Complete` here, in the Engine master plan, and in the roadmap; tasks
  0003-0008 remain `Draft`; no task-0003 specification exists. All Gradle files and advanced
  production sources are unchanged, and `git diff --check` passed.
- Glossary no-change conclusion: `Engine`, composition root, backend, compile, prepare, run,
  publication, and materialization already have established project meanings. Task 0002 adds one
  ordinary construction method and no new reusable project term or changed definition, so editing
  `docs/glossary.md` would add churn rather than needed terminology.

## Implementation notes

- Added one public final ordinary `Engine` that privately retains exactly one final advanced owner.
  `standard()` opens a fresh CPU integration, transfers ownership once, and rolls back the current
  owner on a later unchecked construction failure while preserving primary/suppressed identity.
- `isClosed()` and `close()` delegate to task 0001's established thread-safe lifecycle. No lock,
  worker, shutdown hook, global instance, discovery, backend registry, or Runtime work was added.
- The ordinary public shape remains construction-only. The advanced compile/prepare/borrow/run
  surface and its four public types are unchanged and do not leak through any ordinary signature.
- Documentation now presents three distinct states truthfully: current ordinary standard
  construction/lifetime, current advanced representation-level compile/prepare/run, and planned
  task-0003 typed binding/results followed by task-0004 host materialization.

## Completion summary

- Completed changes: added the distinct ordinary CPU-only `Engine.standard()` composition and
  delegated lifetime; added focused public-shape, lifecycle, independence, cleanup-failure, and
  real-factory integration coverage; finalized Javadocs and the public/compile/runtime API
  explanations; and synchronized planning evidence and status.
- Files changed or created: exactly
  `docs/api/public-api.md`, `docs/api/compile-api.md`, `docs/api/runtime-api.md`, this task,
  `docs/planning/modules/engine/master-plan.md`, `docs/planning/roadmap.md`, `Engine.java`, Engine
  `package-info.java`, `EngineStandardCompositionTest.java`, `AdvancedEnginePublicShapeTest.java`,
  and `EngineStandardCompositionIntegrationTest.java`.
- Tests and validation: reused 12/12 Engine, 1/1 standard integration, and fresh 1/1 architecture
  test evidence from implementation context `01a0a004-1539-7ff2-a423-3fe8e4c11bc5`; the
  documentation context passed Engine Javadoc, generated-page/leakage scans, fresh `javap`,
  comment-stripped token proof, Markdown, exact-path/status/Gradle/advanced-source checks, and
  `git diff --check`.
- Documentation-agent review: completed in clean context
  `01a0a00b-e021-76a3-8272-fcef47e1b026` using the General, API/Javadoc, and Planning profiles.
- Documentation impact: the three API pages now distinguish current ordinary construction from
  the current advanced lifecycle and planned typed/result/materialization work. Architecture
  explanations, backend guides, inward API pages/Javadocs, build files, architecture tests,
  backend conformance, and other modules need no update because no contract, dependency, or
  behavior in those owners changed.
- Javadoc review: `Engine.java` and `package-info.java` were finalized without changing executable
  tokens; task-0001 advanced Javadocs remain accurate and unchanged.
- Glossary impact: no edit; all terms retain established meanings and no reusable term was added.
- Unresolved issues: none for task 0002. Tasks 0003-0008 remain Draft follow-up work.
- Follow-up required: none for task 0002.

Status: Complete
