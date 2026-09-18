# Task 0006A: Representative Tuning Execution and Safe Fallback Foundation

## Status

Complete

## Goal

Add the smallest Engine-owned internal lifecycle capability needed before optional model
autotuning can be composed safely: bind one explicit representative Tensor input set in the
authoritative compiled-input order, execute any complete CPU trial recipe synchronously in a fresh
isolated Runtime state, consume completion without materializing results, clean every trial and
borrowed input wrapper deterministically, and permit ordinary CPU heuristic preparation only when
failure isolation and cleanup are known to have completed.

This task establishes package-private execution and fallback machinery only. It does not expose an
autotuning method, translate `ModelAutotuningConfig`, invoke `WorkloadTuning`, rank candidates,
access a cache, or select a CPU candidate. Engine 0007 remains the consumer that maps the public
request, obtains the CPU handoff, delegates measurement/ranking/cache work to `tools/tuning`, and
uses this task's lifecycle boundary for every requested execution and final preparation.

```text
exact compiled graph + explicit representative Tensor values
  -> validate and order by final compiled inputs
  -> snapshot caller-owned HostTensorStorage and borrow wrappers once
  -> for each later-requested warmup or timed execution:
       prepare one fresh complete trial recipe
       execute in one fresh isolated RunState through every publication
       validate publication count and close the result lease
  -> close borrowed wrappers once in reverse order
  -> after successful cleanup, freshly prepare the selected production recipe
     or, only when policy permits, one ordinary safe-heuristic recipe
```

## Scope

- Add one package-private Engine-owned representative-execution session under
  `io.github.pho001.synaptik.engine`.
- Reuse Engine 0003 binding semantics: join supplied `Tensor` values by `TensorId`, validate them
  against final `CompiledGraph.Input` descriptors, and order them only by
  `CompiledGraph.inputs()`.
- Snapshot each selected Tensor's current `HostTensorStorage` once before the first trial. Validate
  the complete input set and every storage before borrowing any wrapper.
- Borrow CPU-compatible input representations once for the synchronous session. Engine owns and
  closes only the non-owning wrappers; caller storage and memory scopes remain caller-owned and
  must remain live, accessible, and unmodified through the session.
- Execute one supplied immutable Runtime `PreparedExecution` at a time with the existing
  `PreparedExecutionRunner`. Every call reuses only the ordered borrowed-input list and creates a
  fresh isolated `RunState`; it reuses no trial result or mutable execution state.
- Treat synchronous execution through all scheduled publication actions as sufficient output
  consumption for timing. Validate the expected publication count and immediately close the
  Runtime result. Do not inspect, compare, checksum, copy, or materialize publication payloads.
- Make result cleanup part of every complete candidate-execution action. A successful action
  returns only after result cleanup succeeds.
- On trial failure, preserve the exact primary `RuntimeException` or `Error`, attempt all owned
  cleanup once, suppress distinct cleanup failures in deterministic order, and poison the session.
  Do not retry or continue with another trial.
- Make session cleanup idempotent, close wrappers in reverse acquisition order, reject execution
  after cleanup starts, and retain the first cleanup failure with later distinct failures
  suppressed.
- Define the package-private final-preparation/fallback boundary for Engine 0007:
  - prepare a selected production recipe freshly only after tuning and representative-session
    cleanup succeed;
  - never promote a timed trial recipe or `RunState` to production;
  - required tuning propagates unavailability/failure without heuristic preparation;
  - allowed fallback invokes ordinary `CpuBackendIntegration.prepare(artifacts)` exactly once,
    only after representative cleanup and failure isolation are proved complete;
  - forbid fallback for `Error`, Engine closure, ownership/lifecycle rejection, or any failed or
    uncertain cleanup; and
  - if ordinary heuristic preparation fails, make its exact failure primary and suppress the
    earlier recoverable tuning failure when distinct.
- Add focused native-free tests for binding order and validation, storage snapshot and ownership,
  repeated isolated execution, publication completion, cleanup ordering, failure suppression,
  poisoned-session rejection, fresh selected preparation, and strict/allowed fallback decisions.
- Finalize affected Javadocs/package documentation in the mandatory separate clean documentation
  context during implementation, and review explanatory documentation and glossary impact.

## Out of scope

- a public `Engine`, `AdvancedEngine`, handle, result, request, or configuration API
- representative-value storage in `ModelAutotuningConfig`, which intentionally carries identity
  only
- model fingerprint construction, profile lookup, occurrence extraction/weight/context, or
  multi-partition aggregation
- implementing `BackendWorkloadTuning` or `ColdCandidateMeasurement`, invoking `WorkloadTuning`,
  timing, ranking, tie breaking, evidence, cache access, compatibility, codecs, or persistence
- enumerating or interpreting CPU candidate, route, representation, thread, provider, or hardware
  fields
- changing CPU, Prepare, Runtime, Config, Compiler, Model, or tuning public contracts
- output correctness comparison or host materialization
- asynchronous execution, concurrent trials, retry, timeout, cancellation, or failed-session
  recovery
- tools/tuning 0002 graph/plan search or CPU 0016 peer-route generalization
- hidden defaults, global state, service lookup, reflection, a registry, or generic callback
  framework

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants, Engine,
  Runtime, Prepare, Concrete backend modules, Performance evidence and optimization tooling, and
  Dependency rules
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Engine master plan](../master-plan.md)
- Engine [0003](0003-typed-logical-input-binding-and-published-result-access.md) and
  [0006](0006-one-shot-scalar-objective-backward-convenience.md)
- [CPU 0010I](../../../backends/cpu/tasks/0010i-supported-cpu-local-workload-tuning-composition-adapter.md)
- [Config 0006A](../../config/tasks/0006a-model-autotuning-request-configuration.md)
- [tools/tuning 0001](../../../tools/tuning/tasks/0001-exact-default-model-guided-workload-tuning-and-reusable-cache.md)
- [Prepare 0004](../../prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)

## Architecture constraints

- Engine owns logical input binding, orchestration, result consumption, and lifecycle policy.
- Runtime remains unaware of tuning. It executes one prepared recipe with ordered borrowed inputs
  in isolated per-run state through existing contracts.
- CPU owns valid candidate generation and exact trial, selected, and ordinary heuristic
  preparation. Engine must not derive CPU-private facts.
- `tools/tuning` solely owns invocation counts, timing, ranking, cache coordination, persistence,
  and evidence. This task creates only the lifecycle action that 0007 will supply to it.
- Representative values/storage are caller-owned and retained only for the synchronous admitted
  operation. Engine owns only its wrappers and trial run state.
- All work is cold and precedes production Runtime execution. No tuning branch, cache access, or
  selection may enter a prepared hot loop.
- Fallback cannot grant eligibility, treat incompatible/corrupt data as a hit, accept partial
  selection, or conceal preparation/cleanup failure.
- Stop if implementation needs Runtime tuning state, CPU ownership of representative inputs, or
  Engine interpretation of backend candidate fields.

## Package impact

Only `io.github.pho001.synaptik.engine` changes. No package, public type, or public/protected member
is added. Prefer one package-private `RepresentativeExecutionSession` plus private orchestration
methods in `AdvancedEngine`; permit a second package-private final-preparation helper only if it
makes the fallback state machine independently testable without a generic callback abstraction.

`EngineBackendComposition` continues to own borrowing and ordinary heuristic preparation.
`CpuEngineBackendComposition` may expose its retained `CpuLocalWorkloadTuning` through a concrete
package-private method in 0007; this task does not widen the shared composition seam.

## Affected files

Expected production/test paths:

- add `modules/engine/src/main/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSession.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java`
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/EngineBackendComposition.java` only
  if a named fallback operation is necessary instead of existing `prepare`
- add `modules/engine/src/test/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSessionTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineTypedLifecycleTest.java` only
  when its binding/cleanup fixture is required
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/AdvancedEngineLifecycleTest.java`
  only for admission-versus-close coverage

Expected documentation/planning paths during implementation:

- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/package-info.java`
- `docs/architecture/performance-evidence-and-tuning.md`
- `docs/architecture/runtime-prepare-backend-boundary.md` only if its lifecycle explanation is
  incomplete
- `docs/glossary.md` only for a genuinely new reusable term
- this task, [Engine master plan](../master-plan.md), and [roadmap](../../../roadmap.md)

Review without modification unless contradicted: `ARCHITECTURE.md`, ADRs, Config, tuning, CPU,
Prepare and Runtime contracts, Gradle files, architecture tests, conformance tests, and integration
tests.

## Maximum scope

At most 13 implementation paths: 3 Engine production, 3 Engine test, 1 Engine package-documentation,
3 explanatory documentation, and 3 planning paths. The glossary may replace but not increase the
documentation allowance. No build or dependency change is expected because this task does not
invoke `tools/tuning`. Stop if another module's source/public API, a new dependency, or more than
one new production type is required.

## Failure semantics

1. Engine admission and ownership checks precede representative argument inspection.
2. Input identity, descriptor, storage presence/capacity/liveness/accessibility are validated
   completely before the first borrow or trial preparation.
3. Partial borrow failure closes acquired wrappers in reverse order and preserves the borrow
   failure as primary.
4. Each action uses a fresh trial recipe and runner-created `RunState`.
5. Publication-count mismatch is trial failure, never a partial result.
6. Result closure follows every successful run and every later validation failure.
7. Any trial or cleanup failure poisons the session; it permits no later trial, selected
   preparation, or fallback through that session.
8. Repeated close is idempotent and consistently rethrows the retained exact cleanup failure.
9. Engine close waits for the synchronous admitted operation; normal rollback closes any result
   that cannot be published after closure begins.
10. Only a recoverable `RuntimeException` from tuning work is fallback-eligible. `Error`, Engine
    lifecycle/ownership rejection, and cleanup failure are not.
11. Required tuning reports missing handoff or cache/candidate/trial/selected-preparation failure
    unchanged after cleanup and performs no ordinary preparation.
12. Allowed fallback performs exactly one ordinary preparation after proven cleanup. If it fails,
    that preparation failure is primary and the earlier tuning failure is suppressed when distinct.

## Hot-path and performance constraints

- Snapshot list/storage once and borrow each required input once per session. Do not repeat Tensor
  maps, storage lookup, descriptor validation, or wrapper allocation per sample.
- Still use fresh CPU preparation and Runtime state for each complete execution.
- Do not materialize outputs or allocate canonical host byte arrays in the measured action.
- Add no hot-path tuning branch, map lookup, dispatch, reflection, measurement, or cache access.
- Prefer direct private/package-private calls over a general event/callback framework.

## Acceptance criteria

1. Engine 0006A is `Complete`, ordered after 0006/CPU 0010I and before Blocked 0007.
2. No public API or non-Engine module changes.
3. Representative inputs follow exact Engine 0003 identity/descriptor/storage/final-order rules;
   validation precedes borrowing and execution.
4. Storage is snapshotted/borrowed once, remains caller-owned, and wrappers close once in reverse.
5. Every call uses a fresh recipe/state, completes publications, validates count, closes the result,
   and returns no output value.
6. Warmup and timed calls have identical shape; Engine performs no timing.
7. Failure identity/suppression is deterministic and a failed session cannot be reused.
8. Selected production preparation is fresh and occurs after successful session cleanup.
9. Required tuning never falls back; allowed fallback calls ordinary preparation at most once and
   only after recoverable failure with proven isolation.
10. `Error`, lifecycle/ownership, and cleanup failure never fall back; fallback preparation failure
    remains visible as primary.
11. Measurement/ranking/cache/evidence/candidate interpretation stays outside Engine.
12. Native-free tests cover success and all material state/failure transitions.
13. A clean documentation context finalizes Javadocs/docs and records glossary/no-change findings.
14. Final Engine tests/Javadoc, applicable architecture test, Markdown/scope/status checks, and
    `git diff --check` pass.

## Tests / validation

```bash
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.RepresentativeExecutionSessionTest \
  --tests io.github.pho001.synaptik.engine.EngineTypedLifecycleTest \
  --tests io.github.pho001.synaptik.engine.AdvancedEngineLifecycleTest
./gradlew :modules:engine:test
```

The documentation context reuses successful Java evidence unless it changes executable behavior:

```bash
./gradlew :modules:engine:javadoc
git diff --check
git status --short -uall
```

Run focused Engine architecture tests only if `EngineBackendComposition` changes. No repository-wide,
conformance, integration, real OpenBLAS, or benchmark run is required. Validate changed Markdown
targets/anchors, unique headings, balanced fences, final newlines, trailing whitespace, statuses,
ordering, wording, and exact path scope.

## Dependencies

- Engine 0001–0006 — Complete; 0003 supplies binding and 0006 cleanup conventions.
- Runtime 0010/0015 — Complete; supply synchronous isolated execution, publications, and cleanup.
- CPU 0010I — Complete; supplies exact trial/selected preparation and ordinary heuristic prepare.
- Prepare 0004 — Complete; supplies the opaque partition handoff.
- Config 0006A and tools/tuning 0001 — Complete contracts reviewed for future consumption but not
  invoked here.

Metal, CUDA, provider 0004, CPU 0010D1/0016, and tuning 0002 are not dependencies.

## Follow-up tasks

- Engine 0007 is the next planning frontier but remains `Blocked` pending a separate clean
  reassessment. That planning pass may make it `Ready` only after it can specify the public
  representative-value request,
  stable model/occurrence identities, Config-to-tuning mapping, CPU adapter, `WorkloadTuning.tune`
  call, and selected production handle/evidence surface.
- tools/tuning 0002 remains later graph/plan tuning; CPU 0016 remains later peer-route work.

## Architecture impact

Expected impact: None. This realizes existing Engine lifecycle ownership at the authorized cold
boundary. If asynchronous backend completion requires a synchronization primitive absent from the
current synchronous CPU schedule, stop and return to architecture planning.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik Engine task 0006A. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, the planning guide, Engine master plan, and task 0006A in full,
plus referenced Engine 0003/0006, Runtime 0010/0015, Prepare 0004, CPU 0010I, Config 0006A, and
tools/tuning 0001 contracts and current affected source/tests.

Implement only the package-private Engine representative-execution/fallback foundation within its
path ceiling. Add no public API, tuning algorithm, cache work, candidate interpretation, output
materialization, user Config surface, or tools/tuning dependency. Preserve fresh trial recipe/state,
synchronous publication completion, cleanup/failure identity, and fallback only after proven
isolation. Stop for architecture uncertainty, missing synchronization semantics, another-module
change, or scope expansion.

After Engine tests stabilize, hand the exact diff/evidence to a distinct clean documentation
context in the same change. It must follow documentation-rules.md, finalize affected Javadocs and
focused docs/glossary review, update task/master/roadmap evidence, and not repeat successful Java
suites absent behavior change or concrete stale-evidence risk. Mark Complete only after all gates.
```

## Local decisions

- Use one explicit representative Tensor binding set; public profile lookup belongs to 0007.
- Reuse Engine 0003 binding rather than add tuning-specific identity/raw-representation APIs.
- Borrow once per session to avoid per-sample wrapper overhead; prepare/state remain fresh per run.
- Publication completion, not host materialization, completes current synchronous CPU execution.
- Cleanup success is prerequisite to fallback; uncertain resource state cannot continue.
- Fallback-preparation failure is primary because it is the final production-preparation failure;
  the earlier recoverable tuning failure remains suppressed context.
- Add no Engine dependency on `tools/tuning` before 0007 invokes it.

## Known limitations

- CPU-only because CPU is the sole complete Engine composition.
- One explicit representative binding set; no profile lookup, sample set, weights, or occurrence
  fingerprints.
- Lifecycle completion is validated, not numerical equivalence between candidates.
- No asynchronous device completion contract; future accelerators need an explicit prerequisite.
- No public evidence/diagnostic result; 0007 must select that boundary without exposing internals.

## Validation evidence

Implementation context `01a0b187-a358-7fe1-8c5d-e06f66b143ff` recorded these successful executable
validation results before the clean documentation pass:

- focused Engine lifecycle command: 48 tests;
- `./gradlew :modules:engine:test`: 57 tests, 0 failures, 0 errors, and 0 skips; and
- `git diff --check`: passed.

The documentation context inspected the final XML for all six Engine suites. It totals 57 tests,
including 16 `RepresentativeExecutionSessionTest` tests, with zero failures, errors, or skips. It
did not repeat Java tests because documentation changed no executable Java token and no stale-
evidence risk was found. It ran `./gradlew :modules:engine:javadoc`, relative Markdown target,
changed/new link-anchor, heading, fence, line-ending, whitespace, status, ordering, and exact-scope
checks, an executable-token source audit, and final `git diff --check`; all passed.

## Implementation notes

Implementation added one package-private `RepresentativeExecutionSession` and package-private
orchestration helpers in `AdvancedEngine`. Engine admission now precedes ownership and input
inspection and remains held across representative trials, reverse cleanup, and final selected or
fallback preparation. Exact final compiled inputs determine binding membership and order; all
current caller-owned storage associations are validated and snapshotted before the first borrow.

Each supplied complete trial recipe runs through the stateless Runtime runner, which creates fresh
run state and result resources. Engine validates complete publication count, closes every result,
and never inspects or materializes a payload. Trial and cleanup failures prevent every later trial,
selected preparation, and fallback through the session. Cleanup is single-attempt and reverse-
ordered, retains the first cleanup failure, and preserves deterministic suppression.

Selected production preparation occurs freshly after successful cleanup. Strict tuning rethrows
its recoverable failure without ordinary preparation. Allowed fallback invokes ordinary heuristic
preparation once after proven cleanup; fallback-preparation or final-publication failure is primary
with the tuning failure suppressed when distinct, while cleanup failure leaves the tuning failure
primary with distinct cleanup suppressed. A concurrent Engine close waits for the admission and
rejects a final recipe if closure wins the publication race.

The clean documentation pass strengthened affected package-private Javadocs and updated the two
focused architecture explanations plus existing glossary entries whose planned/current wording
had become stale. `package-info.java` required no change: it documents the public package surface
and already correctly says tuning is not a current API.

## Completion summary

```text
Completed changes:
- Added exact representative binding, isolated complete trial execution, deterministic cleanup,
  fresh selected preparation, and strict/allowed safe fallback under one Engine admission.
- Finalized affected package-private lifecycle/failure/ownership Javadocs and current-versus-
  planned architecture and glossary wording.
Files changed or created:
- modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java
- modules/engine/src/main/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSession.java
- modules/engine/src/test/java/io/github/pho001/synaptik/engine/RepresentativeExecutionSessionTest.java
- docs/architecture/performance-evidence-and-tuning.md
- docs/architecture/runtime-prepare-backend-boundary.md
- docs/glossary.md
- docs/planning/modules/engine/tasks/0006a-representative-tuning-execution-and-safe-fallback.md
- docs/planning/modules/engine/master-plan.md
- docs/planning/roadmap.md
Tests and validation performed:
- Reused focused Engine lifecycle evidence: 48 tests passed.
- Reused ./gradlew :modules:engine:test evidence and inspected final XML: 57 tests, 0 failures,
  0 errors, 0 skips, including all 16 representative-session tests.
- ./gradlew :modules:engine:javadoc passed.
- Relative Markdown file targets, changed/new link anchors, unique headings, balanced backtick and
  tilde fences, LF line endings, final newlines, trailing whitespace, path scope, status, and
  ordering checks passed. No changed or new Markdown link contains an anchor fragment.
- Executable-token source comparison passed: Java edits changed comments/Javadocs only.
- git diff --check passed.
Documentation review:
- Public API and package-info.java remain unchanged; the package overview remains accurate.
- Updated two focused explanatory architecture documents because their planned/current wording was
  materially stale; no authoritative architecture rule, ADR, or architecture test changed.
- Updated existing glossary entries only; no private implementation term was added.
- No dependency/build, conformance/integration, or other-module change was required.
Unresolved issues: None for 0006A.
Required follow-up: Reassess Engine 0007 in a separate clean planning context; do not mark it Ready
or create its detailed specification until the public representative-value and identity/evidence
boundaries are resolved.
Implementation context: 01a0b187-a358-7fe1-8c5d-e06f66b143ff
Documentation context: 01a0b2e0-df4b-7ae3-b16e-5afb44220b1e

Status: Complete
```
