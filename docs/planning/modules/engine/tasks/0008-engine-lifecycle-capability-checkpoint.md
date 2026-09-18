# Task 0008: Engine Lifecycle Capability Checkpoint

## Status

Complete

## Goal

Close the current Engine foundation with one capability-checkpoint validation and documentation
reconciliation pass. Prove the implemented standard and advanced CPU-only lifecycle from public
composition through compilation, preparation, typed input binding, execution, publication, host
materialization, one-shot forward and scalar-objective backward convenience, optional local-
workload tuning or safe fallback, cleanup, failure handling, and concurrent-run isolation.

This task consolidates evidence for behavior delivered by Engine tasks 0001 through 0007. It does
not add another Engine behavior, API, test seam, backend route, or persistence facility.

The required NCW Conv1d, NCHW Conv2d, and NCDHW Conv3d public Engine fixtures are now valid through
Complete [Compiler 0006B6](../../compiler/tasks/0006b6-final-convolution-logical-layout-closure.md).
Model construction and ordinary inference still preserve unresolved convolution layouts, while
Compiler closes only eligible fully static final descriptors before Planning capability queries;
the CPU provider remains strict. The completed prerequisite supplies the exact positive public
Engine fixtures, so this checkpoint is now actionable. It must consume that evidence without
reimplementing the mechanism, weakening CPU admission, or substituting backend-internal tests.

## Motivation and mental model

Tasks 0001 through 0007 built the lifecycle incrementally. Their focused tests are necessary
evidence, but the project also needs one named checkpoint that verifies the assembled repository
and removes stale documentation that still describes Engine, Prepare, Runtime, and CPU execution
as unimplemented.

```text
Compiler-owned convolution logical-layout prerequisite
  -> positive public Engine Conv1d/Conv2d/Conv3d execution evidence
  -> existing Engine behavior + existing cross-module tests
  -> one fresh repository-wide validation
  -> current-status documentation reconciliation
  -> Engine foundation checkpoint closure

resolved prerequisite:
  eligible final convolution layout -> CPU owner -> positive public Engine execution
```

## Scope

- Run one fresh repository-wide Java test checkpoint and record exact outcomes.
- Confirm that the existing test inventory collectively covers:
  - `Engine.standard()` CPU-only construction and lifetime;
  - `AdvancedEngine.takeOwnership(...)`, representation-level input borrowing, reusable prepared
    execution, and concurrent runs;
  - ordinary owner-bound compile and prepare handles;
  - final Compiler-ordered `TensorId` input metadata and arbitrary-order typed input supply;
  - caller-owned host-storage snapshots, Runtime result leases, publication occurrence order,
    aliases, and detached canonical host materialization;
  - the four automatic-input `compute(...)` conveniences and their aggregate byte bound;
  - scalar-objective `backward(...)`, explicit target order, the absent positive-one seed, source-
    only constant realization, detached objective/gradients, and aggregate byte bound;
  - optional CPU local-workload tuning, cache-first behavior, fresh selected preparation, required
    failure, allowed safe-heuristic fallback, immutable translated evidence, and deterministic
    no-handoff fallback integration;
  - cleanup ordering, primary/suppressed failure identity, close races, repeated/concurrent close,
    and isolated mutable `RunState` per active run; and
  - dependency direction and the absence of Runtime backend discovery, Engine dependencies from
    inward modules, or tuning work in the Runtime hot path.
- Inspect final source and tests to ensure those claims still match implementation; do not
  duplicate stable assertions merely to create a checkpoint-named test.
- Confirm existing public Engine integration evidence positively executes representative NCW
  Conv1d, NCHW Conv2d, and NCDHW Conv3d forward requests through ordinary typed input binding,
  CPU preparation and Runtime publication. Evidence owned only by CPU-internal tests is
  insufficient.
- Reconcile directly affected documentation so it presents the current CPU-only Engine lifecycle
  as runnable and distinguishes ordinary from advanced composition, reusable from one-shot work,
  result leases from detached values, bounded local tuning from generic tuning, and implemented
  CPU behavior from planned multi-backend composition.
- Preserve the completed convolution boundary accurately: Model construction remains unresolved,
  Compiler owns bounded final static descriptor closure, and the public Engine fixtures now
  provide the required positive evidence.
- Keep this task, the Engine master plan, and repository roadmap synchronized through completion.

## Out of scope

- Any production Java, test Java, test resource, Gradle, dependency, architecture-contract, ADR,
  generated-code, or performance-evidence change.
- New Engine methods, request/result types, exception families, caching, prepared persistence,
  asynchronous execution, cancellation, streaming, typed-array results, or caller destinations.
- Layout inference/planning for convolution, relaxed CPU capability admission, implicit output
  materialization, a compiler rewrite, or a new Prepare/Runtime layout contract.
- Implementation of the blocking Compiler-owned descriptor prerequisite. Compiler 0006B6 owns
  its selected final-rewrite mechanism, exact task scope, focused tests, and public fixtures.
- A public Engine convolution test that passes only by bypassing compilation, forging descriptors,
  naming CPU internals, or manually constructing inward artifacts.
- Persistence adapters, checkpoint formats, NN integration, training sessions, optimizer work,
  generic tuning, multi-profile/occurrence tuning, graph/plan tuning, or model-plan persistence.
- A second lifecycle adapter, Metal/CUDA execution, mixed-backend composition, generic backend
  registration, discovery, `ServiceLoader`, or a Runtime service locator.
- New backend conformance or performance proof. Existing CPU evidence remains owned by its
  completed CPU tasks and is not generalized here.
- Conv3d gradients. Compiler 0006C remains a separate Draft boundary.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially the lifecycle, invariants,
  module responsibilities, dependency rules, and compile/prepare/run sections
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Planning guide](../../../planning-guide.md)
- [Repository roadmap](../../../roadmap.md)
- [Engine master plan](../master-plan.md)

## Architecture constraints

- Engine remains the explicit outer composition root. Compiler owns compilation, Prepare owns
  staged preparation and validation, Runtime owns immutable recipes plus isolated per-run state,
  and CPU owns concrete capability, lowering, storage, execution, and host copying.
- Standard composition remains one fresh Engine-owned CPU integration. Advanced composition
  remains explicit ownership of one supported CPU integration. Neither implies mixed-backend
  schedule assembly.
- Runtime executes prepared work only and performs no graph inspection, backend discovery,
  lowering, kernel selection, tuning search, or cache mutation.
- Prepared recipes remain immutable and reusable. Each invocation has one isolated mutable
  `RunState`; borrowed caller storage remains caller-owned, while result leases and detached host
  values retain their completed ownership rules.
- Model Tensors retain no execution method, gradient field, backward lifecycle, Runtime state, or
  Engine dependency.
- Optional model autotuning completes before production Runtime execution. Engine composes public
  lifecycle, tools/tuning owns measurement/cache/ranking/evidence, and CPU owns typed candidates
  and selected preparation.
- Planning chooses backend ownership and never interprets CPU routes or tuning parameters.
- Documentation may correct implementation-status claims but must not change architecture rules.
  If validation reveals a contradiction or required inward contract change, stop rather than
  editing `ARCHITECTURE.md` or inventing a decision.

## Dependency and readiness audit

### Complete dependencies

- Engine 0001–0007 are Complete, including 0005A and 0006A.
- Compiler 0006B3–0006B5 are Complete and supply the complete compile entry, stable caller-input
  identity bindings, and source-only published-constant descriptor closure.
- Prepare 0003, 0003A, 0004, and 0005 are Complete and supply staged preparation, the partition-
  local DAG, opaque tuning handoff, and producerless published-constant assignment.
- Runtime 0010, 0012, 0014, and 0015 are Complete and supply prepared execution, shared-throwable
  cleanup, architecture enforcement, and leased publication-representation access.
- CPU 0008 and 0008A are Complete for internal Conv2d and dimensional-convolution execution; CPU
  0010F–0010I are Complete for lifecycle composition, host snapshots, source-only constants, and
  the local-workload tuning adapter.
- Config 0006A and tools/tuning 0001 are Complete for the bounded optional tuning flow.

### Satisfied convolution prerequisite

Complete Compiler 0006B6 now supplies Planning convolution occurrences with final resolved
layouts exactly when the Conv2d/Conv3d result is fully static and was unresolved. Planning passes
those descriptors unchanged in `OperationCapabilityQuery` and owns only capability evaluation and
backend selection; CPU still admits only resolved, injective convolution outputs. The Compiler
rewrite runs after final optimization, validation, and published-constant closure. Only a direct
axis-two squeeze of a Conv2d result newly closed by that invocation receives the exact view layout
needed by visible Conv1d composition. Model construction and dynamic Shapes remain unresolved,
Planning remains backend-neutral, and the exact public Engine numerical fixtures pass.

Task 0008 is therefore actionable as a validation-only task. It does not choose or implement
another closure rule, mutate Model semantics, relax CPU capability, or add a Planning layout
policy.

### Convolution fixture audit

| Family | Current Model/Compiler form | Current CPU evidence | Checkpoint requirement |
|---|---|---|---|
| NCW Conv1d | visible `EXPAND_DIMS -> CONV2D -> SQUEEZE` composition | exact composition executed in CPU tests | Positive public Engine forward execution after the Conv2d boundary is admissible. |
| NCHW Conv2d | one first-class flat operation | direct grouped CPU execution exists | Positive public Engine grouped forward execution with typed binding and publication. |
| NCDHW Conv3d | one first-class flat forward operation | direct grouped CPU execution exists | Positive public Engine grouped forward execution; backward remains deferred. |

These are checkpoint requirements, not a claim that every Model expression with a backend-
internal kernel must compile through standard Engine. Compiler 0006B6 satisfied the former unmet
exit condition by closing the exact final-descriptor boundary and passing the public fixtures.
This checkpoint consumes that evidence without broadening it; NN convolution integration remains
ordered afterward.

## Package impact

No Java package changes. Documentation describes existing public Engine and inward-owner types
without adding, moving, or changing any type.

## Affected files

The checkpoint may modify only these fourteen documentation/planning paths:

1. `docs/index.md`
2. `docs/getting-started.md`
3. `docs/architecture/current-architecture-plan.md`
4. `docs/architecture/lifecycle.md`
5. `docs/architecture/module-boundaries.md`
6. `docs/architecture/runtime-prepare-backend-boundary.md`
7. `docs/api/public-api.md`
8. `docs/user-guide/compiling-graphs.md`
9. `docs/user-guide/preparing-execution.md`
10. `docs/user-guide/running-models.md`
11. `docs/user-guide/backend-selection.md`
12. this task specification
13. `docs/planning/modules/engine/master-plan.md`
14. `docs/planning/roadmap.md`

Review without modification: all Java/Javadoc, tests, resources, build files,
`ARCHITECTURE.md`, ADRs, glossary, other API/backend pages, other plans, and other task specs.
Record reasoned no-change conclusions for each plausibly affected category.

## Maximum scope

At most the exact fourteen paths above. No Java, test, resource, build, architecture-contract,
ADR, or glossary path is authorized. If accurate documentation needs another path, amend this
specification and its ceiling before editing. If validation needs executable or architectural
change, stop and report the owning gap.

## Acceptance criteria

- One fresh repository-wide Java test run passes with no Java change before or after it.
- Existing Engine, integration, and architecture reports map to every lifecycle claim in Scope;
  missing evidence fails the checkpoint rather than becoming a prose assertion.
- Standard and advanced composition remain CPU-only, explicitly owned, closeable, and free of
  discovery or process-global service location.
- Reusable compile/prepare/run preserves owner identity, ordered typed input metadata, arbitrary
  input order, caller ownership, reusable prepared state, isolated runs, ordered publication
  occurrences, aliases, and result leases.
- Host materialization remains explicit, bounded, canonical, detached, occurrence-authenticated,
  and valid after result/Engine closure.
- `compute(...)` and `backward(...)` retain automatic leaf discovery, Compiler-authoritative
  binding order, fresh lifecycle work, aggregate preflight, detached results, and exact cleanup
  without caching or Tensor-owned execution state.
- Optional tuning retains its bounded CPU-only request/outcome, caller model identity, occurrence-
  0/partition-0/weight-1 mapping, cache-first trials, fresh production preparation, required
  failure, and explicit safe fallback outside Runtime.
- Public Engine integration tests positively execute representative NCW Conv1d, NCHW Conv2d, and
  NCDHW Conv3d forward requests through compilation, CPU ownership, preparation, Runtime
  execution, ordered publication, and detached materialization. CPU-internal execution evidence
  alone does not satisfy this criterion.
- Architecture tests enforce applicable direction, including `engine -> tools/tuning`, no reverse
  edge, backend independence from Engine, and Runtime independence from backends and Engine.
- Documentation entry points, status pages, public API status, and four lifecycle guides describe
  current behavior rather than obsolete model-only or planned-Engine state.
- Documentation promises no Config aggregates, generic backend registration, mixed-backend
  execution, persistence, training orchestration, or generic tuning.
- The convolution evidence is positive and public without bypassing compilation, forging
  descriptors, importing CPU internals, or manually constructing inward artifacts.
- Javadocs are reviewed and generated without source edits; the glossary has a reasoned no-change
  conclusion because no reusable term or definition changes.
- Exactly the fourteen authorized paths or a documented subset changes. Task/master/roadmap
  statuses agree, Markdown validation passes, and `git diff --check` passes.
- A separate clean documentation-focused context finalizes the prose. Task 0008 becomes Complete
  only after all evidence and its completion summary are recorded.

## Tests / validation

After the blocking prerequisite and positive public integration fixtures are Complete, this
capability checkpoint owns one fresh repository-wide run rather than repeating completed focused
suites separately:

```bash
./gradlew test --rerun-tasks
```

Record the outcome and exact aggregate counts from XML reports. Confirm execution of existing
Engine unit suites, `EngineAdvancedLifecycleIntegrationTest`,
`EngineStandardCompositionIntegrationTest`, `EngineTypedLifecycleIntegrationTest`,
  `EngineModelAutotuningIntegrationTest`, the prerequisite-owned public dimensional-convolution
  integration coverage, and `EngineCompositionContractTest`. Do not rerun the same successful
  suites in the documentation context unless executable Java changes or the repository run omitted
  them.

After the documentation-focused pass:

```bash
./gradlew :modules:engine:javadoc
git diff --check
git status --short -uall
```

Also validate local links/anchors, unique headings, balanced fences, LF/final newlines, trailing
whitespace, current/planned wording, exact changed-path scope, empty staging, and status agreement.
Do not run Java tests while creating this planning specification.

## Documentation and Javadoc impact

Apply General style to the index/getting-started pages, Architecture style to the four architecture
explanations, API/Javadoc style to public API status and Javadoc review, User-guide plus Example
profiles to the workflows, and Planning style to the three planning records.

Replace obsolete model-only and planned-lifecycle statements with a current mental model and
runnable CPU-only examples. Preserve ordinary/advanced, reusable/one-shot, lease/detached-value,
local/generic-tuning, and current/planned distinctions. No Javadoc edit is expected because no API
or behavior changes. Generate and inspect Engine Javadoc and record why it remains accurate. The
glossary should remain unchanged because all relevant terms already have established meanings.

## Risks

- Treating a passed repository suite as proof of untested behavior.
- Expanding stale-status cleanup into unrelated documentation rewrites.
- Presenting CPU-internal convolution evidence as public Engine execution or deleting the positive
  convolution exit condition.
- Converting a checkpoint into layout-resolution, multi-backend, persistence, training, generic-
  tuning, or performance work.
- Repeating successful Java suites in the documentation context.

## Dependencies

- Engine 0001–0007 — Complete.
- Compiler 0006B3–0006B5 — Complete.
- [Compiler 0006B6](../../compiler/tasks/0006b6-final-convolution-logical-layout-closure.md) —
  Complete, including its exact positive public Engine fixtures.
- Prepare 0003, 0003A, 0004, and 0005 — Complete.
- Runtime 0010, 0012, 0014, and 0015 — Complete.
- CPU 0008, 0008A, and 0010F–0010I — Complete.
- Config 0006A and tools/tuning 0001 — Complete.
- Compiler-owned convolution logical-layout closure before Planning capability admission —
  complete in Compiler 0006B6.
- Positive public Engine NCW Conv1d, NCHW Conv2d, and NCDHW Conv3d forward integration evidence —
  supplied by Compiler 0006B6's three passing integration tests.

## Follow-up tasks

- This checkpoint is complete. Planning remains the consumer of final descriptors, CPU remains
  strict, and task 0008 introduced no competing closure rule or Java fixture.
- Existing Draft NN 0025 owns dimensional-convolution layer integration, followed by NN 0025A's
  end-to-end user-capability checkpoint; neither may substitute CPU-internal evidence for public
  Engine execution.
- Tools/tuning 0002 remains later graph/plan tuning.
- Persistence, NN, Training, Metal/CUDA, and mixed-backend composition remain with their owners.
- Compiler 0006C remains the separate Draft Conv3d gradient boundary.

## Architecture impact

Expected impact: None. This checkpoint verifies and documents current contracts. It changes no
owner, dependency, module boundary, API, Runtime behavior, or backend behavior. If validation
finds a conflict, stop and report it rather than editing the architecture contract.

## Implementation prompt

Use this prompt in a separate clean checkpoint context:

```text
You are working in /Users/phujka/IdeaProjects/Synaptik on Engine task 0008. Do not use GSD,
commit, or push.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, roadmap, Engine
master plan, and task 0008 in full. Read the completed dependencies and inspect current Engine,
Compiler, Prepare, Runtime, CPU, and tuning source/tests/reports/Javadocs plus directly affected
documentation.

Verify that Compiler 0006B6 and its exact positive public Engine NCW Conv1d, grouped NCHW Conv2d,
and grouped NCDHW Conv3d fixtures remain Complete, then execute the checkpoint exactly. Change
no Java, tests, resources, Gradle, architecture
contract, ADR, glossary, or path outside its fourteen-path allowlist. Run one fresh repository-
wide checkpoint, map executed tests to every claim, and stop if evidence or executable change is
required. Require positive public dimensional-convolution execution; do not substitute backend-
internal evidence.

Hand stable evidence and documentation scope to a distinct clean documentation-focused context.
That pass follows documentation-rules.md, edits only authorized prose/planning paths, generates
Engine Javadoc, validates Markdown/scope, and does not repeat successful Java tests. Update task
evidence, notes, completion summary, and status only after every gate passes.
```

## Local decisions

- Return 0008 to `Ready` only after Compiler 0006B6 and its positive public dimensional-
  convolution fixtures complete; that prerequisite is now satisfied.
- Use existing tests plus one fresh repository run. A new checkpoint Java test would duplicate
  stable lifecycle assertions; Compiler 0006B6 owns the missing public convolution fixtures before
  this documentation-only checkpoint runs.
- Include documentation reconciliation because entry points and lifecycle guides still claim
  Engine or concrete execution is absent.
- Preserve public convolution success as an exit condition. Complete Compiler 0006B6 now closes
  eligible final layouts before ownership planning and supplies the required public fixtures.

## Known limitations

- Public composition is CPU-only and accepts one non-empty maximal CPU partition. Zero-node and
  mixed/multi-owner preparation remain unsupported.
- Ordinary compilation uses fixed settings; Config aggregate facades remain planned.
- Model autotuning is bounded to one CPU-local handoff and is not generic graph/plan tuning.
- Dynamic or partially dynamic convolution results remain unresolved; the required fully static
  public Engine convolution executions are now available as checkpoint input.
- No persistence, NN execution facade, training session, optimizer step, or second backend is
  delivered.

## Validation evidence

- Checkpoint evidence context: `01a0b444-f257-7651-b90e-9a89a5c78580`. Its one actual repository
  checkpoint, `./gradlew test --rerun-tasks`, completed successfully in 2m 52s with all 70
  actionable tasks executed. The 492 XML reports record 3,188 tests, zero failures, zero errors,
  and 28 skips. Every skip belongs to optional CPU performance/evidence/persistence coverage;
  every Engine, integration, architecture, Compiler convolution, Prepare, Runtime, and tuning
  report consumed here has zero skips. No Java test was rerun in this documentation context.
- The seven Engine unit suites report 67 tests. Named cross-module reports contain
  `EngineAdvancedLifecycleIntegrationTest` 2, `EngineStandardCompositionIntegrationTest` 1,
  `EngineTypedLifecycleIntegrationTest` 6, `EngineModelAutotuningIntegrationTest` 1,
  `EngineConvolutionIntegrationTest` 3, and `EngineCompositionContractTest` 1. All report zero
  failures, errors, and skips.
- Supporting green report sets contain all nine architecture tests, 42 Compiler convolution
  closure/GraphCompiler/Conv3d tests, 24 tools/tuning tests, 52 Prepare tests, and 148 Runtime
  tests, again with zero failures, errors, or skips. These reports map the task's ownership,
  reusable lifecycle, isolated-run, cleanup, tuning, and dependency claims to executed evidence.
- The three public convolution fixtures use only Model and Engine APIs. They execute NCW Conv1d,
  grouped NCHW Conv2d, and grouped NCDHW Conv3d forward requests through reusable
  compile/prepare/run plus one-shot compute, reversed typed input order, publication, and detached
  materialization. They import no CPU internals and construct no inward artifacts. Conv3d
  gradients remain outside the claim.
- Documentation-focused context: `01a0b44b-f58b-7d42-b4cb-f33bd50a2e13`. It read the required
  architecture and planning contracts, General/Architecture/API/Javadoc/User-guide/Example/
  Planning profiles, all eleven affected explanatory pages, relevant glossary entries, public
  Engine source/Javadocs, and the saved unit/integration/architecture reports. It changed no
  executable or authoritative architecture artifact.
- `./gradlew :modules:engine:javadoc` ran exactly once after explanatory prose was final and
  passed: `BUILD SUCCESSFUL in 1s`, 15 actionable tasks, one executed and 14 up-to-date. Inspection
  of the generated Engine, RunResult, HostTensorValue, ModelAutotuning request/preparation, and
  package pages confirmed current ownership, CPU-only, lease, detached-value, one-shot, and
  tuning boundaries. Existing source Javadocs remain accurate; no source edit was needed.
- `/tmp/validate_synaptik_markdown.py` passed all fourteen authorized paths, checking local links
  and anchors, unique headings, balanced fences, LF line endings, terminal newlines, and trailing
  whitespace. Manual review checked current/planned language and examples against public source
  signatures. Final scope/status/staging and `git diff --check` checks also passed.

## Implementation notes

- Reconciled the documentation index, contributor start, architecture status/lifecycle/boundaries,
  public API status, and four lifecycle user guides with the current runnable CPU-only Engine.
- Preserved ordinary `Engine.standard()` versus advanced ownership, reusable prepared execution
  versus fresh one-shot calls, leased `RunResult` versus detached `HostTensorValue`, CPU-local
  workload tuning versus future graph/plan tuning, current CPU execution versus future other or
  mixed owners, and Model's unresolved convolution construction versus Compiler's bounded final
  static closure and strict CPU admission.
- Changed exactly the fourteen allowlisted documentation paths. Java, tests, resources, Gradle,
  dependencies, `ARCHITECTURE.md`, ADRs, and every other document remain unchanged.
- Javadoc source required no change because this checkpoint changes no API or behavior and the
  existing contracts already describe parameters, results, failures, ownership, closure, and
  current limitations consistently with source and tests.
- The glossary required no change because all reusable terms and distinctions used here already
  have current entries, including model autotuning, backend ownership/routes, compiled/prepared
  state, publication occurrences, RunResult, HostTensorValue, and RunState.
- Architecture/ADRs and architecture tests required no change because no owner, dependency,
  lifecycle invariant, or module boundary changed. Java/tests/build/dependencies likewise needed
  no change because this is a documentation-only checkpoint over existing behavior.
- Backend conformance and performance evidence required no change or rerun: no backend behavior or
  performance claim changed. Other documentation remains outside the focused stale-status scope.
- The Engine foundation is Complete through 0008. The existing Draft NN 0025 convolution-layer
  task, followed by NN 0025A's user-capability checkpoint, is the next dimensional-convolution
  frontier; this checkpoint creates no new task.

## Completion summary

- Completed changes: closed the lifecycle capability checkpoint and reconciled all eleven affected
  explanatory pages plus task, Engine master plan, and roadmap status.
- Files changed or created: exactly the fourteen authorized existing Markdown paths; no file was
  created.
- Tests and validation: reused the clean 3,188-test repository checkpoint with zero failures or
  errors; generated Engine Javadoc exactly once; passed Markdown, scope, staging, status, and
  whitespace validation.
- Documentation-agent review: clean context `01a0b44b-f58b-7d42-b4cb-f33bd50a2e13`, Complete.
- Documentation impact: current CPU-only compile/prepare/run/materialize, one-shot, advanced
  ownership, and bounded autotuning workflows now replace obsolete planned-only prose.
- Javadoc review: generated output passed inspection; source Javadocs remain accurate unchanged.
- Glossary impact: unchanged because established entries already cover every term and distinction.
- Convolution readiness: public NCW Conv1d, grouped NCHW Conv2d, and grouped NCDHW Conv3d forward
  fixtures pass through both reusable and one-shot public Engine paths.
- Unresolved issues: None for Engine 0008; documented future capabilities remain intentionally
  unsupported.
- Follow-up required: None for this checkpoint. Existing NN 0025 and 0025A remain the planned
  downstream frontier.

Status: Complete
