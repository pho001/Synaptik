# Task 0008: Engine Lifecycle Capability Checkpoint

## Status

Blocked

## Goal

Close the current Engine foundation with one capability-checkpoint validation and documentation
reconciliation pass. Prove the implemented standard and advanced CPU-only lifecycle from public
composition through compilation, preparation, typed input binding, execution, publication, host
materialization, one-shot forward and scalar-objective backward convenience, optional local-
workload tuning or safe fallback, cleanup, failure handling, and concurrent-run isolation.

This task consolidates evidence for behavior delivered by Engine tasks 0001 through 0007. It does
not add another Engine behavior, API, test seam, backend route, or persistence facility.

The required NCW Conv1d, NCHW Conv2d, and NCDHW Conv3d public Engine fixtures are not valid at the
current boundary. Model and Compiler preserve unresolved logical layouts for the
convolution results, including the Conv2d occurrence inside the visible Conv1d composition, while
the current CPU capability provider admits these families only with resolved input and output
layouts. A following `contiguous()` operation resolves its own output but does not retroactively
resolve the convolution occurrence queried during ownership planning. Because positive public
execution was an existing checkpoint exit condition, replacing it with a negative audit would
silently weaken acceptance. This task is therefore blocked until Ready
[Compiler 0006B6](../../compiler/tasks/0006b6-final-convolution-logical-layout-closure.md) closes
the final logical descriptors before Planning queries backend capability and supplies the exact
public Engine execution fixtures. The checkpoint must not reimplement that mechanism, weaken CPU
admission, or treat backend-internal convolution tests as Engine execution.

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

current blocker:
  unresolved convolution result layout -> no CPU owner -> no public Engine execution
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
- Preserve the convolution blocker accurately until its prerequisite completes: Model
  construction, Compiler forward adoption, and CPU-internal execution are current, but the
  checkpoint cannot pass without positive public Engine numerical fixtures.
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

### Blocking dependency

Compiler currently hands Planning convolution occurrences whose final result descriptors retain
unresolved layouts. Planning passes those descriptors unchanged in `OperationCapabilityQuery` and
owns only capability evaluation and backend selection; CPU correctly admits only resolved,
injective convolution outputs. Ready Compiler 0006B6 owns the missing final descriptor fact. It
selects a dedicated rewrite after final optimization, validation, and published-constant closure:
fully static Conv2d/Conv3d results receive canonical contiguous logical layout, and only a direct
axis-two squeeze of a Conv2d result newly closed by that pass receives the exact view layout needed
by visible Conv1d composition. It preserves unresolved Model construction and dynamic Shapes,
keeps Planning backend-neutral and CPU strict, and adds the exact public Engine numerical fixtures.

Until Compiler 0006B6 and its public Engine evidence are Complete, task 0008 is `Blocked` rather
than an actionable validation-only task. This checkpoint does not choose or implement another
closure rule, mutate Model semantics, relax CPU capability, or add a Planning layout policy.

### Convolution fixture audit

| Family | Current Model/Compiler form | Current CPU evidence | Checkpoint requirement |
|---|---|---|---|
| NCW Conv1d | visible `EXPAND_DIMS -> CONV2D -> SQUEEZE` composition | exact composition executed in CPU tests | Positive public Engine forward execution after the Conv2d boundary is admissible. |
| NCHW Conv2d | one first-class flat operation | direct grouped CPU execution exists | Positive public Engine grouped forward execution with typed binding and publication. |
| NCDHW Conv3d | one first-class flat forward operation | direct grouped CPU execution exists | Positive public Engine grouped forward execution; backward remains deferred. |

These are checkpoint requirements, not a claim that every Model expression with a backend-
internal kernel must compile through standard Engine. The current failure is a known unmet exit
condition: the public path cannot select CPU ownership for these convolution occurrences. The
Compiler 0006B6 must close that exact boundary and pass its fixtures before this task can become
`Ready`; NN convolution integration remains ordered afterward.

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
  Ready; this task remains Blocked until it is Complete.
- Prepare 0003, 0003A, 0004, and 0005 — Complete.
- Runtime 0010, 0012, 0014, and 0015 — Complete.
- CPU 0008, 0008A, and 0010F–0010I — Complete.
- Config 0006A and tools/tuning 0001 — Complete.
- Compiler-owned convolution logical-layout closure before Planning capability admission —
  selected as Ready Compiler 0006B6; implementation remains pending.
- Positive public Engine NCW Conv1d, NCHW Conv2d, and NCDHW Conv3d forward integration evidence —
  owned by Compiler 0006B6 and pending its implementation.

## Follow-up tasks

- Complete Compiler 0006B6 before this task becomes `Ready`. Planning remains the consumer of
  final descriptors, CPU remains strict, and this task creates no competing closure rule or Java
  fixture.
- NN convolution integration and its end-to-end checkpoint remain ordered after the prerequisite
  and this Engine checkpoint; they must not consume CPU-internal evidence as a substitute.
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

Do not execute this checkpoint while it is Blocked. First verify that Compiler 0006B6 and its
exact positive public Engine NCW Conv1d, grouped NCHW Conv2d, and grouped NCDHW Conv3d fixtures are
Complete. Once the task is explicitly returned to Ready, execute the checkpoint exactly. Change
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

- Keep 0008 `Blocked`: the original positive dimensional-convolution exit condition is unmet, and
  converting it into a negative audit would hide incomplete acceptance.
- Use existing tests plus one fresh repository run. A new checkpoint Java test would duplicate
  stable lifecycle assertions; Compiler 0006B6 owns the missing public convolution fixtures before
  this documentation-only checkpoint runs.
- Include documentation reconciliation because entry points and lifecycle guides still claim
  Engine or concrete execution is absent.
- Preserve public convolution success as an exit condition. Backend execution is real, but public
  ownership planning still sees unresolved convolution layouts until Compiler 0006B6 is
  implemented, so the checkpoint cannot yet run.

## Known limitations

- Public composition is CPU-only and accepts one non-empty maximal CPU partition. Zero-node and
  mixed/multi-owner preparation remain unsupported.
- Ordinary compilation uses fixed settings; Config aggregate facades remain planned.
- Model autotuning is bounded to one CPU-local handoff and is not generic graph/plan tuning.
- Public Engine convolution execution is blocked by the recorded layout reason and remains a
  required checkpoint input, not an accepted limitation.
- No persistence, NN execution facade, training session, optimizer step, or second backend is
  delivered.

## Validation evidence

- Independent review context: `01a0b33f-71f0-7710-882f-a703a379b35c`.
- The planning-only audit read the required architecture, planning, completed Engine work,
  directly relevant Compiler/Prepare/Runtime/CPU/tuning contracts, documentation rules/profiles,
  current source/tests, and affected documentation.
- Source inspection established that Model Conv2d/Conv3d outputs use unresolved layouts, Conv1d
  visibly composes through Conv2d, CPU convolution admission requires resolved output layout, and
  `contiguous()` resolves only its own result.
- No Java test or Javadoc command was run while creating this planning specification.
- `/tmp/validate_synaptik_markdown.py` passed for this task, the Engine master plan, and the
  roadmap, checking local link targets and anchors, fences, LF line endings, terminal newlines,
  and trailing whitespace.
- Independent review found that the draft's `Ready` status silently removed the historical
  positive public Conv1d/Conv2d/Conv3d exit condition. Source inspection confirms the exact blocker
  occurs because Compiler supplies unresolved final convolution result layouts to Planning's
  capability query and CPU requires resolved injective layouts.
- Exact-scope validation found only the three expected planning paths changed, with an empty
  staging area. Task status, the master-plan row, and the roadmap project-area row now report
  0008 `Blocked` on the Compiler-owned prerequisite and positive public Engine evidence.
- A later owning-frontier audit selected Ready Compiler 0006B6 with a dedicated final graph
  closure and exact public numerical fixtures. That planning selection does not satisfy this
  checkpoint's blocker; 0006B6 must first become Complete.
- Heading/fence inspection passed, and `git diff --check` produced no diagnostic.
- Implementation evidence remains empty until the separate checkpoint context runs.

## Implementation notes

Empty until the checkpoint is executed.

## Completion summary

Use this template after execution:

```text
- Completed changes: <checkpoint evidence and documentation reconciliation>
- Files changed or created: <exact authorized paths>
- Tests and validation: <commands, counts, outcomes, and report mapping>
- Documentation-agent review: <clean context identity and result>
- Documentation impact: <updated current-status and workflow pages>
- Javadoc review: <generation result and no-change conclusion>
- Glossary impact: <reasoned no-change conclusion>
- Convolution readiness: <positive public Conv1d/Conv2d/Conv3d execution evidence>
- Unresolved issues: <None or exact issue>
- Follow-up required: <None for this checkpoint or exact owner-specific follow-up>

Status: Complete
```
