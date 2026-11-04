# Task 0008: [Engine](../../../../glossary.md#term-engine) [Lifecycle](../../../../glossary.md#lifecycle) Capability [Checkpoint](../../../../glossary.md#term-checkpoint)

## Status

Complete

## Goal

Close the current [Engine](../../../../glossary.md#term-engine) foundation with one capability-[checkpoint](../../../../glossary.md#term-checkpoint) [validation](../../../../glossary.md#term-validation) and documentation
reconciliation pass. Prove the implemented standard and advanced [CPU](../../../../glossary.md#term-cpu)-only [lifecycle](../../../../glossary.md#lifecycle) from public
composition through [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), preparation, typed input [binding](../../../../glossary.md#term-binding), execution, [publication](../../../../glossary.md#term-publication), [host](../../../../glossary.md#term-host)
[materialization](../../../../glossary.md#materialization), one-shot forward and [scalar](../../../../glossary.md#term-scalar)-[objective](../../../../glossary.md#term-objective) backward convenience, optional local-
workload tuning or safe fallback, cleanup, failure handling, and concurrent-[run](../../../../glossary.md#term-run) isolation.

This task consolidates evidence for behavior delivered by Engine tasks 0001 through 0007. It does
not add another Engine behavior, [API](../../../../glossary.md#term-application-programming-interface-api), test seam, [backend route](../../../../glossary.md#backend-route), or [persistence](../../../../glossary.md#term-persistence) facility.

The required [NCW](../../../../glossary.md#term-ncw) Conv1d, [NCHW](../../../../glossary.md#term-nchw) [Conv2d](../../../../glossary.md#convolution--conv2d), and [NCDHW](../../../../glossary.md#conv3d--ncdhw) [Conv3d](../../../../glossary.md#conv3d--ncdhw) public Engine fixtures are now valid through
Complete [Compiler 0006B6](../../compiler/tasks/0006b6-final-convolution-logical-layout-closure.md).
[Model](../../../../glossary.md#term-model) construction and ordinary [inference](../../../../glossary.md#term-inference) still preserve unresolved [convolution](../../../../glossary.md#term-channels-first-convolution-layers) [layouts](../../../../glossary.md#layout), while
[Compiler](../../../../glossary.md#term-compiler) closes only eligible fully static final [descriptors](../../../../glossary.md#term-descriptor) before [Planning](../../../../glossary.md#term-planning) capability queries;
the CPU provider remains strict. The completed prerequisite supplies the exact positive public
Engine fixtures, so this checkpoint is now actionable. It must consume that evidence without
reimplementing the mechanism, weakening CPU [admission](../../../../glossary.md#term-admission), or substituting [backend](../../../../glossary.md#backend)-internal tests.

## Motivation and [mental model](../../../../glossary.md#term-mental-model)

Tasks 0001 through 0007 built the [lifecycle](../../../../glossary.md#lifecycle) incrementally. Their focused tests are necessary
evidence, but the project also needs one named [checkpoint](../../../../glossary.md#term-checkpoint) that verifies the assembled repository
and removes stale documentation that still describes [Engine](../../../../glossary.md#term-engine), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), and [CPU](../../../../glossary.md#term-cpu) execution
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

## [Scope](../../../../glossary.md#term-scope)

- [Run](../../../../glossary.md#term-run) one fresh repository-wide Java test [checkpoint](../../../../glossary.md#term-checkpoint) and record exact outcomes.
- Confirm that the existing test inventory collectively covers:
  - `Engine.standard()` [CPU](../../../../glossary.md#term-cpu)-only construction and [lifetime](../../../../glossary.md#term-lifetime);
  - `AdvancedEngine.takeOwnership(...)`, representation-level input borrowing, reusable prepared
    execution, and concurrent runs;<span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
  - ordinary [owner](../../../../glossary.md#term-owner)-bound [compile](../../../../glossary.md#compile) and [prepare](../../../../glossary.md#prepare) handles;
  - final [Compiler](../../../../glossary.md#term-compiler)-ordered [`TensorId`](../../../../glossary.md#tensorid) input [metadata](../../../../glossary.md#term-metadata) and arbitrary-order typed input supply;
  - caller-owned host-storage snapshots, Runtime result leases, publication occurrence order,
    aliases, and detached canonical host materialization;<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [Runtime glossary](../../../../glossary.md#term-runtime); [leases glossary](../../../../glossary.md#term-lease); [publication occurrence glossary](../../../../glossary.md#publication-occurrence); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [detached glossary](../../../../glossary.md#detached-model-results); [canonical glossary](../../../../glossary.md#term-canonical-form); [materialization glossary](../../../../glossary.md#materialization))</span>
  - the four automatic-input `compute(...)` conveniences and their aggregate byte bound;
  - [scalar](../../../../glossary.md#term-scalar)-[objective](../../../../glossary.md#term-objective) `backward(...)`, explicit target order, the absent positive-one seed, source-
    only constant realization, detached objective/gradients, and aggregate byte bound;<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [seed glossary](../../../../glossary.md#term-seed); [constant glossary](../../../../glossary.md#term-constant); [gradients glossary](../../../../glossary.md#term-gradient))</span>
  - optional CPU local-workload tuning, cache-first behavior, fresh selected preparation, required
    failure, allowed safe-heuristic fallback, immutable translated evidence, and deterministic
    no-handoff fallback integration;<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [heuristic glossary](../../../../glossary.md#term-heuristic); [immutable glossary](../../../../glossary.md#term-immutable); [handoff glossary](../../../../glossary.md#term-handoff))</span>
  - cleanup ordering, primary/suppressed failure identity, close races, repeated/concurrent close,
    and isolated mutable <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [mutable glossary](../../../../glossary.md#term-mutable))</span>[`RunState`](../../../../glossary.md#run-state--runstate) per active run; and
  - dependency direction and the absence of Runtime backend discovery, Engine dependencies from
    inward modules, or tuning work in the Runtime hot path.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [backend glossary](../../../../glossary.md#backend); [modules glossary](../../../../glossary.md#term-module); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- Inspect final source and tests to ensure those claims still match implementation; do not
  duplicate stable assertions merely to create a checkpoint-named test.
- Confirm existing public Engine integration evidence positively executes representative NCW
  Conv1d, NCHW Conv2d, and NCDHW Conv3d forward requests through ordinary typed input binding,
  CPU preparation and Runtime publication. Evidence owned only by CPU-internal tests is
  insufficient.<span data-glossary-navigation="true"> ([NCW glossary](../../../../glossary.md#term-ncw); [NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication))</span>
- Reconcile directly affected documentation so it presents the current CPU-only Engine lifecycle
  as runnable and distinguishes ordinary from advanced composition, reusable from one-shot work,
  result leases from detached values, bounded local tuning from generic tuning, and implemented
  CPU behavior from planned multi-backend composition.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [backend composition glossary](../../../../glossary.md#backend-composition))</span>
- Preserve the completed convolution boundary accurately: Model construction remains unresolved,
  Compiler owns bounded final static descriptor closure, and the public Engine fixtures now
  provide the required positive evidence.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Model glossary](../../../../glossary.md#term-model); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Keep this task, the [Engine](../../../../glossary.md#term-engine) master plan, and repository roadmap synchronized through completion.

## Out of [scope](../../../../glossary.md#term-scope)

- Any production Java, test Java, test resource, Gradle, dependency, architecture-contract, ADR,
  generated-code, or performance-evidence change.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- New Engine methods, request/result types, exception families, caching, prepared persistence,
  asynchronous execution, cancellation, streaming, typed-array results, or caller destinations.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [persistence glossary](../../../../glossary.md#term-persistence); [asynchronous glossary](../../../../glossary.md#term-asynchronous); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- Layout inference/planning for convolution, relaxed CPU capability admission, implicit output
  materialization, a compiler rewrite, or a new Prepare/Runtime layout contract.<span data-glossary-navigation="true"> ([Layout glossary](../../../../glossary.md#layout); [inference glossary](../../../../glossary.md#term-inference); [planning glossary](../../../../glossary.md#term-planning); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [CPU glossary](../../../../glossary.md#term-cpu); [admission glossary](../../../../glossary.md#term-admission); [materialization glossary](../../../../glossary.md#materialization); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Implementation of the blocking Compiler-owned descriptor prerequisite. Compiler 0006B6 owns
  its selected final-rewrite mechanism, exact task scope, focused tests, and public fixtures.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- A public Engine convolution test that passes only by bypassing compilation, forging descriptors,
  naming CPU internals, or manually constructing inward artifacts.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Persistence adapters, checkpoint formats, NN integration, training sessions, optimizer work,
  generic tuning, multi-profile/occurrence tuning, graph/plan tuning, or model-plan persistence.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [optimizer glossary](../../../../glossary.md#term-optimizer); [occurrence glossary](../../../../glossary.md#term-occurrence); [graph glossary](../../../../glossary.md#graph); [model glossary](../../../../glossary.md#term-model))</span>
- A second lifecycle adapter, Metal/CUDA execution, mixed-backend composition, generic backend
  registration, discovery, <span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [backend composition glossary](../../../../glossary.md#backend-composition); [backend glossary](../../../../glossary.md#backend))</span>`ServiceLoader`, or a [Runtime](../../../../glossary.md#term-runtime) [service locator](../../../../glossary.md#term-service-locator).
- New backend conformance or performance proof. Existing CPU evidence remains owned by its
  completed CPU tasks and is not generalized here.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- [Conv3d](../../../../glossary.md#conv3d--ncdhw) [gradients](../../../../glossary.md#term-gradient). [Compiler](../../../../glossary.md#term-compiler) 0006C remains a separate Draft boundary.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially the lifecycle, invariants,
  module responsibilities, dependency rules, and compile/prepare/run sections<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend))</span>
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Repository roadmap](../../../roadmap.md)
- [Engine master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>

## Architecture constraints

- Engine remains the explicit outer composition root. Compiler owns compilation, Prepare owns
  staged preparation and validation, Runtime owns immutable recipes plus isolated per-run state,
  and CPU owns concrete capability, lowering, storage, execution, and host copying.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [composition root glossary](../../../../glossary.md#term-composition-root); [Compiler glossary](../../../../glossary.md#term-compiler); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [Prepare glossary](../../../../glossary.md#prepare); [validation glossary](../../../../glossary.md#term-validation); [Runtime glossary](../../../../glossary.md#term-runtime); [run state glossary](../../../../glossary.md#run-state--runstate); [lowering glossary](../../../../glossary.md#lowering); [host glossary](../../../../glossary.md#term-host))</span>
- Standard composition remains one fresh Engine-owned CPU integration. Advanced composition
  remains explicit ownership of one supported CPU integration. Neither implies mixed-backend
  schedule assembly.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend))</span>
- Runtime executes prepared work only and performs no graph inspection, backend discovery,
  lowering, kernel selection, tuning search, or cache mutation.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [kernel glossary](../../../../glossary.md#kernel); [cache glossary](../../../../glossary.md#term-cache))</span>
- Prepared recipes remain [immutable](../../../../glossary.md#term-immutable) and reusable. Each invocation has one isolated [mutable](../../../../glossary.md#term-mutable)
  [`RunState`](../../../../glossary.md#run-state--runstate); borrowed caller storage remains caller-owned, while result leases and detached host
  values retain their completed ownership rules.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [leases glossary](../../../../glossary.md#term-lease); [detached glossary](../../../../glossary.md#detached-model-results))</span>
- Model Tensors retain no execution method, gradient field, backward lifecycle, Runtime state, or
  Engine dependency.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensors glossary](../../../../glossary.md#tensor); [gradient glossary](../../../../glossary.md#term-gradient); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Optional model autotuning completes before production Runtime execution. Engine composes public
  lifecycle, tools/tuning owns measurement/cache/ranking/evidence, and CPU owns typed candidates
  and selected preparation.<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Planning](../../../../glossary.md#term-planning) chooses [backend ownership](../../../../glossary.md#backend-ownership) and never interprets [CPU](../../../../glossary.md#term-cpu) routes or tuning [parameters](../../../../glossary.md#term-parameter).
- Documentation may correct implementation-status claims but must not change architecture rules.
  If validation reveals a contradiction or required inward contract change, stop rather than
  editing `ARCHITECTURE.md` or inventing a decision.

## [Dependency](../../../../glossary.md#term-dependency) and readiness audit

### Complete dependencies

- [Engine](../../../../glossary.md#term-engine) 0001–0007 are Complete, including 0005A and 0006A.
- Compiler 0006B3–0006B5 are Complete and supply the complete compile entry, stable caller-input
  identity bindings, and source-only published-constant descriptor closure.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [compile glossary](../../../../glossary.md#compile); [identity glossary](../../../../glossary.md#term-identity-meanings); [bindings glossary](../../../../glossary.md#term-binding); [constant glossary](../../../../glossary.md#term-constant); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Prepare 0003, 0003A, 0004, and 0005 are Complete and supply staged preparation, the partition-
  local DAG, opaque tuning handoff, and producerless published-constant assignment.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [opaque glossary](../../../../glossary.md#term-opaque); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Runtime 0010, 0012, 0014, and 0015 are Complete and supply prepared execution, shared-throwable
  cleanup, architecture enforcement, and leased publication-representation access.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [publication glossary](../../../../glossary.md#term-publication))</span>
- CPU 0008 and 0008A are Complete for internal Conv2d and dimensional-convolution execution; CPU
  0010F–0010I are Complete for lifecycle composition, host snapshots, source-only constants, and
  the local-workload tuning adapter.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [lifecycle glossary](../../../../glossary.md#lifecycle); [host glossary](../../../../glossary.md#term-host))</span>
- Config 0006A and tools/tuning 0001 are Complete for the bounded optional tuning flow.

### Satisfied [convolution](../../../../glossary.md#term-channels-first-convolution-layers) prerequisite

Complete [Compiler](../../../../glossary.md#term-compiler) 0006B6 now supplies [Planning](../../../../glossary.md#term-planning) convolution occurrences with final resolved
[layouts](../../../../glossary.md#layout) exactly when the [Conv2d](../../../../glossary.md#convolution--conv2d)/[Conv3d](../../../../glossary.md#conv3d--ncdhw) result is fully static and was unresolved. Planning passes
those [descriptors](../../../../glossary.md#term-descriptor) unchanged in [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) and owns only capability [evaluation](../../../../glossary.md#term-evaluation) and
[backend](../../../../glossary.md#backend) selection; [CPU](../../../../glossary.md#term-cpu) still admits only resolved, [injective](../../../../glossary.md#term-injective) convolution outputs. The Compiler
rewrite runs after final optimization, [validation](../../../../glossary.md#term-validation), and published-[constant](../../../../glossary.md#term-constant) closure. Only a direct
[axis](../../../../glossary.md#term-axis)-two squeeze of a Conv2d result newly closed by that invocation receives the exact [view](../../../../glossary.md#term-view-meanings) layout
needed by visible [Conv1d composition](../../../../glossary.md#conv1d-composition). [Model](../../../../glossary.md#term-model) construction and dynamic [Shapes](../../../../glossary.md#shape) remain unresolved,
Planning remains backend-neutral, and the exact public [Engine](../../../../glossary.md#term-engine) numerical fixtures pass.

Task 0008 is therefore actionable as a validation-only task. It does not choose or implement
another closure rule, mutate Model semantics, relax CPU capability, or add a Planning layout
policy.

### [Convolution](../../../../glossary.md#term-channels-first-convolution-layers) [fixture](../../../../glossary.md#term-fixture) audit

| Family | Current [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) form | Current [CPU](../../../../glossary.md#term-cpu) evidence | [Checkpoint](../../../../glossary.md#term-checkpoint) requirement |
|---|---|---|---|
| [NCW](../../../../glossary.md#term-ncw) Conv1d | visible `EXPAND_DIMS -> CONV2D -> SQUEEZE` composition | exact composition executed in CPU tests | Positive public [Engine](../../../../glossary.md#term-engine) forward execution after the [Conv2d](../../../../glossary.md#convolution--conv2d) boundary is admissible. |
| [NCHW](../../../../glossary.md#term-nchw) Conv2d | one first-class flat [operation](../../../../glossary.md#operation) | direct grouped CPU execution exists | Positive public Engine grouped forward execution with typed [binding](../../../../glossary.md#term-binding) and [publication](../../../../glossary.md#term-publication). |
| [NCDHW](../../../../glossary.md#conv3d--ncdhw) [Conv3d](../../../../glossary.md#conv3d--ncdhw) | one first-class flat forward operation | direct grouped CPU execution exists | Positive public Engine grouped forward execution; backward remains deferred. |

These are checkpoint requirements, not a claim that every Model expression with a [backend](../../../../glossary.md#backend)-
internal [kernel](../../../../glossary.md#kernel) must [compile](../../../../glossary.md#compile) through standard Engine. Compiler 0006B6 satisfied the former unmet
exit condition by closing the exact final-[descriptor](../../../../glossary.md#term-descriptor) boundary and passing the public fixtures.
This checkpoint consumes that evidence without broadening it; [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) convolution integration remains
ordered afterward.

## Package impact

No Java package changes. Documentation describes existing public [Engine](../../../../glossary.md#term-engine) and inward-[owner](../../../../glossary.md#term-owner) types
without adding, moving, or changing any type.

## Affected files

The [checkpoint](../../../../glossary.md#term-checkpoint) may modify only these fourteen documentation/[planning](../../../../glossary.md#term-planning) paths:

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

Review without modification: all Java/[Javadoc](../../../../glossary.md#term-javadoc), tests, resources, build files,
`ARCHITECTURE.md`, ADRs, glossary, other [API](../../../../glossary.md#term-application-programming-interface-api)/[backend](../../../../glossary.md#backend) pages, other plans, and other task specs.
Record reasoned no-change conclusions for each plausibly affected category.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact fourteen paths above. No Java, test, resource, build, architecture-contract,
[ADR](../../../../glossary.md#term-architecture-decision-record-adr), or glossary path is authorized. If accurate documentation needs another path, amend this
specification and its ceiling before editing. If [validation](../../../../glossary.md#term-validation) needs executable or architectural
change, stop and report the owning gap.

## Acceptance criteria

- One fresh repository-wide Java test [run](../../../../glossary.md#term-run) passes with no Java change before or after it.
- Existing Engine, integration, and architecture reports map to every lifecycle claim in Scope;
  missing evidence fails the checkpoint rather than becoming a prose assertion.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [Scope glossary](../../../../glossary.md#term-scope); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Standard and advanced composition remain CPU-only, explicitly owned, closeable, and free of
  discovery or process-global service location.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Reusable compile/prepare/run preserves owner identity, ordered typed input metadata, arbitrary
  input order, caller ownership, reusable prepared state, isolated runs, ordered publication
  occurrences, aliases, and result leases.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [owner glossary](../../../../glossary.md#term-owner); [identity glossary](../../../../glossary.md#term-identity-meanings); [metadata glossary](../../../../glossary.md#term-metadata); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [publication glossary](../../../../glossary.md#term-publication); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [leases glossary](../../../../glossary.md#term-lease))</span>
- Host materialization remains explicit, bounded, canonical, detached, occurrence-authenticated,
  and valid after result/Engine closure.<span data-glossary-navigation="true"> ([Host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization); [canonical glossary](../../../../glossary.md#term-canonical-form); [detached glossary](../../../../glossary.md#detached-model-results); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `compute(...)` and `backward(...)` retain automatic leaf discovery, Compiler-authoritative
  binding order, fresh lifecycle work, aggregate preflight, detached results, and exact cleanup
  without caching or Tensor-owned execution state.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [Compiler glossary](../../../../glossary.md#term-compiler); [binding glossary](../../../../glossary.md#term-binding); [preflight glossary](../../../../glossary.md#term-preflight); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Optional tuning retains its bounded CPU-only request/outcome, caller model identity, occurrence-
  0/partition-0/weight-1 mapping, cache-first trials, fresh production preparation, required
  failure, and explicit safe fallback outside Runtime.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [partition glossary](../../../../glossary.md#partition); [weight glossary](../../../../glossary.md#term-weight); [cache glossary](../../../../glossary.md#term-cache); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Public Engine integration tests positively execute representative NCW Conv1d, NCHW Conv2d, and
  NCDHW Conv3d forward requests through compilation, CPU ownership, preparation, Runtime
  execution, ordered publication, and detached materialization. CPU-internal execution evidence
  alone does not satisfy this criterion.<span data-glossary-navigation="true"> ([NCW glossary](../../../../glossary.md#term-ncw); [NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Architecture tests enforce applicable direction, including `engine -> tools/tuning`, no reverse
  edge, backend independence from Engine, and Runtime independence from backends and Engine.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Documentation entry points, status pages, public API status, and four lifecycle guides describe
  current behavior rather than obsolete model-only or planned-Engine state.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation promises no Config aggregates, generic backend registration, mixed-backend
  execution, persistence, training orchestration, or generic tuning.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
- The convolution evidence is positive and public without bypassing compilation, forging
  descriptors, importing CPU internals, or manually constructing inward artifacts.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [descriptors glossary](../../../../glossary.md#term-descriptor); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Javadocs are reviewed and generated without source edits; the glossary has a reasoned no-change
  conclusion because no reusable term or definition changes.
- Exactly the fourteen authorized paths or a documented subset changes. Task/master/roadmap
  statuses agree, Markdown validation passes, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`git diff --check` passes.
- A separate clean documentation-focused context finalizes the prose. Task 0008 becomes Complete
  only after all evidence and its completion summary are recorded.

## Tests / [validation](../../../../glossary.md#term-validation)

After the blocking prerequisite and positive public integration fixtures are Complete, this
capability [checkpoint](../../../../glossary.md#term-checkpoint) owns one fresh repository-wide [run](../../../../glossary.md#term-run) rather than repeating completed focused
suites separately:

```bash
./gradlew test --rerun-tasks
```

Record the outcome and exact aggregate counts from XML reports. Confirm execution of existing
[Engine](../../../../glossary.md#term-engine) unit suites, `EngineAdvancedLifecycleIntegrationTest`,
`EngineStandardCompositionIntegrationTest`, `EngineTypedLifecycleIntegrationTest`,
  `EngineModelAutotuningIntegrationTest`, the prerequisite-owned public dimensional-convolution
  integration coverage, and <span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>`EngineCompositionContractTest`. Do not rerun the same successful
  suites in the documentation context unless executable Java changes or the repository run omitted
  them.

After the documentation-focused pass:

```bash
./gradlew :modules:engine:javadoc
git diff --check
git status --short -uall
```

Also validate local links/anchors, unique headings, balanced fences, LF/final newlines, trailing
whitespace, current/planned wording, exact changed-path [scope](../../../../glossary.md#term-scope), empty staging, and status agreement.
Do not run Java tests while creating this [planning](../../../../glossary.md#term-planning) specification.

## Documentation and [Javadoc](../../../../glossary.md#term-javadoc) impact

Apply General style to the index/getting-started pages, Architecture style to the four architecture
explanations, [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc style to public API status and Javadoc review, User-guide plus Example
profiles to the workflows, and [Planning](../../../../glossary.md#term-planning) style to the three planning records.

Replace obsolete [model](../../../../glossary.md#term-model)-only and planned-[lifecycle](../../../../glossary.md#lifecycle) statements with a current [mental model](../../../../glossary.md#term-mental-model) and
runnable [CPU](../../../../glossary.md#term-cpu)-only examples. Preserve ordinary/advanced, reusable/one-shot, [lease](../../../../glossary.md#term-lease)/[detached](../../../../glossary.md#detached-model-results)-value,
local/generic-tuning, and current/planned distinctions. No Javadoc edit is expected because no API
or behavior changes. Generate and inspect [Engine](../../../../glossary.md#term-engine) Javadoc and record why it remains accurate. The
glossary should remain unchanged because all relevant terms already have established meanings.

## Risks

- Treating a passed repository suite as proof of untested behavior.
- Expanding stale-status cleanup into unrelated documentation rewrites.
- Presenting CPU-internal convolution evidence as public Engine execution or deleting the positive
  convolution exit condition.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Converting a checkpoint into layout-resolution, multi-backend, persistence, training, generic-
  tuning, or performance work.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [layout glossary](../../../../glossary.md#layout); [backend glossary](../../../../glossary.md#backend); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Repeating successful Java suites in the documentation context.

## Dependencies

- [Engine](../../../../glossary.md#term-engine) 0001–0007 — Complete.
- [Compiler](../../../../glossary.md#term-compiler) 0006B3–0006B5 — Complete.
- [Compiler 0006B6](../../compiler/tasks/0006b6-final-convolution-logical-layout-closure.md) —
  Complete, including its exact positive public Engine fixtures.
- [Prepare](../../../../glossary.md#prepare) 0003, 0003A, 0004, and 0005 — Complete.
- [Runtime](../../../../glossary.md#term-runtime) 0010, 0012, 0014, and 0015 — Complete.
- [CPU](../../../../glossary.md#term-cpu) 0008, 0008A, and 0010F–0010I — Complete.
- Config 0006A and tools/tuning 0001 — Complete.
- Compiler-owned convolution logical-layout closure before Planning capability admission —
  complete in Compiler 0006B6.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [layout glossary](../../../../glossary.md#layout); [Planning glossary](../../../../glossary.md#term-planning); [admission glossary](../../../../glossary.md#term-admission))</span>
- Positive public Engine NCW Conv1d, NCHW Conv2d, and NCDHW Conv3d forward integration evidence —
  supplied by Compiler 0006B6's three passing integration tests.<span data-glossary-navigation="true"> ([NCW glossary](../../../../glossary.md#term-ncw); [NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>

## Follow-up tasks

- This checkpoint is complete. Planning remains the consumer of final descriptors, CPU remains
  strict, and task 0008 introduced no competing closure rule or Java fixture.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [Planning glossary](../../../../glossary.md#term-planning); [descriptors glossary](../../../../glossary.md#term-descriptor); [CPU glossary](../../../../glossary.md#term-cpu); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- Existing Draft NN 0025 owns dimensional-convolution layer integration, followed by NN 0025A's
  end-to-end user-capability checkpoint; neither may substitute CPU-internal evidence for public
  Engine execution.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Tools/tuning 0002 remains later [graph](../../../../glossary.md#graph)/plan tuning.
- [Persistence](../../../../glossary.md#term-persistence), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), Training, [Metal](../../../../glossary.md#term-metal)/[CUDA](../../../../glossary.md#term-cuda), and mixed-[backend composition](../../../../glossary.md#backend-composition) remain with their [owners](../../../../glossary.md#term-owner).
- [Compiler](../../../../glossary.md#term-compiler) 0006C remains the separate Draft [Conv3d](../../../../glossary.md#conv3d--ncdhw) [gradient](../../../../glossary.md#term-gradient) boundary.

## Architecture impact

Expected impact: None. This [checkpoint](../../../../glossary.md#term-checkpoint) verifies and documents current contracts. It changes no
[owner](../../../../glossary.md#term-owner), [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module) boundary, [API](../../../../glossary.md#term-application-programming-interface-api), [Runtime](../../../../glossary.md#term-runtime) behavior, or [backend](../../../../glossary.md#backend) behavior. If [validation](../../../../glossary.md#term-validation)
finds a conflict, stop and report it rather than editing the [architecture contract](../../../../glossary.md#architecture-contract).

## Implementation prompt

Use this prompt in a separate clean [checkpoint](../../../../glossary.md#term-checkpoint) context:

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

- Return 0008 to [`Ready`](../../../../glossary.md#term-ready) only after Compiler 0006B6 and its positive public dimensional-
  convolution fixtures complete; that prerequisite is now satisfied.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Use existing tests plus one fresh repository run. A new checkpoint Java test would duplicate
  stable lifecycle assertions; Compiler 0006B6 owns the missing public convolution fixtures before
  this documentation-only checkpoint runs.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Include documentation reconciliation because entry points and lifecycle guides still claim
  Engine or concrete execution is absent.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- Preserve public convolution success as an exit condition. Complete Compiler 0006B6 now closes
  eligible final layouts before ownership planning and supplies the required public fixtures.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [planning glossary](../../../../glossary.md#term-planning))</span>

## Known limitations

- Public composition is CPU-only and accepts one non-empty maximal CPU partition. Zero-node and
  mixed/multi-owner preparation remain unsupported.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [node glossary](../../../../glossary.md#node); [owner glossary](../../../../glossary.md#term-owner))</span>
- Ordinary [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) uses fixed settings; Config aggregate facades remain planned.
- [Model autotuning](../../../../glossary.md#autotuning--model-autotuning) is bounded to one [CPU](../../../../glossary.md#term-cpu)-local [handoff](../../../../glossary.md#term-handoff) and is not generic [graph](../../../../glossary.md#graph)/plan tuning.
- Dynamic or partially dynamic convolution results remain unresolved; the required fully static
  public Engine convolution executions are now available as checkpoint input.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Engine glossary](../../../../glossary.md#term-engine); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- No persistence, NN execution facade, training session, optimizer step, or second backend is
  delivered.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [facade glossary](../../../../glossary.md#term-facade); [training session glossary](../../../../glossary.md#training-session--trainingsession); [optimizer step glossary](../../../../glossary.md#optimizer-and-stochastic-gradient-descent); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- [Checkpoint](../../../../glossary.md#term-checkpoint) evidence context: `01a0b444-f257-7651-b90e-9a89a5c78580`. Its one actual repository
  checkpoint, `./gradlew test --rerun-tasks`, completed successfully in 2m 52s with all 70
  actionable tasks executed. The 492 XML reports record 3,188 tests, zero failures, zero errors,
  and 28 skips. Every skip belongs to optional CPU performance/evidence/persistence coverage;
  every Engine, integration, architecture, Compiler convolution, Prepare, Runtime, and tuning
  report consumed here has zero skips. No Java test was rerun in this documentation context.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [persistence glossary](../../../../glossary.md#term-persistence); [Compiler glossary](../../../../glossary.md#term-compiler); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- The seven [Engine](../../../../glossary.md#term-engine) unit suites report 67 tests. Named cross-[module](../../../../glossary.md#term-module) reports contain
  `EngineAdvancedLifecycleIntegrationTest` 2, `EngineStandardCompositionIntegrationTest` 1,
  `EngineTypedLifecycleIntegrationTest` 6, `EngineModelAutotuningIntegrationTest` 1,
  `EngineConvolutionIntegrationTest` 3, and `EngineCompositionContractTest` 1. All report zero
  failures, errors, and skips.
- Supporting green report sets contain all nine architecture tests, 42 Compiler convolution
  closure/GraphCompiler/Conv3d tests, 24 tools/tuning tests, 52 Prepare tests, and 148 Runtime
  tests, again with zero failures, errors, or skips. These reports map the task's ownership,
  reusable lifecycle, isolated-run, cleanup, tuning, and dependency claims to executed evidence.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [run glossary](../../../../glossary.md#term-run); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- The three public convolution fixtures use only Model and Engine APIs. They execute NCW Conv1d,
  grouped NCHW Conv2d, and grouped NCDHW Conv3d forward requests through reusable
  compile/prepare/run plus one-shot compute, reversed typed input order, publication, and detached
  materialization. They import no CPU internals and construct no inward artifacts. Conv3d
  gradients remain outside the claim.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [NCW glossary](../../../../glossary.md#term-ncw); [NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [compile glossary](../../../../glossary.md#compile); [publication glossary](../../../../glossary.md#term-publication); [detached glossary](../../../../glossary.md#detached-model-results); [materialization glossary](../../../../glossary.md#materialization); [artifacts glossary](../../../../glossary.md#term-artifact); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Documentation-focused context: `01a0b44b-f58b-7d42-b4cb-f33bd50a2e13`. It read the required
  architecture and planning contracts, General/Architecture/API/Javadoc/User-guide/Example/
  Planning profiles, all eleven affected explanatory pages, relevant glossary entries, public
  Engine source/Javadocs, and the saved unit/integration/architecture reports. It changed no
  executable or authoritative architecture artifact.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `./gradlew :modules:engine:javadoc` ran exactly once after explanatory prose was final and
  passed: `BUILD SUCCESSFUL in 1s`, 15 actionable tasks, one executed and 14 up-to-date. Inspection
  of the generated Engine, RunResult, HostTensorValue, ModelAutotuning request/preparation, and
  package pages confirmed current ownership, CPU-only, lease, detached-value, one-shot, and
  tuning boundaries. Existing source Javadocs remain accurate; no source edit was needed.<span data-glossary-navigation="true"> ([RunResult glossary](../../../../glossary.md#run-result--runresult); [HostTensorValue glossary](../../../../glossary.md#host-tensor-value--hosttensorvalue); [lease glossary](../../../../glossary.md#term-lease))</span>
- `/tmp/validate_synaptik_markdown.py` passed all fourteen authorized paths, checking local links
  and anchors, unique headings, balanced fences, LF line endings, terminal newlines, and trailing
  whitespace. Manual review checked current/planned language and examples against public source
  signatures. Final scope/status/staging and <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` checks also passed.

## Implementation notes

- Reconciled the documentation index, contributor start, architecture status/lifecycle/boundaries,
  public API status, and four lifecycle user guides with the current runnable CPU-only Engine.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Preserved ordinary `Engine.standard()` versus advanced ownership, reusable prepared execution
  versus fresh one-shot calls, leased <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>[`RunResult`](../../../../glossary.md#run-result--runresult) versus [detached](../../../../glossary.md#detached-model-results) [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue), CPU-local
  workload tuning versus future graph/plan tuning, current CPU execution versus future other or
  mixed owners, and Model's unresolved convolution construction versus Compiler's bounded final
  static closure and strict CPU admission.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [Model glossary](../../../../glossary.md#term-model); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Compiler glossary](../../../../glossary.md#term-compiler); [admission glossary](../../../../glossary.md#term-admission))</span>
- Changed exactly the fourteen allowlisted documentation paths. Java, tests, resources, Gradle,
  dependencies, <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`ARCHITECTURE.md`, ADRs, and every other document remain unchanged.
- Javadoc source required no change because this checkpoint changes no API or behavior and the
  existing contracts already describe parameters, results, failures, ownership, closure, and
  current limitations consistently with source and tests.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- The glossary required no change because all reusable terms and distinctions used here already
  have current entries, including model autotuning, backend ownership/routes, compiled/prepared
  state, publication occurrences, RunResult, HostTensorValue, and RunState.<span data-glossary-navigation="true"> ([model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [backend ownership glossary](../../../../glossary.md#backend-ownership); [publication glossary](../../../../glossary.md#term-publication); [RunState glossary](../../../../glossary.md#run-state--runstate))</span>
- Architecture/ADRs and architecture tests required no change because no owner, dependency,
  lifecycle invariant, or module boundary changed. Java/tests/build/dependencies likewise needed
  no change because this is a documentation-only checkpoint over existing behavior.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [dependency glossary](../../../../glossary.md#term-dependency); [invariant glossary](../../../../glossary.md#term-invariant); [module glossary](../../../../glossary.md#term-module))</span>
- Backend conformance and performance evidence required no change or rerun: no backend behavior or
  performance claim changed. Other documentation remains outside the focused stale-status scope.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [scope glossary](../../../../glossary.md#term-scope))</span>
- The Engine foundation is Complete through 0008. The existing Draft NN 0025 convolution-layer
  task, followed by NN 0025A's user-capability checkpoint, is the next dimensional-convolution
  frontier; this checkpoint creates no new task.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

## Completion summary

- Completed changes: closed the lifecycle capability checkpoint and reconciled all eleven affected
  explanatory pages plus task, Engine master plan, and roadmap status.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Files changed or created: exactly the fourteen authorized existing Markdown paths; no file was
  created.
- Tests and validation: reused the clean 3,188-test repository checkpoint with zero failures or
  errors; generated Engine Javadoc exactly once; passed Markdown, scope, staging, status, and
  whitespace validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `01a0b44b-f58b-7d42-b4cb-f33bd50a2e13`, Complete.
- Documentation impact: current CPU-only compile/prepare/run/materialize, one-shot, advanced
  ownership, and bounded autotuning workflows now replace obsolete planned-only prose.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: generated output passed inspection; source Javadocs remain accurate unchanged.
- Glossary impact: unchanged because established entries already cover every term and distinction.
- Convolution readiness: public NCW Conv1d, grouped NCHW Conv2d, and grouped NCDHW Conv3d forward
  fixtures pass through both reusable and one-shot public Engine paths.<span data-glossary-navigation="true"> ([Convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [NCW glossary](../../../../glossary.md#term-ncw); [NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- Unresolved issues: None for Engine 0008; documented future capabilities remain intentionally
  unsupported.
- Follow-up required: None for this checkpoint. Existing NN 0025 and 0025A remain the planned
  downstream frontier.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

Status: Complete
