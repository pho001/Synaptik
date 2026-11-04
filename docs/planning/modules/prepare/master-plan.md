# [Prepare](../../../glossary.md#prepare) Master Plan

> **Editorial historical reading key.**
>
> This master plan is a historical implementation map, not semantic authority or a live launch
> instruction. Its dated frontier narratives and task-table statuses must remain distinct from current
> support references. Engine supplies concrete partition collaborators; shared Prepare projects
> Compiler artifacts, invokes backend analysis, assigns slots, finalizes and validates a complete
> schedule. Backends retain authority over lowering, routes and physical geometry; shared code
> transports their plans opaquely.
>
> The [current preparation
> handoff](../../../reference/runtime/orchestration.md#current-prepare-orchestration-assignment-and-backend-finalization)
> details finalizer ownership before return, Prepare rollback afterward, and successful transfer of
> unique persistent resources to Runtime. Schedule occurrences do not multiply owners. [Current batch
> metadata
> binding](../../../reference/compiler/batch-binding.md#ordinary-named-batch-metadata-binding)
> separates Compiler’s descriptor obligations from numeric resource sizing and backend admission.
> Caller payloads remain borrowed; completed run resources belong to result leases, not the planning
> map.

## Goal and authority

[Prepare](../../../glossary.md#prepare) coordinates the validated transition from [immutable](../../../glossary.md#term-immutable) [compile artifacts](../../../glossary.md#compile-artifacts) to one complete
[Runtime](../../../glossary.md#term-runtime)-[ready](../../../glossary.md#term-ready) recipe. This plan is a non-authoritative implementation map;
[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md), especially `Core lifecycle`,
`modules/prepare`, and `Prepare lifecycle`, is authoritative.

Focused explanations and decisions:

- [Lifecycle](../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>
- [Runtime, Prepare, and Backend Boundary](../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Backend glossary](../../../glossary.md#backend))</span>
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#term-architecture-decision-record-adr))</span>

## [Lifecycle](../../../glossary.md#lifecycle) position

```text
CompileArtifacts
  -> partition-local analysis and exact resource declarations
  -> shared buffer/workspace slot assignment
  -> backend finalization against assigned slots
  -> transactional resource aggregation
  -> complete schedule assembly and validation
  -> one-time ownership transfer to PreparedExecution
```

[Prepare](../../../glossary.md#prepare) stages and validates this [transaction](../../../glossary.md#term-transaction). Concrete [backends](../../../glossary.md#backend) analyze and finalize their own
work; [Runtime](../../../glossary.md#term-runtime) later owns and executes the accepted recipe.

## [Scope](../../../glossary.md#term-scope) and non-goals

[Prepare](../../../glossary.md#prepare) owns [partition](../../../glossary.md#partition) projection, [backend](../../../glossary.md#backend)-neutral analysis/finalization collaborations, exact
resource declarations, shared assignment, producerless published-[constant](../../../glossary.md#term-constant) contribution, [opaque](../../../glossary.md#term-opaque)
candidate transport, complete orchestration, schedule assembly, [validation](../../../glossary.md#term-validation), [rollback](../../../glossary.md#term-rollback), and final
[ownership](../../../glossary.md#term-ownership-meanings) transfer.

Prepare does not execute work, implement concrete [CPU](../../../glossary.md#term-cpu)/[Metal](../../../glossary.md#term-metal)/[CUDA](../../../glossary.md#term-cuda) [lowering](../../../glossary.md#lowering) or storage, select a
backend globally, interpret [backend route](../../../glossary.md#backend-route)/candidate fields, measure or [rank](../../../glossary.md#term-rank) tuning candidates,
mutate caches, or allocate per-[run](../../../glossary.md#term-run) physical representations.

## Stable invariants and [ownership](../../../glossary.md#term-ownership-meanings)

- Backend analysis deterministically selects and retains backend-private lowering and route state
  while declaring every shared buffer/workspace requirement before assignment.<span data-glossary-navigation="true"> ([Backend glossary](../../../glossary.md#backend); [lowering glossary](../../../glossary.md#lowering); [buffer glossary](../../../glossary.md#term-buffer))</span>
- Shared Prepare assigns stable Runtime slots across the complete ordered analyses without
  interpreting concrete storage or route vocabulary.<span data-glossary-navigation="true"> ([Runtime glossary](../../../glossary.md#term-runtime))</span>
- Finalization receives the exact analysis, shared plan, and assignments. It constructs the
  executable only after assignment and may not revise the route or introduce undeclared shared
  requirements.
- A finalizer owns every persistent resource until its complete result returns. Shared Prepare
  then owns each exact identity transactionally and rolls back once in deterministic reverse
  acquisition order after any later failure.<span data-glossary-navigation="true"> ([identity glossary](../../../glossary.md#term-identity-meanings))</span>
- Duplicate resource identity is rejected without double close. Rollback preserves the primary
  preparation failure and suppresses later distinct cleanup failures while avoiding
  self-suppression.<span data-glossary-navigation="true"> ([Rollback glossary](../../../glossary.md#term-rollback))</span>
- Successful `PreparedExecution(memoryPlan, schedule, resources)` construction is the sole
  ownership-transfer point. Prepare derives no ownership from executable or schedule occurrences.
- Candidate batches and decisions cross shared code opaquely. Safe preparation never depends on a
  tuning search, and Runtime receives no cache or candidate state.<span data-glossary-navigation="true"> ([batches glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [cache glossary](../../../glossary.md#term-cache))</span>
- [Prepare](../../../glossary.md#prepare) constructs recipes and validates coverage; it performs no execution.

## Dependencies

Current allowed direct dependencies are `modules/runtime`, `modules/planning`,
`modules/compiler`, `modules/config`, `modules/backend-contract`, and `modules/trace`.
Dependencies on [concrete backend](../../../glossary.md#concrete-backend) implementations are forbidden; [Engine](../../../glossary.md#term-engine) supplies concrete
collaborators from the outer [composition root](../../../glossary.md#term-composition-root).

## Package map

| Package | Public or shared role | Internal boundary |
|---|---|---|
| `prepare.analysis` | [Immutable](../../../glossary.md#term-immutable) [partition](../../../glossary.md#partition)-local [DAG](../../../glossary.md#term-directed-acyclic-graph-dag)/context, [opaque](../../../glossary.md#term-opaque) [backend](../../../glossary.md#backend) input/plan/tuning roles, exact resource declarations, and analyzer collaboration. | No `CompileArtifacts` aggregate, complete [graph](../../../glossary.md#graph), tuning interpretation, slot assignment, executable, or physical resource. |
| [`prepare`](../../../glossary.md#prepare) root | Finalization input/result, assignments, [`PreparedPartition`](../../../glossary.md#prepared-partition--preparedpartition), producerless resources, [`GraphPreparation`](../../../glossary.md#graph-preparation--graphpreparation), and schedule assembly/[validation](../../../glossary.md#term-validation) contracts. | Complete-set [handoff](../../../glossary.md#term-handoff) and transactional tracking stay package-private; concrete backends own [lowering](../../../glossary.md#lowering) and physical mechanics. |

The public Prepare surface is cross-[module](../../../glossary.md#term-module) service-provider interface ([SPI](../../../glossary.md#term-service-provider-interface-spi)), not the ordinary
[Engine](../../../glossary.md#term-engine) user [facade](../../../glossary.md#term-facade).

0010 extends existing `prepare.analysis.PrepareContext` validation and root `GraphPreparation`
projection; [`PartitionDag`](../../../glossary.md#partition-dag-partitiondag) retains exact body-bearing [nodes](../../../glossary.md#node). Complete 0011 reuses these [owners](../../../glossary.md#term-owner) for
length-aware [bindings](../../../glossary.md#term-binding)/demand and existing resource handoff. No new production type/package/[API](../../../glossary.md#term-application-programming-interface-api).

The reviewed [Prepare0012 design delivery](tasks/0012-dynamic-batch-geometry-and-resource-contract.md)
supplies the user-approved cross-[owner](../../../glossary.md#term-owner) dynamic-B package/type/[symbol](../../../glossary.md#term-symbol) map. Authority cutover
[Prepare0012A](tasks/0012a-dynamic-batch-authority-cutover.md) is Complete, authority enacted/MAIN-accepted
2026-10-24. [Prepare0013](tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result) implements the
bounded ordinary [FLOAT32](../../../glossary.md#term-float32) [CPU](../../../glossary.md#term-cpu) path; final independent reviews and native-enabled root validation
passed, and MAIN accepted it on 2026-10-24.

## [Dependency](../../../glossary.md#term-dependency) [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) and [authorized frontier](../../../glossary.md#term-authorized-frontier)

Active frontier: the neutral projection/context-copy [slice](../../../glossary.md#slice) of [CPU0081 Ready](../../backends/cpu/tasks/0081-ordinary-unit-general-scan-composition.md), under accepted coordinated authority/frozen APIs and exact dirty-source context. [Prepare0015 symbolic-length Scan CPU vertical](tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
is Complete, MAIN accepted 2026-10-27 after independent code/docs APPROVE (0 P1/P2), owning [validation](../../../glossary.md#term-validation),
the sole bounded probe and configured root test/[Javadoc](../../../glossary.md#term-javadoc) PASS. Complete Model0044 + Compiler0015 +
Prepare0011/0013/0014 + CPU0042 and retained [lifecycle](../../../glossary.md#lifecycle) predecessors feed the delivered [F32](../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../glossary.md#floating-point-precision-and-rounding)
vertical, including bounded outer/control/ROW(1). Existing [owners](../../../glossary.md#term-owner) and old [identities](../../../glossary.md#term-identity-meanings)/bytes remain;
timing is [nonstationary](../../../glossary.md#term-nonstationary), not parity or a [JIT](../../../glossary.md#term-just-in-time-compilation-jit)-cause [qualification](../../../glossary.md#term-qualification). That completion authorized no successor or [memo](../../../glossary.md#cache-and-memoization).

[Planning](../../../glossary.md#term-planning) coordination only: [Model0043](../model/tasks/0043-symbolic-length-scan-integration-design.md),
Complete for [combined F32/F64 Scan](../../length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration)
DESIGN DELIVERY ONLY; Hume the 4th final Class C APPROVE, 0 P1/P2/no unresolved findings;
MAIN accepted 2026-10-26, not new semantics, authority or implementation.
Prepare0015 now joins typed [run](../../../glossary.md#term-run)-size resources, [producer](../../../glossary.md#term-producer)-ordered [preflight](../../../glossary.md#term-preflight) and actual reverse
through the existing [Prepare](../../../glossary.md#prepare)→[CPU](../../../glossary.md#term-cpu)→[Engine](../../../glossary.md#term-engine) [handoff](../../../glossary.md#term-handoff)/owners.
At Model0043 acceptance, planning/source/authority frontiers were None; no Engine memo is admitted. MAIN's serial review/integration
is closed; Parallel group None. Historical PlanningAuthorized=true; the brief retains [admission](../../../glossary.md#term-admission)/launch.

All-valid baseline: Prepare0001–0009 (including 0003A) + Model0034/0035 + Planning0008 +
Compiler0011 → [Prepare0010](tasks/0010-general-scan-projection-and-resource-handoff.md) → CPU0030–0033
F32/F64 domain closure → low-level join: Complete. 0010's accepted projection/resource evidence
remains in its brief. CPU0037 dense actual-read qualification is also Complete; none enables
length-aware execution.

Nearest length-aware DAG: Complete Model0037 → Model0038, plus Complete Compiler0013,
Planning0008 and Prepare0010 → [Prepare0011](tasks/0011-length-aware-scan-projection-and-resource-handoff.md)
Complete → separately gated CPU mixed-boundary/preflight/skipping capability. Cold/v5 and owned [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)
remain separate coarse gates, without detailed successor briefs or automatic launch.

Prepare0011 Complete, MAIN accepted 2026-10-24 13:00:15 UTC. At that completion [checkpoint](../../../glossary.md#term-checkpoint),
Prepare [Ready](../../../glossary.md#term-ready) frontier was None.
Tesla planning APPROVE, zero open P1/P2; MAIN [metadata](../../../glossary.md#term-metadata) accepted 2026-10-24. ImplementationAuthorized=true
records completed Prepare0011 only, authorized by the user's latest “pust se do prace” request.
Clean sole writer started 2026-10-24 12:31 UTC; implementation tested and frozen. MAIN's exact
[SPI](../../../glossary.md#term-service-provider-interface-spi)-consumer [compile](../../../glossary.md#compile) and four retained CPU/cold/[codec](../../../glossary.md#term-codec) guards PASS. Independent Class C code/docs
APPROVE, zero open P1/P2; MAIN overall acceptance is complete.
No new launch, downstream tasks, [commit](../../../glossary.md#term-commit) or push.

Current common base: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus all accepted
uncommitted work, including Complete Model0037/0038 and Compiler0013; MAIN reverified launch admission
2026-10-24 12:25 UTC.
Parallel group None; [scan](../../../glossary.md#general-differentiable-scan)/projection/resource/shared-document conflicts serialized, none active.
MAIN owns shared integration/documents and ran the narrow Prepare0011 handoff checkpoint;
the brief records all seven integration fields, dependency evidence and exact validation commands.
The brief records worker 71-focused/92-full test and final Javadoc evidence. No root/native/performance campaign; metadata tests are not
physical-history proof. Ready metadata and implementation authorization remain distinct gates.

Current priority: [true per-run symbolic batch](../../dynamic-batch-execution.md)<span data-glossary-navigation="true"> ([batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span> through one
[immutable](../../../glossary.md#term-immutable) compiled/[prepared execution](../../../glossary.md#prepared-execution--preparedexecution). **Prepare0012 reviewed DESIGN delivery: Complete**; the
[proposal](dynamic-batch-geometry-contract-proposal.md) has independent Class C APPROVE, zero open
P1/P2, and MAIN delivery acceptance, 2026-10-24. User “ano” explicitly approves architecture first,
then ordinary CPU forward/actual first-order backward and bounded final A/B. Prepare0012A is the
Complete authority cutover, enacted/MAIN-accepted 2026-10-24 after Pasteur definitive APPROVE,
zero open P1/P2, docs/glossary review complete.
[Prepare0013](tasks/0013-ordinary-dynamic-batch-cpu-vertical.md) is Complete: the sole serial
implementation is frozen, bounded ordinary [FLOAT32](../../../glossary.md#term-float32) CPU forward/actual first-order backward is
implemented, and final code/Javadoc/performance and shared-document reviews approve with zero open
P1/P2. MAIN's native-enabled root gate passed; MAIN accepted the capability on 2026-10-24.
Its verified brief owns [scope](../../../glossary.md#term-scope) and validation. Its completion authorized no automatic successor.
Complete Prepare0011 + Runtime0016 + Engine0021 + Compiler0013 + CPU0037 feed this decision,
reverified Complete (plus design Complete) by MAIN at 2026-10-24 17:41:11 UTC at the base above plus accepted uncommitted
work. All seven dependency/integration fields are in Prepare0012A; no competing writer is active,
Parallel group None; MAIN owns shared-document integration. Prepare0013 used one clean executor;
the final documentation context has exclusive serial [ownership](../../../glossary.md#term-ownership-meanings) of the shared support-status edits.
HEAD/origin/index are unchanged; MAIN's baseline captured 2435 files before writer changes.
Prepare0012A precedes conflicting CPU binder/resource [ABI](../../../glossary.md#term-application-binary-interface-abi) work, not static length semantics;
ordinary dynamic-B CPU forward/backward comes before [general Scan](../../../glossary.md#general-differentiable-scan). No automatic Java launch or
all-successor authorization follows; completed Prepare0011 remains Complete.

At the earlier checkpoint the frontier was None; current coupled CPU0081 Ready scope is above. Prepare0015 remains Complete. [Prepare0014 symbolic Scan CPU vertical](tasks/0014-symbolic-scan-cpu-vertical.md)<span data-glossary-navigation="true"> ([symbolic Scan glossary](../../../glossary.md#symbolic-scan))</span>
is Complete, MAIN accepted 2026-10-25 after independent code/docs APPROVE and the configured root gate.
PlanningAuthorized=true, ImplementationAuthorized=true record historical admission. User “jdi na to” authorized source;
MAIN verified Ready 2026-10-25 17:00 UTC on accepted planning, sole writer/inactive conflicts. Complete
Compiler0014 + Prepare0013/0011 + CPU0037 + Runtime0016 + Engine0021 + Model0041/0042 + Planning0008
feed one [atomic](../../../glossary.md#resource-transaction-and-rollback) Prepare→CPU→Engine F32 forward/actual-backward capability through existing owners.
Planning/code/docs review: APPROVE, 0 open P1/P2; source frontier None;
dependency/base/contract verification is recorded in the brief. Parallel group None; MAIN owns shared docs and the
single configured vertical join. No successor cascade or geometry memo is authorized.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Backend partition analysis and resource declaration](tasks/0001-backend-partition-analysis-and-resource-declaration.md) | Complete | [Compiler](../../../glossary.md#term-compiler) 0006; [Planning](../../../glossary.md#term-planning) 0006; [Runtime](../../../glossary.md#term-runtime) 0001; [ADR](../../../glossary.md#term-architecture-decision-record-adr) 0010 | Defines the analysis-side [Prepare](../../../glossary.md#prepare) projection, typed [backend](../../../glossary.md#backend) analyzer, [opaque](../../../glossary.md#term-opaque) selected plan, and exact [buffer](../../../glossary.md#term-buffer)/workspace declarations without assigning slots or finalizing executables. |
| 0002 | [Backend partition finalization handoff](tasks/0002-backend-partition-finalization-handoff.md)<span data-glossary-navigation="true"> ([Backend partition finalization glossary](../../../glossary.md#backend-partition-finalization))</span> | Complete | 0001; Runtime 0002–0004 | Assigns deterministic conservative shared slots across the complete ordered analyses, retains exact source associations, and finalizes each typed backend plan into the minimal [prepared partition](../../../glossary.md#prepared-partition--preparedpartition)/executable association. |
| 0003 | [Prepare orchestration and validation](tasks/0003-prepare-orchestration-and-validation.md) | Complete | 0001–0002; Compiler 0006; Planning 0006; Runtime 0002–0014 | Composes exact [compile](../../../glossary.md#compile) projection, typed backend analysis/finalization, initialized [constant](../../../glossary.md#term-constant) representations, prepared-memory assignment, complete schedule assembly/[validation](../../../glossary.md#term-validation), and final [prepared execution](../../../glossary.md#prepared-execution--preparedexecution) without [concrete backend](../../../glossary.md#concrete-backend) logic. |
| 0003A | [Immutable partition-local DAG analysis projection](tasks/0003a-immutable-partition-local-dag-analysis-projection.md) | Complete | 0001–0003; [CPU](../../../glossary.md#term-cpu) 0008B–0008E as downstream evidence | Added one public [immutable](../../../glossary.md#term-immutable) Prepare-owned projection for exactly one planned [partition](../../../glossary.md#partition), made it [`PrepareContext`](../../../glossary.md#prepare-context-preparecontext)'s sole [node](../../../glossary.md#node)/[topology](../../../glossary.md#term-topology) source, precomputed precise structural occurrences and adjacency, and kept the complete cross-backend [model](../../../glossary.md#term-model) [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) out of concrete backends. |
| 0004 | [Opaque backend candidate-batch and selected-decision handoff](tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md) | Complete | 0001–0003A; CPU 0010E | Added two method-free opaque roles and one immutable typed exact-partition [handoff](../../../glossary.md#term-handoff), then adopted them on CPU 0010E's existing [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) and decision without shared interpretation, measurement, [persistence](../../../glossary.md#term-persistence), compatibility handling, or route selection. |
| 0005 | [Producerless published-constant resource handoff and shared slot assignment](tasks/0005-producerless-published-constant-resource-handoff-and-shared-slot-assignment.md) | Complete | Compiler 0006B5; 0001–0004 | Contributed an explicit producerless published-constant resource to the complete preparation handoff and assigned its shared slot deterministically, without backend selection or physical geometry; preserved ordinary backend-analysis declarations and partition-connected projection. |
| 0006 | [Persistent prepared-resource finalization transaction](tasks/0006-persistent-prepared-resource-finalization-transaction.md)<span data-glossary-navigation="true"> ([transaction glossary](../../../glossary.md#term-transaction))</span> | Complete | Runtime 0016; ADR 0013 | Returns each executable and its acquisition-ordered persistent resources atomically, rolls [identity](../../../glossary.md#term-identity-meanings)-unique resources back across every later preparation failure, and transfers successful [ownership](../../../glossary.md#term-ownership-meanings) once to [`PreparedExecution`](../../../glossary.md#prepared-execution--preparedexecution); CPU remains resource-free. |
| 0007 | [Partition preparer guide status reconciliation](tasks/0007-partition-preparer-guide-status-reconciliation.md) | Complete | 0006; current CPU/[Engine](../../../glossary.md#term-engine)/tuning [lifecycle](../../../glossary.md#lifecycle) | Reconciled the focused guide with current fixed CPU preparation, Engine/tuning composition, per-[run](../../../glossary.md#term-run) representations, and transactional persistent-[resource ownership](../../../glossary.md#term-resource-ownership). |
| 0008 | [Backend partition finalization glossary status reconciliation](tasks/0008-backend-partition-finalization-glossary-status-reconciliation.md) | Complete | 0007; CPU 0010K; Engine 0013; current CPU finalizer/integration evidence | Corrected and independently reviewed the stale CPU composition and topology status in the shared finalization glossary entry. |
| 0009 | [Complete duplicate-resource rollback](tasks/0009-complete-resource-rollback-on-duplicate.md)<span data-glossary-navigation="true"> ([rollback glossary](../../../glossary.md#term-rollback))</span> | Complete | 0006/0008; project review | Retain trailing transferred [owners](../../../glossary.md#term-owner) before duplicate rejection and close each unique resource once. |
| 0010 | [General scan projection and resource handoff](tasks/0010-general-scan-projection-and-resource-handoff.md)<span data-glossary-navigation="true"> ([General scan glossary](../../../glossary.md#general-differentiable-scan))</span> | Complete | 0001–0009/0003A; Model0034/0035; Planning0008; Compiler0011 Complete | Exact scoped body/roles/demand validation and existing resource handoff; Huygens final independent Class C code/docs APPROVE and main exact compatibility [checkpoint](../../../glossary.md#term-checkpoint) accepted 2026-10-22; no real [scan](../../../glossary.md#general-differentiable-scan) execution. |
| 0011 | [Length-aware scan projection and resource handoff](tasks/0011-length-aware-scan-projection-and-resource-handoff.md) | Complete | Model0037/0038; Compiler0013; Planning0008; Prepare0010 Complete | Static projection/resource [metadata](../../../glossary.md#term-metadata) implemented; focused 71/full Prepare 92 tests, zero failures/errors/skips, [Javadoc](../../../glossary.md#term-javadoc) and MAIN [SPI](../../../glossary.md#term-service-provider-interface-spi)/four-[guard](../../../glossary.md#term-guard) checkpoint PASS. Independent code/docs APPROVE, zero open P1/P2; MAIN accepted 2026-10-24; [result](tasks/0011-length-aware-scan-projection-and-resource-handoff.md#result). |
| 0012 | [Dynamic batch geometry and resource contract](tasks/0012-dynamic-batch-geometry-and-resource-contract.md) | Complete | Prepare0011; Runtime0016; Engine0021; Compiler0013; CPU0037 Complete | Reviewed [DESIGN delivery](dynamic-batch-geometry-contract-proposal.md), independent Class C APPROVE and MAIN accepted; authority successor 0012A Complete, design history retained. |
| 0012A | [Dynamic batch authority cutover](tasks/0012a-dynamic-batch-authority-cutover.md) | Complete | Prepare0012 design and its five dependencies Complete | Authority/ADR0050/explanations/glossary enacted/MAIN-accepted 2026-10-24; Pasteur definitive APPROVE, zero open P1/P2. Authority-only checkpoint; 0012A itself enabled no Java capability. |
| 0013 | [Ordinary dynamic-batch CPU vertical](tasks/0013-ordinary-dynamic-batch-cpu-vertical.md) | Complete | 0012A and recorded predecessors Complete | Bounded ordinary [FLOAT32](../../../glossary.md#term-float32) CPU forward/actual first-order backward implemented; final independent reviews and native-enabled root gate passed, MAIN accepted 2026-10-24; [result](tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result). |
| 0014 | [Symbolic Scan CPU vertical](tasks/0014-symbolic-scan-cpu-vertical.md#result)<span data-glossary-navigation="true"> ([Symbolic Scan glossary](../../../glossary.md#symbolic-scan))</span> | Complete | Compiler0014; Prepare0013/0011; CPU0037; Runtime0016; Engine0021; Model0041/0042; Planning0008 Complete | Bounded [F32](../../../glossary.md#floating-point-precision-and-rounding) forward/actual backward implemented; independent code/docs APPROVE and source-matched native root gate PASS; MAIN accepted 2026-10-25. Bounded CPU/Engine probe results and limits in brief. |
| 0015 | [Symbolic-length Scan CPU vertical](tasks/0015-symbolic-length-scan-cpu-vertical.md#result) | Complete | Model0044; Compiler0015; Prepare0011/0013/0014; CPU0042; retained lifecycle predecessors Complete | F32/[F64](../../../glossary.md#floating-point-precision-and-rounding) named-B plus runtime lengths forward/actual reverse, bounded typed prerequisites; independent code/docs APPROVE and configured gates PASS, MAIN accepted 2026-10-27. Probe is [nonstationary](../../../glossary.md#term-nonstationary), not performance parity; frontier None. |

## Milestones and current frontier

- Active frontier: coupled CPU0081 [Ready](../../../glossary.md#term-ready) neutral activity/typed projection and context-copy closure; source/actual [qualification](../../../glossary.md#term-qualification) pending. Prepare0015 remains Complete with unchanged historical evidence.
- Prepare0013/0014 are Complete, MAIN-accepted 2026-10-24/25; their completion authorized no launch.
  Bounded symbolic F32 Scan execution is implemented; NN/TrainingSession reuse stays gated.<span data-glossary-navigation="true"> ([F32 glossary](../../../glossary.md#floating-point-precision-and-rounding); [Scan glossary](../../../glossary.md#general-differentiable-scan); [NN glossary](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [TrainingSession glossary](../../../glossary.md#training-session--trainingsession))</span>
- All-valid general-scan projection/resource handoff Complete through 0010; CPU F32/F64 closure
  and low-level join Complete. Length-aware 0011 Complete, MAIN accepted 2026-10-24 after worker gates
  and independent code/docs APPROVE, zero open P1/P2. At 0011 completion, Prepare Ready frontier was None. CPU length
  execution, cold/v5 and owned length-aware NN remain gated.<span data-glossary-navigation="true"> ([handoff glossary](../../../glossary.md#term-handoff); [CPU glossary](../../../glossary.md#term-cpu); [F64 glossary](../../../glossary.md#floating-point-precision-and-rounding); [Prepare glossary](../../../glossary.md#prepare))</span>
- Analysis/declaration, shared assignment/finalization, complete orchestration, and immutable
  partition-local projection are Complete through 0003A.<span data-glossary-navigation="true"> ([immutable glossary](../../../glossary.md#term-immutable); [partition glossary](../../../glossary.md#partition))</span>
- Opaque tuning transport, producerless published-constant assignment, and the persistent-resource
  transaction are Complete through 0006.<span data-glossary-navigation="true"> ([Opaque glossary](../../../glossary.md#term-opaque); [constant glossary](../../../glossary.md#term-constant); [resource transaction glossary](../../../glossary.md#resource-transaction-and-rollback))</span>
- Runtime 0016 and Engine 0010 complete the adjacent owner and outward-handle lifecycle.
  Model 0032's profile-free cutover and independently reviewed repair 0009 are Complete;
  historical Metal 0051 is not a pending Prepare prerequisite.<span data-glossary-navigation="true"> ([Runtime glossary](../../../glossary.md#term-runtime); [Engine glossary](../../../glossary.md#term-engine); [owner glossary](../../../glossary.md#term-owner); [lifecycle glossary](../../../glossary.md#lifecycle); [Model glossary](../../../glossary.md#term-model); [Metal glossary](../../../glossary.md#term-metal))</span>

## Live risks and gates

- Prepare0012's redesign, B=0/limits and actual publication with model-free Runtime/run-owned
  geometry are enacted/MAIN-accepted through Complete 0012A, 2026-10-24. Prepare0013 implements
  the bounded ordinary FLOAT32 CPU exception, MAIN-accepted after final reviews and the root gate on 2026-10-24.
  No mandatory public B_max or per-B full-preparation cache is the selected semantics.<span data-glossary-navigation="true"> ([publication glossary](../../../glossary.md#term-publication); [model glossary](../../../glossary.md#term-model); [Runtime glossary](../../../glossary.md#term-runtime); [run glossary](../../../glossary.md#term-run); [FLOAT32 glossary](../../../glossary.md#term-float32); [CPU glossary](../../../glossary.md#term-cpu); [cache glossary](../../../glossary.md#term-cache))</span>
- Dynamic dimension binding outside that qualified ordinary named-B domain remains unsupported.
  Route choice and complete resource declarations stay fixed before assignment/finalization;
  the accepted numeric recipes resolve actual run geometry without changing routes or identities.<span data-glossary-navigation="true"> ([identities glossary](../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([dimension glossary](../../../glossary.md#dimension); [binding glossary](../../../glossary.md#term-binding))</span>
- Keep backend candidates opaque and colocated with their owner. Later route work may use the
  existing transport but must not move backend fields, compatibility, numerical meaning, or
  route choice into Prepare.<span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend); [opaque glossary](../../../glossary.md#term-opaque); [owner glossary](../../../glossary.md#term-owner); [Prepare glossary](../../../glossary.md#prepare))</span>
- Preserve analysis-before-assignment-before-finalization and the one-time ownership transaction;
  never let finalization add undeclared shared resources or let Prepare execute physical work.<span data-glossary-navigation="true"> ([ownership glossary](../../../glossary.md#term-ownership-meanings); [transaction glossary](../../../glossary.md#term-transaction))</span>
- Producerless published constants contribute to complete assignment only; backend selection,
  physical geometry, initialization, and materialization stay with their existing owners.<span data-glossary-navigation="true"> ([constants glossary](../../../glossary.md#term-constant); [materialization glossary](../../../glossary.md#materialization))</span>

## Status [normalization](../../../glossary.md#term-normalization)

The task table and linked task status/results are controlling. The former downstream [lifecycle](../../../glossary.md#lifecycle)-
gate wording is normalized to the current state: [Engine](../../../glossary.md#term-engine) 0010, [Metal](../../../glossary.md#term-metal) 0002, Metal 0003, and [Prepare](../../../glossary.md#prepare)
0008 are Complete. Repair 0009 enforces the existing [ownership](../../../glossary.md#term-ownership-meanings) rule; it introduces no new [API](../../../glossary.md#term-application-programming-interface-api).

## History and update policy

Detailed results, [validation](../../../glossary.md#term-validation) commands, context identifiers, past ordering exceptions, and
completed frontier narratives remain in linked task files and Git history. Update this map only
when [ownership](../../../glossary.md#term-ownership-meanings), dependencies, task order/status, a milestone, or a live gate changes. Put
executable [scope](../../../glossary.md#term-scope) and evidence in the task brief and follow the
[planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>.
