# [Runtime](../../../glossary.md#term-runtime) Master Plan

## Goal and authority

[Runtime](../../../glossary.md#term-runtime) executes already-prepared schedules and owns the logical state, resources, [residency](../../../glossary.md#residency),
[publication](../../../glossary.md#term-publication), and cleanup of each [run](../../../glossary.md#term-run). This plan is a non-authoritative implementation map;
[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md), especially `Core lifecycle`, `Core invariants`,
`modules/runtime`, and `Run lifecycle`, is authoritative.

Focused explanations and decisions:

- [Lifecycle](../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../glossary.md#term-module))</span>
- [Dependency rules](../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>
- [Runtime, Prepare, and Backend Boundary](../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../glossary.md#prepare); [Backend glossary](../../../glossary.md#backend))</span>
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#term-architecture-decision-record-adr))</span>

## [Lifecycle](../../../glossary.md#lifecycle) position

```text
compile -> prepare -> PreparedExecution -> run lease -> isolated RunState
                                      -> prepared schedule -> RunResult lease -> cleanup
                 close ----------------^         -> deferred prepared-resource cleanup if active
```

[`PreparedExecution`](../../../glossary.md#prepared-execution--preparedexecution) is an [immutable](../../../glossary.md#term-immutable) reusable recipe and the unique lifecycle [owner](../../../glossary.md#term-owner) of persistent
prepared resources. Each active invocation owns a distinct [mutable](../../../glossary.md#term-mutable) [`RunState`](../../../glossary.md#run-state--runstate); published results
[lease](../../../glossary.md#term-lease) that state until `RunResult.close()`.

## [Scope](../../../glossary.md#term-scope) and non-goals

[Runtime](../../../glossary.md#term-runtime) owns prepared memory [identities](../../../glossary.md#term-identity-meanings) and geometry, [prepared executable](../../../glossary.md#prepared-executable--preparedexecutable)/transfer/schedule
contracts, per-[run](../../../glossary.md#term-run) [binding](../../../glossary.md#term-binding) and validity, explicit [residency](../../../glossary.md#residency) and transfer work, [publication](../../../glossary.md#term-publication),
results, the runner, persistent-[resource lifetime](../../../glossary.md#ownership-borrowing-and-lease), and passive observation through [Trace](../../../glossary.md#trace).

Runtime does not optimize or inspect [graphs](../../../glossary.md#graph) on the [hot path](../../../glossary.md#term-hot-path), construct [autograd](../../../glossary.md#autograd), discover or
select [backends](../../../glossary.md#backend), lower partitions, select [kernels](../../../glossary.md#kernel), interpret tuning state, or implement [concrete
backend](../../../glossary.md#concrete-backend) storage and mechanics.

## Stable invariants and [ownership](../../../glossary.md#term-ownership-meanings)

- [Runtime](../../../glossary.md#term-runtime) executes only work fixed during preparation. The [hot path](../../../glossary.md#term-hot-path) sees neither [`Operation`](../../../glossary.md#operation) nor
  `CompiledNode` and performs no [backend](../../../glossary.md#backend) [lookup](../../../glossary.md#term-lookup), reflection, string [dispatch](../../../glossary.md#term-dispatch), or route selection.
- [`PreparedExecution`](../../../glossary.md#prepared-execution--preparedexecution), its plan, schedule, executable recipes, and persistent prepared resources
  are immutable and reusable. The owner has identity semantics and exposes no resource lookup.<span data-glossary-navigation="true"> ([immutable glossary](../../../glossary.md#term-immutable); [owner glossary](../../../glossary.md#term-owner); [identity glossary](../../../glossary.md#term-identity-meanings))</span>
- Persistent resources are snapshotted by exact identity once, closed in reverse acquisition
  order, and never inferred from executable or schedule occurrences.
- A synchronous run acquires the sole prepared-resource lease before creating state. Close is
  idempotent, rejects later leases, never waits for admitted runs, and defers reverse attempt-all
  cleanup to the final lease when necessary.<span data-glossary-navigation="true"> ([synchronous glossary](../../../glossary.md#term-synchronous); [resource lease glossary](../../../glossary.md#ownership-borrowing-and-lease); [idempotent glossary](../../../glossary.md#term-idempotent); [leases glossary](../../../glossary.md#term-lease))</span>
- Every active logical [run](../../../glossary.md#term-run) has one isolated [`RunState`](../../../glossary.md#run-state--runstate). Caller inputs are borrowed; internal
  buffers and workspaces are run-owned; a <span data-glossary-navigation="true"> ([borrowed glossary](../../../glossary.md#ownership-borrowing-and-lease); [buffers glossary](../../../glossary.md#term-buffer))</span>[`RunResult`](../../../glossary.md#run-result--runresult) leases the complete state. Cleanup never
  releases borrowed inputs.<span data-glossary-navigation="true"> ([releases glossary](../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Cold binding checks concrete representation compatibility once and creates backend-owned typed
  direct-reference actions. Concrete backends own allocation, release, transfer, and access.<span data-glossary-navigation="true"> ([binding glossary](../../../glossary.md#term-binding); [release glossary](../../../glossary.md#term-release))</span>
- Cleanup preserves the first unchecked failure or error, suppresses later distinct failures in
  encounter order, skips repeated identity of the primary throwable, and remains idempotent.
- Runtime owns no [concrete backend](../../../glossary.md#concrete-backend) [dependency](../../../glossary.md#term-dependency), backend selection, or persistent global state.

## Dependencies

Current direct dependencies are `modules/config`, `modules/backend-contract`, and `modules/trace`.
Architecture-approved [Model](../../../glossary.md#term-model) contracts may be added only for a concrete [Runtime](../../../glossary.md#term-runtime) need; none is
currently declared. Dependencies on `modules/engine` and every [concrete backend](../../../glossary.md#concrete-backend) are forbidden.

## Package map

| Package | Public or shared role | Internal boundary |
|---|---|---|
| `runtime.memory` | Plan-local [buffer](../../../glossary.md#term-buffer)/workspace slots and [immutable](../../../glossary.md#term-immutable) final geometry. | No [graph](../../../glossary.md#graph) [identity](../../../glossary.md#term-identity-meanings), allocation, [residency](../../../glossary.md#residency), or physical storage. |
| `runtime.resource` | [Nominal](../../../glossary.md#term-nominal) physical representation roles, immutable creation descriptions, and [`PreparedResource`](../../../glossary.md#prepared-resource--preparedresource). | Physical implementations and [release](../../../glossary.md#term-release) mechanics stay in concrete [backends](../../../glossary.md#backend). |
| `runtime.execution` | [Prepared execution](../../../glossary.md#prepared-execution--preparedexecution) [owner](../../../glossary.md#term-owner), executable/transfer recipes, and cold-bound actions. | [Binding](../../../glossary.md#term-binding) validates once; bound actions retain direct typed references. |
| `runtime.schedule` | Immutable exact-plan ordered creation, execution, transfer, and [publication](../../../glossary.md#term-publication) recipe. | The schedule neither discovers work nor owns resources by [occurrence](../../../glossary.md#term-occurrence). |
| `runtime.run` | Per-[run](../../../glossary.md#term-run) [ownership](../../../glossary.md#term-ownership-meanings), creation, validity, publication, result [leases](../../../glossary.md#term-lease), and runner orchestration. | `RunStateCreation` remains package-private; [mutable](../../../glossary.md#term-mutable) state is never shared across runs. |

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Prepared buffer-slot identity](tasks/0001-prepared-buffer-slot-identity.md) | Complete | [Compiler](../../../glossary.md#term-compiler) 0006; [Planning](../../../glossary.md#term-planning) 0006; [Backend](../../../glossary.md#backend) Contract 0004; [Trace](../../../glossary.md#trace) 0002 | Replaced the placeholder with one [immutable](../../../glossary.md#term-immutable) prepared-plan-local [`BufferSlot`](../../../glossary.md#buffer-slot--bufferslot) [identity](../../../glossary.md#term-identity-meanings) required to bind later prepared-unit inputs and outputs, without physical storage, allocation, [graph](../../../glossary.md#graph) values, workspace, execution, or [run state](../../../glossary.md#run-state--runstate). |
| 0002 | [Prepared memory and workspace contracts](tasks/0002-prepared-memory-and-workspace-contracts.md) | Complete | 0001; [Prepare](../../../glossary.md#prepare) 0001 | Added [`WorkspaceSlot`](../../../glossary.md#workspace-slot--workspaceslot) and immutable final per-[buffer](../../../glossary.md#term-buffer)-slot/per-workspace-slot byte-size/[alignment](../../../glossary.md#term-alignment) geometry without importing Prepare/[Model](../../../glossary.md#term-model) facts; Complete Prepare 0002 retains exact requirement associations and conservatively assigns distinct slots. |
| 0003 | [Run-state and runtime resource foundation](tasks/0003-run-state-and-runtime-resource-foundation.md) | Complete | 0002; [ADR](../../../glossary.md#term-architecture-decision-record-adr) 0011 | Added [nominal](../../../glossary.md#term-nominal) backend-owned buffer/workspace representation roles, [borrowed](../../../glossary.md#ownership-borrowing-and-lease)/[run](../../../glossary.md#term-run)-owned buffer [bindings](../../../glossary.md#term-binding), and one array-backed closeable [`RunState`](../../../glossary.md#run-state--runstate) per complete logical run without executable binding, [residency](../../../glossary.md#residency), scheduling, transfer, [publication](../../../glossary.md#term-publication), or allocation. |
| 0004 | [Prepared executable and bound invocation](tasks/0004-prepared-executable-and-bound-invocation.md)<span data-glossary-navigation="true"> ([Prepared executable glossary](../../../glossary.md#prepared-executable--preparedexecutable))</span> | Complete | 0001–0003; ADR 0011 | Added immutable dense resource selections, final checked cold binding, and one per-run backend-owned typed [bound invocation](../../../glossary.md#bound-invocation--boundinvocation) without a redundant prepared-unit wrapper. |
| 0005 | [Prepared schedule contract](tasks/0005-prepared-schedule-contract.md)<span data-glossary-navigation="true"> ([Prepared schedule glossary](../../../glossary.md#prepared-schedule--preparedschedule))</span> | Complete | 0002–0004; Prepare 0002 finalization | Added one immutable exact-plan schedule and its execution-step variant; no `PreparedUnit`, transfer, [materialization](../../../glossary.md#materialization), or publication [payload](../../../glossary.md#artifact-envelope-and-atomic-publication) is invented before its [Runtime](../../../glossary.md#term-runtime)-owned facts exist. |
| 0006 | [Prepared execution aggregate](tasks/0006-prepared-execution-aggregate.md)<span data-glossary-navigation="true"> ([Prepared execution glossary](../../../glossary.md#prepared-execution--preparedexecution))</span> | Complete | 0002–0005 | Added the smallest exact-plan/exact-schedule immutable Runtime root while keeping every invocation mutation and resource [lifecycle](../../../glossary.md#lifecycle) in `RunState`. |
| 0007 | [Representation creation and residency foundation](tasks/0007-representation-creation-and-residency-foundation.md) | Complete | 0003; 0005–0006; Prepare 0002 | Added immutable caller-input/backend-creation descriptions, deterministic per-run creation and [rollback](../../../glossary.md#term-rollback), schedule reachability, and explicit per-copy validity without transfer, execution, or Config. |
| 0008 | [Prepared buffer transfer and materialization schedule](tasks/0008-prepared-buffer-transfer-and-materialization-schedule.md)<span data-glossary-navigation="true"> ([Prepared buffer transfer glossary](../../../glossary.md#prepared-buffer-transfer--preparedbuffertransfer))</span> | Complete | 0004–0005; 0007 | Added one backend-supplied prepared/bound buffer-transfer pair and schedule [occurrence](../../../glossary.md#term-occurrence); materialization is the same action between distinct already-created representations, with destination-valid no-op and success-only validity transition. |
| 0009 | [Publication and result schedule steps](tasks/0009-publication-and-result-schedule-steps.md) | Complete | 0005; 0007–0008; stable publication/result [ownership](../../../glossary.md#term-ownership-meanings) | Added exact prepared/run [coordinates](../../../glossary.md#axis-rank-scalar-vector-and-matrix) and a dense publication suffix, cold-bound direct selected representations, and leased the complete `RunState` to an ordered [`RunResult`](../../../glossary.md#run-result--runresult) without importing Compiler publication identities or exposing output values. |
| 0010 | [Prepared runner and dynamic execution](tasks/0010-prepared-runner-and-dynamic-execution.md) | Complete | 0003; 0005–0009; preserve Trace 0001–0002 boundary | Cold-creates one isolated state, binds every direct occurrence before ordered traversal, applies explicit executable read/write validity, and transfers the whole-state result [lease](../../../glossary.md#term-lease) without hot graph/backend [lookup](../../../glossary.md#term-lookup). |
| 0011 | [Runtime contract closure audit](tasks/0011-runtime-contract-closure-audit.md) | Complete | 0001–0010 | Recorded `BLOCKING_GAP`: the audit is complete, but shared-throwable cleanup, stale general architecture status, and absent Runtime architecture enforcement keep the milestone open. |
| 0012 | [Run-state shared-throwable cleanup](tasks/0012-run-state-shared-throwable-cleanup.md) | Complete | 0003; 0007; 0009–0011 | Repaired `RUNTIME-CLEANUP-001`: closed-first reverse cleanup now skips impossible self-suppression when distinct owned resources throw the same exact primary `Throwable`, preserves that primary, and attempts all remaining owned resources. |
| 0013 | [General architecture status correction](tasks/0013-general-architecture-status-correction.md) | Complete | 0011; 0012 | Corrected `DOCUMENTATION-STATUS-001` in four implicated explanatory documents, preserving authoritative architecture and leaving enforcement to task 0014. |
| 0014 | [Runtime architecture enforcement](tasks/0014-runtime-architecture-enforcement.md) | Complete | 0011; 0012–0013 | Added one [dependency](../../../glossary.md#term-dependency)-free focused architecture suite that locks Runtime's exact project edges, exhaustively classifies production sources, and rejects [`Operation`](../../../glossary.md#operation)/`CompiledNode` in the explicit [hot path](../../../glossary.md#term-hot-path). |
| [0015](tasks/0015-leased-publication-representation-access.md) | Leased publication representation access | Complete | 0009–0010; 0012; [Engine](../../../glossary.md#term-engine) 0003 | Added one result-indexed borrowed `BufferRepresentation` reference while the exact `RunResult` lease is open, preserving [aliases](../../../glossary.md#aliasing-and-arena) and whole-state cleanup without adding [host](../../../glossary.md#term-host) copying, concrete-backend knowledge, or ordinary Engine leakage. |
| [0016](tasks/0016-persistent-prepared-resource-lifecycle.md) | Persistent prepared-resource lifecycle | Complete | 0003–0015; ADR 0013; [Metal](../../../glossary.md#term-metal) 0002 blocker audit | Added the nominal backend-implemented [prepared resource](../../../glossary.md#prepared-resource--preparedresource), made [`PreparedExecution`](../../../glossary.md#prepared-execution--preparedexecution) its identity-unique lifecycle [owner](../../../glossary.md#term-owner), and leased it around [synchronous](../../../glossary.md#term-synchronous) runner calls with non-waiting deferred reverse cleanup. |

## Milestones and current frontier

- Prepared memory, execution recipes, per-run state, transfers, publication, and the runner are
  Complete through 0010.<span data-glossary-navigation="true"> ([run state glossary](../../../glossary.md#run-state--runstate))</span>
- The 0011 audit findings are resolved by Complete tasks 0012–0014; 0011 remains a completed
  historical `BLOCKING_GAP` result, not a live blocker.
- Leased [publication](../../../glossary.md#term-publication) access and persistent prepared-resource [ownership](../../../glossary.md#term-ownership) are Complete through 0016.
- No [Runtime](../../../glossary.md#term-runtime) implementation task is [Ready](../../../glossary.md#term-ready). The [roadmap](../../roadmap.md) owns cross-area
  frontiers; Model 0032 is Complete and Runtime remains profile-free.<span data-glossary-navigation="true"> ([Model glossary](../../../glossary.md#term-model))</span>

## Live risks and gates

- Preserve the single Runtime lease authority, per-run isolation, non-waiting close, and exact
  reverse attempt-all cleanup when adding any prepared-resource consumer.<span data-glossary-navigation="true"> ([Runtime glossary](../../../glossary.md#term-runtime); [run glossary](../../../glossary.md#term-run))</span>
- Do not turn prepared transfer/materialization into route search, allocation, hidden coherence,
  or on-demand backend selection.<span data-glossary-navigation="true"> ([materialization glossary](../../../glossary.md#materialization); [backend glossary](../../../glossary.md#backend))</span>
- Keep executable access declarations explicit and finish all cold binding before traversal; do
  not infer reads/writes from selection order or introduce lookup/dispatch in the hot path.<span data-glossary-navigation="true"> ([binding glossary](../../../glossary.md#term-binding); [lookup glossary](../../../glossary.md#term-lookup); [dispatch glossary](../../../glossary.md#term-dispatch); [hot path glossary](../../../glossary.md#term-hot-path))</span>
- New public output forms remain Engine composition concerns. Runtime may lend a representation
  only while the exact <span data-glossary-navigation="true"> ([Engine glossary](../../../glossary.md#term-engine))</span>[`RunResult`](../../../glossary.md#run-result--runresult) [lease](../../../glossary.md#term-lease) is open.

## Status [normalization](../../../glossary.md#term-normalization)

The task table and linked task status/results are controlling. The former downstream-frontier
wording is normalized to the current state: [Prepare](../../../glossary.md#prepare) 0006, [Engine](../../../glossary.md#term-engine) 0010, and current [Metal](../../../glossary.md#term-metal) [lifecycle](../../../glossary.md#lifecycle)
work are Complete; [Model](../../../glossary.md#term-model) 0032 is Complete. No [Runtime](../../../glossary.md#term-runtime) task
status, order, [dependency](../../../glossary.md#term-dependency), [ownership](../../../glossary.md#term-ownership-meanings) rule, [API](../../../glossary.md#term-application-programming-interface-api), or executable behavior changes here.

## History and update policy

Detailed results, [validation](../../../glossary.md#term-validation) commands, context identifiers, audits, and past frontier narratives
remain in linked task files and Git history. Update this map only when [ownership](../../../glossary.md#term-ownership-meanings), dependencies,
task order/status, a milestone, or a live gate changes. Put executable [scope](../../../glossary.md#term-scope) and evidence in the
task brief and follow the [planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span>.
