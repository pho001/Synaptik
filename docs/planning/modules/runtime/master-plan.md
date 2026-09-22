# Runtime Master Plan

## Goal and authority

Runtime executes already-prepared schedules and owns the logical state, resources, residency,
publication, and cleanup of each run. This plan is a non-authoritative implementation map;
[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md), especially `Core lifecycle`, `Core invariants`,
`modules/runtime`, and `Run lifecycle`, is authoritative.

Focused explanations and decisions:

- [Lifecycle](../../../architecture/lifecycle.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)

## Lifecycle position

```text
compile -> prepare -> PreparedExecution -> run lease -> isolated RunState
                                      -> prepared schedule -> RunResult lease -> cleanup
                 close ----------------^         -> deferred prepared-resource cleanup if active
```

`PreparedExecution` is an immutable reusable recipe and the unique lifecycle owner of persistent
prepared resources. Each active invocation owns a distinct mutable `RunState`; published results
lease that state until `RunResult.close()`.

## Scope and non-goals

Runtime owns prepared memory identities and geometry, prepared executable/transfer/schedule
contracts, per-run binding and validity, explicit residency and transfer work, publication,
results, the runner, persistent-resource lifetime, and passive observation through Trace.

Runtime does not optimize or inspect graphs on the hot path, construct autograd, discover or
select backends, lower partitions, select kernels, interpret tuning state, or implement concrete
backend storage and mechanics.

## Stable invariants and ownership

- Runtime executes only work fixed during preparation. The hot path sees neither `Operation` nor
  `CompiledNode` and performs no backend lookup, reflection, string dispatch, or route selection.
- `PreparedExecution`, its plan, schedule, executable recipes, and persistent prepared resources
  are immutable and reusable. The owner has identity semantics and exposes no resource lookup.
- Persistent resources are snapshotted by exact identity once, closed in reverse acquisition
  order, and never inferred from executable or schedule occurrences.
- A synchronous run acquires the sole prepared-resource lease before creating state. Close is
  idempotent, rejects later leases, never waits for admitted runs, and defers reverse attempt-all
  cleanup to the final lease when necessary.
- Every active logical run has one isolated `RunState`. Caller inputs are borrowed; internal
  buffers and workspaces are run-owned; a `RunResult` leases the complete state. Cleanup never
  releases borrowed inputs.
- Cold binding checks concrete representation compatibility once and creates backend-owned typed
  direct-reference actions. Concrete backends own allocation, release, transfer, and access.
- Cleanup preserves the first unchecked failure or error, suppresses later distinct failures in
  encounter order, skips repeated identity of the primary throwable, and remains idempotent.
- Runtime owns no concrete backend dependency, backend selection, or persistent global state.

## Dependencies

Current direct dependencies are `modules/config`, `modules/backend-contract`, and `modules/trace`.
Architecture-approved Model contracts may be added only for a concrete Runtime need; none is
currently declared. Dependencies on `modules/engine` and every concrete backend are forbidden.

## Package map

| Package | Public or shared role | Internal boundary |
|---|---|---|
| `runtime.memory` | Plan-local buffer/workspace slots and immutable final geometry. | No graph identity, allocation, residency, or physical storage. |
| `runtime.resource` | Nominal physical representation roles, immutable creation descriptions, and `PreparedResource`. | Physical implementations and release mechanics stay in concrete backends. |
| `runtime.execution` | Prepared execution owner, executable/transfer recipes, and cold-bound actions. | Binding validates once; bound actions retain direct typed references. |
| `runtime.schedule` | Immutable exact-plan ordered creation, execution, transfer, and publication recipe. | The schedule neither discovers work nor owns resources by occurrence. |
| `runtime.run` | Per-run ownership, creation, validity, publication, result leases, and runner orchestration. | `RunStateCreation` remains package-private; mutable state is never shared across runs. |

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Prepared buffer-slot identity](tasks/0001-prepared-buffer-slot-identity.md) | Complete | Compiler 0006; Planning 0006; Backend Contract 0004; Trace 0002 | Replaced the placeholder with one immutable prepared-plan-local `BufferSlot` identity required to bind later prepared-unit inputs and outputs, without physical storage, allocation, graph values, workspace, execution, or run state. |
| 0002 | [Prepared memory and workspace contracts](tasks/0002-prepared-memory-and-workspace-contracts.md) | Complete | 0001; Prepare 0001 | Added `WorkspaceSlot` and immutable final per-buffer-slot/per-workspace-slot byte-size/alignment geometry without importing Prepare/Model facts; Complete Prepare 0002 retains exact requirement associations and conservatively assigns distinct slots. |
| 0003 | [Run-state and runtime resource foundation](tasks/0003-run-state-and-runtime-resource-foundation.md) | Complete | 0002; ADR 0011 | Added nominal backend-owned buffer/workspace representation roles, borrowed/run-owned buffer bindings, and one array-backed closeable `RunState` per complete logical run without executable binding, residency, scheduling, transfer, publication, or allocation. |
| 0004 | [Prepared executable and bound invocation](tasks/0004-prepared-executable-and-bound-invocation.md) | Complete | 0001–0003; ADR 0011 | Added immutable dense resource selections, final checked cold binding, and one per-run backend-owned typed bound invocation without a redundant prepared-unit wrapper. |
| 0005 | [Prepared schedule contract](tasks/0005-prepared-schedule-contract.md) | Complete | 0002–0004; Prepare 0002 finalization | Added one immutable exact-plan schedule and its execution-step variant; no `PreparedUnit`, transfer, materialization, or publication payload is invented before its Runtime-owned facts exist. |
| 0006 | [Prepared execution aggregate](tasks/0006-prepared-execution-aggregate.md) | Complete | 0002–0005 | Added the smallest exact-plan/exact-schedule immutable Runtime root while keeping every invocation mutation and resource lifecycle in `RunState`. |
| 0007 | [Representation creation and residency foundation](tasks/0007-representation-creation-and-residency-foundation.md) | Complete | 0003; 0005–0006; Prepare 0002 | Added immutable caller-input/backend-creation descriptions, deterministic per-run creation and rollback, schedule reachability, and explicit per-copy validity without transfer, execution, or Config. |
| 0008 | [Prepared buffer transfer and materialization schedule](tasks/0008-prepared-buffer-transfer-and-materialization-schedule.md) | Complete | 0004–0005; 0007 | Added one backend-supplied prepared/bound buffer-transfer pair and schedule occurrence; materialization is the same action between distinct already-created representations, with destination-valid no-op and success-only validity transition. |
| 0009 | [Publication and result schedule steps](tasks/0009-publication-and-result-schedule-steps.md) | Complete | 0005; 0007–0008; stable publication/result ownership | Added exact prepared/run coordinates and a dense publication suffix, cold-bound direct selected representations, and leased the complete `RunState` to an ordered `RunResult` without importing Compiler publication identities or exposing output values. |
| 0010 | [Prepared runner and dynamic execution](tasks/0010-prepared-runner-and-dynamic-execution.md) | Complete | 0003; 0005–0009; preserve Trace 0001–0002 boundary | Cold-creates one isolated state, binds every direct occurrence before ordered traversal, applies explicit executable read/write validity, and transfers the whole-state result lease without hot graph/backend lookup. |
| 0011 | [Runtime contract closure audit](tasks/0011-runtime-contract-closure-audit.md) | Complete | 0001–0010 | Recorded `BLOCKING_GAP`: the audit is complete, but shared-throwable cleanup, stale general architecture status, and absent Runtime architecture enforcement keep the milestone open. |
| 0012 | [Run-state shared-throwable cleanup](tasks/0012-run-state-shared-throwable-cleanup.md) | Complete | 0003; 0007; 0009–0011 | Repaired `RUNTIME-CLEANUP-001`: closed-first reverse cleanup now skips impossible self-suppression when distinct owned resources throw the same exact primary `Throwable`, preserves that primary, and attempts all remaining owned resources. |
| 0013 | [General architecture status correction](tasks/0013-general-architecture-status-correction.md) | Complete | 0011; 0012 | Corrected `DOCUMENTATION-STATUS-001` in four implicated explanatory documents, preserving authoritative architecture and leaving enforcement to task 0014. |
| 0014 | [Runtime architecture enforcement](tasks/0014-runtime-architecture-enforcement.md) | Complete | 0011; 0012–0013 | Added one dependency-free focused architecture suite that locks Runtime's exact project edges, exhaustively classifies production sources, and rejects `Operation`/`CompiledNode` in the explicit hot path. |
| [0015](tasks/0015-leased-publication-representation-access.md) | Leased publication representation access | Complete | 0009–0010; 0012; Engine 0003 | Added one result-indexed borrowed `BufferRepresentation` reference while the exact `RunResult` lease is open, preserving aliases and whole-state cleanup without adding host copying, concrete-backend knowledge, or ordinary Engine leakage. |
| [0016](tasks/0016-persistent-prepared-resource-lifecycle.md) | Persistent prepared-resource lifecycle | Complete | 0003–0015; ADR 0013; Metal 0002 blocker audit | Added the nominal backend-implemented prepared resource, made `PreparedExecution` its identity-unique lifecycle owner, and leased it around synchronous runner calls with non-waiting deferred reverse cleanup. |

## Milestones and current frontier

- Prepared memory, execution recipes, per-run state, transfers, publication, and the runner are
  Complete through 0010.
- The 0011 audit findings are resolved by Complete tasks 0012–0014; 0011 remains a completed
  historical `BLOCKING_GAP` result, not a live blocker.
- Leased publication access and persistent prepared-resource ownership are Complete through 0016.
- No Runtime task is `Ready` or `In progress`. The [roadmap](../../roadmap.md) owns the repository
  frontier and currently selects Draft Metal 0004 for reassessment.

## Live risks and gates

- Preserve the single Runtime lease authority, per-run isolation, non-waiting close, and exact
  reverse attempt-all cleanup when adding any prepared-resource consumer.
- Do not turn prepared transfer/materialization into route search, allocation, hidden coherence,
  or on-demand backend selection.
- Keep executable access declarations explicit and finish all cold binding before traversal; do
  not infer reads/writes from selection order or introduce lookup/dispatch in the hot path.
- New public output forms remain Engine composition concerns. Runtime may lend a representation
  only while the exact `RunResult` lease is open.

## Status normalization

The task table and linked task status/results are controlling. The former downstream-frontier
wording is normalized to the current state: Prepare 0006, Engine 0010, Metal 0002, and Metal 0003
are Complete; Metal 0004 is the current Draft planning frontier. No task status, order,
dependency, ownership rule, API, or executable behavior changes here.

## History and update policy

Detailed results, validation commands, context identifiers, audits, and past frontier narratives
remain in linked task files and Git history. Update this map only when ownership, dependencies,
task order/status, a milestone, or a live gate changes. Put executable scope and evidence in the
task brief and follow the [planning guide](../../planning-guide.md).
