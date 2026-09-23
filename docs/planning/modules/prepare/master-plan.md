# Prepare Master Plan

## Goal and authority

Prepare coordinates the validated transition from immutable compile artifacts to one complete
Runtime-ready recipe. This plan is a non-authoritative implementation map;
[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md), especially `Core lifecycle`,
`modules/prepare`, and `Prepare lifecycle`, is authoritative.

Focused explanations and decisions:

- [Lifecycle](../../../architecture/lifecycle.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../../../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)

## Lifecycle position

```text
CompileArtifacts
  -> partition-local analysis and exact resource declarations
  -> shared buffer/workspace slot assignment
  -> backend finalization against assigned slots
  -> transactional resource aggregation
  -> complete schedule assembly and validation
  -> one-time ownership transfer to PreparedExecution
```

Prepare stages and validates this transaction. Concrete backends analyze and finalize their own
work; Runtime later owns and executes the accepted recipe.

## Scope and non-goals

Prepare owns partition projection, backend-neutral analysis/finalization collaborations, exact
resource declarations, shared assignment, producerless published-constant contribution, opaque
candidate transport, complete orchestration, schedule assembly, validation, rollback, and final
ownership transfer.

Prepare does not execute work, implement concrete CPU/Metal/CUDA lowering or storage, select a
backend globally, interpret backend route/candidate fields, measure or rank tuning candidates,
mutate caches, or allocate per-run physical representations.

## Stable invariants and ownership

- Backend analysis deterministically selects and retains backend-private lowering and route state
  while declaring every shared buffer/workspace requirement before assignment.
- Shared Prepare assigns stable Runtime slots across the complete ordered analyses without
  interpreting concrete storage or route vocabulary.
- Finalization receives the exact analysis, shared plan, and assignments. It constructs the
  executable only after assignment and may not revise the route or introduce undeclared shared
  requirements.
- A finalizer owns every persistent resource until its complete result returns. Shared Prepare
  then owns each exact identity transactionally and rolls back once in deterministic reverse
  acquisition order after any later failure.
- Duplicate resource identity is rejected without double close. Rollback preserves the primary
  preparation failure and suppresses later distinct cleanup failures while avoiding
  self-suppression.
- Successful `PreparedExecution(memoryPlan, schedule, resources)` construction is the sole
  ownership-transfer point. Prepare derives no ownership from executable or schedule occurrences.
- Candidate batches and decisions cross shared code opaquely. Safe preparation never depends on a
  tuning search, and Runtime receives no cache or candidate state.
- Prepare constructs recipes and validates coverage; it performs no execution.

## Dependencies

Current allowed direct dependencies are `modules/runtime`, `modules/planning`,
`modules/compiler`, `modules/config`, `modules/backend-contract`, and `modules/trace`.
Dependencies on concrete backend implementations are forbidden; Engine supplies concrete
collaborators from the outer composition root.

## Package map

| Package | Public or shared role | Internal boundary |
|---|---|---|
| `prepare.analysis` | Immutable partition-local DAG/context, opaque backend input/plan/tuning roles, exact resource declarations, and analyzer collaboration. | No `CompileArtifacts` aggregate, complete graph, tuning interpretation, slot assignment, executable, or physical resource. |
| `prepare` root | Finalization input/result, assignments, `PreparedPartition`, producerless resources, `GraphPreparation`, and schedule assembly/validation contracts. | Complete-set handoff and transactional tracking stay package-private; concrete backends own lowering and physical mechanics. |

The public Prepare surface is cross-module service-provider interface (SPI), not the ordinary
Engine user facade.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Backend partition analysis and resource declaration](tasks/0001-backend-partition-analysis-and-resource-declaration.md) | Complete | Compiler 0006; Planning 0006; Runtime 0001; ADR 0010 | Defines the analysis-side Prepare projection, typed backend analyzer, opaque selected plan, and exact buffer/workspace declarations without assigning slots or finalizing executables. |
| 0002 | [Backend partition finalization handoff](tasks/0002-backend-partition-finalization-handoff.md) | Complete | 0001; Runtime 0002–0004 | Assigns deterministic conservative shared slots across the complete ordered analyses, retains exact source associations, and finalizes each typed backend plan into the minimal prepared partition/executable association. |
| 0003 | [Prepare orchestration and validation](tasks/0003-prepare-orchestration-and-validation.md) | Complete | 0001–0002; Compiler 0006; Planning 0006; Runtime 0002–0014 | Composes exact compile projection, typed backend analysis/finalization, initialized constant representations, prepared-memory assignment, complete schedule assembly/validation, and final prepared execution without concrete backend logic. |
| 0003A | [Immutable partition-local DAG analysis projection](tasks/0003a-immutable-partition-local-dag-analysis-projection.md) | Complete | 0001–0003; CPU 0008B–0008E as downstream evidence | Added one public immutable Prepare-owned projection for exactly one planned partition, made it `PrepareContext`'s sole node/topology source, precomputed precise structural occurrences and adjacency, and kept the complete cross-backend model DAG out of concrete backends. |
| 0004 | [Opaque backend candidate-batch and selected-decision handoff](tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md) | Complete | 0001–0003A; CPU 0010E | Added two method-free opaque roles and one immutable typed exact-partition handoff, then adopted them on CPU 0010E's existing batch and decision without shared interpretation, measurement, persistence, compatibility handling, or route selection. |
| 0005 | [Producerless published-constant resource handoff and shared slot assignment](tasks/0005-producerless-published-constant-resource-handoff-and-shared-slot-assignment.md) | Complete | Compiler 0006B5; 0001–0004 | Contributed an explicit producerless published-constant resource to the complete preparation handoff and assigned its shared slot deterministically, without backend selection or physical geometry; preserved ordinary backend-analysis declarations and partition-connected projection. |
| 0006 | [Persistent prepared-resource finalization transaction](tasks/0006-persistent-prepared-resource-finalization-transaction.md) | Complete | Runtime 0016; ADR 0013 | Returns each executable and its acquisition-ordered persistent resources atomically, rolls identity-unique resources back across every later preparation failure, and transfers successful ownership once to `PreparedExecution`; CPU remains resource-free. |
| 0007 | [Partition preparer guide status reconciliation](tasks/0007-partition-preparer-guide-status-reconciliation.md) | Complete | 0006; current CPU/Engine/tuning lifecycle | Reconciled the focused guide with current fixed CPU preparation, Engine/tuning composition, per-run representations, and transactional persistent-resource ownership. |
| 0008 | [Backend partition finalization glossary status reconciliation](tasks/0008-backend-partition-finalization-glossary-status-reconciliation.md) | Ready | 0007; CPU 0010K; Engine 0013; current CPU finalizer/integration evidence | Correct the stale CPU composition and topology status in the shared finalization glossary entry. |

## Milestones and current frontier

- Analysis/declaration, shared assignment/finalization, complete orchestration, and immutable
  partition-local projection are Complete through 0003A.
- Opaque tuning transport, producerless published-constant assignment, and the persistent-resource
  transaction are Complete through 0006.
- Runtime 0016 and Engine 0010 complete the adjacent owner and outward-handle lifecycle. Metal
  0002 and 0003 are Complete; Metal 0004 is the next Draft repository planning frontier.
- Documentation-only 0007 is Complete. Documentation-only 0008 is the sole authorized `Ready`
  frontier and changes no Prepare contract or executable behavior.

## Live risks and gates

- Dynamic dimension binding remains unsupported. Any fact needed for route choice or exact
  resource geometry must be resolved before analysis unless a future explicit contract preserves
  route and assignment stability.
- Keep backend candidates opaque and colocated with their owning backend. Metal 0004 remains
  Draft without a task brief; its gate is completed 0002–0003 plus the opaque Prepare/tuning
  boundary and artifact versioning. It must not move Metal fields, compatibility, or route choice
  into Prepare.
- Preserve analysis-before-assignment-before-finalization and the one-time ownership transaction;
  never let finalization add undeclared shared resources or let Prepare execute physical work.
- Producerless published constants contribute to complete assignment only; backend selection,
  physical geometry, initialization, and materialization stay with their existing owners.

## Status normalization

The task table and linked task status/results are controlling. The former downstream lifecycle-
gate wording is normalized to the current state: Engine 0010, Metal 0002, and Metal 0003 are
Complete; Prepare 0008 is the sole `Ready` documentation frontier. No product task status, order,
dependency, ownership rule, API, or executable behavior changes here.

## History and update policy

Detailed results, validation commands, context identifiers, past ordering exceptions, and
completed frontier narratives remain in linked task files and Git history. Update this map only
when ownership, dependencies, task order/status, a milestone, or a live gate changes. Put
executable scope and evidence in the task brief and follow the
[planning guide](../../planning-guide.md).
