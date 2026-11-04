# Task 0019: [CPU](../../../../glossary.md#term-cpu) [Partition](../../../../glossary.md#partition) Size Independent of [Fusion](../../../../glossary.md#term-fusion)-Unit Size

## Reader context — historical integration boundary

Planning's CPU partition is an ownership region; a CPU computation unit is a smaller execution group
inside that region. A long all-CPU graph need not change backend owner just because a single fused
loop has a size budget. CPU decomposes it into a stable sequence of legal units, shared Prepare
assigns their declared buffers/workspaces, and Runtime invokes the resulting composite recipe.
Removing the partition admission ceiling therefore does not remove per-unit fusion budgets or make
complete-plan enumeration exhaustive for long graphs. Internal split buffers hold produced values
between units; external borrowed inputs remain owned by the component that supplied them.

Current reading: [Partition and computation-unit boundaries](../../../../backend-guide/cpu-backend.md#general-partition-dag-and-bounded-pointwise-fusion); [Complete-plan tuning collaboration](../../../../backend-guide/cpu-backend.md#cold-complete-plan-candidate-collaboration).

## Status

Complete

The user explicitly authorized this separate [graph](../../../../glossary.md#graph)-size correction while [CPU](../../../../glossary.md#term-cpu) 0018 remains
Blocked. CPU 0008B, 0008D, 0008E, 0008F, 0017, and [Engine](../../../../glossary.md#term-engine) 0018 are Complete. CPU 0018 is not a
[dependency](../../../../glossary.md#term-dependency): its partial, uncommitted recognition-[validation](../../../../glossary.md#term-validation) changes must be preserved, not
completed or reverted here. No concurrent CPU 0018 writer may [run](../../../../glossary.md#term-run) during this task.

## Change class

Class C — accepted complete [CPU](../../../../glossary.md#term-cpu) partitions, cold route/plan [identity](../../../../glossary.md#term-identity-meanings), resource declaration, and
public [Engine](../../../../glossary.md#term-engine) execution change. Implementation and independent documentation/review contexts are
required.

## Goal

Accept a complete [CPU](../../../../glossary.md#term-cpu)-owned [partition](../../../../glossary.md#partition) containing more than eight compiled [nodes](../../../../glossary.md#node) without relaxing
the existing eight-node ceiling of any single [fused](../../../../glossary.md#specialization-and-fusion) [pointwise](../../../../glossary.md#term-pointwise) computation unit. A long partition
must execute as a valid sequence of units, not require [Compiler](../../../../glossary.md#term-compiler)/[Planning](../../../../glossary.md#term-planning) to change their maximal
same-[owner](../../../../glossary.md#term-owner) partitioning contract.

## [Scope](../../../../glossary.md#term-scope)

- Remove the eight-node admission check on the complete CPU partition and the eight-unit check on
  its direct selected execution plan. Retain exact node coverage, stable dependencies, distinct
  declarations, checked geometry, workspace ownership, and atomic finalization/rollback.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [admission glossary](../../../../glossary.md#term-admission); [CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Preserve the per-unit eight-node pointwise/affine lowering and structural budgets. A candidate
  exceeding a unit budget remains split; it never invalidates the otherwise supported partition.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [lowering glossary](../../../../glossary.md#lowering))</span>
- For partitions outside the existing bounded complete-candidate domain, use a deterministic,
  supported direct multi-unit preparation path. Its candidate/tuning API must report incomplete
  or unavailable candidates and reject an explicit selected identity; it must not imply exhaustive
  enumeration or enable a representation copy whose fixed IDs collide with later unit workspaces.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Keep bounded local pointwise fusion available throughout a long partition, including near its
  end; do not spend one global attempt budget on unrelated earlier unit pairs and silently cease
  consideration of all later local opportunities. Preserve existing <=8 behavior and identities.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion))</span>
- Cover a long mixed-family partition and an intrinsic workspace owned by a unit at position >=8.
  Include a public or backend-conformance numerical execution test, not preparation alone.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Non-goals

No [Compiler](../../../../glossary.md#term-compiler)/[Planning](../../../../glossary.md#term-planning) [partition](../../../../glossary.md#partition) splitting, new global [graph](../../../../glossary.md#graph) cap, complete-plan [autotuning](../../../../glossary.md#autotuning--model-autotuning) of an
arbitrarily long partition, cross-window [fusion](../../../../glossary.md#term-fusion) guarantee, change to one-unit emitted loops,
[OpenBLAS](../../../../glossary.md#term-openblas) route expansion, [CPU](../../../../glossary.md#term-cpu) 0018 recognition fix, or change to numerical semantics.

## Contracts

- `ARCHITECTURE.md` — Core lifecycle, Core invariants, Module-ownership routing, Testing
  requirements: CPU analysis declares exact resources; Runtime executes a fixed prepared plan.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Module glossary](../../../../glossary.md#term-module); [CPU glossary](../../../../glossary.md#term-cpu); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `docs/architecture/contracts/compiler-autograd.md` — Partition scoring and Compile lifecycle:
  maximal same-owner partitioning stays unchanged.<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition); [Compile glossary](../../../../glossary.md#compile); [owner glossary](../../../../glossary.md#term-owner))</span>
- `docs/architecture/contracts/backend-execution.md` — Concrete backend modules and CPU backend
  routes: CPU privately owns decomposition, fusion, and route choice.<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [backend glossary](../../../../glossary.md#backend); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- `docs/architecture/contracts/runtime-prepare-engine.md` — `modules/prepare`, Prepare lifecycle,
  Run lifecycle: exact declarations precede slots and finalization.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Run glossary](../../../../glossary.md#term-run))</span>

If these contracts prove ambiguous or require a new [ownership](../../../../glossary.md#term-ownership-meanings) rule, stop and report it before
changing architecture.

## Dependencies and integration

- Depends on: [CPU](../../../../glossary.md#term-cpu) 0008B/0008D/0008E/0008F/0017 and [Engine](../../../../glossary.md#term-engine) 0018 (Complete); not CPU 0018.
- Conflicts with: concurrent CPU 0018 work in recognition/preparation/guide and any CPU DAG,
  plan, finalization, or relevant test edits. CPU 0018 remains paused and Blocked.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- Parallel group: None; explicit user-authorized out-of-order CPU frontier.
- Common base revision: N/A — existing dirty worktree is the integration base; preserve all
  unrelated and overlapping uncommitted changes.
- Integration order: 0019 may finish before the independent, Blocked 0018; no concurrent writes.
- Integration validation: focused CPU tests, affected CPU module test/Javadoc, backend conformance
  or integration execution test, documentation link/diff checks. Run a broader checkpoint only for
  a concrete cross-module failure or changed architecture/dependency contract.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Run glossary](../../../../glossary.md#term-run); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Shared-document integration owner: Main planning context for master plan and roadmap; separate
  documentation/review context for affected guide/Javadocs/glossary.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [planning glossary](../../../../glossary.md#term-planning))</span>

## Files and symbols

- `CpuPartitionDagDecomposer` — whole-partition admission and local fusion search; keep the
  eight-node per-unit contraction limit.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [admission glossary](../../../../glossary.md#term-admission); [fusion glossary](../../../../glossary.md#term-fusion); [node glossary](../../../../glossary.md#node); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- `CpuPartitionPreparer`, `CpuPartitionPreparationPlan` — direct long-partition analysis, truthful
  tuning exposure, plan validation and exact unit-indexed workspace declarations.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Focused CPU lowering/preparation/finalization tests and one backend-conformance or Engine
  integration test — >8 nodes, >8 units, late legal fusion, workspace at unit >=8, malformed
  topology failure, and unchanged <=8 behavior.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [integration test glossary](../../../../glossary.md#term-integration-test); [topology glossary](../../../../glossary.md#term-topology))</span>
- `docs/backend-guide/cpu-backend.md` and affected Javadocs — current direct-long behavior and
  explicit complete-candidate/representation-copy boundary.

## Acceptance criteria

- One 9-node and one materially longer CPU-only DAG prepare and execute as one planned partition,
  with correct numerical output and no undeclared or colliding resource requirement.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [prepare glossary](../../../../glossary.md#prepare); [partition glossary](../../../../glossary.md#partition))</span>
- A legal late pointwise pair is considered independently of early unrelated units; every
  selected fused pointwise unit has <=8 compiled nodes.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
- >8-unit plans retain all node ordinals, dependency order, publications, boundary bindings,
  intrinsic workspaces, and per-run isolation. Unsupported nodes still fail before declarations.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [bindings glossary](../../../../glossary.md#term-binding); [run glossary](../../../../glossary.md#term-run))</span>
- Complete-plan candidate enumeration/explicit selection and copy materialization for long
  partitions fail closed or report incompleteness honestly; existing <=8 behavior remains stable.<span data-glossary-navigation="true"> ([Complete-plan candidate glossary](../../../../glossary.md#complete-plan-candidate); [materialization glossary](../../../../glossary.md#materialization); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Existing [CPU](../../../../glossary.md#term-cpu) 0018 uncommitted edits are preserved without extending or claiming its result.

## [Validation](../../../../glossary.md#term-validation)

Worker validation: focused new and adjacent [CPU](../../../../glossary.md#term-cpu) tests, then `./gradlew :backends:cpu:test` once
after code stabilizes; focused [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) or integration execution test. Independent
documentation context checks affected Javadocs, guide terminology/examples/links, glossary
impact, [Javadoc](../../../../glossary.md#term-javadoc) generation, and `git diff --check` without repeating successful Java suites.

## Documentation and review impact

Class C independent targeted review/documentation is mandatory. No architecture-contract change
is expected because [ownership](../../../../glossary.md#term-ownership-meanings) and maximal partitioning remain intact; stop if that assumption
fails. Keep [CPU](../../../../glossary.md#term-cpu) 0018's unrelated guide and [planning](../../../../glossary.md#term-planning) edits intact.

## Result

- CPU decomposition and preparation now accept supported partitions above eight nodes as direct,
  deterministic multi-unit plans; each contracted pointwise/affine unit retains its eight-node
  limit. Long plans keep exact unit-indexed workspaces and publication/resource validation.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [nodes glossary](../../../../glossary.md#node); [pointwise glossary](../../../../glossary.md#term-pointwise); [publication glossary](../../../../glossary.md#term-publication); [validation glossary](../../../../glossary.md#term-validation))</span>
- Complete-plan enumeration, explicit selected-plan input, and representation-copy candidates
  remain unavailable above eight nodes; the candidate API reports an empty, incomplete set.
  The <=8 candidate path and identities are unchanged.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Changed [CPU](../../../../glossary.md#term-cpu) production files: `CpuPartitionDagDecomposer`, `CpuPartitionPreparer`, and
  `CpuPartitionPreparationPlan`. Focused CPU tests and
  `EngineCpuLongPartitionIntegrationTest` cover 9/12/32-node behavior, late fusion, workspace
  position 9, malformed late input, one maximal planned partition, numerical execution, and
  repeated runs.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [partition glossary](../../../../glossary.md#partition))</span>
- Focused Gradle runs passed: decomposer 13, preparer 35, finalizer 17, resource 8, Engine
  integration 2. The full <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [Engine glossary](../../../../glossary.md#term-engine))</span>`:backends:cpu:test` passed with 2 GB test-worker heap, serial forks,
  and <span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>`forkEvery = 30`: 1,039 tests, 28 skipped, zero failures/errors in 6m 36s. Its first
  default-heap attempt failed with `Java heap space`; a four-minute single-[fork](../../../../glossary.md#term-fork-meanings) retry was stopped.
- Independent documentation context finalized the CPU guide, affected Javadocs, and glossary;
  CPU Javadoc generation, guide links/anchors, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` passed. No architecture
  contract, Compiler/Planning code, generated-kernel loop, or backend dependency changed.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [kernel glossary](../../../../glossary.md#kernel); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Existing CPU 0018 and FLOAT16 worktree edits were preserved; CPU 0018 remains Blocked. A
  concurrent autograd-only Compiler edit during full validation does not affect this task's
  forward-only Engine fixture. No commit or push was requested or performed.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [autograd glossary](../../../../glossary.md#autograd); [fixture glossary](../../../../glossary.md#term-fixture); [commit glossary](../../../../glossary.md#term-commit))</span>

Status: Complete
