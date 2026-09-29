# Task 0019: CPU Partition Size Independent of Fusion-Unit Size

## Status

Complete

The user explicitly authorized this separate graph-size correction while CPU 0018 remains
Blocked. CPU 0008B, 0008D, 0008E, 0008F, 0017, and Engine 0018 are Complete. CPU 0018 is not a
dependency: its partial, uncommitted recognition-validation changes must be preserved, not
completed or reverted here. No concurrent CPU 0018 writer may run during this task.

## Change class

Class C — accepted complete CPU partitions, cold route/plan identity, resource declaration, and
public Engine execution change. Implementation and independent documentation/review contexts are
required.

## Goal

Accept a complete CPU-owned partition containing more than eight compiled nodes without relaxing
the existing eight-node ceiling of any single fused pointwise computation unit. A long partition
must execute as a valid sequence of units, not require Compiler/Planning to change their maximal
same-owner partitioning contract.

## Scope

- Remove the eight-node admission check on the complete CPU partition and the eight-unit check on
  its direct selected execution plan. Retain exact node coverage, stable dependencies, distinct
  declarations, checked geometry, workspace ownership, and atomic finalization/rollback.
- Preserve the per-unit eight-node pointwise/affine lowering and structural budgets. A candidate
  exceeding a unit budget remains split; it never invalidates the otherwise supported partition.
- For partitions outside the existing bounded complete-candidate domain, use a deterministic,
  supported direct multi-unit preparation path. Its candidate/tuning API must report incomplete
  or unavailable candidates and reject an explicit selected identity; it must not imply exhaustive
  enumeration or enable a representation copy whose fixed IDs collide with later unit workspaces.
- Keep bounded local pointwise fusion available throughout a long partition, including near its
  end; do not spend one global attempt budget on unrelated earlier unit pairs and silently cease
  consideration of all later local opportunities. Preserve existing <=8 behavior and identities.
- Cover a long mixed-family partition and an intrinsic workspace owned by a unit at position >=8.
  Include a public or backend-conformance numerical execution test, not preparation alone.

## Non-goals

No Compiler/Planning partition splitting, new global graph cap, complete-plan autotuning of an
arbitrarily long partition, cross-window fusion guarantee, change to one-unit emitted loops,
OpenBLAS route expansion, CPU 0018 recognition fix, or change to numerical semantics.

## Contracts

- `ARCHITECTURE.md` — Core lifecycle, Core invariants, Module-ownership routing, Testing
  requirements: CPU analysis declares exact resources; Runtime executes a fixed prepared plan.
- `docs/architecture/contracts/compiler-autograd.md` — Partition scoring and Compile lifecycle:
  maximal same-owner partitioning stays unchanged.
- `docs/architecture/contracts/backend-execution.md` — Concrete backend modules and CPU backend
  routes: CPU privately owns decomposition, fusion, and route choice.
- `docs/architecture/contracts/runtime-prepare-engine.md` — `modules/prepare`, Prepare lifecycle,
  Run lifecycle: exact declarations precede slots and finalization.

If these contracts prove ambiguous or require a new ownership rule, stop and report it before
changing architecture.

## Dependencies and integration

- Depends on: CPU 0008B/0008D/0008E/0008F/0017 and Engine 0018 (Complete); not CPU 0018.
- Conflicts with: concurrent CPU 0018 work in recognition/preparation/guide and any CPU DAG,
  plan, finalization, or relevant test edits. CPU 0018 remains paused and Blocked.
- Parallel group: None; explicit user-authorized out-of-order CPU frontier.
- Common base revision: N/A — existing dirty worktree is the integration base; preserve all
  unrelated and overlapping uncommitted changes.
- Integration order: 0019 may finish before the independent, Blocked 0018; no concurrent writes.
- Integration validation: focused CPU tests, affected CPU module test/Javadoc, backend conformance
  or integration execution test, documentation link/diff checks. Run a broader checkpoint only for
  a concrete cross-module failure or changed architecture/dependency contract.
- Shared-document integration owner: Main planning context for master plan and roadmap; separate
  documentation/review context for affected guide/Javadocs/glossary.

## Files and symbols

- `CpuPartitionDagDecomposer` — whole-partition admission and local fusion search; keep the
  eight-node per-unit contraction limit.
- `CpuPartitionPreparer`, `CpuPartitionPreparationPlan` — direct long-partition analysis, truthful
  tuning exposure, plan validation and exact unit-indexed workspace declarations.
- Focused CPU lowering/preparation/finalization tests and one backend-conformance or Engine
  integration test — >8 nodes, >8 units, late legal fusion, workspace at unit >=8, malformed
  topology failure, and unchanged <=8 behavior.
- `docs/backend-guide/cpu-backend.md` and affected Javadocs — current direct-long behavior and
  explicit complete-candidate/representation-copy boundary.

## Acceptance criteria

- One 9-node and one materially longer CPU-only DAG prepare and execute as one planned partition,
  with correct numerical output and no undeclared or colliding resource requirement.
- A legal late pointwise pair is considered independently of early unrelated units; every
  selected fused pointwise unit has <=8 compiled nodes.
- >8-unit plans retain all node ordinals, dependency order, publications, boundary bindings,
  intrinsic workspaces, and per-run isolation. Unsupported nodes still fail before declarations.
- Complete-plan candidate enumeration/explicit selection and copy materialization for long
  partitions fail closed or report incompleteness honestly; existing <=8 behavior remains stable.
- Existing CPU 0018 uncommitted edits are preserved without extending or claiming its result.

## Validation

Worker validation: focused new and adjacent CPU tests, then `./gradlew :backends:cpu:test` once
after code stabilizes; focused backend-conformance or integration execution test. Independent
documentation context checks affected Javadocs, guide terminology/examples/links, glossary
impact, Javadoc generation, and `git diff --check` without repeating successful Java suites.

## Documentation and review impact

Class C independent targeted review/documentation is mandatory. No architecture-contract change
is expected because ownership and maximal partitioning remain intact; stop if that assumption
fails. Keep CPU 0018's unrelated guide and planning edits intact.

## Result

- CPU decomposition and preparation now accept supported partitions above eight nodes as direct,
  deterministic multi-unit plans; each contracted pointwise/affine unit retains its eight-node
  limit. Long plans keep exact unit-indexed workspaces and publication/resource validation.
- Complete-plan enumeration, explicit selected-plan input, and representation-copy candidates
  remain unavailable above eight nodes; the candidate API reports an empty, incomplete set.
  The <=8 candidate path and identities are unchanged.
- Changed CPU production files: `CpuPartitionDagDecomposer`, `CpuPartitionPreparer`, and
  `CpuPartitionPreparationPlan`. Focused CPU tests and
  `EngineCpuLongPartitionIntegrationTest` cover 9/12/32-node behavior, late fusion, workspace
  position 9, malformed late input, one maximal planned partition, numerical execution, and
  repeated runs.
- Focused Gradle runs passed: decomposer 13, preparer 35, finalizer 17, resource 8, Engine
  integration 2. The full `:backends:cpu:test` passed with 2 GB test-worker heap, serial forks,
  and `forkEvery = 30`: 1,039 tests, 28 skipped, zero failures/errors in 6m 36s. Its first
  default-heap attempt failed with `Java heap space`; a four-minute single-fork retry was stopped.
- Independent documentation context finalized the CPU guide, affected Javadocs, and glossary;
  CPU Javadoc generation, guide links/anchors, and `git diff --check` passed. No architecture
  contract, Compiler/Planning code, generated-kernel loop, or backend dependency changed.
- Existing CPU 0018 and FLOAT16 worktree edits were preserved; CPU 0018 remains Blocked. A
  concurrent autograd-only Compiler edit during full validation does not affect this task's
  forward-only Engine fixture. No commit or push was requested or performed.

Status: Complete
