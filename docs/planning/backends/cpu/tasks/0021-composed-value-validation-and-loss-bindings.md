# CPU 0021 — Repair composed value validation and loss bindings

## Status

Review needed

## Goal and class

Class C CPU preparation/execution correctness under existing capability and lifecycle contracts.
Close project-review C2/C3/C4 without widening operation domains or changing shared Runtime.

## Confirmed defects and owned scope

- `CpuSoftmaxInputValidator` sends FLOAT16 through its wide reader: short arrays fail a cast and
  native short storage is read as FLOAT32. Add exact F16 decode and the declared working precision.
- `CpuPreparedExecutable` scans softmax/loss/mask values while all partition children are bound,
  before earlier units produce intermediates. Retain metadata/carrier/span checks in binding;
  move required value validation to invocation before consumer writes, after producers. Inspect
  canonical-BOOL and attention-mask checks too; retain cold-prebound validation mechanics and
  avoid per-element dispatch/allocation. Index/scatter execution validation is an existing seam.
- `CpuPartitionPreparationPlan.unitAccessesAgree` compares loss bindings with rank-zero generated
  cache projections. Use actual occurrence `CpuLossIr.boundaryAccesses` while preserving cache IR.

Owned CPU files are those classes and smallest related bound-call/validator owners plus focused
CPU tests. Add backend-conformance/public CPU Engine integration tests in new focused files when
needed. Do not edit Engine mixed-Metal fixture or Compiler tests owned by parallel repair tasks.
Review CPU softmax guide: its finite-only route is a deliberate limitation, not undefined Model
special-value semantics. No unrelated reference-kernel or activation-emitter change.

## Required reading

AGENTS; root scope index, Core invariants, ownership/dependency rules and Testing; scoped
`backend-execution.md` headings Concrete backend modules, CPU backend routes, Profile-free backend
capability and identity; `runtime-prepare-engine.md` Run lifecycle and modules/runtime; Model
SoftmaxKind/LossKind and mask contracts; affected lowering/IR/binding and focused tests.

## Acceptance and validation

- FLOAT16 SOFTMAX/LOG_SOFTMAX execute valid nonempty heap/native inputs with offsets and strided
  geometry. Provider→prepare→execute tests must not only call the decoder directly.
- `[0,1000].exp().softmax(0)` rejects the produced infinity just as direct `[1,+Inf]` does;
  consumer output remains untouched. Cover computed index-loss targets/masks and repeated runs
  so stale workspace contents cannot stand in for producer values.
- Composed non-scalar MSE and categorical losses prepare/run correctly, including strided
  boundaries; retain rejection of forged incompatible bindings and generated-cache projection.
- Validation occurs once on the orchestrator before worker submission/output mutation, not in
  element loops, with no new hot-path allocation or Runtime policy.
- Focused regressions then one CPU test/Javadoc run; selected conformance/integration tests.
  No historical performance/five-fork matrix or full repository run in worker.
- Review/update affected Javadoc and explanation. Independent Class C review checks source,
  placement, domain preservation and semantic/conformance evidence.
- Commit scoped files, no push/plans; report files/tests/doc impact and completion status.

## Dependencies and integration

- Depends on: CPU 0019/0020 Complete; project-review C2/C3/C4.
- Conflicts with: CPU prepare/executable/validator writers and CPU 0022 implementation.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS (other module owners only).
- Common base revision: production `728680ea` plus planning-only briefs.
- Integration order: 0021 before 0022; otherwise independent of module repairs.
- Integration validation: focused/module evidence, independent review, combined repository
  checkpoint after ordered integration.
- Shared-document integration owner: coordinator; worker owns focused CPU explanation only.

Launch verified 2026-10-13: diagnostic counterexamples and source agree; current contracts suffice;
no CPU writer is active. User-authorized repair, complete predecessors, clean context/worktree.
