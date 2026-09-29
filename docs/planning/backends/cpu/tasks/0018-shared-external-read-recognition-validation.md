# Task 0018: Shared external-read recognition validation

## Status

Blocked

The user authorized this public CPU regression hotfix from clean `e42d938d`; CPU
0008C/0008D/0008F and 0017, and Engine 0018 are Complete. Implementation is partial, not
accepted. Independent Class C review found an unresolved P2: a virtualized recognized baseline
output can claim the relative boundary position of a shared external weight. The red adversarial
test shows `CpuPartitionPreparationPlan` construction accepts that forged topology. Current
CPU-preparation facts do not authenticate a `ValueId` for the virtual output, so the implementer
stopped at the architecture decision gate. No CPU task is `Ready`. CPU 0007A1D remains
independently `Review needed`; its performance gate is not acceptance evidence for this fix.

## Change class

Class C — a private cold-validation error rejects a supported public Engine graph at the
backend/Prepare resource-topology boundary. Preserve fail-closed checks for actual cross-unit
values; use a clean implementation context and an independent targeted review/documentation
context.

## Goal

Allow a CPU partition containing two chained `FLOAT32` `MATMUL` occurrences, optionally with a
`RELU` suffix, to prepare and run when both matmuls read the same caller-owned, read-only weight
`Tensor`. The repeated input remains an external read, not a cross-unit produced value.

## Scope

- Correct only the retained-recognition baseline boundary-role comparison in
  `CpuPartitionPreparationPlan.retainedBaselineMatches`. Its current `occurrences > 1` test
  incorrectly demands `CROSS_UNIT` for a repeated `EXTERNAL_READ`, causing
  `retained recognition baseline IR or resource topology disagrees`.
- Compare the recognition snapshot with the selected compatibility baseline's actual boundary
  roles and producer topology. Continue rejecting forged or mismatched structural keys,
  specialization, dependencies, access/byte geometry, workspace, publication/write roles, and
  genuine produced `CROSS_UNIT` values. Keep route and resource declarations unchanged.
- Add focused CPU regression coverage and public Engine numerical integration coverage for the
  bounded graph and its controls. Add focused backend conformance coverage for the changed
  preparation behavior.

## Non-goals

No bare-`MATMUL` Vector API route bug fix, vector preference/default change, generated-kernel or
OpenBLAS change, new capability, ownership/slot policy, public API, or graph with more than eight
nodes. The separate graph-size limit is not relaxed. Do not use this hotfix to close CPU 0007A1D's
independent performance review.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
  — backend analysis declares exact resources before shared assignment; Runtime receives fixed
  prepared work. [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements)
  route backend and end-to-end regressions to their respective test modules.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  — CPU owns lowering, recognition, route choice, and exact declarations under one CPU identity.
- [Runtime/Prepare/Engine — `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  and [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — shared Prepare does not reinterpret CPU-private facts; public Engine composes the prepared
  result. [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)
  keeps per-run resources isolated.

If an applicable contract proves missing or ambiguous, stop and request clarification rather
than inventing a boundary rule.

## Dependencies and integration

- Depends on: CPU 0008C, 0008D, 0008F, 0017; Engine 0018 (all Complete)
- Conflicts with: concurrent edits to CPU retained recognition/selection validation, CPU
  preparation tests, or the affected CPU guide; CPU 0007A1D review remains independent
- Parallel group: None
- Common base revision: `e42d938d` (verified clean planning base; reverify before execution)
- Integration order: serial hotfix before any overlapping CPU preparation work
- Integration validation: focused CPU, backend-conformance, and public Engine integration tests
  listed below, then affected CPU module tests/Javadoc and Markdown/diff checks once
- Shared-document integration owner: Main planner for CPU master-plan and roadmap status; the
  implementation/review contexts own only this brief and affected CPU guidance

## Files and symbols

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
  — `validateRetainedSpecializedSubgraphs`, `retainedBaselineMatches`, and affected Javadoc.
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
  — selected-baseline shared-read regression and fail-closed role controls.
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuSharedExternalReadConformanceTest.java`
  — focused CPU preparation conformance test.
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineCpuSharedExternalReadIntegrationTest.java`
  — focused public `Engine.standard()` CPU numerical integration test.
- `docs/backend-guide/cpu-backend.md` — review the CPU-private recognition and bounded
  profitability explanation against the corrected behavior.

## Acceptance criteria

- A focused CPU test first reproduces the stated failure on the base revision. After the fix, two
  chained `FLOAT32` `MATMUL`s with the same external weight, both without and with `RELU`, retain
  matching recognition/selected-baseline facts and prepare successfully. Assert the repeated
  weight boundary is `EXTERNAL_READ` and the intermediate produced value is `CROSS_UNIT` where
  represented across units; do not infer its role merely from occurrence count.
- A distinct-weights chain and a single `MATMUL` still prepare. Negative tests keep forged role,
  resource geometry, and other retained-baseline mismatches rejected; true cross-unit values do
  not become external reads. The fix does not bypass recognition validation or change the
  eight-node/fact bounds.
- Public `Engine.standard()` compile/prepare/run succeeds for each shared-weight chain and
  publishes numerically correct outputs from an independent small `FLOAT32` reference (including
  a negative value exercised by `RELU`). Use a small scalar-eligible matrix shape so this test
  neither requires nor masks the separate bare-`MATMUL` vector-route fix. Verify single and
  distinct-weight controls at the public boundary.
- Selected route, resource requirements, generated artifact identity, and execution behavior
  outside this repeated external-read case remain unchanged; no Runtime or Engine production
  edits are needed.

## Validation

Worker validation after adding the focused tests:

```bash
./gradlew :backends:cpu:test --tests '*CpuPartitionPreparerTest'
./gradlew :testing:backend-conformance:test --tests '*CpuSharedExternalReadConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuSharedExternalReadIntegrationTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
git diff --check
```

Integration/repository validation: run the focused conformance and integration tests once on
the integrated diff if they were not already run against that exact code. No repository-wide
suite is required absent a dependency, architecture, shared-build, or additional module change.
Validate Markdown links/anchors, terminology, and new-file whitespace after the final review.

## Follow-up

- Investigate the separate bare-`MATMUL` vector-route bug in its own task; it is not acceptance
  for this scalar-eligible shared-read regression.
- Investigate graphs with more than eight nodes separately; do not widen this task's recognition
  or decomposition budgets. The separately requested CPU graph-size architecture choice remains
  open and does not authorize widening this hotfix.
- User decision required before resuming 0018: add a CPU-private authenticated `ValueId` binding
  for virtualized recognized outputs, or retain the fail-closed invariant through a safe
  nonfused fallback. Either choice needs an updated bounded brief, implementation, negative
  topology tests, and independent Class C review. Do not infer the choice from this plan.

## Documentation and review impact

- This is a planning brief. The implementation must review/update the affected plan-validation
  Javadoc and CPU backend guide, following
  [documentation rules](../../../../developer-guide/documentation-rules.md), and search the glossary
  for relevant existing recognition/boundary terminology. No new term is planned.
- A separate clean targeted review/documentation context is mandatory for this Class C public
  behavior/resource-topology fix. It inspects the final diff and focused evidence, finalizes
  Javadoc/guide and link/terminology checks, and reuses passing executable tests unless code
  changes or a concrete risk requires a rerun.

## Result

- Partial implementation in the current worktree changes
  `CpuPartitionPreparationPlan.java` and `CpuPartitionPreparerTest.java`; it adds
  `CpuSharedExternalReadConformanceTest.java` and
  `EngineCpuSharedExternalReadIntegrationTest.java`. A CPU backend-guide/Javadoc draft is present
  but not finalized or accepted. No Runtime or Engine production edit is part of 0018.
- Clean base: `git rev-parse --short HEAD` reports `e42d938d`; the original validator's
  `occurrences > 1` comparison is visible in that revision. No retained clean-base JUnit report
  or exact initial red-test command is available here, so the clean-base reproduction is not
  recorded as passing validation.
- Retained JUnit XML on the current dirty tree: focused backend conformance has 1 test, 0
  failures; the adversarial `virtualizedBaselineOutputCannotClaimSharedExternalWeightPosition`
  has 1 test, 1 failure (`assertThrows` received no exception); public Engine integration has
  2 tests, 2 failures (`MatchException` in `CpuCarrierEmitter.layoutLocalOffset`). These reports
  do not establish clean-base or final integrated validation. The full CPU module/Javadoc and
  documentation/link gates have no recorded passing result for 0018.
- Independent Class C review identified the P2 above; the affected guide/Javadoc and glossary
  impact need final review after a user-selected fix. The separate >8-node architecture choice
  and CPU 0007A1D performance review remain open. No completion claim or next CPU frontier.

Status: Incomplete
Follow-up required: obtain the user's authenticated-`ValueId`-binding versus safe-nonfused-fallback decision, repair the P2, rerun affected validation, and complete independent Class C review.
