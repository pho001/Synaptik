# Task 0018: Shared external-read recognition validation

## Status

Complete — the user selected CPU-private authenticated `ValueId` binding on 2026-10-09. At clean
`a756baad`, CPU 0008C/0008D/0008F/0017 and Engine 0018 are Complete; CPU 0019 is independently
Complete. This was the sole serial CPU recognition/preparation frontier, with no overlapping
writer. The virtual-output validation hole is closed without losing legal MATMUL+RELU fusion.
CPU 0007A1D remains independently `Review needed` and supplies no acceptance evidence here.

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

- Complete the retained-recognition boundary-role fix: repeated reads of the same caller-owned
  weight remain `EXTERNAL_READ`, while a produced value used by another unit is `CROSS_UNIT`.
- Bind every recognized compatibility-baseline boundary, including a virtualized MATMUL epilogue
  output, to its actual graph `ValueId` and producer/consumer relation from the trusted projected
  DAG. Validate claimed relative positions against that CPU-private binding before accepting the
  retained selection. A candidate's own role/position labels cannot authenticate themselves.
  Keep `CpuFusionDecision`'s graph-identity-free candidate, tuning, and generated-artifact
  identities unchanged; do not add graph identity to Runtime, shared Prepare, or the hot path.
- Reject forged or mismatched structural keys, specialization, dependencies, access/byte geometry,
  workspace, publication/write roles, and genuine produced `CROSS_UNIT` values. Preserve the
  selected legal MATMUL+RELU epilogue route and exact resource declarations. If a candidate cannot
  be authenticated, reject it; a split candidate may be selected only when independently valid,
  never by swallowing a plan-validation failure.
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

- Depends on: CPU 0008C, 0008D, 0008F, 0017; Engine 0018 (all Complete). CPU 0019 is Complete and
  independent of this task.
- Conflicts with: concurrent edits to CPU retained recognition/selection validation, CPU
  preparation tests, or the affected CPU guide; CPU 0007A1D review remains independent
- Parallel group: None
- Common base revision: `a756baad` on clean `main`; serial work, no parallel group
- Integration order: authenticated CPU-private binding → focused validation → independent Class C
  review → main-context status/documentation integration; no overlapping CPU preparation work
- Integration validation: focused CPU, backend-conformance, and public Engine integration tests
  listed below, then affected CPU module tests/Javadoc and Markdown/diff checks once
- Shared-document integration owner: Main planner for CPU master-plan and roadmap status; the
  implementation/review contexts own only this brief and affected CPU guidance
- Frontier verification (2026-10-09): all declared predecessors are Complete, CPU 0019 is
  independently Complete, no conflicting writer is active, contracts are stable at `a756baad`,
  and the user resolved the only 0018 decision in favor of authenticated `ValueId` binding.

## Files and symbols

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
  — `validateRetainedSpecializedSubgraphs`, `retainedBaselineMatches`, trusted virtual-boundary
  binding, and affected Javadoc. Follow the existing CPU analysis/DAG path only as needed to
  carry or recompute that binding before plan validation.
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
  — selected-baseline shared-read regression and fail-closed role controls.
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuSharedExternalReadConformanceTest.java`
  — focused CPU preparation conformance test.
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineCpuSharedExternalReadIntegrationTest.java`
  — focused public `Engine.standard()` CPU numerical integration test.
- `docs/backend-guide/cpu-backend.md` — review the CPU-private recognition and bounded
  profitability explanation against the corrected behavior.

## Acceptance criteria

- On the current base, the existing shared-weight test and public Engine integration are green,
  while the adversarial virtual-output alias is accepted. Change the adversarial test to require
  rejection. Two chained `FLOAT32` `MATMUL`s with the same external weight, both without and with
  `RELU`, retain
  matching recognition/selected-baseline facts and prepare successfully. Assert the repeated
  weight boundary is `EXTERNAL_READ` and the intermediate produced value is `CROSS_UNIT` where
  represented across units; do not infer its role merely from occurrence count. The legal
  MATMUL+RELU epilogue remains selected in the tested eligible graph.
- A distinct-weights chain and a single `MATMUL` still prepare. Negative tests keep forged role,
  resource geometry, virtual-output/weight position alias, and other retained-baseline mismatches
  rejected; true cross-unit values do not become external reads. The fix does not bypass
  recognition validation or change the eight-node/fact bounds.
- Public `Engine.standard()` compile/prepare/run succeeds for each shared-weight chain and
  publishes numerically correct outputs from an independent small `FLOAT32` reference (including
  a negative value exercised by `RELU`). Use a small scalar-eligible matrix shape so this test
  neither requires nor masks the separate bare-`MATMUL` vector-route fix. Verify single and
  distinct-weight controls at the public boundary.
- Selected route, resource requirements, generated artifact identity, and execution behavior
  outside this repeated external-read case remain unchanged; no Runtime or Engine production
  edits are needed. Authentication runs only during cold CPU preparation, not per execution step
  or tensor element. A proven legal fusion cannot silently inherit a split fallback.

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
- A future independent optimization may measure the exact fused versus split candidate; this
  correctness task makes no unmeasured speedup claim.

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

CPU preparation now captures immutable unfused baseline boundary provenance from checked DAG
ports and binds it to the selected execution-unit objects. Validation rejects a forged
virtual-output/weight position and a binding transplanted from another plan, while retaining the
legal selected MATMUL+RELU epilogue. Existing decision-forgery tests carry the genuine binding and
check their intended failure reasons. No Runtime, Engine production, generated-artifact, or tuning
identity change was needed. The CPU guide and affected Javadocs were reviewed and updated; no
glossary change was needed.

Final focused CPU, backend-conformance, and public Engine integration validation passed 46/46
tests without failures, errors, or skips.
The final full `:backends:cpu:test` run passed 1,045 tests, 0 failures/errors, 28 skipped.
`:backends:cpu:javadoc` passed with 94 existing warnings. Independent Class C re-review approved
the corrected implementation without P0/P1/P2 findings; Markdown links/anchors and
`git diff --check` passed. CPU 0007A1D and the separate bare-MATMUL vector route remain outside
this task.

Status: Complete
