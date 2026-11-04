# Task 0018: Shared external-read recognition [validation](../../../../glossary.md#term-validation)

## Reader context — historical integration boundary

‘Caller-owned weight Tensor’ can originate in application code or in an NN/training component acting
as Engine's caller. It is a leaf input to this compiled graph, with its current values held in
external storage; the expression/descriptor alone is not that storage. If two MATMUL units read the
same weight, neither produced it, so the repeated boundary is still an external read. A value
produced by an earlier unit is instead a cross-unit dependency. CPU authenticates this distinction
from the trusted projected node/port graph during preparation, not from a candidate's self-declared
labels. Borrowing repeated reads does not transfer storage cleanup to CPU.

Current reading: [Retained subgraph recognition](../../../../backend-guide/cpu-backend.md#cpu-private-specialized-subgraph-recognition); [Partition and computation-unit boundaries](../../../../backend-guide/cpu-backend.md#general-partition-dag-and-bounded-pointwise-fusion).

## Status

Complete — the user selected [CPU](../../../../glossary.md#term-cpu)-private authenticated [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) [binding](../../../../glossary.md#term-binding) on 2026-10-09. At clean
`a756baad`, CPU 0008C/0008D/0008F/0017 and [Engine](../../../../glossary.md#term-engine) 0018 are Complete; CPU 0019 is independently
Complete. This was the sole serial CPU recognition/preparation frontier, with no overlapping
writer. The virtual-output [validation](../../../../glossary.md#term-validation) hole is closed without losing legal [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)+RELU [fusion](../../../../glossary.md#term-fusion).
CPU 0007A1D remains independently `Review needed` and supplies no acceptance evidence here.

## Change class

Class C — a private cold-[validation](../../../../glossary.md#term-validation) error rejects a supported public [Engine](../../../../glossary.md#term-engine) [graph](../../../../glossary.md#graph) at the
[backend](../../../../glossary.md#backend)/[Prepare](../../../../glossary.md#prepare) resource-[topology](../../../../glossary.md#term-topology) boundary. Preserve [fail-closed](../../../../glossary.md#term-fail-closed) checks for actual cross-unit
values; use a clean implementation context and an independent targeted review/documentation
context.

## Goal

Allow a [CPU](../../../../glossary.md#term-cpu) [partition](../../../../glossary.md#partition) containing two chained [`FLOAT32`](../../../../glossary.md#term-float32) [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) occurrences, optionally with a
`RELU` suffix, to [prepare](../../../../glossary.md#prepare) and [run](../../../../glossary.md#term-run) when both matmuls read the same caller-owned, read-only [weight](../../../../glossary.md#term-weight)
[`Tensor`](../../../../glossary.md#tensor). The repeated input remains an external read, not a cross-unit produced value.

## [Scope](../../../../glossary.md#term-scope)

- Complete the retained-recognition boundary-role fix: repeated reads of the same caller-owned
  weight remain <span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight))</span>`EXTERNAL_READ`, while a produced value used by another unit is `CROSS_UNIT`.
- Bind every recognized compatibility-baseline boundary, including a virtualized MATMUL epilogue
  output, to its actual graph <span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [epilogue glossary](../../../../glossary.md#term-epilogue); [graph glossary](../../../../glossary.md#graph))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) and producer/consumer relation from the trusted projected
  DAG. Validate claimed relative positions against that CPU-private binding before accepting the
  retained selection. A candidate's own role/position labels cannot authenticate themselves.
  Keep <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [CPU glossary](../../../../glossary.md#term-cpu); [binding glossary](../../../../glossary.md#term-binding))</span>`CpuFusionDecision`'s graph-identity-free candidate, tuning, and generated-artifact
  identities unchanged; do not add graph identity to Runtime, shared Prepare, or the hot path.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [artifact glossary](../../../../glossary.md#term-artifact); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- Reject forged or mismatched structural keys, specialization, dependencies, access/byte geometry,
  workspace, publication/write roles, and genuine produced <span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [publication glossary](../../../../glossary.md#term-publication))</span>`CROSS_UNIT` values. Preserve the
  selected legal MATMUL+RELU epilogue route and exact resource declarations. If a candidate cannot
  be authenticated, reject it; a split candidate may be selected only when independently valid,
  never by swallowing a plan-validation failure.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Add focused CPU regression coverage and public Engine numerical integration coverage for the
  bounded graph and its controls. Add focused backend conformance coverage for the changed
  preparation behavior.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Non-goals

No bare-[`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) [Vector API](../../../../glossary.md#term-vector-api) route bug fix, [vector](../../../../glossary.md#term-vector) preference/default change, generated-[kernel](../../../../glossary.md#kernel) or
[OpenBLAS](../../../../glossary.md#term-openblas) change, new capability, [ownership](../../../../glossary.md#term-ownership-meanings)/slot policy, public [API](../../../../glossary.md#term-application-programming-interface-api), or [graph](../../../../glossary.md#graph) with more than eight
[nodes](../../../../glossary.md#node). The separate graph-size limit is not relaxed. Do not use this hotfix to close [CPU](../../../../glossary.md#term-cpu) 0007A1D's
independent performance review.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>
  — backend analysis declares exact resources before shared assignment; Runtime receives fixed
  prepared work. <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Runtime glossary](../../../../glossary.md#term-runtime))</span>[Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements)
  route backend and end-to-end regressions to their respective test modules.<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
  — CPU owns lowering, recognition, route choice, and exact declarations under one CPU identity.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [Runtime/Prepare/Engine — `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>,
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  and [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — shared Prepare does not reinterpret CPU-private facts; public Engine composes the prepared
  result. [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>
  keeps per-run resources isolated.

If an applicable contract proves missing or ambiguous, stop and request clarification rather
than inventing a boundary rule.

## Dependencies and integration

- Depends on: CPU 0008C, 0008D, 0008F, 0017; Engine 0018 (all Complete). CPU 0019 is Complete and
  independent of this task.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Conflicts with: concurrent edits to CPU retained recognition/selection validation, CPU
  preparation tests, or the affected CPU guide; CPU 0007A1D review remains independent<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Parallel group: None
- Common base revision: `a756baad` on clean `main`; serial work, no parallel group
- Integration order: authenticated CPU-private binding → focused validation → independent Class C
  review → main-context status/documentation integration; no overlapping CPU preparation work
- Integration validation: focused CPU, backend-conformance, and public Engine integration tests
  listed below, then affected CPU module tests/Javadoc and Markdown/diff checks once<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration owner: Main planner for CPU master-plan and roadmap status; the
  implementation/review contexts own only this brief and affected CPU guidance<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Frontier verification (2026-10-09): all declared predecessors are Complete, CPU 0019 is
  independently Complete, no conflicting writer is active, contracts are stable at `a756baad`,
  and the user resolved the only 0018 decision in favor of authenticated [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) [binding](../../../../glossary.md#term-binding).

## Files and symbols

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
  — `validateRetainedSpecializedSubgraphs`, `retainedBaselineMatches`, trusted virtual-boundary
  binding, and affected Javadoc. Follow the existing CPU analysis/DAG path only as needed to
  carry or recompute that binding before plan validation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Javadoc glossary](../../../../glossary.md#term-javadoc); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [carry glossary](../../../../glossary.md#term-carry); [validation glossary](../../../../glossary.md#term-validation))</span>
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
  — selected-baseline shared-read regression and fail-closed role controls.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuSharedExternalReadConformanceTest.java`
  — focused CPU preparation conformance test.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineCpuSharedExternalReadIntegrationTest.java`
  — focused public `Engine.standard()` [CPU](../../../../glossary.md#term-cpu) numerical [integration test](../../../../glossary.md#term-integration-test).
- `docs/backend-guide/cpu-backend.md` — review the CPU-private recognition and bounded
  profitability explanation against the corrected behavior.

## Acceptance criteria

- On the current base, the existing shared-weight test and public Engine integration are green,
  while the adversarial virtual-output alias is accepted. Change the adversarial test to require
  rejection. Two chained <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>[`FLOAT32`](../../../../glossary.md#term-float32) [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul)s with the same external [weight](../../../../glossary.md#term-weight), both without and with
  `RELU`, retain
  matching recognition/selected-baseline facts and prepare successfully. Assert the repeated
  weight boundary is <span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>`EXTERNAL_READ` and the intermediate produced value is `CROSS_UNIT` where
  represented across units; do not infer its role merely from occurrence count. The legal
  MATMUL+RELU epilogue remains selected in the tested eligible graph.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [epilogue glossary](../../../../glossary.md#term-epilogue); [graph glossary](../../../../glossary.md#graph))</span>
- A distinct-weights chain and a single `MATMUL` still prepare. Negative tests keep forged role,
  resource geometry, virtual-output/weight position alias, and other retained-baseline mismatches
  rejected; true cross-unit values do not become external reads. The fix does not bypass
  recognition validation or change the eight-node/fact bounds.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [node glossary](../../../../glossary.md#node))</span>
- Public `Engine.standard()` compile/prepare/run succeeds for each shared-weight chain and
  publishes numerically correct outputs from an independent small <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run))</span>`FLOAT32` reference (including
  a negative value exercised by `RELU`). Use a small scalar-eligible matrix shape so this test
  neither requires nor masks the separate bare-<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [shape glossary](../../../../glossary.md#shape); [masks glossary](../../../../glossary.md#term-boolean-mask))</span>`MATMUL` vector-route fix. Verify single and
  distinct-weight controls at the public boundary.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>
- Selected route, resource requirements, generated artifact identity, and execution behavior
  outside this repeated external-read case remain unchanged; no Runtime or Engine production
  edits are needed. Authentication runs only during cold CPU preparation, not per execution step
  or tensor element. A proven legal fusion cannot silently inherit a split fallback.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [Runtime glossary](../../../../glossary.md#term-runtime); [Authentication glossary](../../../../glossary.md#term-authentication); [CPU glossary](../../../../glossary.md#term-cpu); [tensor glossary](../../../../glossary.md#tensor); [element glossary](../../../../glossary.md#term-element); [fusion glossary](../../../../glossary.md#term-fusion))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation after adding the focused tests:

```bash
./gradlew :backends:cpu:test --tests '*CpuPartitionPreparerTest'
./gradlew :testing:backend-conformance:test --tests '*CpuSharedExternalReadConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuSharedExternalReadIntegrationTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
git diff --check
```

Integration/repository validation: [run](../../../../glossary.md#term-run) the focused [conformance](../../../../glossary.md#term-conformance) and integration tests once on
the integrated diff if they were not already run against that exact code. No repository-wide
suite is required absent a [dependency](../../../../glossary.md#term-dependency), architecture, shared-build, or additional [module](../../../../glossary.md#term-module) change.
Validate Markdown links/anchors, terminology, and new-file whitespace after the final review.

## Follow-up

- Investigate the separate bare-[`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) vector-route bug in its own task; it is not acceptance
  for this scalar-eligible shared-read regression.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [scalar glossary](../../../../glossary.md#term-scalar); [regression glossary](../../../../glossary.md#term-regression))</span>
- Investigate graphs with more than eight nodes separately; do not widen this task's recognition
  or decomposition budgets. The separately requested CPU graph-size architecture choice remains
  open and does not authorize widening this hotfix.<span data-glossary-navigation="true"> ([graphs glossary](../../../../glossary.md#graph); [nodes glossary](../../../../glossary.md#node); [decomposition glossary](../../../../glossary.md#term-decomposition); [CPU glossary](../../../../glossary.md#term-cpu); [widening glossary](../../../../glossary.md#term-widening))</span>
- A future independent optimization may measure the exact fused versus split candidate; this
  correctness task makes no unmeasured speedup claim.<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion))</span>

## Documentation and review impact

- This is a planning brief. The implementation must review/update the affected plan-validation
  Javadoc and CPU backend guide, following<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
  [documentation rules](../../../../developer-guide/documentation-rules.md), and search the glossary
  for relevant existing recognition/boundary terminology. No new term is planned.
- A separate clean targeted review/documentation context is mandatory for this Class C public
  behavior/resource-topology fix. It inspects the final diff and focused evidence, finalizes
  Javadoc/guide and link/terminology checks, and reuses passing executable tests unless code
  changes or a concrete risk requires a rerun.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>

## Result

[CPU](../../../../glossary.md#term-cpu) preparation now [captures](../../../../glossary.md#term-capture-meanings) [immutable](../../../../glossary.md#term-immutable) unfused baseline boundary [provenance](../../../../glossary.md#provenance) from checked [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag)
ports and binds it to the selected execution-unit objects. [Validation](../../../../glossary.md#term-validation) rejects a forged
virtual-output/[weight](../../../../glossary.md#term-weight) position and a [binding](../../../../glossary.md#term-binding) transplanted from another plan, while retaining the
legal selected [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)+RELU [epilogue](../../../../glossary.md#term-epilogue). Existing decision-forgery tests [carry](../../../../glossary.md#term-carry) the genuine binding and
check their intended failure reasons. No [Runtime](../../../../glossary.md#term-runtime), [Engine](../../../../glossary.md#term-engine) production, generated-[artifact](../../../../glossary.md#term-artifact), or tuning
[identity](../../../../glossary.md#term-identity-meanings) change was needed. The CPU guide and affected Javadocs were reviewed and updated; no
glossary change was needed.

Final focused CPU, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and public Engine integration validation passed 46/46
tests without failures, errors, or skips.
The final full `:backends:cpu:test` [run](../../../../glossary.md#term-run) passed 1,045 tests, 0 failures/errors, 28 skipped.
`:backends:cpu:javadoc` passed with 94 existing warnings. Independent Class C re-review approved
the corrected implementation without P0/P1/P2 findings; Markdown links/anchors and
`git diff --check` passed. CPU 0007A1D and the separate bare-MATMUL [vector](../../../../glossary.md#term-vector) route remain outside
this task.

Status: Complete
