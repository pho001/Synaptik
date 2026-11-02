# Task 0010: General scan projection and resource handoff

## Status

Complete — main accepted 2026-10-22 after independent Class C code/docs APPROVE and the exact compatibility checkpoint.
Historical ImplementationAuthorized=true; user authorization 2026-10-22 covered Prepare0010 ONLY.

## Change class

Class C — scoped compiled-body projection and analysis/resource/finalization boundary.

## Goal

Deliver validated general-scan facts to one partition owner through existing Prepare contracts:
exact compiled body and ordered demand → backend analysis declarations → shared assigned slots
→ backend finalization. Prepare validates the handoff; it does not execute or differentiate bodies.

## Scope

- Extend existing `PrepareContext` validation and `GraphPreparation` projection/demand checks.
  Retain `PartitionDag`'s exact `CompiledNode` references and their `CompiledScanBody` attachments.
- Validate fully static body-local descriptors independently of outer value IDs, ordered formal/
  actual and role/result descriptor associations, and final outer-port demand before analysis.
- Exercise the existing exact resource-declaration/assignment/finalization path with bounded
  test collaborators; preserve ordinary and fixed recurrent-scan behavior.

## Non-goals

No automatic differentiation (AD), operation arithmetic/inference framework, CPU lowering/emitted
loop, backend capability widening, concrete-backend dependency, body flattening/partitioning,
new graph/body representation, runtime tape/session/executor, mutable prepared run state, NN,
padding, persistence, native build or performance campaign. No generic proof framework or fake CPU
loop implementation. Physical history allocation/stores and real scan execution belong to CPU.

## Contracts

Read the [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), then:

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Runtime/Prepare/Engine: [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)
  for immutable prepared versus run-owned resource state and ownership transfer.
- Recurrent scan: [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
  including `Source body, attachments and scoped identities`, `Canonical residual ports and demand`,
  `Traversal, empty domains and declared qualification`, and `Qualification, performance and migration gates`.
- Foundational modules: [Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries).

These contracts own semantics; ambiguity/conflict requires clarification, not architecture/AD expansion.

## Dependencies and integration

- Depends on: Prepare0001–0009 including 0003A; Model0034/0035; Planning0008; Compiler0011 — Complete.
- Conflicts with: scan API/IR/roles/demand, projection/resource/lifecycle, backend scan admission and
  shared scan documentation writers; serialized, none active at launch admission 2026-10-22.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` PLUS all
  accepted uncommitted authority/Model0034/0035, Planning0008 and Compiler0011 work. These are not
  merged predecessors; preserve the complete dirty working tree, including unrelated edits.
- Integration order: accepted upstream → independently approved plan/main acceptance → separately
  authorized clean implementation → independent Class C code/docs approval → main checkpoint below.
- Integration validation: **Prepare0010 scoped-scan handoff compatibility checkpoint**, commands below.
- Shared-document integration owner: main; closure writer owns only the nine authorized planning records.

Historical admission 2026-10-22: Ready at sole authorized frontier, predecessors Complete,
Beauvoir planning APPROVE/main acceptance, matching base, empty index, inactive conflicts;
accepted dirty predecessors preserved, not merged. Execution/review closed; all six frontiers None.
Next CPU F32 pointwise forward + automatic reverse milestone Draft/not briefed/not Ready; no launch.

## Files and symbols

Paths are repository-relative; Java prefix is `io.github.pho001.synaptik.`.
- `modules/prepare/src/main/java/.../prepare/analysis/PrepareContext.java`: canonical and
  compatibility constructors; bounded private validation within this existing owner. Scan-local
  descriptor lookup never shares the outer `values`/logical-requirement map.
- `.../prepare/analysis/PartitionDag.java`: existing constructor, `nodes`, port occurrences and
  adjacency retain outer topology only; attachment transport already works, no new representation.
- `modules/prepare/src/main/java/.../prepare/GraphPreparation.java`: `project`, private `Projection`
  constructor/`context`, `createInvocation`, `validateCrossPartitionDeclarations`. Validate global
  demand from final graph outputs and all outer input uses before projecting any backend context;
  use complete logical requirements to retain external-consumer/publication obligations.
- Existing analysis/resource and finalization/assignment owners retain exact declaration/source/
  geometry checks; no opaque-plan interpretation or new resource categories (criteria 6–7).
- Tests: `modules/prepare/src/test/java/.../prepare/analysis/{PrepareContextTest,PartitionDagTest,
  AnalysisPublicShapeTest}.java` and `.../prepare/{GraphPreparationTest,
  BackendPartitionFinalizationHandoffTest,BackendPartitionFinalizationTest}.java`.
  Existing fixture/input/plan/finalizer/event-order seams only; no new production type.
- Read-only upstream: Model body/graph/header, Planning partition/logical requirements and Compiler
  artifacts/final queries/demand. Compiler inference remains package-private, not Prepare-callable.

## Acceptance criteria

1. Projection retains exact scan node, body and local graph identities, ordered `nextCarry`,
   `emissions`, `residuals`, and `demandedOutputIndices`. No port deduplication, body flattening,
   local nodes in outer DAG adjacency, or body-local values in outer logical resource inventories.
2. Before any analyzer runs, reject dynamic local descriptors, formal/actual dtype/shape/gradient/
   canonical-formal-layout mismatch, sequence time/step mismatch, next-carry mismatch and invalid
   stacked role/output descriptors, including conservative outer gradient flags and T=0.
   Check every local descriptor, including unused formals/siblings, in its own scope; test that
   T=0 skips none of these checks. Valid known PURE Model bodies outside the initial CPU operation/
   dtype whitelist remain projectable (e.g. FLOAT32 RELU and FLOAT16 TANH). Model/Compiler retain
   operation admission/inference/AD, backends retain execution qualification; no second algebra.
3. Missing body, non-scan attachment, malformed header/roles/demand structure fail closed through
   existing Model constructors (test those earliest reachable failures, without bypassing records).
   Structurally legal but descriptor-inconsistent bodies fail at Prepare before collaborators.
4. Demand equals final outer publication/consumer use, not every canonical sibling or internal
   body use. Reject stale missing/extra demanded ports before analysis, including external-partition
   consumers, direct residual publication, forward-to-reverse residual edges and dead reverse
   residuals. Preserve increasing outer-port order and the original attachment; do not repair it.
   Direct `PrepareContext` construction checks locally knowable use/requirement consistency;
   `GraphPreparation` checks exact complete-graph demand, not partition-local sinks alone.
5. Numeric local/outer NodeId and ValueId collisions are legal and cannot alias descriptors, slots
   or ownership. Test deliberately different descriptors at equal numeric IDs, tied outer actuals,
   repeated role roots, unused formals, two carries, and two independent reused-template occurrences.
   Each complete body reaches exactly its selected partition owner once; no mixed-owner interior
   assignment, child-only support shortcut or new partition/executable aggregate.
6. Test collaborators declare explicit outer output/live-residual `Buffer` and analysis-local
   scratch `Workspace` requirements before assignment. Assert exact declaration identity, checked
   geometry, stable slots/representation positions, retained opaque plan and event order at
   finalization. A local-only ID absent from outer values cannot declare an outer buffer.
   A coincident ID alone cannot authenticate local storage; workspaces use existing local IDs.
7. Reject undeclared/extra assignments, changed declaration identity, insufficient buffer geometry
   and mismatched workspace geometry through existing `BackendPartitionFinalization` contracts.
   Preserve valid larger shared buffers: assignment aggregates buffer size/alignment maxima and
   finalization accepts geometry >= each declaration; workspace geometry must equal its declaration.
   Foreign memory-plan results reject through the handoff. Missing required cross-partition
   residual declarations reject before finalization.
   Do not claim shared Prepare can detect hidden allocations inside an opaque backend executable.
8. T=0 retains static empty stacked descriptors, exact role/demand metadata and legal zero-byte
   declarations, without executing a body. Test that T=0 bypasses no scoped validation, including
   unused formals/siblings, and preserves valid outside-CPU-whitelist projection from criterion 2.
   Metadata/resource tests do not prove actual no-history
   bytes/stores, exact numerical final carry, or CPU execution; those remain backend gates.
   Existing ordinary/fixed constructors, projection, assignment and ownership/rollback tests pass.

## Validation

Accepted worker commands (2026-10-22):

```bash
./gradlew :modules:prepare:test --tests '*PrepareContextTest' --tests '*PartitionDagTest' --tests '*AnalysisPublicShapeTest' --tests '*GraphPreparationTest' --tests '*BackendPartitionFinalizationHandoffTest' --tests '*BackendPartitionFinalizationTest'
./gradlew :modules:prepare:test :modules:prepare:javadoc
```

Focused: PASS, 61 tests/6 suites; final module: PASS, 79 tests/12 suites; both zero failures,
errors or skips. Final Javadoc: zero warnings. Main verified fresh XML timestamp 09:29:37Z.
All eight Java files remained byte-identical to the tested versions through final review.
The corrected intermediate fixture compilation issue is resolved, not an open bug.

Main's exact accepted compatibility checkpoint (2026-10-22):

```bash
./gradlew :backends:cpu:compileTestJava :backends:metal:compileTestJava :modules:engine:compileTestJava
./gradlew :testing:architecture-tests:test
./gradlew :testing:integration-tests:test --tests '*EngineCpuRecurrentScanIntegrationTest'
git diff --check
```

Compilation PASS, 489 ms, 18 actionable tasks all UP-TO-DATE: valid compilation evidence, not a
fresh recompile. Architecture PASS, 454 ms, 27 tasks all UP-TO-DATE: reused 25 tests/11 suites,
zero failures/errors/skips, not fresh execution; source-checking test inputs unchanged, no edges changed.
Integration PASS, 3 s, 42 tasks (3 executed/39 UP-TO-DATE): fresh 5 tests/1 suite, zero failures/
errors/skips. Combined `git diff --check` PASS. Existing FFM native-access/incubator-vector warnings
are not Javadoc warnings and require no scope fix. No root/native/performance campaign here;
real scan conformance and the single native-enabled root `test javadoc` remain at the low-level join.

## Documentation and review impact

Affected validation/demand/failure-mode Javadocs and package Javadoc reviewed/updated, including
parameters/results/exceptions; final Javadoc follows edits. Huygens independently approved exact
Class C code/docs/contracts/tests/API compatibility; successful executable evidence reused.
Main owns shared explanations/planning. Twelve factual status/boundary updates, no architecture/
ADR semantic change or new terminology; targeted glossary review covers general scan, formal/primal
binding, residual/history demand, partition DAG and backend finalization. See Result for validation.

## Result

- Delivered exact scoped-body/role projection, static descriptor and final outer-demand validation,
  fail-closed pre-analysis rejection and existing resource declaration/assignment/finalization handoff.
- Changed existing production `modules/prepare/src/main/java/.../prepare/GraphPreparation.java`,
  `analysis/PrepareContext.java` and `analysis/package-info.java` (Javadoc).
- Changed five existing tests under `modules/prepare/src/test/java/.../prepare/`:
  `GraphPreparationTest`, `BackendPartitionFinalizationHandoffTest`, `BackendPartitionFinalizationTest`,
  `analysis/PrepareContextTest` and `analysis/PartitionDagTest`; eight Java files, no added types/files.
- Worker 61 focused/79 module tests + zero-warning Javadoc PASS; exact commands and fresh/reused
  main compilation, architecture and fixed-scan integration evidence are above.
- Huygens final Class C code/docs APPROVE, zero P1/P2; Java unchanged; main checkpoint accepted 2026-10-22.
- Documentation: 12 factual status/boundary updates across `ARCHITECTURE.md`, scoped contracts,
  `docs/architecture/`, `docs/backend-guide/partition-preparer.md` and `docs/glossary.md`;
  no authority semantic change. Affected Javadocs reviewed/updated; no new glossary terminology.
- Independent documentation validation PASS: 512 local Markdown links/anchors, six glossary
  headings and whitespace. This closure synchronizes only the nine authorized planning records.
- No real backend execution/history-allocation proof, F64 domain completion, NN or persistence grant.
  Next bounded CPU F32 milestone remains Draft/not briefed/not Ready; no automatic successor launch.
- Unresolved issues: none. Required follow-up for this task: none.

Status: Complete
