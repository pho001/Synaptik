# Task 0010: [General scan](../../../../glossary.md#general-differentiable-scan) projection and resource [handoff](../../../../glossary.md#term-handoff)

> **Editorial historical reading key.**
>
> This completed historical task concerns a metadata and resource handoff, not a Scan executor.
> Compiler supplies the exact scoped body, role bindings and final demand; GraphPreparation checks
> complete-graph obligations before sending each partition owner a PrepareContext. Local IDs remain
> local even when their numbers equal outer IDs. Model/Compiler own operation semantics and
> differentiation; backend qualification is separate from Prepare acceptance.
>
> Backend analyzers declare outer buffers and local workspaces; shared Prepare assigns slots, and
> backend finalizers receive the exact declarations and shared plan. Under the [current finalization
> protocol](../../../../reference/runtime/orchestration.md#current-prepare-orchestration-assignment-and-backend-finalization),
> backend failure rollback precedes Prepare rollback responsibility, then successful ownership
> transfers to Runtime. The [current capability
> contract](../../../../reference/compiler/planning.md#current-operation-capability-contracts)
> separately requires whole-body/binder support. Historical metadata tests, including T=0, do not
> establish physical history allocation or numerical execution.

## Status

Complete — main accepted 2026-10-22 after independent Class C code/docs APPROVE and the exact compatibility [checkpoint](../../../../glossary.md#term-checkpoint).
Historical ImplementationAuthorized=true; user authorization 2026-10-22 covered Prepare0010 ONLY.

## Change class

Class C — scoped compiled-body projection and analysis/resource/finalization boundary.

## Goal

Deliver validated general-[scan](../../../../glossary.md#term-cumulative-scan) facts to one [partition](../../../../glossary.md#partition) [owner](../../../../glossary.md#term-owner) through existing [Prepare](../../../../glossary.md#prepare) contracts:
exact compiled body and ordered demand → [backend](../../../../glossary.md#backend) analysis declarations → shared assigned slots
→ backend finalization. Prepare validates the [handoff](../../../../glossary.md#term-handoff); it does not execute or differentiate bodies.

## [Scope](../../../../glossary.md#term-scope)

- Extend existing [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) [validation](../../../../glossary.md#term-validation) and [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) projection/demand checks.
  Retain [`PartitionDag`](../../../../glossary.md#partition-dag-partitiondag)'s exact `CompiledNode` references and their `CompiledScanBody` attachments.
- Validate fully static body-local descriptors independently of outer value IDs, ordered formal/
  actual and role/result descriptor associations, and final outer-port demand before analysis.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Exercise the existing exact resource-declaration/assignment/finalization path with bounded
  test collaborators; preserve ordinary and fixed recurrent-scan behavior.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>

## Non-goals

No [automatic differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)), [operation](../../../../glossary.md#operation) arithmetic/[inference](../../../../glossary.md#term-inference) framework, [CPU](../../../../glossary.md#term-cpu) [lowering](../../../../glossary.md#lowering)/emitted
loop, [backend capability](../../../../glossary.md#backend-capability) [widening](../../../../glossary.md#term-widening), concrete-[backend](../../../../glossary.md#backend) [dependency](../../../../glossary.md#term-dependency), body flattening/partitioning,
new [graph](../../../../glossary.md#graph)/body representation, [runtime](../../../../glossary.md#term-runtime) tape/session/executor, [mutable](../../../../glossary.md#term-mutable) prepared [run state](../../../../glossary.md#run-state--runstate), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context),
[padding](../../../../glossary.md#padding), [persistence](../../../../glossary.md#term-persistence), native build or performance campaign. No generic proof framework or fake CPU
loop implementation. Physical history allocation/stores and real [scan](../../../../glossary.md#term-cumulative-scan) execution belong to CPU.

## Contracts

Read the [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>, then:

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span> and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine): [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>
  for immutable prepared versus run-owned resource state and ownership transfer.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Recurrent [scan](../../../../glossary.md#term-cumulative-scan): [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
  including `Source body, attachments and scoped identities`, `Canonical residual ports and demand`,
  `Traversal, empty domains and declared qualification`, and `Qualification, performance and migration gates`.
- Foundational [modules](../../../../glossary.md#term-module): [Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries).

These contracts own semantics; ambiguity/conflict requires clarification, not architecture/[AD](../../../../glossary.md#autograd) expansion.

## Dependencies and integration

- Depends on: Prepare0001–0009 including 0003A; Model0034/0035; Planning0008; Compiler0011 — Complete.
- Conflicts with: scan API/IR/roles/demand, projection/resource/lifecycle, backend scan admission and
  shared scan documentation writers; serialized, none active at launch admission 2026-10-22.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lifecycle glossary](../../../../glossary.md#lifecycle); [backend glossary](../../../../glossary.md#backend))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` PLUS all
  accepted uncommitted authority/Model0034/0035, Planning0008 and Compiler0011 work. These are not
  merged predecessors; preserve the complete dirty working tree, including unrelated edits.
- Integration order: accepted upstream → independently approved plan/main acceptance → separately
  authorized clean implementation → independent Class C code/docs approval → main checkpoint below.
- Integration [validation](../../../../glossary.md#term-validation): **Prepare0010 scoped-[scan](../../../../glossary.md#term-cumulative-scan) [handoff](../../../../glossary.md#term-handoff) compatibility [checkpoint](../../../../glossary.md#term-checkpoint)**, commands below.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main; closure writer owns only the nine authorized [planning](../../../../glossary.md#term-planning) records.

Historical [admission](../../../../glossary.md#term-admission) 2026-10-22: [Ready](../../../../glossary.md#term-ready) at sole [authorized frontier](../../../../glossary.md#term-authorized-frontier), predecessors Complete,
Beauvoir planning APPROVE/main acceptance, matching base, empty index, inactive conflicts;
accepted dirty predecessors preserved, not merged. Execution/review closed; all six frontiers None.
Next [CPU](../../../../glossary.md#term-cpu) [F32](../../../../glossary.md#floating-point-precision-and-rounding) [pointwise](../../../../glossary.md#term-pointwise) forward + automatic reverse milestone Draft/not briefed/not Ready; no launch.

## Files and symbols

Paths are repository-relative; Java prefix is `io.github.pho001.synaptik.`.
- `modules/prepare/src/main/java/.../prepare/analysis/PrepareContext.java`: canonical and
  compatibility constructors; bounded private validation within this existing owner. Scan-local
  descriptor lookup never shares the outer <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation); [owner glossary](../../../../glossary.md#term-owner); [Scan glossary](../../../../glossary.md#term-cumulative-scan); [descriptor glossary](../../../../glossary.md#term-descriptor); [lookup glossary](../../../../glossary.md#term-lookup))</span>`values`/logical-requirement map.
- `.../prepare/analysis/PartitionDag.java`: existing constructor, [`nodes`](../../../../glossary.md#node), port occurrences and
  adjacency retain outer topology only; attachment transport already works, no new representation.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>
- `modules/prepare/src/main/java/.../prepare/GraphPreparation.java`: `project`, private `Projection`
  constructor/`context`, `createInvocation`, `validateCrossPartitionDeclarations`. Validate global
  demand from final graph outputs and all outer input uses before projecting any backend context;
  use complete logical requirements to retain external-consumer/publication obligations.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [publication glossary](../../../../glossary.md#term-publication))</span>
- Existing analysis/resource and finalization/assignment owners retain exact declaration/source/
  geometry checks; no opaque-plan interpretation or new resource categories (criteria 6–7).<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- Tests: `modules/prepare/src/test/java/.../prepare/analysis/{PrepareContextTest,PartitionDagTest,
  AnalysisPublicShapeTest}.java` and `.../prepare/{GraphPreparationTest,
  BackendPartitionFinalizationHandoffTest,BackendPartitionFinalizationTest}.java`.
  Existing fixture/input/plan/finalizer/event-order seams only; no new production type.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>
- Read-only upstream: Model body/graph/header, Planning partition/logical requirements and Compiler
  artifacts/final queries/demand. Compiler inference remains package-private, not Prepare-callable.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [partition glossary](../../../../glossary.md#partition); [Compiler glossary](../../../../glossary.md#term-compiler); [artifacts glossary](../../../../glossary.md#term-artifact); [inference glossary](../../../../glossary.md#term-inference); [Prepare glossary](../../../../glossary.md#prepare))</span>

## Acceptance criteria

1. Projection retains exact [scan](../../../../glossary.md#term-cumulative-scan) [node](../../../../glossary.md#node), body and local [graph](../../../../glossary.md#graph) [identities](../../../../glossary.md#term-identity-meanings), ordered `nextCarry`,
   `emissions`, `residuals`, and `demandedOutputIndices`. No port deduplication, body flattening,
   local nodes in outer DAG adjacency, or body-local values in outer logical resource inventories.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
2. Before any analyzer runs, reject dynamic local descriptors, formal/actual dtype/shape/gradient/
   canonical-formal-layout mismatch, sequence time/step mismatch, next-carry mismatch and invalid
   stacked role/output descriptors, including conservative outer gradient flags and T=0.
   Check every local descriptor, including unused formals/siblings, in its own scope; test that
   T=0 skips none of these checks. Valid known PURE Model bodies outside the initial CPU operation/
   dtype whitelist remain projectable (e.g. FLOAT32 RELU and FLOAT16 TANH). Model/Compiler retain
   operation admission/inference/AD, backends retain execution qualification; no second algebra.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [dtype glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [canonical glossary](../../../../glossary.md#term-canonical-form); [layout glossary](../../../../glossary.md#layout); [carry glossary](../../../../glossary.md#term-carry); [scope glossary](../../../../glossary.md#term-scope); [PURE glossary](../../../../glossary.md#term-pure); [Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu); [operation glossary](../../../../glossary.md#operation); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT16 glossary](../../../../glossary.md#term-float16); [TANH glossary](../../../../glossary.md#term-tanh); [Compiler glossary](../../../../glossary.md#term-compiler); [admission glossary](../../../../glossary.md#term-admission); [inference glossary](../../../../glossary.md#term-inference); [AD glossary](../../../../glossary.md#autograd); [backends glossary](../../../../glossary.md#backend); [qualification glossary](../../../../glossary.md#term-qualification))</span>
3. Missing body, non-scan attachment, malformed header/roles/demand structure fail closed through
   existing Model constructors (test those earliest reachable failures, without bypassing records).
   Structurally legal but descriptor-inconsistent bodies fail at Prepare before collaborators.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [Prepare glossary](../../../../glossary.md#prepare))</span>
4. Demand equals final outer publication/consumer use, not every canonical sibling or internal
   body use. Reject stale missing/extra demanded ports before analysis, including external-partition
   consumers, direct residual publication, forward-to-reverse residual edges and dead reverse
   residuals. Preserve increasing outer-port order and the original attachment; do not repair it.
   Direct <span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) construction checks locally knowable use/requirement consistency;
   [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) checks exact complete-graph demand, not [partition](../../../../glossary.md#partition)-local sinks alone.
5. Numeric local/outer NodeId and ValueId collisions are legal and cannot alias descriptors, slots
   or ownership. Test deliberately different descriptors at equal numeric IDs, tied outer actuals,
   repeated role roots, unused formals, two carries, and two independent reused-template occurrences.
   Each complete body reaches exactly its selected partition owner once; no mixed-owner interior
   assignment, child-only support shortcut or new partition/executable aggregate.<span data-glossary-navigation="true"> ([NodeId glossary](../../../../glossary.md#nodeid); [ValueId glossary](../../../../glossary.md#logical-valueid-versus-physical-memory-slot); [alias glossary](../../../../glossary.md#aliasing-and-arena); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [owner glossary](../../../../glossary.md#term-owner))</span>
6. Test collaborators declare explicit outer output/live-[residual](../../../../glossary.md#scan-residual-port-and-history-demand) [`Buffer`](../../../../glossary.md#term-buffer) and analysis-local
   scratch `Workspace` requirements before assignment. Assert exact declaration identity, checked
   geometry, stable slots/representation positions, retained opaque plan and event order at
   finalization. A local-only ID absent from outer values cannot declare an outer buffer.
   A coincident ID alone cannot authenticate local storage; workspaces use existing local IDs.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [opaque glossary](../../../../glossary.md#term-opaque))</span>
7. Reject undeclared/extra assignments, changed declaration identity, insufficient buffer geometry
   and mismatched workspace geometry through existing `BackendPartitionFinalization` contracts.
   Preserve valid larger shared buffers: assignment aggregates buffer size/alignment maxima and
   finalization accepts geometry >= each declaration; workspace geometry must equal its declaration.
   Foreign memory-plan results reject through the handoff. Missing required cross-partition
   residual declarations reject before finalization.
   Do not claim shared Prepare can detect hidden allocations inside an opaque backend executable.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [handoff glossary](../../../../glossary.md#term-handoff))</span>
8. T=0 retains static empty stacked descriptors, exact role/demand metadata and legal zero-byte
   declarations, without executing a body. Test that T=0 bypasses no scoped validation, including
   unused formals/siblings, and preserves valid outside-CPU-whitelist projection from criterion 2.
   Metadata/resource tests do not prove actual no-history
   bytes/stores, exact numerical final carry, or CPU execution; those remain backend gates.
   Existing ordinary/fixed constructors, projection, assignment and ownership/rollback tests pass.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation); [rollback glossary](../../../../glossary.md#term-rollback))</span>

## [Validation](../../../../glossary.md#term-validation)

Accepted worker commands (2026-10-22):

```bash
./gradlew :modules:prepare:test --tests '*PrepareContextTest' --tests '*PartitionDagTest' --tests '*AnalysisPublicShapeTest' --tests '*GraphPreparationTest' --tests '*BackendPartitionFinalizationHandoffTest' --tests '*BackendPartitionFinalizationTest'
./gradlew :modules:prepare:test :modules:prepare:javadoc
```

Focused: PASS, 61 tests/6 suites; final [module](../../../../glossary.md#term-module): PASS, 79 tests/12 suites; both zero failures,
errors or skips. Final [Javadoc](../../../../glossary.md#term-javadoc): zero warnings. Main verified fresh XML timestamp 09:29:37Z.
All eight Java files remained byte-identical to the tested versions through final review.
The corrected intermediate [fixture](../../../../glossary.md#term-fixture) [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) issue is resolved, not an open bug.

Main's exact accepted compatibility [checkpoint](../../../../glossary.md#term-checkpoint) (2026-10-22):

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
errors/skips. Combined `git diff --check` PASS. Existing [FFM](../../../../glossary.md#term-ffm) native-access/incubator-[vector](../../../../glossary.md#term-vector) warnings
are not Javadoc warnings and require no [scope](../../../../glossary.md#term-scope) fix. No root/native/performance campaign here;
real [scan](../../../../glossary.md#term-cumulative-scan) [conformance](../../../../glossary.md#term-conformance) and the single native-enabled root `test javadoc` remain at the low-level join.

## Documentation and review impact

Affected [validation](../../../../glossary.md#term-validation)/demand/failure-mode Javadocs and package [Javadoc](../../../../glossary.md#term-javadoc) reviewed/updated, including
[parameters](../../../../glossary.md#term-parameter)/results/exceptions; final Javadoc follows edits. Huygens independently approved exact
Class C code/docs/contracts/tests/[API](../../../../glossary.md#term-application-programming-interface-api) compatibility; successful executable evidence reused.
Main owns shared explanations/[planning](../../../../glossary.md#term-planning). Twelve factual status/boundary updates, no architecture/
[ADR](../../../../glossary.md#term-architecture-decision-record-adr) semantic change or new terminology; targeted glossary review covers [general scan](../../../../glossary.md#general-differentiable-scan), formal/[primal](../../../../glossary.md#term-primal)
[binding](../../../../glossary.md#term-binding), [residual](../../../../glossary.md#scan-residual-port-and-history-demand)/[history demand](../../../../glossary.md#scan-residual-port-and-history-demand), [partition](../../../../glossary.md#partition) [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) and [backend](../../../../glossary.md#backend) finalization. See Result for validation.

## Result

- Delivered exact scoped-body/role projection, static descriptor and final outer-demand validation,
  fail-closed pre-analysis rejection and existing resource declaration/assignment/finalization handoff.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Changed existing production `modules/prepare/src/main/java/.../prepare/GraphPreparation.java`,
  `analysis/PrepareContext.java` and `analysis/package-info.java` ([Javadoc](../../../../glossary.md#term-javadoc)).
- Changed five existing tests under `modules/prepare/src/test/java/.../prepare/`:
  `GraphPreparationTest`, `BackendPartitionFinalizationHandoffTest`, `BackendPartitionFinalizationTest`,
  `analysis/PrepareContextTest` and `analysis/PartitionDagTest`; eight Java files, no added types/files.
- Worker 61 focused/79 module tests + zero-warning Javadoc PASS; exact commands and fresh/reused
  main compilation, architecture and fixed-scan integration evidence are above.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Huygens final Class C code/docs APPROVE, zero P1/P2; Java unchanged; main [checkpoint](../../../../glossary.md#term-checkpoint) accepted 2026-10-22.
- Documentation: 12 factual status/boundary updates across `ARCHITECTURE.md`, scoped contracts,
  `docs/architecture/`, `docs/backend-guide/partition-preparer.md` and `docs/glossary.md`;
  no authority semantic change. Affected Javadocs reviewed/updated; no new glossary terminology.
- Independent documentation validation PASS: 512 local Markdown links/anchors, six glossary
  headings and whitespace. This closure synchronizes only the nine authorized planning records.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- No real backend execution/history-allocation proof, F64 domain completion, NN or persistence grant.
  Next bounded CPU F32 milestone remains Draft/not briefed/not Ready; no automatic successor launch.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [persistence glossary](../../../../glossary.md#term-persistence); [CPU glossary](../../../../glossary.md#term-cpu); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Unresolved issues: none. Required follow-up for this task: none.

Status: Complete
