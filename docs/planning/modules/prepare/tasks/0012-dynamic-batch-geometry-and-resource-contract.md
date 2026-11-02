# Task Prepare0012: Dynamic batch geometry and resource contract

## Status

Complete — reviewed DESIGN delivery; independent Class C design/docs APPROVE, zero open P1/P2,
and MAIN delivery acceptance, 2026-10-24. User “ano” now explicitly accepts the reviewed architecture,
authority first, then ordinary CPU forward/actual first-order backward and bounded final A/B.
[Prepare0012A](0012a-dynamic-batch-authority-cutover.md) is Complete: authority enacted/MAIN-accepted
2026-10-24, Pasteur definitive APPROVE, zero open P1/P2. Current Java remains unsupported for dynamic
batch; no Ready Java brief until MAIN separately admits planner output. Historical launch evidence is below.

## Change class

Class C — proposed Runtime/Prepare resource, input-binding and publication boundaries affect
ownership, concurrency and generated hot paths. This task produces a decision packet only;
current architecture contracts and executable behavior remain unchanged.

## Goal

Produce one source-grounded proposed cross-owner contract for actual per-run symbolic batch B
through one immutable compiled/prepared execution, with an exact package/type/symbol map and
ADR/authority-change packet for independent review and user acceptance before Java changes.
The [selected program](../../../dynamic-batch-execution.md) fixes the direction: no mandatory
public B_max, required per-B full-preparation cache or per-batch graph rebuild. Do not reopen it.

## Scope

- Specify the complete input-metadata → checked numeric geometry → resource creation → cold
  invocation binding → actual output-descriptor handoff through existing owners.
- Choose concrete proposed size/creator/metadata/publication signatures, packages and ownership
  in the design output. These choices are this task's deliverable, not preapproved new types.
- Retain DynamicDimension/DimensionExpressions; first shape scope varies B with fixed feature
  and parameter dimensions, and initially fixed T for general Scan. Variable T remains later.
- Produce one planning proposal at `docs/planning/modules/prepare/dynamic-batch-geometry-contract-proposal.md`
  under the user's design-only authorization. Include proposed ADR text and exact authority changes
  there; do not apply them to live contracts. MAIN owns shared index integration.
- Once the contract is stable and accepted, provide at most one compact immediate implementation
  brief for MAIN admission; otherwise identify the exact unresolved decisions and withhold it.

## Non-goals

No production or test types, Java/Javadoc changes, builds, native work, benchmarks, live authority
edits, dependency changes, implementation executor, commit or push. No new session, executor,
second graph/DAG, certificate or general proof framework. No dtype-wide capability promise,
general variable-T solution, automatic model-wide masking, higher-order or native/backend grant.

## Contracts

Read the root scope index first and use only its incorporated owners. These are current limits
to preserve or explicitly propose changing, not rules overridden by this brief.

- Root: [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing) and
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules).
- Runtime/Prepare/Engine: [`modules/runtime`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
  [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) and
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution).
- Compiler: [Compile artifacts](../../../../architecture/contracts/compiler-autograd.md#compile-artifacts)
  and [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation).
- Backend: [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
- Scan: [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)
  and [Length-aware whole-batch scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan).
- Extensions: [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models),
  [`modules/training`](../../../../architecture/contracts/extensions-training.md#modulestraining) and
  [Generic graph source and values](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values),
  only for high-level reuse, lifecycle and separate cold/persistence gates.

If an applicable contract is missing or ambiguous, surface the exact issue for decision. Do not
invent architecture or apply a proposed authority change while executing this design task.

## Dependencies and integration

- Depends on: [Prepare0011](0011-length-aware-scan-projection-and-resource-handoff.md#result),
  [Runtime0016](../../runtime/tasks/0016-persistent-prepared-resource-lifecycle.md),
  [Engine0021](../../engine/tasks/0021-prepared-execution-fresh-value-binding.md#result),
  [Compiler0013](../../compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result) and
  [CPU0037](../../../backends/cpu/tasks/0037-general-scan-dense-read-views.md#result), all Complete.
- Conflicts with: shared geometry/declaration/creator/publication contracts, CPU length binder or
  resource ABI decisions, and shared planning/authority writes. No competing writer active at
  planning handoff; recheck before design launch.
- Parallel group: None; serial design and separately scoped independent review.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted uncommitted tree; predecessors are not described as merged.
- Integration order: independent planning review → MAIN Ready admission → subsequent user design
  go-ahead → design packet → independent decision review/user acceptance → separately admitted
  authority cutover before any Java. No downstream implementation authorization.
- Integration validation: MAIN docs-only source/symbol, links/fragments, size and diff checkpoint;
  the accepted design must name later module/boundary integration gates, without running them here.
- Shared-document integration owner: MAIN.

Admission verification, 2026-10-24: MAIN verified all five dependency briefs Complete; the planner
checked their current Status/Result and the supplied base. Current source/contracts remain the
accepted dirty tree. No competing writer/conflict is active; shared planning integration remains
MAIN-owned. Independent planning APPROVE and MAIN
admission establish Prepare0012 DESIGN as the sole Ready planning frontier. MAIN reverified the
Ready brief, authorized frontier, five Complete dependencies, unchanged HEAD/origin and accepted
dirty base on 2026-10-24 16:24:42 UTC. No competing writer is active. The user's subsequent go-ahead
authorizes this design context only; production implementation remains unauthorized.

## Files and symbols

Use the program's [focused current inventory](../../../dynamic-batch-execution.md#current-truth)
and source links rather than a new repository-wide audit. Focus on these existing seams:

- `model.shape.DynamicDimension`, `DimensionExpressions`, `model.tensor.TensorDescriptor`;
  `planning.memory.LogicalMemoryRequirement` and compiler `CompileArtifacts`/`CompileDiagnostics`.
- `prepare.analysis.PrepareContext`, `PreparationResourceRequirement.Buffer/Workspace`, shared
  `GraphPreparation`/assignments; `runtime.memory.PreparedMemoryPlan`,
  `runtime.resource.PreparedRepresentationPlan` creators and `runtime.run.RunStateCreation/RunState`.
- CPU `internal.memory.CpuRepresentationRecipes`, `internal.executable.CpuPreparedExecutable`,
  `internal.lowering.CpuMatmulLowering.Geometry.pack` and existing loop/MATMUL/Scan emitters.
- Engine `AdvancedEngine.runOrdinaryOpen`/`PreparedExecution`/`RunResult.Publication`;
  NN `ModelExecutionState`; Training `TrainingSession.bindValues` and gradient accumulation.
- Focused evidence: `PrepareContextTest`, `CompileDiagnosticsTest`, `EngineTypedLifecycleTest`,
  `RunStateCreationTest`, `TrainingSessionValuesTest`, existing Scan source/inference/AD/CPU tests.

The design map must name every proposed added/moved production type, including nested types,
owning module/package, visibility, changed existing symbols, dependency consumers and compatibility
impact. This task adds none; exact future type names are deliberately not selected by this brief.

## Acceptance criteria

1. Deliver the exact proposed cross-owner handoff and package/type/symbol map, distinguishing
   existing APIs from proposed changes. Assign evaluation of semantic constraints, backend numeric
   recipes and Engine output descriptors to explicit owners, without Runtime Model/compiler edges.
2. Explicitly redesign fixed byteSize declarations/memory entries and zero-argument buffer/workspace
   creators. Specify signatures, units, checked limits, alignment, validation order, ownership,
   null/failure behavior and static compatibility. No fake byte counts, hidden max-capacity default,
   prepared mutation, late resource discovery or callback escape that bypasses today's contract.
3. Keep declared requirement/slot identities, routes and code fixed across 64→40→1→64. Run bytes derive from
   prepared typed recipes; Prepare does not execute or interpret backend geometry. Runtime does
   not interpret Shape, DimensionExpression, Operation or graphs, search routes, generate code,
   look up caches or bind per element/step. Numeric arrays/banks/resources belong to one run.
4. Specify consistent B extraction across inputs, fixed axes/parameters, dtype/requiresGrad/layout
   policy, checked counts/spans/offsets/overflow, storage admission and actual output publication.
   Preserve eight CompileArtifacts components unless the packet explicitly requests their change;
   do not use diagnostic strings as executable predicates.
5. Explicitly settle proposed B=0 admission, empty reduction/output/gradient behavior and concrete
   representability/resource limits, or surface named unresolved choices for user decision. Do not
   silently narrow to B>0 or impose public B_max. Retain every old static empty behavior; no
   implementation brief becomes actionable while these decisions remain unresolved.
6. Separate shape/byte preflight at run admission from length VALUE preflight after producers and
   before this Scan's first writes. Cover produced lengths, invalid last length, T=0 and B=0 under
   the selected policy. Lengths do not solve variable T or define an outer loss mask/MEAN denominator.
7. Preserve borrow/result/prepared leases, close precedence, attempt-all cleanup, rollback/retry,
   Training transaction/overlap rules, NN exclusion and different-B concurrent-run isolation.
   Actual parameter gradients remain fixed-shaped; dynamic reduction/MATMUL geometry must honor
   existing AD count algebra. Accumulation stays a sum, not implicit sample weighting.
8. Set a bounded ordinary CPU forward AND actual first-order backward vertical milestone before
   generic Scan extension, then high-level reuse of one parameter/optimizer state. Specify focused
   static regressions, numeric/empty/failure/concurrency tests, clean-Java/Class-File checks and
   bounded matched-work evidence for later implementation. Native/mixed/persistence/tuning remain
   separate gates; static length skipping is independent and not held on distant NN/persistence.
   Gate symbolic `Model.graph`/`defineTensor` acquisition on Compiler/Checkpoint cold support;
   symbolic graph save/load additionally needs wire qualification. Qualify builder acquisition
   independently and retain static saved Models/readers. None of these NN gates blocks the ordinary
   low-level Engine+CPU vertical.
9. Include draft ADR, exact proposed root/owning-contract changes, affected explanations/Javadoc
   and any needed architecture/conformance/integration validation. Independent decision review
   and user acceptance precede live authority changes and Java. Draft only the stable immediate
   successor; unresolved design is reported, never disguised as implementation readiness.

## Validation

Design execution uses focused source/contract comparison and documentation checks only: validate
local links/fragments with MAIN's evidence checker, check the complete proposed symbol/dependency
map and approval gates, and run `wc -l -c` plus `git diff --check` on its admitted planning outputs.
Compare changes with MAIN's pre-task baseline, not the unrelated dirty HEAD diff. No Java/Gradle,
native or performance reruns. MAIN owns the serial docs checkpoint and future gate selection.

## Documentation and review impact

Planning plus general/example profiles applied. Independent clean Class C design/docs review
APPROVE, zero open P1/P2; MAIN accepts delivery under its conditional admission. Targeted glossary
lookup covered dimension/expression, deferred-constraint, prepared-plan and run-state terms.
No active meaning/API or Java changes: existing glossary/Javadoc remain accurate and unchanged;
the proposal identifies their later cutover impact. User architecture approval is now recorded;
Prepare0012A authority/docs/glossary review and MAIN acceptance are complete, 2026-10-24; no Java change.

## Result

[Proposal](../dynamic-batch-geometry-contract-proposal.md) and six task/navigation deltas reviewed
against MAIN's baseline: exact handoff/nine-type/static-retention map, actual AD closure, B=0/limits,
resource/lease ordering, generated identity and draft ADR/authority packet. Prior P2s closed: explicit
batch-required mode and constant-five-limb F32 emission. Independent final APPROVE; MAIN delivery
acceptance complete. Only seven admitted planning files changed; no Java/authority/build/benchmark.
Documentation impact: planning only; later Javadoc/glossary impact identified, not applied.
Validation: focused source/contracts reused; final reviewer ran `node` with MAIN's check-links.mjs
on all seven files: 303 links/85 fragments, zero errors; `wc -l -c`, scoped `git diff --check` and
new-file whitespace PASS. `node /private/tmp/synaptik-prepare0012-design.bpgzxE/audit.mjs`: 2428 files
outside seven, index/native hashes and HEAD/origin match baseline.json. No executable change/test run.
Status: Complete
Current follow-up: user architecture approval recorded; [Prepare0012A](0012a-dynamic-batch-authority-cutover.md)
is Complete, authority enacted/MAIN-accepted 2026-10-24 after Pasteur definitive APPROVE, zero open
P1/P2. Current Java dynamic batch remains unsupported; no Ready Java brief until MAIN separately
admits planner output. No automatic successor; historical design admission/evidence is unchanged.
