# Task Prepare0012: Dynamic [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) geometry and resource contract

> **Editorial historical reading key.**
>
> This is a completed design delivery, not the implementation of its proposed signatures. Its packet
> assigns semantic dimension constraints to Compiler, stable declaration identities to Prepare,
> numeric geometry recipes to CPU, per-run resources to Runtime, and actual publication descriptors to
> Engine. It explicitly separates shape/byte admission from later producer-ordered length-value
> checks; neither establishes an outer loss mask.
>
> The historical statement that Java dynamic batch was unsupported belongs to this design checkpoint.
> For implemented metadata behavior, see [named-batch
> binding](../../../../reference/compiler/batch-binding.md#ordinary-named-batch-metadata-binding); for
> qualified execution, see the [current ordinary
> lifecycle](../../../../reference/engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle).
> Caller input storage remains borrowed through result closure, while Runtime owns run resources and
> preparation owns persistent resources. Detached materialization is a separate copy. The design's
> approval does not itself grant Scan, NN, persistence or native capability.

## Status

Complete — reviewed DESIGN delivery; independent Class C design/docs APPROVE, zero open P1/P2,
and MAIN delivery acceptance, 2026-10-24. User “ano” now explicitly accepts the reviewed architecture,
authority first, then ordinary [CPU](../../../../glossary.md#term-cpu) forward/actual first-order backward and bounded final A/B.
[Prepare0012A](0012a-dynamic-batch-authority-cutover.md) is Complete: authority enacted/MAIN-accepted
2026-10-24, Pasteur definitive APPROVE, zero open P1/P2. Current Java remains unsupported for dynamic
[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); no [Ready](../../../../glossary.md#term-ready) Java brief until MAIN separately admits planner output. Historical launch evidence is below.

## Change class

Class C — proposed [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare) resource, input-[binding](../../../../glossary.md#term-binding) and [publication](../../../../glossary.md#term-publication) boundaries affect
[ownership](../../../../glossary.md#term-ownership-meanings), [concurrency](../../../../glossary.md#term-concurrency) and generated hot paths. This task produces a decision packet only;
current architecture contracts and executable behavior remain unchanged.

## Goal

Produce one source-grounded proposed cross-[owner](../../../../glossary.md#term-owner) contract for actual per-[run](../../../../glossary.md#term-run) symbolic [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) B
through one [immutable](../../../../glossary.md#term-immutable) compiled/[prepared execution](../../../../glossary.md#prepared-execution--preparedexecution), with an exact package/type/[symbol](../../../../glossary.md#term-symbol) map and
[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/authority-change packet for independent review and user acceptance before Java changes.
The [selected program](../../../dynamic-batch-execution.md) fixes the direction: no mandatory
public B_max, required per-B full-preparation [cache](../../../../glossary.md#term-cache) or per-batch [graph](../../../../glossary.md#graph) rebuild. Do not reopen it.

## [Scope](../../../../glossary.md#term-scope)

- Specify the complete input-metadata → checked numeric geometry → resource creation → cold
  invocation binding → actual output-descriptor handoff through existing owners.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [binding glossary](../../../../glossary.md#term-binding); [descriptor glossary](../../../../glossary.md#term-descriptor); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Choose concrete proposed size/creator/metadata/publication signatures, packages and ownership
  in the design output. These choices are this task's deliverable, not preapproved new types.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [signatures glossary](../../../../glossary.md#term-signature); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Retain DynamicDimension/DimensionExpressions; first shape scope varies B with fixed feature
  and parameter dimensions, and initially fixed T for general Scan. Variable T remains later.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [feature glossary](../../../../glossary.md#term-software-feature); [parameter glossary](../../../../glossary.md#term-parameter); [dimensions glossary](../../../../glossary.md#dimension); [general Scan glossary](../../../../glossary.md#general-differentiable-scan))</span>
- Produce one [planning](../../../../glossary.md#term-planning) proposal at `docs/planning/modules/prepare/dynamic-batch-geometry-contract-proposal.md`
  under the user's design-only authorization. Include proposed ADR text and exact authority changes
  there; do not apply them to live contracts. MAIN owns shared index integration.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Once the contract is stable and accepted, provide at most one compact immediate implementation
  brief for MAIN admission; otherwise identify the exact unresolved decisions and withhold it.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>

## Non-goals

No production or test types, Java/[Javadoc](../../../../glossary.md#term-javadoc) changes, builds, native work, benchmarks, live authority
edits, [dependency](../../../../glossary.md#term-dependency) changes, implementation executor, [commit](../../../../glossary.md#term-commit) or push. No new session, executor,
second [graph](../../../../glossary.md#graph)/[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag), [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) or general proof framework. No [dtype](../../../../glossary.md#data-type--datatype)-wide capability promise,
general variable-T solution, automatic [model](../../../../glossary.md#term-model)-wide masking, higher-order or native/[backend](../../../../glossary.md#backend) grant.

## Contracts

Read the root [scope](../../../../glossary.md#term-scope) index first and use only its incorporated [owners](../../../../glossary.md#term-owner). These are current limits
to preserve or explicitly propose changing, not rules overridden by this brief.

- Root: [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>,
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span> and
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>.
- [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine): [`modules/runtime`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
  [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span> and
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>.
- [Compiler](../../../../glossary.md#term-compiler): [Compile artifacts](../../../../architecture/contracts/compiler-autograd.md#compile-artifacts)<span data-glossary-navigation="true"> ([Compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
  and [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>.
- [Backend](../../../../glossary.md#backend): [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>.
- [Scan](../../../../glossary.md#term-cumulative-scan): [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>
  and [Length-aware whole-batch scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>.
- Extensions: [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>,
  [`modules/training`](../../../../architecture/contracts/extensions-training.md#modulestraining) and
  [Generic graph source and values](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values)<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>,
  only for high-level reuse, lifecycle and separate cold/persistence gates.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>

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
  planning handoff; recheck before design launch.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [CPU glossary](../../../../glossary.md#term-cpu); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Parallel group: None; serial design and separately scoped independent review.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted uncommitted tree; predecessors are not described as merged.
- Integration order: independent planning review → MAIN Ready admission → subsequent user design
  go-ahead → design packet → independent decision review/user acceptance → separately admitted
  authority cutover before any Java. No downstream implementation authorization.
- Integration validation: MAIN docs-only source/symbol, links/fragments, size and diff checkpoint;
  the accepted design must name later module/boundary integration gates, without running them here.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [symbol glossary](../../../../glossary.md#term-symbol); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [module glossary](../../../../glossary.md#term-module))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

[Admission](../../../../glossary.md#term-admission) verification, 2026-10-24: MAIN verified all five [dependency](../../../../glossary.md#term-dependency) briefs Complete; the planner
checked their current Status/Result and the supplied base. Current source/contracts remain the
accepted dirty tree. No competing writer/conflict is active; shared [planning](../../../../glossary.md#term-planning) integration remains
MAIN-owned. Independent planning APPROVE and MAIN
admission establish Prepare0012 DESIGN as the sole [Ready](../../../../glossary.md#term-ready) planning frontier. MAIN reverified the
Ready brief, [authorized frontier](../../../../glossary.md#term-authorized-frontier), five Complete dependencies, unchanged HEAD/origin and accepted
dirty base on 2026-10-24 16:24:42 UTC. No competing writer is active. The user's subsequent go-ahead
authorizes this design context only; production implementation remains unauthorized.

## Files and symbols

Use the program's [focused current inventory](../../../dynamic-batch-execution.md#current-truth)
and source links rather than a new repository-wide audit. Focus on these existing seams:

- `model.shape.DynamicDimension`, `DimensionExpressions`, `model.tensor.TensorDescriptor`;
  `planning.memory.LogicalMemoryRequirement` and [compiler](../../../../glossary.md#term-compiler) `CompileArtifacts`/`CompileDiagnostics`.
- `prepare.analysis.PrepareContext`, `PreparationResourceRequirement.Buffer/Workspace`, shared
  [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation)/assignments; `runtime.memory.PreparedMemoryPlan`,
  `runtime.resource.PreparedRepresentationPlan` creators and `runtime.run.RunStateCreation/RunState`.
- [CPU](../../../../glossary.md#term-cpu) `internal.memory.CpuRepresentationRecipes`, `internal.executable.CpuPreparedExecutable`,
  `internal.lowering.CpuMatmulLowering.Geometry.pack` and existing loop/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[Scan](../../../../glossary.md#term-cumulative-scan) emitters.
- [Engine](../../../../glossary.md#term-engine) `AdvancedEngine.runOrdinaryOpen`/[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution)/`RunResult.Publication`;
  NN <span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>`ModelExecutionState`; Training `TrainingSession.bindValues` and [gradient accumulation](../../../../glossary.md#term-gradient-accumulation).
- Focused evidence: `PrepareContextTest`, `CompileDiagnosticsTest`, `EngineTypedLifecycleTest`,
  `RunStateCreationTest`, `TrainingSessionValuesTest`, existing Scan source/[inference](../../../../glossary.md#term-inference)/[AD](../../../../glossary.md#autograd)/CPU tests.

The design map must name every proposed added/moved production type, including nested types,
owning [module](../../../../glossary.md#term-module)/package, visibility, changed existing symbols, [dependency](../../../../glossary.md#term-dependency) consumers and compatibility
impact. This task adds none; exact future type names are deliberately not selected by this brief.

## Acceptance criteria

1. Deliver the exact proposed cross-owner handoff and package/type/symbol map, distinguishing
   existing APIs from proposed changes. Assign evaluation of semantic constraints, backend numeric
   recipes and Engine output descriptors to explicit owners, without Runtime Model/compiler edges.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [handoff glossary](../../../../glossary.md#term-handoff); [symbol glossary](../../../../glossary.md#term-symbol); [evaluation glossary](../../../../glossary.md#term-evaluation); [backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [descriptors glossary](../../../../glossary.md#term-descriptor); [Runtime glossary](../../../../glossary.md#term-runtime); [Model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler))</span>
2. Explicitly redesign fixed byteSize declarations/memory entries and zero-argument buffer/workspace
   creators. Specify signatures, units, checked limits, alignment, validation order, ownership,
   null/failure behavior and static compatibility. No fake byte counts, hidden max-capacity default,
   prepared mutation, late resource discovery or callback escape that bypasses today's contract.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [buffer glossary](../../../../glossary.md#term-buffer); [signatures glossary](../../../../glossary.md#term-signature); [alignment glossary](../../../../glossary.md#term-alignment); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [callback glossary](../../../../glossary.md#term-callback))</span>
3. Keep declared requirement/slot identities, routes and code fixed across 64→40→1→64. Run bytes derive from
   prepared typed recipes; Prepare does not execute or interpret backend geometry. Runtime does
   not interpret Shape, DimensionExpression, Operation or graphs, search routes, generate code,
   look up caches or bind per element/step. Numeric arrays/banks/resources belong to one run.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [Prepare glossary](../../../../glossary.md#prepare); [Shape glossary](../../../../glossary.md#shape); [Operation glossary](../../../../glossary.md#operation); [graphs glossary](../../../../glossary.md#graph); [element glossary](../../../../glossary.md#term-element))</span>
4. Specify consistent B extraction across inputs, fixed axes/parameters, dtype/requiresGrad/layout
   policy, checked counts/spans/offsets/overflow, storage admission and actual output publication.
   Preserve eight CompileArtifacts components unless the packet explicitly requests their change;
   do not use diagnostic strings as executable predicates.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [parameters glossary](../../../../glossary.md#term-parameter); [dtype glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [overflow glossary](../../../../glossary.md#term-overflow); [admission glossary](../../../../glossary.md#term-admission); [publication glossary](../../../../glossary.md#term-publication))</span>
5. Explicitly settle proposed B=0 admission, empty reduction/output/gradient behavior and concrete
   representability/resource limits, or surface named unresolved choices for user decision. Do not
   silently narrow to B>0 or impose public B_max. Retain every old static empty behavior; no
   implementation brief becomes actionable while these decisions remain unresolved.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
6. Separate shape/byte preflight at run admission from length VALUE preflight after producers and
   before this Scan's first writes. Cover produced lengths, invalid last length, T=0 and B=0 under
   the selected policy. Lengths do not solve variable T or define an outer loss mask/MEAN denominator.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [Scan glossary](../../../../glossary.md#term-cumulative-scan); [loss glossary](../../../../glossary.md#loss); [mask glossary](../../../../glossary.md#term-boolean-mask); [MEAN glossary](../../../../glossary.md#term-mean))</span>
7. Preserve borrow/result/prepared leases, close precedence, attempt-all cleanup, rollback/retry,
   Training transaction/overlap rules, NN exclusion and different-B concurrent-run isolation.
   Actual parameter gradients remain fixed-shaped; dynamic reduction/MATMUL geometry must honor
   existing AD count algebra. Accumulation stays a sum, not implicit sample weighting.<span data-glossary-navigation="true"> ([leases glossary](../../../../glossary.md#term-lease); [rollback glossary](../../../../glossary.md#term-rollback); [transaction glossary](../../../../glossary.md#term-transaction); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [AD glossary](../../../../glossary.md#autograd); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
8. Set a bounded ordinary CPU forward AND actual first-order backward vertical milestone before
   generic Scan extension, then high-level reuse of one parameter/optimizer state. Specify focused
   static regressions, numeric/empty/failure/concurrency tests, clean-Java/Class-File checks and
   bounded matched-work evidence for later implementation. Native/mixed/persistence/tuning remain
   separate gates; static length skipping is independent and not held on distant NN/persistence.
   Gate symbolic <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [optimizer state glossary](../../../../glossary.md#term-optimizer-state); [concurrency glossary](../../../../glossary.md#term-concurrency); [persistence glossary](../../../../glossary.md#term-persistence))</span>`Model.graph`/`defineTensor` acquisition on Compiler/Checkpoint cold support;
   symbolic graph save/load additionally needs wire qualification. Qualify builder acquisition
   independently and retain static saved Models/readers. None of these NN gates blocks the ordinary
   low-level Engine+CPU vertical.<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint); [qualification glossary](../../../../glossary.md#term-qualification))</span>
9. Include draft ADR, exact proposed root/owning-contract changes, affected explanations/Javadoc
   and any needed architecture/conformance/integration validation. Independent decision review
   and user acceptance precede live authority changes and Java. Draft only the stable immediate
   successor; unresolved design is reported, never disguised as implementation readiness.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation)

Design execution uses focused source/contract comparison and documentation checks only: validate
local links/fragments with MAIN's evidence checker, check the complete proposed [symbol](../../../../glossary.md#term-symbol)/[dependency](../../../../glossary.md#term-dependency)
map and approval gates, and [run](../../../../glossary.md#term-run) `wc -l -c` plus `git diff --check` on its admitted [planning](../../../../glossary.md#term-planning) outputs.
Compare changes with MAIN's pre-task baseline, not the unrelated dirty HEAD diff. No Java/[Gradle](../../../../glossary.md#term-gradle),
native or performance reruns. MAIN owns the serial docs [checkpoint](../../../../glossary.md#term-checkpoint) and future gate selection.

## Documentation and review impact

[Planning](../../../../glossary.md#term-planning) plus general/example profiles applied. Independent clean Class C design/docs review
APPROVE, zero open P1/P2; MAIN accepts delivery under its conditional [admission](../../../../glossary.md#term-admission). Targeted glossary
[lookup](../../../../glossary.md#term-lookup) covered [dimension](../../../../glossary.md#dimension)/expression, deferred-constraint, prepared-plan and [run](../../../../glossary.md#term-run)-state terms.
No active meaning/[API](../../../../glossary.md#term-application-programming-interface-api) or Java changes: existing glossary/[Javadoc](../../../../glossary.md#term-javadoc) remain accurate and unchanged;
the proposal identifies their later cutover impact. User architecture approval is now recorded;
Prepare0012A authority/docs/glossary review and MAIN acceptance are complete, 2026-10-24; no Java change.

## Result

[Proposal](../dynamic-batch-geometry-contract-proposal.md) and six task/navigation deltas reviewed
against MAIN's baseline: exact [handoff](../../../../glossary.md#term-handoff)/nine-type/static-retention map, actual [AD](../../../../glossary.md#autograd) closure, B=0/limits,
resource/[lease](../../../../glossary.md#term-lease) ordering, generated [identity](../../../../glossary.md#term-identity-meanings) and draft [ADR](../../../../glossary.md#term-architecture-decision-record-adr)/authority packet. Prior P2s closed: explicit
[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-required mode and [constant](../../../../glossary.md#term-constant)-five-[limb](../../../../glossary.md#exact-superaccumulator-and-limbs) [F32](../../../../glossary.md#floating-point-precision-and-rounding) emission. Independent final APPROVE; MAIN delivery
acceptance complete. Only seven admitted [planning](../../../../glossary.md#term-planning) files changed; no Java/authority/build/benchmark.
Documentation impact: planning only; later [Javadoc](../../../../glossary.md#term-javadoc)/glossary impact identified, not applied.
[Validation](../../../../glossary.md#term-validation): focused source/contracts reused; final reviewer ran [`node`](../../../../glossary.md#node) with MAIN's check-links.mjs
on all seven files: 303 links/85 fragments, zero errors; `wc -l -c`, scoped `git diff --check` and
new-file whitespace PASS. `node /private/tmp/synaptik-prepare0012-design.bpgzxE/audit.mjs`: 2428 files
outside seven, index/native [hashes](../../../../glossary.md#term-hash) and HEAD/origin match baseline.json. No executable change/test [run](../../../../glossary.md#term-run).
Status: Complete
Current follow-up: user architecture approval recorded; [Prepare0012A](0012a-dynamic-batch-authority-cutover.md)
is Complete, authority enacted/MAIN-accepted 2026-10-24 after Pasteur definitive APPROVE, zero open
P1/P2. Current Java dynamic batch remains unsupported; no [Ready](../../../../glossary.md#term-ready) Java brief until MAIN separately
admits planner output. No automatic successor; historical design [admission](../../../../glossary.md#term-admission)/evidence is unchanged.
