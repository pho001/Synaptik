# Task Prepare0011: Length-aware [Scan](../../../../glossary.md#general-differentiable-scan) projection and resource [handoff](../../../../glossary.md#term-handoff)

> **Editorial historical reading key.**
>
> This historical extension admits length-aware metadata through Prepare, not length-value execution.
> Compiler supplies proven row roles, exact saved-primal bindings and actual reverse bodies. Prepare
> separates the final lengths actual from body formals, validates static boundary descriptors and
> complete outer demand, then projects the unchanged attachment to its selected backend analyzer.
> Equal extents do not turn SHARED histories into row histories; numerical ID collisions do not merge
> scopes.
>
> Test analyzers declare lengths, boundary and demanded-history geometry; Prepare assigns declarations
> rather than allocating their bytes. The [current finalization
> handoff](../../../../reference/runtime/orchestration.md#current-prepare-orchestration-assignment-and-backend-finalization)
> explains backend acquisition, staged rollback and Runtime ownership. The [current capability
> contract](../../../../reference/compiler/planning.md#current-operation-capability-contracts) keeps
> complete-body support separate. Historical CPU/cold/codec rejection checks record then-retained
> gates, not a permanent current prohibition; these fixtures prove neither skipping nor physical
> history stores.

## Status

Complete — MAIN accepted 2026-10-24 13:00:15 UTC after independent Class C code/docs APPROVE, zero open P1/P2.
Clean sole writer started 2026-10-24 12:31 UTC after Tesla [planning](../../../../glossary.md#term-planning) APPROVE and MAIN [admission](../../../../glossary.md#term-admission).
ImplementationAuthorized=true records completed Prepare0011 only. Launch authorization was from the user's
latest “pust se do prace” request. Worker gates and MAIN [SPI](../../../../glossary.md#term-service-provider-interface-spi)/retained-[guard](../../../../glossary.md#term-guard) [checkpoint](../../../../glossary.md#term-checkpoint) PASS;
no new launch, downstream tasks, [commit](../../../../glossary.md#term-commit) or push.

## Change class

Class C — scoped [scan](../../../../glossary.md#general-differentiable-scan) [bindings](../../../../glossary.md#term-binding), saved-value demand and the analysis/resource-assignment boundary.
Clean implementation and independent targeted code/documentation review complete; MAIN accepted 2026-10-24.

## Goal

Project Compiler0013 length-aware forward and actual first-order reverse scans through existing
[Prepare](../../../../glossary.md#prepare) [owners](../../../../glossary.md#term-owner), validating static [bindings](../../../../glossary.md#term-binding)/demand before analysis and preserving the existing
declaration → assignment → finalization [handoff](../../../../glossary.md#term-handoff). This supplies [metadata](../../../../glossary.md#term-metadata)/recipes, not execution.

## [Scope](../../../../glossary.md#term-scope) and non-goals

Extend existing [Prepare](../../../../glossary.md#prepare) [validation](../../../../glossary.md#term-validation) and focused tests only; no new production type, public [API](../../../../glossary.md#term-application-programming-interface-api),
package, [dependency](../../../../glossary.md#term-dependency), [graph](../../../../glossary.md#graph), runner or interpreter. [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) retain body row-algebra proof
and [AD](../../../../glossary.md#autograd) [admission](../../../../glossary.md#term-admission). Prepare never reads length values, computes [tensor](../../../../glossary.md#tensor) results or allocates physical
[buffers](../../../../glossary.md#term-buffer). [CPU](../../../../glossary.md#term-cpu) mixed-boundary [binding](../../../../glossary.md#term-binding), length [preflight](../../../../glossary.md#term-preflight), skipping/execution, cold/v5/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), fixed [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt),
higher-order AD, low types, generated [identities](../../../../glossary.md#term-identity-meanings) and performance work stay held.

## Contracts

Read AGENTS.md and root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>, then:

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>, [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates),
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine) [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  and [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle).
- Recurrent-[scan](../../../../glossary.md#general-differentiable-scan) [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>:
  [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities),
  [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>,
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)<span data-glossary-navigation="true"> ([primal glossary](../../../../glossary.md#term-primal); [bindings glossary](../../../../glossary.md#term-binding))</span>.
- [Length-aware whole-batch scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>, including
  [Header, bindings and source admission](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission)<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>,
  [Bounded row-role analysis](../../../../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>,
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>.

These are accepted authority, not new rules. Stop on missing/ambiguous authority or a required
[scope](../../../../glossary.md#term-scope) expansion; do not invent a Prepare row-proof engine or widen [backend capability](../../../../glossary.md#backend-capability).

## Dependencies and integration

- Depends on: Complete Model0037 authority, Model0038 source/IR, Compiler0013 capture/first-order AD,
  Planning0008 queries and Prepare0010 projection/resource handoff.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [capture glossary](../../../../glossary.md#term-capture-meanings); [AD glossary](../../../../glossary.md#autograd))</span>
- Conflicts with: scan header/roles/demand, Prepare projection/resource lifecycle, Compiler/CPU/cold/
  codec/NN scan writers and shared-document writers; none active, serialize these scopes.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#general-differentiable-scan); [lifecycle glossary](../../../../glossary.md#lifecycle); [CPU glossary](../../../../glossary.md#term-cpu); [codec glossary](../../../../glossary.md#term-codec); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus all
  accepted uncommitted work at planning verification, 2026-10-24; predecessors are not merged commits.<span data-glossary-navigation="true"> ([commits glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Integration order: accepted dependencies → Ready/MAIN admission → user implementation authorization
  → clean implementation → independent review and MAIN checkpoint → MAIN acceptance.
- Integration [validation](../../../../glossary.md#term-validation): named narrow Prepare0011 [handoff](../../../../glossary.md#term-handoff) [checkpoint](../../../../glossary.md#term-checkpoint) below, once after code freeze.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; exclusive independent docs writer after worker freeze.

[Planning](../../../../glossary.md#term-planning) verification (2026-10-24): predecessor Status/Result and current authority/source/tests agree:
[Model0037](../../model/tasks/0037-length-aware-general-scan-authority.md#result),
[Model0038](../../model/tasks/0038-length-aware-scan-source-foundation.md#result),
[Compiler0013](../../compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result),
[Planning0008](../../planning/tasks/0008-general-scan-body-capability-query.md#result) and
[Prepare0010](0010-general-scan-projection-and-resource-handoff.md#result) are Complete.
At verification, [Model](../../../../glossary.md#term-model) separated actual/formal counts, [Compiler](../../../../glossary.md#term-compiler) emitted final mixed roles/demand
and [Prepare](../../../../glossary.md#prepare) rejected length mode. MAIN launch [admission](../../../../glossary.md#term-admission) 2026-10-24 12:25 UTC verified the [Ready](../../../../glossary.md#term-ready)
brief/all five planning [hashes](../../../../glossary.md#term-hash) unchanged, all five dependencies Complete, the same accepted base,
2432 Git files, index empty and authority/source matching the accepted [snapshot](../../../../glossary.md#term-snapshot). Accepted dirty
work was preserved; [authorized frontier](../../../../glossary.md#term-authorized-frontier) Prepare0011, no live conflicting writer/agent.
ImplementationAuthorized=true from the user's latest “pust se do prace” request, Prepare0011 only;
no downstream tasks, [commit](../../../../glossary.md#term-commit) or push. Clean sole writer launched at 12:31 UTC.

## Files and symbols

Existing package prefix `io.github.pho001.synaptik.`; production root
`modules/prepare/src/main/java/`, matching tests under `modules/prepare/src/test/java/`:

- `prepare.analysis.PrepareContext#validateScan` — separate lengths and formal [binding](../../../../glossary.md#term-binding) checks;
  `prepare.analysis.PartitionDag` — retain exact attachment and outer port occurrences.
- `prepare.GraphPreparation#Projection`, `project`, `prepareInternal` — complete-graph demand and
  all-context validation before the first analyzer.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation))</span>
- Existing `prepare.analysis.BackendPartitionAnalysis` / `PreparationResourceRequirement`,
  `prepare.PreparationResourceAssignment` / `BackendPartitionFinalizationHandoff` — retain staged
  declaration/assignment semantics; no new resource API or backend geometry policy.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend))</span>
- `analysis.PrepareContextTest`, `analysis.PartitionDagTest`, `GraphPreparationTest` and existing
  `BackendPartitionFinalizationHandoffTest` / `BackendPartitionFinalizationTest` — focused fixtures.
- Read-only [handoff](../../../../glossary.md#term-handoff) references: [Compiler](../../../../glossary.md#term-compiler) `GraphCompiler` final queries, `ScanBodyInference` demand,
  `LengthAwareScanBodyInferenceTest`, `LengthAwareScanGradientRulesTest`, `GeneralScanCompilerTest`;
  CPU <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>`CpuGeneralScanLowering#analyze` and `CpuGeneralScanPreparedExecutable`; cold/[codec](../../../../glossary.md#term-codec) guards below.

## Acceptance criteria

1. Use checked `formalCount=c+s+i`, `actualCount=formalCount+1` in length mode. Bind only the first
   formalCount actuals; keep the final lengths actual in the outer projection and logical-use checks,
   never in body formals or the outer gradient OR. All-valid counts/flags remain unchanged.
2. Validate static T/B, [homogeneous](../../../../glossary.md#term-homogeneous) [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) body/formal types, non-[gradient](../../../../glossary.md#term-gradient) [INT64](../../../../glossary.md#data-type--datatype) [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `[B]`
   lengths, leading ROW(0) carry/next-carry/emission B extents and declared sequence-axis extents,
   including unused roles. ROW(1) requires rank two; SHARED has no inferred batch axis. Check
   canonical formal/result layouts, exact positional dtype/shape/gradient bindings and checked geometry. Preserve outer
   actual layouts; do not import CPU read predicates or add a new canonical-lengths restriction.
   Invalid descriptors/roles, dynamic locals/siblings and overflow fail before any analyzer,
   including a bad later partition, T=0, B=0/1 and empty features. Never inspect length values.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [axis glossary](../../../../glossary.md#term-axis); [rank glossary](../../../../glossary.md#term-rank); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [canonical glossary](../../../../glossary.md#term-canonical-form); [layouts glossary](../../../../glossary.md#layout); [dtype glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [bindings glossary](../../../../glossary.md#term-binding); [CPU glossary](../../../../glossary.md#term-cpu); [descriptors glossary](../../../../glossary.md#term-descriptor); [overflow glossary](../../../../glossary.md#term-overflow); [partition glossary](../../../../glossary.md#partition); [features glossary](../../../../glossary.md#term-feature))</span>
3. Retain exact attrs/body/local graph, ordered next-carry/emission/residual bindings and demand;
   repeated actual ports and aliased roles stay positional. Local IDs resolve only inside their
   body, even when numerically equal to outer IDs or IDs in another reused-template occurrence.
   Validate boundary facts without re-deriving body numerical/dataflow semantics.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [occurrence glossary](../../../../glossary.md#term-occurrence); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>
4. Project broader valid forward sources with SHARED carry suffixes and SHARED/ROW(1) sequences,
   and actual Compiler-generated reverse scans with leading row cotangents plus SHARED parameter
   accumulators and mixed ROW(0)/ROW(1)/SHARED saved-primal/seed sequences. Do not apply Compiler's
   restricted original-source AD admission to these projections. Test both dtypes/traversals,
   false-flag carry propagation, tied positions and pruned sequence roles using Compiler output.
   At F=B, shared transpose history stays <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [parameter glossary](../../../../glossary.md#term-parameter); [primal glossary](../../../../glossary.md#term-primal); [seed glossary](../../../../glossary.md#term-seed); [AD glossary](../../../../glossary.md#autograd); [admission glossary](../../../../glossary.md#term-admission); [transpose glossary](../../../../glossary.md#term-transpose))</span>`[T,F,H]`, not `[T,B,F,H]`; transposed row history keeps
   ROW(1). Preserve exact saved primal bindings, not recomputation or role inference from shape.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
5. Revalidate ordered demand against final publications and every outer consumer, including
   forward-to-reverse and cross-partition edges, before analysis. Direct contexts check locally
   knowable obligations. Missing/extra/stale demand rejects. Body-internal uses do not demand
   history; dead forward/reverse residuals stay undemanded, with no history declaration/store duty.
6. Test-only analyzers declare exact checked buffer/workspace requirements before assignment:
   separate INT64 lengths bytes from F32/F64 boundaries, carry/staging/temporary needs and demanded
   histories with their original local shapes. Preserve zero dead-history bytes and no added B×P
   parameter bank. Geometry remains backend-owned; shared Prepare only validates/transports and
   assigns declarations. Finalizers receive exact declaration/slot/representation associations;
   no body-local value becomes an outer buffer by ID coincidence or undeclared storage at
   finalization. Missing cross-partition declarations fail before finalization. Retain rollback,
   one-time ownership transfer and run isolation. Fixtures invoke no physical creators/executables;
   metadata evidence does not claim physical history allocation or runtime skipping.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [backend glossary](../../../../glossary.md#backend); [Prepare glossary](../../../../glossary.md#prepare); [rollback glossary](../../../../glossary.md#term-rollback); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [run glossary](../../../../glossary.md#term-run); [metadata glossary](../../../../glossary.md#term-metadata); [runtime glossary](../../../../glossary.md#term-runtime))</span>
7. Preserve old all-valid source/Prepare acceptance beyond CPU's whitelist, canonical residual
   ordering/demand, old constructor/default paths and fixed scans. Real CPU pointwise/shaped
   admission, cold capture/reconstruction and direct codec rejection remain closed for length mode;
   v1–v4/default identities are unchanged. No dependency/build/backend/Runtime mutation is needed.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [capture glossary](../../../../glossary.md#term-capture-meanings); [codec glossary](../../../../glossary.md#term-codec); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## [Validation](../../../../glossary.md#term-validation)

[JDK](../../../../glossary.md#term-java-development-kit-jdk) 26. Worker: focused iteration, then one final [Prepare](../../../../glossary.md#prepare) test/[Javadoc](../../../../glossary.md#term-javadoc) gate:

```bash
./gradlew :modules:prepare:test --tests '*PrepareContextTest' --tests '*PartitionDagTest' --tests '*GraphPreparationTest' --tests '*BackendPartitionFinalizationHandoffTest' --tests '*BackendPartitionFinalizationTest'
./gradlew :modules:prepare:test :modules:prepare:javadoc
```

MAIN's narrow [handoff](../../../../glossary.md#term-handoff) [checkpoint](../../../../glossary.md#term-checkpoint) checks existing Prepare [SPI](../../../../glossary.md#term-service-provider-interface-spi) consumers and retained [CPU](../../../../glossary.md#term-cpu)/cold/[codec](../../../../glossary.md#term-codec)
rejection; reuse successful worker evidence, no duplicate [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :backends:cpu:compileTestJava :backends:metal:compileTestJava :modules:engine:compileTestJava
./gradlew -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false \
  :backends:cpu:test --tests '*CpuGeneralScanCapabilityTest.lengthModeRejectsBeforeBothRealPointwiseAndShapedFallbackPaths' \
  :modules:compiler:test --tests '*SourceGraphReconstructionTest.lengthModeColdPreflightRejectsLateDeadAndZeroTimeBeforeAnyFormalOrProducerId' \
  :extensions:checkpoint:test --tests '*GraphArtifactCodecTest.directCodecLengthGuardPrecedesV4SelectionSizingAndStorageEvenWithoutColdPreflight' \
  --tests '*GraphArtifactV4Test.lengthModeRejectsColdCopyRebindAndSaveBeforeStateCopyOrOutputIdsIncludingZeroTime'
git diff --check
```

CPU maps to `modules/backend/cpu` (`:backends:cpu`), Checkpoint to `modules/checkpoint`
(`:extensions:checkpoint`). No root/native/performance campaign or execution/[conformance](../../../../glossary.md#term-conformance) claim;
later CPU capability owns its independent execution [qualification](../../../../glossary.md#term-qualification). Stop if a new edge/contract or
non-Prepare executable change is needed. Worker does not [run](../../../../glossary.md#term-run) MAIN's checkpoint.

## Documentation and review impact

Independent Class C review finalizes [Prepare](../../../../glossary.md#prepare) Javadocs, factual root/scoped status, boundary/[scan](../../../../glossary.md#general-differentiable-scan)
explanations, [partition](../../../../glossary.md#partition)-preparer guide, [API](../../../../glossary.md#term-application-programming-interface-api) references, targeted glossary entries and five plans.
It uses repo general/architecture/API/[backend](../../../../glossary.md#backend)/[planning](../../../../glossary.md#term-planning) styles and write-page/writing_quality for
surgical prose, without Pages. No new rule, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), term, API or [owner](../../../../glossary.md#term-owner) is selected. Executable evidence
is reused; final Prepare [Javadoc](../../../../glossary.md#term-javadoc) follows comment-only edits. MAIN retains overall acceptance.

## Result

Implemented [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) static lengths/formal, [dtype](../../../../glossary.md#data-type--datatype), geometry and row-boundary checks. Exact
mixed forward/actual-reverse roles, scoped IDs and ordered demand use the unchanged staged [handoff](../../../../glossary.md#term-handoff).
Seven Java files changed: four production files (only `PrepareContext` executable) and the three
focused test classes above. Seventeen Markdown files reconcile factual status; no new type/[API](../../../../glossary.md#term-application-programming-interface-api)/edge.
Worker commands above, [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26.0.1, both [scan](../../../../glossary.md#general-differentiable-scan) performance flags false: focused 71 tests, full [Prepare](../../../../glossary.md#prepare)
92 tests/12 suites, zero failures/errors/skips; [Javadoc](../../../../glossary.md#term-javadoc) PASS. Development wrapper-[cache](../../../../glossary.md#term-cache) denial and
test-only initialized-[constant](../../../../glossary.md#term-constant) [fixture](../../../../glossary.md#term-fixture) failure were resolved before these successful gates.
MAIN combined the [SPI](../../../../glossary.md#term-service-provider-interface-spi) [compile](../../../../glossary.md#compile) and four named guards above: PASS, 34 tasks, 7 executed/27 up-to-date.
Frozen 39-Java [hashes](../../../../glossary.md#term-hash) matched; independent Class C code/docs APPROVE, zero open P1/P2. Review adds comments only,
reuses executable evidence, and runs final `:modules:prepare:javadoc` with the same flags: PASS,
9 tasks, 2 executed/7 up-to-date, no warnings. Normal escalated retry resolved the wrapper lock.
Documentation impact: root/scoped status, explanations, API/guide/glossary and Javadocs finalized.
17-file links/anchors/format, 39-file comment-only [identity](../../../../glossary.md#term-identity-meanings), [scope](../../../../glossary.md#term-scope)/native/index and diff checks PASS.
Unresolved issues: none. No physical allocation, execution or skipping claim.
Status: Complete
MAIN accepted 2026-10-24 13:00:15 UTC. Follow-up required: none.
