# Task Prepare0011: Length-aware Scan projection and resource handoff

## Status

Complete — MAIN accepted 2026-10-24 13:00:15 UTC after independent Class C code/docs APPROVE, zero open P1/P2.
Clean sole writer started 2026-10-24 12:31 UTC after Tesla planning APPROVE and MAIN admission.
ImplementationAuthorized=true records completed Prepare0011 only. Launch authorization was from the user's
latest “pust se do prace” request. Worker gates and MAIN SPI/retained-guard checkpoint PASS;
no new launch, downstream tasks, commit or push.

## Change class

Class C — scoped scan bindings, saved-value demand and the analysis/resource-assignment boundary.
Clean implementation and independent targeted code/documentation review complete; MAIN accepted 2026-10-24.

## Goal

Project Compiler0013 length-aware forward and actual first-order reverse scans through existing
Prepare owners, validating static bindings/demand before analysis and preserving the existing
declaration → assignment → finalization handoff. This supplies metadata/recipes, not execution.

## Scope and non-goals

Extend existing Prepare validation and focused tests only; no new production type, public API,
package, dependency, graph, runner or interpreter. Model/Compiler retain body row-algebra proof
and AD admission. Prepare never reads length values, computes tensor results or allocates physical
buffers. CPU mixed-boundary binding, length preflight, skipping/execution, cold/v5/NN, fixed BPTT,
higher-order AD, low types, generated identities and performance work stay held.

## Contracts

Read AGENTS.md and root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), then:

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle), [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates),
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Runtime/Prepare/Engine [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)
  and [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle).
- Recurrent-scan [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities),
  [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand),
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings).
- [Length-aware whole-batch scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan), including
  [Header, bindings and source admission](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission),
  [Bounded row-role analysis](../../../../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories),
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates).

These are accepted authority, not new rules. Stop on missing/ambiguous authority or a required
scope expansion; do not invent a Prepare row-proof engine or widen backend capability.

## Dependencies and integration

- Depends on: Complete Model0037 authority, Model0038 source/IR, Compiler0013 capture/first-order AD,
  Planning0008 queries and Prepare0010 projection/resource handoff.
- Conflicts with: scan header/roles/demand, Prepare projection/resource lifecycle, Compiler/CPU/cold/
  codec/NN scan writers and shared-document writers; none active, serialize these scopes.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus all
  accepted uncommitted work at planning verification, 2026-10-24; predecessors are not merged commits.
- Integration order: accepted dependencies → Ready/MAIN admission → user implementation authorization
  → clean implementation → independent review and MAIN checkpoint → MAIN acceptance.
- Integration validation: named narrow Prepare0011 handoff checkpoint below, once after code freeze.
- Shared-document integration owner: MAIN; exclusive independent docs writer after worker freeze.

Planning verification (2026-10-24): predecessor Status/Result and current authority/source/tests agree:
[Model0037](../../model/tasks/0037-length-aware-general-scan-authority.md#result),
[Model0038](../../model/tasks/0038-length-aware-scan-source-foundation.md#result),
[Compiler0013](../../compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result),
[Planning0008](../../planning/tasks/0008-general-scan-body-capability-query.md#result) and
[Prepare0010](0010-general-scan-projection-and-resource-handoff.md#result) are Complete.
At verification, Model separated actual/formal counts, Compiler emitted final mixed roles/demand
and Prepare rejected length mode. MAIN launch admission 2026-10-24 12:25 UTC verified the Ready
brief/all five planning hashes unchanged, all five dependencies Complete, the same accepted base,
2432 Git files, index empty and authority/source matching the accepted snapshot. Accepted dirty
work was preserved; authorized frontier Prepare0011, no live conflicting writer/agent.
ImplementationAuthorized=true from the user's latest “pust se do prace” request, Prepare0011 only;
no downstream tasks, commit or push. Clean sole writer launched at 12:31 UTC.

## Files and symbols

Existing package prefix `io.github.pho001.synaptik.`; production root
`modules/prepare/src/main/java/`, matching tests under `modules/prepare/src/test/java/`:

- `prepare.analysis.PrepareContext#validateScan` — separate lengths and formal binding checks;
  `prepare.analysis.PartitionDag` — retain exact attachment and outer port occurrences.
- `prepare.GraphPreparation#Projection`, `project`, `prepareInternal` — complete-graph demand and
  all-context validation before the first analyzer.
- Existing `prepare.analysis.BackendPartitionAnalysis` / `PreparationResourceRequirement`,
  `prepare.PreparationResourceAssignment` / `BackendPartitionFinalizationHandoff` — retain staged
  declaration/assignment semantics; no new resource API or backend geometry policy.
- `analysis.PrepareContextTest`, `analysis.PartitionDagTest`, `GraphPreparationTest` and existing
  `BackendPartitionFinalizationHandoffTest` / `BackendPartitionFinalizationTest` — focused fixtures.
- Read-only handoff references: Compiler `GraphCompiler` final queries, `ScanBodyInference` demand,
  `LengthAwareScanBodyInferenceTest`, `LengthAwareScanGradientRulesTest`, `GeneralScanCompilerTest`;
  CPU `CpuGeneralScanLowering#analyze` and `CpuGeneralScanPreparedExecutable`; cold/codec guards below.

## Acceptance criteria

1. Use checked `formalCount=c+s+i`, `actualCount=formalCount+1` in length mode. Bind only the first
   formalCount actuals; keep the final lengths actual in the outer projection and logical-use checks,
   never in body formals or the outer gradient OR. All-valid counts/flags remain unchanged.
2. Validate static T/B, homogeneous F32/F64 body/formal types, non-gradient INT64 rank-one `[B]`
   lengths, leading ROW(0) carry/next-carry/emission B extents and declared sequence-axis extents,
   including unused roles. ROW(1) requires rank two; SHARED has no inferred batch axis. Check
   canonical formal/result layouts, exact positional dtype/shape/gradient bindings and checked geometry. Preserve outer
   actual layouts; do not import CPU read predicates or add a new canonical-lengths restriction.
   Invalid descriptors/roles, dynamic locals/siblings and overflow fail before any analyzer,
   including a bad later partition, T=0, B=0/1 and empty features. Never inspect length values.
3. Retain exact attrs/body/local graph, ordered next-carry/emission/residual bindings and demand;
   repeated actual ports and aliased roles stay positional. Local IDs resolve only inside their
   body, even when numerically equal to outer IDs or IDs in another reused-template occurrence.
   Validate boundary facts without re-deriving body numerical/dataflow semantics.
4. Project broader valid forward sources with SHARED carry suffixes and SHARED/ROW(1) sequences,
   and actual Compiler-generated reverse scans with leading row cotangents plus SHARED parameter
   accumulators and mixed ROW(0)/ROW(1)/SHARED saved-primal/seed sequences. Do not apply Compiler's
   restricted original-source AD admission to these projections. Test both dtypes/traversals,
   false-flag carry propagation, tied positions and pruned sequence roles using Compiler output.
   At F=B, shared transpose history stays `[T,F,H]`, not `[T,B,F,H]`; transposed row history keeps
   ROW(1). Preserve exact saved primal bindings, not recomputation or role inference from shape.
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
   metadata evidence does not claim physical history allocation or runtime skipping.
7. Preserve old all-valid source/Prepare acceptance beyond CPU's whitelist, canonical residual
   ordering/demand, old constructor/default paths and fixed scans. Real CPU pointwise/shaped
   admission, cold capture/reconstruction and direct codec rejection remain closed for length mode;
   v1–v4/default identities are unchanged. No dependency/build/backend/Runtime mutation is needed.

## Validation

JDK 26. Worker: focused iteration, then one final Prepare test/Javadoc gate:

```bash
./gradlew :modules:prepare:test --tests '*PrepareContextTest' --tests '*PartitionDagTest' --tests '*GraphPreparationTest' --tests '*BackendPartitionFinalizationHandoffTest' --tests '*BackendPartitionFinalizationTest'
./gradlew :modules:prepare:test :modules:prepare:javadoc
```

MAIN's narrow handoff checkpoint checks existing Prepare SPI consumers and retained CPU/cold/codec
rejection; reuse successful worker evidence, no duplicate module suite:

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
(`:extensions:checkpoint`). No root/native/performance campaign or execution/conformance claim;
later CPU capability owns its independent execution qualification. Stop if a new edge/contract or
non-Prepare executable change is needed. Worker does not run MAIN's checkpoint.

## Documentation and review impact

Independent Class C review finalizes Prepare Javadocs, factual root/scoped status, boundary/scan
explanations, partition-preparer guide, API references, targeted glossary entries and five plans.
It uses repo general/architecture/API/backend/planning styles and write-page/writing_quality for
surgical prose, without Pages. No new rule, ADR, term, API or owner is selected. Executable evidence
is reused; final Prepare Javadoc follows comment-only edits. MAIN retains overall acceptance.

## Result

Implemented `PrepareContext` static lengths/formal, dtype, geometry and row-boundary checks. Exact
mixed forward/actual-reverse roles, scoped IDs and ordered demand use the unchanged staged handoff.
Seven Java files changed: four production files (only `PrepareContext` executable) and the three
focused test classes above. Seventeen Markdown files reconcile factual status; no new type/API/edge.
Worker commands above, JDK 26.0.1, both scan performance flags false: focused 71 tests, full Prepare
92 tests/12 suites, zero failures/errors/skips; Javadoc PASS. Development wrapper-cache denial and
test-only initialized-constant fixture failure were resolved before these successful gates.
MAIN combined the SPI compile and four named guards above: PASS, 34 tasks, 7 executed/27 up-to-date.
Frozen 39-Java hashes matched; independent Class C code/docs APPROVE, zero open P1/P2. Review adds comments only,
reuses executable evidence, and runs final `:modules:prepare:javadoc` with the same flags: PASS,
9 tasks, 2 executed/7 up-to-date, no warnings. Normal escalated retry resolved the wrapper lock.
Documentation impact: root/scoped status, explanations, API/guide/glossary and Javadocs finalized.
17-file links/anchors/format, 39-file comment-only identity, scope/native/index and diff checks PASS.
Unresolved issues: none. No physical allocation, execution or skipping claim.
Status: Complete
MAIN accepted 2026-10-24 13:00:15 UTC. Follow-up required: none.
