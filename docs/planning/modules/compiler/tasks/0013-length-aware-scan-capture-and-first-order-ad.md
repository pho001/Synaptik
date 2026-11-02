# Task Compiler0013: Length-aware Scan capture and first-order AD

## Status

Complete — MAIN accepted implementation, independent review and validation on 2026-10-24.
ImplementationAuthorized=true. Code frozen; independent code/docs APPROVE, 0 open P1/P2.

Class C: scoped identities, saved primals, first-order automatic differentiation (AD), query/demand.

## Goal and scope

Deliver length-body inference, exact primal/cotangent bindings, reverse construction and combined
capture/query/demand/tests/docs through existing owners; no execution or new type/API/dependency.

## Non-goals

No execution/storage/skipping, cold/copy/rebind/v5, NN/RNN migration, model-wide mask/loss change,
fixed BPTT, higher-order waiver, low/vector/parallel/Metal/OpenBLAS, performance campaign or
runtime/build/dependency/semantic-contract change. Factual documentation updates only.

## Contracts

Read AGENTS.md and root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), then these exact headings:

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle), [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates),
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Recurrent-scan [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities),
  [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand),
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings),
  [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification),
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates).
- [Length-aware whole-batch scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan), ALL subheadings:
  [Header, bindings and source admission](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission),
  [Bounded row-role analysis](../../../../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories),
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates).
- Compiler/autograd [modules/compiler](../../../../architecture/contracts/compiler-autograd.md#modulescompiler),
  [Compile artifacts](../../../../architecture/contracts/compiler-autograd.md#compile-artifacts), [Training graph model](../../../../architecture/contracts/compiler-autograd.md#training-graph-model),
  [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation),
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle), including its bounded cold guard.
- Foundational [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel),
  [modules/planning](../../../../architecture/contracts/foundational-modules.md#modulesplanning), [Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries).

Status: [length plan](../../../length-aware-general-scan.md). Stop on contract ambiguity; plans cannot override authority.

## Dependencies and integration

- Depends on: Complete Model0037/0038, Compiler0011, Planning0008; preserve Compiler0012 cold baseline.
- Conflicts with: scan API/roles, Compiler capture/AD/demand/query/cold, Prepare/CPU/codec,
  NN0032 and shared-document writers; none active, serial only.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus the
  accepted uncommitted tree (302 dirty paths / 2429 Git-listed files); not merged predecessors.
- Integration order: accepted plan → authorized clean executor → independent Class C code/docs
  review → MAIN checkpoint/acceptance.
- Integration validation: **Compiler0013 length-role compatibility checkpoint**, commands below.
- Shared-document integration owner: MAIN; exclusive six-path planning delegation here.

MAIN verified sole Ready frontier/dependencies/conflicts and launched 2026-10-24 after user
authorization. Preserve accepted dirty base/native; no cascade or commits/push. Backup/evidence:
`/private/tmp/synaptik-compiler0013-implementation.ffc6iw/` (`accepted/`, `baseline.tar`).

## Files and symbols

Java prefix: `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/`. No production additions/moves.

- `ScanBodyInference.java`: infer/capture/formals/validateResiduals/refreshDemand; independent local-ID roles.
- `ScanGradientRules.java`: preflight/apply/primalBindings/usedStep/usedSequences/formal/constants;
  positional derivatives, no lengths derivative.
- `AutogradPreflight.java`: whole-original inventory, preflightBody/roles, pre-allocation rejection;
  `FirstOrderAutograd.java`: expandBody, normalization/accumulation, DerivativeConstants closed
  bindings; no registry/proof framework or new public API.
- MAIN added `ReductionGradientRules.java` on 2026-10-24: primal-bound SUM/sumToShape restoration
  only, using checked lexical roles and ordinary row-leading layout edits; all-valid unchanged.
- Existing CSE/exact-rewrite owners preserve length actual-binding sites; no all-valid eligibility change.
- `GraphCapture.java`, `CapturedGraphInference.java`, `GraphCompiler.java`: scopes, validation,
  refresh-before-query and complete OperationCapabilityQuery; existing rebuild owners only as needed
  to preserve headers/attachments, no new optimization.
- Read-only seams: Model `Scan.java#apply/validateRows/rowResult`, `ScanAttrs`, `CompiledNode`,
  `CompiledScanBody`; Compiler `SourceGraphReconstruction.preflight` remains fail-closed.
- Focused tests at `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/`:
  `ScanBodyInferenceTest`, `ScanGradientRulesTest`, `GeneralScanCompilerTest`; affected guards in
  `GraphCaptureTest`, `SourceGraphReconstructionTest`, `FirstOrderGradientCoverageTest`; added
  `LengthAwareScanBodyInferenceTest` and `LengthAwareScanGradientRulesTest` (tests only).

## Acceptance criteria

1. Independent inference admits the accepted length subset, including forward SHARED suffix carries
   and ROW(1)/SHARED sequences outside original-source AD. Compiled fixtures bypass Scan.apply:
   recheck lengths INT64[B]/nongrad/static metadata, common T/B, roles/formals/flags, checked counts/
   layouts, whole bodies/siblings, compatible tied-actual roles, terminal-reduction taint, exact
   shared contribution shape and carry use. Reject malformed T=0/B=0 cases without index failures.
   Outer count is c+s+i+1; formal count c+s+i; lengths has no formal and is excluded from gradient OR.
   Never read length storage/values or allocate physical state.
2. Preflight every original occurrence, including unrelated branches and unused/nongrad positions,
   before seeds/derivative IDs. Original length AD requires all carries ROW(0) and EVERY original
   sequence axis present zero. Preserve existing unsupported-body, residual-root/path/target and
   higher-order/createGraph guards. Generated mixed-role reverse inputs are not original sources.
3. Reverse retains the same lengths actual, opposite traversal, joint final/emission seeds,
   false-flag carry flow, exact saved activation/previous-state primalBindings and closed constants.
   Use ordinary FirstOrderAutograd.expandBody/family formulas; no replay, numerical derivative
   rewrite or ValueId-to-Tensor AD map. No lengths cotangent; preserve compatible tied positional/
   root and cross-occurrence additions.
4. Reverse carry has original row-cotangent prefix and invariant-shaped SHARED gradient suffix.
   Each suffix root is exactly its own old.add(contribution), matching shape/dtype, independent of
   all shared carry formals, with a distinct ADD site; no B×parameter bank or per-row accumulator.
   Normalize EACH incoming cotangent for an original SHARED value BEFORE contribution append/
   accumulation, including intermediates, not merely before fanout/propagation or final targets.
   Convert ROW via explicit batch-axis SUM, including B=1, preserving keepDimensions/exact shape,
   ordinary SUM/rounding and active contributors. Never re-reduce SHARED or infer roles from target
   shape; equal-shape sumToShape cannot erase ROW. No mixed ROW+terminal-SHARED accumulation.
5. Disconnected row carry/sequence cotangents bind exact typed positive-zero constants through
   explicit ROW seed/sequence declarations, not SHARED expand inventing ROW. Cover nextCarry=xStep
   (disconnected carry), unused step (disconnected sequence) and false flags with poisoned unused
   primals/seeds. No 0*x, compute-masking, new operation or extra source/AD capability rejection.
   Both closure requirements implement accepted semantics; ordinary all-valid AD remains unchanged.
6. Derive roles from checked bindings/dataflow for original sequences, emission/zero seeds and
   retained primals. Prune axes IN LOCKSTEP with step formals/actuals, including the time binder.
   Shared w-transpose history stays [T,F,H], row seed [T,B,H], transposed row history ROW(1);
   never infer from B=F, rank or flags.
   Roles survive combined capture, canonicalization/rewrites/DCE/CSE, demand refresh and final query.
   Reused templates have independent scopes; saved sites and original wrappers/port order persist.
7. Only ordinary publication/use demands history; forward/reverse dead residuals stay undemanded.
   Final queries carry complete attrs/body/ordered descriptors/demand, lengths and no extra formal.
   Test accepting spy and binder-rejecting providers: child support cannot grant ownership.
8. Cold stays closed: captureSource → GraphCapture.captureSource → SourceCapture constructor →
   SourceGraphReconstruction.preflight; reconstructSource → reconstruct → same preflight. Preserve
   test BOTH public paths/direct SourceCapture and late/dead/T=0 raw fixtures: explicit rejection,
   no reconstruction IDs/storage copy. General inference success is not cold success. Prepare,
   real CPU provider/lowering (pointwise AND shaped), and direct codec still reject.

Matrix: F32/F64, both traversals, disabled/standard optimization; inspect actual generated bodies.

| Case | Required evidence in this task |
|---|---|
| B=F/1/0; T=0; empty features | Roles/history geometry, whole empty-body checks, O(body) graph size. |
| B=1; q=a.matmul(w) SHARED[1,H]; xStep ROW0[1,1]; next=h.add(q).add(xStep.matmul(q)) | q gets direct ROW and terminal-SHARED MATMUL terms: keepDims batch SUM the incoming ROW BEFORE append/accumulation, keep SHARED unchanged; upstream a/w adjoints remain SHARED, no target-shape heuristic. |
| Final/emission/joint/repeated seeds; two carries | primalBindings, positional sums, zero/one bindings, false flags. |
| Unused/nongrad sequences; ties | Check all positions; ROW zeros and lockstep pruning. |
| Malformed roles/lengths/feedback; unrelated bad scan | Independent inference/pre-allocation failures, no new rejection policy. |

## Validation

Worker focused iteration, then ONE final module/Javadoc gate:

```bash
./gradlew :modules:compiler:test --tests '*ScanBodyInferenceTest' --tests '*ScanGradientRulesTest' --tests '*GeneralScanCompilerTest' --tests '*GraphCaptureTest' --tests '*SourceGraphReconstructionTest' --tests '*FirstOrderGradientCoverageTest'
./gradlew :modules:compiler:test :modules:compiler:javadoc :modules:model:javadoc
```

MAIN bounded compatibility checkpoint (reuse worker evidence; no duplicate successful suites):

```bash
./gradlew :modules:planning:compileTestJava :modules:prepare:compileTestJava :backends:cpu:compileTestJava :extensions:checkpoint:compileTestJava
./gradlew :modules:prepare:test --tests '*PrepareContextTest' :backends:cpu:test --tests '*CpuGeneralScanCapabilityTest' :extensions:checkpoint:test --tests '*GraphArtifactCodecTest' --tests '*GraphArtifactV4Test'
git diff --check
```

Mappings: CPU `modules/backend/cpu` (:backends:cpu), Checkpoint `modules/checkpoint`
(:extensions:checkpoint). Report non-Compiler executable/guard edits before expanding scope.
Evidence proves construction/roles and retained rejection, not runtime length validation/skipping,
poisoned-padding execution or speed. Oracle.scan remains all-valid. No root/native/performance gate;
reuse source-matched evidence. Execution/conformance join belongs to later owners.

## Documentation and review impact

Finalize Compiler/Model Scan Javadocs, root/scoped facts, three API references, scan/training/module
explanations, glossary and six active plans. Keep cold/Prepare/CPU/codec/v5/NN closed; no API/ADR/
authority/dependency redesign. Independent review reuses tests; MAIN integrates.

## Planning review

Popper APPROVE, 0 P1/P2; MAIN accepted planning/Ready 2026-10-24.

## Result

Status: Complete
Delivered scoped length inference/first-order reverse/query/demand. All eight criteria source/test-reviewed;
code APPROVE, zero open P1/P2. R1 restoration, typed ROW literal/zero closure and before-append
SHARED normalization preserve ordinary formulas and all-valid defaults.
Worker focused: 80 tests / 9 suites, zero failures/errors/skips.
Worker module/Javadoc: 362 tests / 48 suites, zero failures/errors/skips; Javadocs PASS.
MAIN compatibility compile/guards PASS: Prepare 12 + CPU 18 + Checkpoint 21 = 51,
zero failures/errors/skips; 28 tasks, 8 executed/20 up-to-date, both performance flags false.
Logs: `worker-result.txt` and `compatibility.log` in the evidence directory above.
Independent docs/links/anchors/comment-only/whitespace PASS; final Compiler/Model Javadocs PASS
(8 tasks, 4 executed). Tests reused, no reruns. Review APPROVE, zero open P1/P2.
Evidence proves construction/roles/query and retained rejection, not runtime values/skipping,
poisoned-padding execution or performance. Cold/Prepare/CPU/codec/v5/NN remain closed.
MAIN accepted overall completion 2026-10-24. Ready frontier None; no successor/cascade.
