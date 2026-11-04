# Task Compiler0013: Length-aware [Scan](../../../../glossary.md#term-cumulative-scan) [capture](../../../../glossary.md#term-capture-meanings) and first-order [AD](../../../../glossary.md#autograd)

> **Editorial reading key — not part of the historical record.**
>
> Length-aware inference sees an INT64 lengths descriptor, not its runtime contents. ROW(axis)
> explicitly names a batch-coordinate role; SHARED means no such coordinate, not merely an
> equal-looking shape. Compiler preserves these roles while constructing joint reverse expressions
> and normalizing shared contributions before accumulation. A symbolic history description
> allocates no history storage, and the historical rejection gates below distinguish compiler
> construction from later skipping, cold reconstruction and real execution.
>
> Current reading: [scoped Scan capture, roles and first-order differentiation](../../../../reference/compiler/verification-scan.md#current-scoped-scan-capture-and-first-order-ad).

## Status

Complete — MAIN accepted implementation, independent review and [validation](../../../../glossary.md#term-validation) on 2026-10-24.
ImplementationAuthorized=true. Code frozen; independent code/docs APPROVE, 0 open P1/P2.

Class C: scoped [identities](../../../../glossary.md#term-identity-meanings), saved primals, first-order [automatic differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)), query/demand.

## Goal and [scope](../../../../glossary.md#term-scope)

Deliver length-body [inference](../../../../glossary.md#term-inference), exact [primal](../../../../glossary.md#term-primal)/[cotangent](../../../../glossary.md#cotangent) [bindings](../../../../glossary.md#term-binding), reverse construction and combined
[capture](../../../../glossary.md#term-capture-meanings)/query/demand/tests/docs through existing [owners](../../../../glossary.md#term-owner); no execution or new type/[API](../../../../glossary.md#term-application-programming-interface-api)/[dependency](../../../../glossary.md#term-dependency).

## Non-goals

No execution/storage/skipping, cold/copy/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction)/v5, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) migration, [model](../../../../glossary.md#term-model)-wide [mask](../../../../glossary.md#term-boolean-mask)/[loss](../../../../glossary.md#loss) change,
fixed [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), higher-order waiver, low/[vector](../../../../glossary.md#term-vector)/parallel/[Metal](../../../../glossary.md#term-metal)/[OpenBLAS](../../../../glossary.md#term-openblas), performance campaign or
[runtime](../../../../glossary.md#term-runtime)/build/[dependency](../../../../glossary.md#term-dependency)/semantic-contract change. Factual documentation updates only.

## Contracts

Read AGENTS.md and root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>, then these exact headings:

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>, [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, [Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates),
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Recurrent-[scan](../../../../glossary.md#term-cumulative-scan) [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>:
  [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities),
  [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>,
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)<span data-glossary-navigation="true"> ([primal glossary](../../../../glossary.md#term-primal); [bindings glossary](../../../../glossary.md#term-binding))</span>,
  [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>,
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates).
- [Length-aware whole-batch scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>, ALL subheadings:
  [Header, bindings and source admission](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission)<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>,
  [Bounded row-role analysis](../../../../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>,
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [preflight glossary](../../../../glossary.md#term-preflight))</span>.
- [Compiler](../../../../glossary.md#term-compiler)/[autograd](../../../../glossary.md#autograd) [modules/compiler](../../../../architecture/contracts/compiler-autograd.md#modulescompiler)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>,
  [Compile artifacts](../../../../architecture/contracts/compiler-autograd.md#compile-artifacts)<span data-glossary-navigation="true"> ([Compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>, [Training graph model](../../../../architecture/contracts/compiler-autograd.md#training-graph-model)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph); [model glossary](../../../../glossary.md#term-model))</span>,
  [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>,
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>, including its bounded cold [guard](../../../../glossary.md#term-guard).
- Foundational [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel),
  [modules/planning](../../../../architecture/contracts/foundational-modules.md#modulesplanning)<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>, [Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries).

Status: [length plan](../../../length-aware-general-scan.md). Stop on contract ambiguity; plans cannot override authority.

## Dependencies and integration

- Depends on: Complete Model0037/0038, Compiler0011, Planning0008; preserve Compiler0012 cold baseline.
- Conflicts with: scan API/roles, Compiler capture/AD/demand/query/cold, Prepare/CPU/codec,
  NN0032 and shared-document writers; none active, serial only.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [AD glossary](../../../../glossary.md#autograd); [Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [codec glossary](../../../../glossary.md#term-codec))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus the
  accepted uncommitted tree (302 dirty paths / 2429 Git-listed files); not merged predecessors.
- Integration order: accepted plan → authorized clean executor → independent Class C code/docs
  review → MAIN checkpoint/acceptance.
- Integration [validation](../../../../glossary.md#term-validation): **Compiler0013 length-role compatibility [checkpoint](../../../../glossary.md#term-checkpoint)**, commands below.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; exclusive six-path [planning](../../../../glossary.md#term-planning) delegation here.

MAIN verified sole [Ready](../../../../glossary.md#term-ready) frontier/dependencies/conflicts and launched 2026-10-24 after user
authorization. Preserve accepted dirty base/native; no cascade or [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/push. Backup/evidence:
`/private/tmp/synaptik-compiler0013-implementation.ffc6iw/` (`accepted/`, `baseline.tar`).

## Files and symbols

Java prefix: `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/`. No production additions/moves.

- `ScanBodyInference.java`: infer/[capture](../../../../glossary.md#term-capture-meanings)/formals/validateResiduals/refreshDemand; independent local-ID roles.
- `ScanGradientRules.java`: preflight/apply/primalBindings/usedStep/usedSequences/formal/constants;
  positional derivatives, no lengths derivative.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [constants glossary](../../../../glossary.md#term-constant); [derivatives glossary](../../../../glossary.md#term-derivative))</span>
- `AutogradPreflight.java`: whole-original inventory, preflightBody/roles, pre-allocation rejection;
  `FirstOrderAutograd.java`: expandBody, normalization/accumulation, DerivativeConstants closed
  bindings; no registry/proof framework or new public API.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [registry glossary](../../../../glossary.md#term-registry); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- MAIN added `ReductionGradientRules.java` on 2026-10-24: primal-bound SUM/sumToShape restoration
  only, using checked lexical roles and ordinary row-leading layout edits; all-valid unchanged.<span data-glossary-navigation="true"> ([primal glossary](../../../../glossary.md#term-primal); [layout glossary](../../../../glossary.md#layout))</span>
- Existing [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse)/exact-rewrite [owners](../../../../glossary.md#term-owner) preserve length actual-[binding](../../../../glossary.md#term-binding) sites; no all-valid eligibility change.
- `GraphCapture.java`, `CapturedGraphInference.java`, `GraphCompiler.java`: scopes, validation,
  refresh-before-query and complete OperationCapabilityQuery; existing rebuild owners only as needed
  to preserve headers/attachments, no new optimization.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [OperationCapabilityQuery glossary](../../../../glossary.md#operation-capability-query--operationcapabilityquery))</span>
- Read-only seams: [Model](../../../../glossary.md#term-model) `Scan.java#apply/validateRows/rowResult`, `ScanAttrs`, `CompiledNode`,
  `CompiledScanBody`; [Compiler](../../../../glossary.md#term-compiler) `SourceGraphReconstruction.preflight` remains [fail-closed](../../../../glossary.md#term-fail-closed).
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
   Never read length storage/values or allocate physical state.<span data-glossary-navigation="true"> ([AD glossary](../../../../glossary.md#autograd); [INT64 glossary](../../../../glossary.md#data-type--datatype); [metadata glossary](../../../../glossary.md#term-metadata); [layouts glossary](../../../../glossary.md#layout); [carry glossary](../../../../glossary.md#term-carry); [gradient glossary](../../../../glossary.md#term-gradient))</span>
2. Preflight every original occurrence, including unrelated branches and unused/nongrad positions,
   before seeds/derivative IDs. Original length AD requires all carries ROW(0) and EVERY original
   sequence axis present zero. Preserve existing unsupported-body, residual-root/path/target and
   higher-order/createGraph guards. Generated mixed-role reverse inputs are not original sources.<span data-glossary-navigation="true"> ([Preflight glossary](../../../../glossary.md#term-preflight); [occurrence glossary](../../../../glossary.md#term-occurrence); [derivative glossary](../../../../glossary.md#term-derivative); [axis glossary](../../../../glossary.md#term-axis); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>
3. Reverse retains the same lengths actual, opposite traversal, joint final/emission seeds,
   false-flag carry flow, exact saved activation/previous-state primalBindings and closed constants.
   Use ordinary FirstOrderAutograd.expandBody/family formulas; no replay, numerical derivative
   rewrite or ValueId-to-Tensor AD map. No lengths cotangent; preserve compatible tied positional/
   root and cross-occurrence additions.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [constants glossary](../../../../glossary.md#term-constant); [ValueId glossary](../../../../glossary.md#logical-valueid-versus-physical-memory-slot); [Tensor glossary](../../../../glossary.md#tensor); [cotangent glossary](../../../../glossary.md#cotangent))</span>
4. Reverse carry has original row-cotangent prefix and invariant-shaped SHARED gradient suffix.
   Each suffix root is exactly its own old.add(contribution), matching shape/dtype, independent of
   all shared carry formals, with a distinct ADD site; no B×parameter bank or per-row accumulator.
   Normalize EACH incoming cotangent for an original SHARED value BEFORE contribution append/
   accumulation, including intermediates, not merely before fanout/propagation or final targets.
   Convert ROW via explicit batch-axis SUM, including B=1, preserving keepDimensions/exact shape,
   ordinary SUM/rounding and active contributors. Never re-reduce SHARED or infer roles from target
   shape; equal-shape sumToShape cannot erase ROW. No mixed ROW+terminal-SHARED accumulation.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant); [dtype glossary](../../../../glossary.md#data-type--datatype); [parameter glossary](../../../../glossary.md#term-parameter); [accumulator glossary](../../../../glossary.md#accumulator-type); [rounding glossary](../../../../glossary.md#term-rounding))</span>
5. Disconnected row carry/sequence cotangents bind exact typed positive-zero constants through
   explicit ROW seed/sequence declarations, not SHARED expand inventing ROW. Cover nextCarry=xStep
   (disconnected carry), unused step (disconnected sequence) and false flags with poisoned unused
   primals/seeds. No 0*x, compute-masking, new operation or extra source/AD capability rejection.
   Both closure requirements implement accepted semantics; ordinary all-valid AD remains unchanged.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [operation glossary](../../../../glossary.md#operation))</span>
6. Derive roles from checked bindings/dataflow for original sequences, emission/zero seeds and
   retained primals. Prune axes IN LOCKSTEP with step formals/actuals, including the time binder.
   Shared w-transpose history stays [T,F,H], row seed [T,B,H], transposed row history ROW(1);
   never infer from B=F, rank or flags.
   Roles survive combined capture, canonicalization/rewrites/DCE/CSE, demand refresh and final query.
   Reused templates have independent scopes; saved sites and original wrappers/port order persist.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [transpose glossary](../../../../glossary.md#term-transpose); [rank glossary](../../../../glossary.md#term-rank); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
7. Only ordinary publication/use demands history; forward/reverse dead residuals stay undemanded.
   Final queries carry complete attrs/body/ordered descriptors/demand, lengths and no extra formal.
   Test accepting spy and binder-rejecting providers: child support cannot grant ownership.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [descriptors glossary](../../../../glossary.md#term-descriptor); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
8. Cold stays closed: captureSource → GraphCapture.captureSource → SourceCapture constructor →
   SourceGraphReconstruction.preflight; reconstructSource → reconstruct → same preflight. Preserve
   test BOTH public paths/direct SourceCapture and late/dead/T=0 raw fixtures: explicit rejection,
   no reconstruction IDs/storage copy. General inference success is not cold success. Prepare,
   real CPU provider/lowering (pointwise AND shaped), and direct codec still reject.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [pointwise glossary](../../../../glossary.md#term-pointwise); [codec glossary](../../../../glossary.md#term-codec))</span>

[Matrix](../../../../glossary.md#term-matrix-meanings): [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding), both traversals, disabled/standard optimization; inspect actual generated bodies.

| Case | Required evidence in this task |
|---|---|
| B=F/1/0; T=0; empty [features](../../../../glossary.md#term-software-feature) | Roles/history geometry, whole empty-body checks, O(body) [graph](../../../../glossary.md#graph) size. |
| B=1; q=a.matmul(w) SHARED[1,H]; xStep ROW0[1,1]; next=h.add(q).add(xStep.matmul(q)) | q gets direct ROW and terminal-SHARED [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) terms: keepDims [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) SUM the incoming ROW BEFORE append/accumulation, keep SHARED unchanged; upstream a/w adjoints remain SHARED, no [target](../../../../glossary.md#term-target)-[shape](../../../../glossary.md#shape) [heuristic](../../../../glossary.md#term-heuristic). |
| Final/emission/joint/repeated [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); two carries | primalBindings, positional sums, zero/one [bindings](../../../../glossary.md#term-binding), false flags. |
| Unused/nongrad sequences; ties | Check all positions; ROW zeros and lockstep pruning. |
| Malformed roles/lengths/feedback; unrelated bad [scan](../../../../glossary.md#term-cumulative-scan) | Independent [inference](../../../../glossary.md#term-inference)/pre-allocation failures, no new rejection policy. |

## [Validation](../../../../glossary.md#term-validation)

Worker focused iteration, then ONE final [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc) gate:

```bash
./gradlew :modules:compiler:test --tests '*ScanBodyInferenceTest' --tests '*ScanGradientRulesTest' --tests '*GeneralScanCompilerTest' --tests '*GraphCaptureTest' --tests '*SourceGraphReconstructionTest' --tests '*FirstOrderGradientCoverageTest'
./gradlew :modules:compiler:test :modules:compiler:javadoc :modules:model:javadoc
```

MAIN bounded compatibility [checkpoint](../../../../glossary.md#term-checkpoint) (reuse worker evidence; no duplicate successful suites):

```bash
./gradlew :modules:planning:compileTestJava :modules:prepare:compileTestJava :backends:cpu:compileTestJava :extensions:checkpoint:compileTestJava
./gradlew :modules:prepare:test --tests '*PrepareContextTest' :backends:cpu:test --tests '*CpuGeneralScanCapabilityTest' :extensions:checkpoint:test --tests '*GraphArtifactCodecTest' --tests '*GraphArtifactV4Test'
git diff --check
```

Mappings: [CPU](../../../../glossary.md#term-cpu) `modules/backend/cpu` (:[backends](../../../../glossary.md#backend):cpu), Checkpoint `modules/checkpoint`
(:extensions:checkpoint). Report non-[Compiler](../../../../glossary.md#term-compiler) executable/[guard](../../../../glossary.md#term-guard) edits before expanding [scope](../../../../glossary.md#term-scope).
Evidence proves construction/roles and retained rejection, not [runtime](../../../../glossary.md#term-runtime) length validation/skipping,
poisoned-[padding](../../../../glossary.md#padding) execution or speed. [Oracle](../../../../glossary.md#term-test-oracle).[scan](../../../../glossary.md#term-cumulative-scan) remains all-valid. No root/native/performance gate;
reuse source-matched evidence. Execution/[conformance](../../../../glossary.md#term-conformance) join belongs to later [owners](../../../../glossary.md#term-owner).

## Documentation and review impact

Finalize [Compiler](../../../../glossary.md#term-compiler)/[Model](../../../../glossary.md#term-model) [Scan](../../../../glossary.md#term-cumulative-scan) Javadocs, root/scoped facts, three [API](../../../../glossary.md#term-application-programming-interface-api) references, scan/training/[module](../../../../glossary.md#term-module)
explanations, glossary and six active plans. Keep cold/[Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu)/[codec](../../../../glossary.md#term-codec)/v5/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) closed; no API/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/
authority/[dependency](../../../../glossary.md#term-dependency) redesign. Independent review reuses tests; MAIN integrates.

## [Planning](../../../../glossary.md#term-planning) review

Popper APPROVE, 0 P1/P2; MAIN accepted planning/[Ready](../../../../glossary.md#term-ready) 2026-10-24.

## Result

Status: Complete
Delivered scoped length [inference](../../../../glossary.md#term-inference)/first-order reverse/query/demand. All eight criteria source/test-reviewed;
code APPROVE, zero open P1/P2. R1 restoration, typed ROW literal/zero closure and before-append
SHARED [normalization](../../../../glossary.md#term-normalization) preserve ordinary formulas and all-valid defaults.
Worker focused: 80 tests / 9 suites, zero failures/errors/skips.
Worker [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc): 362 tests / 48 suites, zero failures/errors/skips; Javadocs PASS.
MAIN compatibility [compile](../../../../glossary.md#compile)/guards PASS: [Prepare](../../../../glossary.md#prepare) 12 + [CPU](../../../../glossary.md#term-cpu) 18 + [Checkpoint](../../../../glossary.md#term-checkpoint) 21 = 51,
zero failures/errors/skips; 28 tasks, 8 executed/20 up-to-date, both performance flags false.
[Logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings): `worker-result.txt` and `compatibility.log` in the evidence directory above.
Independent docs/links/anchors/comment-only/whitespace PASS; final [Compiler](../../../../glossary.md#term-compiler)/[Model](../../../../glossary.md#term-model) Javadocs PASS
(8 tasks, 4 executed). Tests reused, no reruns. Review APPROVE, zero open P1/P2.
Evidence proves construction/roles/query and retained rejection, not [runtime](../../../../glossary.md#term-runtime) values/skipping,
poisoned-[padding](../../../../glossary.md#padding) execution or performance. Cold/Prepare/CPU/[codec](../../../../glossary.md#term-codec)/v5/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) remain closed.
MAIN accepted overall completion 2026-10-24. [Ready](../../../../glossary.md#term-ready) frontier None; no successor/cascade.
