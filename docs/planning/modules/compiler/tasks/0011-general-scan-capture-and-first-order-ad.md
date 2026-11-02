# Task Compiler0011: General scan capture and first-order AD

## Status

Complete; accepted by main 2026-10-22 after independent Class C code/docs APPROVE and compatibility checkpoint.
ImplementationAuthorized=true; user-authorized 2026-10-22 (“ok, pust se do toho”); launch admission verified.

## Change class

Class C — scoped identity, pre-capture automatic differentiation (AD), saved-value bindings and demand.
Clean executor and independent targeted code/docs review required.

## Goal

Compile scoped Scan bodies and first-order cotangents (reverse sensitivities) using existing rules:
forward scan → demanded residuals (saved primal values) → opposite-traversal reverse sibling →
one combined capture. Deliver validated Tensor-free bodies/demand, not execution.

## Scope

Capture/inference, AD, rule bindings, attachment-preserving transforms and final query demand in all
compile modes. Reuse completed Model0035/Planning0008 foundations.
Only new package-private production types: `io.github.pho001.synaptik.compiler.ScanBodyInference` and
`ScanGradientRules` in the same root package; no other production type/package.

## Non-goals

No authority/API/dependency/build changes, second AD algebra, registry, IR AD, executor/tape/storage,
Prepare/CPU implementation or backend/performance claim. No fixed RNN/GRU/LSTM backpropagation through
time (BPTT), cold captureSource/reconstructSource or save/load widening, NN training/lengths/padding,
if/while/nesting/regions. Preserve fixed four-dtype/read-layout/static routes; low backend qualification is later.

## Contracts

Read `AGENTS.md`, affected source/tests, root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), then only these headings:

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle), [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- compiler-autograd: [`modules/compiler`](../../../../architecture/contracts/compiler-autograd.md#modulescompiler),
  [Compile artifacts](../../../../architecture/contracts/compiler-autograd.md#compile-artifacts),
  [Training graph model](../../../../architecture/contracts/compiler-autograd.md#training-graph-model),
  [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation),
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle).
- recurrent-scan: [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan) and all five subheadings:
  [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities),
  [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand),
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings),
  [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification),
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates).
- foundational-modules: [`modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel),
  [`modules/planning`](../../../../architecture/contracts/foundational-modules.md#modulesplanning) and
  [Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries), only source/IR and query boundaries.

Stop and report any missing/ambiguous contract or required extra type/boundary; do not invent architecture.

## Dependencies and integration

- Depends on: Model0034, Model0035, Planning0008 Complete; Compiler0006A/0006D/0008/0009 Complete baseline.
- Conflicts with: capture/inference/AD/closure/query/shared-scan writers; none active.
  Prepare/CPU/NN/persistence remains inactive. Serial, no concurrent writer.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus ALL accepted
  dirty changes, including Model0034/0035 and Planning0008, not merged predecessors.
- Integration order: frozen plan → independent planning review/main acceptance → explicit executor
  authorization → clean Compiler implementation → independent code/docs APPROVE → main compatibility
  checkpoint/acceptance; Prepare/CPU/NN stay None/Draft.
- Integration validation: **Compiler0011 scoped-body compatibility checkpoint**, commands below;
  native root test/Javadoc gate at low-level join.
- Shared-document integration owner: main (plans/status); executor: Compiler code/tests and
  explanatory/Javadoc edits; planning delegated.

Main verified BEFORE launch, 2026-10-22: brief Ready, sole authorized frontier; all named dependencies
Complete with accepted results/gates; Galileo plan APPROVE, zero findings, main acceptance today.
HEAD=origin/main as above; index empty; conflicts inactive; no other managed writer/test. All 147
accepted dirty files match final freeze (none new/missing/changed); 119 source/test/protected files
unchanged, including THIRD_PARTY_NOTICES.md, CPU cast oracle and Superseded Compiler0010.
Authorization preceded launch; implementation, independent review and main checkpoint are complete; no successor Ready.

## Files and symbols

Paths use `modules/compiler/src/{main,test}/java/io/github/pho001/synaptik/compiler/`.

- New `ScanBodyInference.java`: local inference and formal/role/outer descriptors.
  New `ScanGradientRules.java`: body inventory, joint-output rules and reverse Scan.
- `GraphCapture.java` (`captureInternal`, `captureCombined`), `CapturedGraphInference.java`
  (`inferAndValidate`, dispatch): unused formals, lexical scope/escape checks and attachments.
- `AutogradPreflight.java` (`inventory`, `preflight`, `preflightStage`, `validateOccurrence`, `rejectForwardOnlyOccurrences`),
  `FirstOrderGradientCoverage.java` (`classify`), `FirstOrderAutograd.java` (`expand`, `apply`,
  `DerivativeConstants`): preflight, joint seeds, positional accumulation, closed constants.
- `ElementwiseGradientRules.java`, `LinearAlgebraGradientRules.java`, `LayoutGradientRules.java`,
  `ReductionGradientRules.java` (`apply`/helpers): primal binding; non-scan callers retain identity binding/formulas/guards.
- `GraphCanonicalization.java` (`rebuild`), `ForwardExactArithmeticRewriting.java`, `ForwardConstantFolding.java`,
  `ForwardDeadCodeElimination.java`, `ForwardCommonSubexpressionElimination.java`,
  `ForwardGraphOptimization.java`, `StaticResultLogicalLayoutClosure.java`: preserve/revalidate body,
  role/demand facts and saved sites in every rebuild branch; no new rewrite/fusion implementation.
- `GraphCompiler.java` (`compile`, query construction), `GraphCompilation.java`, `GraphCompilationPort.java`:
  one graph, preserved publication/caller identities, final body-bearing queries (currently three-argument).
- Read-only foundations: Model `tensor/{ScanBody,Scan,ScanResult,TensorProducer}.java`,
  `graph/{CompiledNode,CompiledScanBody}.java`; Planning `capability/OperationCapabilityQuery.java`,
  `memory/LogicalMemoryPlanning.java`. No foundation API/algorithm changes.
- New focused tests `ScanBodyInferenceTest.java`, `ScanGradientRulesTest.java`, `GeneralScanCompilerTest.java`;
  extend existing capture/inference/preflight/gradient/graph-optimization/layout/compiler/port tests as affected.

## Acceptance criteria

1. Capture/inference admits valid PURE Model source independently of the CPU whitelist (FLOAT16 TANH,
   F32 RELU included), without AD/backend qualification. Preserve wrappers, retained order/siblings, unused
   declared formals and repeatable roles over unique local outputs. Reused templates get independent
   scope-qualified IDs; reject escape/sharing across different templates, not reuse. No outer IDs inward.
2. Inventory exact operation/attrs/role/Shape/dtype support and the entire generated derivative closure.
   Required baseline: homogeneous F32/F64 static canonical boundaries/algebra below, not all Model sources. Test actual
   generated kinds/attrs; reject unsupported requirements before derivative/seed/formal/Scan allocation.

   | Source/rule owner | Required formula/closure inventory |
   |---|---|
   | Tensor/scalar ADD/SUB/MUL/DIV; Elementwise | Existing NEG/product/DIV formulas, broadcast `sumToShape`; no new masks/numerics. |
   | NEG/TANH/SIGMOID; Elementwise | NEG; saved output, MUL/SUB, typed one/expand; no replay. |
   | Rank-two MATMUL; LinearAlgebra | Both positional operands, `[1,0]` transposes, MATMUL, shape/type normalization. |
   | Rank-two `[1,0]` PERMUTE; Layout | Inverse permutation; generated EXPAND/RESHAPE/CONTIGUOUS and required rank-editing views. |
   | Generated reductions/constants; Reduction + existing owners | SUM/`SumToShapeAttrs`, broadcast axes/keep-dim restoration, expand/views, required floating casts, registered typed zero/one splats. |

3. Every formula primal resolves via an explicit compile-local original-Tensor → residual/step/invariant
   reverse-formal function; contribution keys stay original identities. Carry/body primals use original
   residual slots; step/invariants use ordered explicit inputs. Canonical TANH/SIGMOID saved
   outputs resolve to residual formals, never replay. oneLike constants close through non-gradient
   invariant formals/registered outer splats; no free leaves or AD ValueId→Tensor map.
4. Replace per-output-slot scan dispatch with ONE joint final-carry/emission reverse Scan per occurrence,
   opposite traversal, before capture. Missing internal port seeds/parameter-sum initializers are typed zeros;
   public seed guards unchanged. Both seed classes coexist. dSequences/dInitialCarry/invariant sums and
   tied roles/repeated roots/cross-occurrence terms accumulate positionally without deduplication.
   Internal carry cotangents traverse even when original h0/formal/body values have requiresGrad=false;
   adapt local preflightStage/validateOccurrence without changing flags/public guards. Preserve outer G (OR of actual flags).
5. Preflight rejects residual-root differentiation (including paths through residuals), every scan higher-order/
   createGraph request and unsupported body occurrences anywhere in the original forward inventory, including
   an unrelated sibling branch. Pin unchanged Tensor-ID counter on known rejections; no partial derivative work.
   Non-scan higher-order support and fixed-scan/Conv3d rejection stay unchanged.
6. One phase-aware capture retains original wrappers, independent bodies, publication/caller TensorIds and
   derivative order. Every four-argument CompiledNode rebuild must retain its five-argument body,
   in EVERY branch; no CSE across scans/scopes/live saved sites. Test each pass, optimization on/off,
   repeated-template occurrences and rebuilds triggered by a neighboring ordinary rewrite.
7. ForwardDCE.eliminateGraph sibling-liveness is NOT port demand. Derive demandedOutputIndices from final
   outer publication/ordinary uses AFTER rewrites/layout closure, not siblings/internal uses/latent body roots.
   Revalidate each changed candidate. Dead forward/reverse histories impose no history demand/fusion restriction.
   Keep live saved-site identity/rounding boundaries through closure, reindexing, DCE/CSE and future fusion handoff.
   Logical value records are not storage; bytes/stores are later Prepare/CPU acceptance.
8. Final OperationCapabilityQuery carries validated complete body/roles/descriptors/demand; no child-only
   ownership shortcut/inner mixed-owner partitioning. Test providers prove closure delivery and unsupported
   binder rejection. Ordinary/fixed queries stay empty-body; real capability/cold source admission unchanged.
9. Tests cover two carries, both seed classes, shared parameters, tied actual roles and occurrences/template reuse,
   closed constants, exact saved TANH/SIGMOID bindings/special values, broadcasts, DIV zero/NaN/Inf/signed zero,
   MATMUL/transpose and generated closure. Compare at least three steps against unrolled gradients with zero
   non-gradient h0 and trainable w/u, in both traversal directions, using a bounded test-only oracle (no production runner).
   Check original-time indices and previous-carry-before-step. T=0 has exact initial final carry,
   dInitialCarry=dFinalCarry, empty emissions/residuals and zero invariant sums. Empty axes retain constituent
   reduction semantics and never bypass unsupported preflight. Vary T to prove O(body), not T-fold graph growth.

## Validation

Recorded validation, repository root (outcomes in Result):

```bash
./gradlew :modules:compiler:test --tests '*ScanBodyInferenceTest' --tests '*ScanGradientRulesTest' --tests '*GeneralScanCompilerTest'
./gradlew :modules:compiler:test :modules:compiler:javadoc
```

One final module run after executable stabilization; regenerate Javadoc after final comment edits.
Main's **Compiler0011 scoped-body compatibility checkpoint**, once after independent code/docs approval:

```bash
./gradlew :modules:prepare:compileTestJava :backends:cpu:compileTestJava :backends:metal:compileTestJava :modules:engine:compileTestJava :extensions:checkpoint:compileTestJava
./gradlew :testing:architecture-tests:test
./gradlew :backends:cpu:test --tests '*CpuCapabilityProviderTest' --tests '*CpuCapabilityProviderPublicShapeTest' :backends:metal:test --tests '*MetalCapabilityProviderTest'
git diff --check
```

Check local links/anchors and new-file whitespace. No Gradle/benchmark during planning.
No root/native/five-fork/performance campaign. Scan conformance/integration and native root gate are
later CPU/join work. Review reuses evidence.

## Documentation and review impact

Implementation/review updated Javadoc (bindings, scope, flags, demand/rejections, params/results/failures)
and targeted `docs/architecture/general-differentiable-scan.md`, `docs/architecture/training-graph.md`,
`docs/api/compile-api.md`; check `docs/api/public-api.md` and glossary scan/primal/cotangent/residual entries for status drift.
Applied documentation-rules/general/planning and API/Javadoc/architecture profiles to corresponding edits.
Independent code/docs review checked diff/tests/contracts: closure, false-flag carry, saved activation
identity/constants, scope/demand. No new term/authority; shared/authority status edits need main delegation.
Planning changes no Java/Javadoc/capability.

## Result

Status: Complete; main acceptance 2026-10-22 after Kant implementation, Euclid final independent Class C code/docs APPROVE (zero P1/P2) and compatibility checkpoint.
Scope complete: scoped capture/inference, one joint reverse Scan via ordinary AD/exact primal bindings and closed constants, non-gradient carry flow, whole-original preflight/Tensor-allocation-free rejections, attachments/exact saved boundaries and final demand/body queries.
Changed 25 Compiler Java files: two new package-private production owners, three new scan tests, existing owners/passes/GraphCompilation[Port]/GraphCapture and two old tests; full file/symbol scope above.
Numeric oracle: three-step unrolled existing gradients, F32/F64, both directions, two carries/joint seeds, broadcast and MATMUL/transpose; T=0, previous carry/original time, ties/reuse, DIV specials, exact saved TANH/SIGMOID, T=0/3/100000 graph size and Tensor-ID rejection guard.
Worker focused: `./gradlew :modules:compiler:test --tests '*GraphCaptureTest' --tests '*ScanBodyInferenceTest' --tests '*ScanGradientRulesTest' --tests '*GeneralScanCompilerTest' --tests '*FirstOrderGradientCoverageTest'` — PASS 47/47.
Final `./gradlew :modules:compiler:test :modules:compiler:javadoc` — PASS 340 tests/46 suites, zero failures/errors/skips; final comment-only Javadoc regeneration PASS, zero warnings.
Euclid reviewed 25 Java files: 24 byte-identical; only ScanGradientRules.java Javadoc changed to “Tensor-allocation-free” (SHA-256 `2d6962add1c3eda3a2534efff63ac8fe106846ba3be8b7f97ca00ae2c7ca168d`). No executable change; reused 340 tests; own final `./gradlew :modules:compiler:javadoc` PASS, zero warnings.
Review finalized 11 factual-status/explanation docs: ARCHITECTURE; compiler-autograd/recurrent-scan/foundational-modules contracts; general-differentiable-scan/training-graph/module-boundaries explanations; compile-api/public-api; ADR0045; glossary. No semantic redesign; 75 targeted links/anchors/examples/glossary/whitespace/diff checks PASS.
MAIN compatibility compile command in Validation — PASS, 1s, 21 tasks: 6 executed/15 UP-TO-DATE (Prepare/CPU/Metal/Engine/Checkpoint).
MAIN `./gradlew :testing:architecture-tests:test` — PASS FROM-CACHE, not freshly executed: 25 tests/11 suites, zero failures/errors/skips; 627ms, 27 tasks: 1 cached/26 UP-TO-DATE. Main verified unchanged declared inputs/source consumers and valid cache; no live Compiler/ARCHITECTURE source access.
MAIN provider command in Validation — PASS CPU 30 tests/2 suites + Metal 19 tests/1 suite, zero failures/errors/skips; 1s, 26 tasks: 3 executed/23 UP-TO-DATE. Existing incubator-vector warning is not a Javadoc warning.
HEAD/index/protected files unchanged; source Model/Planning and 119 protected/source hashes match acceptance; no backend code delta. All predecessor dirty work retained.
No CPU execution, Prepare, NN training, cold-source or save widening. Root/native/full-repository/performance gates remain deferred to low-level join; no commit/push or successor launch.
Closure reuses accepted code/docs/test evidence; eight planning files: 623 local links/59 anchors, whitespace and `git diff --check` PASS. No unresolved acceptance issues; Compiler frontier None.
Follow-up: existing Prepare projection Draft only, not yet briefed/not Ready; CPU/NN/Planning/Model frontiers None, padding/NN/persistence gates unchanged.
Size exception: atomic Class C acceptance, admission, validation and compact result remain together above 200 lines/15 KB; below 25 KB, no acceptance split or reduction.
