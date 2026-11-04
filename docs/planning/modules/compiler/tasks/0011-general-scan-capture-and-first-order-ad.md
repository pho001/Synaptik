# Task Compiler0011: [General scan](../../../../glossary.md#general-differentiable-scan) [capture](../../../../glossary.md#term-capture-meanings) and first-order [AD](../../../../glossary.md#autograd)

> **Editorial reading key — not part of the historical record.**
>
> Compiler receives a scoped step expression whose formals are local placeholders, not stored
> per-step values. It checks the body, binds exact forward primals to residual/step/invariant
> reverse formals, constructs one joint opposite-traversal derivative Scan, then captures forward
> and derivative work together. A residual port names needed forward information; it does not
> itself allocate a history buffer. Final demand and capability queries cross into later
> preparation/backend responsibility without granting execution.
>
> Current reading: [scoped Scan capture, roles and first-order differentiation](../../../../reference/compiler/verification-scan.md#current-scoped-scan-capture-and-first-order-ad); [capability queries and hard-eligible owner selection](../../../../reference/compiler/planning.md#current-operation-capability-contracts).

## Status

Complete; accepted by main 2026-10-22 after independent Class C code/docs APPROVE and compatibility [checkpoint](../../../../glossary.md#term-checkpoint).
ImplementationAuthorized=true; user-authorized 2026-10-22 (“ok, pust se do toho”); launch [admission](../../../../glossary.md#term-admission) verified.

## Change class

Class C — scoped [identity](../../../../glossary.md#term-identity-meanings), pre-[capture](../../../../glossary.md#term-capture-meanings) [automatic differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)), saved-value [bindings](../../../../glossary.md#term-binding) and demand.
Clean executor and independent targeted code/docs review required.

## Goal

[Compile](../../../../glossary.md#compile) scoped [Scan](../../../../glossary.md#general-differentiable-scan) bodies and first-order cotangents (reverse sensitivities) using existing rules:
forward scan → demanded residuals (saved [primal](../../../../glossary.md#term-primal) values) → opposite-traversal reverse sibling →
one combined [capture](../../../../glossary.md#term-capture-meanings). Deliver validated [Tensor](../../../../glossary.md#tensor)-free bodies/demand, not execution.

## [Scope](../../../../glossary.md#term-scope)

[Capture](../../../../glossary.md#term-capture-meanings)/[inference](../../../../glossary.md#term-inference), [AD](../../../../glossary.md#autograd), rule [bindings](../../../../glossary.md#term-binding), attachment-preserving transforms and final query demand in all
[compile](../../../../glossary.md#compile) modes. Reuse completed Model0035/Planning0008 foundations.
Only new package-private production types: `io.github.pho001.synaptik.compiler.ScanBodyInference` and
`ScanGradientRules` in the same root package; no other production type/package.

## Non-goals

No authority/[API](../../../../glossary.md#term-application-programming-interface-api)/[dependency](../../../../glossary.md#term-dependency)/build changes, second [AD](../../../../glossary.md#autograd) algebra, [registry](../../../../glossary.md#term-registry), [IR](../../../../glossary.md#term-intermediate-representation-ir) AD, executor/tape/storage,
[Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu) implementation or [backend](../../../../glossary.md#backend)/performance claim. No fixed [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) [backpropagation through
time](../../../../glossary.md#term-backpropagation-through-time-bptt) ([BPTT](../../../../glossary.md#backpropagation-through-time-and-recurrent-state)), cold captureSource/reconstructSource or save/load [widening](../../../../glossary.md#term-widening), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) training/lengths/[padding](../../../../glossary.md#padding),
if/while/nesting/regions. Preserve fixed four-[dtype](../../../../glossary.md#data-type--datatype)/read-[layout](../../../../glossary.md#layout)/static routes; low backend [qualification](../../../../glossary.md#term-qualification) is later.

## Contracts

Read `AGENTS.md`, affected source/tests, root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>, then only these headings:

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>, [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [compiler](../../../../glossary.md#term-compiler)-[autograd](../../../../glossary.md#autograd): [`modules/compiler`](../../../../architecture/contracts/compiler-autograd.md#modulescompiler),
  [Compile artifacts](../../../../architecture/contracts/compiler-autograd.md#compile-artifacts)<span data-glossary-navigation="true"> ([Compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>,
  [Training graph model](../../../../architecture/contracts/compiler-autograd.md#training-graph-model)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph); [model glossary](../../../../glossary.md#term-model))</span>,
  [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>,
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>.
- recurrent-[scan](../../../../glossary.md#general-differentiable-scan): [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span> and all five subheadings:
  [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities)<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>,
  [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>,
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)<span data-glossary-navigation="true"> ([primal glossary](../../../../glossary.md#term-primal); [bindings glossary](../../../../glossary.md#term-binding))</span>,
  [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>,
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates).
- foundational-[modules](../../../../glossary.md#term-module): [`modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel),
  [`modules/planning`](../../../../architecture/contracts/foundational-modules.md#modulesplanning) and
  [Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries), only source/[IR](../../../../glossary.md#term-intermediate-representation-ir) and query boundaries.

Stop and report any missing/ambiguous contract or required extra type/boundary; do not invent architecture.

## Dependencies and integration

- Depends on: Model0034, Model0035, Planning0008 Complete; Compiler0006A/0006D/0008/0009 Complete baseline.
- Conflicts with: capture/inference/AD/closure/query/shared-scan writers; none active.
  Prepare/CPU/NN/persistence remains inactive. Serial, no concurrent writer.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [AD glossary](../../../../glossary.md#autograd); [scan glossary](../../../../glossary.md#general-differentiable-scan); [Prepare glossary](../../../../glossary.md#prepare); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus ALL accepted
  dirty changes, including Model0034/0035 and Planning0008, not merged predecessors.
- Integration order: frozen plan → independent planning review/main acceptance → explicit executor
  authorization → clean Compiler implementation → independent code/docs APPROVE → main compatibility
  checkpoint/acceptance; Prepare/CPU/NN stay None/Draft.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Integration [validation](../../../../glossary.md#term-validation): **Compiler0011 scoped-body compatibility [checkpoint](../../../../glossary.md#term-checkpoint)**, commands below;
  native root test/Javadoc gate at low-level join.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration owner: main (plans/status); executor: Compiler code/tests and
  explanatory/Javadoc edits; planning delegated.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

Main verified BEFORE launch, 2026-10-22: brief [Ready](../../../../glossary.md#term-ready), sole [authorized frontier](../../../../glossary.md#term-authorized-frontier); all named dependencies
Complete with accepted results/gates; Galileo plan APPROVE, zero findings, main acceptance today.
HEAD=origin/main as above; index empty; conflicts inactive; no other managed writer/test. All 147
accepted dirty files match final freeze (none new/missing/changed); 119 source/test/protected files
unchanged, including THIRD_PARTY_NOTICES.md, [CPU](../../../../glossary.md#term-cpu) cast [oracle](../../../../glossary.md#term-test-oracle) and Superseded Compiler0010.
Authorization preceded launch; implementation, independent review and main checkpoint are complete; no successor Ready.

## Files and symbols

Paths use `modules/compiler/src/{main,test}/java/io/github/pho001/synaptik/compiler/`.

- New `ScanBodyInference.java`: local inference and formal/role/outer descriptors.
  New <span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>`ScanGradientRules.java`: body inventory, joint-output rules and reverse [Scan](../../../../glossary.md#general-differentiable-scan).
- `GraphCapture.java` (`captureInternal`, `captureCombined`), `CapturedGraphInference.java`
  (`inferAndValidate`, [dispatch](../../../../glossary.md#term-dispatch)): unused formals, [lexical scope](../../../../glossary.md#term-lexical-scope)/escape checks and attachments.
- `AutogradPreflight.java` (`inventory`, [`preflight`](../../../../glossary.md#term-preflight), `preflightStage`, `validateOccurrence`, `rejectForwardOnlyOccurrences`),
  `FirstOrderGradientCoverage.java` (`classify`), `FirstOrderAutograd.java` (`expand`, `apply`,
  `DerivativeConstants`): preflight, joint [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), positional accumulation, closed [constants](../../../../glossary.md#term-constant).
- `ElementwiseGradientRules.java`, `LinearAlgebraGradientRules.java`, `LayoutGradientRules.java`,
  `ReductionGradientRules.java` (`apply`/helpers): [primal](../../../../glossary.md#term-primal) [binding](../../../../glossary.md#term-binding); non-scan callers retain [identity](../../../../glossary.md#term-identity-meanings) binding/formulas/guards.
- `GraphCanonicalization.java` (`rebuild`), `ForwardExactArithmeticRewriting.java`, `ForwardConstantFolding.java`,
  `ForwardDeadCodeElimination.java`, `ForwardCommonSubexpressionElimination.java`,
  `ForwardGraphOptimization.java`, `StaticResultLogicalLayoutClosure.java`: preserve/revalidate body,
  role/demand facts and saved sites in every rebuild branch; no new rewrite/fusion implementation.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion))</span>
- `GraphCompiler.java` ([`compile`](../../../../glossary.md#compile), query construction), `GraphCompilation.java`, `GraphCompilationPort.java`:
  one graph, preserved publication/caller identities, final body-bearing queries (currently three-argument).<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication); [argument glossary](../../../../glossary.md#term-argument))</span>
- Read-only foundations: [Model](../../../../glossary.md#term-model) `tensor/{ScanBody,Scan,ScanResult,TensorProducer}.java`,
  `graph/{CompiledNode,CompiledScanBody}.java`; [Planning](../../../../glossary.md#term-planning) `capability/OperationCapabilityQuery.java`,
  `memory/LogicalMemoryPlanning.java`. No foundation [API](../../../../glossary.md#term-application-programming-interface-api)/algorithm changes.
- New focused tests `ScanBodyInferenceTest.java`, `ScanGradientRulesTest.java`, `GeneralScanCompilerTest.java`;
  extend existing capture/inference/preflight/gradient/graph-optimization/layout/compiler/port tests as affected.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout); [compiler glossary](../../../../glossary.md#term-compiler))</span>

## Acceptance criteria

1. Capture/inference admits valid PURE Model source independently of the CPU whitelist (FLOAT16 TANH,
   F32 RELU included), without AD/backend qualification. Preserve wrappers, retained order/siblings, unused
   declared formals and repeatable roles over unique local outputs. Reused templates get independent
   scope-qualified IDs; reject escape/sharing across different templates, not reuse. No outer IDs inward.<span data-glossary-navigation="true"> ([Capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [PURE glossary](../../../../glossary.md#term-pure); [Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu); [FLOAT16 glossary](../../../../glossary.md#term-float16); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [AD glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend); [qualification glossary](../../../../glossary.md#term-qualification); [scope glossary](../../../../glossary.md#term-scope))</span>
2. Inventory exact operation/attrs/role/Shape/dtype support and the entire generated derivative closure.
   Required baseline: homogeneous F32/F64 static canonical boundaries/algebra below, not all Model sources. Test actual
   generated kinds/attrs; reject unsupported requirements before derivative/seed/formal/Scan allocation.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [dtype glossary](../../../../glossary.md#data-type--datatype); [derivative glossary](../../../../glossary.md#term-derivative); [homogeneous glossary](../../../../glossary.md#term-homogeneous); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [canonical glossary](../../../../glossary.md#term-canonical-form); [seed glossary](../../../../glossary.md#term-seed); [Scan glossary](../../../../glossary.md#general-differentiable-scan))</span>

   | Source/rule [owner](../../../../glossary.md#term-owner) | Required formula/closure inventory |
   |---|---|
   | [Tensor](../../../../glossary.md#tensor)/[scalar](../../../../glossary.md#term-scalar) ADD/SUB/MUL/DIV; [Elementwise](../../../../glossary.md#term-elementwise) | Existing NEG/product/DIV formulas, broadcast `sumToShape`; no new [masks](../../../../glossary.md#term-boolean-mask)/numerics. |
   | NEG/[TANH](../../../../glossary.md#term-tanh)/[SIGMOID](../../../../glossary.md#term-sigmoid); Elementwise | NEG; saved output, MUL/SUB, typed one/expand; no replay. |
   | [Rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul); LinearAlgebra | Both positional operands, `[1,0]` transposes, MATMUL, [shape](../../../../glossary.md#shape)/type [normalization](../../../../glossary.md#term-normalization). |
   | Rank-two `[1,0]` PERMUTE; [Layout](../../../../glossary.md#layout) | Inverse [permutation](../../../../glossary.md#permutation); generated EXPAND/RESHAPE/[CONTIGUOUS](../../../../glossary.md#term-contiguous) and required [rank](../../../../glossary.md#term-rank)-editing [views](../../../../glossary.md#term-view-meanings). |
   | Generated reductions/[constants](../../../../glossary.md#term-constant); Reduction + existing owners | SUM/`SumToShapeAttrs`, broadcast [axes](../../../../glossary.md#term-axis)/keep-dim restoration, expand/views, required floating casts, registered typed zero/one splats. |

3. Every formula primal resolves via an explicit compile-local original-Tensor → residual/step/invariant
   reverse-formal function; contribution keys stay original identities. Carry/body primals use original
   residual slots; step/invariants use ordered explicit inputs. Canonical TANH/SIGMOID saved
   outputs resolve to residual formals, never replay. oneLike constants close through non-gradient
   invariant formals/registered outer splats; no free leaves or AD ValueId→Tensor map.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([primal glossary](../../../../glossary.md#term-primal); [compile glossary](../../../../glossary.md#compile); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [invariant glossary](../../../../glossary.md#term-invariant); [Carry glossary](../../../../glossary.md#term-carry); [gradient glossary](../../../../glossary.md#term-gradient); [ValueId glossary](../../../../glossary.md#logical-valueid-versus-physical-memory-slot))</span>
4. Replace per-output-slot scan dispatch with ONE joint final-carry/emission reverse Scan per occurrence,
   opposite traversal, before capture. Missing internal port seeds/parameter-sum initializers are typed zeros;
   public seed guards unchanged. Both seed classes coexist. dSequences/dInitialCarry/invariant sums and
   tied roles/repeated roots/cross-occurrence terms accumulate positionally without deduplication.
   Internal carry cotangents traverse even when original h0/formal/body values have requiresGrad=false;
   adapt local preflightStage/validateOccurrence without changing flags/public guards. Preserve outer G (OR of actual flags).<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [occurrence glossary](../../../../glossary.md#term-occurrence); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [parameter glossary](../../../../glossary.md#term-parameter))</span>
5. Preflight rejects residual-root differentiation (including paths through residuals), every scan higher-order/
   createGraph request and unsupported body occurrences anywhere in the original forward inventory, including
   an unrelated sibling branch. Pin unchanged Tensor-ID counter on known rejections; no partial derivative work.
   Non-scan higher-order support and fixed-scan/Conv3d rejection stay unchanged.<span data-glossary-navigation="true"> ([Preflight glossary](../../../../glossary.md#term-preflight); [partial derivative glossary](../../../../glossary.md#term-partial-derivative); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
6. One phase-aware capture retains original wrappers, independent bodies, publication/caller TensorIds and
   derivative order. Every four-argument CompiledNode rebuild must retain its five-argument body,
   in EVERY branch; no CSE across scans/scopes/live saved sites. Test each pass, optimization on/off,
   repeated-template occurrences and rebuilds triggered by a neighboring ordinary rewrite.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [argument glossary](../../../../glossary.md#term-argument); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
7. ForwardDCE.eliminateGraph sibling-liveness is NOT port demand. Derive demandedOutputIndices from final
   outer publication/ordinary uses AFTER rewrites/layout closure, not siblings/internal uses/latent body roots.
   Revalidate each changed candidate. Dead forward/reverse histories impose no history demand/fusion restriction.
   Keep live saved-site identity/rounding boundaries through closure, reindexing, DCE/CSE and future fusion handoff.
   Logical value records are not storage; bytes/stores are later Prepare/CPU acceptance.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [history demand glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [fusion glossary](../../../../glossary.md#term-fusion); [identity glossary](../../../../glossary.md#term-identity-meanings); [rounding glossary](../../../../glossary.md#term-rounding); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [handoff glossary](../../../../glossary.md#term-handoff); [Prepare glossary](../../../../glossary.md#prepare))</span>
8. Final OperationCapabilityQuery carries validated complete body/roles/descriptors/demand; no child-only
   ownership shortcut/inner mixed-owner partitioning. Test providers prove closure delivery and unsupported
   binder rejection. Ordinary/fixed queries stay empty-body; real capability/cold source admission unchanged.<span data-glossary-navigation="true"> ([OperationCapabilityQuery glossary](../../../../glossary.md#operation-capability-query--operationcapabilityquery); [descriptors glossary](../../../../glossary.md#term-descriptor); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [admission glossary](../../../../glossary.md#term-admission))</span>
9. Tests cover two carries, both seed classes, shared parameters, tied actual roles and occurrences/template reuse,
   closed constants, exact saved TANH/SIGMOID bindings/special values, broadcasts, DIV zero/NaN/Inf/signed zero,
   MATMUL/transpose and generated closure. Compare at least three steps against unrolled gradients with zero
   non-gradient h0 and trainable w/u, in both traversal directions, using a bounded test-only oracle (no production runner).
   Check original-time indices and previous-carry-before-step. T=0 has exact initial final carry,
   dInitialCarry=dFinalCarry, empty emissions/residuals and zero invariant sums. Empty axes retain constituent
   reduction semantics and never bypass unsupported preflight. Vary T to prove O(body), not T-fold graph growth.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero); [transpose glossary](../../../../glossary.md#term-transpose); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [oracle glossary](../../../../glossary.md#term-test-oracle); [graph glossary](../../../../glossary.md#graph))</span>

## [Validation](../../../../glossary.md#term-validation)

Recorded validation, repository root (outcomes in Result):

```bash
./gradlew :modules:compiler:test --tests '*ScanBodyInferenceTest' --tests '*ScanGradientRulesTest' --tests '*GeneralScanCompilerTest'
./gradlew :modules:compiler:test :modules:compiler:javadoc
```

One final [module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run) after executable stabilization; regenerate [Javadoc](../../../../glossary.md#term-javadoc) after final comment edits.
Main's **Compiler0011 scoped-body compatibility [checkpoint](../../../../glossary.md#term-checkpoint)**, once after independent code/docs approval:

```bash
./gradlew :modules:prepare:compileTestJava :backends:cpu:compileTestJava :backends:metal:compileTestJava :modules:engine:compileTestJava :extensions:checkpoint:compileTestJava
./gradlew :testing:architecture-tests:test
./gradlew :backends:cpu:test --tests '*CpuCapabilityProviderTest' --tests '*CpuCapabilityProviderPublicShapeTest' :backends:metal:test --tests '*MetalCapabilityProviderTest'
git diff --check
```

Check local links/anchors and new-file whitespace. No [Gradle](../../../../glossary.md#term-gradle)/benchmark during [planning](../../../../glossary.md#term-planning).
No root/native/five-[fork](../../../../glossary.md#term-fork-meanings)/performance campaign. [Scan](../../../../glossary.md#general-differentiable-scan) [conformance](../../../../glossary.md#term-conformance)/integration and native root gate are
later [CPU](../../../../glossary.md#term-cpu)/join work. Review reuses evidence.

## Documentation and review impact

Implementation/review updated [Javadoc](../../../../glossary.md#term-javadoc) ([bindings](../../../../glossary.md#term-binding), [scope](../../../../glossary.md#term-scope), flags, demand/rejections, params/results/failures)
and targeted `docs/architecture/general-differentiable-scan.md`, `docs/architecture/training-graph.md`,
`docs/api/compile-api.md`; check `docs/api/public-api.md` and glossary [scan](../../../../glossary.md#general-differentiable-scan)/[primal](../../../../glossary.md#term-primal)/[cotangent](../../../../glossary.md#cotangent)/[residual](../../../../glossary.md#scan-residual-port-and-history-demand) entries for status drift.
Applied documentation-rules/general/[planning](../../../../glossary.md#term-planning) and [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc/architecture profiles to corresponding edits.
Independent code/docs review checked diff/tests/contracts: closure, false-flag [carry](../../../../glossary.md#term-carry), saved [activation](../../../../glossary.md#term-activation)
[identity](../../../../glossary.md#term-identity-meanings)/[constants](../../../../glossary.md#term-constant), scope/demand. No new term/authority; shared/authority status edits need main delegation.
Planning changes no Java/Javadoc/capability.

## Result

Status: Complete; main acceptance 2026-10-22 after Kant implementation, Euclid final independent Class C code/docs APPROVE (zero P1/P2) and compatibility [checkpoint](../../../../glossary.md#term-checkpoint).
[Scope](../../../../glossary.md#term-scope) complete: scoped [capture](../../../../glossary.md#term-capture-meanings)/[inference](../../../../glossary.md#term-inference), one joint reverse [Scan](../../../../glossary.md#general-differentiable-scan) via ordinary [AD](../../../../glossary.md#autograd)/exact [primal](../../../../glossary.md#term-primal) [bindings](../../../../glossary.md#term-binding) and closed [constants](../../../../glossary.md#term-constant), non-[gradient](../../../../glossary.md#term-gradient) [carry](../../../../glossary.md#term-carry) flow, whole-original [preflight](../../../../glossary.md#term-preflight)/[Tensor](../../../../glossary.md#tensor)-allocation-free rejections, attachments/exact saved boundaries and final demand/body queries.
Changed 25 [Compiler](../../../../glossary.md#term-compiler) Java files: two new package-private production [owners](../../../../glossary.md#term-owner), three new scan tests, existing owners/passes/GraphCompilation[Port]/GraphCapture and two old tests; full file/[symbol](../../../../glossary.md#term-symbol) scope above.
Numeric [oracle](../../../../glossary.md#term-test-oracle): three-step [unrolled](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) existing gradients, [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding), both directions, two carries/joint [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), broadcast and [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[transpose](../../../../glossary.md#term-transpose); T=0, previous carry/original time, ties/reuse, DIV specials, exact saved [TANH](../../../../glossary.md#term-tanh)/[SIGMOID](../../../../glossary.md#term-sigmoid), T=0/3/100000 [graph](../../../../glossary.md#graph) size and Tensor-ID rejection [guard](../../../../glossary.md#term-guard).
Worker focused: `./gradlew :modules:compiler:test --tests '*GraphCaptureTest' --tests '*ScanBodyInferenceTest' --tests '*ScanGradientRulesTest' --tests '*GeneralScanCompilerTest' --tests '*FirstOrderGradientCoverageTest'` — PASS 47/47.
Final `./gradlew :modules:compiler:test :modules:compiler:javadoc` — PASS 340 tests/46 suites, zero failures/errors/skips; final comment-only [Javadoc](../../../../glossary.md#term-javadoc) regeneration PASS, zero warnings.
Euclid reviewed 25 Java files: 24 byte-identical; only ScanGradientRules.java Javadoc changed to “Tensor-allocation-free” ([SHA-256](../../../../glossary.md#term-sha-256) `2d6962add1c3eda3a2534efff63ac8fe106846ba3be8b7f97ca00ae2c7ca168d`). No executable change; reused 340 tests; own final `./gradlew :modules:compiler:javadoc` PASS, zero warnings.
Review finalized 11 factual-status/explanation docs: ARCHITECTURE; compiler-[autograd](../../../../glossary.md#autograd)/recurrent-scan/foundational-[modules](../../../../glossary.md#term-module) contracts; general-[differentiable](../../../../glossary.md#derivative-and-chain-rule)-scan/training-graph/module-boundaries explanations; [compile](../../../../glossary.md#compile)-api/public-api; ADR0045; glossary. No semantic redesign; 75 targeted links/anchors/examples/glossary/whitespace/diff checks PASS.
MAIN compatibility compile command in [Validation](../../../../glossary.md#term-validation) — PASS, 1s, 21 tasks: 6 executed/15 UP-TO-DATE ([Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal)/[Engine](../../../../glossary.md#term-engine)/Checkpoint).
MAIN `./gradlew :testing:architecture-tests:test` — PASS FROM-[CACHE](../../../../glossary.md#term-cache), not freshly executed: 25 tests/11 suites, zero failures/errors/skips; 627ms, 27 tasks: 1 cached/26 UP-TO-DATE. Main verified unchanged declared inputs/source consumers and valid cache; no live Compiler/ARCHITECTURE source access.
MAIN provider command in Validation — PASS CPU 30 tests/2 suites + Metal 19 tests/1 suite, zero failures/errors/skips; 1s, 26 tasks: 3 executed/23 UP-TO-DATE. Existing incubator-[vector](../../../../glossary.md#term-vector) warning is not a Javadoc warning.
HEAD/index/protected files unchanged; source [Model](../../../../glossary.md#term-model)/[Planning](../../../../glossary.md#term-planning) and 119 protected/source [hashes](../../../../glossary.md#term-hash) match acceptance; no [backend](../../../../glossary.md#backend) code delta. All predecessor dirty work retained.
No CPU execution, Prepare, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) training, cold-source or save [widening](../../../../glossary.md#term-widening). Root/native/full-repository/performance gates remain deferred to low-level join; no [commit](../../../../glossary.md#term-commit)/push or successor launch.
Closure reuses accepted code/docs/test evidence; eight planning files: 623 local links/59 anchors, whitespace and `git diff --check` PASS. No unresolved acceptance issues; Compiler frontier None.
Follow-up: existing Prepare projection Draft only, not yet briefed/not [Ready](../../../../glossary.md#term-ready); CPU/NN/Planning/Model frontiers None, [padding](../../../../glossary.md#padding)/NN/[persistence](../../../../glossary.md#term-persistence) gates unchanged.
Size exception: [atomic](../../../../glossary.md#resource-transaction-and-rollback) Class C acceptance, [admission](../../../../glossary.md#term-admission), validation and compact result remain together above 200 lines/15 KB; below 25 KB, no acceptance split or reduction.
