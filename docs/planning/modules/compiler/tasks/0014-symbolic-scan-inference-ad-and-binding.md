# Task Compiler0014: [Symbolic Scan](../../../../glossary.md#symbolic-scan) [inference](../../../../glossary.md#term-inference), first-order [AD](../../../../glossary.md#autograd) and scoped [binding](../../../../glossary.md#term-binding)

> **Editorial reading key — not part of the historical record.**
>
> The named batch dimension is a metadata variable, not a sampled batch of stored values. Compiler
> verifies each body scope and constructs actual derivative expressions; BatchBindingPlan later
> checks fresh concrete descriptors and all local obligations for each binding. Equal numeric
> local IDs in two reused bodies do not identify the same value. This metadata binding owns
> neither buffers nor backend executables and cannot substitute a successful symbolic proof for
> physical execution qualification.
>
> Current reading: [scoped Scan capture, roles and first-order differentiation](../../../../reference/compiler/verification-scan.md#current-scoped-scan-capture-and-first-order-ad); [descriptor verification and deferred constraints](../../../../reference/compiler/verification-scan.md#current-package-private-verification-inference).

## Status

Complete — independently reviewed and MAIN accepted 2026-10-25 after the named compatibility [checkpoint](../../../../glossary.md#term-checkpoint).
PlanningAuthorized=true; ImplementationAuthorized=true (historical). Implementation frontier: None.
[Compiler](../../../../glossary.md#term-compiler) [metadata](../../../../glossary.md#term-metadata)/actual first-order [AD](../../../../glossary.md#autograd) only; no symbolic [backend](../../../../glossary.md#backend) execution is authorized.

## Change class

Class C — scoped [inference](../../../../glossary.md#term-inference), saved primals, [automatic differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)) and [metadata](../../../../glossary.md#term-metadata) [admission](../../../../glossary.md#term-admission).
Clean implementation and independent review required.

## Goal

Extend existing [Compiler](../../../../glossary.md#term-compiler) [owners](../../../../glossary.md#term-owner) to symbolic [inference](../../../../glossary.md#term-inference), actual first-order reverse construction and
fresh scoped all-port `BatchBindingPlan` [binding](../../../../glossary.md#term-binding). Bind [metadata](../../../../glossary.md#term-metadata), not resources. Retain static
[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) paths and ordinary dynamic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) behavior.

## [Scope](../../../../glossary.md#term-scope)

All-valid [general SCAN](../../../../glossary.md#general-differentiable-scan), one [canonical](../../../../glossary.md#term-canonical-form) named B and fixed time T/[features](../../../../glossary.md#term-software-feature) F/state widths H.
[Inference](../../../../glossary.md#term-inference) independently checks accepted [Model](../../../../glossary.md#term-model) [PURE](../../../../glossary.md#term-pure) semantics/applicable [dtypes](../../../../glossary.md#data-type--datatype); do not import
the [AD](../../../../glossary.md#autograd) inventory or a [CPU](../../../../glossary.md#term-cpu) whitelist into forward inference. First-order AD is separately bounded
to [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) and the accepted algebra. Initial CPU FLOAT32 [qualification](../../../../glossary.md#term-qualification) is later.
Include necessary bounded [rank](../../../../glossary.md#term-rank)-three ordinary outer [seed](../../../../glossary.md#term-seed)/[constant](../../../../glossary.md#term-constant)/[pointwise](../../../../glossary.md#term-pointwise)/[layout](../../../../glossary.md#layout) plumbing,
scoped private obligations and canonical actual [descriptors](../../../../glossary.md#term-descriptor) for every [Scan](../../../../glossary.md#term-cumulative-scan) output position.

## Non-goals

No [Model](../../../../glossary.md#term-model) behavior/[API](../../../../glossary.md#term-application-programming-interface-api)/header change; [Scan](../../../../glossary.md#term-cumulative-scan).java class support-status [Javadoc](../../../../glossary.md#term-javadoc) only. No symbolic length mode, variable T/F/H, higher-order,
fixed-scan [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), arithmetic/rewrite [widening](../../../../glossary.md#term-widening), cold/copy/wire, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training, [Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu)/[Engine](../../../../glossary.md#term-engine)
enablement or geometry [memo](../../../../glossary.md#cache-and-memoization). No native build, performance campaign, stage/[commit](../../../../glossary.md#term-commit)/push or successor
launch. No [graph](../../../../glossary.md#graph)/session/[constant](../../../../glossary.md#term-constant) type, public [signature](../../../../glossary.md#term-signature), [ABI](../../../../glossary.md#term-application-binary-interface-abi) or [dependency](../../../../glossary.md#term-dependency) change is selected.

## Contracts

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span> and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Compiler](../../../../glossary.md#term-compiler) [Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [metadata glossary](../../../../glossary.md#term-metadata); [binding glossary](../../../../glossary.md#term-binding))</span>,
  [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>,
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span> and
  [Bounded general-scan cold reconstruction](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)<span data-glossary-navigation="true"> ([cold reconstruction glossary](../../../../glossary.md#term-cold-reconstruction))</span>.
- [Scan](../../../../glossary.md#term-cumulative-scan) [Accepted symbolic-batch scan](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan),
  including ALL four subsections; [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities)<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>,
  [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>,
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)<span data-glossary-navigation="true"> ([primal glossary](../../../../glossary.md#term-primal))</span>,
  [Header, bindings and source admission](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission)<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>
  for retained static length mode. Stop on ambiguity; contracts win.

## Dependencies and integration

- Depends on: [Model0041](../../model/tasks/0041-symbolic-scan-authority-cutover.md#result),
  [Model0042](../../model/tasks/0042-symbolic-scan-source-foundation.md#result),
  [Compiler0011](0011-general-scan-capture-and-first-order-ad.md#result),
  [Compiler0013](0013-length-aware-scan-capture-and-first-order-ad.md#result),
  [Planning0008](../../planning/tasks/0008-general-scan-body-capability-query.md#result) and
  [Prepare0013 ordinary binding baseline](../../prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result)<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>, all Complete.
- Conflicts with: inference/capture/AD/binding/layout/query/demand, cold and shared Scan/Compiler
  contract/docs writers; future numeric vertical. None active.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [capture glossary](../../../../glossary.md#term-capture-meanings); [AD glossary](../../../../glossary.md#autograd); [layout glossary](../../../../glossary.md#layout); [Scan glossary](../../../../glossary.md#term-cumulative-scan); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Parallel group: None; serial writer, read-only reviewer.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus ALL
  accepted dirty work including Model0042, not merged predecessors; empty index.
- Integration order: approved plan → verified Ready/source admission → clean implementation
  → independent code/docs approval → MAIN scoped-binding compatibility checkpoint → acceptance; no next launch.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Integration validation: named Compiler0014 scoped-binding compatibility checkpoint below;
  one configured root gate belongs to the separately admitted numeric vertical join.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; sole clean writer until freeze; reviewer read-only.

[Ready](../../../../glossary.md#term-ready) verification (MAIN, 2026-10-25): explicit approval, all six dependencies/reviews/gates Complete,
stable authority/base, inactive conflicts and sole writer/frontier Compiler0014 verified.
Implementation baseline: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/pre-compiler0014-implementation.json`
and `-files/` copies. Preserve all accepted dirty work. No downstream launch, [commit](../../../../glossary.md#term-commit) or push.

## Files and symbols

Production paths: `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/`.
MAIN admits only [Model](../../../../glossary.md#term-model) `Scan.java` class support-status [Javadoc](../../../../glossary.md#term-javadoc); stripped code must remain identical.
Added/moved production types: None proposed, not proof. If needed, stop with exact
package/name/purpose/[signature](../../../../glossary.md#term-signature) and reasoning for MAIN/user approval; no silent expansion.

- `ScanBodyInference.java` ([`capture`](../../../../glossary.md#term-capture-meanings), `infer`, [residual](../../../../glossary.md#scan-residual-port-and-history-demand) [validation](../../../../glossary.md#term-validation), `refreshDemand`) and
  `CapturedGraphInference.java` (`inferAndValidate`): all local declarations, ordinary family
  inference, actual/formal/result equality, typed obligations and independent scope validation.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [scope glossary](../../../../glossary.md#term-scope))</span>
- `BatchBindingPlan.java` ([`prepare`](../../../../glossary.md#prepare), `bind`, `Binding.descriptor`, dimension/count recipes):
  private per-containing-occurrence/scope preindexing and exact SCAN all-port layout intent.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [occurrence glossary](../../../../glossary.md#term-occurrence); [layout glossary](../../../../glossary.md#layout))</span>
  `DeferredGraphConstraint.java` (`GraphPredicate`, `GraphPredicateProof.bind`) is the existing
  private predicate owner; preserve <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>`CompileDiagnostics`/eight-[artifact](../../../../glossary.md#term-artifact)/public [binding](../../../../glossary.md#term-binding) signatures.
- `ScanGradientRules.java` ([`preflight`](../../../../glossary.md#term-preflight), `apply`, `formal`), `AutogradPreflight.java`,
  `FirstOrderAutograd.java` (`expandBody`, `DerivativeConstants`): symbolic closure and joint seeds.
  Existing <span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>`ElementwiseGradientRules`, `LinearAlgebraGradientRules`, `ReductionGradientRules`
  (SUM restoration, MEAN/count) and <span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean))</span>`LayoutGradientRules` own any necessary formula adjustments.
- `GraphCapture`, `GraphCompiler` final body queries, `StaticResultLogicalLayoutClosure` and existing
  optimization/reindex/demand owners: inspect/preserve phase, scoped IDs, saved sites and unresolved
  templates. Change only a demonstrated obstruction; no unrelated pass refactor or body optimizer.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `GraphCompilationPort.java`/`SourceGraphReconstruction.java`: only preserve explicit cold rejection.
- Same-package tests: `ScanBodyInferenceTest`, `ScanGradientRulesTest`,
  `GeneralScanCompilerTest`, `BatchBindingPlanTest`, `StaticResultLogicalLayoutClosureTest` and
  retained `LengthAwareScanBodyInferenceTest`/`LengthAwareScanGradientRulesTest`/cold tests.
  Split blanket symbolic-rejection tests into admission/retained guards. Model <span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>[`Scan`](../../../../glossary.md#term-cumulative-scan)/`ScanBody` and
  `ScanSymbolicBatchSourceTest` are read-only source/[shape](../../../../glossary.md#shape) references, not new owners to modify.

## Acceptance criteria

1. Independently infer every local value/formal, actual binding and canonical final/emission/residual
   port, including unused declarations and canonical siblings. Verify single named B, static T,
   shape/type/flags, canonical static versus unresolved symbolic layouts, retained order and gradient
   OR. Raw malformed IR tests bypass Model checks. Reused templates remain separate body scopes;
   coincident local/outer IDs or two occurrences' equal local IDs never share proof identity.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [canonical glossary](../../../../glossary.md#term-canonical-form); [residual port glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [Model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
2. Forward inference retains accepted Model PURE/dtype breadth (for example low TANH/F32 RELU and
   discrete siblings), without claiming AD/provider support. Actual first-order AD independently
   closes tensor/scalar ADD/SUB/MUL/DIV, NEG/TANH/SIGMOID, rank-two MATMUL/transpose,
   SUM/sumToShape, constants and necessary expand/rank edits/contiguous/same-dtype casts.
   Prove named-axis preservation for feature edits; unproved B mixing rejects, not sample binding.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [PURE glossary](../../../../glossary.md#term-pure); [dtype glossary](../../../../glossary.md#data-type--datatype); [TANH glossary](../../../../glossary.md#term-tanh); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [AD glossary](../../../../glossary.md#autograd); [tensor glossary](../../../../glossary.md#tensor); [scalar glossary](../../../../glossary.md#term-scalar); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [transpose glossary](../../../../glossary.md#term-transpose); [constants glossary](../../../../glossary.md#term-constant); [rank glossary](../../../../glossary.md#term-rank); [contiguous glossary](../../../../glossary.md#term-contiguous); [axis glossary](../../../../glossary.md#term-axis); [feature glossary](../../../../glossary.md#term-feature); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
3. Generate the real opposite-traversal sibling once per occurrence with joint final/emission seeds,
   missing typed-zero seeds and explicit invariant constants. Load exact saved TANH/SIGMOID outputs,
   never replay them. Shared parameter-shaped additive carries contract actual B (dWx K=B), including
   bias unbroadcast at B=0/1. Preserve ties, repeated roles, cross-occurrence sums and false-flag carry
   flow. No B-by-parameter bank, handwritten reverse substitute or prototype MEAN denominator.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [invariant glossary](../../../../glossary.md#term-invariant); [parameter glossary](../../../../glossary.md#term-parameter); [bias glossary](../../../../glossary.md#term-bias); [carry glossary](../../../../glossary.md#term-carry); [prototype glossary](../../../../glossary.md#term-prototype); [MEAN glossary](../../../../glossary.md#term-mean))</span>
4. Ordinary outer `[T,B,*]` [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/zeros/splats, [pointwise](../../../../glossary.md#term-pointwise)/[layout](../../../../glossary.md#layout) edits and [scalar objective](../../../../glossary.md#loss)
   `mean(finalCarry)` close through existing [owners](../../../../glossary.md#term-owner). Count remains `oneLike(input).sum(...)` with
   actual B*H. Test actual generated expressions numerically against independent nonuniform ordinary
   Tensor unrolling in both traversals/F32/F64, joint and emission-only seeds, tied parameters,
   B=0/1/equal-to-F-or-H and 64→40→1→64 binding reuse. Extend only test-local bound evaluation
   of the actual symbolic expressions; do not rebuild a static reverse or use cold reconstruction.<span data-glossary-navigation="true"> ([unrolling glossary](../../../../glossary.md#term-unrolling); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [evaluation glossary](../../../../glossary.md#term-evaluation); [cold reconstruction glossary](../../../../glossary.md#term-cold-reconstruction))</span>
5. [`prepare`](../../../../glossary.md#prepare) preindexes all local typed obligations by exact containing occurrence/scope, all extent
   validity and SCAN canonical actual-layout recipes. <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [extent glossary](../../../../glossary.md#term-extent); [SCAN glossary](../../../../glossary.md#term-cumulative-scan))</span>`bind` checks every input occurrence and every
   scoped obligation freshly on EVERY call, even at repeated B, including unused/dead declarations.
   Unknown symbolic multi-output kinds still reject. No diagnostic-string execution, new public
   predicate protocol, per-call graph walk/inference/recapture/optimization/backend decision or memo.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [memo glossary](../../../../glossary.md#cache-and-memoization))</span>
6. Resolve canonical non-view actual descriptors for ALL final/emission/residual ports, including
   dead siblings; preserve lazy dense outer value-list indexing and current supplied input-backed
   descriptors (<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [lazy glossary](../../../../glossary.md#term-lazy-evaluation))</span>`isView` included). Shared histories stay `[T,F,H]`, not `[T,B,F,H]`. Templates stay
   unresolved, static closure unchanged; refreshed queries/demand/phase/roles/saved sites survive
   standard/disabled optimization. Unknowns/foreign symbols/unproved expressions and overflow fail
   at the owning preflight/bind boundary, not deferred until an undemanded descriptor is requested.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [preflight glossary](../../../../glossary.md#term-preflight))</span>
7. Separate fixed-T=0 fixtures from T>0: no body work, distinct canonical raw final carry,
   dInitialCarry=seed, empty stacked ports/dX and typed-zero invariant sums. At B=0 do not globally
   skip nonempty shared work/parameter contractions; empty SUM/K=0 gives typed zero and empty MEAN
   gives NaN. Test B=0/T=0 separately and together; unsupported/overflowing dead bodies still reject.
   These are Compiler/test-oracle semantics, not physical bytes, runtime skipping or execution proof.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [NaN glossary](../../../../glossary.md#term-nan); [Compiler glossary](../../../../glossary.md#term-compiler); [oracle glossary](../../../../glossary.md#term-test-oracle); [runtime glossary](../../../../glossary.md#term-runtime))</span>
8. Deliberate whole-original backward preflight rejects unsupported rules/dtypes, residual roots/paths
   and higher-order/createGraph before derivative IDs, including unrelated branches/T=0/dead siblings.
   Unsupported forward metadata rejects independently. Retain static length-mode inference/AD and its
   exact header/roles, but reject symbolic length mode. Cold capture/reconstruct stays static-only
   with no IDs allocated on rejection, even B present only in an unused declaration/static outer ports.
   Keep actual backend/provider rejection and ordinary/static compatibility; no execution enablement.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker focused development, then final affected-[module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run) after source stabilization:

```bash
./gradlew --no-daemon :modules:compiler:test --tests '*Scan*Test' --tests '*BatchBindingPlanTest' --tests '*AutogradPreflightTest' --tests '*FirstOrderGradientCoverageTest' --tests '*StaticResultLogicalLayoutClosureTest' --tests '*SourceGraphReconstructionTest'
./gradlew --no-daemon :modules:compiler:test :modules:compiler:javadoc
```

Final module/[Javadoc](../../../../glossary.md#term-javadoc) once after stabilization; reviewer reuses evidence.
MAIN runs the **Compiler0014 scoped-[binding](../../../../glossary.md#term-binding) compatibility [checkpoint](../../../../glossary.md#term-checkpoint)** once after ordered review:

```bash
./gradlew --no-daemon :modules:prepare:compileTestJava :backends:cpu:compileTestJava :backends:metal:compileTestJava :modules:engine:compileTestJava :extensions:checkpoint:compileTestJava
./gradlew --no-daemon :modules:prepare:test --tests '*GraphPreparationTest' --tests '*BackendPartitionFinalizationHandoffTest' :backends:cpu:test --tests '*CpuGeneralScanCapabilityTest' :modules:engine:test --tests '*EngineOrdinaryDynamicBatchTest' :extensions:checkpoint:test --tests '*GraphCheckpointGeneralScanTest'
```

Performance opt-ins stay false; no native build/full-root/performance campaign. The later numeric
vertical owns [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance)/[Engine](../../../../glossary.md#term-engine) integration, specialized clean-Java/Class-File/performance
evidence and one configured root join. No [dependency](../../../../glossary.md#term-dependency)/[ABI](../../../../glossary.md#term-application-binary-interface-abi) change.

## Documentation and review impact

Independent Class C review finalizes affected [Javadoc](../../../../glossary.md#term-javadoc) and support status:
`docs/api/compile-api.md`, `docs/architecture/general-differentiable-scan.md`, root, owning [Compiler](../../../../glossary.md#term-compiler)/
[Scan](../../../../glossary.md#term-cumulative-scan) contracts, glossary, this brief/Compiler master/roadmap/dynamic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) map and [Model](../../../../glossary.md#term-model) master/
symbolic-batch proposal. No new semantics/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/[dependency](../../../../glossary.md#term-dependency). Search glossary [`Symbolic Scan`](../../../../glossary.md#symbolic-scan)/
[`Symbolic batch binding`](../../../../glossary.md#symbolic-batch-binding); update capability status only. Stop on new decisions.

## Result

MAIN accepted 2026-10-25 after independent Class C code/docs APPROVE, 0 open P1/P2.
Three existing [owners](../../../../glossary.md#term-owner) implement scoped [PURE](../../../../glossary.md#term-pure) [inference](../../../../glossary.md#term-inference), actual [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) reverse and fresh all-port [binding](../../../../glossary.md#term-binding).
Nonuniform unroll witnesses cover both traversals/[seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), B=0/1/F/H, 64→40→1→64, actual [MEAN](../../../../glossary.md#term-mean),
ties, saved outputs, T=0 and shared work; malformed [IR](../../../../glossary.md#term-intermediate-representation-ir)/dead obligations reject eagerly.
[Compiler](../../../../glossary.md#term-compiler) [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc) PASS: 52 suites/384 tests; [Model](../../../../glossary.md#term-model) status-only Javadoc PASS.
MAIN compatibility PASS: consumer [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) and 5 suites/68 tests; no failures/errors/skips.
Evidence/acceptance [matrix](../../../../glossary.md#term-matrix-meanings)/[hashes](../../../../glossary.md#term-hash): `/private/tmp/synaptik-compiler0014.JPHriy/worker-handoff.txt`;
`independent-review.txt`, `main-acceptance.txt` and `main-compatibility-counts.json` in that directory.
Public [API](../../../../glossary.md#term-application-programming-interface-api)/[scope](../../../../glossary.md#term-scope)/links/diff PASS; baseline/index preserved. Model executable text unchanged.
No new types/[signatures](../../../../glossary.md#term-signature), [backend](../../../../glossary.md#backend) execution, cold/wire/[memo](../../../../glossary.md#cache-and-memoization) enablement, successor, [commit](../../../../glossary.md#term-commit) or push.
Status: Complete
