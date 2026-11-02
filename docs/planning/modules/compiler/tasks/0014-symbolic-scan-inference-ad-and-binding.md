# Task Compiler0014: Symbolic Scan inference, first-order AD and scoped binding

## Status

Complete — independently reviewed and MAIN accepted 2026-10-25 after the named compatibility checkpoint.
PlanningAuthorized=true; ImplementationAuthorized=true (historical). Implementation frontier: None.
Compiler metadata/actual first-order AD only; no symbolic backend execution is authorized.

## Change class

Class C — scoped inference, saved primals, automatic differentiation (AD) and metadata admission.
Clean implementation and independent review required.

## Goal

Extend existing Compiler owners to symbolic inference, actual first-order reverse construction and
fresh scoped all-port `BatchBindingPlan` binding. Bind metadata, not resources. Retain static
FLOAT32/FLOAT64 paths and ordinary dynamic-batch behavior.

## Scope

All-valid general SCAN, one canonical named B and fixed time T/features F/state widths H.
Inference independently checks accepted Model PURE semantics/applicable dtypes; do not import
the AD inventory or a CPU whitelist into forward inference. First-order AD is separately bounded
to homogeneous FLOAT32/FLOAT64 and the accepted algebra. Initial CPU FLOAT32 qualification is later.
Include necessary bounded rank-three ordinary outer seed/constant/pointwise/layout plumbing,
scoped private obligations and canonical actual descriptors for every Scan output position.

## Non-goals

No Model behavior/API/header change; Scan.java class support-status Javadoc only. No symbolic length mode, variable T/F/H, higher-order,
fixed-scan BPTT, arithmetic/rewrite widening, cold/copy/wire, NN/Training, Prepare/CPU/Engine
enablement or geometry memo. No native build, performance campaign, stage/commit/push or successor
launch. No graph/session/constant type, public signature, ABI or dependency change is selected.

## Contracts

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Compiler [Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding),
  [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation),
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle) and
  [Bounded general-scan cold reconstruction](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).
- Scan [Accepted symbolic-batch scan](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan),
  including ALL four subsections; [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities),
  [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand),
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings),
  [Header, bindings and source admission](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission)
  for retained static length mode. Stop on ambiguity; contracts win.

## Dependencies and integration

- Depends on: [Model0041](../../model/tasks/0041-symbolic-scan-authority-cutover.md#result),
  [Model0042](../../model/tasks/0042-symbolic-scan-source-foundation.md#result),
  [Compiler0011](0011-general-scan-capture-and-first-order-ad.md#result),
  [Compiler0013](0013-length-aware-scan-capture-and-first-order-ad.md#result),
  [Planning0008](../../planning/tasks/0008-general-scan-body-capability-query.md#result) and
  [Prepare0013 ordinary binding baseline](../../prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result), all Complete.
- Conflicts with: inference/capture/AD/binding/layout/query/demand, cold and shared Scan/Compiler
  contract/docs writers; future numeric vertical. None active.
- Parallel group: None; serial writer, read-only reviewer.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus ALL
  accepted dirty work including Model0042, not merged predecessors; empty index.
- Integration order: approved plan → verified Ready/source admission → clean implementation
  → independent code/docs approval → MAIN scoped-binding compatibility checkpoint → acceptance; no next launch.
- Integration validation: named Compiler0014 scoped-binding compatibility checkpoint below;
  one configured root gate belongs to the separately admitted numeric vertical join.
- Shared-document integration owner: MAIN; sole clean writer until freeze; reviewer read-only.

Ready verification (MAIN, 2026-10-25): explicit approval, all six dependencies/reviews/gates Complete,
stable authority/base, inactive conflicts and sole writer/frontier Compiler0014 verified.
Implementation baseline: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/pre-compiler0014-implementation.json`
and `-files/` copies. Preserve all accepted dirty work. No downstream launch, commit or push.

## Files and symbols

Production paths: `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/`.
MAIN admits only Model `Scan.java` class support-status Javadoc; stripped code must remain identical.
Added/moved production types: None proposed, not proof. If needed, stop with exact
package/name/purpose/signature and reasoning for MAIN/user approval; no silent expansion.

- `ScanBodyInference.java` (`capture`, `infer`, residual validation, `refreshDemand`) and
  `CapturedGraphInference.java` (`inferAndValidate`): all local declarations, ordinary family
  inference, actual/formal/result equality, typed obligations and independent scope validation.
- `BatchBindingPlan.java` (`prepare`, `bind`, `Binding.descriptor`, dimension/count recipes):
  private per-containing-occurrence/scope preindexing and exact SCAN all-port layout intent.
  `DeferredGraphConstraint.java` (`GraphPredicate`, `GraphPredicateProof.bind`) is the existing
  private predicate owner; preserve `CompileDiagnostics`/eight-artifact/public binding signatures.
- `ScanGradientRules.java` (`preflight`, `apply`, `formal`), `AutogradPreflight.java`,
  `FirstOrderAutograd.java` (`expandBody`, `DerivativeConstants`): symbolic closure and joint seeds.
  Existing `ElementwiseGradientRules`, `LinearAlgebraGradientRules`, `ReductionGradientRules`
  (SUM restoration, MEAN/count) and `LayoutGradientRules` own any necessary formula adjustments.
- `GraphCapture`, `GraphCompiler` final body queries, `StaticResultLogicalLayoutClosure` and existing
  optimization/reindex/demand owners: inspect/preserve phase, scoped IDs, saved sites and unresolved
  templates. Change only a demonstrated obstruction; no unrelated pass refactor or body optimizer.
- `GraphCompilationPort.java`/`SourceGraphReconstruction.java`: only preserve explicit cold rejection.
- Same-package tests: `ScanBodyInferenceTest`, `ScanGradientRulesTest`,
  `GeneralScanCompilerTest`, `BatchBindingPlanTest`, `StaticResultLogicalLayoutClosureTest` and
  retained `LengthAwareScanBodyInferenceTest`/`LengthAwareScanGradientRulesTest`/cold tests.
  Split blanket symbolic-rejection tests into admission/retained guards. Model `Scan`/`ScanBody` and
  `ScanSymbolicBatchSourceTest` are read-only source/shape references, not new owners to modify.

## Acceptance criteria

1. Independently infer every local value/formal, actual binding and canonical final/emission/residual
   port, including unused declarations and canonical siblings. Verify single named B, static T,
   shape/type/flags, canonical static versus unresolved symbolic layouts, retained order and gradient
   OR. Raw malformed IR tests bypass Model checks. Reused templates remain separate body scopes;
   coincident local/outer IDs or two occurrences' equal local IDs never share proof identity.
2. Forward inference retains accepted Model PURE/dtype breadth (for example low TANH/F32 RELU and
   discrete siblings), without claiming AD/provider support. Actual first-order AD independently
   closes tensor/scalar ADD/SUB/MUL/DIV, NEG/TANH/SIGMOID, rank-two MATMUL/transpose,
   SUM/sumToShape, constants and necessary expand/rank edits/contiguous/same-dtype casts.
   Prove named-axis preservation for feature edits; unproved B mixing rejects, not sample binding.
3. Generate the real opposite-traversal sibling once per occurrence with joint final/emission seeds,
   missing typed-zero seeds and explicit invariant constants. Load exact saved TANH/SIGMOID outputs,
   never replay them. Shared parameter-shaped additive carries contract actual B (dWx K=B), including
   bias unbroadcast at B=0/1. Preserve ties, repeated roles, cross-occurrence sums and false-flag carry
   flow. No B-by-parameter bank, handwritten reverse substitute or prototype MEAN denominator.
4. Ordinary outer `[T,B,*]` seeds/zeros/splats, pointwise/layout edits and scalar objective
   `mean(finalCarry)` close through existing owners. Count remains `oneLike(input).sum(...)` with
   actual B*H. Test actual generated expressions numerically against independent nonuniform ordinary
   Tensor unrolling in both traversals/F32/F64, joint and emission-only seeds, tied parameters,
   B=0/1/equal-to-F-or-H and 64→40→1→64 binding reuse. Extend only test-local bound evaluation
   of the actual symbolic expressions; do not rebuild a static reverse or use cold reconstruction.
5. `prepare` preindexes all local typed obligations by exact containing occurrence/scope, all extent
   validity and SCAN canonical actual-layout recipes. `bind` checks every input occurrence and every
   scoped obligation freshly on EVERY call, even at repeated B, including unused/dead declarations.
   Unknown symbolic multi-output kinds still reject. No diagnostic-string execution, new public
   predicate protocol, per-call graph walk/inference/recapture/optimization/backend decision or memo.
6. Resolve canonical non-view actual descriptors for ALL final/emission/residual ports, including
   dead siblings; preserve lazy dense outer value-list indexing and current supplied input-backed
   descriptors (`isView` included). Shared histories stay `[T,F,H]`, not `[T,B,F,H]`. Templates stay
   unresolved, static closure unchanged; refreshed queries/demand/phase/roles/saved sites survive
   standard/disabled optimization. Unknowns/foreign symbols/unproved expressions and overflow fail
   at the owning preflight/bind boundary, not deferred until an undemanded descriptor is requested.
7. Separate fixed-T=0 fixtures from T>0: no body work, distinct canonical raw final carry,
   dInitialCarry=seed, empty stacked ports/dX and typed-zero invariant sums. At B=0 do not globally
   skip nonempty shared work/parameter contractions; empty SUM/K=0 gives typed zero and empty MEAN
   gives NaN. Test B=0/T=0 separately and together; unsupported/overflowing dead bodies still reject.
   These are Compiler/test-oracle semantics, not physical bytes, runtime skipping or execution proof.
8. Deliberate whole-original backward preflight rejects unsupported rules/dtypes, residual roots/paths
   and higher-order/createGraph before derivative IDs, including unrelated branches/T=0/dead siblings.
   Unsupported forward metadata rejects independently. Retain static length-mode inference/AD and its
   exact header/roles, but reject symbolic length mode. Cold capture/reconstruct stays static-only
   with no IDs allocated on rejection, even B present only in an unused declaration/static outer ports.
   Keep actual backend/provider rejection and ordinary/static compatibility; no execution enablement.

## Validation

Worker focused development, then final affected-module run after source stabilization:

```bash
./gradlew --no-daemon :modules:compiler:test --tests '*Scan*Test' --tests '*BatchBindingPlanTest' --tests '*AutogradPreflightTest' --tests '*FirstOrderGradientCoverageTest' --tests '*StaticResultLogicalLayoutClosureTest' --tests '*SourceGraphReconstructionTest'
./gradlew --no-daemon :modules:compiler:test :modules:compiler:javadoc
```

Final module/Javadoc once after stabilization; reviewer reuses evidence.
MAIN runs the **Compiler0014 scoped-binding compatibility checkpoint** once after ordered review:

```bash
./gradlew --no-daemon :modules:prepare:compileTestJava :backends:cpu:compileTestJava :backends:metal:compileTestJava :modules:engine:compileTestJava :extensions:checkpoint:compileTestJava
./gradlew --no-daemon :modules:prepare:test --tests '*GraphPreparationTest' --tests '*BackendPartitionFinalizationHandoffTest' :backends:cpu:test --tests '*CpuGeneralScanCapabilityTest' :modules:engine:test --tests '*EngineOrdinaryDynamicBatchTest' :extensions:checkpoint:test --tests '*GraphCheckpointGeneralScanTest'
```

Performance opt-ins stay false; no native build/full-root/performance campaign. The later numeric
vertical owns backend conformance/Engine integration, specialized clean-Java/Class-File/performance
evidence and one configured root join. No dependency/ABI change.

## Documentation and review impact

Independent Class C review finalizes affected Javadoc and support status:
`docs/api/compile-api.md`, `docs/architecture/general-differentiable-scan.md`, root, owning Compiler/
Scan contracts, glossary, this brief/Compiler master/roadmap/dynamic-batch map and Model master/
symbolic-batch proposal. No new semantics/ADR/dependency. Search glossary `Symbolic Scan`/
`Symbolic batch binding`; update capability status only. Stop on new decisions.

## Result

MAIN accepted 2026-10-25 after independent Class C code/docs APPROVE, 0 open P1/P2.
Three existing owners implement scoped PURE inference, actual F32/F64 reverse and fresh all-port binding.
Nonuniform unroll witnesses cover both traversals/seeds, B=0/1/F/H, 64→40→1→64, actual MEAN,
ties, saved outputs, T=0 and shared work; malformed IR/dead obligations reject eagerly.
Compiler module/Javadoc PASS: 52 suites/384 tests; Model status-only Javadoc PASS.
MAIN compatibility PASS: consumer compilation and 5 suites/68 tests; no failures/errors/skips.
Evidence/acceptance matrix/hashes: `/private/tmp/synaptik-compiler0014.JPHriy/worker-handoff.txt`;
`independent-review.txt`, `main-acceptance.txt` and `main-compatibility-counts.json` in that directory.
Public API/scope/links/diff PASS; baseline/index preserved. Model executable text unchanged.
No new types/signatures, backend execution, cold/wire/memo enablement, successor, commit or push.
Status: Complete
