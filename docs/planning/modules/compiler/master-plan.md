# Compiler Master Plan

## Goal and authority

Compile Tensor expressions into immutable compile artifacts through phase-aware capture,
validation, exact graph transformation, compiler-owned automatic differentiation (autograd), and
Planning orchestration. This plan records implementation order and current gates; it does not
define architecture. [`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative.

Focused contracts:

- [core invariants](../../../../ARCHITECTURE.md#core-invariants) and
  [Compiler ownership](../../../architecture/contracts/compiler-autograd.md#modulescompiler);
- [compile artifacts](../../../architecture/contracts/compiler-autograd.md#compile-artifacts),
  [training graph](../../../architecture/contracts/compiler-autograd.md#training-graph-model), and
  [compiler-owned autograd](../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation);
- [compile lifecycle](../../../architecture/contracts/compiler-autograd.md#compile-lifecycle) and
  [fixed recurrent scan](../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions);
- [CPU generated-code ownership](../../../architecture/contracts/backend-execution.md#cpu-backend-routes),
  [architecture decision record (ADR) 0009](../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md),
  and [ADR 0012](../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md).

## Scope and non-goals

Compiler owns graph capture and indexing; Shape, data-type, and logical-layout inference and
validation; deterministic canonicalization and exact optimization; compiler-owned pre-capture
autograd and combined forward/backward graph construction; immutable publication, constant,
diagnostic, partition, and logical-memory artifacts; and complete valid backend-neutral graph
candidates for bounded tuning.

Compiler does not own Tensor gradient lifecycle state, model-owned derivative rules, another
gradient algebra, physical buffers, prepared schedules or executions, runtime state,
backend-specific lowering, generated JVM bytecode, or concrete kernel selection.

## Stable invariants

- Tensor expressions are Compiler ingress, not graph-local intermediate representation (IR).
  Request-local Tensor identity maps are ephemeral; one phase-aware capture assigns graph-local
  IDs once for the complete forward or combined graph.
- `CompiledGraphModel` and `CompileArtifacts` are immutable. Compiler may close only proved
  semantic descriptors and logical layouts; it never allocates or materializes physical storage.
- Per-node `GraphPhase` is authoritative. Exact rewrite/folding and dead-code elimination apply
  only under their proved guards, common-subexpression elimination (CSE) is phase-local, and every
  changed candidate is revalidated.
- Compiler owns autograd preflight, named derivative-rule dispatch, deterministic reverse
  accumulation, and derivative-order metadata. Generated formulas use ordinary public Tensor
  operations and the same inference, numerical semantics, validation, and exact optimization as
  forward expressions.
- Autograd is fail-closed before derivative allocation when an exact operation, attribute, role,
  Shape/data-type normalization, or derivative policy is unsupported. Boolean, index,
  random-number-generator (RNG) state, mask, configuration, and other declared
  non-differentiable roles receive no cotangent.
- Fixed recurrent neural network (RNN), gated recurrent unit (GRU), and long short-term memory
  (LSTM) scans remain identity-distinct ordinary flat multi-output nodes and are forward-only; CSE
  must not merge occurrences. Backpropagation through time (BPTT) requires a separate explicit
  decision.
- Conv3d remains one ordinary flat CSE-eligible batch/channel/depth/height/width (NCDHW) node.
  Shared grouped Conv2d/Conv3d adjoints now use public unfold, MATMUL, overlap-add fold and
  reductions, restoring each selected target's exact Shape/type. Backend closure is independent.
- Compiler owns graph-candidate semantics and validity. Tuning may measure bounded complete
  candidates but may not construct or reinterpret them.
- The supported public boundary stays narrow: immutable compile/publication contracts, the
  bounded functional gradient request/result contracts, and the constant-free
  `GraphCompilationPort` for module integration. Implementation entries and explicit-constant
  ingress remain package-private.
- Compiler orchestrates Planning but owns no execution, backend route, or kernel. CPU analysis
  owns generated-code lowering, specialization, route choice, and exact resource declarations;
  CPU finalization may generate and define the selected JVM kernel only after shared Prepare
  assigns slots.

## Dependencies and package map

Allowed dependencies: `modules/model`, `modules/config`, `modules/planning`,
`modules/backend-contract`, and `modules/trace`.

Forbidden dependencies: Runtime, Prepare, Engine, and concrete backend modules.

```text
io.github.pho001.synaptik.compiler/
  <root>  cohesive compiler front end: capture, inference/constraints, canonicalization,
          exact rewrites/folding, DCE/CSE, autograd rules and accumulation, immutable artifacts,
          Planning orchestration, functional gradient contracts, module-integration compile port,
          caller-input identity bindings, and family-specific inference owners
```

The root package must not become a catch-all public facade, pass or gradient registry, generic
algebra builder, planning adapter, backend lowering layer, or execution owner.

Implemented 0008 additions stay in that package: nested public `GraphCompilationPort.SourceCapture`
and package-private `SourceGraphReconstruction`; existing `GraphCapture` owns aligned leaf capture.
No new graph representation, module, dependency, generic validator or execution type.

0009 changes only existing `StaticResultLogicalLayoutClosure` in this package, routing SELECT to
`IndexingInference` and coordinate SLICE to `LayoutInference`; no new production type or owner.

Completed 0011 adds only package-private `ScanBodyInference` and `ScanGradientRules` in this existing
root package; all other changes extend existing owners. No public API, AD system or execution layer.

Completed 0013 extends these same inference/scan-rule owners, existing `AutogradPreflight`,
`FirstOrderAutograd`/`DerivativeConstants`, capture, final query and demand paths. Production
additions/moves: None. Bounded role-aware cotangent normalization/zero bindings use ordinary Tensor
algebra and existing closed protocols, not a new gradient framework or public abstraction.

[0014](tasks/0014-symbolic-scan-inference-ad-and-binding.md) extends these existing owners
and `BatchBindingPlan` for symbolic inference, actual first-order AD and scoped all-port binding.
Added/moved production types: None. No public signature or protocol change.
Forward inference retains accepted Model PURE/dtype breadth; bounded F32/F64 AD and later initial
CPU F32 execution are independent admissions. Compiler0014 is independently reviewed and MAIN
accepted 2026-10-25 after the scoped-binding compatibility checkpoint; it enables no execution.

## Task list

| ID | Task | Status | Depends on | One-line result or intent |
|---|---|---|---|---|
| 0001 | [Tensor expression graph capture](tasks/0001-tensor-expression-graph-capture.md) | Complete | Completed model graph/provenance/RNG-state foundations and model milestone closure | Added deterministic package-private forward capture with exact producer, output-slot, boundary, and opaque-state identity. |
| 0002 | [Captured-graph inference and validation](tasks/0002-captured-graph-inference-and-validation.md) | Complete | 0001 | Added independent descriptor verification and retained only unresolved typed Shape constraints. |
| 0003 | [Canonicalization and forward optimization](tasks/0003-canonicalization-and-forward-optimization.md) | Complete | 0002 | Added deterministic reindexing and config-controlled DCE/CSE/DCE with revalidation. |
| 0003A | [Exact arithmetic rewriting](tasks/0003a-exact-arithmetic-rewriting.md) | Complete | 0003 | Added the guarded seven-rule exact arithmetic matrix with no relaxed algebra. |
| 0003B | [Compile-time constants and constant folding](tasks/0003b-compile-time-constants-and-constant-folding.md) | Complete | 0003A | Added logical-splat ingress, bounded folding, and sidecar-aware constant pruning without storage reads. |
| 0004 | [Compiler-owned pre-capture autograd and graph compilation](tasks/0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md) | Complete | Model 0025; Compiler 0001–0003B; Config 0002 | Added fail-closed first-order preflight, Tensor-expression formulas, one combined capture, and phase-safe exact optimization. |
| 0004A | [Exact-composition gradient-rule extensions](tasks/0004a-exact-composition-gradient-rule-extensions.md) | Complete | 0004 | Added exact ERF, SUM, MATMUL, and selected data-movement adjoints. |
| 0004B | [Shared-algebra cotangent normalization and local derivative rules](tasks/0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md) | Complete | 0004A | Added mixed-floating cotangent normalization and selected DIV, MEAN, FLOOR, CEIL, and SIGN rules in the shared algebra. |
| 0005 | [Publication, planning orchestration, and compile artifacts](tasks/0005-publication-planning-orchestration-and-compile-artifacts.md) | Complete | 0001–0004B; Planning 0006 closure; stable Config 0001–0003 and Backend Contract 0001–0004 inputs | Added complete compile orchestration and immutable publication, constant, diagnostic, partition, and logical-memory artifacts. |
| 0005A | [Derivative policy and elementwise/activation gradient completion](tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md) | Complete | Model 0025A; Compiler 0005 | Closed the elementwise/activation derivative inventory and boundary policies. |
| 0005B | [Reduction, scan, softmax, statistics, and normalization gradient completion](tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md) | Complete | Model 0025B; 0005A | Closed binding-aware reduction, scan, softmax, statistics, norm, and normalization gradients. |
| 0005C | [Layout, window, indexing, scatter, ordering, and stochastic gradient completion](tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md) | Complete | Models 0025C–0025D; 0005B | Closed layout, window, indexing, scatter, ordering, and dropout gradients with dynamic obligations retained. |
| 0005D | [Attention, convolution, pooling, and loss gradient completion](tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md) | Complete | 0005B, 0005C | Closed the representable structured-ML gradient inventory and recorded intentional fail-closed cases. |
| 0005E | [First-order gradient coverage closure checkpoint](tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md) | Complete | 0005A, 0005B, 0005C, 0005D | Proved source-backed 37-family/107-kind/128-signature first-order role coverage and bounded transitive formula closure. |
| 0006 | [Explicit functional gradient requests and higher-order differentiation](tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md) | Complete | 0005E and the stable public compile/artifact boundary from 0005 | Added immutable one/two-stage functional requests, seeds, disconnected policy, ordered results, and derivative-order metadata. |
| 0006A | [Fixed recurrent-scan forward adoption and explicit BPTT boundary](tasks/0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md) | Complete | accepted ADR 0012 and NN 0021A architecture decision; Model 0025E–0025F; 0001–0006 | Added flat forward RNN/GRU/LSTM compilation and allocation-free fail-closed BPTT preflight. |
| 0006B | [Conv3d forward adoption and explicit gradient boundary](tasks/0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md) | Complete | Model 0025H; 0001–0006A | Added grouped NCDHW Conv3d forward compilation at 39/111/132 coverage while first-order support stayed 37/107/128 with four deferred signatures. |
| 0006B1 | [Pool3d and 3D-window forward adoption and explicit gradient boundary](tasks/0006b1-pool3d-and-3d-window-forward-adoption-and-explicit-gradient-boundary.md) | Complete | Model 0025J–0025K; 0006B | Added five Pool3d/3D-window forward signatures at 40/115/137 while first-order support stayed 37/107/128 with nine deferred signatures. |
| 0006B2 | [Pool3d and 3D-window gradient closure](tasks/0006b2-pool3d-and-3d-window-gradient-closure.md) | Complete | Model 0025K; 0006B1; 0005D | Closed five adjoints through Tensor algebra at 38/111/133 first-order support; only three recurrent signatures and Conv3d remain deferred. |
| 0006B3 | [Engine-facing complete compile integration port](tasks/0006b3-public-constant-free-complete-compile-entry.md) | Complete | 0005; 0006B2; current Config 0001–0003 leaves; Engine frontier reassessment | Added one public constant-free module-integration port while retaining package-private compile implementation. |
| 0006B4 | [Stable caller-input Tensor identity bindings](tasks/0006b4-stable-caller-input-tensor-identity-bindings.md) | Complete | 0005; 0006; 0006B3; blocked Engine 0003 reassessment | Added ordered caller `TensorId` to final `ValueId` bindable-input associations without retaining Tensor objects. |
| 0006B5 | [Published compile-time constant descriptor closure](tasks/0006b5-published-compile-time-constant-descriptor-closure.md) | Complete | 0006; 0006B4; Planning 0005–0006; Engine 0006 prerequisite diagnosis | Closed logical layouts for eligible static source-only published splat constants. |
| 0006B6 | [Final convolution logical-layout closure](tasks/0006b6-final-convolution-logical-layout-closure.md) | Complete | 0006B, 0006B3–0006B5; blocked Engine 0008 reassessment; current Model/Planning/CPU contracts | Closed eligible final Conv2d/Conv3d layouts and the direct Conv1d squeeze view before Planning queries. |
| 0006B7 | [Final NEG logical-layout closure](tasks/0006b7-final-neg-logical-layout-closure.md) | Complete | 0006B3–0006B6; Metal 0002 prerequisite review; current Model/Planning contracts | Closed eligible exact NEG and directly consumed splat-input layouts before final artifact derivation. |
| 0006B8 | [Bounded functional autograd contract and documentation reconciliation](tasks/0006b8-bounded-functional-autograd-contract-and-documentation-reconciliation.md) | Complete | 0006; completed Engine 0011; user-authorized drift-remediation sequence | Reconciled and independently reviewed the autograd contract and focused product documentation against current bounded one/two-stage behavior. |
| 0006B9 | [Compile-artifacts contract reconciliation](tasks/0006b9-compile-artifacts-contract-reconciliation.md) | Complete | 0006B8; completed Engine 0012; current `CompileArtifacts` source, test, and API references | Reconciled and independently reviewed the normative block against the exact public eight-component compile-time recipe. |
| 0006B10 | [Compile integration status reconciliation](tasks/0006b10-compile-integration-status-reconciliation.md) | Complete | 0006B9; completed Planning 0007 review; current Engine and Compiler integration source/API references | Reconciled and independently reviewed the integration SPI and shared compile status against current fixed CPU-only Engine composition. |
| 0006B11 | [Static-result logical-layout closure for public CPU execution](tasks/0006b11-static-result-logical-layout-closure-for-public-cpu-execution.md) | Complete | 0006B5–0006B10; current CPU complete-partition lowering; public Engine lifecycle | Completed canonical layouts only for explicitly proven materialized results and splats, preserved affine/view mappings, and restored public CPU compute and reusable prepared execution. |
| 0006C | Conv3d adjoint expressibility and gradient closure | Complete | 0006B; existing public 3D unfold/fold algebra; explicit user selection | Shared grouped Conv2d/Conv3d formula, checked metadata, selected-role and two-stage closure; CPU F32/F64 and Metal-only F32 numerical gradients, persisted NN training and full-body CPU Scan passed. Independent reviews APPROVE0, root test/build/distribution and fresh Maven/no-Git source consumers passed. |
| 0006D | [Autograd audit correctness repairs](tasks/0006d-autograd-audit-correctness-repairs.md) | Complete | 0006/0006B11; project review | Repair batched matrix/vector gradients and reject required fail-closed sibling paths; no new derivative capability. |
| 0007 | Exact constant identities under profile-free semantics | Draft | 0006; explicit legality proof under current Model semantics | Reassess exact identities without broadening constant evidence or using test tolerances as rewrite permission. |
| 0008 | [Cold source capture and reconstruction](tasks/0008-cold-source-capture-and-reconstruction.md) | Complete | Approved P2/ADR0039; completed capture/inference/integration boundaries; Training0007 serial base only | Bounded raw-source API and authority cutover implemented; final independent code/docs APPROVE and coordinator Class C checkpoint passed. |
| 0009 | [Static SELECT and SLICE logical layout closure](tasks/0009-static-select-and-slice-logical-layout-closure.md) | Complete | Compiler0006B11; Compiler0008; CPU0025 — Complete | Final independent APPROVE, zero findings; exact views support NN0029. Main's unchanged-snapshot native root gate passed (2026-10-19). |
| 0010 | [First RNN_TANH scan BPTT proposal](tasks/0010-rnn-tanh-scan-bptt.md) | Superseded | User selected general scan; Model0034 decision replaces this direction | Historical fixed-adjoint proposal/evidence retained; no fixed public adjoint selected. |
| 0011 | [General scan capture and first-order AD](tasks/0011-general-scan-capture-and-first-order-ad.md) | Complete | Model0034/0035 + Planning0008 Complete; Compiler0006A/0006D/0008/0009 baseline | Scoped capture/inference, joint first-order AD and final body queries/demand accepted by main 2026-10-22 after independent code/docs APPROVE and compatibility checkpoint. |
| 0012 | [General-scan cold source reconstruction](tasks/0012-general-scan-cold-source-reconstruction.md) | Complete | Approved NN0030; Complete Compiler0011/foundation and low-level join | Bounded cold body reconstruction accepted 2026-10-23 after independent review/final gates; later CP0012 and [NN0031](../../extensions/nn/tasks/0031-general-scan-owned-model-workflow.md#result) completed bounded consumer gates. Body wire stays closed. |
| 0013 | [Length-aware Scan capture and first-order AD](tasks/0013-length-aware-scan-capture-and-first-order-ad.md) | Complete | Model0037/0038, Compiler0011 and Planning0008 Complete; Compiler0012 cold baseline preserved | MAIN accepted 2026-10-24 after independent code/docs APPROVE, 0 P1/P2, worker module, compatibility and final documentation/Javadoc PASS; [evidence](tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result). No execution/cold/wire grant. |
| 0014 | [Symbolic Scan inference, first-order AD and scoped binding](tasks/0014-symbolic-scan-inference-ad-and-binding.md) | Complete | Model0041/0042, Compiler0011/0013, Planning0008 and Prepare0013 ordinary binding baseline Complete | MAIN accepted 2026-10-25 after independent code/docs APPROVE (0 P1/P2), module/Javadoc PASS (384 tests) and scoped compatibility PASS (68 tests). No backend enablement; evidence in brief. |
| 0015 | [Symbolic-length Scan inference, first-order AD and scoped binding](tasks/0015-symbolic-length-scan-inference-ad-and-binding.md#result) | Complete | Model0041/0044, Compiler0011/0013/0014, Planning0008 and Prepare0013 baseline Complete; pre-Java authority/Ready accepted | MAIN accepted 2026-10-26 after independent source/docs APPROVE and recorded final gates. Inference/actual sibling AD/fresh all-port binding only; no downstream execution grant. |
| 0016 | [Omit redundant MATMUL-gradient shape reductions](tasks/0016-matmul-gradient-noop-reduction-elision.md#result) | Complete | Compiler0015; CPU0045 accepted Complete | MAIN accepted 2026-10-28: exact-Shape normalization, preserved broadcast/cast order, Compiler 402 tests/Javadoc and CPU actual-Scan 101 tests PASS; context exception recorded. |
| 0017 | [Typed all-PURE Scan demand, first-order proof and activity](tasks/0017-per-value-typed-length-scan-inference-and-ad.md) | Ready | Complete Model0046/0047/0048, Compiler0017A/B and CPU0078/0080; frozen coupledCPU0081 interface, both independent authority APPROVE0 reviews | MAIN admits exact dirty-source/disjoint contexts for finite selected-pair/same-engine proof/typed history/component-eight control. Full all-valid/length source and actual qualification pending, no narrower milestone. |
| 0017A | [Ordinary lexical primal binding](tasks/0017a-ordinary-lexical-primal-binding.md#result) | Complete | Existing ordinary AD protocol; actual before witness; independent combined source APPROVE0 | Nine ordinary-owner files; 629 affected tests passed/20 CPU skips, all unchanged 18/84/12 lexical consumers and 84 ordinary regressions passed; no new Scan capability. |
| 0017B | [Shared checked ordinary gradient construction](tasks/0017b-shared-checked-ordinary-gradient-construction.md#result) | Complete | Complete0017A/Model0046/0047/CPU0078; final independent source/evidence/docs APPROVE0 | Seventeen paths integrated; affected2,463pass/60existing skips, original84numerical+18/84/12lexical, all-eleven-family metadata zero-ID CLI and85oldartifact bytes pass. No new Scan admission. |
| 0018 | [Static length-aware Scan cold reconstruction](tasks/0018-static-length-scan-cold-reconstruction.md#result) | Complete | Complete Compiler0012/0013/0015 and Model0038/0044; approved coordinated consumer authority | MAIN accepted independent code/docs APPROVE, 1025 affected tests/Javadocs and eight public cold roundtrip cases; Checkpoint/NN cold effects qualified, no v5 grant. |
| 0019 | [Static materializing RESHAPE logical closure](tasks/0019-static-materializing-reshape-logical-layout-closure.md#result) | Complete | Compiler0009; governing authority/ADR0062; final independent source/evidence/docs APPROVE0 | Exact two-file source integrated;418Compiler tests/Javadoc/jar and15actualcapturedgraphs pass. CPU/Metal physical execution remains separate; raw source/wire unchanged. |

## Dependency DAG and authorized frontiers

Compiler0016 is Complete, accepted by MAIN on 2026-10-28; its
[Result](tasks/0016-matmul-gradient-noop-reduction-elision.md#result) owns the focused/module/consumer
evidence and explicit user-approved context exception. Compiler0018 is Complete, accepted by MAIN
on 2026-11-01 after independent code/docs APPROVE, the public cold smoke and the recorded
Compiler/Checkpoint/NN gate. Compiler0019 is Complete after independent final acceptance and its
full module/captured-geometry proof. Compiler0017B ordinary construction parity is Complete after
final independent source/evidence/documentation acceptance and its recorded actual consumer gates.
[0017A ordinary lexical binding](tasks/0017a-ordinary-lexical-primal-binding.md#result) is Complete
after independent source/Javadoc APPROVE0, MAIN's affected gate and unchanged real lexical/
ordinary numerical before/after proof. The accepted cold/v5/practical RNN workflow remains Complete.

[Compiler0017](tasks/0017-per-value-typed-length-scan-inference-and-ad.md) remains Draft.
Its older length-only closed-inventory plan is not the mandatory all-PURE target. Complete
Model0046 resolves empty/no-ROW representation and Model0047 supplies shared ordinary guards.
Selected mixed cotangent demand, coordinate/presence proof and physical composition remain
substantive work. ADR0061's broader direction is not Ready; 0017B is the independent parity seam.
0017A completes the already-authorized ordinary lexical resolver independently, without enabling
new body operations or changing preflight/coverage. Its new manifest is distinct from the old
historical Compiler0017 preservation snapshot; that old evidence is not recaptured or relabeled.
Downstream/cold/wire/NN/higher-order domains remain unchanged until separately qualified.

Planning coordination: [Model0043](../model/tasks/0043-symbolic-length-scan-integration-design.md)
is Complete for the [combined F32/F64 Scan proposal](../../length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration),
DESIGN DELIVERY ONLY, MAIN accepted 2026-10-26 after Hume the 4th final Class C APPROVE
(0 P1/P2, no unresolved findings). Compiler0015 delivers the later approved inference/actual AD/scoped
binding phase together, preserving ROW(1) constant sequences and fresh all-port obligations.
At that design-delivery checkpoint, planning/source/authority frontiers were None and its serial
review/integration closed. Parallel group None; that delivery itself granted no authority or implementation.

Symbolic branch: Complete Model0041 authority → Complete Model0042 source, joining Complete
Compiler0011/0013 + Planning0008 + Prepare0013 ordinary binding → Compiler0014 →
[Prepare0014 vertical Complete](../prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result).
The latter is MAIN accepted 2026-10-25 after independent code/docs APPROVE and the configured
root gate. PlanningAuthorized=true; ImplementationAuthorized=true record historical admission;
Prepare/CPU/Engine frontiers None. User “jdi na to” approved source, MAIN verified Ready 2026-10-25 17:00 UTC.
Compiler production is read-only for that existing-owner Prepare/CPU/Engine packet; fresh full
scoped BatchBindingPlan.bind remains mandatory per invocation. MAIN owns serial integration,
Parallel group None. Compiler0014 is Complete, PlanningAuthorized=true,
ImplementationAuthorized=true records the user's historical source admission. MAIN verified all dependencies,
accepted base/contracts, inactive conflicts and sole frontier on 2026-10-25 before clean launch.
Seven integration fields and scoped validation are in its brief; Parallel group None, MAIN owns
integration. Independent planning/code/docs APPROVE, zero open P1/P2; module/Javadoc and MAIN
compatibility checkpoint PASS, MAIN accepted 2026-10-25. Implementation frontier: None;
Prepare0014 is Complete; Prepare/CPU/Engine frontiers None and Compiler production stayed read-only.

Immediate length-aware DAG: Model0037 authority → Model0038 source/IR, both Complete, joins
Complete Compiler0011 + Planning0008 → [Compiler0013 Complete](tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result).
MAIN accepted Compiler0013 on 2026-10-24 after independent code/docs APPROVE (0 P1/P2), worker
module/Javadoc, compatibility and final docs/Javadoc PASS. ImplementationAuthorized=true is historical admission.
Active frontier: None; no downstream successor or new Planning-module task is authorized.
Later Prepare/CPU execution and cold/v5 remain coarse owner gates, not a detailed task cascade.
0013's seven integration fields and Ready verification record the accepted uncommitted base,
inactive conflicts, Parallel group None and MAIN-owned serial integration/compatibility checkpoint.

Compiler0006B11 + Compiler0008 + CPU0025 → Compiler0009 → NN0029 → shared native root gate: all Complete.
At that completed checkpoint Compiler frontier was None; neither Compiler0009 nor CPU0025 depends
on NN0029. [0012 Result](tasks/0012-general-scan-cold-source-reconstruction.md#result) records Complete
bounded cold reconstruction, independent code/docs APPROVE and main's final gates (2026-10-23);
historical pre-code APPROVE/Ready verification remains in its brief. Later CP0012 copy/rebind and
[NN0031 owned workflow/native join](../../extensions/nn/tasks/0031-general-scan-owned-model-workflow.md#result)
are Complete; at that checkpoint Compiler/NN/Checkpoint frontiers were None, body wire closed, no successor launch.
[0011](tasks/0011-general-scan-capture-and-first-order-ad.md) is Complete.
[Prepare0010](../prepare/tasks/0010-general-scan-projection-and-resource-handoff.md) is Complete:
scoped projection/demand validation and resource handoff, main accepted 2026-10-22 after Huygens
final Class C code/docs APPROVE and the exact compatibility checkpoint. CPU/NN/Checkpoint frontiers: None;
Compiler0012 is Complete as recorded above; no successor launch.
Bounded CPU F32 pointwise forward + actual automatic reverse
[CPU0030](../../backends/cpu/tasks/0030-general-scan-f32-pointwise-vertical.md) is Complete, accepted
by main 2026-10-22 after independent review and focused checkpoint PASS. CPU0031–0033 declared
algebra/closure + F64 and low-level join are Complete; no NN/padding/persistence implementation grant.
Other Compiler tasks remain Draft/Superseded.

[0010](tasks/0010-rnn-tanh-scan-bptt.md) is Superseded by the
[Model0034 decision](../model/tasks/0034-general-differentiable-scan-decision.md) and
[general scan program](../../general-differentiable-scan.md). Model0034 authority decision/cutover
is Complete, accepted by main 2026-10-21 after Ramanujan the 2nd's final independent Class C
APPROVE, zero blockers; the brief owns result evidence. No authority frontier remains.
Base main=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`; Parallel group None;
main owns shared integration. DAG: approved authority → Model body/IR → Planning body query →
Compiler scoped capture/joint-output AD → Prepare projection → complete bounded CPU F32 pointwise
forward/automatic-reverse milestone → remaining declared forms/closure + F64 → one low-level root join.
Later padding/NN and persistence are separate gates. Model0035 source/IR is Complete, accepted by main
2026-10-22 after independent approval and its compatibility checkpoint; Model frontier None;
Model0034 owns seven metadata fields. Implemented `compiler.ScanBodyInference` and `ScanGradientRules`
stay in the existing root package. Actual user approval (“ok, tak tedy pokracuj”) covers the concrete
body/reverse-scan/residual/primal-binding proposal and authority cutover, not all Java slices.
The brief records historical verified Complete predecessors/base/inactive conflicts on 2026-10-21.
Planning0008 body query is Complete, accepted by main 2026-10-22 after final independent APPROVE and
compatibility gates; Planning frontier None. Compiler0011 is Complete, accepted by main 2026-10-22
after Euclid final independent Class C code/docs APPROVE and the compatibility checkpoint.
Prepare0010 projection/resource handoff and CPU0030 bounded F32 pointwise forward/actual-reverse
execution are Complete; CPU0031–0033 algebra/closure + F64 and low-level join are also Complete.
Compiler0012 is Complete; at that checkpoint Compiler/NN/Checkpoint frontiers were None, no successor launch.
Initial tensor/scalar floating DIV reuses ElementwiseGradientRules' ordinary derivatives; inventory
full generated closure (broadcast reductions, views/contiguous, constants) under existing family numerics.

0011 metadata: Depends on Model0034/0035, Planning0008 and Compiler0006A/0006D/0008/0009 Complete;
conflicts with capture/inference/AD/closure/query/shared-scan writers, none active. Parallel group None;
common base is HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus ALL accepted dirty work.
Integration order: frozen plan → independent plan review/main acceptance → explicit launch → clean
implementation → independent code/docs APPROVE → main's **Compiler0011 scoped-body compatibility checkpoint**.
The brief owns exact commands/evidence. Shared-document integration owner main;
Prepare0010's completed brief retains its historical Ready admission and implementation authorization.

Historical 0009 metadata: conflicts were closure/inference, scoped tests/docs, NN0029 and shared planning writers.
Parallel group: None. Common base: `7e880fb1044168043e45b10419a570d64c5283ae` plus the full dirty
snapshot, including accepted CP0009, frozen NN partial and completed CPU0025/Compiler0009; preserve all.
Integration order: reviewed Compiler repair → completed/reviewed NN0029 → main's one native-enabled
root gate passed on the unchanged reviewed snapshot (2026-10-19). Main owns integration; the 0009 brief records
worker validation and exact metadata. Existing contracts suffice; no authority cutover or extra plan review.

## P2 source reconstruction dependency

Joint [CP0010](../../extensions/checkpoint/tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md)
joins completed Compiler0008/0009, Checkpoint0008/0009, Training0008 and NN0028/0029 for bounded
fluent CNN/static-unroll persistence, with [CP0011 scan supplement](../../extensions/checkpoint/tasks/0011-flat-recurrent-scan-persistence.md)
using completed Model0025E/0025F/Compiler0006A. Both Complete on 2026-10-20 after independent
APPROVE and main's focused/native root gates; [CP0010 Result](../../extensions/checkpoint/tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md#result)
owns evidence. Raw inference stays separate from optimized layout closure; cold persistence grants
no execution or BPTT. CPU0028/CPU0029 and NN0021B subsequently completed forward execution only.
Base `8d5357e9a8698721a2252cb2ca3cb94150e705e8`;
at that checkpoint Compiler/Checkpoint/NN frontiers None, no successor or independent Compiler task;
Parallel group None, main integration.

The [source-verified brief](tasks/0008-cold-source-capture-and-reconstruction.md) supplies the
[program](../../model-execution-refactor.md#phases-and-owner-dag)'s Compiler prerequisite to
Checkpoint0008. GraphCompilationPort.captureSource returns the validated **raw** source graph and
exact input-aligned leaves from the same capture traversal, without optimization/layout closure.
reconstructSource preflights the bounded static FORWARD slice and calls ordinary Tensor constructors.
Descriptor-compatible replacement expressions support cold symbolic composition; captured leaves
remain provenance-free. SourceCapture is a short-lived integration sidecar, not another DAG/runner.
First slice: ADD/MUL/NEG/MATMUL/PERMUTE with exact [1,0] attributes for ordinary Linear; no all-family gate.

Root Explicit non-goals and compiler-autograd modules/compiler/Compile lifecycle/Compile artifacts
now enact only the call-local, post-preflight cold reconstruction table exception, discarded on
return/failure; no gradient-map exception, generic Tensor injection or Engine proxy.
Checkpoint0008 completed its separate Compiler-edge cutover; Compiler gained no dependency edge.
Existing autograd and planning remain unchanged.

Compiler0008 is Complete after Einstein's final independent code/docs APPROVE and the coordinator's
Class C checkpoint (2026-10-17); its brief owns evidence. At that checkpoint Compiler frontier was None.
Checkpoint0008, Training0008 and NN0028 are Complete, including high-level Model/withEngine;
the selected program stays Complete. Engine0021/Training0007 remain Complete. Completed execution
bases and evidence remain in their briefs, not new launch authorization.

## Atomic activation-source successor

[Checkpoint0009](../../extensions/checkpoint/tasks/0009-owned-model-activation-source-and-persistence.md)
is the completed cross-module capability adding RELU/TANH/SIGMOID to cold source admission and
reconstruction together with Checkpoint wire support and owned-Model integration tests/docs.
Status: Complete, accepted 2026-10-19 after Archimedes final APPROVE and main's integration/native root gates.
Compiler0008 + Checkpoint0008 + Training0008 + NN0028 (all Complete) → Checkpoint0009.
No separate source-widening task was needed: the former Checkpoint encoder's
fallback-to-PERMUTE required an atomic cutover. Compiler retains reconstruction ownership;
Checkpoint retains wire ownership. Existing compiler-package types suffice; no new production type.

Current policy: keep five-kind graph-model.v1 bytes/readers, use graph-model.v2 only when saving
added activations, and fail closed on unsupported version/kind pairs. Exact wires and the required
root/owning-contract cutover before Java are in the brief; no authority changes here.
Common base: main `7e880fb1044168043e45b10419a570d64c5283ae`; parallel group None. Shared admission,
wire and contract writes serialize; main owns shared documents and the combined integration gate.
At CP0009 completion, Checkpoint/Compiler/NN frontiers were None. The eight-kind/selective-v2 cutover is Complete;
the brief records final review/gate evidence and all seven integration fields. No follow-up remains.

## Milestones and current frontier

- Capture, validation, and exact forward optimization are Complete through 0003B.
- Compiler-owned autograd, current first-order inventory closure, and bounded two-stage functional
  differentiation are Complete through 0006.
- Recurrent, Conv3d, Pool3d/3D-window, public integration, caller binding, operation-aware static-
  result logical-layout completion, bounded-autograd documentation reconciliation, compile-artifact
  contract reconciliation, and compile-integration status reconciliation work are Complete through
  0006B11.
- Project-review repair 0006D is Complete after independent review and the combined checkpoint.
- Bounded raw source capture/reconstruction 0008 and its authority cutover are Complete after
  independent review and the coordinator checkpoint.
- Complete: 0009, the SELECT/coordinate-SLICE exact logical-view repair, accepted after final
  independent APPROVE, zero findings, and main's native root gate (2026-10-19). At that checkpoint Compiler/NN frontiers: None.
- 0006C is Complete under explicit user authorization. Shared formulas and coupled CPU/Metal
  execution passed numerical/train/persist programs, root test/build/distribution and fresh
  Maven/no-Git source consumers after independent APPROVE0. 0007 remains Draft and unauthorized.
- 0011 is Complete, accepted by main 2026-10-22 after independent code/docs APPROVE and compatibility checkpoint.
  Prepare0010 projection/resource handoff and CPU0030 bounded execution also Complete; CPU/NN/Checkpoint frontiers None;
  later domain-completion, low-level join, NN and persistence gates unchanged.
- Compiler0012 is Complete after independent code/docs APPROVE and main's final module/consumer gates;
  its Result owns evidence. Downstream CP0012 and [NN0031](../../extensions/nn/tasks/0031-general-scan-owned-model-workflow.md#result)
  separately completed bounded copy/rebind and owned workflow qualification, including the configured
  native program join; no body persistence or fixed-scan BPTT. At that checkpoint Compiler frontier was None.
- Compiler0013 is Complete, MAIN accepted 2026-10-24; independent code/docs APPROVE (0 P1/P2),
  worker module/Javadoc, compatibility and final docs/Javadoc PASS; its Result owns evidence.
  Ready frontier None; no downstream task is launched.

## Live gates and risks

- **Conv3d qualification:** 0006C implements group-isolated adjoints, stride/padding/dilation,
  overlap, symbolic Shape retention, selected roles and ordinary two-stage formula closure.
  First-order inventory is 39/112/134; only three fixed recurrent signatures remain deferred.
  CPU F32/F64 general Scan full-body forward/reverse composition passed all-valid/length-aware
  and both-direction numerical cases. This does not widen ordinary public CPU named-B Conv3d,
  CPU low arithmetic, Metal Scan or persistence inventories. MAIN accepted the source-matched root
  gate and fresh installed-artifact/no-Git-source runtime proof.
- **Recurrent differentiation:** 0006A covers forward compilation only. The exact RNN, GRU, and
  LSTM signatures remain outside gradient coverage. CPU0028/CPU0029 and
  [NN0021B](../../extensions/nn/tasks/0021b-fixed-recurrent-scan-implementation-program.md) are Complete
  for forward execution; BPTT remains unsupported. [0010](tasks/0010-rnn-tanh-scan-bptt.md) is
  Superseded by general scan, not a fixed-adjoint primitive. Completed Model0034 authority covers source/IR and original
  canonical residual ports, pre-capture joint-output body AD and truthful capability projection.
  Existing fixed scans and NN early training rejection remain unchanged; no successor implementation authorized.
- **Exact versus relaxed algebra:** preserve the guarded scalar `POW(+1) -> input`. Any exact
  `POW(0)` needs a typed shape-correct one-splat and proof across exceptional values, constant
  sidecars, publication identity, phase, autograd, and descriptors. Do not replace `POW(0.5)` with
  `SQRT` or infer Tensor constants from storage or factory history. Historical Config 0006 was
  removed by Model 0032; it is not a pending permission switch. A rewrite outside current Model
  semantics requires an explicit architecture decision; hardware or measurement never grants it.
- **Backend boundary:** the coupled [CPU](../../backends/cpu/master-plan.md) and
  [Metal](../../backends/metal/master-plan.md) Conv3d training closure implements Compiler's
  ordinary formulas; forward admission alone is still no derivative proof. Compiler absorbs no
  lowering, route, generated-code, physical storage or execution ownership. Fixed-scan BPTT stays closed.
- **Structural risks:** do not merge recurrent occurrences, split graph-local ID assignment across
  captures, merge CSE across phases, construct partial backward expressions before preflight,
  reconstruct canonical sibling outputs, or turn logical saved edges into a runtime tape.

## Status normalization

The task table and linked task `Status`/completion summaries take precedence over historical prose.
Current completion extends through 0006B11 plus independently reviewed 0006D, 0008, 0009, 0011 and 0012;
the historical coverage checkpoint is not proof of every value-level higher-order derivative.

## History and update policy

Detailed results, commands, inventories, audits, and past ordering exceptions remain in linked
task files and Git history. Read them only for a current unresolved question. Update this plan only
when Compiler ownership, package direction, task order/status, a live dependency or risk, or the
current frontier changes. Use the [planning guide](../../planning-guide.md) for any new or
materially revised brief, and stop if proposed work conflicts with the architecture contract.
