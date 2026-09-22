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
  Forward compilation is complete, but backward-capable requests remain fail-closed until task
  0006C proves and implements its exact adjoints.
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
| 0006C | Conv3d adjoint expressibility and gradient closure | Draft | 0006B; current public Tensor algebra; any separately selected Model prerequisite | Prove exact grouped NCDHW input/weight/bias adjoints; implement only if current public algebra closes every required case, otherwise select the smallest Model prerequisite. |
| 0007 | Exact constant identities and permission-aware algebra | Draft | 0006; Config 0006 before any relaxed rule | Reassess exact identities and separately permissioned relaxed rewrites without broadening current constant evidence or completed exact rules. |

## Milestones and current frontier

- Capture, validation, and exact forward optimization are Complete through 0003B.
- Compiler-owned autograd, current first-order inventory closure, and bounded two-stage functional
  differentiation are Complete through 0006.
- Recurrent, Conv3d, Pool3d/3D-window, public integration, caller binding, final logical-layout,
  and bounded-autograd documentation reconciliation work are Complete through 0006B8.
- The user-authorized drift-remediation sequence returned to Compiler after completed Engine 0011.
  Documentation-only 0006B8 completed its clean implementation and independent Class C review
  in exactly seven documentation/planning paths; the explicit repository-order interleave is
  closed.
- 0006C and 0007 remain `Draft`, have no detailed task brief, and are independent side branches;
  0006B8 grants neither implementation authorization.

## Live gates and risks

- **Conv3d adjoints:** 0006C must prove group isolation, dilation/padding, overlap accumulation,
  symbolic Shape behavior, and higher-order formula closure through the current public Tensor
  algebra. If any part is inexpressible, select the smallest prerequisite in the
  [Model plan](../model/master-plan.md) first and keep Conv3d fail-closed.
- **Recurrent differentiation:** 0006A covers forward compilation only. The exact RNN, GRU, and
  LSTM signatures remain outside gradient coverage; no BPTT task brief or saved-state policy
  exists. [Neural-network (NN) 0021B](../../extensions/nn/master-plan.md) also remains Draft until
  truthful concrete-backend recurrent execution is identified, and no current backend claims it.
- **Exact versus relaxed algebra:** preserve the guarded scalar `POW(+1) -> input`. Any exact
  `POW(0)` needs a typed shape-correct one-splat and proof across exceptional values, constant
  sidecars, publication identity, phase, autograd, and descriptors. Do not replace `POW(0.5)` with
  `SQRT` or infer Tensor constants from storage or factory history. Relaxed rules additionally wait
  for Draft [Config 0006](../config/master-plan.md) and its dependency chain; hardware or
  measurement never grants permission.
- **Backend boundary:** completed [CPU Conv3d execution](../../backends/cpu/master-plan.md) does not
  prove Conv3d adjoints or BPTT. Compiler must not absorb lowering, route selection, generated-code
  construction, physical layout/materialization, or execution ownership.
- **Structural risks:** do not merge recurrent occurrences, split graph-local ID assignment across
  captures, merge CSE across phases, construct partial backward expressions before preflight,
  reconstruct canonical sibling outputs, or turn logical saved edges into a runtime tape.

## Status normalization

The task table and linked task `Status`/completion summaries take precedence over historical prose.
The current roadmap records documentation-only 0006B8 as `Complete`; no next Compiler
implementation frontier is selected.

## History and update policy

Detailed results, commands, inventories, audits, and past ordering exceptions remain in linked
task files and Git history. Read them only for a current unresolved question. Update this plan only
when Compiler ownership, package direction, task order/status, a live dependency or risk, or the
current frontier changes. Use the [planning guide](../../planning-guide.md) for any new or
materially revised brief, and stop if proposed work conflicts with the architecture contract.
