# [Compiler](../../../glossary.md#term-compiler) Master Plan

> **Editorial reading key — not part of the historical record.**
>
> The application supplies Tensor expression roots through Engine; Engine composes the call into
> Compiler. Compiler captures identities, checks descriptions, constructs requested derivative
> expressions and orchestrates Planning to produce an immutable recipe. Backend providers answer
> capability questions; later preparation and execution own physical resources and values. A leaf
> means a Tensor with no producing expression, not necessarily a Tensor containing stored numbers.
> The task rows below retain their original milestone-specific statuses and evidence.
>
> Current reading: [structural capture and exact leaf identities](../../../reference/compiler/capture.md#current-package-private-structural-capture); [ordinary and advanced Engine compile boundaries](../../../reference/compiler/entry-points.md#current-ordinary-and-advanced-engine-compile-boundaries).

## Goal and authority

[Compile](../../../glossary.md#compile) [Tensor](../../../glossary.md#tensor) expressions into [immutable](../../../glossary.md#term-immutable) [compile artifacts](../../../glossary.md#compile-artifacts) through phase-aware [capture](../../../glossary.md#term-capture-meanings),
[validation](../../../glossary.md#term-validation), exact [graph](../../../glossary.md#graph) transformation, [compiler](../../../glossary.md#term-compiler)-owned [automatic differentiation](../../../glossary.md#autograd) ([autograd](../../../glossary.md#autograd)), and
[Planning](../../../glossary.md#term-planning) orchestration. This plan records implementation order and current gates; it does not
define architecture. [`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative.

Focused contracts:

- [core invariants](../../../../ARCHITECTURE.md#core-invariants) and
  [Compiler ownership](../../../architecture/contracts/compiler-autograd.md#modulescompiler)<span data-glossary-navigation="true"> ([ownership glossary](../../../glossary.md#term-ownership-meanings))</span>;
- [compile artifacts](../../../architecture/contracts/compiler-autograd.md#compile-artifacts),
  [training graph](../../../architecture/contracts/compiler-autograd.md#training-graph-model)<span data-glossary-navigation="true"> ([training graph glossary](../../../glossary.md#training-graph))</span>, and
  [compiler-owned autograd](../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation);
- [compile lifecycle](../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../glossary.md#lifecycle))</span> and
  [fixed recurrent scan](../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)<span data-glossary-navigation="true"> ([scan glossary](../../../glossary.md#term-cumulative-scan))</span>;
- [CPU generated-code ownership](../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../glossary.md#term-cpu))</span>,
  [architecture decision record (ADR) 0009](../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#adr-ci-and-qualification))</span><span data-glossary-navigation="true"> ([architecture decision record glossary](../../../glossary.md#term-architecture-decision-record-adr))</span>,
  and [ADR 0012](../../../design/decisions/0012-fixed-recurrent-scan-without-regions.md).

## [Scope](../../../glossary.md#term-scope) and non-goals

[Compiler](../../../glossary.md#term-compiler) owns [graph capture](../../../glossary.md#capture-graph-ports-and-reconstruction) and indexing; [Shape](../../../glossary.md#shape), data-type, and logical-[layout](../../../glossary.md#layout) [inference](../../../glossary.md#term-inference) and
[validation](../../../glossary.md#term-validation); deterministic [canonicalization](../../../glossary.md#occurrence-identity-and-canonical-form) and exact optimization; compiler-owned pre-[capture](../../../glossary.md#term-capture-meanings)
[autograd](../../../glossary.md#autograd) and combined forward/[backward graph](../../../glossary.md#backward-graph) construction; [immutable](../../../glossary.md#term-immutable) [publication](../../../glossary.md#term-publication), [constant](../../../glossary.md#term-constant),
diagnostic, [partition](../../../glossary.md#partition), and logical-memory [artifacts](../../../glossary.md#term-artifact); and complete valid [backend](../../../glossary.md#backend)-neutral [graph](../../../glossary.md#graph)
candidates for bounded tuning.

Compiler does not own [Tensor](../../../glossary.md#tensor) [gradient](../../../glossary.md#term-gradient) [lifecycle](../../../glossary.md#lifecycle) state, [model](../../../glossary.md#term-model)-owned [derivative](../../../glossary.md#term-derivative) rules, another
gradient algebra, physical [buffers](../../../glossary.md#term-buffer), prepared schedules or executions, [runtime](../../../glossary.md#term-runtime) state,
backend-specific [lowering](../../../glossary.md#lowering), generated [JVM](../../../glossary.md#term-java-virtual-machine-jvm) [bytecode](../../../glossary.md#term-bytecode), or concrete [kernel](../../../glossary.md#kernel) selection.

## Stable invariants

- Tensor expressions are Compiler ingress, not graph-local intermediate representation (IR).
  Request-local Tensor identity maps are ephemeral; one phase-aware capture assigns graph-local
  IDs once for the complete forward or combined graph.<span data-glossary-navigation="true"> ([IR glossary](../../../glossary.md#intermediate-representation-dag-and-ssa))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../glossary.md#tensor); [Compiler glossary](../../../glossary.md#term-compiler); [graph glossary](../../../glossary.md#graph); [intermediate representation glossary](../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../glossary.md#term-identity-meanings); [capture glossary](../../../glossary.md#term-capture-meanings))</span>
- [`CompiledGraphModel`](../../../glossary.md#compiled-graph--compiledgraphmodel) and `CompileArtifacts` are immutable. Compiler may close only proved
  semantic descriptors and logical layouts; it never allocates or materializes physical storage.<span data-glossary-navigation="true"> ([immutable glossary](../../../glossary.md#term-immutable); [descriptors glossary](../../../glossary.md#term-descriptor); [layouts glossary](../../../glossary.md#layout))</span>
- Per-[node](../../../glossary.md#node) `GraphPhase` is authoritative. Exact rewrite/folding and dead-code elimination apply
  only under their proved guards, common-subexpression elimination (CSE) is phase-local, and every
  changed candidate is revalidated.<span data-glossary-navigation="true"> ([CSE glossary](../../../glossary.md#compiler-transformations-and-dominance))</span><span data-glossary-navigation="true"> ([dead-code elimination glossary](../../../glossary.md#term-dead-code-elimination-dce); [common-subexpression elimination glossary](../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- Compiler owns autograd preflight, named derivative-rule dispatch, deterministic reverse
  accumulation, and derivative-order metadata. Generated formulas use ordinary public Tensor
  operations and the same inference, numerical semantics, validation, and exact optimization as
  forward expressions.<span data-glossary-navigation="true"> ([autograd glossary](../../../glossary.md#autograd); [preflight glossary](../../../glossary.md#term-preflight); [derivative glossary](../../../glossary.md#term-derivative); [dispatch glossary](../../../glossary.md#term-dispatch); [metadata glossary](../../../glossary.md#term-metadata); [inference glossary](../../../glossary.md#term-inference); [validation glossary](../../../glossary.md#term-validation))</span>
- Autograd is fail-closed before derivative allocation when an exact operation, attribute, role,
  Shape/data-type normalization, or derivative policy is unsupported. Boolean, index,
  random-number-generator (RNG) state, mask, configuration, and other declared
  non-differentiable roles receive no cotangent.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../glossary.md#term-fail-closed); [operation glossary](../../../glossary.md#operation); [Shape glossary](../../../glossary.md#shape); [normalization glossary](../../../glossary.md#term-normalization); [Boolean glossary](../../../glossary.md#data-type--datatype); [RNG glossary](../../../glossary.md#graph-rng-state--graphrngstate); [mask glossary](../../../glossary.md#term-boolean-mask); [differentiable glossary](../../../glossary.md#derivative-and-chain-rule); [cotangent glossary](../../../glossary.md#cotangent))</span>
- Fixed recurrent neural network (RNN), gated recurrent unit (GRU), and long short-term memory
  (LSTM) scans remain identity-distinct ordinary flat multi-output nodes and are forward-only; CSE
  must not merge occurrences. Backpropagation through time (BPTT) requires a separate explicit
  decision.<span data-glossary-navigation="true"> ([RNN glossary](../../../glossary.md#neural-network-families); [GRU glossary](../../../glossary.md#neural-network-families); [LSTM glossary](../../../glossary.md#neural-network-families); [BPTT glossary](../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span><span data-glossary-navigation="true"> ([recurrent neural network glossary](../../../glossary.md#term-recurrent-neural-network-rnn); [gated recurrent unit glossary](../../../glossary.md#term-gated-recurrent-unit-gru); [long short-term memory glossary](../../../glossary.md#term-long-short-term-memory-lstm); [Backpropagation through time glossary](../../../glossary.md#term-backpropagation-through-time-bptt))</span>
- Conv3d remains one ordinary flat CSE-eligible batch/channel/depth/height/width (NCDHW) node.
  Shared grouped Conv2d/Conv3d adjoints now use public unfold, MATMUL, overlap-add fold and
  reductions, restoring each selected target's exact Shape/type. Backend closure is independent.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../glossary.md#conv3d--ncdhw); [batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../../glossary.md#term-channel); [NCDHW glossary](../../../glossary.md#conv3d--ncdhw); [Conv2d glossary](../../../glossary.md#convolution--conv2d); [MATMUL glossary](../../../glossary.md#matrix-multiplication--matmul); [target glossary](../../../glossary.md#term-target); [Backend glossary](../../../glossary.md#backend))</span>
- Compiler owns graph-candidate semantics and validity. Tuning may measure bounded complete
  candidates but may not construct or reinterpret them.
- The supported public boundary stays narrow: immutable compile/publication contracts, the
  bounded functional gradient request/result contracts, and the constant-free<span data-glossary-navigation="true"> ([compile glossary](../../../glossary.md#compile); [publication glossary](../../../glossary.md#term-publication); [functional gradient request glossary](../../../glossary.md#functional-gradient-request); [constant glossary](../../../glossary.md#term-constant))</span>
  `GraphCompilationPort` for module integration. Implementation entries and explicit-constant
  ingress remain package-private.<span data-glossary-navigation="true"> ([module glossary](../../../glossary.md#term-module))</span>
- Compiler orchestrates Planning but owns no execution, backend route, or kernel. CPU analysis
  owns generated-code lowering, specialization, route choice, and exact resource declarations;
  CPU finalization may generate and define the selected JVM kernel only after shared Prepare
  assigns slots.<span data-glossary-navigation="true"> ([Planning glossary](../../../glossary.md#term-planning); [backend route glossary](../../../glossary.md#backend-route); [kernel glossary](../../../glossary.md#kernel); [CPU glossary](../../../glossary.md#term-cpu); [lowering glossary](../../../glossary.md#lowering); [specialization glossary](../../../glossary.md#term-specialization); [JVM glossary](../../../glossary.md#term-java-virtual-machine-jvm); [Prepare glossary](../../../glossary.md#prepare))</span>

## Dependencies and package map

Allowed dependencies: `modules/model`, `modules/config`, `modules/planning`,
`modules/backend-contract`, and `modules/trace`.

Forbidden dependencies: [Runtime](../../../glossary.md#term-runtime), [Prepare](../../../glossary.md#prepare), [Engine](../../../glossary.md#term-engine), and [concrete backend](../../../glossary.md#concrete-backend) [modules](../../../glossary.md#term-module).

```text
io.github.pho001.synaptik.compiler/
  <root>  cohesive compiler front end: capture, inference/constraints, canonicalization,
          exact rewrites/folding, DCE/CSE, autograd rules and accumulation, immutable artifacts,
          Planning orchestration, functional gradient contracts, module-integration compile port,
          caller-input identity bindings, and family-specific inference owners
```

The root package must not become a catch-all public [facade](../../../glossary.md#term-facade), pass or [gradient](../../../glossary.md#term-gradient) [registry](../../../glossary.md#term-registry), generic
algebra builder, [planning](../../../glossary.md#term-planning) adapter, [backend](../../../glossary.md#backend) [lowering](../../../glossary.md#lowering) layer, or execution [owner](../../../glossary.md#term-owner).

Implemented 0008 additions stay in that package: nested public `GraphCompilationPort.SourceCapture`
and package-private `SourceGraphReconstruction`; existing `GraphCapture` owns aligned [leaf](../../../glossary.md#term-leaf) [capture](../../../glossary.md#term-capture-meanings).
No new [graph](../../../glossary.md#graph) representation, module, [dependency](../../../glossary.md#term-dependency), generic validator or execution type.

0009 changes only existing `StaticResultLogicalLayoutClosure` in this package, routing SELECT to
`IndexingInference` and coordinate [SLICE](../../../glossary.md#slice) to `LayoutInference`; no new production type or owner.

Completed 0011 adds only package-private `ScanBodyInference` and `ScanGradientRules` in this existing
root package; all other changes extend existing owners. No public [API](../../../glossary.md#term-application-programming-interface-api), [AD](../../../glossary.md#autograd) system or execution layer.

Completed 0013 extends these same [inference](../../../glossary.md#term-inference)/[scan](../../../glossary.md#term-cumulative-scan)-rule owners, existing `AutogradPreflight`,
`FirstOrderAutograd`/`DerivativeConstants`, capture, final query and demand paths. Production
additions/moves: None. Bounded role-aware [cotangent normalization](../../../glossary.md#cotangent-normalization)/zero [bindings](../../../glossary.md#term-binding) use ordinary [Tensor](../../../glossary.md#tensor)
algebra and existing closed protocols, not a new gradient framework or public abstraction.

[0014](tasks/0014-symbolic-scan-inference-ad-and-binding.md) extends these existing owners
and `BatchBindingPlan` for symbolic inference, actual first-order AD and scoped all-port binding.
Added/moved production types: None. No public [signature](../../../glossary.md#term-signature) or protocol change.
Forward inference retains accepted [Model](../../../glossary.md#term-model) [PURE](../../../glossary.md#term-pure)/[dtype](../../../glossary.md#data-type--datatype) breadth; bounded [F32](../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../glossary.md#floating-point-precision-and-rounding) AD and later initial
[CPU](../../../glossary.md#term-cpu) F32 execution are independent admissions. Compiler0014 is independently reviewed and MAIN
accepted 2026-10-25 after the scoped-binding compatibility [checkpoint](../../../glossary.md#term-checkpoint); it enables no execution.

## Task list

| ID | Task | Status | Depends on | One-line result or intent |
|---|---|---|---|---|
| 0001 | [Tensor expression graph capture](tasks/0001-tensor-expression-graph-capture.md)<span data-glossary-navigation="true"> ([graph capture glossary](../../../glossary.md#capture-graph-ports-and-reconstruction))</span> | Complete | Completed [model](../../../glossary.md#term-model) [graph](../../../glossary.md#graph)/[provenance](../../../glossary.md#provenance)/[RNG](../../../glossary.md#graph-rng-state--graphrngstate)-state foundations and model milestone closure | Added deterministic package-private forward [capture](../../../glossary.md#term-capture-meanings) with exact [producer](../../../glossary.md#term-producer), output-slot, boundary, and [opaque](../../../glossary.md#term-opaque)-state [identity](../../../glossary.md#term-identity-meanings). |
| 0002 | [Captured-graph inference and validation](tasks/0002-captured-graph-inference-and-validation.md)<span data-glossary-navigation="true"> ([validation glossary](../../../glossary.md#term-validation))</span> | Complete | 0001 | Added independent [descriptor](../../../glossary.md#term-descriptor) verification and retained only unresolved typed [Shape](../../../glossary.md#shape) constraints. |
| 0003 | [Canonicalization and forward optimization](tasks/0003-canonicalization-and-forward-optimization.md)<span data-glossary-navigation="true"> ([Canonicalization glossary](../../../glossary.md#occurrence-identity-and-canonical-form))</span> | Complete | 0002 | Added deterministic reindexing and config-controlled [DCE](../../../glossary.md#term-dead-code-elimination-dce)/[CSE](../../../glossary.md#term-common-subexpression-elimination-cse)/DCE with revalidation. |
| 0003A | [Exact arithmetic rewriting](tasks/0003a-exact-arithmetic-rewriting.md) | Complete | 0003 | Added the guarded seven-rule exact arithmetic [matrix](../../../glossary.md#term-matrix-meanings) with no relaxed algebra. |
| 0003B | [Compile-time constants and constant folding](tasks/0003b-compile-time-constants-and-constant-folding.md)<span data-glossary-navigation="true"> ([constant folding glossary](../../../glossary.md#term-constant-folding))</span> | Complete | 0003A | Added logical-splat ingress, bounded folding, and sidecar-aware [constant](../../../glossary.md#term-constant) pruning without storage reads. |
| 0004 | [Compiler-owned pre-capture autograd and graph compilation](tasks/0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md) | Complete | Model 0025; [Compiler](../../../glossary.md#term-compiler) 0001–0003B; Config 0002 | Added [fail-closed](../../../glossary.md#term-fail-closed) first-order [preflight](../../../glossary.md#term-preflight), [Tensor](../../../glossary.md#tensor)-expression formulas, one combined capture, and phase-safe exact optimization. |
| 0004A | [Exact-composition gradient-rule extensions](tasks/0004a-exact-composition-gradient-rule-extensions.md) | Complete | 0004 | Added exact [ERF](../../../glossary.md#term-erf), SUM, [MATMUL](../../../glossary.md#matrix-multiplication--matmul), and selected data-movement adjoints. |
| 0004B | [Shared-algebra cotangent normalization and local derivative rules](tasks/0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md)<span data-glossary-navigation="true"> ([local derivative glossary](../../../glossary.md#derivative-and-chain-rule))</span> | Complete | 0004A | Added mixed-floating [cotangent normalization](../../../glossary.md#cotangent-normalization) and selected DIV, [MEAN](../../../glossary.md#term-mean), FLOOR, CEIL, and SIGN rules in the shared algebra. |
| 0005 | [Publication, planning orchestration, and compile artifacts](tasks/0005-publication-planning-orchestration-and-compile-artifacts.md)<span data-glossary-navigation="true"> ([compile artifacts glossary](../../../glossary.md#compile-artifacts))</span> | Complete | 0001–0004B; [Planning](../../../glossary.md#term-planning) 0006 closure; stable Config 0001–0003 and [Backend](../../../glossary.md#backend) Contract 0001–0004 inputs | Added complete [compile](../../../glossary.md#compile) orchestration and [immutable](../../../glossary.md#term-immutable) [publication](../../../glossary.md#term-publication), constant, diagnostic, [partition](../../../glossary.md#partition), and logical-memory [artifacts](../../../glossary.md#term-artifact). |
| 0005A | [Derivative policy and elementwise/activation gradient completion](tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md) | Complete | Model 0025A; Compiler 0005 | Closed the [elementwise](../../../glossary.md#term-elementwise)/[activation](../../../glossary.md#term-activation) [derivative](../../../glossary.md#term-derivative) inventory and boundary policies. |
| 0005B | [Reduction, scan, softmax, statistics, and normalization gradient completion](tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md) | Complete | Model 0025B; 0005A | Closed [binding](../../../glossary.md#term-binding)-aware reduction, [scan](../../../glossary.md#term-cumulative-scan), [softmax](../../../glossary.md#softmax--log-softmax), statistics, [norm](../../../glossary.md#term-norm), and [normalization](../../../glossary.md#term-normalization) [gradients](../../../glossary.md#term-gradient). |
| 0005C | [Layout, window, indexing, scatter, ordering, and stochastic gradient completion](tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md) | Complete | Models 0025C–0025D; 0005B | Closed [layout](../../../glossary.md#layout), window, indexing, scatter, ordering, and [dropout](../../../glossary.md#dropout--inverted-dropout) gradients with dynamic obligations retained. |
| 0005D | [Attention, convolution, pooling, and loss gradient completion](tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([convolution glossary](../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../glossary.md#term-pooling); [loss glossary](../../../glossary.md#loss))</span> | Complete | 0005B, 0005C | Closed the representable structured-[ML](../../../glossary.md#term-machine-learning-ml) gradient inventory and recorded intentional fail-closed cases. |
| 0005E | [First-order gradient coverage closure checkpoint](tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md) | Complete | 0005A, 0005B, 0005C, 0005D | Proved source-backed 37-family/107-kind/128-[signature](../../../glossary.md#term-signature) first-order role coverage and bounded transitive formula closure. |
| 0006 | [Explicit functional gradient requests and higher-order differentiation](tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md) | Complete | 0005E and the stable public compile/artifact boundary from 0005 | Added immutable one/two-stage functional requests, [seeds](../../../glossary.md#commit-release-distribution-log-and-seed-meanings), disconnected policy, ordered results, and derivative-order [metadata](../../../glossary.md#term-metadata). |
| 0006A | [Fixed recurrent-scan forward adoption and explicit BPTT boundary](tasks/0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md) | Complete | accepted [ADR](../../../glossary.md#term-architecture-decision-record-adr) 0012 and [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0021A architecture decision; Model 0025E–0025F; 0001–0006 | Added flat forward [RNN](../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../glossary.md#term-long-short-term-memory-lstm) [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names) and allocation-free fail-closed [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt) preflight. |
| 0006B | [Conv3d forward adoption and explicit gradient boundary](tasks/0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md) | Complete | Model 0025H; 0001–0006A | Added grouped [NCDHW](../../../glossary.md#conv3d--ncdhw) [Conv3d](../../../glossary.md#conv3d--ncdhw) forward compilation at 39/111/132 coverage while first-order support stayed 37/107/128 with four deferred signatures. |
| 0006B1 | [Pool3d and 3D-window forward adoption and explicit gradient boundary](tasks/0006b1-pool3d-and-3d-window-forward-adoption-and-explicit-gradient-boundary.md) | Complete | Model 0025J–0025K; 0006B | Added five [Pool3d](../../../glossary.md#pool3d--ncdhw-pooling)/3D-window forward signatures at 40/115/137 while first-order support stayed 37/107/128 with nine deferred signatures. |
| 0006B2 | [Pool3d and 3D-window gradient closure](tasks/0006b2-pool3d-and-3d-window-gradient-closure.md) | Complete | Model 0025K; 0006B1; 0005D | Closed five adjoints through Tensor algebra at 38/111/133 first-order support; only three recurrent signatures and Conv3d remain deferred. |
| 0006B3 | [Engine-facing complete compile integration port](tasks/0006b3-public-constant-free-complete-compile-entry.md) | Complete | 0005; 0006B2; current Config 0001–0003 leaves; [Engine](../../../glossary.md#term-engine) frontier reassessment | Added one public constant-free [module](../../../glossary.md#term-module)-[integration port](../../../glossary.md#term-integration-port) while retaining package-private compile implementation. |
| 0006B4 | [Stable caller-input Tensor identity bindings](tasks/0006b4-stable-caller-input-tensor-identity-bindings.md) | Complete | 0005; 0006; 0006B3; blocked Engine 0003 reassessment | Added ordered caller [`TensorId`](../../../glossary.md#tensorid) to final [`ValueId`](../../../glossary.md#logical-valueid-versus-physical-memory-slot) bindable-input associations without retaining Tensor objects. |
| 0006B5 | [Published compile-time constant descriptor closure](tasks/0006b5-published-compile-time-constant-descriptor-closure.md) | Complete | 0006; 0006B4; Planning 0005–0006; Engine 0006 prerequisite diagnosis | Closed logical layouts for eligible static source-only published splat constants. |
| 0006B6 | [Final convolution logical-layout closure](tasks/0006b6-final-convolution-logical-layout-closure.md) | Complete | 0006B, 0006B3–0006B5; blocked Engine 0008 reassessment; current Model/Planning/[CPU](../../../glossary.md#term-cpu) contracts | Closed eligible final [Conv2d](../../../glossary.md#convolution--conv2d)/Conv3d layouts and the direct Conv1d squeeze [view](../../../glossary.md#term-view-meanings) before Planning queries. |
| 0006B7 | [Final NEG logical-layout closure](tasks/0006b7-final-neg-logical-layout-closure.md) | Complete | 0006B3–0006B6; [Metal](../../../glossary.md#term-metal) 0002 prerequisite review; current Model/Planning contracts | Closed eligible exact NEG and directly consumed splat-input layouts before final artifact derivation. |
| 0006B8 | [Bounded functional autograd contract and documentation reconciliation](tasks/0006b8-bounded-functional-autograd-contract-and-documentation-reconciliation.md) | Complete | 0006; completed Engine 0011; user-authorized drift-remediation sequence | Reconciled and independently reviewed the [autograd](../../../glossary.md#autograd) contract and focused product documentation against current bounded one/two-stage behavior. |
| 0006B9 | [Compile-artifacts contract reconciliation](tasks/0006b9-compile-artifacts-contract-reconciliation.md) | Complete | 0006B8; completed Engine 0012; current `CompileArtifacts` source, test, and [API](../../../glossary.md#term-application-programming-interface-api) references | Reconciled and independently reviewed the [normative](../../../glossary.md#term-normative) block against the exact public eight-component compile-time recipe. |
| 0006B10 | [Compile integration status reconciliation](tasks/0006b10-compile-integration-status-reconciliation.md) | Complete | 0006B9; completed Planning 0007 review; current Engine and Compiler integration source/API references | Reconciled and independently reviewed the integration [SPI](../../../glossary.md#term-service-provider-interface-spi) and shared compile status against current fixed CPU-only Engine composition. |
| 0006B11 | [Static-result logical-layout closure for public CPU execution](tasks/0006b11-static-result-logical-layout-closure-for-public-cpu-execution.md) | Complete | 0006B5–0006B10; current CPU complete-partition [lowering](../../../glossary.md#lowering); public Engine [lifecycle](../../../glossary.md#lifecycle) | Completed [canonical](../../../glossary.md#term-canonical-form) layouts only for explicitly proven materialized results and splats, preserved affine/view mappings, and restored public CPU compute and reusable [prepared execution](../../../glossary.md#prepared-execution--preparedexecution). |
| 0006C | Conv3d [adjoint](../../../glossary.md#term-adjoint) expressibility and gradient closure | Complete | 0006B; existing public 3D unfold/fold algebra; explicit user selection | Shared grouped Conv2d/Conv3d formula, checked metadata, selected-role and two-stage closure; CPU [F32](../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../glossary.md#floating-point-precision-and-rounding) and Metal-only F32 numerical gradients, persisted NN training and full-body CPU Scan passed. Independent reviews APPROVE0, root test/build/[distribution](../../../glossary.md#term-distribution) and fresh [Maven](../../../glossary.md#term-maven)/no-Git source consumers passed. |
| 0006D | [Autograd audit correctness repairs](tasks/0006d-autograd-audit-correctness-repairs.md) | Complete | 0006/0006B11; project review | Repair batched matrix/[vector](../../../glossary.md#term-vector) gradients and reject required fail-closed sibling paths; no new derivative capability. |
| 0007 | Exact constant identities under profile-free semantics | Draft | 0006; explicit legality proof under current Model semantics | Reassess exact identities without broadening constant evidence or using test tolerances as rewrite permission. |
| 0008 | [Cold source capture and reconstruction](tasks/0008-cold-source-capture-and-reconstruction.md)<span data-glossary-navigation="true"> ([capture glossary](../../../glossary.md#term-structural-capture))</span> | Complete | Approved P2/ADR0039; completed capture/[inference](../../../glossary.md#term-inference)/integration boundaries; Training0007 serial base only | Bounded raw-source API and authority cutover implemented; final independent code/docs APPROVE and coordinator Class C [checkpoint](../../../glossary.md#term-checkpoint) passed. |
| 0009 | [Static SELECT and SLICE logical layout closure](tasks/0009-static-select-and-slice-logical-layout-closure.md)<span data-glossary-navigation="true"> ([SLICE glossary](../../../glossary.md#slice))</span> | Complete | Compiler0006B11; Compiler0008; CPU0025 — Complete | Final independent APPROVE, zero findings; exact views support NN0029. Main's unchanged-[snapshot](../../../glossary.md#term-snapshot) native root gate passed (2026-10-19). |
| 0010 | [First RNN_TANH scan BPTT proposal](tasks/0010-rnn-tanh-scan-bptt.md) | Superseded | User selected [general scan](../../../glossary.md#general-differentiable-scan); Model0034 decision replaces this direction | Historical fixed-adjoint proposal/evidence retained; no fixed public adjoint selected. |
| 0011 | [General scan capture and first-order AD](tasks/0011-general-scan-capture-and-first-order-ad.md) | Complete | Model0034/0035 + Planning0008 Complete; Compiler0006A/0006D/0008/0009 baseline | Scoped capture/inference, joint first-order [AD](../../../glossary.md#autograd) and final body queries/demand accepted by main 2026-10-22 after independent code/docs APPROVE and compatibility checkpoint. |
| 0012 | [General-scan cold source reconstruction](tasks/0012-general-scan-cold-source-reconstruction.md) | Complete | Approved NN0030; Complete Compiler0011/foundation and low-level join | Bounded cold body reconstruction accepted 2026-10-23 after independent review/final gates; later CP0012 and [NN0031](../../extensions/nn/tasks/0031-general-scan-owned-model-workflow.md#result) completed bounded consumer gates. Body wire stays closed. |
| 0013 | [Length-aware Scan capture and first-order AD](tasks/0013-length-aware-scan-capture-and-first-order-ad.md) | Complete | Model0037/0038, Compiler0011 and Planning0008 Complete; Compiler0012 cold baseline preserved | MAIN accepted 2026-10-24 after independent code/docs APPROVE, 0 P1/P2, worker module, compatibility and final documentation/[Javadoc](../../../glossary.md#term-javadoc) PASS; [evidence](tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result). No execution/cold/wire grant. |
| 0014 | [Symbolic Scan inference, first-order AD and scoped binding](tasks/0014-symbolic-scan-inference-ad-and-binding.md)<span data-glossary-navigation="true"> ([Symbolic Scan glossary](../../../glossary.md#symbolic-scan))</span> | Complete | Model0041/0042, Compiler0011/0013, Planning0008 and Prepare0013 ordinary binding baseline Complete | MAIN accepted 2026-10-25 after independent code/docs APPROVE (0 P1/P2), module/Javadoc PASS (384 tests) and scoped compatibility PASS (68 tests). No backend enablement; evidence in brief. |
| 0015 | [Symbolic-length Scan inference, first-order AD and scoped binding](tasks/0015-symbolic-length-scan-inference-ad-and-binding.md#result) | Complete | Model0041/0044, Compiler0011/0013/0014, Planning0008 and Prepare0013 baseline Complete; pre-Java authority/[Ready](../../../glossary.md#term-ready) accepted | MAIN accepted 2026-10-26 after independent source/docs APPROVE and recorded final gates. Inference/actual sibling AD/fresh all-port binding only; no downstream execution grant. |
| 0016 | [Omit redundant MATMUL-gradient shape reductions](tasks/0016-matmul-gradient-noop-reduction-elision.md#result) | Complete | Compiler0015; CPU0045 accepted Complete | MAIN accepted 2026-10-28: exact-Shape normalization, preserved broadcast/cast order, Compiler 402 tests/Javadoc and CPU actual-Scan 101 tests PASS; context exception recorded. |
| 0017 | [Typed all-PURE Scan demand, first-order proof and activity](tasks/0017-per-value-typed-length-scan-inference-and-ad.md)<span data-glossary-navigation="true"> ([PURE glossary](../../../glossary.md#term-pure))</span> | Ready | Complete Model0046/0047/0048, Compiler0017A/B and CPU0078/0080; frozen coupledCPU0081 interface, both independent authority APPROVE0 reviews | MAIN admits exact dirty-source/disjoint contexts for [finite](../../../glossary.md#term-finite) selected-pair/same-engine proof/typed history/component-eight control. Full all-valid/length source and actual [qualification](../../../glossary.md#term-qualification) pending, no narrower milestone. |
| 0017A | [Ordinary lexical primal binding](tasks/0017a-ordinary-lexical-primal-binding.md#result)<span data-glossary-navigation="true"> ([primal glossary](../../../glossary.md#term-primal))</span> | Complete | Existing ordinary AD protocol; actual before witness; independent combined source APPROVE0 | Nine ordinary-[owner](../../../glossary.md#term-owner) files; 629 affected tests passed/20 CPU skips, all unchanged 18/84/12 lexical consumers and 84 ordinary regressions passed; no new Scan capability. |
| 0017B | [Shared checked ordinary gradient construction](tasks/0017b-shared-checked-ordinary-gradient-construction.md#result) | Complete | Complete0017A/Model0046/0047/CPU0078; final independent source/evidence/docs APPROVE0 | Seventeen paths integrated; affected2,463pass/60existing skips, original84numerical+18/84/12lexical, all-eleven-family metadata zero-ID [CLI](../../../glossary.md#term-cli) and85oldartifact bytes pass. No new Scan [admission](../../../glossary.md#term-admission). |
| 0018 | [Static length-aware Scan cold reconstruction](tasks/0018-static-length-scan-cold-reconstruction.md#result)<span data-glossary-navigation="true"> ([cold reconstruction glossary](../../../glossary.md#term-cold-reconstruction))</span> | Complete | Complete Compiler0012/0013/0015 and Model0038/0044; approved coordinated consumer authority | MAIN accepted independent code/docs APPROVE, 1025 affected tests/Javadocs and eight public cold roundtrip cases; Checkpoint/NN cold effects qualified, no v5 grant. |
| 0019 | [Static materializing RESHAPE logical closure](tasks/0019-static-materializing-reshape-logical-layout-closure.md#result) | Complete | Compiler0009; governing authority/ADR0062; final independent source/evidence/docs APPROVE0 | Exact two-file source integrated;418Compiler tests/Javadoc/jar and15actualcapturedgraphs pass. CPU/Metal physical execution remains separate; raw source/wire unchanged. |

## [Dependency](../../../glossary.md#term-dependency) [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) and authorized frontiers

Compiler0016 is Complete, accepted by MAIN on 2026-10-28; its
[Result](tasks/0016-matmul-gradient-noop-reduction-elision.md#result) owns the focused/[module](../../../glossary.md#term-module)/consumer
evidence and explicit user-approved context exception. Compiler0018 is Complete, accepted by MAIN
on 2026-11-01 after independent code/docs APPROVE, the public cold [smoke](../../../glossary.md#term-smoke-test) and the recorded
[Compiler](../../../glossary.md#term-compiler)/[Checkpoint](../../../glossary.md#term-checkpoint)/[NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) gate. Compiler0019 is Complete after independent final acceptance and its
full module/captured-geometry proof. Compiler0017B ordinary construction parity is Complete after
final independent source/evidence/documentation acceptance and its recorded actual consumer gates.
[0017A ordinary lexical binding](tasks/0017a-ordinary-lexical-primal-binding.md#result) is Complete
after independent source/[Javadoc](../../../glossary.md#term-javadoc) APPROVE0, MAIN's affected gate and unchanged real lexical/
ordinary numerical before/after proof. The accepted cold/v5/practical [RNN](../../../glossary.md#term-recurrent-neural-network-rnn) workflow remains Complete.

[Compiler0017](tasks/0017-per-value-typed-length-scan-inference-and-ad.md) remains Draft.
Its older length-only closed-inventory plan is not the mandatory all-[PURE](../../../glossary.md#term-pure) [target](../../../glossary.md#term-target). Complete
Model0046 resolves empty/no-ROW representation and Model0047 supplies shared ordinary guards.
Selected mixed [cotangent](../../../glossary.md#cotangent) demand, [coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix)/presence proof and physical composition remain
substantive work. ADR0061's broader direction is not [Ready](../../../glossary.md#term-ready); 0017B is the independent parity seam.
0017A completes the already-authorized ordinary lexical resolver independently, without enabling
new body operations or changing [preflight](../../../glossary.md#term-preflight)/coverage. Its new [manifest](../../../glossary.md#term-manifest) is distinct from the old
historical Compiler0017 preservation [snapshot](../../../glossary.md#term-snapshot); that old evidence is not recaptured or relabeled.
Downstream/cold/wire/NN/higher-order domains remain unchanged until separately qualified.

[Planning](../../../glossary.md#term-planning) coordination: [Model0043](../model/tasks/0043-symbolic-length-scan-integration-design.md)
is Complete for the [combined F32/F64 Scan proposal](../../length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration),
DESIGN DELIVERY ONLY, MAIN accepted 2026-10-26 after Hume the 4th final Class C APPROVE
(0 P1/P2, no unresolved findings). Compiler0015 delivers the later approved [inference](../../../glossary.md#term-inference)/actual [AD](../../../glossary.md#autograd)/scoped
[binding](../../../glossary.md#term-binding) phase together, preserving ROW(1) [constant](../../../glossary.md#term-constant) sequences and fresh all-port obligations.
At that design-delivery checkpoint, planning/source/authority frontiers were None and its serial
review/integration closed. Parallel group None; that delivery itself granted no authority or implementation.

Symbolic branch: Complete Model0041 authority → Complete Model0042 source, joining Complete
Compiler0011/0013 + Planning0008 + Prepare0013 ordinary binding → Compiler0014 →
[Prepare0014 vertical Complete](../prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result).
The latter is MAIN accepted 2026-10-25 after independent code/docs APPROVE and the configured
root gate. PlanningAuthorized=true; ImplementationAuthorized=true record historical [admission](../../../glossary.md#term-admission);
[Prepare](../../../glossary.md#prepare)/[CPU](../../../glossary.md#term-cpu)/[Engine](../../../glossary.md#term-engine) frontiers None. User “jdi na to” approved source, MAIN verified Ready 2026-10-25 17:00 UTC.
Compiler production is read-only for that existing-[owner](../../../glossary.md#term-owner) Prepare/CPU/Engine packet; fresh full
scoped BatchBindingPlan.bind remains mandatory per invocation. MAIN owns serial integration,
Parallel group None. Compiler0014 is Complete, PlanningAuthorized=true,
ImplementationAuthorized=true records the user's historical source admission. MAIN verified all dependencies,
accepted base/contracts, inactive conflicts and sole frontier on 2026-10-25 before clean launch.
Seven integration fields and scoped [validation](../../../glossary.md#term-validation) are in its brief; Parallel group None, MAIN owns
integration. Independent planning/code/docs APPROVE, zero open P1/P2; module/Javadoc and MAIN
compatibility checkpoint PASS, MAIN accepted 2026-10-25. Implementation frontier: None;
Prepare0014 is Complete; Prepare/CPU/Engine frontiers None and Compiler production stayed read-only.

Immediate length-aware DAG: Model0037 authority → Model0038 source/[IR](../../../glossary.md#term-intermediate-representation-ir), both Complete, joins
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
bounded [cold reconstruction](../../../glossary.md#term-cold-reconstruction), independent code/docs APPROVE and main's final gates (2026-10-23);
historical pre-code APPROVE/Ready verification remains in its brief. Later CP0012 copy/[rebind](../../../glossary.md#capture-graph-ports-and-reconstruction) and
[NN0031 owned workflow/native join](../../extensions/nn/tasks/0031-general-scan-owned-model-workflow.md#result)
are Complete; at that checkpoint Compiler/NN/Checkpoint frontiers were None, body wire closed, no successor launch.
[0011](tasks/0011-general-scan-capture-and-first-order-ad.md) is Complete.
[Prepare0010](../prepare/tasks/0010-general-scan-projection-and-resource-handoff.md) is Complete:
scoped projection/demand validation and resource [handoff](../../../glossary.md#term-handoff), main accepted 2026-10-22 after Huygens
final Class C code/docs APPROVE and the exact compatibility checkpoint. CPU/NN/Checkpoint frontiers: None;
Compiler0012 is Complete as recorded above; no successor launch.
Bounded CPU [F32](../../../glossary.md#floating-point-precision-and-rounding) [pointwise](../../../glossary.md#term-pointwise) forward + actual automatic reverse
[CPU0030](../../backends/cpu/tasks/0030-general-scan-f32-pointwise-vertical.md) is Complete, accepted
by main 2026-10-22 after independent review and focused checkpoint PASS. CPU0031–0033 declared
algebra/closure + [F64](../../../glossary.md#floating-point-precision-and-rounding) and low-level join are Complete; no NN/[padding](../../../glossary.md#padding)/[persistence](../../../glossary.md#term-persistence) implementation grant.
Other Compiler tasks remain Draft/Superseded.

[0010](tasks/0010-rnn-tanh-scan-bptt.md) is Superseded by the
[Model0034 decision](../model/tasks/0034-general-differentiable-scan-decision.md) and
[general scan program](../../general-differentiable-scan.md)<span data-glossary-navigation="true"> ([general scan glossary](../../../glossary.md#general-differentiable-scan))</span>. Model0034 authority decision/cutover
is Complete, accepted by main 2026-10-21 after Ramanujan the 2nd's final independent Class C
APPROVE, zero blockers; the brief owns result evidence. No authority frontier remains.
Base main=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`; Parallel group None;
main owns shared integration. DAG: approved authority → [Model](../../../glossary.md#term-model) body/IR → Planning body query →
Compiler scoped [capture](../../../glossary.md#term-capture-meanings)/joint-output AD → Prepare projection → complete bounded CPU F32 pointwise
forward/automatic-reverse milestone → remaining declared forms/closure + F64 → one low-level root join.
Later padding/NN and persistence are separate gates. Model0035 source/IR is Complete, accepted by main
2026-10-22 after independent approval and its compatibility checkpoint; Model frontier None;
Model0034 owns seven [metadata](../../../glossary.md#term-metadata) fields. Implemented `compiler.ScanBodyInference` and `ScanGradientRules`
stay in the existing root package. Actual user approval (“ok, tak tedy pokracuj”) covers the concrete
body/reverse-[scan](../../../glossary.md#term-cumulative-scan)/[residual](../../../glossary.md#scan-residual-port-and-history-demand)/[primal](../../../glossary.md#term-primal)-binding proposal and authority cutover, not all Java slices.
The brief records historical verified Complete predecessors/base/inactive conflicts on 2026-10-21.
Planning0008 body query is Complete, accepted by main 2026-10-22 after final independent APPROVE and
compatibility gates; Planning frontier None. Compiler0011 is Complete, accepted by main 2026-10-22
after Euclid final independent Class C code/docs APPROVE and the compatibility checkpoint.
Prepare0010 projection/resource handoff and CPU0030 bounded F32 pointwise forward/actual-reverse
execution are Complete; CPU0031–0033 algebra/closure + F64 and low-level join are also Complete.
Compiler0012 is Complete; at that checkpoint Compiler/NN/Checkpoint frontiers were None, no successor launch.
Initial [tensor](../../../glossary.md#tensor)/[scalar](../../../glossary.md#term-scalar) floating DIV reuses ElementwiseGradientRules' ordinary [derivatives](../../../glossary.md#term-derivative); inventory
full generated closure (broadcast reductions, [views](../../../glossary.md#term-view-meanings)/[contiguous](../../../glossary.md#term-contiguous), constants) under existing family numerics.

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

## P2 source reconstruction [dependency](../../../glossary.md#term-dependency)

Joint [CP0010](../../extensions/checkpoint/tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md)
joins completed Compiler0008/0009, Checkpoint0008/0009, Training0008 and NN0028/0029 for bounded
fluent [CNN](../../../glossary.md#term-convolutional-neural-network-cnn)/static-unroll [persistence](../../../glossary.md#term-persistence), with [CP0011 scan supplement](../../extensions/checkpoint/tasks/0011-flat-recurrent-scan-persistence.md)<span data-glossary-navigation="true"> ([scan glossary](../../../glossary.md#term-cumulative-scan))</span>
using completed Model0025E/0025F/Compiler0006A. Both Complete on 2026-10-20 after independent
APPROVE and main's focused/native root gates; [CP0010 Result](../../extensions/checkpoint/tasks/0010-fluent-cnn-and-static-recurrent-graph-persistence.md#result)
owns evidence. Raw [inference](../../../glossary.md#term-inference) stays separate from optimized [layout](../../../glossary.md#layout) closure; cold persistence grants
no execution or [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt). CPU0028/CPU0029 and NN0021B subsequently completed forward execution only.
Base `8d5357e9a8698721a2252cb2ca3cb94150e705e8`;
at that [checkpoint](../../../glossary.md#term-checkpoint) [Compiler](../../../glossary.md#term-compiler)/Checkpoint/[NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) frontiers None, no successor or independent Compiler task;
Parallel group None, main integration.

The [source-verified brief](tasks/0008-cold-source-capture-and-reconstruction.md) supplies the
[program](../../model-execution-refactor.md#phases-and-owner-dag)'s Compiler prerequisite to
Checkpoint0008. GraphCompilationPort.captureSource returns the validated **raw** source [graph](../../../glossary.md#graph) and
exact input-aligned leaves from the same [capture](../../../glossary.md#term-capture-meanings) traversal, without optimization/layout closure.
reconstructSource preflights the bounded static FORWARD [slice](../../../glossary.md#slice) and calls ordinary [Tensor](../../../glossary.md#tensor) constructors.
[Descriptor](../../../glossary.md#term-descriptor)-compatible replacement expressions support cold symbolic composition; captured leaves
remain [provenance](../../../glossary.md#provenance)-free. SourceCapture is a short-lived integration sidecar, not another [DAG](../../../glossary.md#term-directed-acyclic-graph-dag)/runner.
First slice: ADD/MUL/NEG/[MATMUL](../../../glossary.md#matrix-multiplication--matmul)/PERMUTE with exact [1,0] attributes for ordinary [Linear](../../../glossary.md#term-linear-meanings); no all-family gate.

Root Explicit non-goals and compiler-[autograd](../../../glossary.md#autograd) [modules](../../../glossary.md#term-module)/compiler/[Compile](../../../glossary.md#compile) [lifecycle](../../../glossary.md#lifecycle)/[Compile artifacts](../../../glossary.md#compile-artifacts)
now enact only the call-local, post-[preflight](../../../glossary.md#term-preflight) [cold reconstruction](../../../glossary.md#term-cold-reconstruction) table exception, discarded on
return/failure; no [gradient](../../../glossary.md#term-gradient)-map exception, generic Tensor injection or [Engine](../../../glossary.md#term-engine) proxy.
Checkpoint0008 completed its separate Compiler-edge cutover; Compiler gained no [dependency edge](../../../glossary.md#term-dependency-edge).
Existing autograd and [planning](../../../glossary.md#term-planning) remain unchanged.

Compiler0008 is Complete after Einstein's final independent code/docs APPROVE and the coordinator's
Class C checkpoint (2026-10-17); its brief owns evidence. At that checkpoint Compiler frontier was None.
Checkpoint0008, Training0008 and NN0028 are Complete, including high-level [Model](../../../glossary.md#term-model)/withEngine;
the selected program stays Complete. Engine0021/Training0007 remain Complete. Completed execution
bases and evidence remain in their briefs, not new launch authorization.

## [Atomic](../../../glossary.md#resource-transaction-and-rollback) [activation](../../../glossary.md#term-activation)-source successor

[Checkpoint0009](../../extensions/checkpoint/tasks/0009-owned-model-activation-source-and-persistence.md)
is the completed cross-[module](../../../glossary.md#term-module) capability adding RELU/[TANH](../../../glossary.md#term-tanh)/[SIGMOID](../../../glossary.md#term-sigmoid) to cold source [admission](../../../glossary.md#term-admission) and
reconstruction together with [Checkpoint](../../../glossary.md#term-checkpoint) wire support and owned-[Model](../../../glossary.md#term-model) integration tests/docs.
Status: Complete, accepted 2026-10-19 after Archimedes final APPROVE and main's integration/native root gates.
Compiler0008 + Checkpoint0008 + Training0008 + NN0028 (all Complete) → Checkpoint0009.
No separate source-[widening](../../../glossary.md#term-widening) task was needed: the former Checkpoint encoder's
fallback-to-PERMUTE required an atomic cutover. [Compiler](../../../glossary.md#term-compiler) retains reconstruction [ownership](../../../glossary.md#term-ownership-meanings);
Checkpoint retains wire ownership. Existing compiler-package types suffice; no new production type.

Current policy: keep five-kind [graph](../../../glossary.md#graph)-model.v1 bytes/readers, use graph-model.v2 only when saving
added activations, and [fail closed](../../../glossary.md#admission-preflight-and-fail-closed-validation) on unsupported version/kind pairs. Exact wires and the required
root/owning-contract cutover before Java are in the brief; no authority changes here.
Common base: main `7e880fb1044168043e45b10419a570d64c5283ae`; parallel group None. Shared admission,
wire and contract writes serialize; main owns shared documents and the combined integration gate.
At CP0009 completion, Checkpoint/Compiler/[NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) frontiers were None. The eight-kind/selective-v2 cutover is Complete;
the brief records final review/gate evidence and all seven integration fields. No follow-up remains.

## Milestones and current frontier

- [Capture](../../../glossary.md#term-capture-meanings), [validation](../../../glossary.md#term-validation), and exact forward optimization are Complete through 0003B.
- Compiler-owned autograd, current first-order inventory closure, and bounded two-stage functional
  differentiation are Complete through 0006.<span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler); [autograd glossary](../../../glossary.md#autograd))</span>
- Recurrent, Conv3d, Pool3d/3D-window, public integration, caller binding, operation-aware static-
  result logical-layout completion, bounded-autograd documentation reconciliation, compile-artifact
  contract reconciliation, and compile-integration status reconciliation work are Complete through
  0006B11.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../glossary.md#conv3d--ncdhw); [Pool3d glossary](../../../glossary.md#pool3d--ncdhw-pooling); [binding glossary](../../../glossary.md#term-binding); [operation glossary](../../../glossary.md#operation); [layout glossary](../../../glossary.md#layout); [compile glossary](../../../glossary.md#compile); [artifact glossary](../../../glossary.md#term-artifact))</span>
- Project-review repair 0006D is Complete after independent review and the combined [checkpoint](../../../glossary.md#term-checkpoint).
- Bounded raw source capture/reconstruction 0008 and its authority cutover are Complete after
  independent review and the coordinator checkpoint.<span data-glossary-navigation="true"> ([capture glossary](../../../glossary.md#term-structural-capture))</span>
- Complete: 0009, the SELECT/coordinate-SLICE exact logical-view repair, accepted after final
  independent APPROVE, zero findings, and main's native root gate (2026-10-19). At that checkpoint Compiler/NN frontiers: None.<span data-glossary-navigation="true"> ([coordinate glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix); [SLICE glossary](../../../glossary.md#slice); [view glossary](../../../glossary.md#term-view-meanings); [NN glossary](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- 0006C is Complete under explicit user authorization. Shared formulas and coupled CPU/Metal
  execution passed numerical/train/persist programs, root test/build/distribution and fresh
  Maven/no-Git source consumers after independent APPROVE0. 0007 remains Draft and unauthorized.<span data-glossary-navigation="true"> ([CPU glossary](../../../glossary.md#term-cpu); [Metal glossary](../../../glossary.md#term-metal); [distribution glossary](../../../glossary.md#term-distribution); [Maven glossary](../../../glossary.md#term-maven))</span>
- 0011 is Complete, accepted by main 2026-10-22 after independent code/docs APPROVE and compatibility checkpoint.
  Prepare0010 projection/resource handoff and CPU0030 bounded execution also Complete; CPU/NN/Checkpoint frontiers None;
  later domain-completion, low-level join, NN and persistence gates unchanged.<span data-glossary-navigation="true"> ([handoff glossary](../../../glossary.md#term-handoff); [persistence glossary](../../../glossary.md#term-persistence))</span>
- Compiler0012 is Complete after independent code/docs APPROVE and main's final module/consumer gates;
  its Result owns evidence. Downstream CP0012 and <span data-glossary-navigation="true"> ([module glossary](../../../glossary.md#term-module))</span>[NN0031](../../extensions/nn/tasks/0031-general-scan-owned-model-workflow.md#result)
  separately completed bounded copy/rebind and owned workflow qualification, including the configured
  native program join; no body persistence or fixed-scan BPTT. At that checkpoint Compiler frontier was None.<span data-glossary-navigation="true"> ([rebind glossary](../../../glossary.md#capture-graph-ports-and-reconstruction); [qualification glossary](../../../glossary.md#term-qualification); [scan glossary](../../../glossary.md#term-cumulative-scan); [BPTT glossary](../../../glossary.md#term-backpropagation-through-time-bptt))</span>
- Compiler0013 is Complete, MAIN accepted 2026-10-24; independent code/docs APPROVE (0 P1/P2),
  worker module/Javadoc, compatibility and final docs/Javadoc PASS; its Result owns evidence.
  Ready frontier None; no downstream task is launched.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../glossary.md#term-javadoc); [Ready glossary](../../../glossary.md#term-ready))</span>

## Live gates and risks

- **[Conv3d](../../../glossary.md#conv3d--ncdhw) [qualification](../../../glossary.md#term-qualification):** 0006C implements group-isolated adjoints, stride/padding/dilation,
  overlap, symbolic Shape retention, selected roles and ordinary two-stage formula closure.
  First-order inventory is 39/112/134; only three fixed recurrent signatures remain deferred.
  CPU F32/F64 general Scan full-body forward/reverse composition passed all-valid/length-aware
  and both-direction numerical cases. This does not widen ordinary public CPU named-B Conv3d,
  CPU low arithmetic, Metal Scan or persistence inventories. MAIN accepted the source-matched root
  gate and fresh installed-artifact/no-Git-source runtime proof.<span data-glossary-navigation="true"> ([stride glossary](../../../glossary.md#term-stride); [padding glossary](../../../glossary.md#padding); [dilation glossary](../../../glossary.md#term-dilation); [Shape glossary](../../../glossary.md#shape); [signatures glossary](../../../glossary.md#term-signature); [CPU glossary](../../../glossary.md#term-cpu); [F32 glossary](../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../glossary.md#floating-point-precision-and-rounding); [general Scan glossary](../../../glossary.md#general-differentiable-scan); [Metal glossary](../../../glossary.md#term-metal); [Scan glossary](../../../glossary.md#term-cumulative-scan); [persistence glossary](../../../glossary.md#term-persistence); [artifact glossary](../../../glossary.md#term-artifact); [runtime glossary](../../../glossary.md#term-runtime))</span>
- **Recurrent differentiation:** 0006A covers forward compilation only. The exact RNN, GRU, and
  LSTM signatures remain outside gradient coverage. CPU0028/CPU0029 and<span data-glossary-navigation="true"> ([compilation glossary](../../../glossary.md#compiler-engine-and-execution-stage-names); [RNN glossary](../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../glossary.md#term-long-short-term-memory-lstm); [gradient glossary](../../../glossary.md#term-gradient))</span>
  [NN0021B](../../extensions/nn/tasks/0021b-fixed-recurrent-scan-implementation-program.md) are Complete
  for forward execution; BPTT remains unsupported. <span data-glossary-navigation="true"> ([BPTT glossary](../../../glossary.md#term-backpropagation-through-time-bptt))</span>[0010](tasks/0010-rnn-tanh-scan-bptt.md) is
  Superseded by general scan, not a fixed-adjoint primitive. Completed Model0034 authority covers source/IR and original
  canonical residual ports, pre-capture joint-output body AD and truthful capability projection.
  Existing fixed scans and NN early training rejection remain unchanged; no successor implementation authorized.<span data-glossary-navigation="true"> ([adjoint glossary](../../../glossary.md#term-adjoint); [IR glossary](../../../glossary.md#term-intermediate-representation-ir); [canonical glossary](../../../glossary.md#term-canonical-form); [residual glossary](../../../glossary.md#scan-residual-port-and-history-demand); [capture glossary](../../../glossary.md#term-capture-meanings); [AD glossary](../../../glossary.md#autograd); [NN glossary](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- **Exact versus relaxed algebra:** preserve the guarded [scalar](../../../glossary.md#term-scalar) `POW(+1) -> input`. Any exact
  `POW(0)` needs a typed shape-correct one-splat and proof across exceptional values, constant
  sidecars, publication identity, phase, autograd, and descriptors. Do not replace <span data-glossary-navigation="true"> ([constant glossary](../../../glossary.md#term-constant); [publication glossary](../../../glossary.md#term-publication); [identity glossary](../../../glossary.md#term-identity-meanings); [autograd glossary](../../../glossary.md#autograd); [descriptors glossary](../../../glossary.md#term-descriptor))</span>`POW(0.5)` with
  `SQRT` or infer Tensor constants from storage or factory history. Historical Config 0006 was
  removed by Model 0032; it is not a pending permission switch. A rewrite outside current Model
  semantics requires an explicit architecture decision; hardware or measurement never grants it.<span data-glossary-navigation="true"> ([Tensor glossary](../../../glossary.md#tensor); [Model glossary](../../../glossary.md#term-model))</span>
- **[Backend](../../../glossary.md#backend) boundary:** the coupled [CPU](../../backends/cpu/master-plan.md) and
  [Metal](../../backends/metal/master-plan.md) Conv3d training closure implements Compiler's
  ordinary formulas; forward admission alone is still no derivative proof. Compiler absorbs no
  lowering, route, generated-code, physical storage or execution ownership. Fixed-scan BPTT stays closed.<span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler); [admission glossary](../../../glossary.md#term-admission); [derivative glossary](../../../glossary.md#term-derivative); [lowering glossary](../../../glossary.md#lowering); [ownership glossary](../../../glossary.md#term-ownership-meanings))</span>
- **Structural risks:** do not merge recurrent occurrences, split graph-local ID assignment across
  captures, merge CSE across phases, construct partial backward expressions before preflight,
  reconstruct canonical sibling outputs, or turn logical saved edges into a runtime tape.<span data-glossary-navigation="true"> ([graph glossary](../../../glossary.md#graph); [CSE glossary](../../../glossary.md#term-common-subexpression-elimination-cse); [preflight glossary](../../../glossary.md#term-preflight))</span>

## Status [normalization](../../../glossary.md#term-normalization)

The task table and linked task `Status`/completion summaries take precedence over historical prose.
Current completion extends through 0006B11 plus independently reviewed 0006D, 0008, 0009, 0011 and 0012;
the historical coverage [checkpoint](../../../glossary.md#term-checkpoint) is not proof of every value-level higher-order [derivative](../../../glossary.md#term-derivative).

## History and update policy

Detailed results, commands, inventories, audits, and past ordering exceptions remain in linked
task files and Git history. Read them only for a current unresolved question. Update this plan only
when [Compiler](../../../glossary.md#term-compiler) [ownership](../../../glossary.md#term-ownership-meanings), package direction, task order/status, a live [dependency](../../../glossary.md#term-dependency) or risk, or the
current frontier changes. Use the [planning guide](../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../glossary.md#term-planning))</span> for any new or
materially revised brief, and stop if proposed work conflicts with the [architecture contract](../../../glossary.md#architecture-contract).
