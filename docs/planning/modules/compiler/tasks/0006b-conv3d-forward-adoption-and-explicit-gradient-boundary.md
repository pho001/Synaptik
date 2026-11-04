# Task 0006B: [Conv3d](../../../../glossary.md#conv3d--ncdhw) Forward Adoption and Explicit [Gradient](../../../../glossary.md#term-gradient) Boundary

## Status

Complete

## Goal

Adopt [Model](../../../../glossary.md#term-model)'s first-class grouped [NCDHW](../../../../glossary.md#conv3d--ncdhw) [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) [operation](../../../../glossary.md#operation) in [Compiler](../../../../glossary.md#term-compiler)'s ordinary flat
forward-[graph](../../../../glossary.md#graph) pipeline without [lowering](../../../../glossary.md#lowering), [decomposition](../../../../glossary.md#term-decomposition), execution, or [gradient](../../../../glossary.md#term-gradient) construction.

For each biased or unbiased [occurrence](../../../../glossary.md#term-occurrence), Compiler must preserve the exact operation kind,
`Conv3dAttrs`, ordered inputs, and one output; independently infer and verify the complete
promoted [descriptor](../../../../glossary.md#term-descriptor) and NCDHW geometry from captured graph state; retain every unresolved
[channel](../../../../glossary.md#term-channel), group, [bias](../../../../glossary.md#term-bias), and [spatial](../../../../glossary.md#term-spatial-axes)-fit relation as an ordered [deferred graph constraint](../../../../glossary.md#deferred-graph-constraint); and [run](../../../../glossary.md#term-run)
the same checks during initial and final [validation](../../../../glossary.md#term-validation).

`CompileMode.FORWARD_ONLY` must [compile](../../../../glossary.md#compile) the occurrence through the existing [capture](../../../../glossary.md#term-capture-meanings),
optimization, diagnostics, [publication](../../../../glossary.md#term-publication), and [Planning](../../../../glossary.md#term-planning) [handoff](../../../../glossary.md#term-handoff). `FORWARD_AND_BACKWARD` and
`TRAINING_STEP` must reject any complete original forward inventory containing `CONV3D` during
allocation-free [autograd](../../../../glossary.md#autograd) [preflight](../../../../glossary.md#term-preflight), before a [seed](../../../../glossary.md#term-seed), [derivative](../../../../glossary.md#term-derivative) [constant](../../../../glossary.md#term-constant), formula [Tensor](../../../../glossary.md#tensor), Tensor ID,
or combined graph is created. Conv3d derivative expressibility and formula construction remain
owned only by Draft Compiler task 0006C.

The representation remains:

```text
one Conv3d TensorProducer occurrence
  -> one ordinary flat CompiledNode
     operation = CONV3D + exact Conv3dAttrs
     inputs    = [input, weight] or [input, weight, bias]
     output    = one promoted NCDHW ValueId
  -> existing diagnostics, publication, and OperationCapabilityQuery paths
```

## [Scope](../../../../glossary.md#term-scope)

- Route `Conv3dKind.CONV3D` through the current structured-[operation](../../../../glossary.md#operation) [inference](../../../../glossary.md#term-inference) boundary.
- Independently validate its exact `Conv3dAttrs`, two-or-three-input cardinality, one-output
  cardinality, role types, ranks, static kernel extents, ordered floating promotion, output
  gradient eligibility, and exact output descriptor.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality); [kernel glossary](../../../../glossary.md#kernel); [gradient glossary](../../../../glossary.md#term-gradient); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Derive NCDHW result geometry from graph descriptors and attributes without trusting the stored
  Model output descriptor.<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [graph glossary](../../../../glossary.md#graph); [Model glossary](../../../../glossary.md#term-model))</span>
- Prove or retain, in deterministic order, exact obligations for:
  1. input-[channel](../../../../glossary.md#term-channel) divisibility by `groups`;
  2. output-channel divisibility by `groups`;
  3. `weightChannelsPerGroup * groups == inputChannels`;
  4. optional [bias](../../../../glossary.md#term-bias) length equal to output channels;
  5. non-negative padded depth numerator;
  6. non-negative padded height numerator; and
  7. non-negative padded width numerator.
- Preserve the exact input batch Dimension and weight output-channel Dimension in the inferred
  result and derive the three spatial Dimensions using the existing canonical Dimension
  expression vocabulary.<span data-glossary-navigation="true"> ([batch Dimension glossary](../../../../glossary.md#batch-dimension); [weight glossary](../../../../glossary.md#term-weight); [Dimension glossary](../../../../glossary.md#dimension); [spatial glossary](../../../../glossary.md#term-spatial-axes))</span>
- Preserve one ordinary flat node. Do not lower or decompose it into window, matrix, fold,
  pointwise, loop, region, or backend operations.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [pointwise glossary](../../../../glossary.md#term-pointwise); [backend glossary](../../../../glossary.md#backend))</span>
- Preserve ordinary exact common-subexpression elimination (CSE) semantics: graph-output
  producers remain ineligible as today; equal eligible internal Conv3d occurrences may merge only
  when phase, derivative order, complete operation value including attributes, remapped inputs,
  and output descriptors are equal; distinct attributes or inputs must not merge. Conv3d has no
  recurrent-style identity exclusion.<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#compiler-transformations-and-dominance); [producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [derivative glossary](../../../../glossary.md#term-derivative); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Preserve whole-[node](../../../../glossary.md#node) [dead-code elimination](../../../../glossary.md#term-dead-code-elimination-dce) and ordinary [canonical](../../../../glossary.md#term-canonical-form) reindexing.
- Carry surviving exact operation attributes and ordered descriptors through publication and the
  existing <span data-glossary-navigation="true"> ([Carry glossary](../../../../glossary.md#term-carry); [publication glossary](../../../../glossary.md#term-publication))</span>[`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery); add no provider capability.
- Add an allocation-free, deterministic Conv3d guard to the complete-original-forward-inventory
  autograd preflight for both backward-capable modes.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [autograd glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Keep `FirstOrderGradientCoverage.SIGNATURES`, its 128 supported rows, and its family-owner
  vocabulary unchanged. Update only the source-backed boundary test so the complete 132-signature
  Model inventory is partitioned into 128 supported signatures plus four deferred signatures:
  three recurrent signatures and one Conv3d signature.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [signature glossary](../../../../glossary.md#term-signature))</span>
- Update directly affected Compiler/Model explanatory documentation and planning status in the
  implementation change, with a separate clean documentation-focused pass.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [Conv3d](../../../../glossary.md#conv3d--ncdhw) input, [weight](../../../../glossary.md#term-weight), or [bias](../../../../glossary.md#term-bias) [gradient](../../../../glossary.md#term-gradient) formulas; `ConvolutionGradientRules` changes; a Conv3d
  gradient family owner; adding Conv3d to <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>`FirstOrderGradientCoverage.SIGNATURES`; or claiming
  adjoint or higher-order expressibility<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- any part of Draft Compiler 0006C, including its proof, Model prerequisite selection, detailed
  specification, saved-state policy, overlap-accumulation formula, or gradient closure
- changes to [Model](../../../../glossary.md#term-model) Conv3d semantics, `Conv3dAttrs`, `Conv3dKind`, Tensor methods, descriptors,
  provenance, factory behavior, numerical policy, or tests beyond focused unchanged-boundary
  validation<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation))</span>
- a `CONV1D` kind or [Compiler](../../../../glossary.md#term-compiler) inventory row; [NCW](../../../../glossary.md#term-ncw) Conv1d remains the visible
  `EXPAND_DIMS -> CONV2D -> SQUEEZE` composition from Model 0025G and continues to use existing
  inference and gradient rows<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- Conv3d lowering or decomposition, window extraction, matrix multiplication, fold, explicit
  loops, graph regions, nested graphs, backend-specific graphs, executable units, kernels,
  generated code, or numerical evaluation<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [decomposition glossary](../../../../glossary.md#term-decomposition); [matrix multiplication glossary](../../../../glossary.md#matrix-multiplication--matmul); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [kernels glossary](../../../../glossary.md#kernel); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- capability advertisement, owner-policy changes, concrete backend work, CPU 0008/0008A, general
  partition-DAG work, fusion, profitability, materialization, or performance claims<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization))</span>
- changes to Planning, Prepare, Runtime, Engine, Config, Trace, backend-contract, concrete
  backends, extensions, public Compiler API, dependencies, Gradle, architecture rules, or build
  structure<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [Trace glossary](../../../../glossary.md#trace); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- dynamic-rank, channels-last, asymmetric intrinsic padding, transposed, deformable,
  causal-specialized, quantized, sparse, depthwise-specific, separable, or arbitrary-rank
  convolution<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [channels-last glossary](../../../../glossary.md#spatial-axes-channels-and-window-sampling); [padding glossary](../../../../glossary.md#padding); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Compiler 0007 exact/relaxed algebra or any later detailed task specification
- architecture, backend-conformance, or integration tests unless implementation reveals an actual
  boundary change, in which case stop rather than expanding this task<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially Model and Compiler
  ownership, flat graph state, autograd preflight, compile lifecycle, and forbidden Compiler
  dependencies<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [graph glossary](../../../../glossary.md#graph); [autograd glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight); [compile glossary](../../../../glossary.md#compile); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [ADR 0009: Compiler-owned pre-capture Tensor-expression autograd](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [pre-capture Tensor-expression autograd glossary](../../../../glossary.md#pre-capture-tensor-expression-autograd))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0002: Captured-graph inference and validation](0002-captured-graph-inference-and-validation.md)<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation))</span>
- [Compiler 0004: Compiler-owned pre-capture autograd](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- [Compiler 0005D: Attention, convolution, pooling, and loss gradients](0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [loss glossary](../../../../glossary.md#loss); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- [Compiler 0005E: First-order gradient coverage closure](0005e-first-order-gradient-coverage-closure-checkpoint.md)
- [Compiler 0006: Explicit functional gradient requests](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
- [Compiler 0006A: Fixed recurrent-scan forward adoption](0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- [Model 0020: NCHW Conv2d semantics](../../model/tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- [Model 0025G: NCW Conv1d composition](../../model/tasks/0025g-ncw-conv1d-composition.md)<span data-glossary-navigation="true"> ([NCW glossary](../../../../glossary.md#term-ncw); [Conv1d composition glossary](../../../../glossary.md#conv1d-composition))</span>
- [Model 0025H: NCDHW Conv3d semantics](../../model/tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)

## Architecture constraints

- Model remains the owner of Conv3d meaning, immutable attributes, public Tensor construction,
  floating numerical policy, descriptor metadata, and canonical provenance.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [metadata glossary](../../../../glossary.md#term-metadata); [canonical glossary](../../../../glossary.md#term-canonical-form); [provenance glossary](../../../../glossary.md#provenance))</span>
- Compiler independently derives and validates captured semantics; it must not trust a stored
  output descriptor merely because Model construction produced it.
- One [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) [occurrence](../../../../glossary.md#term-occurrence) remains one ordinary flat `CompiledNode`. Compiler adds no convolution
  IR, nested graph, region, decomposition, executable work, physical state, or backend payload.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [graph glossary](../../../../glossary.md#graph); [decomposition glossary](../../../../glossary.md#term-decomposition); [backend glossary](../../../../glossary.md#backend); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- [Compiler](../../../../glossary.md#term-compiler) may retain unresolved logical [Shape](../../../../glossary.md#shape) obligations only through the existing typed
  `DeferredGraphConstraint`/`GraphPredicate` boundary. It must not bind Dimensions or move
  validation into Runtime.<span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension); [validation glossary](../../../../glossary.md#term-validation); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Compiler-owned autograd must fail closed before derivative Tensor allocation when a complete
  forward inventory contains an unsupported forward-only occurrence.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Forward inference support and first-order gradient coverage remain separate closed inventories.
  This task changes the former from 131 to 132 signatures and leaves the latter at 128.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient); [signatures glossary](../../../../glossary.md#term-signature))</span>
- CSE remains phase- and derivative-order-local and uses exact operation value, ordered remapped
  inputs, and complete output descriptors. Conv3d is functionally pure and gains no identity-
  distinct exemption.<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Compiler has no dependency on Runtime, Prepare, Engine, or concrete backends and introduces no
  public facade or registry.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry))</span>
- If implementation requires a public API, another module, a new predicate vocabulary, an
  architecture change, or a Conv3d derivative formula, stop and report the conflict.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing package used and changed:

- `io.github.pho001.synaptik.compiler` — retains all implementation and tests in the existing
  package-private Compiler front-end boundary.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>

Packages added or changed:

- no package is added;
- no public package surface is widened.

Type placement:

- `io.github.pho001.synaptik.compiler.CapturedGraphInference` — adds only closed [dispatch](../../../../glossary.md#term-dispatch) of
  `Conv3dKind` to the existing structured [inference](../../../../glossary.md#term-inference) [owner](../../../../glossary.md#term-owner).
- `io.github.pho001.synaptik.compiler.StructuredOperationInference` — owns the rank-five Conv3d
  descriptor derivation and ordered constraint requests beside its existing Conv2d inference.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [descriptor glossary](../../../../glossary.md#term-descriptor); [Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- `io.github.pho001.synaptik.compiler.AutogradPreflight` — owns the complete-forward-inventory
  forward-only boundary and deterministic pre-allocation rejection.
- `io.github.pho001.synaptik.compiler.FirstOrderGradientCoverage` — remains unchanged; only its
  source-backed package-local test records Conv3d as the fourth deferred signature.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>

Do not add a generic [convolution](../../../../glossary.md#term-channels-first-convolution-layers) helper, `ConvNd` abstraction, public diagnostic type, [gradient](../../../../glossary.md#term-gradient)
[registry](../../../../glossary.md#term-registry), or another package.

## Forward [inference](../../../../glossary.md#term-inference) and [validation](../../../../glossary.md#term-validation) contract

### Accepted [operation](../../../../glossary.md#operation), attributes, and [cardinality](../../../../glossary.md#term-cardinality)

Accept only:

```text
kind:       Conv3dKind.CONV3D
attributes: Conv3dAttrs
inputs:     2 or 3
outputs:    exactly 1
```

Ordered inputs are `[input, weight]` or `[input, weight, bias]`. The exact attribute object and
ordered inputs captured in the `Operation`/`CompiledNode` remain unchanged. Any unknown kind,
wrong attribute type, invalid input/output count, or later unrecognized operation continues to
[fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) through the current structural or [inference](../../../../glossary.md#term-inference) boundary.

### [Descriptor](../../../../glossary.md#term-descriptor) derivation

[Inference](../../../../glossary.md#term-inference) derives the result in this order:

1. Require floating `input`, then floating [`weight`](../../../../glossary.md#term-weight); if present, require floating [`bias`](../../../../glossary.md#term-bias).
2. Promote `input` with `weight`, then promote the intermediate type with present `bias`, using
   `DataTypePromotion.promoteFloating`.
3. Require input [Shape](../../../../glossary.md#shape) `[N, C_in, D, H, W]`, weight Shape
   `[C_out, C_in/groups, K_d, K_h, K_w]`, and optional bias Shape `[C_out]` by exact ranks
   five, five, and one.
4. Require `K_d`, `K_h`, and `K_w` to be statically known and positive, in depth, height, width
   order.
5. Compute each effective [kernel](../../../../glossary.md#kernel) with checked signed-`long` arithmetic:

   ```text
   effectiveKernel_x = dilation_x * (K_x - 1) + 1
   offset_x          = 2 * padding_x - effectiveKernel_x
   numerator_x       = X + offset_x
   X_out             = floor(numerator_x / stride_x) + 1
   ```

6. Construct the result Shape
   `[N, C_out, D_out, H_out, W_out]`, retaining the exact input batch Dimension and weight output-
   channel Dimension references.<span data-glossary-navigation="true"> ([batch Dimension glossary](../../../../glossary.md#batch-dimension); [channel glossary](../../../../glossary.md#term-channel); [Dimension glossary](../../../../glossary.md#dimension))</span>
7. Derive unresolved [layout](../../../../glossary.md#layout) and `requiresGrad` equal to the logical OR of input, weight, and
   present bias.

The inferred descriptor must exactly equal the stored output descriptor. Existing [node](../../../../glossary.md#node)-index,
[`NodeId`](../../../../glossary.md#nodeid), kind, output-index, [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), expected-descriptor, and stored-descriptor context remains
the final mismatch diagnostic.

### Ordered constraints and diagnostics

[Inference](../../../../glossary.md#term-inference) emits candidate constraints in this exact order:

```text
conv3d input channels divisible by groups
conv3d output channels divisible by groups
conv3d weight channels per group
conv3d bias channels                         only when bias is present
conv3d depth numerator non-negative
conv3d height numerator non-negative
conv3d width numerator non-negative
```

Use existing predicates:

- `DimensionDivisible(C_in, groups)`;
- `DimensionDivisible(C_out, groups)`;
- `DimensionEqual(multiply(weightChannelsPerGroup, groups), C_in)`;
- `DimensionEqual(biasLength, C_out)` when biased; and
- `DimensionAtLeast(numerator_x, 0)` for each [spatial axis](../../../../glossary.md#spatial-axes-channels-and-window-sampling).

Static contradictions fail during inference with existing full [node](../../../../glossary.md#node) context. Proven constraints
are discarded. Unresolved constraints become exact ordered `DeferredGraphConstraint` values and
public `CompileDiagnostics.DeferredConstraintDiagnostic` projections. The constraint subjects are
stable diagnostic text for this task; no new public diagnostic taxonomy or rejection aggregate
is added.

For a static [spatial](../../../../glossary.md#term-spatial-axes) input, checked addition computes the numerator and negative values fail. For
an unresolved input, use the same [canonical](../../../../glossary.md#term-canonical-form) `DimensionExpressions.addConstant`, `floorDivide`,
and final `addConstant(..., 1)` sequence as [Model](../../../../glossary.md#term-model). Checked arithmetic [overflow](../../../../glossary.md#term-overflow) is reported through
the current node-context [descriptor](../../../../glossary.md#term-descriptor)-derivation failure path; it is never treated as a deferred
valid [binding](../../../../glossary.md#term-binding).

### Repeated [validation](../../../../glossary.md#term-validation) and downstream [handoff](../../../../glossary.md#term-handoff)

The same `CapturedGraphInference.inferAndValidate` boundary must [run](../../../../glossary.md#term-run) before [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and
after canonicalization or every changed optional optimization candidate, as it does for current
operations. Final deferred constraints and their order must reflect only the final validated
[graph](../../../../glossary.md#graph).

Existing [publication](../../../../glossary.md#term-publication) and [Planning](../../../../glossary.md#term-planning) orchestration then [carry](../../../../glossary.md#term-carry):

- the exact surviving [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) [operation](../../../../glossary.md#operation) and attributes;
- exact ordered input [descriptors](../../../../glossary.md#term-descriptor);
- the exact one-output descriptor; and
- the final ordered deferred diagnostics.

A recording test provider may accept the query to prove the unchanged handoff. It must not be a
production capability claim, and no current provider is changed.

## [Autograd](../../../../glossary.md#autograd) and inventory boundary

For both backward-capable modes, `AutogradPreflight` builds the complete original forward [producer](../../../../glossary.md#term-producer)
inventory without allocating a [Tensor](../../../../glossary.md#tensor), then scans deterministic producer postorder before request
stage, [seed](../../../../glossary.md#term-seed), route, [occurrence](../../../../glossary.md#term-occurrence)-policy, or formula [validation](../../../../glossary.md#term-validation). The first [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) occurrence fails
with stable context containing producer-postorder index, exact kind class/name, exact attributes
class, and this explanation:

```text
Conv3d is forward-only until Compiler task 0006C closes its gradients
```

The [guard](../../../../glossary.md#term-guard) applies even when Conv3d is a separate forward root and the requested [target](../../../../glossary.md#term-target) lies on an
unrelated non-Conv3d branch. Rejection must leave the next Tensor ID unchanged and construct no
default or explicit seed [normalization](../../../../glossary.md#term-normalization), typed splat, local formula, [gradient](../../../../glossary.md#term-gradient) [accumulator](../../../../glossary.md#accumulator-type),
matching auxiliary, second-stage expression, combined [capture](../../../../glossary.md#term-capture-meanings), or partial [backward graph](../../../../glossary.md#backward-graph).

Do not add Conv3d to `FirstOrderGradientCoverage.SIGNATURES`, make it resolve to
`FamilyOwner.CONVOLUTION`, or add it to `ConvolutionGradientRules`. The existing [Conv2d](../../../../glossary.md#convolution--conv2d) row and
formulas remain unchanged. The
source-backed test must prove:

```text
Model production inventory:           39 families, 111 constants, 132 signatures
Compiler supported gradient inventory: 37 families, 107 kinds, 128 signatures
Deferred exact signatures:             4 = 3 recurrent + 1 Conv3d
Supported intersection deferred:       empty
Supported union deferred:              exact complete Model inventory
```

Every Conv3d output/input boundary classification remains `FC` with exact existing reason
`unknown or unclassified operation kind/attributes pairing`, without Tensor-ID allocation.
Forward [inference](../../../../glossary.md#term-inference) adoption must not be mistaken for gradient support.

## Affected files

Expected [Compiler](../../../../glossary.md#term-compiler) production source (3):

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CapturedGraphInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StructuredOperationInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`

Expected Compiler tests (4):

- add `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/Conv3dCompilerTest.java`
- update `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/StructuredOperationInferenceTest.java`
- update `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- update `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverageTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) (8):

- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/master-plan.md`
- this task specification
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless current evidence makes a statement inaccurate: `GraphCapture`,
`ForwardCommonSubexpressionElimination`, `CompileDiagnostics`, `FirstOrderGradientCoverage`,
`ConvolutionGradientRules`, their focused tests, [Model](../../../../glossary.md#term-model) [Conv3d](../../../../glossary.md#conv3d--ncdhw) source/tests/Javadocs, the
authoritative architecture and focused architecture pages, [Runtime](../../../../glossary.md#term-runtime) and Training APIs, Planning,
[Prepare](../../../../glossary.md#prepare), Runtime, [Engine](../../../../glossary.md#term-engine), [CPU](../../../../glossary.md#term-cpu) and other [backend](../../../../glossary.md#backend) plans, architecture/[conformance](../../../../glossary.md#term-conformance)/integration tests,
[Gradle](../../../../glossary.md#term-gradle), dependencies, and other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly fifteen paths maximum: three [Compiler](../../../../glossary.md#term-compiler) production files, four Compiler test files, and
eight documentation/[planning](../../../../glossary.md#term-planning) files. No [Model](../../../../glossary.md#term-model) Java or test, public [API](../../../../glossary.md#term-application-programming-interface-api), other [module](../../../../glossary.md#term-module), build,
architecture, [backend](../../../../glossary.md#backend), [conformance](../../../../glossary.md#term-conformance), or integration path may change.

If implementation needs another path, a new [graph](../../../../glossary.md#graph) predicate, another module, a public contract,
or a [gradient](../../../../glossary.md#term-gradient) formula, stop and update this task through a separate planning decision before
continuing. Do not silently use the maximum as permission for unrelated cleanup.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Review and finalize the class and method Javadocs of all three changed Compiler production
  files. Explain independent Conv3d inference, ordered deferred obligations, repeated final
  validation, and allocation-free backward rejection. Every changed or added method contract must
  have meaningful <span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation))</span>`@param`, `@return`, and applicable `@throws` documentation.
- Update the Compile API to make the 132-signature forward inventory current, retain the exact
  128-signature first-order inventory, explain the four-signature deferred set, and distinguish
  successful forward compilation from absent execution/capability and absent gradients.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [signature glossary](../../../../glossary.md#term-signature); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Update only the Conv3d downstream-status sentence in the Tensor API; Model semantics and public
  API remain unchanged.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model))</span>
- Update the glossary's Conv3d and inventory entries to distinguish current Compiler forward
  inference/final proof from Draft 0006C gradients and later execution.
- Update Model capabilities and the current Model master-plan status wording only where they call
  Compiler forward adoption absent or Conv3d planned. Do not change completed Model 0025H's
  historical task record.
- Synchronize this task, [Compiler](../../../../glossary.md#term-compiler) master plan, and roadmap to `Complete` only after implementation,
  validation, and the separate documentation pass. Keep 0006C and 0007 Draft without detailed
  specifications; keep Model 0025H Complete and Model 0026 Draft without a specification.
- Record reasoned no-change conclusions for Tensor public signatures/Javadocs, Runtime and
  Training APIs, Model capabilities beyond status, architecture/ADRs/tests, Planning/Prepare/
  Runtime/Engine, backend plans and capability providers, conformance/integration, Gradle,
  dependencies, and other modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- `FORWARD_ONLY` accepts biased and unbiased [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) as one ordinary flat captured node with
  exact operation/attribute/input/output preservation and no decomposition or execution work.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Inference independently checks exact kind/attributes/cardinalities, role types/ranks, ordered
  floating promotion, positive static K-d/K-h/K-w, checked geometry, result descriptor, unresolved
  layout, and gradient-eligibility OR.<span data-glossary-navigation="true"> ([Inference glossary](../../../../glossary.md#term-inference); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Static valid geometry and zero batch/output-channel cases pass. Every static contradiction and
  overflow category fails with deterministic node context. Dynamic channel/group/bias and three
  spatial-fit obligations are proven, rejected, or retained exactly in the specified order.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../../../glossary.md#term-channel); [overflow glossary](../../../../glossary.md#term-overflow); [bias glossary](../../../../glossary.md#term-bias); [spatial glossary](../../../../glossary.md#term-spatial-axes))</span>
- Final stored-[descriptor](../../../../glossary.md#term-descriptor) mismatch diagnostics identify [node](../../../../glossary.md#node) position, [`NodeId`](../../../../glossary.md#nodeid), kind, output
  position, [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), expected descriptor, and stored descriptor.
- Ordinary CSE semantics remain unchanged: equal eligible internal Conv3d expressions may merge;
  differing attributes or inputs do not; graph-output producers remain unmerged. DCE,
  canonicalization, phase/order metadata, and unrelated operation behavior remain unchanged.<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [graph glossary](../../../../glossary.md#graph); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Publication and a recording Planning provider receive the exact surviving operation attributes
  and ordered descriptors; no production provider advertises Conv3d.<span data-glossary-navigation="true"> ([Publication glossary](../../../../glossary.md#term-publication); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Both backward-capable modes reject the first Conv3d in complete producer postorder with the
  stable task-owned diagnostic before any derivative Tensor/ID/node allocation, including an
  unrelated-gradient-branch case.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [derivative glossary](../../../../glossary.md#term-derivative); [Tensor glossary](../../../../glossary.md#tensor))</span>
- The source-backed inventories are exactly 39 families, 111 constants, and 132 Model signatures;
  128 supported gradient signatures plus the exact four deferred signatures partition that set;
  Conv3d remains fail-closed with the existing unknown/unclassified reason.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [Model glossary](../../../../glossary.md#term-model); [signatures glossary](../../../../glossary.md#term-signature); [partition glossary](../../../../glossary.md#partition); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Conv1d remains visible rank-edit/Conv2d/rank-edit composition and gains no kind, inference row,
  gradient row, or special Compiler treatment.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- No Conv3d formula, `FirstOrderGradientCoverage` production row, `FamilyOwner` change,
  `ConvolutionGradientRules` change, public API, shared predicate type, other module, dependency,
  Gradle, architecture, backend, conformance, or integration behavior changes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Focused tests, unchanged Model boundary tests, one final Compiler suite, Compiler Javadoc,
  documentation validation, exact fifteen-path scope, status/order/specification-absence checks,
  empty staging, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [validation glossary](../../../../glossary.md#term-validation); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` pass.
- A separate clean documentation-focused context independently finalizes all affected Javadocs,
  explanatory docs, glossary impact, planning status/evidence, and reasoned no-change conclusions
  in the same overall change.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) the focused [Compiler](../../../../glossary.md#term-compiler) [matrix](../../../../glossary.md#term-matrix-meanings):

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.Conv3dCompilerTest \
  --tests io.github.pho001.synaptik.compiler.StructuredOperationInferenceTest \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest \
  --tests io.github.pho001.synaptik.compiler.ForwardCommonSubexpressionEliminationTest \
  --tests io.github.pho001.synaptik.compiler.CompileDiagnosticsTest
```

Run the committed [Model](../../../../glossary.md#term-model) boundary once without changing Model:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.convolution.Conv3dSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorConv3dExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorConv2dExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorConv1dExpressionTest
```

After executable Compiler code stabilizes, run exactly one final affected-[module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:compiler:test
```

Record [Gradle](../../../../glossary.md#term-gradle) outcomes and [JUnit](../../../../glossary.md#term-junit) XML suite/test/skip/failure/error counts. Do not run the full
Model suite because Model executable behavior does not change. Repository-wide, architecture,
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration validation is deferred to [continuous integration](../../../../glossary.md#term-continuous-integration-ci): this task
changes one executable module, no [dependency](../../../../glossary.md#term-dependency) or architecture rule, no shared build configuration,
and no backend or end-to-end path.

The implementation context must hand the actual diff and exact executable evidence to a separate
clean documentation-focused context. That context must not repeat successful Java tests unless it
changes executable behavior or records a concrete stale-evidence risk.

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --cached --name-only
git status --short
```

The documentation pass must also inspect the generated/public Compiler Javadocs and the source
Javadocs of package-private changed types; validate local Markdown links and anchors, unique
headings, balanced fences, final newlines, and trailing whitespace; verify exact [operation](../../../../glossary.md#operation),
constraint, diagnostic, inventory, and status wording; confirm the exact fifteen authorized
paths; confirm no [Conv3d](../../../../glossary.md#conv3d--ncdhw) [gradient](../../../../glossary.md#term-gradient) or provider-capability claim; confirm 0006C and 0007 are Draft
without detailed task files; confirm Model 0025H remains Complete and Model 0026 remains Draft
without a specification; preserve roadmap order Model 0025H -> Compiler 0006B -> [CPU](../../../../glossary.md#term-cpu) 0008 -> CPU
0008A -> CPU 0008B–0008E; and confirm an empty Git index.

## Dependencies

Hard prerequisites, all Complete:

- Model 0018K–0018N and 0025 provide operation-signature, symbolic-Dimension, descriptor, and
  canonical-producer foundations.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [Dimension glossary](../../../../glossary.md#dimension); [descriptor glossary](../../../../glossary.md#term-descriptor); [canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer))</span>
- [Model 0020](../../model/tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md) provides the
  rank-two grouped-convolution semantic and inference oracle.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [inference glossary](../../../../glossary.md#term-inference); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- [Model 0025G](../../model/tasks/0025g-ncw-conv1d-composition.md) fixes Conv1d as visible ordinary
  composition with no new operation kind.
- [Model 0025H](../../model/tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md) provides
  exact first-class Conv3d semantics, attributes, signatures, static/symbolic geometry, numerical
  policy, and canonical provenance.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [provenance glossary](../../../../glossary.md#provenance))</span>
- Compiler 0001–0003B provide identity capture, inference/final-validation orchestration,
  canonicalization, exact optimization, DCE/CSE, and constant sidecars.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [constant glossary](../../../../glossary.md#term-constant))</span>
- Compiler 0004–0005E provide allocation-free autograd preflight, combined capture, publication/
  Planning orchestration, and the source-backed 128-signature first-order closure.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight); [publication glossary](../../../../glossary.md#term-publication))</span>
- [Compiler 0006](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
  provides current one/two-stage functional requests and backward-capable modes.
- [Compiler 0006A](0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md) provides the
  complete-original-forward-inventory forward-only guard precedent and the intentional split
  between forward inference and deferred gradient inventories.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [gradient glossary](../../../../glossary.md#term-gradient))</span>

Existing [Compiler](../../../../glossary.md#term-compiler) dependencies on [Model](../../../../glossary.md#term-model), Config, [Planning](../../../../glossary.md#term-planning), [Backend](../../../../glossary.md#backend) Contract, and [Trace](../../../../glossary.md#trace) are
sufficient. No new [dependency](../../../../glossary.md#term-dependency) or architecture decision is required.

## Follow-up tasks

- Draft Compiler 0006C remains the sole owner of Conv3d adjoint-expressibility proof and gradient
  closure. It must prove group isolation, dilation/padding, overlap accumulation, symbolic Shape,
  and higher-order formula closure through current public Tensor algebra, or select the smallest
  separate Model prerequisite. It is not implemented or specified here.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [adjoint glossary](../../../../glossary.md#term-adjoint); [gradient glossary](../../../../glossary.md#term-gradient); [dilation glossary](../../../../glossary.md#term-dilation); [padding glossary](../../../../glossary.md#padding); [Shape glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model))</span>
- Draft CPU 0008 remains the portable Conv2d execution foundation after 0006B. Draft CPU 0008A
  then validates visible Conv1d composition and adds direct grouped Conv3d forward execution.
  Compiler 0006C does not block those forward CPU tasks.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv1d composition glossary](../../../../glossary.md#conv1d-composition))</span>
- CPU 0008B–0008E remain the later ordered general decomposition, recognition, profitability, and
  bounded multi-input materialization program. No need for those facilities is inferred here.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [materialization glossary](../../../../glossary.md#materialization))</span>
- Engine 0004 and NN 0025/0025A remain later public lifecycle and layer consumers. Model 0026
  remains the separate Draft FLOAT16 semantic foundation.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [lifecycle glossary](../../../../glossary.md#lifecycle); [FLOAT16 glossary](../../../../glossary.md#term-float16))</span>
- [Compiler](../../../../glossary.md#term-compiler) 0007 remains the separate Draft exact/permission-aware algebra [owner](../../../../glossary.md#term-owner).

Do not create any follow-up detailed specification in this task.

## Architecture impact

Expected impact: None.

This task implements the existing [Compiler](../../../../glossary.md#term-compiler) [ownership](../../../../glossary.md#term-ownership-meanings) and pre-[capture](../../../../glossary.md#term-capture-meanings) [fail-closed](../../../../glossary.md#term-fail-closed) contract with
the current flat [graph](../../../../glossary.md#graph), typed constraints, exact optimization, [publication](../../../../glossary.md#term-publication), diagnostics, and
[Planning](../../../../glossary.md#term-planning) [handoff](../../../../glossary.md#term-handoff). It changes no [module](../../../../glossary.md#term-module) ownership, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), public [API](../../../../glossary.md#term-application-programming-interface-api), architecture
rule, or execution contract. If implementation reveals a need for any such change, stop and
report the exact conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit, stage, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the Compiler master plan, and
docs/planning/modules/compiler/tasks/0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md.
Read the task's directly referenced Model Conv3d/Conv2d/Conv1d, Compiler capture/inference/
constraint/validation/autograd/coverage/CSE/diagnostic/publication/Planning source, tests, and
documentation contracts.

Implement Compiler 0006B exactly within its fifteen authorized paths. Adopt Conv3d only in the
ordinary flat forward pipeline, independently derive its complete descriptor and ordered deferred
obligations, and reject both backward-capable modes before derivative Tensor allocation. Preserve
the 128 supported gradient rows plus exact four-signature deferred boundary. Do not implement
0006C, Conv3d formulas, decomposition, execution, capability advertisement, public API, another
module, dependencies, Gradle, architecture, conformance, integration, or later specifications.
Stop on any architecture, inventory, or scope conflict.

After executable implementation and the recorded focused, Model-boundary, and final Compiler
validation, hand the actual diff and exact evidence to a separate documentation-focused agent or
thread with clean context. That pass must follow docs/developer-guide/documentation-rules.md and
finalize affected Javadocs, APIs, glossary/capability impact, planning status/evidence, and
documentation validation in the same overall change without repeating successful Java tests
unless executable behavior changes or a concrete stale-evidence risk is recorded.

Update this task with local decisions, exact validation evidence, implementation notes,
completion summary, and final status. Do not mark it Complete before every acceptance criterion
and the documentation pass finish.
```

## Local decisions

- [Conv3d](../../../../glossary.md#conv3d--ncdhw) uses the existing ordinary `CompiledNode`, structured inference, typed constraint,
  diagnostics, optimization, publication, and Planning boundaries. No new IR or public type is
  justified.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [Planning glossary](../../../../glossary.md#term-planning); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- [Rank](../../../../glossary.md#term-rank)-five Conv3d [inference](../../../../glossary.md#term-inference) remains beside [Conv2d](../../../../glossary.md#convolution--conv2d) in `StructuredOperationInference`; no generic
  ConvNd helper or new package is introduced.
- [Spatial](../../../../glossary.md#term-spatial-axes)-fit validity is an explicit `DimensionAtLeast(numerator, 0)` obligation for each axis.
  This makes deferred binding requirements observable rather than treating a symbolic output
  expression as proof of fit.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-normative))</span><span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- Conv3d remains eligible for ordinary exact CSE. Unlike recurrent scan, the architecture does
  not make equivalent Conv3d producer identities semantically distinct.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [scan glossary](../../../../glossary.md#term-cumulative-scan); [producer glossary](../../../../glossary.md#term-producer))</span>
- Backward-capable compilation rejects Conv3d from the complete original forward inventory, not
  only from the selected gradient ancestry, so <span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [gradient glossary](../../../../glossary.md#term-gradient))</span>`FORWARD_ONLY` is the sole adopting mode until
  0006C.
- Production first-order coverage remains byte-for-byte unchanged. Its source-backed test expands
  only the deferred set and complete Model inventory counts.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>

## Known limitations

- No current production capability provider is expected to accept [`CONV3D`](../../../../glossary.md#conv3d--ncdhw), so forward Compiler
  adoption does not make the operation executable through Engine or a concrete backend.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [operation glossary](../../../../glossary.md#operation); [Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- Deferred constraints are compile diagnostics and later binding obligations; this task does not
  provide a binding API or prove arbitrary runtime Shapes.<span data-glossary-navigation="true"> ([compile diagnostics glossary](../../../../glossary.md#compile-diagnostics); [binding glossary](../../../../glossary.md#term-binding); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [runtime glossary](../../../../glossary.md#term-runtime); [Shapes glossary](../../../../glossary.md#shape))</span>
- Conv3d [gradients](../../../../glossary.md#term-gradient) and every [differentiable](../../../../glossary.md#derivative-and-chain-rule) role remain [fail-closed](../../../../glossary.md#term-fail-closed) until separate task 0006C.
- Conv3d execution, algorithms, resources, numerical realization, and performance remain future
  CPU/backend work.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
- Repository-wide and downstream suites remain deferred under the recorded task-[validation](../../../../glossary.md#term-validation) tier.

## [Validation](../../../../glossary.md#term-validation) evidence

- Reused the implementation context's focused Compiler matrix: 6 suites, 40 tests, 0 skipped,
  failures, or errors.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Reused the focused unchanged Model Conv1d/Conv2d/Conv3d boundary: 4 suites, 30 tests, 0 skipped,
  failures, or errors.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- Reused the single final `./gradlew :modules:compiler:test`: 34 suites, 231 tests, 0 skipped,
  failures, or errors. One intermediate focused fixture was corrected before this final passing
  evidence; no production defect remained.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>
- The documentation-focused pass ran `./gradlew :modules:compiler:javadoc`, inspected generated
  public Compiler Javadocs and all three changed package-private source Javadocs, validated local
  Markdown links and anchors, heading uniqueness, balanced fences, final newlines, CR/trailing
  whitespace, status/inventory wording, exact fifteen-path scope, and the empty Git index, then
  passed final <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.
- Java tests were not repeated because the documentation pass changed no executable Java behavior
  and found no stale-evidence risk.

## Implementation notes

- `CapturedGraphInference` dispatches [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) to the existing structured [owner](../../../../glossary.md#term-owner).
  `StructuredOperationInference` independently derives its promoted rank-five descriptor and
  requests the seven channel/spatial obligations in the specified order; final graph validation
  checks the stored descriptor and retained constraints.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [descriptor glossary](../../../../glossary.md#term-descriptor); [channel glossary](../../../../glossary.md#term-channel); [spatial glossary](../../../../glossary.md#term-spatial-axes); [graph glossary](../../../../glossary.md#graph); [validation glossary](../../../../glossary.md#term-validation))</span>
- Conv3d remains an ordinary [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse)-eligible flat [node](../../../../glossary.md#node). Existing [publication](../../../../glossary.md#term-publication), diagnostics, and
  [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery)/Planning handoff carry it unchanged; no production provider accepts
  it and no lowering, preparation, execution, or gradient behavior was added.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [handoff glossary](../../../../glossary.md#term-handoff); [carry glossary](../../../../glossary.md#term-carry); [lowering glossary](../../../../glossary.md#lowering); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- `AutogradPreflight` rejects the first Conv3d occurrence in complete original-forward postorder
  for both backward-capable modes before seeds or derivative Tensors. Production<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [derivative glossary](../../../../glossary.md#term-derivative); [Tensors glossary](../../../../glossary.md#tensor))</span>
  `FirstOrderGradientCoverage` and `ConvolutionGradientRules` remain unchanged; the boundary test
  proves 39/111/132 forward inventory against unchanged 37/107/128 first-order support plus the
  exact four deferred signatures.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- The affected Javadocs are meaningful and complete for their package-private contracts. No
  public Tensor or Compiler API signature changed.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Reviewed and recorded no-change conclusions for Runtime and Training APIs; architecture, ADRs,
  and architecture tests; Planning, Prepare, Runtime, and Engine behavior; capability providers
  and backend plans; conformance and integration tests; Gradle and dependencies; Model Java and
  tests; and all other modules. None required an update within task scope.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [Model glossary](../../../../glossary.md#term-model); [modules glossary](../../../../glossary.md#term-module); [scope glossary](../../../../glossary.md#term-scope))</span>

## Completion summary

Completed [Compiler](../../../../glossary.md#term-compiler) 0006B within the authorized fifteen paths. Three package-private production
files adopt first-class [Conv3d](../../../../glossary.md#conv3d--ncdhw) for independent forward [inference](../../../../glossary.md#term-inference)/final [validation](../../../../glossary.md#term-validation) and preserve a
pre-allocation backward boundary; four tests prove forward behavior, ordered constraints,
diagnostics, [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse)/[publication](../../../../glossary.md#term-publication)/[Planning](../../../../glossary.md#term-planning) behavior, and the exact supported/deferred inventories.
The documentation pass finalized the Compiler and [Tensor](../../../../glossary.md#tensor) APIs, glossary, [Model](../../../../glossary.md#term-model) downstream-status
statements, task evidence, Compiler master plan, and roadmap. Compiler 0006C and 0007 remain Draft
without specifications; Model 0025H remains Complete and 0026 remains Draft without a
specification; [CPU](../../../../glossary.md#term-cpu) 0008 remains the next Draft task to plan separately, followed by CPU 0008A and
CPU 0008B–0008E. No unresolved issue remains and no follow-up is required for 0006B.

Status: Complete
