# Task 0023: [Adjoint](../../../../glossary.md#term-adjoint) Expressibility Audit

**Editorial reading key (current context, not a status update).**

This completed record is a planning audit, not a gradient implementation; its six Draft follow-ups
describe the historical frontier, not present API absence. An adjoint maps an incoming output
cotangent (sensitivity) to an input sensitivity; `requiresGrad` marks eligibility, not a derivative
policy. Model defines meanings; compiler owns differentiation and saved-value capture; execution
providers realize supported routes without redefining semantics. These responsibilities do not
assign buffer ownership. Caller Tensor references, Shapes and attributes describe expressions;
construction neither reads values nor executes or owns result buffers.

For today's broadcast-reversal contract, follow [binding-aware
sum-to-Shape](../../../../reference/tensor/reductions.md#binding-aware-sum-to-shape-expressions):
reducing shape `[8, 4]` to `[1, 4]` combines rows, while `[8, 4]` preserves positions. Follow
[Gather-compatible Scatter
Add](../../../../reference/tensor/scatter.md#gather-compatible-scatter-add-expressions) to
distinguish its rank-changing updates from Scatter Elements: the receiver supplies the base, indices
select destinations, and updates contribute values only during execution. Duplicate destinations
accumulate; scatter describes a new result, not base mutation. These current descriptions do not
settle the audit's deferred derivative policies or erase NaN, infinity, empty-domain and signed-zero
obligations.

**Mathematical reading illustration.** For a differentiable real function $y=f(x)$ with
$x\in\mathbb{R}^{n}$ and $y\in\mathbb{R}^{m}$, the adjoint maps an output sensitivity
$u\in\mathbb{R}^{m}$ to input sensitivity $J_f(x)^\mathsf{T}u$, where $J_f(x)$ is the $m$-by-$n$
Jacobian. For scalar $f(x)=x^2$, $x=3$, and $u=1$, the input sensitivity is six. This explains the
audit's representational question; it does not choose derivatives at nonsmooth or exceptional
values.

## Status

Complete

## Goal

Audit whether the [adjoint](../../../../glossary.md#term-adjoint) of every selected [differentiable](../../../../glossary.md#derivative-and-chain-rule) public [Tensor](../../../../glossary.md#tensor) [operation](../../../../glossary.md#operation) can be written
exactly as a [backend](../../../../glossary.md#backend)-neutral [compiler](../../../../glossary.md#term-compiler) composition of current general [model](../../../../glossary.md#term-model) operations and current
shared [producer](../../../../glossary.md#term-producer) outputs.

The task produces one [planning](../../../../glossary.md#term-planning)-only expressibility [matrix](../../../../glossary.md#term-matrix-meanings), identifies the minimum generally useful
public Tensor primitives needed to close proven gaps, and inserts concise Draft follow-up rows
before task 0024. It does not add an operation kind, Tensor method, [gradient](../../../../glossary.md#term-gradient) rule, [backward graph](../../../../glossary.md#backward-graph),
compiler pass, executable behavior, or detailed follow-up specification.

[Mental model](../../../../glossary.md#term-mental-model):

```text
current exact forward semantics
  -> exact adjoint formula for each differentiable input role
     -> current composition, existing auxiliary output, public-primitive gap,
        genuine semantic gap, non-differentiable role, or deferred policy
        -> smallest evidence-backed follow-up queue before 0024
```

This is deliberately an audit-only task. The selected operation surface spans [elementwise](../../../../glossary.md#term-elementwise),
reduction, [linear](../../../../glossary.md#term-linear-meanings) algebra, stochastic, [normalization](../../../../glossary.md#term-normalization), attention, [convolution](../../../../glossary.md#term-channels-first-convolution-layers), [pooling](../../../../glossary.md#term-pooling), [layout](../../../../glossary.md#layout),
indexing, ordering, and [loss](../../../../glossary.md#loss) contracts. Current evidence already exposes independent questions
about dynamic-[shape](../../../../glossary.md#shape) unbroadcasting, zero-base construction for scatter, public general-[axis](../../../../glossary.md#term-axis) fold,
[slice](../../../../glossary.md#slice) placement, convolution correlation geometry, pooling selection, and [derivative](../../../../glossary.md#term-derivative) conventions.
Choosing or implementing one primitive before the complete audit would preselect a solution and
could hide a second generally useful gap. No small implementation slice is logically inseparable
from recording these decisions.

## [Scope](../../../../glossary.md#term-scope)

- Create `docs/planning/modules/model/adjoint-expressibility-audit.md` as the sole detailed
  planning artifact for the audit result.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Define the artifact's notation for an output cotangent, input adjoint, reduction-to-input-Shape,
  saved forward value, recomputed forward value, and non-differentiable input.<span data-glossary-navigation="true"> ([cotangent glossary](../../../../glossary.md#cotangent); [adjoint glossary](../../../../glossary.md#term-adjoint); [Shape glossary](../../../../glossary.md#shape); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Inventory every current public Tensor semantic operation family whose floating input or output
  can participate in a differentiable expression. Treat conveniences such as <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [operation glossary](../../../../glossary.md#operation))</span>[`linear`](../../../../glossary.md#term-linear-meanings),
  [`embedding`](../../../../glossary.md#embedding), [`transpose`](../../../../glossary.md#term-transpose), `flip`, `clampMin`, `clampMax`, and `unstack` through their exact
  current primitive producer chains rather than inventing separate semantic kinds.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Inventory public operations and output roles that are non-differentiable so the boundary is
  explicit rather than silently omitted.
- Write exact backend-neutral vector-Jacobian-product or adjoint compositions at an appropriate
  family level. A family row may cover overloads only when their operation kind, differentiable
  input roles, formula, Shape obligations, special-value obligations, and classification are the
  same. Split rows by input role or semantic variant when any of those differ.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [vector glossary](../../../../glossary.md#term-vector); [Jacobian glossary](../../../../glossary.md#term-jacobian))</span>
- Audit required forward inputs, public outputs, recomputable intermediates, and existing
  producer-described auxiliary outputs. In particular, inspect the dropout keep-mask slot,
  dropout state outputs, batch-normalization saved mean/inverse-standard-deviation slots, top-K
  value/index slots, and the absence of attention weights or maximum-pooling indices.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [mask glossary](../../../../glossary.md#term-boolean-mask); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [mean glossary](../../../../glossary.md#term-mean); [weights glossary](../../../../glossary.md#term-weight); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- Classify every audited row using the fixed decision vocabulary below and explain the evidence
  for the classification.
- Identify only the minimum generally useful public model primitives proved necessary by the
  completed matrix. Insert one concise Draft master-plan and roadmap row per selected primitive or
  cohesive primitive family immediately after 0023 and before 0024. Do not create a detailed task
  specification for any follow-up.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Keep `capabilities.md`, the model master plan, and the roadmap synchronized with the audit
  result, follow-up queue, dependencies, statuses, and preserved completed history.
- Record explicit no-change conclusions for Tensor, Compile, Runtime, and Training API references,
  the glossary, architecture documents and tests, Java source and tests, Gradle, other modules,
  backend conformance, and integration tests.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Use current repository contracts as the primary source. If an external formula or convention is
  needed to resolve a factual question, cite only a primary paper or official specification/API
  reference, minimally, and distinguish that evidence from Synaptik's selected semantics.

### Required family coverage

The [matrix](../../../../glossary.md#term-matrix-meanings) must cover at least these current public families and must add any current public
semantic family found during the final source/[API](../../../../glossary.md#term-application-programming-interface-api) inventory:

- binary and scalar arithmetic, clamp/minimum/maximum/power, unary numeric operations, modern
  activations, floating casts, and conditional selection;<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [unary glossary](../../../../glossary.md#term-unary); [activations glossary](../../../../glossary.md#term-activation))</span>
- full, single-axis, ordered multi-axis, statistical, norm, and masked reductions, plus cumulative
  sum and extrema/arg-extrema distinctions;<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [norm glossary](../../../../glossary.md#term-norm))</span>
- [matrix multiplication](../../../../glossary.md#matrix-multiplication--matmul) and the exact primitive [`linear`](../../../../glossary.md#term-linear-meanings) composition;
- [softmax](../../../../glossary.md#softmax--log-softmax) and [log-softmax](../../../../glossary.md#softmax--log-softmax);
- explicit-state [dropout](../../../../glossary.md#dropout--inverted-dropout) and its hidden keep [mask](../../../../glossary.md#term-boolean-mask);
- layer, RMS, batch-inference, and batch-training normalization, including affine inputs and saved
  training statistics;<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [inference glossary](../../../../glossary.md#term-inference); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- [scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention) with and without explicit/causal masking;
- grouped [NCHW](../../../../glossary.md#term-nchw) [convolution](../../../../glossary.md#term-channels-first-convolution-layers), maximum [pooling](../../../../glossary.md#term-pooling), and count-[padding](../../../../glossary.md#padding) average pooling;
- contiguous, reshape, expand, dimension insertion/removal, permutation/transpose, slice/flip,
  scalar select, padding, tiling, concatenation, stack/unstack, unfold, unfold2d, and fold2d;<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [dimension glossary](../../../../glossary.md#dimension); [permutation glossary](../../../../glossary.md#permutation); [transpose glossary](../../../../glossary.md#term-transpose); [slice glossary](../../../../glossary.md#slice); [scalar select glossary](../../../../glossary.md#scalar-select); [tiling glossary](../../../../glossary.md#tiling))</span>
- Gather, Gather Elements, Gather-ND, embedding, Scatter Elements, and Scatter-ND, including every
  selected scatter reduction whose differentiable roles differ;<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [Gather Elements glossary](../../../../glossary.md#gather-elements); [Gather-ND glossary](../../../../glossary.md#gather-nd); [embedding glossary](../../../../glossary.md#embedding); [Scatter Elements glossary](../../../../glossary.md#scatter-elements); [Scatter-ND glossary](../../../../glossary.md#scatter-nd))</span>
- stable sort and top-K value outputs, with index outputs separated from value outputs; and
- mean-squared error plus dense-target and index-target categorical cross entropy with logits for
  every differentiable operand role and reduction.<span data-glossary-navigation="true"> ([mean-squared error glossary](../../../../glossary.md#mean-squared-error--mse); [target glossary](../../../../glossary.md#term-target); [categorical cross entropy glossary](../../../../glossary.md#categorical-cross-entropy))</span>

Comparison, logical, floating-classification, one-hot, [graph](../../../../glossary.md#graph)-[RNG](../../../../glossary.md#graph-rng-state--graphrngstate)-state, argsort, arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum), and
integral/index roles must appear as non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) or explicitly policy-deferred boundaries as
appropriate. A true `requiresGrad` [metadata](../../../../glossary.md#term-metadata) flag is eligibility evidence only and is not proof that
an exact [derivative](../../../../glossary.md#term-derivative) policy has already been selected.

## Out of [scope](../../../../glossary.md#term-scope)

- Java production or test changes
- a new or changed [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), operation signature, Tensor method, result
  carrier, helper, descriptor, producer, provenance contract, or factory behavior<span data-glossary-navigation="true"> ([operation signature glossary](../../../../glossary.md#operation-signature); [Tensor glossary](../../../../glossary.md#tensor); [carrier glossary](../../../../glossary.md#term-storage-carrier); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- any catalog or advance selection of [operation](../../../../glossary.md#operation)-specific `*_BACKWARD`, `BACKWARD_*`, gradient,
  derivative, or adjoint kinds<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- implementation of `foldAxis`, slice scatter/add, reduce-to-Shape, zero-like expression,
  convolution transpose/correlation, pooling indices, or any other candidate primitive<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [Shape glossary](../../../../glossary.md#shape); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [transpose glossary](../../../../glossary.md#term-transpose); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- a detailed task specification for 0024 or for any follow-up inserted by this audit
- choosing subgradient, discontinuity, extrema-tie, duplicate-target, or undefined-point policy
  merely to make a formula table complete<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [target glossary](../../../../glossary.md#term-target))</span>
- compiler capture, automatic-differentiation traversal, gradient accumulation, backward graph
  construction, saved-value lifetime implementation, optimization, or publication<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradient accumulation glossary](../../../../glossary.md#term-gradient-accumulation); [backward graph glossary](../../../../glossary.md#backward-graph); [lifetime glossary](../../../../glossary.md#term-lifetime); [publication glossary](../../../../glossary.md#term-publication))</span>
- backend capability declarations, lowering, fusion, specialization, algorithms, kernels,
  tolerances, execution, or performance promises<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [specialization glossary](../../../../glossary.md#term-specialization); [kernels glossary](../../../../glossary.md#kernel))</span>
- [runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), [engine](../../../../glossary.md#term-engine), training-extension, or [backend](../../../../glossary.md#backend) changes
- architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [dependency](../../../../glossary.md#term-dependency), [Gradle](../../../../glossary.md#term-gradle), build-logic, [conformance](../../../../glossary.md#term-conformance)-test, or integration-test changes
- copying legacy implementation structure or treating legacy [gradient](../../../../glossary.md#term-gradient) callbacks as current policy

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)
- completed operation, signature, shared-producer, symbolic-Shape, typed-scalar, indexing, window,
  modern-model, normalization, and loss tasks from 0005 through 0022B<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [producer glossary](../../../../glossary.md#term-producer); [Shape glossary](../../../../glossary.md#shape); [scalar glossary](../../../../glossary.md#term-scalar); [normalization glossary](../../../../glossary.md#term-normalization); [loss glossary](../../../../glossary.md#loss))</span>

## Architecture constraints

- `modules/model` owns backend-independent operation semantics, reusable public Tensor primitives,
  descriptors, and pre-capture producer metadata. It does not own gradient rules or backward graph
  construction.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [capture glossary](../../../../glossary.md#term-capture-meanings); [producer glossary](../../../../glossary.md#term-producer); [metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient); [backward graph glossary](../../../../glossary.md#backward-graph))</span>
- `modules/compiler` owns automatic differentiation, exact gradient-rule selection, backward graph
  construction, saved-value capture/lifetime decisions, and post-autograd optimization.<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd); [lifetime glossary](../../../../glossary.md#term-lifetime); [autograd glossary](../../../../glossary.md#autograd))</span>
- Backend prepare owns lowering, fusion, specialization, route selection, and kernels. Complexity,
  operation count, anticipated performance, or fusion opportunity is never evidence for a new
  model semantic kind.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [specialization glossary](../../../../glossary.md#term-specialization); [kernels glossary](../../../../glossary.md#kernel); [model glossary](../../../../glossary.md#term-model))</span>
- A stable tensor transformation useful outside automatic differentiation is planned as a general
  public model primitive and Tensor API capability, not hidden under a backward-only name.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- A compiler-only semantic kind is eligible only when the audit supplies a concrete proof that
  neither current exact composition nor an appropriate general public primitive can represent the
  meaning. Failure to find a short formula is not such a proof.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- Shared [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) output descriptors and indexed provenance are the existing representation
  for genuine auxiliary forward values. The audit may require their compiler capture; it may not
  redesign producer identity, expose private outputs, or invent saved-state storage.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Planning documents remain non-authoritative coordination artifacts. The audit must explain
  current contracts without changing <span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [artifacts glossary](../../../../glossary.md#term-artifact))</span>`ARCHITECTURE.md` or claiming implementation.
- If exact semantics are absent or contradictory, classify the row as deferred policy or report an
  architecture/scope conflict. Do not invent behavior.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Package impact

Existing packages used:

- All current `io.github.pho001.synaptik.model.operation.*` families and
  `io.github.pho001.synaptik.model.tensor` are inspected as evidence only.
- `io.github.pho001.synaptik.model.shape`, `.datatype`, and `.layout` are inspected where exact
  Shape, type, or view semantics constrain expressibility.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [view glossary](../../../../glossary.md#term-view-meanings))</span>

Packages added or changed:

- None. This task changes [planning](../../../../glossary.md#term-planning) documentation only.

Type placement:

- None. Candidate types or methods may be named only as non-binding follow-up design questions or
  as the minimum selected public primitive in a concise Draft queue row after proof in the matrix.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>

## Affected files

Expected:

- `docs/planning/modules/model/adjoint-expressibility-audit.md`
- `docs/planning/modules/model/tasks/0023-adjoint-expressibility-audit.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Reviewed without modification; if the audit finds an actual contradiction, stop and report it:

- `ARCHITECTURE.md` and focused architecture documentation
- `docs/api/tensor-api.md`, `docs/api/compile-api.md`, `docs/api/runtime-api.md`, and
  `docs/api/training-api.md`
- `docs/glossary.md`
- current [model](../../../../glossary.md#term-model) production and test sources
- completed task specifications relevant to each [matrix](../../../../glossary.md#term-matrix-meanings) row

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify exactly the five expected [planning](../../../../glossary.md#term-planning) paths above. It must not change
Java, tests, [API](../../../../glossary.md#term-application-programming-interface-api) or glossary documentation, architecture or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) files, [Gradle](../../../../glossary.md#term-gradle), another [module](../../../../glossary.md#term-module), or
create another task specification. If a correct audit requires a sixth path, stop and propose the
scope change before editing it.

## Required audit [artifact](../../../../glossary.md#term-artifact)

`adjoint-expressibility-audit.md` is authoritative only for subsequent [planning](../../../../glossary.md#term-planning) decisions. It is
not an [architecture contract](../../../../glossary.md#architecture-contract), public [API](../../../../glossary.md#term-application-programming-interface-api) reference, [gradient](../../../../glossary.md#term-gradient) implementation, or [backend](../../../../glossary.md#backend) promise.
`capabilities.md` must retain only a concise result summary and links; it must not duplicate the
[matrix](../../../../glossary.md#term-matrix-meanings) or become an [autograd](../../../../glossary.md#autograd) manual.

### [Matrix](../../../../glossary.md#term-matrix-meanings) fields

Each matrix row must contain or link locally to all of these fields:

1. public [Tensor](../../../../glossary.md#tensor) [operation](../../../../glossary.md#operation)/family and exact [`OperationKind`](../../../../glossary.md#operationkind)/attributes or primitive [producer](../../../../glossary.md#term-producer) chain;
2. [differentiable](../../../../glossary.md#derivative-and-chain-rule) input role under audit and explicitly non-differentiable roles;
3. forward result and incoming [cotangent](../../../../glossary.md#cotangent) [Shape](../../../../glossary.md#shape)/type assumptions;
4. required forward inputs, public outputs, recomputed intermediates, and exact shared producer
   output positions, if any;
5. exact backend-neutral adjoint formula using current public general operations where possible,
   including every broadcast reversal, axis insertion/removal, permutation, reduction, scatter,
   and gradient-accumulation step needed to restore the input Shape;<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [adjoint glossary](../../../../glossary.md#term-adjoint); [permutation glossary](../../../../glossary.md#permutation); [gradient glossary](../../../../glossary.md#term-gradient))</span>
6. operation-specific axis, rank, type, empty-domain, dynamic-Shape binding, index-bound,
   duplicate-target, random-state/mask, special-value, discontinuity, and tie obligations;<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [target glossary](../../../../glossary.md#term-target); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
7. one fixed classification and a concise proof tied to current semantics;
8. the minimum general public primitive candidate, if classification requires one, plus why it is
   useful beyond autograd and why a narrower backward-only kind is rejected;<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd))</span>
9. architecture [owner](../../../../glossary.md#term-owner) for later implementation; and
10. repository evidence and any minimal primary external reference.

Rows may share a formula block or evidence block, but no field may be silently inherited across
semantically different variants. Mathematical notation must define shapes and [axes](../../../../glossary.md#term-axis). [Pseudocode](../../../../glossary.md#term-pseudocode) or
proposed [API](../../../../glossary.md#term-application-programming-interface-api) names must be labeled as planned and non-[binding](../../../../glossary.md#term-binding).

### Fixed classifications

Use exactly one of these classifications per [differentiable](../../../../glossary.md#derivative-and-chain-rule) input-role row:

- `EXACT_CURRENT_COMPOSITION` — current public operations represent the exact adjoint for every
  accepted current Shape/semantic case, without depending on an unavailable forward output.<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- `EXACT_WITH_EXISTING_AUXILIARY_OUTPUT` — the exact composition uses a producer-described output
  that already exists, such as the dropout keep mask or batch-normalization saved statistics.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [mask glossary](../../../../glossary.md#term-boolean-mask); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- `MISSING_GENERAL_PUBLIC_PRIMITIVE` — exact meaning is representable by a stable tensor
  transformation useful outside autograd, but that public model primitive is absent.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [autograd glossary](../../../../glossary.md#autograd); [model glossary](../../../../glossary.md#term-model))</span>
- `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP` — after rejecting both current composition and every
  appropriate general public primitive, a stable compiler-only semantic meaning remains. This
  classification requires an explicit impossibility argument, not convenience or performance.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [argument glossary](../../../../glossary.md#term-argument))</span>
- `NON_DIFFERENTIABLE` — the role is integral, Boolean, state, index, or otherwise outside the
  selected differentiable domain.<span data-glossary-navigation="true"> ([Boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- `POLICY_DEFERRED` — current forward semantics do not select the derivative/subgradient behavior
  required at discontinuities, ties, duplicate targets, undefined points, or another observable
  boundary. Record the exact missing policy; do not choose it in this task.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [targets glossary](../../../../glossary.md#term-target))</span>

If a family contains roles with different classifications, split it. If a formula works only for
fully static [Shapes](../../../../glossary.md#shape) while the forward [operation](../../../../glossary.md#operation) accepts unresolved Shapes, it is not
`EXACT_CURRENT_COMPOSITION` for the general row.

### Required decision probes

The audit must explicitly resolve or defer all of these questions:

- **Broadcast reversal:** prove how [binary](../../../../glossary.md#term-binary-descriptor-and-metadata), `where`, matmul, attention, affine normalization, and
  loss adjoints reduce to each original input Shape. Treat a Dimension that may bind to one or to
  the result extent as a binding-dependent obligation; do not assume compile-time reduction axes.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [extent glossary](../../../../glossary.md#term-extent); [binding glossary](../../../../glossary.md#term-binding); [compile glossary](../../../../glossary.md#compile); [axes glossary](../../../../glossary.md#term-axis))</span>
- **Zero/one construction:** distinguish [eager](../../../../glossary.md#term-eager-evaluation) fully static `TensorFactory.zerosLike`/`onesLike`
  leaves from graph-expressible constants for unresolved Shapes. Prove how scatter bases, masks,
  and empty identities are constructed without contaminating NaN/infinity semantics.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [constants glossary](../../../../glossary.md#term-constant); [masks glossary](../../../../glossary.md#term-boolean-mask); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- **[Gather](../../../../glossary.md#gather):** test Gather, Gather Elements, and Gather-ND adjoints against current Scatter Elements
  and Scatter-ND Shape, batch-dimension, bounds, duplicate-index, reduction, and data-base
  semantics. Use addition only where repeated indices require accumulation.<span data-glossary-navigation="true"> ([Gather Elements glossary](../../../../glossary.md#gather-elements); [Gather-ND glossary](../../../../glossary.md#gather-nd); [Scatter Elements glossary](../../../../glossary.md#scatter-elements); [Scatter-ND glossary](../../../../glossary.md#scatter-nd); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- **Scatter:** audit adjoints separately for data and updates and for `NONE`, `ADD`, `MUL`, `MAX`,
  and `MIN`; account for replacement uniqueness, duplicate reduction, base participation, extrema
  ties, and saved/recomputed values.
- **[Slice](../../../../glossary.md#slice)/select:** prove whether current scatter operations, padding, or other current primitives
  place signed-step, multi-axis, empty, and dynamic-Shape slice cotangents exactly. If not, assess a
  general slice update/scatter/add primitive rather than a slice-backward kind.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [slice update glossary](../../../../glossary.md#slice-update))</span>
- **Windows:** prove the `unfold` [adjoint](../../../../glossary.md#term-adjoint) against the retained `FOLD_AXIS` semantics and current
  absence of public `Tensor.foldAxis`; distinguish `unfold2d`/`fold2d` and audit the reverse
  direction of every public fold.
- **[Convolution](../../../../glossary.md#term-channels-first-convolution-layers):** write both input and weight adjoints for grouped NCHW cross-correlation,
  including stride, symmetric padding, dilation, groups, bias, output padding/cropping, and dynamic
  Shapes. Test composition through current unfold2d/fold2d, matmul, reductions, and layouts before
  proposing convolution-transpose or correlation primitives.<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight); [NCHW glossary](../../../../glossary.md#term-nchw); [stride glossary](../../../../glossary.md#term-stride); [dilation glossary](../../../../glossary.md#term-dilation); [bias glossary](../../../../glossary.md#term-bias); [layouts glossary](../../../../glossary.md#layout); [transpose glossary](../../../../glossary.md#term-transpose))</span>
- **[Pooling](../../../../glossary.md#term-pooling):** audit average-pool overlap accumulation and fixed count-padding divisor. For
  max-pool, preserve excluded padding, all-padding negative infinity, NaN selection, signed zero,
  and first logical-kernel tie semantics; assess recomputation versus generally useful public
  indices/selection output without inventing a tie rule.<span data-glossary-navigation="true"> ([negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero); [kernel glossary](../../../../glossary.md#kernel))</span>
- **Reduction [extrema](../../../../glossary.md#saturation-pole-and-extremum) and piecewise operations:** distinguish a representation gap from an
  unselected subgradient/tie convention for min/max, clamp, absolute value, ReLU, sign, floor,
  ceil, sort/top-K, and related discontinuities.<span data-glossary-navigation="true"> ([absolute value glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); [ReLU glossary](../../../../glossary.md#term-relu))</span>
- **[Softmax](../../../../glossary.md#softmax--log-softmax), [log-softmax](../../../../glossary.md#softmax--log-softmax), [normalization](../../../../glossary.md#term-normalization), attention, and [losses](../../../../glossary.md#loss):** write exact formulas using public
  outputs or saved values where sufficient. Preserve current masked/all-masked, positive-infinity,
  NaN, empty-domain, correction, epsilon, ignore-index, and reduction-denominator semantics rather
  than relying on a generic textbook formula that covers only finite non-empty inputs.<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon); [finite glossary](../../../../glossary.md#term-finite))</span>
- **Randomness:** dropout adjoints must reuse the exact forward keep mask and scaling. They must not
  resample, infer a mask from the output, advance state, or differentiate RNG-state inputs.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- **Dynamic [Shapes](../../../../glossary.md#shape):** list every formula whose axes, base Shape, count, crop, or index construction
  depends on later binding. State whether current attributes can carry the exact obligation and
  whether a general public primitive is needed.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry))</span>

### Decision gate for follow-up rows

After every [matrix](../../../../glossary.md#term-matrix-meanings) row is complete:

1. merge gaps only when one cohesive general public primitive closes them under one semantic
   contract;
2. select the smallest set of required public [model](../../../../glossary.md#term-model) primitives, with `foldAxis` restoration or
   redesign explicitly decided;
3. add one concise Draft row per selected primitive/capability immediately after 0023 and before
   0024 in the model master plan and roadmap;
4. make each row depend on 0023 and order the rows by semantic [dependency](../../../../glossary.md#term-dependency);
5. make 0024 depend on 0023 and every required inserted model follow-up;
6. leave optional optimizations and [compiler](../../../../glossary.md#term-compiler) implementation work outside the model queue; and
7. create no detailed follow-up task file.

If no public primitive is needed for a candidate, record that no-change conclusion rather than
adding a placeholder row. A compiler-only kind may receive a Draft row only with the required
impossibility proof and exact [backend](../../../../glossary.md#backend)-neutral semantic gap in the matrix.

## Acceptance criteria

- The audit artifact states its planning-only authority and defines its notation and
  classifications before the matrix.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [planning glossary](../../../../glossary.md#term-planning); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Source, public API, operation-kind/signature, producer-output, and completed-task inventories
  account for every current public semantic family and convenience producer chain in scope.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [producer glossary](../../../../glossary.md#term-producer); [scope glossary](../../../../glossary.md#term-scope))</span>
- Every selected differentiable input role has an exact family-level or operation-level adjoint
  formula and every required matrix field.<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- Every non-differentiable role is explicit; no role is omitted merely because its result metadata
  cannot request gradients.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Every formula restores the exact input Shape and type or identifies the precise unresolved
  policy/primitive gap. Static-only constructions are visibly distinguished from the full accepted
  dynamic-Shape contract.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- The required gather/scatter, slice/select, window, convolution, pooling, extrema, softmax,
  normalization, attention, loss, random-state, saved-output, and dynamic-Shape probes are each
  answered with repository evidence.<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather); [slice glossary](../../../../glossary.md#slice); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [loss glossary](../../../../glossary.md#loss))</span>
- The audit does not preselect or catalog operation-specific backward kinds. Formula length,
  performance, kernel availability, or fusion never appears as justification for model semantics.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [fusion glossary](../../../../glossary.md#term-fusion); [model glossary](../../../../glossary.md#term-model))</span>
- Every `MISSING_GENERAL_PUBLIC_PRIMITIVE` row identifies a reusable public transformation and
  rejects a narrower backward-only spelling. Every `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP` row
  contains the required concrete impossibility proof.
- The minimum evidence-backed Draft follow-up rows are inserted before preserved task 0024, in
  dependency order, without detailed specifications. No speculative row is added merely to cover
  a table category.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- `capabilities.md` contains only a concise linked result summary; the detailed formulas and matrix
  exist only in `adjoint-expressibility-audit.md`.
- The model master plan, roadmap, capabilities, task status, dependencies, frontier text, and
  completed history agree. Task 0024 remains Draft and has no detailed specification.
- A separate clean documentation-focused agent reviews and finalizes the five planning paths in
  the same overall change using the General and Planning profiles. It records glossary and API
  no-change reasons and does not rerun successful Java suites.
- Exactly the five permitted planning paths change. There are no Java, test, API, glossary,
  architecture, ADR, Gradle, module, conformance, or integration changes.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Local Markdown links and heading anchors resolve, code fences are balanced, all changed files
  end with one newline, no trailing whitespace exists, and `git diff --check` passes.

## Tests / [validation](../../../../glossary.md#term-validation)

This task changes [planning](../../../../glossary.md#term-planning) documentation only. Do not [run](../../../../glossary.md#term-run) Java tests or [Javadoc](../../../../glossary.md#term-javadoc). Reuse the
recorded post-0022B capability-[checkpoint](../../../../glossary.md#term-checkpoint) evidence: root tests passed with 966 tests across 124
suites, [model](../../../../glossary.md#term-model) Javadoc passed, the public [Tensor](../../../../glossary.md#tensor) surface contained 188 methods, 657 local links and
176 anchors passed, and no architecture conflict was found. Record the originating checkpoint
context and confirm that this audit changed no executable or [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc source afterward.

Run final [scope](../../../../glossary.md#term-scope) and status inspection:

```bash
git status --short
git diff --name-only
git ls-files --others --exclude-standard
rg -n "Task 0023|Task 0024|0023|0024" \
  docs/planning/modules/model/capabilities.md \
  docs/planning/modules/model/master-plan.md \
  docs/planning/roadmap.md \
  docs/planning/modules/model/tasks/0023-adjoint-expressibility-audit.md \
  docs/planning/modules/model/adjoint-expressibility-audit.md
find docs/planning/modules/model/tasks -maxdepth 1 -type f -name '0024*.md' -print
```

The final path inventory must contain exactly the five paths under [Affected files](#affected-files),
and the `find` command must print nothing. Inspect the [matrix](../../../../glossary.md#term-matrix-meanings) for all fixed classification [tokens](../../../../glossary.md#term-token),
every required family, exact shared output indices, required decision probes, no backward-kind
catalog, and no implementation claim.

Run a read-only Markdown validation over the five changed files that checks relative file [targets](../../../../glossary.md#term-target),
GitHub-style heading anchors, balanced code fences, and final newlines. Record the exact checker
command and the number of links and anchors checked. Then run:

```bash
git diff --check
```

Repository-wide validation is not repeated because the recorded post-0022B capability checkpoint
already closed the model-family frontier and this task changes no executable, [dependency](../../../../glossary.md#term-dependency), build,
architecture, or public API contract.

## Dependencies

- Tasks 0005–0006 for [backend](../../../../glossary.md#backend)-neutral semantic [operation](../../../../glossary.md#operation) [ownership](../../../../glossary.md#term-ownership-meanings).
- Tasks 0013, 0018K, and 0018L for exact [producer](../../../../glossary.md#term-producer)/[signature](../../../../glossary.md#term-signature)/indexed multi-output [provenance](../../../../glossary.md#provenance).
- Tasks 0018M–0018M1 and current [Shape](../../../../glossary.md#shape) contracts for symbolic and dynamic [extent](../../../../glossary.md#term-extent) obligations.
- Task 0018N for exact typed [scalar](../../../../glossary.md#term-scalar) configuration.
- Completed operation-family tasks 0014A–0022B, including indexing cleanup 0018O, slice/window
  cleanup 0018R, modern-model tasks 0019–0020A1, normalization tasks 0021–0021C, and loss tasks
  0022–0022B.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [model glossary](../../../../glossary.md#term-model); [normalization glossary](../../../../glossary.md#term-normalization); [loss glossary](../../../../glossary.md#loss))</span>
- The completed post-0022B capability [checkpoint](../../../../glossary.md#term-checkpoint) with the evidence recorded above.

## Follow-up tasks

- Required: the minimum public model primitive tasks selected by the completed matrix. The audit
  inserts only concise Draft rows before 0024 and creates no detailed specifications.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Required after those rows: task 0024, the preserved Draft [model](../../../../glossary.md#term-model) capability selection audit.
- Later compiler planning owns gradient rules, graph traversal/capture, saved-value lifetime,
  gradient accumulation, and backward graph construction after the required model primitives are
  available.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [gradient glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [capture glossary](../../../../glossary.md#term-capture-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime); [gradient accumulation glossary](../../../../glossary.md#term-gradient-accumulation); [backward graph glossary](../../../../glossary.md#backward-graph))</span>
- Later backend/runtime work owns lowering, conformance, algorithms, prepared execution, and
  numerical execution.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [lowering glossary](../../../../glossary.md#lowering); [conformance glossary](../../../../glossary.md#term-conformance); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>

## Architecture impact

Expected impact: None.

This audit applies existing [model](../../../../glossary.md#term-model)/[compiler](../../../../glossary.md#term-compiler)/[backend ownership](../../../../glossary.md#backend-ownership) and does not change [dependency](../../../../glossary.md#term-dependency) rules.
If the audit discovers that an exact required meaning cannot fit those boundaries, stop and report
the conflicting rule and decision needed instead of changing architecture documentation.

## Implementation prompt

Use this prompt in a separate clean agentic task/thread:

```text
You are working in the Synaptik repository on planning/design documentation only.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md, the General and Planning documentation profiles,
and docs/planning/modules/model/tasks/0023-adjoint-expressibility-audit.md. Read the current
operation/source/test/API/task evidence required by that specification.

Execute task 0023 exactly as specified. Create the one planning-only audit artifact and update
only the five permitted planning paths. Do not implement Java/tests, create a backward-kind
catalog, create a detailed follow-up specification, edit architecture/API/glossary/Gradle, commit,
or push. Stop on an architecture conflict, missing exact semantic policy, or required scope
expansion rather than inventing behavior.

After the audit draft and planning validation, hand the exact diff and reused post-0022B
checkpoint evidence to a separate documentation-focused agent/thread with clean context. That
agent must independently finalize the five planning paths under the General and Planning profiles,
validate links/anchors/fences/newlines/whitespace, record API/glossary/Javadoc no-change reasons,
and must not repeat successful Java tests.

Finally update the task file with decisions, limitations, exact validation evidence, documentation
review, completion summary, and status. Do not mark Complete until the matrix, evidence-backed
follow-up queue, synchronization, and documentation pass are final.
```

## Local decisions

- Task 0023 is audit-only. The breadth and independent semantic gaps make implementation before
  the matrix unsafe and non-cohesive.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- The detailed result belongs in one sibling planning artifact rather than capabilities, an API
  guide, the glossary, or architecture documentation.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [artifact glossary](../../../../glossary.md#term-artifact); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Fixed row classifications make absence, non-differentiability, unselected policy, reusable
  primitive gaps, and genuine compiler-only gaps independently reviewable.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- Follow-up rows were selected only after the full matrix; the Ready specification did not
  preselect <span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>`foldAxis`, slice placement, convolution transpose/correlation, sum-to-Shape, or any
  compiler-only kind.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [transpose glossary](../../../../glossary.md#term-transpose); [sum-to-Shape glossary](../../../../glossary.md#sum-to-shape))</span>
- Task 0024 keeps its established ID, Draft status, and lack of a detailed specification.
- Exact typed [scalar](../../../../glossary.md#term-scalar) zero/one leaves followed by `expand(targetShape)` are sufficient graph
  constants for unresolved Shapes. They replace the initial zero-like candidate without reading
  template values or contaminating NaN/infinity semantics.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [constants glossary](../../../../glossary.md#term-constant); [Shapes glossary](../../../../glossary.md#shape); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- Current Scatter Elements ADD exactly expresses Gather Elements adjoints, and Scatter-ND ADD
  exactly expresses Gather-ND adjoints. Rank-changing Gather still needs a distinct generally
  useful axis scatter-add for an unresolved gathered extent because neither current scatter has
  its Shape relation and current one-hot requires positive static depth. Positive static extents
  already compose through one-hot selection and reduction; a statically zero valid domain returns
  the zero base.<span data-glossary-navigation="true"> ([Scatter Elements glossary](../../../../glossary.md#scatter-elements); [Gather Elements glossary](../../../../glossary.md#gather-elements); [Scatter-ND glossary](../../../../glossary.md#scatter-nd); [Gather-ND glossary](../../../../glossary.md#gather-nd); [Rank glossary](../../../../glossary.md#term-rank); [Gather glossary](../../../../glossary.md#gather); [axis glossary](../../../../glossary.md#term-axis); [extent glossary](../../../../glossary.md#term-extent))</span>
- Ordinary [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic, `where`, and `EXPAND` expose only leading or statically known
  singleton expansion axes, so fixed reductions reverse their current accepted broadcasts.
  Binding-aware sum-to-Shape remains required for MATMUL and attention because those helpers
  explicitly defer unresolved singleton-or-equal batch obligations.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [Binding glossary](../../../../glossary.md#term-binding); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Signed non-zero slice coordinates are injective. Draft 0023C needs general target-Shape
  placement/update and target-relative crop, but the audit does not require an additive overlap
  mode or a slice-backward kind.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [injective glossary](../../../../glossary.md#term-injective); [target glossary](../../../../glossary.md#term-target); [target-relative crop glossary](../../../../glossary.md#target-relative-crop))</span>
- Maximum-pool routing does not need a new producer output. Negative-infinity padding, dynamic
  window materialization, existing first-index arg-max, numeric <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [padding glossary](../../../../glossary.md#padding); [materialization glossary](../../../../glossary.md#materialization))</span>`where` over one-hot selection,
  and exact fold/crop compose it once the general window gap is closed.
- The matrix selects six Draft public-capability rows: binding-aware sum-to-Shape,
  Gather-compatible axis scatter-add, signed slice placement plus target-relative dynamic crop,
  public foldAxis plus dynamic/configurable 2D windows, cumulative product, and same-occurrence
  attention weights.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [weights glossary](../../../../glossary.md#term-weight))</span>
- Dynamic 2D windows cannot be obtained by only relaxing current helper validation: canonical
  rank-three columns flatten <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`outputHeight*outputWidth`, while current Dimension expressions do
  not multiply two unresolved extents. Draft 0023D must select either sufficient Shape algebra or
  a non-flattened dynamic window result/target contract.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension))</span>
- No row survives as `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP`; no compiler-only or
  operation-specific backward semantic row is added.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

## Known limitations

- This task does not implement or test automatic differentiation. Its formulas are planning
  evidence for later compiler tasks.<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd); [planning glossary](../../../../glossary.md#term-planning); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Rows may remain `POLICY_DEFERRED`; that is a valid audit result when current forward contracts do
  not select an observable derivative convention. Such rows must name the later decision owner.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [owner glossary](../../../../glossary.md#term-owner))</span>
- The audit identifies minimum model prerequisites but does not plan the later compiler traversal,
  accumulation, capture, optimization, or publication sequence.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [capture glossary](../../../../glossary.md#term-capture-meanings); [publication glossary](../../../../glossary.md#term-publication))</span>
- Regular-domain formulas do not choose derivatives at NaNs, infinities, singularities,
  discontinuities, extrema ties, scatter MUL/extrema edge cases, empty/all-masked domains, or
  ordering cutoffs. The matrix records each as <span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>`POLICY_DEFERRED` rather than inventing behavior.
- Concise Draft rows 0023A–0023F are planning labels only. They do not stabilize method names,
  attributes, overloads, result carriers, package placement, or detailed task scope.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Reused the completed post-0022B capability checkpoint because this task changes planning text
  only: 966 root tests across 124 suites passed, model Javadoc passed, 188 public Tensor methods
  were inventoried, and the checkpoint documentation validation passed 657 links and 176 anchors.
  Java tests and Javadoc were not rerun.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [planning glossary](../../../../glossary.md#term-planning); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Inspected the current public Tensor methods, operation kinds, attributes, signatures, expression
  helpers, shared-producer output positions, completed task evidence, and dynamic Shape contracts.
  The matrix covers every required family and decision probe and selects no operation-specific
  backward kind or compiler-only semantic gap.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature); [producer glossary](../../../../glossary.md#term-producer); [Shape glossary](../../../../glossary.md#shape); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Independent documentation context
  `/root/execute_0023_audit/docs_review_0023` finalized the five planning paths under the General
  and Planning profiles. It corrected broadcast reversal, static Gather composition,
  batch-normalization auxiliary use, injective signed-slice placement, product Shape restoration,
  and the unresolved-product constraint on dynamic two-dimensional windows.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [injective glossary](../../../../glossary.md#term-injective); [slice glossary](../../../../glossary.md#slice))</span>
- Exact five-path inventory passed. No `0024*.md` task specification exists. Tasks 0023A–0023F
  and 0024 remain Draft, and 0024 depends through 0023F.
- The independent read-only Markdown checker and the local bounded checker each passed 486 local
  links and one heading anchor, balanced fences, and single final newlines across the five paths.
  Required family, classification, producer-slot, decision-probe, and no-backward-kind searches
  passed. Trailing-whitespace search and `git diff --check` passed.
- Tensor, Compile, Runtime, and Training API references remain unchanged because no public or
  executable API changed. Javadoc remains accurate because no Java contract changed. No glossary
  update is needed because the audit defines local notation without stabilizing a public term.
  Architecture documents/tests, Java/tests, Gradle, other modules, backend conformance, and
  integration tests remain unchanged because ownership, dependencies, build structure, and
  executable behavior did not change.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Implementation notes

- Created the [planning](../../../../glossary.md#term-planning)-only [adjoint expressibility audit](../adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([adjoint glossary](../../../../glossary.md#term-adjoint))</span>.
- Inventoried all current public Tensor methods, operation families/attributes/signatures,
  expression helpers/tests, completed task evidence, API boundaries, dynamic Shape contracts, and
  shared producer slots.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Shape glossary](../../../../glossary.md#shape); [producer glossary](../../../../glossary.md#term-producer))</span>
- Synchronized capabilities, the model master plan, and the roadmap with the six evidence-backed
  Draft rows before preserved task 0024. No follow-up task specification was created.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- No Java, tests, Javadoc, API, glossary, architecture, ADR, Gradle, other-module, conformance, or
  integration path was changed.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Completion summary

- Completed changes: produced the planning-only adjoint expressibility matrix and selected the six
  minimum general public-capability follow-ups without adding a backward-specific or compiler-only
  semantic kind.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [adjoint glossary](../../../../glossary.md#term-adjoint); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Files changed or created: `adjoint-expressibility-audit.md`, this task specification,
  `capabilities.md`, `master-plan.md`, and the repository roadmap.
- Tests and validation: reused the post-0022B executable/Javadoc checkpoint and passed the exact
  scope, status, matrix-probe, Markdown, link, anchor, fence, newline, whitespace, and diff checks.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation review: the required independent clean-context documentation pass completed with
  no blocker and synchronized all five planning paths.
- Documentation, Javadoc, and glossary impact: planning documents only; the API references,
  Javadoc, glossary, architecture documents, and implementation documentation require no change.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Unresolved issues: none for this audit.
- Required follow-up: execute Draft tasks 0023A–0023F in order before Draft task 0024; their
  detailed specifications are intentionally not part of this task.

Status: Complete
