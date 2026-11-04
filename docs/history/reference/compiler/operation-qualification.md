# Historical reference [snapshot](../../../glossary.md#term-snapshot): [Operation](../../../glossary.md#operation) [capture](../../../glossary.md#term-capture-meanings) and numerical [qualification](../../../glossary.md#term-qualification) notes

This is an archived, composite qualification narrative preserved from the pre-redesign [API](../../../glossary.md#term-application-programming-interface-api) reference at base `f3f7c2face52c1fa886e075a68faa6d11765cbd0`. Its task-era inventories were accumulated at different [checkpoints](../../../glossary.md#term-checkpoint): this is not a single [release](../../../glossary.md#term-release)-wide capability statement. All quoted “current”, “planned”, “unsupported” and version/count claims below belong to those earlier checkpoints, not today's API.

**Known supersessions:** [CPU](../../../glossary.md#term-cpu) low-[precision](../../../glossary.md#term-precision) parity, fixed-cell first-order [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt), bounded static length-aware cold/[owned Model](../../../glossary.md#model-module-buffer-and-scan-distinctions) workflows and wire versions through v7, shared [normalization](../../../glossary.md#term-normalization) [derivatives](../../../glossary.md#term-derivative), and ordinary [backend](../../../glossary.md#backend) preparation/execution are implemented. The archive does not retract or widen any current qualification. Use [current reference goals](../../../reference/index.md), [Compiler AD](../../../reference/compiler/autograd.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler); [AD glossary](../../../glossary.md#autograd))</span>, [owned Model execution](../../../reference/nn/owned-model.md), and [current backend qualification](../../../backend-guide/production-support.md) instead of the inventories below. Symbolic cold/wire, arbitrary [PURE](../../../glossary.md#term-pure)-body copying, higher-order recurrence and unqualified backend routes are not granted by these supersessions.

## Editorial reading key: three different acceptances

The Model module defines an operation's meaning. The application supplies Tensor expressions;
the Compiler captures their producing operations and validates compatible descriptions.
A backend capability claim then concerns executing an exact occurrence, including its input and
output types, shapes and layouts. Acceptance at the first two boundaries is not proof of the third.
Likewise, a publication binding associates identities; it does not contain or own result bytes.

Read [capture and graph-value construction](../../../reference/compiler/capture.md) to understand
why one producer can have several output positions, and
[functional differentiation](../../../reference/compiler/autograd.md) for why constructing a
gradient expression neither evaluates it nor updates parameters. The
[run/result lifecycle](../../../reference/engine/lifecycle.md) explains who keeps input storage
alive and who closes a completed result. The archived operation counts and numerical policies
remain statements of their original checkpoints.

<a id="operation-capture-and-numerical-qualification-notes"></a>

## Preserved historical text

> **[Operation](../../../glossary.md#operation) [capture](../../../glossary.md#term-capture-meanings) and numerical [qualification](../../../glossary.md#term-qualification) notes**
>
> Before a [node](../../../glossary.md#node) enters that container, `Operation` validates its exact kind/attributes pairing and
> derives a family-owned `OperationSignature`. `CompiledNode` preserves its existing local list
> rules and then checks that the final ordered input and output counts lie within that signature's
> inclusive bounds. This catches a unary operation connected to two inputs, for example, without
> claiming that operand Shapes or data types are compatible. Zero-input, bounded, variadic-input,
> and multi-output occurrences remain representable when their kind explicitly declares them.<span data-glossary-navigation="true"> ([signature glossary](../../../glossary.md#term-signature); [unary glossary](../../../glossary.md#term-unary); [variadic glossary](../../../glossary.md#term-variadic))</span>
>
> A `ForwardPublicationBinding` carries only the public Tensor identity and graph-local value
> identity for one requested forward result. A <span data-glossary-navigation="true"> ([identity glossary](../../../glossary.md#term-identity-meanings); [graph glossary](../../../glossary.md#graph))</span>`GradientPublicationBinding` instead carries
> derivative order, stage-local target index, target Tensor identity, and final gradient<span data-glossary-navigation="true"> ([derivative glossary](../../../glossary.md#term-derivative); [target glossary](../../../glossary.md#term-target); [gradient glossary](../../../glossary.md#term-gradient))</span>
> `valueId()`. Current [compiler](../../../glossary.md#term-compiler)-owned `PublicationPlan` groups both ordered binding lists with the
> exact final graph. Neither binding retains a public <span data-glossary-navigation="true"> ([binding glossary](../../../glossary.md#term-binding))</span>[`Tensor`](../../../glossary.md#tensor), publication policy, runtime target,
> storage, backend, or execution state.<span data-glossary-navigation="true"> ([publication glossary](../../../glossary.md#term-publication); [runtime glossary](../../../glossary.md#term-runtime); [backend glossary](../../../glossary.md#backend))</span>
>
> `GraphRngState.initial(key, counter)` is also current model construction. It creates a fresh
> zero-input, one-output `GraphRngKind.INITIAL_STATE` [occurrence](../../../glossary.md#term-occurrence) whose exact
> `GraphRngStateAttrs(long key, long counter)` words are retained in provenance. The private state
> Tensor is fixed <span data-glossary-navigation="true"> ([provenance glossary](../../../glossary.md#provenance))</span>`INT64 Shape[2]`, unresolved-layout, non-gradient, unlabeled, and storage-free,
> with output index zero. Both <span data-glossary-navigation="true"> ([layout glossary](../../../glossary.md#layout))</span>`long` values are unsigned 64-bit bit patterns: key identifies a
> caller-selected stream/domain, and counter identifies the next abstract logical sample position.
> Equal words request replay-equivalent positions but do not merge expression identities in the
> public expression model.<span data-glossary-navigation="true"> ([sample glossary](../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
>
> Current package-private structural capture preserves this state producer and its ordered state
> edges when they are reachable from a requested Tensor result. A future serializer must preserve
> the raw words losslessly. Current exact internal CSE may merge equal initial-state and dropout
> graph occurrences only when their complete operation, ordered inputs, phase, and all output
> descriptors match; this does not merge their pre-capture expression identities or select a random
> algorithm. No current code defines serialization, gradient rules, a byte encoding, a stable enum
> token, prepared state, or execution. Without a selected portable algorithm, no cross-backend or
> cross-version bitstream follows from equal state.<span data-glossary-navigation="true"> ([structural capture glossary](../../../glossary.md#term-structural-capture))</span><span data-glossary-navigation="true"> ([CSE glossary](../../../glossary.md#term-common-subexpression-elimination-cse); [dropout glossary](../../../glossary.md#dropout--inverted-dropout); [serialization glossary](../../../glossary.md#term-serialization); [token glossary](../../../glossary.md#term-token))</span>
>
> The public `Tensor` model is also current. Its seven binary arithmetic methods, six binary
> comparison methods, three boolean logical methods, nineteen unary elementwise methods, three
> floating-classification methods, seven exact-typed plus seven exact-FLOAT64 scalar arithmetic
> methods, and six range/one-bound clamp
> methods, plus one static conditional-selection method and one explicit cast method, fifteen
> full/axis numeric aggregate methods, one binding-aware target-Shape SUM method, six full/axis
> boolean aggregate methods, fourteen ordinary
> multi-axis methods, twelve floating advanced/statistical reduction methods, two axis-removing
> masked aggregate methods, six axis-only <span data-glossary-navigation="true"> ([binary glossary](../../../glossary.md#term-binary-descriptor-and-metadata); [boolean glossary](../../../glossary.md#data-type--datatype); [FLOAT64 glossary](../../../glossary.md#term-float64))</span>`argMin`/`argMax` methods, two one-[axis](../../../glossary.md#term-axis) `cumSum` methods,
> and two one-axis `cumProd` methods, plus one-axis [`softmax`](../../../glossary.md#softmax--log-softmax), `logSoftmax`, two trailing-[Shape](../../../glossary.md#shape)
> `layerNorm`
> methods, and two trailing-Shape `rmsNorm`
> methods, plus explicit-axis five-input `batchNormInference` and `batchNormTraining` methods,
> scalar `select`, two
> tensor-index axis-gather methods, the <span data-glossary-navigation="true"> ([gather glossary](../../../glossary.md#gather))</span>[`embedding`](../../../glossary.md#embedding) convenience over axis-zero Gather, one
> trailing-axis `oneHot` index-encoding method, and two Gather-ND methods, plus two functional
> Scatter Elements methods, one Gather-compatible functional Scatter Add method, and three
> functional Scatter-ND methods, plus one matrix-multiplication method, construct
> storage-free expressions with immutable producer-and-output-index provenance. Four full-ordering
> methods add ascending or explicitly directed <span data-glossary-navigation="true"> ([Gather-ND glossary](../../../glossary.md#gather-nd); [Scatter Elements glossary](../../../glossary.md#scatter-elements); [Scatter Add glossary](../../../glossary.md#scatter-add); [Scatter-ND glossary](../../../glossary.md#scatter-nd); [matrix glossary](../../../glossary.md#term-matrix-meanings); [immutable glossary](../../../glossary.md#term-immutable))</span>`sort` and `argsort` requests. Every current
> single-output expression creates one identity-distinct producer whose ordered descriptor list has
> one entry and whose provenance index is zero. The producer snapshots the exact operation and input
> Tensor references and retains the exact canonical Tensor wrapper for that output. Its public
> indexed lookup returns that same wrapper without reconstruction. The parameterless <span data-glossary-navigation="true"> ([lookup glossary](../../../glossary.md#term-lookup))</span>[`contiguous`](../../../glossary.md#term-contiguous)
> method
> adds the same expression provenance for a canonical-layout request, and the two `reshape`
> overloads add normalized target-shape expressions. Two `expand` overloads add directional
> right-aligned target-shape expressions.
> Binary ADD, SUB, MUL, MIN, and MAX now accept same-category floating or signed-integral operands;
> DIV and POW remain floating-only. All six comparisons accept the same floating or integral
> pairing rules and produce BOOL results. Integral promotion uses `INT32 < INT64`, with conceptual
> sign extension into the INT64 domain, while mixed numeric categories require an explicit cast and
> BOOL is excluded. Numeric unary and conditional-selection results remain floating. Unary<span data-glossary-navigation="true"> ([sign extension glossary](../../../glossary.md#term-sign-extension); [INT64 glossary](../../../glossary.md#data-type--datatype))</span>
> `rsqrt`, `log1p`, `expm1`, exact `gelu`, fixed `geluTanhApproximation`, and [canonical](../../../glossary.md#term-canonical-form) `silu` are
> first-class transforms rather than stored decompositions. Exact GELU selects <span data-glossary-navigation="true"> ([GELU glossary](../../../glossary.md#gaussian-error-linear-unit--gelu))</span>`x * Phi(x)`; the
> tanh spelling selects only its fixed conventional `0.044715` approximation; and [SiLU](../../../glossary.md#sigmoid-linear-unit--silu) selects
> `x * sigmoid(x)`. There is no `swish` [alias](../../../glossary.md#aliasing-and-arena) or configurable approximation. Together with [`exp`](../../../glossary.md#term-expx)
> and [`tanh`](../../../glossary.md#term-tanh), they record portable mathematical requests without selecting an algorithm, bitwise
> result, fixed accuracy bound, gradient rule, or backend route. Floating classifications,
> comparisons, and logical results are unresolved-layout <span data-glossary-navigation="true"> ([backend route glossary](../../../glossary.md#backend-route))</span>[`BOOL`](../../../glossary.md#data-type--datatype) descriptors with false gradient
> eligibility. Logical AND and OR require exact BOOL inputs and derive a local broadcast shape;
> logical NOT requires exact BOOL and retains the exact input shape. Scalar parameters remain exact
> typed <span data-glossary-navigation="true"> ([parameters glossary](../../../glossary.md#term-parameter))</span>[`ScalarValue`](../../../glossary.md#typed-scalar-value--scalarvalue) operation attributes rather than Tensor inputs. Scalar ADD, SUB, MUL, MIN, and
> MAX plus one-bound clamp conveniences accept exact matching floating or integral values; scalar
> DIV, POW, and first-class range CLAMP remain floating-only. Constant padding uses exact equality
> for all seven current data types. Tensor-to-Tensor and scalar arithmetic share the seven semantic
> kinds, while the accepted integral subset is the five operations above; public pairwise extrema
> are named <span data-glossary-navigation="true"> ([Constant glossary](../../../glossary.md#term-constant); [extrema glossary](../../../glossary.md#saturation-pole-and-extremum))</span>`minimum`/`maximum`, while aggregate reductions remain `min`/`max`. Floating extrema
> propagate NaN, order infinities normally, and select negative zero for minimum or positive zero
> for maximum, independent of operand order. Equal nonzero candidates produce their numeric value
> without a selected-operand promise. Floating ordered comparisons are false if either operand is
> NaN and treat negative and positive zero as equal. Floating <span data-glossary-navigation="true"> ([NaN glossary](../../../glossary.md#term-nan); [infinities glossary](../../../glossary.md#nan-infinity-and-signed-zero); [negative zero glossary](../../../glossary.md#nan-infinity-and-signed-zero); [positive zero glossary](../../../glossary.md#nan-infinity-and-signed-zero))</span>`EQUAL` is exact represented numeric
> equality rather than bit or tolerance equality, and <span data-glossary-navigation="true"> ([tolerance glossary](../../../glossary.md#term-tolerance))</span>`NOT_EQUAL` is its logical complement.
> FLOAT64, FLOAT32, BFLOAT16, and FLOAT16 use their represented values under existing promotion. Range<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../glossary.md#term-float16))</span>
> `CLAMP` remains first-class with exactly ordered `MIN(MAX(input, minValue), maxValue)` meaning;
> `clampMin` creates one [scalar](../../../glossary.md#term-scalar) `MAX` [producer](../../../glossary.md#term-producer) and `clampMax` one scalar `MIN` producer. Integral
> ADD, SUB, and MUL have fixed-width two's-complement modular meaning, and integral MIN, MAX, and
> comparisons use signed order. These are current model semantics and metadata-construction facts,
> not compiler validation, derivative-policy, lowering, backend, or execution claims. Package-
> private structural capture preserves them, and the following package-private verification pass
> independently revalidates them. An <span data-glossary-navigation="true"> ([two's-complement glossary](../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks); [metadata glossary](../../../glossary.md#term-metadata); [validation glossary](../../../glossary.md#term-validation); [lowering glossary](../../../glossary.md#lowering))</span>`OperationSignature`
> still validates only the exact attribute class and occurrence cardinality because an <span data-glossary-navigation="true"> ([cardinality glossary](../../../glossary.md#term-cardinality))</span>`Operation`
> has no operand
> descriptor. Current package-private compiler verification revalidates operand domains and exact
> scalar/input data-type equality for captured or otherwise constructed occurrences. `Tensor.where` requires an exact BOOL
> condition, promotes two floating branches, composes branch-first and condition-second local
> broadcasts, propagates gradient eligibility from the branches only, and records exact ordered
> condition/true-branch/false-branch provenance. It constructs no selected values or gradient rule.
> `Tensor.cast` accepts every current source/target data-type pair, retains the exact input shape,
> leaves layout unresolved, and retains a true gradient request only for floating-to-floating casts.
> Every call remains a fresh explicit expression, including a same-type request, with typed target
> attributes and exact one-input provenance. Model's
> [`CastValueConversions` contract](../../../reference/tensor/selection-casts.md#cast-expressions) now fixes the observable value
> meaning for all 49 pairs; Compiler continues only to preserve and validate the expression and to
> apply the existing floating-only reverse rule described above.
> `Tensor.sumToShape(Shape)` currently constructs one fresh numeric SUM expression whose descriptor
> and `SumToShapeAttrs` retain the exact target Shape. The target is right-aligned with the input:
> leading axes reduce, aligned target-one axes reduce and remain, and equal aligned axes preserve.
> Provable static incompatibility fails locally; any pair involving an unresolved Dimension remains
> an obligation for later binding validation. The result preserves exact input type and gradient
> eligibility, has unresolved layout, and records exact one-input/output-index-zero provenance.
> Current package-private capture preserves this model metadata structurally, and current
> package-private verification represents and proves or retains the Shape obligation. Adjoint
> construction is current only for the locally provable floating subset described in the autograd
> matrix above. The current fail-closed CPU path advertises, lowers, prepares, and executes only
> fully static, resolved-layout SUM_TO_SHAPE occurrences over FLOAT64, FLOAT32,
> INT32, or INT64 with exact source/result type, right-aligned target Shape, leading- and target-one-axis
> reduction, preserved equal axes, non-negative layouts, and an injective output. Binding-dependent
> inversion outside the qualified <span data-glossary-navigation="true"> ([Dimension glossary](../../../glossary.md#dimension); [Adjoint glossary](../../../glossary.md#term-adjoint); [autograd glossary](../../../glossary.md#autograd); [fail-closed glossary](../../../glossary.md#term-fail-closed); [CPU glossary](../../../glossary.md#term-cpu); [INT32 glossary](../../../glossary.md#data-type--datatype); [injective glossary](../../../glossary.md#term-injective))</span>[ordinary FLOAT32 named-B slice](../../../reference/compiler/batch-binding.md#ordinary-named-batch-metadata-binding)<span data-glossary-navigation="true"> ([slice glossary](../../../glossary.md#slice))</span>,
> general dynamic binding, and other-backend execution remain planned. That slice also executes
> SUM-to-Shape with actual B, including the raw point-copy case at B=1.<span data-glossary-navigation="true"> ([SUM-to-Shape glossary](../../../glossary.md#sum-to-shape))</span>
> `Tensor.matmul` currently constructs one fresh two-input MATMUL expression with a locally derived
> vector, matrix, or broadcast-batch Shape and same-category promoted numeric type. Unequal static
> contraction dimensions fail locally; unresolved contraction equality and the accepted
> unresolved-versus-static batch singleton-or-equal cases remain obligations for later compiler
> validation or concrete binding. Package-private structural capture plus graph-wide verification
> and conservative constraint proof are current. Current package-private autograd constructs
> role-aware floating MATMUL cotangents for all four vector/matrix rank pairings when the selected
> operand has the output type; integral and cross-floating selected roles remain unsupported. The
> current fail-closed CPU path advertises, lowers, prepares, and executes only the fully static,
> resolved-layout MATMUL subset over FLOAT32/FLOAT64 or integral operands with exact contraction
> and batch geometry, supported same-category promotion, non-negative layouts, and an injective output. The separate ordinary named-B slice
> also executes homogeneous FLOAT32 rank-two MATMUL and its actual first-order adjoints, with fixed
> features/parameters. General dynamic binding, wider dynamic types and other-backend execution
> remain planned.<span data-glossary-navigation="true"> ([vector glossary](../../../glossary.md#term-vector); [batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [contraction glossary](../../../glossary.md#contraction-dimension); [singleton glossary](../../../glossary.md#term-singleton); [rank glossary](../../../glossary.md#term-rank); [homogeneous glossary](../../../glossary.md#term-homogeneous); [rank-two glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix); [features glossary](../../../glossary.md#term-feature))</span>
> `Tensor.linear(weight)` and `Tensor.linear(weight, bias)` are also current model construction, but
> they add no LINEAR operation. Conventional <span data-glossary-navigation="true"> ([LINEAR glossary](../../../glossary.md#term-linear-meanings))</span>`[outFeatures, inFeatures]` weight is explicitly
> transposed through PERMUTE <span data-glossary-navigation="true"> ([weight glossary](../../../glossary.md#term-weight))</span>`[1, 0]`, followed by [MATMUL](../../../glossary.md#matrix-multiplication--matmul) and optional exact [rank-one](../../../glossary.md#axis-rank-scalar-vector-and-matrix)
> `[outFeatures]` ADD bias. Complete caller-controlled validation precedes the first intermediate.
> No-bias returns the MATMUL product after two wrapper/ID allocations; biased construction returns
> ADD after three. The biased final Shape is structurally equal to the product Shape and reuses its
> ordered Dimension references, although the outer Shape object may differ. These visible primitive
> producer chains are current inputs to package-private structural capture. This page does not claim
> linear-pattern recognition, canonicalization, fusion, lowering, backend support, or execution.
> The visible floating PERMUTE/MATMUL/ADD chain is differentiable when every selected primitive row
> meets the current closed autograd guards; there is no separate LINEAR rule.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../glossary.md#occurrence-identity-and-canonical-form); [fusion glossary](../../../glossary.md#term-fusion))</span>
> `Tensor.scaledDotProductAttention` and `Tensor.scaledDotProductAttentionWithWeights` are current
> first-class model construction with three ordered query/key/value inputs and an optional fourth
> BOOL mask input. Both locally derive the promoted type and exact broadcast-batch, weights, and
> output Shapes and retain optional exact scale and causal eligibility. The original four methods
> record an exact one-output <span data-glossary-navigation="true"> ([mask glossary](../../../glossary.md#term-boolean-mask))</span>`SCALED_DOT_PRODUCT_ATTENTION` occurrence with output slot zero and no
> hidden weights descriptor or Tensor. The four explicit methods return
> `ScaledDotProductAttentionResult(output, weights)` whose slots zero and one retain one exact
> shared producer, operation, attributes reference, ordered inputs, and output descriptors. The
> kind's sole signature accepts three to four inputs and one to two outputs; producer descriptor
> count identifies the requested occurrence form without adding another kind. Unresolved embedding
> positivity/equality, key/value sequence equality, batch singleton-or-equal, and mask
> singleton-or-equal facts remain obligations for later compiler validation or concrete binding.
> Package-private structural capture preserves this compiler-visible metadata, and package-private
> verification revalidates it and proves or retains its typed Shape constraints. Legal decomposition
> and saved-value lifetime remain planned. The current fail-closed CPU path advertises, lowers,
> prepares, and executes only fully static, resolved-layout one- or two-output
> FLOAT32/FLOAT64 attention occurrences with exact broadcast, mask, output-role,
> positive-embedding, type-promotion, scratch, and injective-output constraints. Dynamic binding and other-
> backend execution remain planned. Current
> package-private first-order autograd supports both public outputs only for the explicit
> two-output occurrence: values slot zero may select query, key, and value, while canonical weights
> slot one may select query and key. It reuses the exact same-occurrence weights wrapper, constructs
> selection-safe products through <span data-glossary-navigation="true"> ([decomposition glossary](../../../glossary.md#term-decomposition); [lifetime glossary](../../../glossary.md#term-lifetime))</span>`where`, and applies the configured or logical symbolic default
> scale. The one-output occurrence fails closed because it has no canonical weights output.
> `Tensor.conv2d(weight, attrs)` and `Tensor.conv2d(weight, bias, attrs)` are current first-class
> grouped NCHW cross-correlation model construction. Each call records one [`CONV2D`](../../../glossary.md#convolution--conv2d) occurrence with
> exact ordered inputs, intrinsic stride/padding/dilation/group attributes, promoted floating
> descriptor, and static or canonical-symbolic output Shape. Input and weight have Shapes<span data-glossary-navigation="true"> ([stride glossary](../../../glossary.md#term-stride); [dilation glossary](../../../glossary.md#term-dilation))</span>
> `[N, C_in, H, W]` and `[C_out, C_in/groups, K_h, K_w]`; optional [bias](../../../glossary.md#term-bias) has `[C_out]`.
> Unresolved channel divisibility, grouped weight/input equality, bias/output equality, and dynamic
> spatial non-negativity remain obligations for future compiler validation or concrete binding.
> This metadata can be structurally captured, and package-private verification now represents and
> proves or retains its descriptor-only constraints. The current fail-closed CPU path advertises,
> lowers, prepares, and executes only the fully static, resolved-layout
> FLOAT32/FLOAT64 grouped NCHW forward subset with exact group, channel, optional-
> bias, spatial, promoted-type, and injective-output relationships. Metal separately admits its
> occurrence-qualified homogeneous four-floating-type domain and documented promotions.
> This includes authenticated Conv1d singleton-height composition. Current first-order autograd
> constructs selected input, weight, and optional bias cotangents through the shared grouped
> Conv2d/Conv3d rule. Matching unfold/fold operations preserve stride, symmetric padding,
> dilation and the original target Shape. Input gradients contract transposed grouped weights
> with the output cotangent and overlap-add the columns; weight gradients contract unfolded
> inputs with that cotangent and reduce batch; bias gradients reduce batch and every spatial axis.
> Explicit <span data-glossary-navigation="true"> ([channel glossary](../../../glossary.md#term-channel); [spatial glossary](../../../glossary.md#term-spatial-axes); [Metal glossary](../../../glossary.md#term-metal); [cotangent glossary](../../../glossary.md#cotangent); [spatial axis glossary](../../../glossary.md#spatial-axes-channels-and-window-sampling))</span>`contiguous()` nodes materialize canonical-only contraction and fold boundaries.
> The rule restores every selected target's exact Shape and floating type. CPU F32/F64 and
> Metal F32 execute their qualified static complete generated topologies, including overlapping
> image folds. This does not broaden ordinary dynamic admission or mixed/low gradient capability;
> see the <span data-glossary-navigation="true"> ([F32 glossary](../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../glossary.md#floating-point-precision-and-rounding); [admission glossary](../../../glossary.md#term-admission))</span>[production support matrix](../../../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary)<span data-glossary-navigation="true"> ([support matrix glossary](../../../glossary.md#term-review-matrix))</span>.
> `Tensor.conv3d(weight, attrs)` and `Tensor.conv3d(weight, bias, attrs)` are current Model
> construction for one first-class grouped NCDHW <span data-glossary-navigation="true"> ([NCDHW glossary](../../../glossary.md#conv3d--ncdhw))</span>[`CONV3D`](../../../glossary.md#conv3d--ncdhw) occurrence. Input, weight, optional bias,
> and result Shapes are `[N, C_in, D, H, W]`,
> `[C_out, C_in/groups, K_d, K_h, K_w]`, `[C_out]`, and
> `[N, C_out, D_out, H_out, W_out]`. The ten-field `Conv3dAttrs` value retains depth/height/width
> stride, symmetric padding, dilation, and groups. Model validates statically decidable relations
> and retains unresolved relations in exact descriptors and attributes. Complete Compiler 0006B
> includes this kind in the closed forward inventory. Independent inference and final validation
> prove or retain its ordered channel and spatial obligations, and ordinary CSE, publication,
> diagnostics, and Planning handoff preserve the exact operation and ordered descriptors. The
> current fail-closed CPU path advertises, lowers, prepares, and executes only the fully static,
> resolved-layout FLOAT32/FLOAT64 grouped NCDHW forward subset with exact group,
> channel, optional-bias, spatial, promoted-type, and injective-output relationships. Metal
> separately admits its occurrence-qualified forward domain. First-order Conv3d uses the same
> group-preserving formula described above, with <span data-glossary-navigation="true"> ([inference glossary](../../../glossary.md#term-inference); [Planning glossary](../../../glossary.md#term-planning); [handoff glossary](../../../glossary.md#term-handoff))</span>`unfold3d`/`fold3d`; no backward-only kind or
> duplicated rank-specific derivative is added. Actual static CPU F32/F64 and Metal F32
> all-role and selected-target execution is qualified in the production matrix. Compiler retains
> symbolic Dimensions, but ordinary public named-B Conv3d admission is not widened. Pool3d now
> supports qualified static CPU FLOAT32/FLOAT64 and Metal FLOAT32 first-order training;
> CPU low arithmetic and Scan higher-order restrictions remain unchanged.<span data-glossary-navigation="true"> ([Pool3d glossary](../../../glossary.md#pool3d--ncdhw-pooling); [Scan glossary](../../../glossary.md#term-cumulative-scan))</span>
> `Tensor.maxPool2d(attrs)` is current first-class [NCHW](../../../glossary.md#term-nchw) maximum-[pooling](../../../glossary.md#term-pooling) [model](../../../glossary.md#term-model) construction. One
> `MAX_POOL2D` occurrence records exact ordered input `[input]`, `MaxPool2dAttrs`, one output at
> index zero, the unchanged floating type and gradient request, exact batch/channel Dimensions, and
> static or canonical-symbolic floor/ceil spatial extents. Dynamic spatial non-negativity remains a
> future compiler-validation or concrete-binding obligation. Literal ceil mode retains every
> ceiling-grid window, even when the terminal window is all-padding; padding exclusion, negative-
> infinity empty windows, NaN propagation, signed-zero ordering, and first-logical-sample ties are
> semantic metadata. Package-private structural capture and descriptor verification are current.
> The CPU backend executes the fully static resolved-layout non-gradient subset through its direct
> schema-71 generated route. Metal executes the fully static FLOAT64/FLOAT32/BFLOAT16/FLOAT16 subset
> through its fixed custom program, preserving exact gradient metadata for later
> ownership decisions. Current package-private first-order autograd reconstructs the exact first
> eligible logical winner from the original input and the same-occurrence public output, including
> padding exclusion and the specified NaN and signed-zero equality, then routes through public
> one-hot and overlap-accumulating fold expressions. It adds no saved-index output. Qualified
> Metal FLOAT32 maximum-pool backward supports overlapping windows; FLOAT64/low overlap remains
> unsupported. Dynamic binding and other algorithms remain separately owned.<span data-glossary-navigation="true"> ([infinity glossary](../../../glossary.md#term-infinity); [signed-zero glossary](../../../glossary.md#nan-infinity-and-signed-zero); [schema glossary](../../../glossary.md#term-schema); [ownership glossary](../../../glossary.md#term-ownership-meanings))</span>
> `Tensor.averagePool2d(attrs)` is current first-class NCHW average-pooling model construction. One
> `AVERAGE_POOL2D` occurrence records exact ordered input `[input]`, `AveragePool2dAttrs`, one output
> at index zero, the unchanged floating type and gradient request, exact batch/channel Dimensions,
> and static or canonical-symbolic floor/ceil spatial extents. Literal ceil mode retains terminal
> all-padding windows. Its semantic metadata fixes a `kernelHeight * kernelWidth` divisor,
> conceptual positive-zero padding that counts in that divisor, FLOAT32 accumulation/division for
> BFLOAT16, FLOAT16, and FLOAT32, FLOAT64 accumulation/division for FLOAT64, one final division, and the
> documented NaN/infinity/signed-zero/all-padding policies. Dynamic spatial non-negativity remains a
> future compiler-validation or concrete-binding obligation. Package-private structural capture and
> descriptor verification are current. The CPU backend executes the fully static resolved-layout
> non-gradient FLOAT32/FLOAT64 subset through its direct schema-71 generated route. Metal executes fully static
> qualified FLOAT32 through its fixed custom program. Current package-private first-order autograd
> divides by a logical typed `kernelHeight * kernelWidth` count and routes every window position
> through public expansion and overlap-accumulating fold expressions. Explicit `contiguous()`
> materializations make the expanded divisor, seed, and columns canonical. The final Pool2d fold
> descriptor becomes canonical in the compiler's static-result logical-layout closure; the formula
> does not append an expression-level final <span data-glossary-navigation="true"> ([seed glossary](../../../glossary.md#term-seed))</span>`contiguous()`. Metal FLOAT32 generated folds support
> overlap; FLOAT64/low folds retain non-overlap placement. Dynamic binding and other algorithms
> remain separately owned.
> `Tensor.maxPool3d(attrs)` and `Tensor.averagePool3d(attrs)` are current Model construction for
> first-class rank-five NCDHW metadata. Each records exact ordered input `[input]`, its family-
> specific thirteen-field attrs reference, one output at index zero, the unchanged floating type
> and gradient request, exact batch/channel Dimension references, and static or canonical-symbolic
> depth/height/width geometry. Literal ceil mode retains terminal all-padding windows. Maximum
> metadata fixes excluded padding, negative-infinity all-padding results, dominant NaN, positive-
> over-negative-zero ordering, and first depth-height-width winner. Average metadata fixes the
> mathematical `kernelDepth * kernelHeight * kernelWidth` count-padding divisor, selected
> accumulator/division domains, one final division, BFLOAT16 narrowing, and its exceptional-value
> policy. Model construction does not evaluate either operation.<span data-glossary-navigation="true"> ([accumulator glossary](../../../glossary.md#accumulator-type); [narrowing glossary](../../../glossary.md#term-narrowing))</span>
>
> Complete Compiler 0006B1 captures both kinds as ordinary flat forward nodes and independently
> validates their descriptors and spatial obligations. Complete 0006B2 adds their first-order
> gradients through the public three-dimensional window algebra. Average Pool3d divides the
> cotangent by a logical typed `kernelDepth * kernelHeight * kernelWidth` Tensor reduction, expands
> every kernel position, and overlap-accumulates with <span data-glossary-navigation="true"> ([kernel glossary](../../../glossary.md#kernel))</span>`fold3d`; padding still counts in the fixed
> divisor but contributes no input cotangent. Maximum Pool3d reconstructs eligibility from typed
> negative-infinity candidates, a separate direct-positive-zero in-bounds mask, and the exact
> same-occurrence output. Its equality preserves dominant NaN, signed-zero ordering, real negative
> infinity, first-winner, and all-padding behavior before one-hot routing and <span data-glossary-navigation="true"> ([negative infinity glossary](../../../glossary.md#nan-infinity-and-signed-zero))</span>`fold3d` accumulation.
> The maximum selector is a fixed non-differentiable mask for a later derivative stage, while the
> incoming branch remains differentiable. CPU raw forward execution retains the fully static
> resolved-layout domain for all six floating types, scalar compute, optional caller-owned complete
> output-cell ranges and zero workspace/materialization. Qualified matching floating gradient
> metadata and lowering reuse ordinary pooling rules and UNFOLD3D/FOLD3D. Low arithmetic uses
> FLOAT32 working values and narrows at the declared operation-result boundaries.
> Metal retains its qualified raw forward carrier domains and now executes qualified FLOAT32
> maximum and average Pool3d gradients, including overlap. Explicit gradient-formula materialization
> produces canonical columns/results; FLOAT64/low folds retain non-overlap restrictions and
> original-carrier validation. Dynamic binding and Metal Scan are not expanded.
> Pool3d constructibility preflight compares the materialized result's semantic type, Shape, and
> gradient eligibility with the requested input descriptor; canonical layout belongs to the explicit
> final <span data-glossary-navigation="true"> ([materialization glossary](../../../glossary.md#materialization); [carrier glossary](../../../glossary.md#term-storage-carrier))</span>`contiguous()` node rather than being required of the original affine input.
> `Tensor.sort(axis[, descending])` and `Tensor.argsort(axis[, descending])` currently construct
> distinct stable, one-input, one-output ordering expressions. Both normalize the axis, preserve the
> exact input Shape reference, leave layout unresolved, and use fixed NaN-last ordering in both
> directions with stable logical-index ties. Sort preserves input type and gradient eligibility;
> argsort uses non-differentiable INT64. These model-expression and provenance facts are structurally
> capturable, and package-private operand/descriptor revalidation is current. Floating SORT gradient
> construction is also current: the compiler constructs one separate stable `ARGSORT` occurrence
> with the exact original input, normalized axis, and direction, then routes the cotangent through
> that permutation. The current fail-closed CPU path advertises, lowers, prepares, and executes only
> fully static, resolved-layout SORT/ARGSORT occurrences over all seven represented input types with
> an exact normalized axis, input-shaped same-type values or INT64 indices, stable NaN-last and
> signed-zero order, and an injective output. The current Metal path separately admits both kinds
> for canonical dense ranks <span data-glossary-navigation="true"> ([normalized axis glossary](../../../glossary.md#normalized-axis); [permutation glossary](../../../glossary.md#permutation))</span>`1..16` whose positive geometry and logical indices
> fit unsigned 32 bits, and always selects its fixed custom program. Gradient support remains limited
> to the floating SORT role above. Dynamic binding and unlisted-backend execution remain planned.
>
> [Preflight](../../../glossary.md#term-preflight) requires the exact SORT input/output [descriptors](../../../glossary.md#term-descriptor), `SortAttrs`, normalized axis,
> direction, and matching one-input/one-output ARGSORT constructibility before any derivative Tensor
> is allocated. SORT still has one output: this is not a hidden index output, public sort result,
> producer reconstruction, or Model API change. The generated ARGSORT is a BACKWARD occurrence in
> the one combined capture.<span data-glossary-navigation="true"> ([API glossary](../../../glossary.md#term-application-programming-interface-api))</span>
> `Tensor.topK(k, axis[, largest, sorted])` currently constructs one compiler-neutral
> `TopKKind.TOP_K` model occurrence with ordered outputs `[values, indices]`. The values output
> preserves input type and gradient eligibility; indices are non-differentiable INT64. Both share
> one fresh Shape whose selected dimension is static `k`, one exact producer, and provenance slots
> zero and one. A known static selected extent is checked during Tensor construction. When that
> extent is dynamic or expression-based, the obligation <span data-glossary-navigation="true"> ([extent glossary](../../../glossary.md#term-extent))</span>`bound extent >= k` is deliberately
> deferred: future compiler or binding validation must reject an insufficient bound rather than
> clamp, pad, wrap, or reduce the output count. Package-private capture preserves the shared TOP_K
> producer and both output positions. The following package-private pass revalidates both descriptors
> and proves or retains the selected-extent obligation without binding an extent or executing work.
> The current fail-closed CPU path advertises, lowers, prepares, and executes only fully static,
> resolved-layout TOP_K occurrences over all seven represented input types with exact normalized
> axis, `0 <= k <= selected extent`, largest/smallest and sorted/unsorted attributes, same-type
> values plus non-differentiable INT64 indices, equal output Shapes, and two injective outputs. The
> current Metal path separately admits positive K for canonical dense ranks
> `1..16` whose positive geometry and logical indices fit unsigned 32 bits. One fixed custom step
> retains both outputs through target subsets, composition, reuse, and publication. Dynamic selected-
> extent binding and unlisted-backend execution remain planned. Current floating TOP_K values-slot
> autograd uses the exact canonical indices wrapper at producer slot one and never recomputes
> selection. Indices remain non-differentiable; stable cutoff membership, NaN membership, direction,
> and sorted-output order are routed without selected-set averaging.
> `Tensor.embedding(indices)` currently validates a rank-two floating weight receiver and exact
> INT32/INT64 indices, then constructs the existing ordinary axis-zero GATHER occurrence directly.
> The result Shape is the complete indices Shape plus the exact weight axis-one Dimension; result
> type and gradient eligibility come only from weights. This is public model-expression construction
> and is structurally capturable; current Compiler autograd treats it as the ordinary GATHER
> occurrence, including repeated-index accumulation. The current fail-closed CPU path advertises,
> lowers, prepares, and executes the fully static, resolved-layout axis-zero GATHER subset with
> INT32/INT64 indices, exact Shape, and injective output. The current Metal path additionally admits
> the exact canonical positive-rank FLOAT32+INT32 form and validates every index
> before selector dispatch or output writes. Constant-index analysis and dynamic binding remain
> planned. No <span data-glossary-navigation="true"> ([dispatch glossary](../../../glossary.md#term-dispatch))</span>`EMBEDDING` kind or [padding](../../../glossary.md#padding)/sparse/max-[norm](../../../glossary.md#term-norm)/frequency option exists.
> `Tensor.oneHot(depth)` currently validates exact INT32/INT64 receiver metadata before positive
> static depth, preserves every input Dimension reference, and appends one fresh
> `StaticDimension(depth)`. It constructs one storage-free, non-[differentiable](../../../glossary.md#derivative-and-chain-rule) BOOL result with one
> `ONE_HOT` producer and exact sole-input provenance. This is a current model-expression inventory
> entry and is structurally capturable. Construction reads no index values; valid execution requires
> `0 <= i < depth`. The current fail-closed CPU path advertises, lowers, prepares, and executes the
> fully static, resolved-layout INT32/INT64-index subset with exact positive depth, BOOL result Shape,
> and injective output. The current Metal path additionally admits the exact canonical positive-rank
> INT32-to-BOOL form, prevalidates every index, and publishes exact zero/one
> bytes without widening BOOL transfer or consumers. Constant analysis and dynamic bounds
> enforcement remain planned; the index input and BOOL result remain non-differentiable.<span data-glossary-navigation="true"> ([widening glossary](../../../glossary.md#term-widening))</span>
> `Tensor.rsqrt`, `log1p`, `expm1`, `gelu`, `geluTanhApproximation`, and `silu` accept floating input,
> retain its exact type, Shape, and gradient eligibility, leave layout unresolved, and record
> one-input parameterless provenance.
> Their selected special-value semantics distinguish signed zero, infinities, and NaN, but current
> construction neither reads values nor promises correct rounding or a fixed numerical tolerance.<span data-glossary-navigation="true"> ([signed zero glossary](../../../glossary.md#term-signed-zero); [correct rounding glossary](../../../glossary.md#term-correct-rounding))</span>
> `Tensor.isFinite`, `isNaN`, and `isInf` accept floating input and construct fixed BOOL results with
> the exact input Shape, unresolved layout, false gradient eligibility, and one-input parameterless
> provenance. They record graph-visible value classifications without eagerly inspecting host
> storage; package-private capture and descriptor verification are current. The current fail-closed
> CPU path advertises, lowers, prepares, and executes only fully static, resolved-layout
> BFLOAT16/FLOAT16/FLOAT32/FLOAT64 classification occurrences with exact Shape preservation,
> non-differentiable BOOL output, non-negative layouts, and membership in a bounded
> one-through-eight-node pointwise unit. Dynamic binding and other-backend execution remain planned.<span data-glossary-navigation="true"> ([host storage glossary](../../../glossary.md#host-storage); [pointwise glossary](../../../glossary.md#term-pointwise))</span>
> `Tensor.sum`, `prod`, reduction `min`, and reduction `max` accept floating or signed-integral
> inputs; [`mean`](../../../glossary.md#term-mean) remains floating-only. They construct full, axis-removing, or retained-axis
> expressions. Full forms have canonical rank-zero Shape and use the canonical no-attributes
> singleton; axis forms normalize the caller axis and store <span data-glossary-navigation="true"> ([rank-zero glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`AxisReductionAttrs`. Every result
> preserves exact input type and gradient eligibility, leaves layout unresolved, and records
> one-input provenance. Integral sum/product use exact-type modular arithmetic modulo <span data-glossary-navigation="true"> ([modular arithmetic glossary](../../../glossary.md#term-modular-arithmetic))</span>`2^32` or
> `2^64` with reassociation permitted; integral min/max use signed order. Their empty identities are
> zero, one, the input type's maximum, and the input type's minimum, respectively, for full domains
> and empty selected-axis slices. Model construction records these semantics without aggregating or
> comparing values. Current compiler capture, verification, and the closed floating first-order
> autograd matrix remain separate from execution. The current fail-closed CPU path advertises,
> lowers, prepares, and executes only the fully static, resolved-layout ordinary SUM/MEAN/PROD/
> MIN/MAX/ALL/ANY subset with its exact family-specific type, attribute, Shape, selected-domain,
> carrier, and injective-output constraints. Gradient support remains limited to the closed matrix
> above. Ordinary named-B FLOAT32 SUM/MEAN are the bounded dynamic exception, not PROD/MIN/MAX/
> ALL/ANY or wider types; general dynamic binding and other-backend execution remain planned.
> Aggregate <span data-glossary-navigation="true"> ([reassociation glossary](../../../glossary.md#term-reassociation))</span>`MIN`/`MAX` remain
> typed separately from the
> equally named two-input binary elementwise kinds.
> The masked `Tensor.sum(axis, mask)` and `Tensor.mean(axis, mask)` forms require floating input and
> an exact BOOL mask. They require ordinary right-aligned broadcasting of the mask to produce
> exactly the input Shape; callers make other axis intent visible with an explicit reshape,
> dimension insertion, or expansion. Each form removes the normalized axis and records one
> first-class two-input <span data-glossary-navigation="true"> ([broadcasting glossary](../../../glossary.md#broadcasting))</span>`SUM` or `MEAN` occurrence with `MaskedReductionAttrs(axis)` and ordered
> `[input, mask]` provenance. False positions exclude their inputs, including NaN and infinity;
> an all-false sum is zero and an all-false mean is NaN without a payload guarantee. Model
> construction does not align storage, inspect values, select elements, count true positions, or
> compute a result. Current compiler capture and floating autograd are described above. The current
> fail-closed CPU path advertises, lowers, prepares, and executes only fully static, resolved-layout
> FLOAT32/FLOAT64 masked SUM/MEAN occurrences with an exact non-gradient BOOL mask, right-aligned broadcast,
> axis-removing result Shape, and injective output. Dynamic binding and other-backend execution
> remain planned.<span data-glossary-navigation="true"> ([payload glossary](../../../glossary.md#artifact-envelope-and-atomic-publication); [elements glossary](../../../glossary.md#term-element))</span>
> `Tensor.all` and `Tensor.any` require exact BOOL input and construct full, axis-removing, or retained-axis
> expressions with exact BOOL result type, false gradient eligibility, unresolved layout, and
> one-input provenance. Aggregate `ALL`/`ANY` remain typed separately from [elementwise](../../../glossary.md#term-elementwise) `AND`/`OR`.
> Model construction does not inspect truth values. The current fail-closed CPU path advertises,
> lowers, prepares, and executes only fully static, resolved-layout ordinary ALL/ANY occurrences
> with exact BOOL input/output, ordinary reduction attributes, Model-derived Shape, canonical
> represented truth values, and an injective output. Dynamic binding and other-backend execution
> remain planned.
> `Tensor.argMin` and `Tensor.argMax` accept floating or integral input and one positive or negative
> axis. Their convenience forms explicitly use `FIRST_INDEX`, while complete forms retain an
> explicit first- or last-index policy in shared `ArgExtremaAttrs`. Axis removal or retention follows
> the ordinary structural Shape rules, but every result is fixed unresolved-layout INT64 with false
> gradient eligibility, one-input provenance, and output index zero. Integral candidates use signed
> order. Floating candidates prefer NaN for both extrema directions, treat multiple NaNs as ties,
> order negative zero below positive zero, and order infinities normally. Model construction rejects
> a statically empty selected axis, accepts an unselected zero axis and unbound selected extent, and
> does not compare values or select an index. Current compiler capture and verification remain
> separate. The current fail-closed CPU path advertises, lowers, prepares, and executes only the
> fully static, resolved-layout numeric ARG_MIN/ARG_MAX subset with a positive selected
> extent, exact keep/remove-Dimension Shape, fixed non-gradient INT64 result, tie policy, and
> injective output. The current Metal path separately admits the canonical dense subset under both
> profiles for ranks `1..16` whose positive geometry and logical indices fit unsigned 32 bits, using
> the same NaN-preferred extrema and explicit first/last tie semantics. A future binding boundary
> must prove or validate that a dynamic selected extent is positive before index selection; dynamic
> binding and unlisted-backend execution remain planned.
> The 26 multi-axis/statistical methods accept ordered distinct positive or negative axes. Caller
> order is retained in immutable `MultiAxisReductionAttrs` or `StatisticalReductionAttrs`; Shape
> derivation uses membership to remove selected axes or retain them with extent one. An empty axis
> list selects a point domain and creates a fresh occurrence rather than requesting full reduction.
> Ordinary type domains remain unchanged. Floating-only `LOG_SUM_EXP`, [`VARIANCE`](../../../glossary.md#term-variance),
> `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM` preserve exact input type/eligibility and remain
> first-class operations rather than stored decompositions. Variance and standard deviation default
> to correction zero or retain explicit non-negative correction, rejecting a statically known
> domain count at most correction while deferring dynamic proof.<span data-glossary-navigation="true"> ([standard deviation glossary](../../../glossary.md#term-standard-deviation))</span>
>
> The model now fixes floating and BOOL empty/special-value reduction semantics, including NaN,
> infinity, signed zero, positive-zero/one or infinite identities, ALL/ANY identities, stable
> log-sum-exp targets, corrected statistical formulas, and non-negative norm targets. These are
> compiler-visible requested meanings that current package-private capture preserves structurally.
> Current package-private verification revalidates operands and proves, rejects, or retains dynamic
> corrected-domain constraints; the closed floating first-order autograd matrix remains current.
> The current fail-closed CPU path advertises, lowers, prepares, and executes only fully static,
> resolved-layout FLOAT32/FLOAT64 LOG_SUM_EXP, VARIANCE, STANDARD_DEVIATION,
> L1_NORM, and L2_NORM occurrences with exact ordered axes, retained/removed Shape, gradient eligibility,
> positive corrected statistical domain, and injective output. Gradient support remains limited to
> the closed matrix above. Dynamic corrected-domain binding and other-backend execution remain
> planned.<span data-glossary-navigation="true"> ([log-sum-exp glossary](../../../glossary.md#log-sum-exp))</span>
> `Tensor.cumSum` and `Tensor.cumProd` are current model expressibility for shape-preserving
> cumulative addition and multiplication. Each family accepts floating or integral input and one
> positive or negative axis. Each short form explicitly selects inclusive forward traversal; each
> complete form retains exact exclusive and reverse flags. Every result retains the exact input
> Shape, data type, and gradient eligibility, leaves layout unresolved, and records the matching<span data-glossary-navigation="true"> ([data type glossary](../../../glossary.md#data-type--datatype))</span>
> `CUM_SUM` or `CUM_PROD` with exact one-input provenance. Integral cumulative product has exact-
> width modular meaning; floating product has the selected NaN, zero-times-infinity, sign-parity,
> and positive-one identity semantics documented by the Tensor API. Construction does not read or
> accumulate values.
>
>
> ---
>
