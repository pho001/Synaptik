# Historical reference [snapshot](../../../glossary.md#term-snapshot): [Tensor](../../../glossary.md#tensor) [qualification](../../../glossary.md#term-qualification) context

This is an archived, composite qualification narrative preserved from the pre-redesign [API](../../../glossary.md#term-application-programming-interface-api) reference at base `f3f7c2face52c1fa886e075a68faa6d11765cbd0`. Its task-era inventories were accumulated at different [checkpoints](../../../glossary.md#term-checkpoint): this is not a single [release](../../../glossary.md#term-release)-wide capability statement. All quoted “current”, “planned”, “unsupported” and version/count claims below belong to those earlier checkpoints, not today's API.

**Known supersessions:** [CPU](../../../glossary.md#term-cpu) low-[precision](../../../glossary.md#term-precision) parity, fixed-cell first-order [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt), bounded static length-aware cold/[owned Model](../../../glossary.md#model-module-buffer-and-scan-distinctions) workflows and wire versions through v7, shared [normalization](../../../glossary.md#term-normalization) [derivatives](../../../glossary.md#term-derivative), and ordinary [backend](../../../glossary.md#backend) preparation/execution are implemented. The archive does not retract or widen any current qualification. Use [current reference goals](../../../reference/index.md), [Compiler AD](../../../reference/compiler/autograd.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler); [AD glossary](../../../glossary.md#autograd))</span>, [owned Model execution](../../../reference/nn/owned-model.md), and [current backend qualification](../../../backend-guide/production-support.md) instead of the inventories below. Symbolic cold/wire, arbitrary [PURE](../../../glossary.md#term-pure)-body copying, higher-order recurrence and unqualified backend routes are not granted by these supersessions.

<a id="tensor-qualification-context"></a>
<a id="purpose-and-mental-model"></a>

## Preserved historical text

> **[Tensor](../../../glossary.md#tensor) [qualification](../../../glossary.md#term-qualification) context**
>
> **Purpose and [mental model](../../../glossary.md#term-mental-model)**
>
> This reference documents the public [model](../../../glossary.md#term-model) contracts that are implemented today. The [mutable](../../../glossary.md#term-mutable)
> `Tensor` now connects stable logical metadata to an optional borrowed host-storage association,
> and <span data-glossary-navigation="true"> ([metadata glossary](../../../glossary.md#term-metadata); [borrowed glossary](../../../glossary.md#ownership-borrowing-and-lease))</span>`TensorFactory` provides the public construction boundary for completed descriptors, copied
> primitive-array import, independent dense constants including full-value and rectangular identity
> tensors, and integer ranges. <span data-glossary-navigation="true"> ([descriptors glossary](../../../glossary.md#term-descriptor); [constants glossary](../../../glossary.md#term-constant); [identity glossary](../../../glossary.md#term-identity-meanings))</span>`TensorRandoms` is the sole public owner of eager normal,
> continuous-uniform, bounded-integral, and Bernoulli initialization from an explicit caller-owned
> source. <span data-glossary-navigation="true"> ([owner glossary](../../../glossary.md#term-owner); [eager glossary](../../../glossary.md#term-eager-evaluation); [Bernoulli glossary](../../../glossary.md#term-bernoulli-distribution))</span>[`GraphRngState`](../../../glossary.md#graph-rng-state--graphrngstate) separately represents an explicit, opaque random-number-generator (RNG)
> state occurrence for later graph execution; it neither exposes a numerical Tensor nor performs a
> random draw. <span data-glossary-navigation="true"> ([opaque glossary](../../../glossary.md#term-opaque); [RNG glossary](../../../glossary.md#graph-rng-state--graphrngstate); [graph glossary](../../../glossary.md#graph))</span>`Tensor.dropout(probability, state)` constructs one training-only inverted-dropout
> occurrence and returns its public output together with the explicitly threaded next state. The
> current concrete expression surface contains seven tensor-to-tensor binary arithmetic
> methods, of which ADD, SUB, MUL, MIN, and MAX also accept signed integral operands; six floating
> or signed-integral tensor-to-tensor comparison methods; three BOOL-only logical
> methods, nineteen floating unary elementwise methods, three floating-classification methods, seven
> exact-typed plus seven exact-FLOAT64 scalar arithmetic methods, and six range/one-bound clamp
> methods, plus one static conditional-selection method and one explicit cast method. Fifteen
> numeric aggregate methods add full, one-axis, and retained-axis <span data-glossary-navigation="true"> ([dropout glossary](../../../glossary.md#dropout--inverted-dropout); [binary glossary](../../../glossary.md#term-binary-descriptor-and-metadata); [BOOL glossary](../../../glossary.md#data-type--datatype); [unary glossary](../../../glossary.md#term-unary); [elementwise glossary](../../../glossary.md#term-elementwise); [scalar glossary](../../../glossary.md#term-scalar))</span>`sum`, [`mean`](../../../glossary.md#term-mean), `prod`, reduction
> `min`, and reduction `max` expression construction: mean remains floating-only, while the other
> four families also accept signed-integral input. Six BOOL methods provide full/single-axis
> `all`/`any`; ten additional numeric and four BOOL methods add ordered distinct multi-axis forms.
> Twelve floating methods add first-class log-sum-exp, corrected
> variance/standard deviation, and L1/L2 norms. Six axis-only <span data-glossary-navigation="true"> ([log-sum-exp glossary](../../../glossary.md#log-sum-exp); [standard deviation glossary](../../../glossary.md#term-standard-deviation); [L1 glossary](../../../glossary.md#term-l1); [L2 glossary](../../../glossary.md#term-l2))</span>`argMin`/`argMax` methods add fixed
> INT64 index results. Two one-axis <span data-glossary-navigation="true"> ([INT64 glossary](../../../glossary.md#data-type--datatype))</span>`cumSum` methods and two matching `cumProd` methods add
> shape-preserving numeric scan expressions, and one-axis <span data-glossary-navigation="true"> ([scan glossary](../../../glossary.md#term-cumulative-scan))</span>[`softmax`](../../../glossary.md#softmax--log-softmax) and `logSoftmax` add shape-
> preserving floating probability-normalization expressions. Two <span data-glossary-navigation="true"> ([normalization glossary](../../../glossary.md#term-normalization))</span>`layerNorm` methods add population-variance
> normalization over an exact trailing Shape, either without affine operands or with explicit scale
> and bias. Two <span data-glossary-navigation="true"> ([bias glossary](../../../glossary.md#term-bias))</span>`rmsNorm` methods separately add uncentered root-mean-square normalization over an
> exact trailing Shape, either without scale or with one explicit exact-Shape scale. One
> `batchNormInference` method adds stateless per-channel inference from mandatory explicit scale,
> bias, running-mean, and running-variance vectors along a layout-neutral channel axis. One<span data-glossary-navigation="true"> ([channel glossary](../../../glossary.md#term-channel); [inference glossary](../../../glossary.md#term-inference); [vectors glossary](../../../glossary.md#term-vector); [layout glossary](../../../glossary.md#layout))</span>
> `batchNormTraining` method uses the same five Tensor roles plus exact typed momentum and epsilon
> to describe current-batch normalization and an explicit running-statistic transition. It returns
> the normalized output and next running statistics while its shared producer also describes two
> compiler-facing saved outputs. The
> parameterless <span data-glossary-navigation="true"> ([momentum glossary](../../../glossary.md#term-momentum); [epsilon glossary](../../../glossary.md#epsilon); [batch normalization glossary](../../../glossary.md#batch-normalization--batch-normalization-inference-and-training); [statistic glossary](../../../glossary.md#term-statistic); [producer glossary](../../../glossary.md#term-producer); [compiler glossary](../../../glossary.md#term-compiler))</span>[`contiguous`](../../../glossary.md#term-contiguous) method adds a shape-preserving request
> for canonical dense row-major result geometry. Two <span data-glossary-navigation="true"> ([canonical glossary](../../../glossary.md#term-canonical-form); [row-major glossary](../../../glossary.md#term-row-major))</span>`reshape` overloads add ordered-element-
> preserving coordinate changes from either raw <span data-glossary-navigation="true"> ([element glossary](../../../glossary.md#term-element); [coordinate glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`long...` [dimensions](../../../glossary.md#dimension) or an exact normalized
> [`Shape`](../../../glossary.md#shape). Two `expand` overloads add directional right-aligned singleton and leading-axis
> repetition with locally derived zero-stride view geometry when possible. <span data-glossary-navigation="true"> ([stride glossary](../../../glossary.md#term-stride); [view glossary](../../../glossary.md#term-view-meanings))</span>`permute(int...)` adds
> arbitrary complete axis reordering, and `transpose()` adds its [rank-two](../../../glossary.md#axis-rank-scalar-vector-and-matrix) `[1, 0]` convenience.
> Two [`linear`](../../../glossary.md#term-linear-meanings) overloads compose that transpose with MATMUL and optional exact rank-one ADD bias.
> Two <span data-glossary-navigation="true"> ([transpose glossary](../../../glossary.md#term-transpose); [MATMUL glossary](../../../glossary.md#matrix-multiplication--matmul); [rank-one glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`conv1d` overloads compose input/weight singleton-height expansion, ordinary grouped Conv2d,
> and height squeeze, with optional output-channel bias and no new operation kind.
> Two <span data-glossary-navigation="true"> ([weight glossary](../../../glossary.md#term-weight); [Conv2d glossary](../../../glossary.md#convolution--conv2d); [operation glossary](../../../glossary.md#operation))</span>[`conv3d`](../../../glossary.md#conv3d--ncdhw) overloads construct one first-class grouped cross-correlation occurrence in NCDHW
> axis order, with optional output-channel bias and explicit rank-three-spatial geometry.
> One <span data-glossary-navigation="true"> ([NCDHW glossary](../../../glossary.md#conv3d--ncdhw); [rank glossary](../../../glossary.md#term-rank); [spatial glossary](../../../glossary.md#term-spatial-axes))</span>`maxPool3d` and one `averagePool3d` receiver construct first-class NCDHW pooling metadata with
> rank-specific depth, height, and width geometry. They do not evaluate or materialize a result.<span data-glossary-navigation="true"> ([NCDHW pooling glossary](../../../glossary.md#pool3d--ncdhw-pooling))</span>
> `expandDims(int)` inserts one [singleton](../../../glossary.md#term-singleton) [axis](../../../glossary.md#term-axis), while `squeeze(int)` removes one selected statically
> known singleton axis. `slice(long[], long[], int[], long[])` adds general signed-step directional
> half-open selection. <span data-glossary-navigation="true"> ([half-open glossary](../../../glossary.md#term-half-open-interval))</span>`sliceAxis(int, long, long)` supplies its one-axis step-one convenience,
> `sliceAxis(int, long, long, long)` supplies an explicit signed step, and `flip(int...)` reverses
> explicit axes through one slice occurrence.<span data-glossary-navigation="true"> ([slice glossary](../../../glossary.md#slice))</span>
> `select(int, long)` fixes one scalar coordinate and removes its source axis, with locally derived
> logical view geometry when the input layout is resolved and the result is non-empty.
> `gather(Tensor, int)` replaces one data axis with the complete indices Shape.
> `gatherElements(Tensor, int)` retains the exact same-rank indices Shape after checking every
> non-selected Dimension. These are the complete public axis-index gather spellings.<span data-glossary-navigation="true"> ([gather glossary](../../../glossary.md#gather))</span>
> `oneHot(long)` treats an INT32 or INT64 receiver as logical indices and appends one positive
> static class axis. It produces storage-free BOOL metadata for one first-class <span data-glossary-navigation="true"> ([INT32 glossary](../../../glossary.md#data-type--datatype); [class axis glossary](../../../glossary.md#class-axis))</span>`ONE_HOT`
> occurrence without reading any index value.
> `gatherNd(Tensor)` and `gatherNd(Tensor, int)` use coordinate tuples to index several data axes
> after zero or more structurally equal shared batch Dimensions. They require a static positive
> tuple depth, derive the result from the indices prefix plus the untouched data suffix, and never
> inspect index values.
> The two <span data-glossary-navigation="true"> ([batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [tuple depth glossary](../../../glossary.md#term-tuple-depth))</span>`scatterElements` overloads construct functional same-rank axis-scatter expressions from
> ordered `[data, indices, updates]` inputs. They preserve the exact data Shape and type, combine
> data/update gradient eligibility, and never inspect values or mutate the data input.<span data-glossary-navigation="true"> ([gradient glossary](../../../glossary.md#term-gradient))</span>
> `scatterAdd(Tensor, Tensor, int)` constructs the distinct Gather-compatible fixed-add scatter:
> the complete indices Shape replaces one data axis in the required updates Shape, while the fresh
> functional result retains the exact data Shape and accumulates duplicate targets.
> The three <span data-glossary-navigation="true"> ([targets glossary](../../../glossary.md#term-target))</span>`scatterNd` overloads construct functional tuple-index scatter expressions with
> replacement or an explicit reduction and with zero or an explicit shared-batch count. They apply
> the Gather-ND result-Shape formula to <span data-glossary-navigation="true"> ([Gather-ND glossary](../../../glossary.md#gather-nd))</span>`updates`, preserve exact data metadata in a fresh unresolved
> result, and never inspect values or mutate an input.
> `pad(long[], long[], ScalarValue)` adds exactly typed constant-filled positions around every axis;
> its retained `double` overload is an exact-[FLOAT64](../../../glossary.md#term-float64) convenience. `tile(long...)` repeats each
> complete input pattern along every axis. Static `concat` joins an
> ordered non-empty input sequence along an existing axis, static `stack` inserts a count axis for
> same-shaped inputs, and instance `unstack` returns an immutable ordered list of independent scalar
> select expressions. <span data-glossary-navigation="true"> ([immutable glossary](../../../glossary.md#term-immutable); [scalar select glossary](../../../glossary.md#scalar-select))</span>`unfold` adds general-axis window [materialization](../../../glossary.md#materialization) and `foldAxis` restores its
> overlap-summing target transformation. The two `unfold2d` forms add NCHW im2col with conceptual
> positive-zero or exact typed padding, while <span data-glossary-navigation="true"> ([NCHW glossary](../../../glossary.md#term-nchw); [im2col glossary](../../../glossary.md#term-im2col); [padding glossary](../../../glossary.md#padding))</span>`fold2d` adds overlap-summing col2im with exact static
> or symbolic compatibility.<span data-glossary-navigation="true"> ([col2im glossary](../../../glossary.md#term-col2im))</span>
> `HostTensorStorage` exposes current raw [host](../../../glossary.md#term-host)-storage facts, but `Tensor` has no typed element
> getters or setters. The public Engine now owns the separate compile, prepare, and run
> lifecycle for Tensor expressions. Public compilation can request forward and supported
> first-order gradient publication occurrences. Engine.run binds exact input identities in any order;
> Engine.runValues binds fresh fully static exact-descriptor values in compiled order without
> mutating Tensor prototypes. Both use the same prepared owner and an open<span data-glossary-navigation="true"> ([Engine glossary](../../../glossary.md#term-engine); [compile glossary](../../../glossary.md#compile); [prepare glossary](../../../glossary.md#prepare); [run glossary](../../../glossary.md#term-run); [lifecycle glossary](../../../glossary.md#lifecycle); [compilation glossary](../../../glossary.md#compiler-engine-and-execution-stage-names); [publication glossary](../../../glossary.md#term-publication); [Engine.runValues glossary](../../../glossary.md#fresh-value-binding--enginerunvalues); [prototypes glossary](../../../glossary.md#term-prototype))</span>
> [`RunResult`](../../../glossary.md#run-result--runresult) can explicitly materialize one [occurrence](../../../glossary.md#term-occurrence) as a [detached](../../../glossary.md#detached-model-results) host value.
> `Engine.standard()` uses one fixed CPU-only composition; preparation and execution remain bounded
> by exact CPU capability and fail closed for
> unsupported operation, data-type, Shape, or layout combinations. This current lifecycle does not
> move compiler, prepared-execution, runtime-residency, or backend state into Tensor. Tensor-owned
> device residency, generic or mixed-backend composition, and universal execution coverage remain
> absent.
> The current Model contract defines one numerical meaning per family and dtype, with no graph-wide
> selector. Kinds, attributes, mapping, contributors, masks, indices, state, traversal, casts,
> ordering, guards, identities, divisors, stored comparisons and extrema, raw movement and
> selection, and publication remain exact. Denormals-are-zero (DAZ) at named arithmetic inputs and
> flush-to-zero (FTZ) at named arithmetic results are permitted for FLOAT32 and all four low types,
> including low multiplication, but forbidden for FLOAT64. They do not apply to stored comparison,
> extrema, cast, raw movement, selection, or the exact unary ABS/NEG/SIGN/FLOOR/CEIL/RELU sites.
> Every floating scatter MUL includes the base and each addressed update exactly once but may use a
> rounded tree; integral modular multiplication and raw unaddressed cells remain exact.
> Qualified low arithmetic uses FLOAT32 working values and accumulators followed by one final
> ties-to-even narrowing per declared low output, without hidden intermediate low narrowing.
> Finite accuracy thresholds qualify backend routes only in tests; they do not define a public
> envelope or authorize compiler rewrites. Composite and generated-gradient formulas inherit only
> their named primitive-site permissions. Tensor construction performs no numerical evaluation.<span data-glossary-navigation="true"> ([CPU glossary](../../../glossary.md#term-cpu); [fail closed glossary](../../../glossary.md#admission-preflight-and-fail-closed-validation); [runtime glossary](../../../glossary.md#term-runtime); [residency glossary](../../../glossary.md#residency); [backend glossary](../../../glossary.md#backend); [device glossary](../../../glossary.md#term-device); [backend composition glossary](../../../glossary.md#backend-composition); [dtype glossary](../../../glossary.md#data-type--datatype); [masks glossary](../../../glossary.md#term-boolean-mask); [extrema glossary](../../../glossary.md#saturation-pole-and-extremum); [Denormals-are-zero glossary](../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow); [DAZ glossary](../../../glossary.md#term-daz); [flush-to-zero glossary](../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow); [FTZ glossary](../../../glossary.md#term-ftz); [FLOAT32 glossary](../../../glossary.md#term-float32); [ties-to-even glossary](../../../glossary.md#term-ties-to-even); [narrowing glossary](../../../glossary.md#term-narrowing); [Finite glossary](../../../glossary.md#term-finite); [evaluation glossary](../../../glossary.md#term-evaluation))</span>
>
> Metal's exact domain contains the exact unary,
> affine, canonicalization, indexing, BOOL-domain, Task-0059 movement, Task-0060
> replacement/fold/aggregate, Task-0063 ordering/top-K/numeric arg-extrema, no-gradient promoted
> INT32/INT64 MATMUL, Task-0064 maximum pooling, and Task-0065 raw INITIAL_STATE rows. Metal
> also admits the documented FLOAT32 arithmetic, extrema, scalar, reduction, and scan rows;
> every positive-static FLOAT32 MATMUL vector, matrix, batched, and broadcast geometry; no-gradient
> BFLOAT16/FLOAT32 mixed MATMUL with FLOAT32 output; Task-0064 convolution/average pooling; Task-0065
> FLOAT32 dropout; and three Task-0069 source-owned custom programs. Task 0069 admits no-gradient,
> canonical rank-one FLOAT32 <span data-glossary-navigation="true"> ([Metal glossary](../../../glossary.md#term-metal); [canonicalization glossary](../../../glossary.md#occurrence-identity-and-canonical-form); [pooling glossary](../../../glossary.md#term-pooling); [matrix glossary](../../../glossary.md#term-matrix-meanings); [BFLOAT16 glossary](../../../glossary.md#term-bfloat16); [convolution glossary](../../../glossary.md#term-channels-first-convolution-layers))</span>`L1_NORM` with one positive extent; no-gradient, axis-zero, canonical
> rank-one FLOAT32 <span data-glossary-navigation="true"> ([extent glossary](../../../glossary.md#term-extent))</span>`SCATTER_ADD` with positive data and update extents, a materialized canonical
> rank-one INT32 or INT64 index feed, and identical base/result and index/update extents; and
> no-gradient FLOAT32 singleton [`VARIANCE`](../../../glossary.md#term-variance) over input `[1]`, axis `[0]`, correction zero, with
> canonical scalar or retained `[1]` output. Compiler-generated rank-one Gather data cotangents close
> through an explicit canonical zero `EXPAND -> CONTIGUOUS -> SCATTER_ADD`; gradient-bearing
> Scatter-Add and its own cotangents remain unsupported. The resulting Metal capability ledger is
> exactly 88 admitted kinds and 27 remaining false, with 103 structurally executable kinds and 12
> remaining nonexecutable. The later bounded FLOAT32 and homogeneous BFLOAT16/FLOAT16 <span data-glossary-navigation="true"> ([capability ledger glossary](../../../glossary.md#capability-ledger); [homogeneous glossary](../../../glossary.md#term-homogeneous); [FLOAT16 glossary](../../../glossary.md#term-float16))</span>[`EXP`](../../../glossary.md#term-expx) routes
> and sign-guarded composed FLOAT32 [`SIGMOID`](../../../glossary.md#term-sigmoid) route do not make every dtype or occurrence of those
> kinds available. Homogeneous BFLOAT16 and FLOAT16 `SIGMOID` use independently qualified custom
> steps. These counts describe kinds with
> admitted occurrences, not whole-kind admission; every unlisted occurrence fails closed before
> route selection.<span data-glossary-navigation="true"> ([admission glossary](../../../glossary.md#term-admission))</span>
>
> Metal uses ABI 7 with thirteen exports and one bounded schema-20 route-bearing program image.
> Operation wires <span data-glossary-navigation="true"> ([ABI glossary](../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../glossary.md#term-schema))</span>`1..115`, attribute wires `0..41`, and type wires `1..7` cover current structural
> vocabulary. The custom-program image authenticates compact materialized slots, deterministic
> pointwise units, and qualified MATMUL/Conv2d anchor epilogues. Metal-local workload,
> exact-policy, candidate, compatibility, route-policy, and session codec identities are version 40;
> every other identity fails closed rather than falling back. Model remains the sole semantic owner
> of numerical meaning.<span data-glossary-navigation="true"> ([pointwise glossary](../../../glossary.md#term-pointwise); [codec glossary](../../../glossary.md#term-codec))</span>
>
> The authoritative [module](../../../glossary.md#term-module) boundary remains [`ARCHITECTURE.md`](../../../../ARCHITECTURE.md).
>
>
> ---
>
> [Next topic](compiler-qualification.md)
