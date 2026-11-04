# Historical reference [snapshot](../../../glossary.md#term-snapshot): [Model](../../../glossary.md#term-model) construction and semantic boundaries

This is an archived, composite [qualification](../../../glossary.md#term-qualification) narrative preserved from the pre-redesign [API](../../../glossary.md#term-application-programming-interface-api) reference at base `f3f7c2face52c1fa886e075a68faa6d11765cbd0`. Its task-era inventories were accumulated at different [checkpoints](../../../glossary.md#term-checkpoint): this is not a single [release](../../../glossary.md#term-release)-wide capability statement. All quoted “current”, “planned”, “unsupported” and version/count claims below belong to those earlier checkpoints, not today's API.

**Known supersessions:** [CPU](../../../glossary.md#term-cpu) low-[precision](../../../glossary.md#term-precision) parity, fixed-cell first-order [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt), bounded static length-aware cold/[owned Model](../../../glossary.md#model-module-buffer-and-scan-distinctions) workflows and wire versions through v7, shared [normalization](../../../glossary.md#term-normalization) [derivatives](../../../glossary.md#term-derivative), and ordinary [backend](../../../glossary.md#backend) preparation/execution are implemented. The archive does not retract or widen any current qualification. Use [current reference goals](../../../reference/index.md), [Compiler AD](../../../reference/compiler/autograd.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler); [AD glossary](../../../glossary.md#autograd))</span>, [owned Model execution](../../../reference/nn/owned-model.md), and [current backend qualification](../../../backend-guide/production-support.md) instead of the inventories below. Symbolic cold/wire, arbitrary [PURE](../../../glossary.md#term-pure)-body copying, higher-order recurrence and unqualified backend routes are not granted by these supersessions.

## Editorial reading key: two meanings of Model and ownership

The **Model module** discussed here defines Tensor descriptions and operation meanings. It is
not the NN extension's owned `Model`, which manages a network's private state. The application
calls `TensorFactory` to create a leaf (a Tensor without a producing expression), then calls Tensor
methods to describe derived results. A descriptor says what a value looks like; it is not the value's
bytes. A graph value adds graph-local identity, not physical storage.

The overview's flat-array import copies the application's array; its borrowed storage wrapper
instead retains a memory reference without taking responsibility for closing the memory scope.
These are different boundaries, not a rule that every call copies or every Tensor is closeable.
Read [storage association and lifetime](../../../reference/tensor/storage-association.md) for that
distinction and [Compiler capture](../../../reference/compiler/capture.md) for the conversion
from public expressions to graph records. The preserved construction table remains historical.

<a id="model-construction-and-semantic-boundaries"></a>

## Preserved historical text

> **[Model](../../../glossary.md#term-model) construction and semantic boundaries**
>
> The current types describe logical values, construct storage-free arithmetic, comparison, boolean
> logical, conditional-selection, unary, scalar, and value- or index-producing aggregate
> expressions, and provide one bounded host-allocation path without executing a tensor:<span data-glossary-navigation="true"> ([unary glossary](../../../glossary.md#term-unary); [host glossary](../../../glossary.md#term-host))</span>
>
> ```text
> DataType + Shape + optional LayoutDescriptor + requiresGrad = TensorDescriptor
> DataType + physical capacity + exact MemorySegment           = MemorySegmentStorage
> Operation + ordered inputs + ordered output descriptors      = TensorProducer
> TensorProducer + zero-based output index                     = TensorProvenance
> TensorId + exact output descriptor + label + optional provenance + optional host storage = Tensor
> TensorFactory + completed descriptor + optional metadata     = public Tensor construction
> TensorFactory + resolved descriptor                          = exact-span heap storage + Tensor
> TensorFactory + dense descriptor + matching flat array       = copied populated heap Tensor
> TensorFactory + rectangular nested primitive array           = inferred dense descriptor + copied Tensor
> TensorFactory + scalar/shape/type or Tensor template          = independent dense constant Tensor
> TensorFactory + static shape + exact primitive value          = independent dense full-value Tensor
> TensorFactory + rows + columns + DataType                     = independent dense identity matrix
> TensorFactory + integral bounds/step                          = copied dense INT32 or INT64 range Tensor
> TensorRandoms + static shape/type + caller RandomGenerator    = copied dense normal/uniform Tensor
> TensorRandoms + static shape + primitive bounds + source      = copied dense INT32/INT64 random Tensor
> TensorRandoms + static shape + probability + source           = copied dense BOOL random Tensor
> GraphRngState.initial(key, counter)                            = opaque storage-free state expression
> left Tensor + binary kind + right Tensor                       = fresh descriptor + provenance Tensor
> left Tensor + comparison kind + right Tensor                   = fresh BOOL descriptor + provenance Tensor
> left BOOL Tensor + AND/OR + right BOOL Tensor                   = fresh broadcast BOOL expression Tensor
> BOOL Tensor + NOT                                               = fresh shape-preserving BOOL expression Tensor
> BOOL condition + true/false floating branches                  = fresh broadcast selection Tensor
> input Tensor + unary kind                                      = fresh descriptor + provenance Tensor
> floating Tensor + classification kind                         = fresh BOOL descriptor + provenance Tensor
> input Tensor + scalar kind + matching ScalarValue attributes  = fresh descriptor + provenance Tensor
> input Tensor + target DataType                                 = fresh explicit cast Tensor
> numeric Tensor + SUM/PROD/MIN/MAX + full/axis attributes         = fresh reduced-shape Tensor
> floating Tensor + MEAN + full/axis attributes                   = fresh reduced-shape Tensor
> BOOL Tensor + ALL/ANY + full/axis attributes                    = fresh reduced-shape BOOL Tensor
> numeric Tensor + ARG_MIN/ARG_MAX + axis/tie attributes          = fresh reduced-shape INT64 Tensor
> numeric Tensor + CUM_SUM/CUM_PROD + axis/mode attributes        = fresh shape-preserving Tensor
> floating Tensor + SOFTMAX/LOG_SOFTMAX + axis attributes        = fresh shape-preserving Tensor
> Tensor + CONTIGUOUS                                = fresh static-resolved or dynamic-unresolved Tensor
> Tensor + raw/exact target Shape + RESHAPE          = fresh conditional-view reshape Tensor
> Tensor + raw/exact target Shape + EXPAND           = fresh conditional-view expand Tensor
> Tensor + complete output-to-input axes + PERMUTE   = fresh conditional-view permute Tensor
> Tensor + insertion axis + EXPAND_DIMS              = fresh conditional-view rank-expanded Tensor
> Tensor + static singleton axis + SQUEEZE           = fresh conditional-view rank-reduced Tensor
> Tensor + parallel bounds/axes/signed steps + SLICE = fresh conditional-layout sliced Tensor
> Tensor + one axis/bounds/optional step + SLICE     = fresh single-axis sliced Tensor
> Tensor + explicit axes + SLICE                     = fresh one-occurrence flipped Tensor
> Tensor + one axis and scalar coordinate + SELECT  = fresh conditional-view rank-reduced Tensor
> Tensor + arbitrary indices + GATHER               = fresh inserted-Shape gather Tensor
> Tensor + same-rank aligned indices + GATHER_ELEMENTS = fresh exact-indices-Shape Tensor
> INT32/INT64 indices Tensor + positive depth + ONE_HOT = fresh trailing-axis BOOL Tensor
> Tensor + coordinate-tuple indices + GATHER_ND    = fresh indices-prefix-plus-data-suffix Tensor
> Tensor + same-rank indices/updates + SCATTER_ELEMENTS = fresh data-Shape replacement/reduction Tensor
> Tensor + coordinate-tuple indices/updates + SCATTER_ND = fresh data-Shape replacement/reduction Tensor
> Tensor + axis/size/step + UNFOLD_AXIS              = fresh general-axis window Tensor
> window Tensor + axis/output/step + FOLD_AXIS       = fresh overlap-summed target Tensor
> rank-four NCHW Tensor + window/[padding] + UNFOLD2D = fresh canonical column Tensor
> rank-three columns + NCHW Shape/window + FOLD2D    = fresh NCHW fold Tensor
> TensorId / NodeId / ValueId                                  = distinct identity domains
> Operation                                                     = OperationKind + OperationAttrs
> BinaryArithmeticKind                                          = seven parameterless binary arithmetic semantics
> BinaryComparisonKind                                          = six parameterless ordered comparison semantics
> BooleanLogicalKind                                            = three parameterless boolean logical semantics
> WhereSelectionKind                                            = one parameterless ternary conditional-selection semantic
> CastKind + CastAttrs                                          = explicit cast identity + target DataType
> AggregateReductionKind + reduction attributes                = full/single-/multi-axis aggregate + statistics + arg extrema
> SUM/MEAN + MaskedReductionAttrs                              = masked axis semantics + ordinary broadcast mask
> CumulativeScanKind + CumulativeScanAttrs                     = shape-preserving cumulative sum/product semantics
> SoftmaxKind + SoftmaxAttrs                                   = shape-preserving probability normalization semantics
> LayerNormKind + LayerNormAttrs / AffineLayerNormAttrs         = trailing-Shape population normalization semantics
> RmsNormKind + RmsNormAttrs                                   = trailing-Shape uncentered RMS normalization semantics
> BatchNormKind + BatchNormInferenceAttrs                     = explicit five-input per-channel inference semantics
> BatchNormKind + BatchNormTrainingAttrs + BatchNormTrainingResult = pure five-output training/statistic transition
> ContiguousKind                                               = parameterless canonical dense row-major geometry request
> ShapeTransformKind + TargetShapeAttrs                        = reshape/expand meaning + normalized target Shape
> AxisTransformKind + PermutationAttrs / AxisTransformAttrs    = axis reorder/insertion/removal semantics
> SliceKind + SliceAttrs                                      = finite signed-step coordinate-sequence semantics
> PadKind + PadAttrs                                          = constant-padding meaning + normalized widths/typed constant
> TileKind + TileAttrs                                        = complete-pattern per-axis tiling + positive repeat counts
> TensorCompositionKind + CompositionAxisAttrs                = concat/stack meaning + normalized axis
> SelectKind + SelectAttrs                                   = scalar coordinate + normalized source axis/index
> AxisGatherKind + IndexAxisAttrs                            = gather/gather-elements meanings + normalized data axis
> OneHotKind + OneHotAttrs                                 = one-hot meaning + positive static depth
> GatherNdKind + GatherNdAttrs                              = tuple-index meaning + normalized shared batch count
> AxisScatterKind + ScatterElementsAttrs                    = functional scatter-elements meaning + axis/reduction
> ScatterNdKind + ScatterNdAttrs                             = functional tuple-scatter meaning + batch count/reduction
> WindowTransformKind + window attributes                      = unfold/fold meaning + normalized geometry
> UnaryElementwiseKind                                          = nineteen parameterless unary elementwise semantics
> FloatingClassificationKind                                   = three parameterless floating classifications
> ScalarElementwiseKind                                         = eight parameterized one-input scalar semantics
> ScalarValueAttrs / ClampRangeAttrs                            = exact scalar parameters or ordered clamp bounds
> ValueId + TensorDescriptor                                    = GraphValue
> NodeId + Operation + ordered input/output ValueIds            = CompiledNode
> values + nodes + boundaries + node phases                     = CompiledGraphModel
> TensorId + ValueId                                             = ForwardPublicationBinding
> ```
>
> An implemented `TensorDescriptor` keeps the logical element type, shape, explicit layout state,
> and gradient eligibility together as one immutable value. It is still only a description. The
> implemented <span data-glossary-navigation="true"> ([element glossary](../../../glossary.md#term-element); [layout glossary](../../../glossary.md#layout); [gradient glossary](../../../glossary.md#term-gradient))</span>[`Tensor`](../../../glossary.md#tensor) retains that [descriptor](../../../glossary.md#term-descriptor), a stable [`TensorId`](../../../glossary.md#tensorid), and an optional label while
> allowing only its borrowed host-storage association to change. A tensor may also retain immutable
> optional <span data-glossary-navigation="true"> ([borrowed glossary](../../../glossary.md#ownership-borrowing-and-lease))</span>`TensorProvenance`: the backend-independent operation that produced it and an ordered
> immutable snapshot of the exact input-tensor references. Provenance is origin metadata for later
> compiler capture, not graph membership or an intermediate-representation (IR) node. The
> implemented <span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend); [snapshot glossary](../../../glossary.md#term-snapshot); [Provenance glossary](../../../glossary.md#provenance); [metadata glossary](../../../glossary.md#term-metadata); [compiler glossary](../../../glossary.md#term-compiler); [capture glossary](../../../glossary.md#term-capture-meanings); [graph glossary](../../../glossary.md#graph); [IR glossary](../../../glossary.md#term-intermediate-representation-ir); [node glossary](../../../glossary.md#node))</span>`TensorFactory`
> creates tensors from completed descriptors and assigns identity unique among factory allocations
> within the current Java virtual machine (JVM). For a descriptor with resolved layout geometry, its
> allocation overloads also create one matching primitive array whose length is the layout's
> referenced element span and attach the resulting heap-segment storage. Its flat-array overloads
> copy one matching primitive carrier into a resolved dense-contiguous tensor without retaining the
> caller's array. Its nested-array method validates a rank-two-or-greater rectangular primitive-array
> graph, infers the exact carrier type and fully static shape, flattens it in row-major order, and
> delegates destination creation to that flat-import boundary. Its constant methods synthesize
> canonical dense descriptors for rank-zero scalars or fully static caller/template shapes, then
> reuse default-zero allocation or exact-carrier flat import. Its full-value methods fill one exact
> typed carrier from a primitive scalar, and its identity method creates rank-two rectangular
> matrices with typed one on the main diagonal and typed zero elsewhere.
> Its deterministic range methods create non-empty exclusive-end <span data-glossary-navigation="true"> ([JVM glossary](../../../glossary.md#java-platform-and-build-tools))</span><span data-glossary-navigation="true"> ([identity glossary](../../../glossary.md#term-identity-meanings); [Java virtual machine glossary](../../../glossary.md#term-java-virtual-machine-jvm); [referenced element span glossary](../../../glossary.md#referenced-element-span); [heap glossary](../../../glossary.md#term-heap); [carrier glossary](../../../glossary.md#term-storage-carrier); [rank-two glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix); [row-major glossary](../../../glossary.md#term-row-major); [constant glossary](../../../glossary.md#term-constant); [canonical glossary](../../../glossary.md#term-canonical-form); [rank-zero glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix); [matrices glossary](../../../glossary.md#term-matrix-meanings))</span>[`INT32`](../../../glossary.md#data-type--datatype) and [`INT64`](../../../glossary.md#data-type--datatype) tensors,
> synthesize canonical dense descriptors, build one complete temporary carrier, and delegate final
> storage population and identity assignment to flat import. Prefix preparation is test-fixture
> mechanics, not a production TensorFactory capability.
> Public <span data-glossary-navigation="true"> ([fixture glossary](../../../glossary.md#term-fixture))</span>`TensorRandoms` owns normal, continuous-uniform, bounded-integral, and Bernoulli eager leaf
> creation. Each method builds one exact carrier but consumes a transient caller-owned<span data-glossary-navigation="true"> ([Bernoulli glossary](../../../glossary.md#term-bernoulli-distribution); [eager glossary](../../../glossary.md#term-eager-evaluation); [leaf glossary](../../../glossary.md#term-leaf))</span>
> `RandomGenerator` instead of retaining, selecting, or seeding a source. Primitive `int` or `long`
> bounds infer `INT32` or `INT64`, respectively; Bernoulli output is always BOOL; and all integral
> and boolean results disable gradients. This stateless namespace is not a random service,
> lifecycle owner, source abstraction, seed API, distribution hierarchy, or graph RNG contract.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../glossary.md#lifecycle); [owner glossary](../../../glossary.md#term-owner); [seed glossary](../../../glossary.md#term-seed); [API glossary](../../../glossary.md#term-application-programming-interface-api); [distribution glossary](../../../glossary.md#term-distribution); [RNG glossary](../../../glossary.md#graph-rng-state--graphrngstate))</span>
>
> The implemented `HostTensorStorage` boundary describes a raw host-memory region. Its one
> implementation, `MemorySegmentStorage`, borrows an exact JDK memory segment and records physical
> capacity without allocating memory or relating that capacity to a tensor descriptor or layout.<span data-glossary-navigation="true"> ([JDK glossary](../../../glossary.md#term-java-development-kit-jdk); [tensor descriptor glossary](../../../glossary.md#tensor-descriptor))</span>
> `TensorFactory`, rather than the storage wrapper, supplies the implemented allocation policy for
> automatic-scope primitive-array heap segments.<span data-glossary-navigation="true"> ([scope glossary](../../../glossary.md#term-scope))</span>
>
> An [`OperationKind`](../../../glossary.md#operationkind) says which computation is meant, while [`OperationAttrs`](../../../glossary.md#operationattrs) carries its typed
> semantic parameters. The implemented <span data-glossary-navigation="true"> ([parameters glossary](../../../glossary.md#term-parameter))</span>[`Operation`](../../../glossary.md#operation) record keeps those two values together.
> `BinaryArithmeticKind` names seven parameterless tensor-to-tensor arithmetic meanings, and
> `BinaryComparisonKind` names six parameterless ordered tensor-to-tensor comparison meanings.
> `BooleanLogicalKind` names parameterless [boolean](../../../glossary.md#data-type--datatype) conjunction, disjunction, and negation meanings.
> `WhereSelectionKind` names one parameterless elementwise conditional-selection meaning with
> ordered condition, true-branch, and false-branch roles.
> `CastKind` names one parameterized [elementwise](../../../glossary.md#term-elementwise) data-type conversion meaning, and `CastAttrs`
> carries its exact non-null target [`DataType`](../../../glossary.md#data-type--datatype). The public `Tensor.cast` method composes that pair
> with the receiver's source descriptor to create a fresh storage-free expression for every current
> source/target combination, including a same-type request.
> `AggregateReductionKind` names numeric sum, mean, product, minimum, and maximum reductions,
> boolean all and any reductions, floating log-sum-exp, variance, standard deviation, L1/L2 norm,
> and index-producing arg-min and arg-max. <span data-glossary-navigation="true"> ([variance glossary](../../../glossary.md#term-variance); [standard deviation glossary](../../../glossary.md#term-standard-deviation); [L1 glossary](../../../glossary.md#term-l1); [L2 norm glossary](../../../glossary.md#norms-elementary-functions-and-probability-distributions))</span>`AxisReductionAttrs` carries one
> already normalized non-negative axis plus a retained-dimension choice for ordinary single-axis
> forms. Ordinary full forms instead use <span data-glossary-navigation="true"> ([dimension glossary](../../../glossary.md#dimension))</span>`NoOperationAttrs.INSTANCE`, so no negative axis sentinel
> means "all axes." `MultiAxisReductionAttrs` carries a caller-ordered immutable set of distinct
> normalized axes plus retention; `StatisticalReductionAttrs` adds non-negative correction.
> `ArgExtremaAttrs` carries the normalized axis, retained-dimension choice, and an
> explicit <span data-glossary-navigation="true"> ([normalized axis glossary](../../../glossary.md#normalized-axis))</span>`ArgExtremaTiePolicy`. `MaskedReductionAttrs` carries only the normalized reduction axis for
> masked, axis-removing `SUM` and [`MEAN`](../../../glossary.md#term-mean). These reduction semantics are implemented, and public Tensor reduction
> methods currently cover `sum`, `mean`, `prod`, reduction `min`, reduction `max`, boolean `all`,
> boolean `any`, floating [log-sum-exp](../../../glossary.md#log-sum-exp)/statistics/norms, and numeric `argMin`/`argMax`. Ordinary full
> forms produce a canonical rank-zero scalar; single- and multi-axis forms normalize caller axes,
> then remove them or retain them with extent one. Arg extrema
> have no full form, accept floating and integral inputs, and produce fixed INT64 results.
> Masked <span data-glossary-navigation="true"> ([extent glossary](../../../glossary.md#term-extent); [extrema glossary](../../../glossary.md#saturation-pole-and-extremum))</span>`sum(axis, mask)` and `mean(axis, mask)` require ordinary right-aligned broadcasting to
> produce exactly the input Shape, remove the selected axis, and record exact <span data-glossary-navigation="true"> ([broadcasting glossary](../../../glossary.md#broadcasting))</span>`[input, mask]`
> provenance.
> Integral ordinary reduction and arg-extrema ordering/empty-domain semantics are documented below.
> Numerical or truth evaluation, floating ordinary empty-domain behavior, gradients, and execution
> remain planned.<span data-glossary-navigation="true"> ([evaluation glossary](../../../glossary.md#term-evaluation))</span>
> `CumulativeScanKind.CUM_SUM`, `CumulativeScanKind.CUM_PROD`, and `CumulativeScanAttrs` provide
> distinct shape-preserving cumulative-addition and cumulative-multiplication meanings through one
> shared scan family. Public <span data-glossary-navigation="true"> ([scan glossary](../../../glossary.md#term-cumulative-scan))</span>`Tensor.cumSum` and `Tensor.cumProd` each provide an inclusive-forward
> short form and a complete overload retaining caller-selected exclusive and reverse modes. All four
> methods accept floating and integral input, normalize one axis, retain the exact input Shape, data
> type, and gradient eligibility, leave layout unresolved, and record exact one-input provenance.
> They do not inspect or accumulate values.<span data-glossary-navigation="true"> ([data type glossary](../../../glossary.md#data-type--datatype))</span>
> `RecurrentScanKind` provides the separate fixed `RNN_TANH`, `GRU_RESET_AFTER`, and [`LSTM`](../../../glossary.md#term-long-short-term-memory-lstm)
> time-major recurrent meanings. <span data-glossary-navigation="true"> ([time-major glossary](../../../glossary.md#term-time-major))</span>`RecurrentDirection` is each [occurrence](../../../glossary.md#term-occurrence)'s exact `FORWARD` or
> `REVERSE` attribute. The public final field-free `RecurrentScan` namespace has exactly six static
> biased or bias-free <span data-glossary-navigation="true"> ([bias glossary](../../../glossary.md#term-bias))</span>`rnn`, `gru`, and `lstm` methods. Each accepts an explicit time-major input,
> fully static floating descriptors, and one ordinary non-gradient `INT64[batch]` valid-length
> Tensor, then constructs two or three canonical outputs from one flat producer. Model validates
> descriptors but never reads length or data values. Current package-private Compiler forward-only
> inference adopts and independently revalidates the family. Both backward-capable modes remain
> allocation-free fail-closed until backpropagation through time (BPTT) is implemented; Engine
> binding, runtime validation, capability advertisement, and backend execution remain future.<span data-glossary-navigation="true"> ([BPTT glossary](../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span><span data-glossary-navigation="true"> ([producer glossary](../../../glossary.md#term-producer); [fail-closed glossary](../../../glossary.md#term-fail-closed); [backpropagation through time glossary](../../../glossary.md#term-backpropagation-through-time-bptt); [Engine glossary](../../../glossary.md#term-engine); [binding glossary](../../../glossary.md#term-binding); [runtime glossary](../../../glossary.md#term-runtime); [validation glossary](../../../glossary.md#term-validation))</span>
> `SoftmaxKind.SOFTMAX`, `SoftmaxKind.LOG_SOFTMAX`, and `SoftmaxAttrs` provide distinct
> shape-preserving probability and log-probability normalization semantics. The attributes carry
> one already normalized non-negative axis. Public <span data-glossary-navigation="true"> ([normalization glossary](../../../glossary.md#term-normalization); [carry glossary](../../../glossary.md#term-carry))</span>`Tensor.softmax` and `Tensor.logSoftmax`
> accept floating input, normalize a positive or negative caller axis, preserve the exact Shape,
> data type, and gradient-eligibility metadata in an unresolved descriptor, and record exact
> one-input provenance. Construction calculates no probability or logarithm and defines no
> numerical algorithm or gradient rule.
> `LayerNormKind.LAYER_NORM` provides population-variance normalization over an exact non-empty
> trailing Shape. `LayerNormAttrs` selects the exact one-input form, while
> `AffineLayerNormAttrs` selects ordered inputs `[input, scale, bias]`; both attributes retain the
> normalized Shape and exact typed positive epsilon. Public <span data-glossary-navigation="true"> ([normalized Shape glossary](../../../glossary.md#normalized-shape); [epsilon glossary](../../../glossary.md#epsilon))</span>`Tensor.layerNorm` validates static
> trailing extents, defers unresolved equality, derives the result type and metadata, and records
> one output at provenance index zero. It does not calculate mean or variance, expose saved
> statistics, define gradients, or select an execution algorithm.
> `RmsNormKind.RMS_NORM` provides uncentered root-mean-square normalization over an exact non-empty
> trailing Shape. One `RmsNormAttrs` value retains the normalized Shape and exact typed positive
> epsilon for ordered inputs `[input]` or `[input, scale]`, each with exactly one output. Public
> `Tensor.rmsNorm` rejects known static trailing mismatches, defers unresolved equality, requires
> scale Shape exactly equal to the normalized Shape, and retains the exact input result Shape.
> Construction adds no centering, variance, correction, bias, default, broadcast scale, saved
> statistic, numerical evaluation, gradient rule, compiler behavior, or execution behavior.<span data-glossary-navigation="true"> ([saved statistic glossary](../../../glossary.md#saved-statistic))</span>
> `BatchNormKind.BATCH_NORM_INFERENCE` provides stateless per-channel inference with one normalized
> layout-neutral channel axis and exact typed positive epsilon in <span data-glossary-navigation="true"> ([channel glossary](../../../glossary.md#term-channel))</span>`BatchNormInferenceAttrs`.
> Public `Tensor.batchNormInference` consumes ordered
> `[input, scale, bias, runningMean, runningVariance]`, requires each per-channel operand to be an
> exact rank-one <span data-glossary-navigation="true"> ([rank-one glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`[C]` vector, promotes all five floating inputs in that order, preserves the exact
> input Shape, and records one output at provenance index zero. The supplied running variance is
> used directly; construction calculates or updates no statistic, reads no value, creates no saved
> output, and defines no training, gradient, compiler, backend, runtime, or execution behavior.<span data-glossary-navigation="true"> ([vector glossary](../../../glossary.md#term-vector); [statistic glossary](../../../glossary.md#term-statistic))</span>
> `BatchNormKind.BATCH_NORM_TRAINING` instead pairs with
> `BatchNormTrainingAttrs(channelAxis, momentum, epsilon)` and has a fixed five-input/five-output
> signature. Public <span data-glossary-navigation="true"> ([signature glossary](../../../glossary.md#term-signature))</span>`Tensor.batchNormTraining` calculates no values during construction; it records
> the pure transition from explicit old statistics to explicit next statistics, with current-batch
> mean and inverse standard deviation as additional producer-described outputs for later compiler
> capture. <span data-glossary-navigation="true"> ([batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>`BatchNormTrainingResult` exposes only output positions zero through two.
> `ContiguousKind.CONTIGUOUS` provides the parameterless semantic request that one logical input
> retain its values, Shape, DataType, and row-major element order while the result targets canonical
> dense row-major geometry with logical storage offset zero. The kind is distinct from<span data-glossary-navigation="true"> ([storage offset glossary](../../../glossary.md#term-storage-offset))</span>
> `LayoutKind.DENSE_CONTIGUOUS`, which classifies geometry that has already been resolved for a
> static Shape. Public `Tensor.contiguous()` constructs a fresh expression with exact Shape, data
> type, and gradient eligibility. Fully static Shapes receive newly resolved canonical geometry;
> dynamic Shapes remain unresolved. Construction does not inspect input layout or decide aliasing,
> copying, or materialization.<span data-glossary-navigation="true"> ([aliasing glossary](../../../glossary.md#term-aliasing); [materialization glossary](../../../glossary.md#materialization))</span>
> `ShapeTransformKind.RESHAPE` and `ShapeTransformKind.EXPAND` provide distinct one-input
> target-shape meanings. `RESHAPE` preserves the ordered logical element sequence while changing
> the coordinates through which it is interpreted. <span data-glossary-navigation="true"> ([coordinates glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`EXPAND` logically repeats compatible
> singleton dimensions or adds leading dimensions. Both use <span data-glossary-navigation="true"> ([singleton glossary](../../../glossary.md#term-singleton))</span>`TargetShapeAttrs`, whose exact
> non-null [`Shape`](../../../glossary.md#shape) is already normalized model semantics. The attributes therefore contain neither
> raw public request syntax nor a numeric `-1` [inference](../../../glossary.md#term-inference) sentinel. Public `Tensor.reshape` owns
> raw-request normalization, locally provable element-count validation, conditional result-layout
> derivation, and one-input provenance. Public `Tensor.expand` owns directional right-aligned
> compatibility, leading-axis and input-singleton expansion, and conditional zero-stride view
> derivation. The semantic values themselves still do not perform that work. Neither current
> expression family defines gradients, compiler behavior, materialization policy, backend behavior,
> or execution.
> Public <span data-glossary-navigation="true"> ([stride glossary](../../../glossary.md#term-stride); [view glossary](../../../glossary.md#term-view-meanings))</span>`Tensor.permute` owns complete-permutation validation, raw negative-axis normalization,
> exact Dimension-reference reordering, conditional resolved-stride reordering, and one-input
> provenance. <span data-glossary-navigation="true"> ([permutation glossary](../../../glossary.md#permutation))</span>`Tensor.transpose()` supplies normalized [axes](../../../glossary.md#term-axis) `[1, 0]` after requiring rank two. These
> methods construct logical model metadata only; compiler capture and canonicalization, planning and
> prepare-time materialization, backend lowering, storage aliasing, gradients, and execution remain
> outside the current Tensor contract.<span data-glossary-navigation="true"> ([rank glossary](../../../glossary.md#term-rank); [canonicalization glossary](../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../glossary.md#term-planning); [prepare glossary](../../../glossary.md#prepare); [lowering glossary](../../../glossary.md#lowering))</span>
> `SliceKind.SLICE` and `SliceKind.SLICE_UPDATE` define signed extraction and functional replacement.
> At each `SliceAttrs` entry, one normalized inclusive start, selected length, normalized input axis,
> and signed non-zero step identify a finite coordinate sequence. <span data-glossary-navigation="true"> ([finite glossary](../../../glossary.md#term-finite))</span>`CropToShapeAttrs` instead retains
> exact target and prefix Shapes for target-relative extraction or placement. Public `Tensor.slice`
> normalizes directional raw bounds, while `sliceByLength` consumes normalized starts and explicit
> finite lengths and may retain an unresolved selected input extent; both may derive positive-step
> extraction-view metadata. `sliceAxis` and `flip` use the same kind. The array-based
> `sliceUpdate` derives lengths from selected static update Dimensions. The Shape-based overload
> retains the exact update Shape as the target, the exact prefix Shape, and ordered
> `[base, update]`; `cropToShape` retains exact [target](../../../glossary.md#term-target)/prefix references and `[input]`. None reads
> values, mutates inputs, attaches storage, defines gradients, captures a graph, or executes work.
> `PadKind.PAD` pairs with `PadAttrs`, whose [immutable](../../../glossary.md#term-immutable) ordered `before` and `after` lists carry one
> non-negative width per normalized axis position and whose `constantValue` retains one exact typed
> scalar value. <span data-glossary-navigation="true"> ([typed scalar value glossary](../../../glossary.md#typed-scalar-value--scalarvalue))</span>`TileKind.TILE` pairs with `TileAttrs`, whose immutable ordered list
> carries one positive complete-pattern repeat count per normalized axis position. Both attributes
> accept empty scalar-identity lists and `Long.MAX_VALUE` structurally. They contain no input Tensor
> or Shape, so rank matching, checked result-Shape arithmetic, padding-constant conversion, result
> descriptors, layout, provenance, gradients, materialization, compiler behavior, backend behavior,
> ONNX mapping, and execution remain outside these semantic values. Public <span data-glossary-navigation="true"> ([padding glossary](../../../glossary.md#padding); [ONNX glossary](../../../glossary.md#term-onnx))</span>`Tensor.pad` and
> `Tensor.tile` separately perform the input-dependent model validation and metadata construction
> described under [pad and tile expressions](../../../reference/tensor/composition.md#pad-and-tile-expressions).
> `UnaryElementwiseKind` names nineteen parameterless unary arithmetic, transcendental, and
> activation meanings. <span data-glossary-navigation="true"> ([activation glossary](../../../glossary.md#term-activation))</span>`FloatingClassificationKind` separately names three parameterless floating
> classifications with fixed BOOL results. `ScalarElementwiseKind` names eight parameterized
> one-input meanings, with exact typed parameters carried by `ScalarValueAttrs` or
> `ClampRangeAttrs`. The public `Tensor.add`, `sub`, `mul`, `div`, `minimum`,
> `maximum`, and tensor-valued `pow` methods use the binary kinds to construct storage-free expression
> tensors. The public <span data-glossary-navigation="true"> ([binary glossary](../../../glossary.md#term-binary-descriptor-and-metadata))</span>`Tensor.greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`,
> and `notEqualTo` methods use the comparison kinds to construct storage-free [`BOOL`](../../../glossary.md#data-type--datatype) expressions
> from ordered same-category floating or signed-integral inputs. The public `Tensor.abs`, `neg`,
> `reciprocal`, [`log`](../../../glossary.md#term-log), `log1p`, [`exp`](../../../glossary.md#term-expx),
> `expm1`, [`erf`](../../../glossary.md#term-erf), `sqrt`, `rsqrt`, `floor`, `ceil`, `sign`, `relu`, [`sigmoid`](../../../glossary.md#term-sigmoid), [`tanh`](../../../glossary.md#term-tanh), `gelu`,
> `geluTanhApproximation`, and `silu` methods use the unary kinds to create storage-free expressions
> from one floating input. `Tensor.isFinite`,
> `isNaN`, and `isInf` use the classification kinds to create storage-free, non-differentiable BOOL
> expressions from one floating input. The public scalar overloads<span data-glossary-navigation="true"> ([differentiable glossary](../../../glossary.md#derivative-and-chain-rule))</span>
> `Tensor.add(ScalarValue)`, `sub(ScalarValue)`, `mul(ScalarValue)`, `div(ScalarValue)`,
> `minimum(ScalarValue)`, `maximum(ScalarValue)`, and `pow(ScalarValue)` use matching scalar kinds
> and typed attributes to create one-input storage-free expressions with exact receiver/value type
> matching. Their retained `double` overloads are exact-[FLOAT64](../../../glossary.md#term-float64) adapters. First-class
> `clamp(ScalarValue, ScalarValue)` retains one range operation; `clampMin` delegates to [scalar](../../../glossary.md#term-scalar)
> `maximum`, and `clampMax` delegates to scalar `minimum`. The public
> `Tensor.logicalAnd`, `logicalOr`, and `logicalNot` methods use the logical kinds to construct
> storage-free `BOOL` expressions from exact `BOOL` inputs. AND and OR derive a right-aligned
> broadcast shape and preserve receiver/argument order; NOT retains the exact input <span data-glossary-navigation="true"> ([argument glossary](../../../glossary.md#term-argument))</span>`Shape`
> reference without broadcasting. Every logical result has unresolved layout and false gradient
> eligibility. Static `Tensor.where(condition, ifTrue, ifFalse)` requires an exact BOOL condition,
> promotes the two floating branches, broadcasts the branches first and the condition with their
> common shape second, and creates a fresh result whose gradient eligibility is the branch-only OR.
> Its provenance preserves the exact ordered condition, true branch, and false branch references.
> `Tensor.sum`, `prod`, reduction `min`, and reduction `max` construct full or single-axis floating
> or integral aggregate expressions; `mean` remains floating-only. They preserve the input type and gradient-eligibility request,
> derive the reduced shape, leave layout unresolved, and record the receiver as their sole
> provenance input. Product and extrema preserve eligibility metadata without claiming that a
> gradient rule exists.
> `Tensor.all` and `Tensor.any` construct full or single-axis boolean aggregate expressions from exact BOOL
> input. They use the same shape and provenance rules, produce exact BOOL with false gradient
> eligibility, and do not inspect truth values or define empty-domain identities. Aggregate ALL and
> ANY remain typed separately from elementwise AND and OR.
> `Tensor.argMin` and `Tensor.argMax` construct single-axis index-producing expressions from floating
> or integral input. Their convenience forms explicitly use <span data-glossary-navigation="true"> ([axis index glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`FIRST_INDEX`, while complete forms
> retain an explicit first- or last-index policy. Every result is exact INT64 with false gradient
> eligibility. The shared contract fixes floating/integral ordering and rejects a statically empty
> selected axis; construction does not compare values or select an index.
> `Tensor.cumSum` and `Tensor.cumProd` construct single-axis cumulative-addition and cumulative-
> multiplication expressions from floating or integral input. Each short form is inclusive and
> forward; each complete form retains explicit exclusive and reverse choices. Every result preserves
> the exact input Shape, type, and gradient eligibility, leaves layout unresolved, and records
> exactly the receiver as provenance without accumulating values.
> `Tensor.softmax` and `Tensor.logSoftmax` construct single-axis normalization expressions from
> floating input. Both preserve the exact input Shape, type, and gradient eligibility, leave layout
> unresolved, and record exactly the receiver as provenance. They retain distinct first-class
> SOFTMAX or LOG_SOFTMAX semantics without calculating values or selecting a decomposition.<span data-glossary-navigation="true"> ([SOFTMAX glossary](../../../glossary.md#softmax--log-softmax); [decomposition glossary](../../../glossary.md#term-decomposition))</span>
> `Tensor.layerNorm` constructs exact trailing-Shape population-normalization expressions from
> floating input. Its no-affine form retains input type and records `[input]`; its affine form
> promotes floating `[input, scale, bias]` in occurrence order. Both preserve the exact input Shape,
> leave layout unresolved, and produce one fresh storage-free output at provenance index zero. They
> record no saved statistics and perform no numerical evaluation.
> `Tensor.rmsNorm` constructs exact trailing-Shape uncentered RMS-normalization expressions from
> floating input. Its no-scale form retains input type and records `[input]`; its scaled form
> promotes floating `[input, scale]` in occurrence order and requires scale Shape exactly equal to
> the normalized Shape. Both preserve the exact input Shape, leave layout unresolved, and produce
> one fresh storage-free output at provenance index zero. They add no centering, correction, bias,
> default, broadcast scale, saved statistic, numerical evaluation, compiler behavior, or execution
> behavior.
> `Tensor.batchNormInference` constructs one stateless five-input inference expression. It
> normalizes an explicit channel axis against rank-at-least-two input, requires exact rank-one
> scale, bias, running-mean, and running-variance vectors, and records those inputs in that order
> after the receiver. It preserves the exact input Shape, uses ordered floating promotion, leaves
> layout unresolved, combines all five gradient-eligibility flags, and produces one fresh
> storage-free output at provenance index zero. Construction does not read values, calculate or
> update statistics, add saved outputs, select an algorithm, capture a graph, or execute work.
> `Tensor.batchNormTraining` constructs one pure five-input/five-output training occurrence. It
> returns normalized output and explicit next running mean and variance in
> `BatchNormTrainingResult`; saved batch mean and inverse standard deviation remain described at
> producer positions three and four. Construction retains no statistic between calls and does not
> capture, differentiate, publish, lower, or execute the occurrence.
> `Tensor.contiguous()` accepts every current data type and preserves the exact input Shape, type, and
> gradient eligibility. A fully static result receives a newly constructed canonical dense
> row-major layout; a dynamic result remains unresolved. Each call records [`CONTIGUOUS`](../../../glossary.md#term-contiguous) with
> `NoOperationAttrs.INSTANCE` and exactly the receiver as provenance, without reading input layout,
> storage, or values or performing a copy.
> `TensorProvenance` retains the exact operation and ordered public-tensor inputs. All current
> expression families derive result descriptors, but none assigns graph identity, captures
> a graph, calculates values, defines gradient rules, or executes computation.
>
> Comparison expression construction validates same-category numeric compatibility and local
> broadcasting, then derives an unresolved non-differentiable `BOOL` descriptor and exact ordered
> provenance. It does not inspect values or execute the comparison. Its fixed represented-value
> semantics use ordinary floating numeric order, make every ordered relation false when either
> operand is NaN, and treat negative and positive zero as equal. <span data-glossary-navigation="true"> ([NaN glossary](../../../glossary.md#term-nan); [positive zero glossary](../../../glossary.md#nan-infinity-and-signed-zero))</span>`EQUAL` is exact numeric equality
> rather than raw-bit or tolerance equality; <span data-glossary-navigation="true"> ([tolerance glossary](../../../glossary.md#term-tolerance))</span>`NOT_EQUAL` is its logical complement. Integral
> relations retain exact signed comparison after promotion. Compiler capture, gradients, backend
> lowering, and execution remain planned for their owning layers.
>
> An implemented `GraphValue` describes logical data. An implemented `CompiledNode` describes one
> place where operation semantics consume and produce that data. The implemented
> [`CompiledGraphModel`](../../../glossary.md#compiled-graph--compiledgraphmodel) contains those elements, declares ordered graph boundaries, classifies every
> node with a `GraphPhase`, and validates structural closure. It is compile-time model state, not a
> compiler, compile artifact, prepared program, or runtime schedule. The standalone implemented<span data-glossary-navigation="true"> ([compile glossary](../../../glossary.md#compile); [artifact glossary](../../../glossary.md#term-artifact))</span>
> `ForwardPublicationBinding` associates public tensor identity with graph-local value identity for
> a compiler-owned forward-publication plan.<span data-glossary-navigation="true"> ([publication plan glossary](../../../glossary.md#publication-plan))</span>
>
>
> ---
>
