# Historical reference [snapshot](../../../glossary.md#term-snapshot): Historical implementation-status notes

This is an archived, composite [qualification](../../../glossary.md#term-qualification) narrative preserved from the pre-redesign [API](../../../glossary.md#term-application-programming-interface-api) reference at base `f3f7c2face52c1fa886e075a68faa6d11765cbd0`. Its task-era inventories were accumulated at different [checkpoints](../../../glossary.md#term-checkpoint): this is not a single [release](../../../glossary.md#term-release)-wide capability statement. All quoted “current”, “planned”, “unsupported” and version/count claims below belong to those earlier checkpoints, not today's API.

**Known supersessions:** [CPU](../../../glossary.md#term-cpu) low-[precision](../../../glossary.md#term-precision) parity, fixed-cell first-order [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt), bounded static length-aware cold/[owned Model](../../../glossary.md#model-module-buffer-and-scan-distinctions) workflows and wire versions through v7, shared [normalization](../../../glossary.md#term-normalization) [derivatives](../../../glossary.md#term-derivative), and ordinary [backend](../../../glossary.md#backend) preparation/execution are implemented. The archive does not retract or widen any current qualification. Use [current reference goals](../../../reference/index.md), [Compiler AD](../../../reference/compiler/autograd.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler); [AD glossary](../../../glossary.md#autograd))</span>, [owned Model execution](../../../reference/nn/owned-model.md), and [current backend qualification](../../../backend-guide/production-support.md) instead of the inventories below. Symbolic cold/wire, arbitrary [PURE](../../../glossary.md#term-pure)-body copying, higher-order recurrence and unqualified backend routes are not granted by these supersessions.

<a id="historical-implementation-status-notes"></a>
<a id="planned-contracts"></a>

## Preserved historical text

> **Historical implementation-status notes**
>
> The following earlier implementation-status material is retained for [qualification](../../../glossary.md#term-qualification) history, not as a current absence list. [Engine](../../../glossary.md#term-engine) [compile](../../../glossary.md#compile)/[prepare](../../../glossary.md#prepare)/[run](../../../glossary.md#term-run), shared [Compiler](../../../glossary.md#term-compiler) [AD](../../../glossary.md#autograd), Training, and bounded cold [graph](../../../glossary.md#graph)/[owned Model](../../../glossary.md#model-module-buffer-and-scan-distinctions) [persistence](../../../glossary.md#term-persistence) are implemented. Symbolic execution, [cold reconstruction](../../../glossary.md#term-cold-reconstruction) and wire [admission](../../../glossary.md#term-admission) remain separate qualification boundaries. Consult [Engine lifecycle](../../../reference/engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../glossary.md#lifecycle))</span>, [graph persistence](../../../reference/checkpoint/graph.md), and the [current backend matrix](../../../backend-guide/production-support.md)<span data-glossary-navigation="true"> ([matrix glossary](../../../glossary.md#term-matrix-meanings))</span>.
>
> > **Planned contracts**
> >
> > The following contracts appear in the architecture and [planning](../../../glossary.md#term-planning) documents but are not implemented:
> >
> > - random Operations and typed [tensor](../../../glossary.md#tensor) access or export;
> > - native, mapped, [runtime](../../../glossary.md#term-runtime), and [backend](../../../glossary.md#backend) allocation with deterministic resource [ownership](../../../glossary.md#term-ownership);
> > - expression families beyond binary arithmetic, binary comparison, boolean logical, conditional
> >   selection, cast, unary elementwise, floating-classification, scalar elementwise, the current
> >   value- and
> >   index-producing aggregate operations, cumulative sum/product scans, softmax, layer, RMS, and batch-
> >   normalization inference/training transition, and
> >   contiguous, reshape, expand, permute, expand-dimensions, squeeze, slice, slice update,
> >   target-relative crop, pad, tile, concat,
> >   stack, repeated-select unstack, scalar select, Gather, Gather Elements, Gather-ND Tensor
> >   construction, Scatter Elements construction, Scatter-ND construction, unfold, unfold2d, and
> >   fold2d requests; plus gradient
> >   and trainable state and publication behavior;<span data-glossary-navigation="true"> ([binary glossary](../../../glossary.md#term-binary-descriptor-and-metadata); [unary glossary](../../../glossary.md#term-unary); [elementwise glossary](../../../glossary.md#term-elementwise); [scalar glossary](../../../glossary.md#term-scalar); [softmax glossary](../../../glossary.md#softmax--log-softmax); [batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../glossary.md#term-normalization); [inference glossary](../../../glossary.md#term-inference); [contiguous glossary](../../../glossary.md#term-contiguous); [slice glossary](../../../glossary.md#slice); [slice update glossary](../../../glossary.md#slice-update); [target-relative crop glossary](../../../glossary.md#target-relative-crop); [scalar select glossary](../../../glossary.md#scalar-select); [Gather Elements glossary](../../../glossary.md#gather-elements); [Gather-ND glossary](../../../glossary.md#gather-nd); [Scatter Elements glossary](../../../glossary.md#scatter-elements); [Scatter-ND glossary](../../../glossary.md#scatter-nd); [gradient glossary](../../../glossary.md#term-gradient))</span>
> > - operation-kind families beyond the current binary arithmetic, binary comparison, boolean
> >   logical, unary elementwise, floating-classification, scalar elementwise, conditional selection,
> >   cast, aggregate
> >   reduction, cumulative scan, softmax, layer, RMS, and batch-normalization inference/training,
> >   contiguous-request, reshape/expand,
> >   axis-transform, slice, pad, tile, tensor-composition, scalar-select, axis-gather, one-hot,
> >   gather-ND,
> >   axis-scatter, scatter-ND, and window-transform semantics, plus
> >   family-specific attribute values beyond those documented above;<span data-glossary-navigation="true"> ([aggregate reduction glossary](../../../glossary.md#aggregate-reduction); [cumulative scan glossary](../../../glossary.md#cumulative-scan))</span>
> > - public compiler entry points, concrete [dimension](../../../glossary.md#dimension) [binding](../../../glossary.md#term-binding), [publication](../../../glossary.md#term-publication) delivery, and the engine
> >   `CompiledGraph` [facade](../../../glossary.md#term-facade); compiler-owned `PublicationPlan` and `CompileArtifacts` are current
> >   output contracts whose producing entry remains package-private;
> >   package-private forward-only and phase-aware combined capture, bounded one/two-stage autograd,
> >   binding-free verification inference, canonicalization, exact rewriting, bounded logical-splat
> >   folding, whole-graph DCE, and phase- plus derivative-order-local CSE are current; and<span data-glossary-navigation="true"> ([capture glossary](../../../glossary.md#term-capture-meanings); [autograd glossary](../../../glossary.md#autograd); [canonicalization glossary](../../../glossary.md#occurrence-identity-and-canonical-form); [DCE glossary](../../../glossary.md#term-dead-code-elimination-dce); [derivative glossary](../../../glossary.md#term-derivative); [CSE glossary](../../../glossary.md#term-common-subexpression-elimination-cse))</span>
> > - planning, prepare, runtime, publication execution, and backend execution.
> >
> > General-[scan](../../../glossary.md#term-cumulative-scan) source construction and structural compiled attachments are current as described
> > [above](../../../reference/tensor/general-scan.md#general-scan-source-construction). Compiler scoped capture, first-order AD and demand
> > emission, and Prepare scoped binding/demand validation are implemented. The CPU0030–CPU0033
> > pointwise/broadcast/SUM/layout/MATMUL and FLOAT64 inventories record historical layers, not a
> > current finite body whitelist. Current general Scan composes every ordinary CPU-supported PURE
> > unit under the same ordinary dtype/layout/attribute guards, including supported symbolic
> > occurrences with fixed time/features. Each actual forward and selected ordinary-VJP reverse body
> > qualifies independently. ND roles cut derivatives; unsupported required finite paths reject.
> > Model itself owns none of those downstream stages. CPU BFLOAT16, FLOAT16 and both FP8 types
> > reuse ordinary CPU/shared Compiler AD owners under the same family and geometry constraints.
> > Cold/wire, NN workflows and Metal support remain separately bounded.
> > CPU0042 additionally qualifies fixed T/B/features FLOAT32/FLOAT64 length-mode forward and actual
> > first-order reverse. Prepare0011 adds static length-mode boundary/demand
> > projection, preserving mixed generated reverse roles and outer layouts without CPU read predicates.
> > Model0044's static/one-B source extension and Compiler0015's independent inference/actual AD/all-port
> > binding alone grant no execution. Prepare0015 separately implements one-B/fixed-T/features F32/F64
> > length-mode projection/execution and bounded typed outer/control closure, with root validation and the sole
> > bounded probe completed (evidence and limits:<span data-glossary-navigation="true"> ([owners glossary](../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([validation glossary](../../../glossary.md#term-validation); [pointwise glossary](../../../glossary.md#term-pointwise); [layout glossary](../../../glossary.md#layout); [MATMUL glossary](../../../glossary.md#matrix-multiplication--matmul); [FLOAT64 glossary](../../../glossary.md#term-float64); [finite glossary](../../../glossary.md#term-finite); [general Scan glossary](../../../glossary.md#general-differentiable-scan); [CPU glossary](../../../glossary.md#term-cpu); [PURE glossary](../../../glossary.md#term-pure); [dtype glossary](../../../glossary.md#data-type--datatype); [features glossary](../../../glossary.md#term-feature); [VJP glossary](../../../glossary.md#term-vector-jacobian-product-vjp); [ND glossary](../../../glossary.md#term-nd); [Model glossary](../../../glossary.md#term-model); [BFLOAT16 glossary](../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../glossary.md#term-float16); [FP8 glossary](../../../glossary.md#term-fp8); [NN glossary](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Metal glossary](../../../glossary.md#term-metal); [FLOAT32 glossary](../../../glossary.md#term-float32); [F32 glossary](../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../glossary.md#floating-point-precision-and-rounding))</span>
> > [Prepare0015 Result](../../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
> > final MAIN acceptance remains pending. Cold reconstruction, encoding and owned length-aware NN
> > workflows remain unavailable. Metadata-only test providers do not establish production backend support.<span data-glossary-navigation="true"> ([Metadata glossary](../../../glossary.md#term-metadata))</span>
> >
> > [`OperationKind`](../../../glossary.md#operationkind), [`OperationAttrs`](../../../glossary.md#operationattrs), [`NoOperationAttrs`](../../../glossary.md#nooperationattrs), [`Operation`](../../../glossary.md#operation), `BinaryArithmeticKind`,
> > `BinaryComparisonKind`, `BooleanLogicalKind`, `WhereSelectionKind`, `UnaryElementwiseKind`,
> > `FloatingClassificationKind`, `ScalarElementwiseKind`, `ScalarValueAttrs`, `ClampRangeAttrs`,
> > `CastKind`, `CastAttrs`,
> > `AggregateReductionKind`, `AxisReductionAttrs`, `MultiAxisReductionAttrs`,
> > `StatisticalReductionAttrs`, `ArgExtremaTiePolicy`, `ArgExtremaAttrs`,
> > `MaskedReductionAttrs`, `SumToShapeAttrs`, `CumulativeScanKind`, `CumulativeScanAttrs`, `SoftmaxKind`, `SoftmaxAttrs`,
> > `LayerNormKind`, `LayerNormAttrs`, `AffineLayerNormAttrs`, `RmsNormKind`, `RmsNormAttrs`,
> > `BatchNormKind`, `BatchNormInferenceAttrs`, `BatchNormTrainingAttrs`,
> > `BatchNormTrainingResult`,
> > `ScaledDotProductAttentionKind`, `ScaledDotProductAttentionAttrs`,
> > `ScaledDotProductAttentionResult`,
> > `ContiguousKind`, `ShapeTransformKind`, `TargetShapeAttrs`, `AxisTransformKind`,
> > `PermutationAttrs`, `AxisTransformAttrs`, `SliceKind`, `SliceAttrs`, `CropToShapeAttrs`,
> > `PadKind`, `PadAttrs`,
> > `TileKind`, `TileAttrs`, `TensorCompositionKind`, `CompositionAxisAttrs`,
> > `SelectKind`, `SelectAttrs`, `AxisGatherKind`, `IndexAxisAttrs`, `OneHotKind`, `OneHotAttrs`,
> > `GatherNdKind`, `GatherNdAttrs`, `AxisScatterKind`, `ScatterReduction`, `ScatterElementsAttrs`,
> > `ScatterNdKind`, `ScatterNdAttrs`,
> > `WindowTransformKind`, `UnfoldAxisAttrs`, `FoldAxisAttrs`,
> > `Window2dAttrs`, `Unfold2dAttrs`, `Fold2dAttrs`, `GraphValue`, `CompiledNode`,
> > `GraphPhase`, [`CompiledGraphModel`](../../../glossary.md#compiled-graph--compiledgraphmodel), `CompiledScanBody`, `ScanBody`, `Scan`, `ScanResult`,
> > `ScanKind`, `ScanAttrs`, and
> > `ForwardPublicationBinding`, plus [`TensorProducer`](../../../glossary.md#producer--tensorproducer) and `TensorProvenance`, are current Java API
> > contracts.
> > The producer/provenance pair represents current shared multi-output occurrences including
> > dropout, top-K, batch-normalization training, and explicitly requested attention weights. Binary
> > arithmetic, binary comparison,
> > boolean logical, unary
> > elementwise, scalar elementwise, conditional selection, and cast semantics are the current
> > production concrete kind families with matching public Tensor expression construction. Aggregate
> > reduction is also a current production semantic family; <span data-glossary-navigation="true"> ([API glossary](../../../glossary.md#term-application-programming-interface-api); [producer glossary](../../../glossary.md#term-producer); [provenance glossary](../../../glossary.md#provenance); [dropout glossary](../../../glossary.md#dropout--inverted-dropout); [weights glossary](../../../glossary.md#term-weight))</span>`sum`, [`mean`](../../../glossary.md#term-mean), `prod`, reduction `min`,
> > reduction `max`, [boolean](../../../glossary.md#data-type--datatype) `all`, boolean `any`, and [axis](../../../glossary.md#term-axis)-only `argMin`/`argMax` have matching public Tensor
> > expression methods, including the masked `sum(axis, mask)`, `mean(axis, mask)`, and binding-aware
> > `sumToShape(Shape)` forms. The graph
> > records can compose these, current cumulative-scan expressions, or test-local semantics, but they
> > do not provide a compiler entry point or executable support. Softmax and log-softmax semantic
> > values and their public Tensor expression construction are current. The contiguous semantic value
> > and public Tensor expression construction are also current, including static resolved and dynamic
> > unresolved result-layout rules. Reshape and expand semantic values and both public Tensor
> > expression families are current, with their distinct count-preserving and directional-
> > broadcasting validation and layout rules.
> > MATMUL semantics and public <span data-glossary-navigation="true"> ([log-softmax glossary](../../../glossary.md#softmax--log-softmax); [broadcasting glossary](../../../glossary.md#broadcasting))</span>`Tensor.matmul` expression construction are current for rank-one and
> > higher floating or signed-integral operands. Construction promotes within one numeric category,
> > derives exact vector/matrix/broadcast-batch Shape metadata, defers only the documented unresolved
> > obligations, and records fresh ordered two-input provenance. It performs no multiplication,
> > gradient construction, graph transformation, backend selection, or execution. Compiler capture,
> > verification, and generated first-order formulas are current; concrete execution remains
> > carrier-, gradient-, and backend-qualified. CPU realizes its documented domain subject to the<span data-glossary-navigation="true"> ([rank-one glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix); [vector glossary](../../../glossary.md#term-vector); [Shape glossary](../../../glossary.md#shape); [carrier glossary](../../../glossary.md#term-storage-carrier))</span>
> > [temporary low-operand arithmetic lock](../../../backend-guide/production-support.md#temporary-java-low-computation-lock).
> > Metal realizes promoted no-gradient INT32/INT64 pairs and also
> > realizes general FLOAT32 with generated gradients and no-gradient BFLOAT16/FLOAT32 mixed pairs.
> > The two public [`linear`](../../../glossary.md#term-linear-meanings) overloads are current explicit composition over rank-two PERMUTE, MATMUL,
> > and optional exact rank-one ADD bias. They introduce no LINEAR kind or layer state, fully
> > prevalidate caller-controlled failures before intermediate IDs, and leave the primitive producer
> > chain visible. Concrete support follows the visible operations rather than a separate linear route.
> > The four public <span data-glossary-navigation="true"> ([rank-two glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix); [bias glossary](../../../glossary.md#term-bias))</span>`scaledDotProductAttention` overloads and four
> > `scaledDotProductAttentionWithWeights` overloads are current first-class model construction. They
> > derive exact broadcast-batch, score/weights, and output Shapes for floating query/key/value
> > inputs; optionally retain an exact BOOL mask; and record default-or-explicit scale plus causal
> > eligibility in <span data-glossary-navigation="true"> ([BOOL glossary](../../../glossary.md#data-type--datatype); [mask glossary](../../../glossary.md#term-boolean-mask))</span>`ScaledDotProductAttentionAttrs`. The original family creates exactly one output
> > and no hidden weights or random-state output. The explicit family returns
> > `ScaledDotProductAttentionResult` with output slot zero and normalized weights slot one from one
> > exact shared producer. Deferred embedding, sequence, batch, and mask obligations are current
> > package-private compiler-verification inputs and may remain for later concrete binding. The CPU
> > backend currently executes a deliberately narrow forward subset: fully static, resolved,
> > non-negative-layout FLOAT32 or FLOAT64 occurrences through one direct generated
> > scalar route, with optional right-broadcast BOOL masking, top-left causal eligibility, and either public
> > output count. That backend capability does not change this Tensor API's backend-independent
> > meaning or imply dropout, dynamic shapes, negative strides, overlap/in-place execution,
> > decomposed-attention recognition, fusion, vector/native/packed/flash routing, or universal
> > performance. Gradient construction and saved-value lifetime remain owned by Compiler; dropout and
> > other unsupported execution forms remain planned in their owning layers.
> > Axis permutation, singleton-axis insertion, and selected singleton-axis removal semantic values
> > are current. Public permutation, rank-two transpose, expand-dimensions, and squeeze expression
> > construction is also current. Signed-step slice semantics and public general/single-axis slice and
> > flip expression construction are current, including directional static selected-dimension
> > normalization, canonical zero-extent state, positive-only resolved view geometry, and
> > negative-step unresolved layout. Length-defined extraction is also current: it retains exact
> > static requested lengths across unresolved selected extents after proving non-negative coordinates
> > and defers only the unresolved upper bound. Functional signed multi-axis slice update,
> > target-relative symbolic placement, and target-relative crop are current. Array update derives
> > finite selected lengths from static update Dimensions. Shape-based update retains the exact update
> > Shape as its target and the exact prefix Shape while returning the exact base Shape. Crop retains
> > exact target/prefix Shapes, including symbolic Dimensions. Updates leave layout unresolved; crop
> > and unresolved/empty/negative-step extraction do likewise. Fully static bounds fail locally,
> > while only the documented unresolved fits remain later obligations. Current compiler
> > inference/preflight/autograd have adopted both construction forms and retain those unresolved
> > fits as occurrence-owned constraints.
> > Constant-padding and complete-pattern tiling semantic kinds and immutable normalized attributes
> > are current, as are public pad/tile Tensor request validation, checked Shape derivation, unresolved
> > result-layout policy, exact metadata retention, and one-input provenance. Padding-constant
> > conversion, gradients, compiler behavior, materialization, backend/ONNX behavior, and execution
> > remain planned.
> > Tensor-composition semantic kinds and public concat/stack construction are current. Public unstack
> > is ordered repeated scalar select after upfront count validation. Its immutable result list uses
> > independent one-output producers, provenance index zero, and SELECT's conditional layout behavior.
> > The general producer/output-index representation remains current. Compiler capture or decomposition, gradients,
> > materialization, lowering, ONNX mapping, and execution remain planned.
> > Scalar-select semantics and public <span data-glossary-navigation="true"> ([embedding glossary](../../../glossary.md#embedding); [backend capability glossary](../../../glossary.md#backend-capability); [strides glossary](../../../glossary.md#term-stride); [fusion glossary](../../../glossary.md#term-fusion); [lifetime glossary](../../../glossary.md#term-lifetime); [permutation glossary](../../../glossary.md#permutation); [singleton glossary](../../../glossary.md#term-singleton); [transpose glossary](../../../glossary.md#term-transpose); [canonical glossary](../../../glossary.md#term-canonical-form); [extent glossary](../../../glossary.md#term-extent); [view glossary](../../../glossary.md#term-view-meanings); [coordinates glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix); [target glossary](../../../glossary.md#term-target); [preflight glossary](../../../glossary.md#term-preflight); [occurrence glossary](../../../glossary.md#term-occurrence); [Constant glossary](../../../glossary.md#term-constant); [tiling glossary](../../../glossary.md#tiling); [immutable glossary](../../../glossary.md#term-immutable); [materialization glossary](../../../glossary.md#materialization); [ONNX glossary](../../../glossary.md#term-onnx); [decomposition glossary](../../../glossary.md#term-decomposition); [lowering glossary](../../../glossary.md#lowering))</span>`Tensor.select` expression construction are current. The public
> > method normalizes a raw axis, normalizes and bounds-checks an index when the selected extent is
> > static, accepts a non-negative index on a dynamic selected extent with its upper bound deferred,
> > removes the selected axis, derives conditional resolved logical-view geometry, and records fresh
> > one-input provenance. Value access, a physical alias guarantee, gradients, compiler capture and
> > canonicalization, materialization, backend lowering, and execution remain planned.
> > Gather and Gather Elements semantics and public Tensor construction are current. <span data-glossary-navigation="true"> ([alias glossary](../../../glossary.md#aliasing-and-arena))</span>[`GATHER`](../../../glossary.md#gather) and
> > `GATHER_ELEMENTS` preserve distinct ordered `[data, indices]` meanings and Shape relationships. Expression
> > construction validates [`INT32`](../../../glossary.md#data-type--datatype)/[`INT64`](../../../glossary.md#data-type--datatype) index type, normalizes the data axis, derives and validates
> > family-specific Shapes, preserves data metadata, leaves layout unresolved, and records fresh
> > two-input provenance. Index-value access
> > and bounds checks, gradients, compiler behavior, lowering, and execution remain planned or
> > separately owned.
> > Gather-ND semantics and public Tensor construction are current. `GATHER_ND` uses ordered
> > `[data, indices]` roles, takes [tuple depth](../../../glossary.md#term-tuple-depth) from the final indices Dimension, and pairs with
> > `GatherNdAttrs(batchDimensions)`. The public methods validate exact index type, both ranks,
> > structurally equal batch prefixes, and static positive tuple depth before deriving an
> > indices-prefix-plus-data-suffix Shape and recording exact provenance. Results retain data type and
> > gradient eligibility, leave layout unresolved, and remain unlabeled and storage-free. Index-value
> > bounds, gradients, compiler behavior, lowering, and execution remain separately owned.
> > Scatter Elements semantics and public Tensor construction are current. <span data-glossary-navigation="true"> ([data type glossary](../../../glossary.md#data-type--datatype))</span>`SCATTER_ELEMENTS` uses
> > ordered `[data, indices, updates]`, preserves the exact data Shape in a new functional result, and
> > uses `ScatterElementsAttrs` with explicit
> > `NONE`/`ADD`/`MUL`/`MAX`/`MIN` reduction meaning. Public construction validates exact integral
> > index type, matching data/update type, reduction eligibility, raw axis, and same-rank Shapes.
> > Results retain exact data Shape/type, combine
> > data/update gradient eligibility, leave layout unresolved, and record exact ordered provenance.<span data-glossary-navigation="true"> ([rank glossary](../../../glossary.md#term-rank))</span>
> > `NONE` requires unique targets, but duplicate detection, index bounds, value writes/reductions,
> > gradient rules, compiler behavior, lowering, and execution remain later responsibilities.
> > Scatter-ND semantics and public Tensor construction are current.
> > `SCATTER_ND` uses ordered `[data, indices, updates]` roles, takes tuple depth from the final
> > indices Dimension, and pairs with `ScatterNdAttrs(batchDimensions, reduction)`. Updates follow the
> > exact Gather-ND result-Shape formula, and the functional result has the exact data Shape without
> > in-place mutation. Public construction validates exact index and update types, reduction
> > eligibility, ranks, shared batch prefix, static positive tuple depth, and the exact updates Shape.
> > It returns a fresh unresolved-layout, unlabeled, storage-free result with exact data Shape/type,
> > data/update eligibility OR, and exact ordered provenance. Index values and bounds, duplicate-target
> > detection, value writes or reductions, derivatives, compiler behavior, lowering, backend behavior,
> > and execution remain planned or separately owned. Configurable `MUL`, `MIN`, and `MAX` combine the
> > base and every addressed update scalar exactly once, preserve an unaddressed base representation,
> > and have the portable order-independent floating and integral represented-value contract described
> > above; duplicate tuples remain distinct scalar contributions.
> > Window-transform semantic kinds and immutable normalized attributes are current. Public `unfold`,
> > `foldAxis`, direct and exact-typed-[padding](../../../glossary.md#padding) `unfold2d`, and `fold2d` expression construction is
> > current. General-axis fold validates static window geometry; the 2D forms retain exact dynamic
> > channel/spatial formulas and require structural fold compatibility. Every result has unresolved
> > layout, preserved eligibility, exact attributes, and one-input provenance. Gradient construction, compiler
> > capture and canonicalization, materialization, lowering, backend/ONNX behavior, and execution
> > remain planned.<span data-glossary-navigation="true"> ([channel glossary](../../../glossary.md#term-channel); [spatial glossary](../../../glossary.md#term-spatial-axes))</span>
> >
>
> ---
>
