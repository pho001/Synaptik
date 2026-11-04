# Historical reference [snapshot](../../../glossary.md#term-snapshot): [Compiler](../../../glossary.md#term-compiler) [qualification](../../../glossary.md#term-qualification) context

This is an archived, composite qualification narrative preserved from the pre-redesign [API](../../../glossary.md#term-application-programming-interface-api) reference at base `f3f7c2face52c1fa886e075a68faa6d11765cbd0`. Its task-era inventories were accumulated at different [checkpoints](../../../glossary.md#term-checkpoint): this is not a single [release](../../../glossary.md#term-release)-wide capability statement. All quoted “current”, “planned”, “unsupported” and version/count claims below belong to those earlier checkpoints, not today's API.

**Known supersessions:** [CPU](../../../glossary.md#term-cpu) low-[precision](../../../glossary.md#term-precision) parity, fixed-cell first-order [BPTT](../../../glossary.md#term-backpropagation-through-time-bptt), bounded static length-aware cold/[owned Model](../../../glossary.md#model-module-buffer-and-scan-distinctions) workflows and wire versions through v7, shared [normalization](../../../glossary.md#term-normalization) [derivatives](../../../glossary.md#term-derivative), and ordinary [backend](../../../glossary.md#backend) preparation/execution are implemented. The archive does not retract or widen any current qualification. Use [current reference goals](../../../reference/index.md), [Compiler AD](../../../reference/compiler/autograd.md)<span data-glossary-navigation="true"> ([AD glossary](../../../glossary.md#autograd))</span>, [owned Model execution](../../../reference/nn/owned-model.md), and [current backend qualification](../../../backend-guide/production-support.md) instead of the inventories below. Symbolic cold/wire, arbitrary [PURE](../../../glossary.md#term-pure)-body copying, higher-order recurrence and unqualified backend routes are not granted by these supersessions.

<a id="compiler-qualification-context"></a>

## Preserved historical text

> **[Compiler](../../../glossary.md#term-compiler) [qualification](../../../glossary.md#term-qualification) context**
>
> Completed model task 0025 supplies one narrow producer contract needed by pre-capture autograd:
> every producer retains and returns the canonical exact Tensor wrapper for each ordered output
> slot. <span data-glossary-navigation="true"> ([producer glossary](../../../glossary.md#term-producer); [capture glossary](../../../glossary.md#term-capture-meanings); [autograd glossary](../../../glossary.md#autograd))</span>`TensorFactory` constructs the producer, all wrappers, and their indexed provenance as one
> unpublished immutable occurrence before returning a result. The resulting
> Tensor/provenance/producer/output cycle is ordinary collectable model metadata, not graph
> membership or runtime ownership.<span data-glossary-navigation="true"> ([provenance glossary](../../../glossary.md#provenance); [immutable glossary](../../../glossary.md#term-immutable); [occurrence glossary](../../../glossary.md#term-occurrence); [metadata glossary](../../../glossary.md#term-metadata); [graph glossary](../../../glossary.md#graph); [runtime glossary](../../../glossary.md#term-runtime); [ownership glossary](../../../glossary.md#term-ownership-meanings))</span>
>
> Compiler task 0004 is Complete. Package-private named gradient rules reuse existing public Tensor
> expression methods directly on the original forward expression, and package-private
> `GraphCompiler` captures forward outputs and generated gradient roots together once. This changes
> no public Tensor method or ergonomic result carrier. Its mode-neutral internal
> `GraphCompilation` is not a [Tensor](../../../glossary.md#tensor) result [carrier](../../../glossary.md#term-storage-carrier) or the current `CompileArtifacts` aggregate. The
> implementation adds no
> `Tensor.gradient`, `Tensor.backward`, [mutable](../../../glossary.md#term-mutable) [gradient](../../../glossary.md#term-gradient) state, placeholder Tensor for a captured
> [`ValueId`](../../../glossary.md#logical-valueid-versus-physical-memory-slot), or [model](../../../glossary.md#term-model)-owned [derivative](../../../glossary.md#term-derivative) rule.
>
> The current functional boundary accepts one or two stages with exact forward or generated
> first-stage-gradient output references, aligned explicit seeds or exact typed scalar defaults,
> and ordered identity-unique targets from the complete original forward inventory. The compiler
> preflights each selected slice before creating any default seed or formula Tensor. Exact
> target/producer object identity remains temporary compiler bookkeeping, not public Tensor state
> or graph membership.<span data-glossary-navigation="true"> ([identity glossary](../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([seeds glossary](../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [slice glossary](../../../glossary.md#slice); [seed glossary](../../../glossary.md#term-seed))</span>
>
> The closed [matrix](../../../glossary.md#term-matrix-meanings) currently covers same-floating-type [binary](../../../glossary.md#term-binary-descriptor-and-metadata) and exact-[scalar](../../../glossary.md#term-scalar)
> `ADD`/`SUB`/`MUL`, same-type branch-only `WHERE`, same-type floating `CAST`,
> `NEG`/[`EXP`](../../../glossary.md#term-expx)/`EXPM1`/[`SIGMOID`](../../../glossary.md#term-sigmoid)/[`TANH`](../../../glossary.md#term-tanh), ordinary full/single-/multi-[axis](../../../glossary.md#term-axis) `SUM`, `CUM_SUM`,
> [`CONTIGUOUS`](../../../glossary.md#term-contiguous), `RESHAPE`, `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, and `PERMUTE`. Generated BFLOAT16,
> FLOAT32, and FLOAT64 positive-zero/one leaves are storage-free and explicitly registered as
> logical splats. Everything else fails closed on a selected route until a follow-up selects the
> required formula or tie, boundary, discontinuity, singularity, exceptional-value, empty-domain,
> or mixed-floating conversion policy.<span data-glossary-navigation="true"> ([leaves glossary](../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../glossary.md#term-bfloat16); [FLOAT64 glossary](../../../glossary.md#term-float64))</span>
>
> The current semantic vocabulary also includes `BatchNormKind.BATCH_NORM_INFERENCE` and
> `BATCH_NORM_TRAINING`. Inference has one output. Training has five ordered outputs: normalized
> output, next running mean, next running variance, saved batch mean, and saved inverse standard
> deviation. Only its first three Tensor wrappers are public results; all five descriptors and
> output positions remain current producer metadata. The current semantic vocabulary also includes<span data-glossary-navigation="true"> ([Inference glossary](../../../glossary.md#term-inference); [variance glossary](../../../glossary.md#term-variance); [batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [standard deviation glossary](../../../glossary.md#term-standard-deviation); [descriptors glossary](../../../glossary.md#term-descriptor))</span>
> `ContiguousKind.CONTIGUOUS`, a parameterless
> request for logically equivalent canonical dense row-major, zero-offset result geometry. Public<span data-glossary-navigation="true"> ([row-major glossary](../../../glossary.md#term-row-major))</span>
> `Tensor.contiguous()` now constructs that request without allocating or copying storage.
> `ShapeTransformKind.RESHAPE` and `ShapeTransformKind.EXPAND` are also current semantic identities.
> Both pair with `TargetShapeAttrs`, which stores an exact normalized model [`Shape`](../../../glossary.md#shape).
> `Tensor.reshape(long...)` normalizes raw dimensions and performs locally provable count
> validation, while <span data-glossary-navigation="true"> ([validation glossary](../../../glossary.md#term-validation))</span>`Tensor.reshape(Shape)` retains an exact normalized [target](../../../glossary.md#term-target). The matching
> `Tensor.expand` overloads retain literal or exact target Shapes, validate directional
> right-aligned compatibility, and derive logical view strides when geometry is resolved.
> Package-private compiler verification of captured descriptors and retained reshape obligations is
> current. The closed first-order matrix also currently reverses <span data-glossary-navigation="true"> ([view glossary](../../../glossary.md#term-view-meanings); [strides glossary](../../../glossary.md#term-stride))</span>`CONTIGUOUS`, `RESHAPE`, `EXPAND`,
> `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE` through public Tensor metadata operations. Mandatory
> graph-local ID canonicalization and whole-graph DCE plus phase-local CSE are current internal
> behavior. CPU physical execution is current for its documented bounded affine units. Metal forward
> execution has an exact occurrence baseline: canonical <span data-glossary-navigation="true"> ([canonicalization glossary](../../../glossary.md#occurrence-identity-and-canonical-form); [DCE glossary](../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../glossary.md#term-common-subexpression-elimination-cse); [CPU glossary](../../../glossary.md#term-cpu); [Metal glossary](../../../glossary.md#term-metal))</span>[`FLOAT32`](../../../glossary.md#term-float32) `NEG`, `ABS`,
> `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, and `CONTIGUOUS`; bounded [canonical](../../../glossary.md#term-canonical-form)
> `FLOAT32` `UNFOLD_AXIS`; canonical positive-[rank](../../../glossary.md#term-rank) `FLOAT32` data [`GATHER`](../../../glossary.md#gather) with canonical [`INT32`](../../../glossary.md#data-type--datatype)
> indices; positive-rank `INT32`-to-[`BOOL`](../../../glossary.md#data-type--datatype) `ONE_HOT`; and canonical positive-rank
> `FLOAT32`/`INT32`/`FLOAT32` `SCATTER_ELEMENTS/NONE`. Metal UNFOLD_AXIS accepts input rank 1..15,
> size 1..16, a positive step, and size no larger than the selected extent. It materializes the
> exact floor-count rank-plus-one Shape, repeats overlapping source representations, omits incomplete
> tails, applies no padding, and leaves input storage unchanged. Metal validates every index before
> selector dispatch or target writes; scatter completes its bounds pass before complete-target
> uniqueness. Bounds and duplicates publish the exact Model exception text in stable node and
> row-major ordinal order, and targets and inputs remain unchanged on failure. Metal caller ingress,
> authenticated SELECT/SLICE publication, and CPU/Metal transfer cover all seven carriers over their
> exact supported descriptors. Cross-owner values may be rank-zero through rank-sixteen and use a
> canonical or positive-stride non-overlapping storage layout with a checked physical span; BOOL
> validates logical zero/one bytes while leaving storage holes uninterpreted. Metal additionally
> executes its documented FLOAT32 arithmetic/reduction/scan rows, every positive-static FLOAT32
> MATMUL geometry, no-gradient BFLOAT16/FLOAT32 mixed MATMUL, and no-gradient promoted INT32/INT64
> MATMUL. Independently admitted homogeneous BFLOAT16/FLOAT16 counterparts use the custom route.
> Existing rank-two FLOAT32 matrix products retain direct MPSGraph, while new MATMUL forms use the
> fixed custom program. General FLOAT32 MATMUL also executes the Compiler-generated
> explicitly seeded first-order formulas for both operands through exact local last-two-axis
> transposes beneath the existing shape-restoration boundaries. This introduces no complete
> window/indexing backward, implicit seeding, or unrestricted Metal training claim.<span data-glossary-navigation="true"> ([extent glossary](../../../glossary.md#term-extent); [padding glossary](../../../glossary.md#padding); [dispatch glossary](../../../glossary.md#term-dispatch); [node glossary](../../../glossary.md#node); [publication glossary](../../../glossary.md#term-publication); [owner glossary](../../../glossary.md#term-owner); [rank-zero glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix); [layout glossary](../../../glossary.md#layout); [scan glossary](../../../glossary.md#term-cumulative-scan); [MATMUL glossary](../../../glossary.md#matrix-multiplication--matmul); [homogeneous glossary](../../../glossary.md#term-homogeneous); [FLOAT16 glossary](../../../glossary.md#term-float16); [rank-two glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix); [MPSGraph glossary](../../../glossary.md#term-mpsgraph))</span>
>
> `AxisTransformKind.PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE` are current semantic [identities](../../../glossary.md#term-identity-meanings).
> `PermutationAttrs` stores a complete normalized output-to-input axis [permutation](../../../glossary.md#permutation), while
> `AxisTransformAttrs` stores one normalized non-negative insertion or removal position. Public
> `Tensor.permute` now owns input-rank checks, one-time negative-axis normalization, result
> Shape/layout derivation, and provenance; rank-two <span data-glossary-navigation="true"> ([normalization glossary](../../../glossary.md#term-normalization))</span>`Tensor.transpose()` uses the same `PERMUTE`
> meaning with `[1, 0]`. Public `Tensor.expandDims` and `Tensor.squeeze` now own their distinct
> insertion-position and existing-axis normalization, result Shape/layout derivation, static
> singleton proof for removal, and provenance.<span data-glossary-navigation="true"> ([singleton glossary](../../../glossary.md#term-singleton))</span>
>
> `SliceKind.SLICE` and `SLICE_UPDATE` are current extraction and functional-replacement identities.
> `SliceAttrs` stores normalized [finite](../../../glossary.md#term-finite) signed [coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix) sequences shared by both meanings, while
> `CropToShapeAttrs` stores exact target/prefix Shapes for target-relative extraction or placement.
> Public general and single-axis slicing normalize raw directional half-open bounds against
> selected static dimensions; <span data-glossary-navigation="true"> ([half-open glossary](../../../glossary.md#term-half-open-interval))</span>`flip(int...)` constructs the same negative-step extraction directly.
> Public `sliceByLength` instead accepts normalized non-negative starts and explicit finite lengths,
> so an unresolved selected extent can defer only its upper-bound proof. Positive non-empty
> extraction may derive logical view geometry, while empty or negative-step extraction remains
> layout-unresolved. The two `sliceUpdate` forms perform signed coordinate-sequence or exact
> target-relative replacement; `cropToShape` performs exact target-relative extraction. Updates and
> crop always leave layout unresolved and retain their unresolved fit obligations without a
> Model-owned constraint object. Current compiler verification and first-order autograd have
> adopted length-defined extraction and target-relative placement, including their retained
> constraints and fail-closed preflight. Transformation, concrete binding,
> materialization, backend lowering, ONNX mapping, and execution remain planned in their owning
> layers.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../glossary.md#term-fail-closed); [preflight glossary](../../../glossary.md#term-preflight); [binding glossary](../../../glossary.md#term-binding); [materialization glossary](../../../glossary.md#materialization); [backend glossary](../../../glossary.md#backend); [lowering glossary](../../../glossary.md#lowering); [ONNX glossary](../../../glossary.md#term-onnx))</span>
>
> `PadKind.PAD` with `PadAttrs` and `TileKind.TILE` with `TileAttrs` are current semantic contracts.
> Padding attributes store immutable ordered before/after widths plus one exact typed scalar value.
> Tiling attributes store immutable positive complete-pattern repeat counts. Empty lists
> describe rank-zero scalar identity parameters. Public <span data-glossary-navigation="true"> ([typed scalar value glossary](../../../glossary.md#typed-scalar-value--scalarvalue); [Tiling glossary](../../../glossary.md#tiling); [parameters glossary](../../../glossary.md#term-parameter))</span>`Tensor.pad` and `Tensor.tile` now own exact
> rank validation, checked result-Shape derivation, unresolved result layout, exact metadata
> retention, and fresh one-input provenance. The semantic values themselves still do not construct
> Tensors or define conversion of a padding constant, gradients, materialization, compiler or
> backend behavior, ONNX mapping, or execution.<span data-glossary-navigation="true"> ([constant glossary](../../../glossary.md#term-constant))</span>
>
> `TensorCompositionKind.CONCAT` and `STACK` are current semantic identities.
> `CompositionAxisAttrs` stores one normalized non-negative axis shared by concat and stack, while
> the kind distinguishes an existing concat axis from a newly inserted stack result-axis position.
> Public composition methods, input and result validation, Shape derivation, immutable result
> collection construction, and provenance attachment are current. Public unstack is an ordered
> convenience over scalar `SELECT`, so every result has its own independent single-output producer
> and provenance output index zero; compiler capture or decomposition, gradients,
> materialization, and execution remain planned.<span data-glossary-navigation="true"> ([decomposition glossary](../../../glossary.md#term-decomposition))</span>
>
> `SelectKind.SELECT` is a current scalar-index semantic [identity](../../../glossary.md#term-identity-meanings). It pairs with
> `SelectAttrs(axis, index)`, whose normalized non-negative axis and scalar coordinate fix one
> logical position on one source axis and remove that axis from the conceptual result. For source
> Shape `[2, 3, 4]`, axis `1` and index `2` therefore [mean](../../../glossary.md#term-mean) conceptual result Shape `[2, 4]`.
> The attributes contain no input Shape, so they cannot prove that the axis exists or that the index
> is in bounds. Public `Tensor.select` now owns caller-facing negative normalization, local bounds
> validation, result Shape/layout construction, and exact one-input provenance. Gradients, compiler
> capture, materialization, backend lowering, and execution remain planned.
>
> `AxisGatherKind.GATHER` and `GATHER_ELEMENTS` are current tensor-index semantic identities. Each
> pairs with `IndexAxisAttrs`, which stores one already normalized non-negative data-axis position,
> and each has ordered logical inputs `[data, indices]`. Their index alignment and conceptual result
> Shapes deliberately differ. These semantic values do not inspect either
> input, validate index data type or Shape compatibility, construct a Tensor or provenance, define
> gradients, or execute indexing. Public <span data-glossary-navigation="true"> ([alignment glossary](../../../glossary.md#term-alignment); [data type glossary](../../../glossary.md#data-type--datatype))</span>`Tensor.gather` and `gatherElements` own raw-axis
> normalization, exact `INT32`/[`INT64`](../../../glossary.md#data-type--datatype) index representation, family-specific structural Shape checks
> and construction, and exact ordered two-input provenance. Both consume an existing index Tensor.
> None of the methods interprets index values or checks their bounds.
>
> `OneHotKind.ONE_HOT` is the current index-encoding semantic identity. It pairs with
> `OneHotAttrs(depth)`, which stores one positive static trailing-axis extent, and consumes ordered
> logical input `[indices]`. Public `Tensor.oneHot(depth)` requires exact INT32 or INT64 index
> metadata, preserves every receiver Dimension reference, appends one fresh
> `StaticDimension(depth)`, and constructs a fixed-BOOL, non-differentiable result. Construction
> reads no values. Valid eventual execution requires every index <span data-glossary-navigation="true"> ([differentiable glossary](../../../glossary.md#derivative-and-chain-rule))</span>`i` to satisfy `0 <= i < depth`;
> invalid values do not wrap, clamp, select a default, or produce an all-false row. Compiler
> capture, constant analysis, bounds enforcement, lowering, backend support, and execution are not
> implemented by this model contract.
>
> `GatherNdKind.GATHER_ND` is the current tuple-index semantic identity. It pairs with
> `GatherNdAttrs(batchDimensions)`, which stores only the normalized non-negative count of shared
> leading batch Dimensions. Ordered logical inputs are `[data, indices]`; the final indices
> Dimension supplies tuple depth `K`, so that occurrence-specific fact is not duplicated in the
> attributes. Public `Tensor.gatherNd` construction now owns exact `INT32`/`INT64` index-type checks,
> input-rank and batch compatibility, structural batch-prefix equality, static positive tuple-depth
> validation, result Shape construction, and exact ordered provenance. Index-value bounds,
> gradients, compiler behavior, backend behavior, and execution remain planned or separately owned.
>
> `AxisScatterKind.SCATTER_ELEMENTS` is the current functional axis-scatter semantic identity. It
> has ordered logical inputs `[data, indices, updates]` and conceptually produces a new result with
> the exact `data` Shape without mutating `data` in place. It uses
> `ScatterElementsAttrs(axis, reduction)` and the reusable
> `ScatterReduction` vocabulary `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`. Replacement with `NONE`
> requires unique target coordinates; value-aware duplicate detection is deferred. Public Tensor
> construction now owns caller-axis normalization, exact index and data/update type checks, its
> same-rank Shape rule, result metadata, and ordered provenance. Index-value bounds,
> duplicate detection, gradients, compiler behavior, backend behavior, and execution remain planned
> or separately owned.
>
> `AxisScatterKind.SCATTER_ADD` is the fixed additive Gather-compatible scatter identity. It pairs
> with `IndexAxisAttrs(axis)`, consumes ordered logical inputs `[data, indices, updates]`, and returns
> a new result with the exact `data` Shape. Its updates Shape is exactly the Shape that ordinary
> axis Gather would read for the same data and indices; the base and every addressed update
> contribute, so duplicate indices accumulate. Public Tensor construction owns normalized axis,
> exact index/data/update type and Shape checks, result metadata, gradient eligibility, and ordered
> provenance. It does not mutate inputs, inspect index bounds, execute work, or construct gradients.<span data-glossary-navigation="true"> ([normalized axis glossary](../../../glossary.md#normalized-axis))</span>
>
> `ScatterNdKind.SCATTER_ND` is the current functional tuple-index scatter semantic identity. It
> pairs with `ScatterNdAttrs(batchDimensions, reduction)`, which stores an already normalized
> non-negative shared-batch count and one explicit `ScatterReduction`. Ordered logical inputs are
> `[data, indices, updates]`; the final indices [Dimension](../../../glossary.md#dimension) supplies [tuple depth](../../../glossary.md#term-tuple-depth) `K`, and updates have
> the Shape that Gather-ND would read from the same data and indices. The conceptual result is a new
> value with the exact data Shape, so the operation does not mutate <span data-glossary-navigation="true"> ([Gather-ND glossary](../../../glossary.md#gather-nd); [operation glossary](../../../glossary.md#operation))</span>`data`. Public Scatter-ND Tensor
> construction now owns exact index and update type checks, reduction eligibility, input-rank and
> batch compatibility, structural batch-prefix equality, static positive tuple depth, exact updates-
> Shape validation, result metadata, and ordered provenance. Index bounds, <span data-glossary-navigation="true"> ([Scatter-ND glossary](../../../glossary.md#scatter-nd))</span>`NONE` duplicate
> detection, gradients, compiler behavior, backend behavior, and execution remain planned or
> separately owned.
>
> `WindowTransformKind.UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, `FOLD2D`, `UNFOLD3D`, and `FOLD3D`
> are current semantic identities with normalized immutable window attributes. Public
> `Tensor.unfold`, `foldAxis`, both `unfold2d` and `unfold3d` variants, and `fold2d`/`fold3d` own
> their input-aware Shape and compatibility checks, checked static or canonical symbolic
> arithmetic, unresolved result layout, and exact one-input provenance.
> These contracts describe materialized windows or overlap-summing
> scatter-add without reading values, attaching storage, capturing a graph, generating gradients,
> or executing.
>
>
> ---
>
