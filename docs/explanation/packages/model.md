# Model: values, expressions, and graph meaning

This atlas explains how `modules/model` describes a computation without executing it. Start with the [learning journey](../../tutorials/index.md) for a runnable application, or the [framework explanation](../framework.md) for the lifecycle around these packages.

> This is explanatory documentation. The [architecture root](../../../ARCHITECTURE.md), [Model ownership contract](../../architecture/contracts/foundational-modules.md#modulesmodel), [numerical semantics](../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics), and [scan contract](../../architecture/contracts/recurrent-scan.md) remain authoritative.

## Read the model in three layers

```text
application expression                         captured graph
Tensor --provenance--> TensorProducer            CompiledNode (NodeId)
  |                    | ordered inputs              | ordered ValueIds
  | descriptor         | canonical outputs           v
  v                    v                         GraphValue (ValueId)
TensorDescriptor <-----+                              | descriptor
  | dtype + Shape + optional LayoutDescriptor <-------+
  |
Tensor --optional borrowed association--> HostTensorStorage --> caller-owned memory
```

Read the upper arrows as semantic relationships, not execution. A Tensor has stable expression origin even when its borrowed storage is replaced or cleared. Capture creates graph-local identities; it does not turn a Tensor into a node. The bottom arrow concerns host memory only, not a device allocation or prepared executable.

The operation-family sections below share a specific boundary: kind and attributes are immutable semantic values consumed by Tensor construction and Compiler inspection. They do not own operand memory, gradients, kernels, runtime scratch, or backend capability. Family-specific guards and input/output relationships differ; those differences are the reason for separate packages. Public Tensor methods are the application composition surface. Package-private `Tensor*Expressions` helpers are implementation details, not supported application API. A source-valid expression is not proof of differentiation, serialization, or backend execution support.

## A concrete walk through LearningJourney

This is a paper trace of the canonical program, not a claim that it was executed for this document. Use the runnable [examples entry point](../../examples/README.md) and [training guide](../../user-guide/training.md) for prerequisites and commands.

The program supplies FLOAT32 input `[2,1]` containing `[1,3]`, a gradient-eligible weight `[1,1]` containing zero, no bias, and target `[2,1]` containing `[2,6]`. The public builder copies the supplied weight into model-owned parameter state; that ownership is above this Model module.

1. `linear(weight)` describes `input @ transpose(weight)`. Transpose is a `PERMUTE` with axes `[1,0]`; MATMUL contracts the feature axes. Its result descriptor is FLOAT32 `[2,1]`, not a computed array.
2. Squared-error SUM means `sum((prediction-target)^2)`. The loss family can represent this objective as one semantic occurrence; its mathematical formula does not require separate public SUB/MUL/SUM occurrences. Initial prediction is `[0,0]` and objective is `4+36=40`.
3. The Compiler captures producer occurrences and assigns NodeIds and ValueIds. It owns automatic differentiation. `requiresGrad` in a descriptor requests eligibility; no gradient field is installed on Tensor.
4. The weight sensitivity is `2*((0-2)*1 + (0-6)*3) = -40`. `Sgd(0.05)` updates the weight to `0 - 0.05*(-40) = 2`. Training/NN owns that state transition, not an operation-kind enum or Tensor provenance.
5. Re-execution predicts `[2,6]`; fresh values `[2,4]` under the same `[2,1]` signature predict `[4,8]`. Replacing values is different from changing shape: a mismatching static signature is an error, not implicit recompilation.

The direct path therefore exercises datatype, shape, layout, storage, tensor, graph, operation, operation.layout, operation.linalg, and operation.loss. Binary/scalar/reduction families explain the equivalent elementary formula and derivative algebra; this does not promise its exact captured decomposition. Other families below are intentionally absent from this tiny linear example.

## io.github.pho001.synaptik.model.datatype

Important entry points: [DataType](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataType.java), [DataTypePromotion](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataTypePromotion.java), and [ScalarValue](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/ScalarValue.java).

This package answers what one element means: logical category, byte width, differentiability, promotion, and low-precision representation. FLOAT64, FLOAT32, BFLOAT16, FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2, INT32, INT64, and BOOL are Model meanings, not statements about installed native hardware. Promotion is used by input-aware expression construction; arbitrary cross-category arithmetic is not implied by the existence of casts.

A `ScalarValue` stores a dtype and exact primitive bits. It is operation metadata, not a rank-zero Tensor or borrowed storage. Its equality distinguishes signed floating zeros and raw NaN payloads, and a wrong-type accessor fails rather than converting. [BFloat16Bits](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/BFloat16Bits.java), [Float16Bits](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/Float16Bits.java), and the two Float8 bit utilities define representation conversions. This keeps low-format semantics independent of CPU/Metal availability. These immutable values own no memory lifetime. LearningJourney uses FLOAT32; its learning rate belongs to optimizer configuration, not a Tensor dtype promotion request.

## io.github.pho001.synaptik.model.shape

Important entry points: [Shape](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/Shape.java), [Dimension](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/Dimension.java), [DimensionExpressions](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DimensionExpressions.java), and [ShapeBroadcast](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/ShapeBroadcast.java).

A Shape is an immutable ordered list of static extents, named dynamic dimensions, or dimension expressions. Rank zero is a scalar with one logical element; a static zero extent makes an empty tensor. Static sizes cannot be negative. Numeric `-1` reshape syntax is normalized by the expression boundary and is not stored as a negative dimension.

Shape transformations consume dimensions and return dimensions; they do not bind runtime values or provide strides. Broadcasting aligns trailing axes, permitting equal dimensions or singleton expansion and retaining unresolved obligations where proof must wait. Expression dimensions describe derived geometry rather than inventing a sampled extent. Checked element-count arithmetic may overflow even though individual dimensions are valid. Compiler/Prepare owns later symbolic binding. In LearningJourney, `[2,1] @ [1,1]` yields `[2,1]`; the objective is scalar. See the [tensor guide](../../user-guide/tensors.md) for shape-facing operations.

## io.github.pho001.synaptik.model.layout

Important entry points: [LayoutDescriptor](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/layout/LayoutDescriptor.java), [LayoutGeometry](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/layout/LayoutGeometry.java), and [LayoutKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/layout/LayoutKind.java).

Layout describes resolved element geometry, not memory ownership. For a coordinate `i`, the element position is `storageOffset + sum(i[axis]*stride[axis])`. Strides and offsets are in **elements**, not bytes. A descriptor holds copied strides, derived classification, an explicit view flag, and referenced span; it does not retain the originating Shape or a storage object.

Only fully static shapes admit numeric layouts. Stride count must equal rank; strides and offset are nonnegative; repeating a dimension larger than one with stride zero requires `isView=true`. Geometry arithmetic is checked. Empty geometry references no elements, while scalar geometry references one. Canonical contiguity and alias metadata answer different questions: a view may have contiguous geometry without owning storage. TensorDescriptor separately validates the layout against its paired Shape.

LearningJourney's dense host inputs use resolved geometry. Its weight transpose describes a view mapping; this does not itself attach the weight's storage to the fresh result Tensor. Most computed expression layouts may remain unresolved until later inference/materialization decisions.

## io.github.pho001.synaptik.model.storage

Important entry points: [HostTensorStorage](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/storage/HostTensorStorage.java) and [MemorySegmentStorage](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/storage/MemorySegmentStorage.java).

This is the borrowed host-memory boundary. `MemorySegmentStorage` retains an exact caller-supplied JDK MemorySegment, logical dtype, physical element capacity, and checked byte size. It neither allocates nor copies nor closes the segment. Its constructor requires nonnegative capacity, exact `capacity * byteWidth == segment.byteSize()`, and an initially live scope. Zero capacity requires a live zero-byte segment. It makes no alignment or byte-order promise and does not itself check a Tensor's layout.

The caller must keep the arena alive and obey its thread-access/read-only policy. Metadata remains readable after closure, but accessing the dead segment fails under JDK rules. Tensor association validates compatibility separately; synchronized reference replacement does not synchronize underlying memory writes. Logical element count, referenced layout span, and physical capacity must not be conflated. LearningJourney's input/target/weight values eventually need host values, but operation construction alone does not read them. Device storage and native buffers belong elsewhere; there is no native component in `modules/model`.

## io.github.pho001.synaptik.model.tensor

Important entry points: [Tensor](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java), [TensorFactory](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java), [TensorDescriptor](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java), [TensorProducer](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java), and [TensorProvenance](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProvenance.java).

A Tensor has immutable TensorId, descriptor, diagnostic label, and optional provenance. Its only mutable state is an optional borrowed host-storage association. Object equality is identity equality, not descriptor equality or matching ID text. Factory allocations receive JVM-unique TensorIds; this is a different identity domain from a graph's NodeId/ValueId.

A descriptor contains dtype, Shape, optional resolved layout, and `requiresGrad`. A fully static Shape may still have unresolved layout; a dynamic Shape cannot have a resolved numeric layout. A true gradient request is rejected for a nondifferentiable dtype and never proves an operation has a derivative implementation.

Expression methods validate operand metadata and create storage-free results. A producer owns one exact operation occurrence, ordered input references and output descriptors, and all canonical output wrappers, created together by the factory. Provenance identifies the producer and output slot. Repeated access returns the same wrapper, not a reconstructed equal Tensor. Multi-output results such as TopK, dropout, attention, batch normalization and recurrence retain sibling relationships; keeping one result can keep its siblings alive. These ordinary Java reference cycles own no external resource.

The important distinction is visible in LearningJourney: its prediction expression knows how it was requested even before any numerical prediction exists. Rebinding or clearing host storage cannot rewrite that origin. Tensor owns neither accumulated gradients nor optimizer state. Read [autograd](../../user-guide/autograd.md) for the separate derivative workflow.

### Scan bodies and lexical containment

[ScanBody](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/ScanBody.java), [Scan](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Scan.java), and [ScanResult](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/ScanResult.java) describe fixed-count composition over ordinary Tensor algebra. This is not a callback retained for runtime invocation.

```text
outer occurrence: initial carry + sequences + invariants
                          |
                          v
body scope: carry formal + step formal + invariant formal
                          |
                     ordinary producers
                          |
                    next carry + emissions
                          |
outer results: final carry + stacked emissions + residual ports
```

The diagram separates outer actual bindings from body-local formals. Body leaves must be the exact declared formals by identity; hidden Tensor constants must instead be explicit invariants. Formals are distinct, storage/provenance-free and disjoint from actual bindings. Unused declared formals are allowed; repeated actual identities in different positions are not deduplicated. Free leaves, host-backed constants, mutation, RNG, and nested/recursive bodies reject.

The immutable template may be reused by several scan occurrences, but producers cannot escape into the outer expression or be shared across different lexical templates. Captured body ValueIds are local: the same number in two bodies does not name the same value. Retained residual ports identify semantic body values potentially needed later; they do not allocate saved buffers. `ScanBody.describeOrdinary` checks output descriptors without creating Tensors, IDs or storage. Its descriptor proof is not backend admission or Compiler inference.

Current source admits more PURE algebra/type combinations than every downstream execution or persistence route. Symbolic-batch and length-aware forms have additional exact role/coordinate constraints; the [focused Scan explanation](../../architecture/general-differentiable-scan.md) and [owning contract](../../architecture/contracts/recurrent-scan.md#general-differentiable-scan) separate those domains. None participates in LearningJourney.

## io.github.pho001.synaptik.model.graph

Important entry points: [CompiledGraphModel](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledGraphModel.java), [CompiledNode](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledNode.java), [GraphValue](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/GraphValue.java), [CompiledScanBody](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledScanBody.java), and [ForwardPublicationBinding](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/ForwardPublicationBinding.java).

Compiler emits immutable graph values (ValueId plus descriptor) and topologically ordered computation nodes (NodeId, operation, ordered ports, optional compiled Scan body). The graph snapshots values, nodes, ordered boundaries and per-node forward/backward phases. It validates structural closure: IDs are unique within their domain/scope; every reference resolves; inputs are producer-free; every non-input has one producer; node inputs are available before use; phases cover exactly the nodes. Cycles, later-node dependencies and duplicate producers reject. Repeated node inputs, unused inputs and a zero-node pass-through are legitimate.

Graph structural validation does not infer family descriptors, allocate IDs, capture public Tensors, select backends or build an executable. Forward publication bindings describe the public Tensor/graph-value relationship without embedding mutable public Tensors in graph structure. A compiled Scan body contains the one nested captured graph and ordered role/result bindings; graph outputs are unique even when role bindings intentionally repeat a value.

LearningJourney reaches this layer during compilation, with separate forward and derivative work as needed. Graph IDs describe that capture, not the globally allocated input TensorIds or physical memory slots. See [compiling graphs](../../user-guide/compiling-graphs.md).

## io.github.pho001.synaptik.model.operation

Important entry points: [Operation](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/Operation.java), [OperationKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationKind.java), [OperationAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationAttrs.java), and [OperationSignature](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationSignature.java).

An Operation pairs a typed semantic kind with immutable attributes. A family declares accepted **exact attribute classes** and inclusive input/output count ranges. Kind signature resolution fails closed for undeclared pairings or malformed declarations. Parameterless operations use `NoOperationAttrs.INSTANCE`, never null. A diagnostic enum name is not a kernel name, registry key or wire token; two families may use the same text without becoming the same kind.

There are three different checks: attributes validate their intrinsic parameters; Operation validates kind/attribute pairing; input-aware Tensor methods validate ranks, dtype/Shape compatibility and derive results. Signature cardinality alone cannot establish a valid MATMUL. Attributes contain no Tensor, graph, storage, service or body closure. TensorProducer and CompiledNode supply occurrence identity and ordered operands. These packages describe LearningJourney's transpose, multiplication and loss; execution occurs only in later layers.

## io.github.pho001.synaptik.model.operation.attention

[ScaledDotProductAttentionKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionKind.java) consumes `[query,key,value]` with an optional fourth Boolean mask. It produces attention output and, when explicitly requested, normalized weights as a second canonical output. It contracts query/key embeddings, applies exact mask/causal eligibility, normalizes eligible scores, and aggregates value rows; it is not just an alias for MATMUL.

[ScaledDotProductAttentionAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionAttrs.java) stores an optional finite positive typed scale and causal flag. Absence means `1/sqrt(E)` for positive bound embedding extent. Scale metadata accepts FLOAT64, FLOAT32 or BFLOAT16; Tensor construction owns operand-dependent geometry and compatibility. No eligible positions produce positive-zero output, and eligibility cannot be approximated. The attributes own no cache or saved weights; auxiliary weights are ordinary producer results. Attention is absent from LearningJourney.

## io.github.pho001.synaptik.model.operation.convolution

[Conv2dKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv2dKind.java) and [Conv3dKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv3dKind.java) describe grouped NCHW/NCDHW **cross-correlation**, not a spatially reversed mathematical convolution. Ordered inputs are `[input,weight]` with optional per-output-channel bias. Each output contracts the exact channel/kernel contributors; symmetric padding supplies positive zero.

[Conv2dAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/convolution/Conv2dAttrs.java) and Conv3dAttrs require positive stride, dilation and groups and nonnegative padding. Kernel extents and bias presence remain operand/occurrence facts. Channel divisibility, ranks and output geometry are input-aware checks, not guaranteed by valid attributes. Conv1dAttrs is a convenience translated to singleton-height 2D geometry; no Conv1d occurrence retains it. No weights, workspace or native handle are owned here. This spatial family is not used by LearningJourney's linear layer.

## io.github.pho001.synaptik.model.operation.elementwise.binary

[BinaryArithmeticKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/binary/BinaryArithmeticKind.java) defines ordered two-input ADD, SUB, MUL, DIV, MIN, MAX and POW with no attributes. Tensor construction right-aligns broadcast shapes and promotes same-category operands. Integral ADD/SUB/MUL wrap at their fixed width; DIV/POW remain floating-only. MIN/MAX select using represented values and explicit NaN/signed-zero rules rather than arithmetic tolerance.

Only the request is stored: no broadcast buffer or computed element array exists in the kind. LearningJourney's mathematical error and derivative formulas involve subtraction and multiplication, but the first-class loss may keep its own occurrence instead of materializing this elementary graph.

## io.github.pho001.synaptik.model.operation.elementwise.cast

[CastKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastKind.java), [CastAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastAttrs.java), and [CastValueConversions](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastValueConversions.java) separate a one-input cast request from the scalar conversion oracle. Attributes store only target dtype; source dtype comes from the input descriptor. Every current source/target pair has a defined conversion.

The oracle preserves object identity for same-type ScalarValue conversion, rounds directly to floating targets with ties-to-even, and defines saturation, NaN, infinity, Boolean and integral-width cases. Casts are exact conversions, not floating arithmetic sites: arithmetic denormals-are-zero/flush-to-zero permissions do not apply. Tensor preserves Shape and gradient eligibility only across floating-to-floating casts. The oracle reads no Tensor storage and is not a backend element loop. LearningJourney deliberately stays FLOAT32 and needs no cast.

## io.github.pho001.synaptik.model.operation.elementwise.classification

[FloatingClassificationKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/classification/FloatingClassificationKind.java) describes one-input finite/infinite/NaN tests, each producing same-Shape nondifferentiable BOOL metadata through Tensor construction. It reads the meaning of the stored representation, including subnormals, not an approximate arithmetic predicate. There are no parameters, counters or diagnostic services: these are graph-visible Boolean values, unlike trace events. Nonfloating input rejects at expression construction. LearningJourney has no classification branch.

## io.github.pho001.synaptik.model.operation.elementwise.comparison

[BinaryComparisonKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/comparison/BinaryComparisonKind.java) describes ordered two-input relations with singleton attributes. Tensor methods validate same-category numeric operands, promote/broadcast them and publish BOOL results without gradients. Floating equality compares represented numeric values, not raw bits: signed zeros are equal, NaN is unequal to everything, and ordered relations with NaN are false. No epsilon is available.

These semantics differ from ScalarValue metadata equality and ordering-family winner rules. The kind retains no comparator or operand storage. LearningJourney's arithmetic objective does not use Boolean comparisons.

## io.github.pho001.synaptik.model.operation.elementwise.logical

[BooleanLogicalKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/logical/BooleanLogicalKind.java) gives two-input AND/OR and one-input NOT exact truth-table meaning. Tensor construction requires BOOL, broadcasts binary operands and creates nondifferentiable BOOL results. Numeric truthiness requires an explicit cast rather than implicit logical coercion. There is no mutable Boolean state or short-circuit control-flow body here. LearningJourney does not use logical operations.

## io.github.pho001.synaptik.model.operation.elementwise.scalar

[ScalarElementwiseKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarElementwiseKind.java) refines one Tensor input with [ScalarValueAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarValueAttrs.java) or [ClampRangeAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ClampRangeAttrs.java). The scalar is metadata, not a second Tensor leaf. Exact matching scalar dtype avoids implicit promotion; Java-double conveniences mean FLOAT64, which matters for a FLOAT32 receiver.

Range clamp means ordered `MIN(MAX(input,min),max)` but remains one occurrence. Bounds must have equal numeric dtype and not be strictly inverted; NaN endpoints and either signed-zero ordering are structurally accepted under the stated comparison rule. DIV, POW and range CLAMP expression forms are floating-only, while selected scalar arithmetic also accepts integrals. No scalar buffer is allocated by the attrs. This family can explain elementary learning formulas, but SGD's learning-rate configuration is not itself a scalar Tensor operation.

## io.github.pho001.synaptik.model.operation.elementwise.selection

[WhereSelectionKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/selection/WhereSelectionKind.java) defines ordered `[condition,trueBranch,falseBranch]` inputs and one output. Tensor methods require BOOL condition, floating branches, branch promotion and compatible combined broadcasting. Gradient eligibility comes from branches, not the predicate.

WHERE chooses exactly before any required cast; same-type selection preserves chosen bits, including NaN payloads and subnormals. It neither specifies eager versus lazy evaluation nor creates a conditional graph region. It owns no branch closure. LearningJourney has no masked selection.

## io.github.pho001.synaptik.model.operation.elementwise.unary

[UnaryElementwiseKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKind.java) gives nineteen parameterless one-input floating transforms/activations their formulas. Tensor construction preserves logical Shape/type and eligibility; the enum does not install derivative rules. ABS/NEG sign operations and SIGN/FLOOR/CEIL/RELU have exact represented-value rules. LOG1P and EXPM1 are first-class targets, not separately rounded `LOG(1+x)` or `EXP(x)-1` graphs.

Numerical domains and infinity/NaN/zero behavior belong to each formula, not construction-time evaluation of stored input values. No activation array or cache is owned here. LearningJourney is deliberately bias-free and activation-free, so this family is outside its forward path.

## io.github.pho001.synaptik.model.operation.index

Important entry points: [AxisGatherKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisGatherKind.java), [AxisScatterKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisScatterKind.java), [GatherNdKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/GatherNdKind.java), [ScatterNdKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterNdKind.java), and [ScatterReduction](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterReduction.java).

Gather consumes `[data,indices]`; scatter consumes `[data,indices,updates]` and returns a new value rather than mutating data. Axis gather replaces one data axis with the indices Shape; gather-elements aligns same-rank coordinates. ND indexing uses a shared batch prefix B and final indices tuple depth K: output/update Shape is `indices[:-1] + data[B+K:]`. Scalar SELECT instead stores one normalized axis/index and removes that axis. ONE_HOT adds a positive-depth final BOOL axis to signed-integral indices.

Attributes check intrinsic nonnegative axes/batch counts and positive depth, but cannot know input rank or index values. Tensor construction requires INT32/INT64 indices and checks structural Shape/type relationships. Runtime-value bounds and duplicate replacement targets need later value-aware validation. NONE replacement requires unique targets; reductions combine the base and every addressed update once, including duplicate contributions. Unaddressed coordinates preserve the original base. Selected raw payloads and mappings remain exact. This package owns no index buffer or atomic-update implementation, and LearningJourney does not index or scatter.

## io.github.pho001.synaptik.model.operation.layout

Important entry points: [AxisTransformKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/AxisTransformKind.java), [ShapeTransformKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/ShapeTransformKind.java), [SliceAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/SliceAttrs.java), [WindowTransformKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/WindowTransformKind.java), and [ContiguousKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/ContiguousKind.java).

This package describes **changes of logical mapping**, unlike `model.layout`, which describes resolved geometry. PERMUTE reorders axes; rank-two transpose is `[1,0]`, the mapping used by LearningJourney. EXPAND_DIMS/SQUEEZE insert/remove singleton axes. RESHAPE preserves logical element order; EXPAND repeats singleton coordinates. CONCAT joins an ordered nonempty input list along an existing axis; STACK introduces a new one. PAD inserts exact typed constants; TILE repeats whole patterns rather than individual cells.

Intrinsic guards matter: permutations are complete and duplicate-free; padding widths are nonnegative and paired; tile counts are positive. Input-aware methods normalize caller axes and validate rank/Shape compatibility. TargetShapeAttrs stores normalized Shape, not an unresolved `-1` request. Expansion can retain target-one-or-equal binding obligations. CONTIGUOUS is a materialization request, not merely the `DENSE_CONTIGUOUS` geometric classifier.

SLICE stores finite normalized sequences `start + k*step`, with nonnegative starts/lengths, distinct axes and signed nonzero steps. Empty entries have canonical start zero. Negative logical traversal is valid even though resolved physical LayoutDescriptor strides cannot be negative. Whether a view can be proved or materialization is needed is separate from the slice request. SLICE_UPDATE is functional replacement; CropToShapeAttrs can express a symbolic region by target and prefix Shapes without storing a runtime coordinate.

UNFOLD_AXIS replaces an axis with window positions and appends window size; FOLD_AXIS scatters overlap sums into an explicit output extent. Image unfold/fold uses NCHW or NCDHW canonical columns. Windows require positive kernels/strides/dilations and nonnegative padding; floor versus literal ceil geometry affects terminal samples. Fold sums overlaps without dividing by coverage; explicit output size is necessary because trailing uncovered positions cannot be recovered from columns. These immutable parameters own no aliased memory or im2col workspace. Other than transpose, these operations are absent from the canonical forward path.

## io.github.pho001.synaptik.model.operation.linalg

[MatmulKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/linalg/MatmulKind.java) declares two ordered inputs and one output with no intrinsic attrs. It contracts the left final axis with the right final axis for a vector or penultimate axis for a matrix, promotes vector ranks temporarily, and broadcasts leading batch dimensions. Tensor construction validates ranks, floating promotion and contraction compatibility.

Every mapped product participates once in the declared contraction. FLOAT64 uses FLOAT64 working accumulation; FLOAT32 and low floating results use FLOAT32, with final low narrowing where applicable. This does not select a BLAS call, tile shape, native library or backend. LearningJourney's `[2,1] @ [1,1]` is the small rank-two instance, yielding two rows with one prediction each.

## io.github.pho001.synaptik.model.operation.loss

[LossKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/LossKind.java), [MeanSquaredErrorAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/MeanSquaredErrorAttrs.java), and [LossReduction](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/LossReduction.java) describe ordered prediction/target objectives, not a training session. MSE computes one subtraction and squared delta per logical position, then explicit NONE, SUM or MEAN reduction. LearningJourney uses SUM, so its two squared errors produce 40, not the mean 20.

Dense/index categorical cross-entropy operates directly on logits using guarded log-sum-exp semantics, not a request to first round softmax probabilities. Class axes are normalized metadata. Index targets use signed integral indices; optional typed ignore-index matches before bounds or logits evaluation, and mean divides by nonignored target count. An empty/all-ignored sum is positive zero and mean is NaN. Attributes cannot know class extent or target bounds; expression and execution boundaries own those checks. No target values, accumulated loss, gradients or optimizer resources are retained by this family.

## io.github.pho001.synaptik.model.operation.normalization

Important entry points: [SoftmaxKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/SoftmaxKind.java), [LayerNormKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/LayerNormKind.java), [RmsNormKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/RmsNormKind.java), and [BatchNormKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/BatchNormKind.java).

Softmax/log-softmax normalize one axis without reducing Shape. All-finite slices use maximum shifts; log-softmax is not the logarithm of already rounded probabilities. Layer norm standardizes exact trailing axes using population variance and epsilon inside sqrt, with optional explicit scale/bias. RMS norm uses uncentered mean square, optionally scale. Attributes require nonempty normalized Shape rank and finite positive typed epsilon; matching actual trailing extents is an input-aware proof.

Batch normalization consumes `[input,scale,bias,runningMean,runningVariance]`. Inference returns one output. Training returns five ordinary outputs, including next running statistics and saved batch values, rather than mutating hidden statistics. Momentum is the new-batch weight and must be finite in `[0,1]`; epsilon is positive finite. Channel axes are logical, not hard-coded layout positions. Exact scalar metadata/type compatibility is checked by the owning expression boundary. Model attrs own no mutable running state or saved-buffer lifetime; NN/training and later execution layers do. LearningJourney contains no normalization or running-statistic transition.

## io.github.pho001.synaptik.model.operation.ordering

[OrderingKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/ordering/OrderingKind.java), [SortAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/ordering/SortAttrs.java), [TopKKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/ordering/TopKKind.java), and [TopKAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/ordering/TopKAttrs.java) describe axis-local value ordering and original logical indices. SORT/ARGSORT each have one output; TopK has canonical value and index outputs from one occurrence.

Stable equal keys retain increasing logical input-index order; NaNs are last in either direction, and signed zeros have a defined order. These decisions use represented values without epsilon. Attributes require normalized nonnegative axis and nonnegative k; Tensor construction checks rank and k against known extent. Selection does not choose a sort implementation or own scratch memory. Index outputs are nondifferentiable INT64, not storage addresses. LearningJourney needs no ranking.

## io.github.pho001.synaptik.model.operation.pooling

[Pool2dKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/Pool2dKind.java), [Pool3dKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/pooling/Pool3dKind.java), and their distinct max/average attrs describe one floating input and one spatial output. Positive kernels/strides/dilations, nonnegative symmetric padding and literal ceil mode determine exact sample membership. The 1D attrs translate to singleton-height 2D occurrences rather than a separate Pool1d kind.

Max pooling excludes padding, prefers NaN, orders positive zero above negative zero and retains the first equal logical sample. An all-padding window yields negative infinity, or canonical NaN for E4M3. Average pooling counts every kernel position in a fixed divisor, with out-of-bounds positive-zero contributions; it has no divisor override. Thus max and average cannot share an inferred generic padding identity. No winner cache or pooling workspace is owned by these attrs. LearningJourney uses no spatial windows.

## io.github.pho001.synaptik.model.operation.random

[GraphRngKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/GraphRngKind.java), [GraphRngStateAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/GraphRngStateAttrs.java), [DropoutKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/DropoutKind.java), and [DropoutAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/DropoutAttrs.java) make random state explicit in provenance. An initial-state occurrence records exact unsigned key/counter words, exposed through the opaque GraphRngState carrier rather than an ordinary numerical Tensor API.

Dropout consumes `[input,state]` and has `[output,keepMask,nextState]` canonical outputs; the keep mask is compiler-facing auxiliary state. Probability must be finite in `[0,1)`. One abstract draw per logical element advances the counter modulo `2^64`; probability zero still advances it. Dropped values are positive zero, kept values use inverse-keep scaling. The immutable attrs hold neither a mutable generator nor a chosen sampling algorithm. This effect-like explicit state is not admitted inside a pure general Scan body. LearningJourney supplies a deterministic zero weight and uses no graph RNG/dropout.

## io.github.pho001.synaptik.model.operation.recurrent

[RecurrentScanKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/recurrent/RecurrentScanKind.java) and [RecurrentDirection](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/recurrent/RecurrentDirection.java) describe fixed RNN_TANH, GRU_RESET_AFTER and LSTM equations. They consume explicit sequences, valid lengths, initial state, packed weights and optional input-side bias; result carriers expose dense outputs and final hidden state, plus final cell for LSTM. Gate packing and equations are fixed, not a configurable cell callback.

FORWARD/REVERSE traverse each row's valid prefix; reverse does not traverse padding or reverse dense output coordinates. Length bounds `[0,time]` require runtime values. All carried state is explicit and no parameter, hidden mutation or persistent recurrent workspace lives in Model. Crucially, this is one flat multi-output producer/node, not the body-bearing Scan described below. LearningJourney's linear layer has no recurrent state. See the [fixed-family contract](../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions) for exact equations and empty-prefix behavior.

## io.github.pho001.synaptik.model.operation.reduction

[AggregateReductionKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/AggregateReductionKind.java), [MultiAxisReductionAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/MultiAxisReductionAttrs.java), [StatisticalReductionAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/StatisticalReductionAttrs.java), and [SumToShapeAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/SumToShapeAttrs.java) describe combining exact contributor domains. Full reductions use no attrs; axis/multi-axis forms store normalized axes and retention flags. An empty axis list means a point domain, not all axes. Lists are immutable snapshots and reject negative/duplicate axes.

Numeric SUM/PROD/MIN/MAX and Boolean ALL/ANY have different type and empty-identity rules. Masked SUM/MEAN consumes `[input,mask]`; the BOOL mask must broadcast exactly to input Shape, and false entries exclude even NaN/infinity before arithmetic. Statistical correction is nonnegative and requires `N > correction`; a provably invalid static domain rejects while dynamic proof may defer. Arg-extrema produces INT64 indices with explicit first/last logical-index tie policy, preferring NaNs and distinguishing zero signs; a statically empty selected axis rejects.

SUM_TO_SHAPE right-aligns an exact target: remove leading axes, reduce aligned target-one axes and preserve equal axes. It retains binding obligations rather than guessing axes for unresolved dimensions. The package owns semantic parameters, not accumulation arrays or a chosen reduction tree. LearningJourney's SUM objective uses this mathematics through the loss family; derivative construction may also need reductions to reverse broadcasts.

## io.github.pho001.synaptik.model.operation.scan

[CumulativeScanKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/CumulativeScanKind.java) and [CumulativeScanAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/CumulativeScanAttrs.java) describe same-Shape cumulative sum/product along one axis. Exclusive mode omits the current position; reverse changes traversal, not output coordinates. For input `[1,2,3]`, sum gives `[1,3,6]`, exclusive sum `[0,1,3]`, reverse sum `[6,5,3]`, and exclusive reverse sum `[5,3,0]`. These are formula examples, not runtime output collected here. A zero-length axis has no position to receive an identity.

[ScanKind](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/ScanKind.java) and [ScanAttrs](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/scan/ScanAttrs.java) instead describe the bounded general body-bearing scan. Attrs contain scalar time/count/direction/length-mode/row-role header metadata only. They validate counts and role-header consistency with checked count arithmetic; the actual source body belongs on TensorProducer and the captured body on CompiledNode, never inside attrs or a registry. Source body scoping is explained in the tensor section above.

A valid header neither executes iterations nor allocates residual storage. Compiler, Prepare and backend each independently admit the actual body, types and layouts. Neither cumulative nor general Scan participates in LearningJourney; its two input rows are a matrix batch, not two recurrent time steps.

## Where to go next

- [Tensor guide](../../user-guide/tensors.md): creation, metadata and expression use.
- [Declarative models](../../user-guide/declarative-models.md): the higher-level builder used by the learning program.
- [Autograd](../../user-guide/autograd.md): derivative ownership beyond `requiresGrad`.
- [General differentiable Scan](../../architecture/general-differentiable-scan.md): body roles and bounded support.
- [Package atlas index](index.md) and [reference entry point](../../reference/index.md): adjacent layers and API navigation.

This atlas accounts for all 28 Model packages. It introduces no operation capability or public API change; source links and paper arithmetic explain existing meaning, not an independently exercised backend route.
