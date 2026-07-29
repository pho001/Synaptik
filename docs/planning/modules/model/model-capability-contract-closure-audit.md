# Model Capability and Contract Closure Audit

## Executive conclusion and closure verdict

**Verdict: `BLOCKING_GAP`.**

The audit completed and found one blocking documentation-contract gap in a review-only Java path:
[`GraphValue`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/GraphValue.java)
calls the already-implemented public mutable `Tensor` API “planned.” The wording contradicts the
current source and architecture invariant. Task 0024 cannot correct Java/Javadoc, so the selected
`modules/model` milestone remains open for one bounded follow-up.

Apart from that stale Javadoc sentence, current source provides a coherent backend-independent tensor model, exact operation occurrence
signatures, public expression construction for the selected inference/training baseline, dynamic
Shape metadata, typed scalar attributes, shared multi-output provenance, and immutable graph
contracts. No behavioral model gap or architecture decision remains.

This verdict does not claim graph capture, automatic differentiation (autograd), backend support,
kernels, numerical execution, prepared state, runtime residency, or general tensor-library
completeness. Those are later-layer responsibilities. The closure test is:

```text
selected model meaning + honest metadata + valid occurrence construction
  = closed model representation

compiler + planning + prepare + backend + runtime
  = downstream implementation, not implied by this verdict
```

The final checkpoint evidence and the bounded follow-up appear in the last section.

## Authority, scope, and method

[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative. This audit is a planning result,
not a new architecture rule or API contract. It applied the General, Planning, and API/Javadoc
documentation profiles and task [0024](tasks/0024-model-capability-and-contract-closure-audit.md).

Current source and tests were primary evidence. Completed tasks and the
[adjoint-expressibility result](adjoint-expressibility-audit.md) supplied decision history only.
The inventory commands were:

```bash
rg --files --hidden modules/model/src/main/java | sort
rg --files --hidden modules/model/src/test/java | sort
javap -classpath modules/model/build/classes/java/main -public \
  io.github.pho001.synaptik.model.tensor.Tensor
javac -cp modules/model/build/classes/java/main -d /tmp /tmp/ModelAudit.java
java -cp /tmp:modules/model/build/classes/java/main ModelAudit \
  modules/model/build/classes/java/main
rg -l 'implements OperationAttrs' modules/model/src/main/java | sort
rg -l 'implements OperationKind' modules/model/src/main/java | sort
rg -n 'new Operation|TensorFactory\.createDerived' \
  modules/model/src/main/java/io/github/pho001/synaptik/model/tensor
```

The sorted inventory contained 176 production Java files and 129 test-tree paths: 128 Java files
plus one `.gitkeep`, 305 paths total. Every path in both lists was reviewed by package and
responsibility; focused source/tests were then read
for every foundation, operation family, expression helper, producer, and result carrier. The
reflection helper scanned every top-level compiled model class rather than using a hand-maintained
kind registry. Temporary helpers remained under `/tmp`.

The audit changed documentation/planning only. It did not change Java, tests, Gradle, architecture,
architecture tests, another module, conformance tests, or integration tests.

## Architecture-boundary assessment

The source matches the model boundary in the architecture contract:

| Check | Evidence | Conclusion |
|---|---|---|
| Model semantics are backend-independent | [`Operation`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/Operation.java), [`OperationKind`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationKind.java), and an import scan of all 176 production files | No backend identity, support query, lowering, kernel, prepare, or execution route occurs in model semantics. |
| `Tensor` is public mutable API state, not graph IR | [`Tensor`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java), [`TensorProducer`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java), and [`CompiledNode`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledNode.java) | Tensor identity/storage association and pre-capture provenance remain distinct from graph-local node/value identity. |
| Runtime hot-path types remain outside model | Production inventory and architecture-test source scan | No runtime, prepared-execution, residency, physical-buffer, or workspace type appears under `modules/model`. |
| Compiler/autograd ownership is preserved | [Compile API](../../../api/compile-api.md), [Training API](../../../api/training-api.md), and the [adjoint audit](adjoint-expressibility-audit.md) | Model exposes forward semantics and reusable primitives; compiler remains the owner of capture, gradient rules, saved-value lifetime, and backward construction. |
| Dependency direction is preserved | Model source-import scan plus the final architecture-test checkpoint | `modules/model` does not import planning, compiler, prepare, runtime, engine, concrete backends, or extensions. |

The explanatory architecture pages and architecture tests required no change because no ownership,
dependency, or lifecycle rule changed. The audit found no conflict requiring an ADR or edit to
`ARCHITECTURE.md`.

## Foundation-contract inventory

The table records readiness rather than every accessor. Source links identify the owning contract;
the matching package tests are included in the 128-file test inventory.

| Foundation | Current contract and readiness |
|---|---|
| Data types and promotion | [`DataType`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataType.java) defines FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and BOOL; [`DataTypePromotion`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataTypePromotion.java) keeps floating and signed-integral promotion explicit. Ready for the selected baseline. |
| Typed scalar values | [`ScalarValue`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/ScalarValue.java) is the sole exact scalar-attribute carrier for all six types, preserving raw floating/BFLOAT16 bits and exact integral/BOOL values. Ready. |
| Dimensions and Shapes | [`Dimension`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/Dimension.java), [`StaticDimension`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/StaticDimension.java), [`DynamicDimension`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DynamicDimension.java), [`ExpressionDimension`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/ExpressionDimension.java), and [`Shape`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/Shape.java) represent static, named, canonical expression, and constrained-unknown extents without runtime binding. Ready. |
| Layouts and descriptors | [`LayoutDescriptor`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/layout/LayoutDescriptor.java), [`LayoutGeometry`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/layout/LayoutGeometry.java), and [`TensorDescriptor`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java) distinguish logical metadata from optional representable physical geometry. Ready; negative-stride results honestly remain layout-unresolved. |
| Typed identifiers | [`TensorId`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorId.java), [`NodeId`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/NodeId.java), and [`ValueId`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/ValueId.java) remain separate domains. Ready. |
| Tensor identity and host storage | [`Tensor`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java), [`HostTensorStorage`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/storage/HostTensorStorage.java), and [`MemorySegmentStorage`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/storage/MemorySegmentStorage.java) provide mutable borrowed host-storage association without device residency. Ready. |
| Eager construction | [`TensorFactory`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java) owns identity, allocation, imports, and constants; [`TensorRandoms`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRandoms.java) owns explicit-source eager random leaves; package-private [`TensorRanges`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRanges.java) owns integer-range mechanics. Ready. |
| Producer/provenance | [`TensorProducer`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java) owns one pre-capture occurrence and ordered descriptors; [`TensorProvenance`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProvenance.java) selects one zero-based output. Ready. |
| Immutable graph model | [`GraphValue`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/GraphValue.java), [`CompiledNode`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledNode.java), [`CompiledGraphModel`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledGraphModel.java), [`GraphPhase`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/GraphPhase.java), and the current [`ForwardPublicationBinding`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/ForwardPublicationBinding.java) provide structurally validated compile-time model values without implementing compilation. The later terminology correction from the historical `PublicationBinding` name does not change this audit conclusion. Behavior is ready; `GraphValue` has one stale public-Tensor status sentence that blocks documentation closure. |
| Public multi-output boundaries | [`DropoutResult`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/DropoutResult.java), [`BatchNormTrainingResult`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/BatchNormTrainingResult.java), [`TopKResult`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TopKResult.java), and [`ScaledDotProductAttentionResult`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/ScaledDotProductAttentionResult.java) expose operation-specific roles without a generic tuple hierarchy. Ready. |

## Public Tensor and construction inventory

Java 26 reflection found exactly **200 declared public `Tensor` methods**. The following groups
contain every exact signature; array parameters shown below correspond to Java varargs where the
source declares them. Category subtotals sum to 200.

| Owning family or documented convenience | Exact signatures, grouped by method name | Count |
|---|---|---:|
| Identity, descriptor, provenance, storage, diagnostics | `id()`, `descriptor()`, `label()`, `provenance()`, `hostStorage()`, `replaceHostStorage(HostTensorStorage)`, `clearHostStorage()`, `toString()` | 8 |
| Seven binary/scalar arithmetic families | `add/sub/mul/div/minimum/maximum/pow` each with `(Tensor)`, `(ScalarValue)`, `(double)` | 21 |
| Range and one-bound clamp | `clamp(ScalarValue,ScalarValue)`, `clamp(double,double)`, `clampMin(ScalarValue)`, `clampMin(double)`, `clampMax(ScalarValue)`, `clampMax(double)` | 6 |
| Unary numeric and activation primitives | `abs()`, `neg()`, `reciprocal()`, `log()`, `log1p()`, `exp()`, `expm1()`, `erf()`, `sqrt()`, `rsqrt()`, `floor()`, `ceil()`, `sign()`, `relu()`, `sigmoid()`, `tanh()`, `gelu()`, `geluTanhApproximation()`, `silu()` | 19 |
| Floating classification and cast | `isFinite()`, `isNaN()`, `isInf()`, `cast(DataType)` | 4 |
| Comparison, logical, and selection | Six `(Tensor)` comparisons; `logicalAnd(Tensor)`, `logicalOr(Tensor)`, `logicalNot()`; static `where(Tensor,Tensor,Tensor)` | 10 |
| Aggregate, statistical, and arg reductions | `sum` 6 forms; `mean` 6; `prod/min/max/all/any` 5 each; `sumToShape(Shape)`; `logSumExp` 2; `variance` 3; `standardDeviation` 3; `l1Norm/l2Norm` 2 each; `argMin/argMax` 3 each | 56 |
| Cumulative scans | `cumSum(int)`, `cumSum(int,boolean,boolean)`, `cumProd(int)`, `cumProd(int,boolean,boolean)` | 4 |
| Softmax normalization | `softmax(int)`, `logSoftmax(int)` | 2 |
| Layout and rank transformations | `contiguous()`; `reshape(long[])`, `reshape(Shape)`; `expand(long[])`, `expand(Shape)`; `expandDims(int)`, `squeeze(int)`, `permute(int[])`, `transpose()` | 9 |
| Slice and placement transformations | `slice(long[],long[],int[],long[])`; two `sliceAxis` forms; `flip(int[])`; `sliceUpdate(Tensor,long[],int[],long[])`; `cropToShape(Shape,Shape)` | 6 |
| Indexing and indexed update | `select`; `gather`; `embedding`; `oneHot`; `gatherElements`; two `gatherNd`; three `scatterNd`; two `scatterElements`; `scatterAdd` | 13 |
| Composition, padding, tiling, windows | Static `concat` and `stack`; `unstack`; two `pad`; `tile`; `unfold`; `foldAxis`; two `unfold2d`; `fold2d` | 11 |
| Linear algebra and linear convenience | `matmul(Tensor)`, `linear(Tensor)`, `linear(Tensor,Tensor)` | 3 |
| Ordering and top-K | Two `sort`; two `argsort`; two `topK` | 6 |
| Convolution and pooling | Two `conv2d`; `maxPool2d`; `averagePool2d` | 4 |
| Normalization | Two `layerNorm`; two `rmsNorm`; `batchNormInference`; `batchNormTraining` | 6 |
| Losses | `meanSquaredError`; two `categoricalCrossEntropyWithLogits` | 3 |
| Attention | Four `scaledDotProductAttention`; four `scaledDotProductAttentionWithWeights` | 8 |
| Explicit graph RNG/dropout | `dropout(double,GraphRngState)` | 1 |
| **Total** | Reflection over `Tensor.getDeclaredMethods()` with `public` filter | **200** |

Direct one-occurrence construction is package-owned by focused `Tensor*Expressions` helpers plus
`GraphRngState`; all reach package-private
`TensorFactory.createDerived` or `createDerivedOutputs`, which creates `TensorProducer` and
therefore validates selected signature counts before allocating result identities. Source search
found 58 `new Operation(...)` sites and 51 derived-factory call sites. Repeated family helpers
account for the difference.

Conveniences preserve visible primitive ownership:

| Convenience | Exact producer chain |
|---|---|
| `clampMin` / `clampMax` | Scalar `MAX` / scalar `MIN` |
| `transpose` | `PERMUTE[1,0]` |
| `linear` | `PERMUTE -> MATMUL -> optional ADD` |
| `embedding` | `GATHER(axis=0)` |
| `flip` | One signed `SLICE` |
| `unstack` | Ordered independent `SELECT` occurrences |

No convenience hides a fused or backward-only kind. All public eager leaves are provenance-free;
all direct expression results are storage-free and have validated provenance.

## Operation kind, attributes, and signature inventory

Reflection found **37 concrete kind enum types, 107 constants, 47 concrete `OperationAttrs`
implementations, and 127 signatures**. Every attributes implementation appears in at least one
signature. Every constant returns the same non-empty immutable list across calls, each list has
one variant per exact attributes class, every range satisfies `0 <= minInputs <= maxInputs` and
`1 <= minOutputs <= maxOutputs`, and mutation attempts fail. `Operation.signatureFor` performs
exact-class matching, so no permissive kind/attributes pairing exists.

Notation in the table is `Attrs inputs -> outputs`; `1..N` means one through
`Integer.MAX_VALUE` inputs.

| Kind family | Constants and exact accepted signatures | Construction owner |
|---|---|---|
| `ScaledDotProductAttentionKind` | `SCALED_DOT_PRODUCT_ATTENTION`: `ScaledDotProductAttentionAttrs 3..4 -> 1..2` | `TensorScaledDotProductAttentionExpressions` |
| `Conv2dKind` | `CONV2D`: `Conv2dAttrs 2..3 -> 1` | `TensorConv2dExpressions` |
| `BinaryArithmeticKind` | `ADD, SUB, MUL, DIV, MIN, MAX, POW`: `NoOperationAttrs 2 -> 1` | `TensorBinaryExpressions` |
| `CastKind` | `CAST`: `CastAttrs 1 -> 1` | `TensorCastExpressions` |
| `FloatingClassificationKind` | `IS_FINITE, IS_NAN, IS_INF`: `NoOperationAttrs 1 -> 1` | `TensorFloatingClassifications` |
| `BinaryComparisonKind` | `GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL, EQUAL, NOT_EQUAL`: `NoOperationAttrs 2 -> 1` | `TensorComparisonExpressions` |
| `BooleanLogicalKind` | `AND, OR`: `NoOperationAttrs 2 -> 1`; `NOT`: `NoOperationAttrs 1 -> 1` | `TensorLogicalExpressions` |
| `ScalarElementwiseKind` | `ADD, SUB, MUL, DIV, MIN, MAX, POW`: `ScalarValueAttrs 1 -> 1`; `CLAMP`: `ClampRangeAttrs 1 -> 1` | `TensorScalarExpressions` |
| `WhereSelectionKind` | `WHERE`: `NoOperationAttrs 3 -> 1` | `TensorWhereExpressions` |
| `UnaryElementwiseKind` | `ABS, NEG, RECIPROCAL, LOG, LOG1P, EXP, EXPM1, ERF, SQRT, RSQRT, FLOOR, CEIL, SIGN, RELU, SIGMOID, TANH, GELU, GELU_TANH_APPROXIMATION, SILU`: `NoOperationAttrs 1 -> 1` | `TensorUnaryExpressions` |
| `AxisGatherKind` | `GATHER, GATHER_ELEMENTS`: `IndexAxisAttrs 2 -> 1` | `TensorAxisGatherExpressions` |
| `AxisScatterKind` | `SCATTER_ELEMENTS`: `ScatterElementsAttrs 3 -> 1`; `SCATTER_ADD`: `IndexAxisAttrs 3 -> 1` | `TensorAxisScatterExpressions` |
| `GatherNdKind` | `GATHER_ND`: `GatherNdAttrs 2 -> 1` | `TensorGatherNdExpressions` |
| `OneHotKind` | `ONE_HOT`: `OneHotAttrs 1 -> 1` | `TensorOneHotExpressions` |
| `ScatterNdKind` | `SCATTER_ND`: `ScatterNdAttrs 3 -> 1` | `TensorScatterNdExpressions` |
| `SelectKind` | `SELECT`: `SelectAttrs 1 -> 1` | `TensorSelectExpressions` |
| `AxisTransformKind` | `PERMUTE`: `PermutationAttrs 1 -> 1`; `EXPAND_DIMS, SQUEEZE`: `AxisTransformAttrs 1 -> 1` | permutation/rank-edit helpers |
| `ContiguousKind` | `CONTIGUOUS`: `NoOperationAttrs 1 -> 1` | `TensorContiguousExpressions` |
| `PadKind` | `PAD`: `PadAttrs 1 -> 1` | `TensorPadTileExpressions` |
| `ShapeTransformKind` | `RESHAPE, EXPAND`: `TargetShapeAttrs 1 -> 1` | reshape/expand helpers |
| `SliceKind` | `SLICE`: `SliceAttrs 1 -> 1` and `CropToShapeAttrs 1 -> 1`; `SLICE_UPDATE`: `SliceAttrs 2 -> 1` | slice/slice-placement helpers |
| `TensorCompositionKind` | `CONCAT, STACK`: `CompositionAxisAttrs 1..N -> 1` | `TensorCompositionExpressions` |
| `TileKind` | `TILE`: `TileAttrs 1 -> 1` | `TensorPadTileExpressions` |
| `WindowTransformKind` | `UNFOLD_AXIS`: `UnfoldAxisAttrs 1 -> 1`; `FOLD_AXIS`: `FoldAxisAttrs 1 -> 1`; `UNFOLD2D`: `Window2dAttrs 1 -> 1` and `Unfold2dAttrs 1 -> 1`; `FOLD2D`: `Fold2dAttrs 1 -> 1` | `TensorWindowExpressions` |
| `MatmulKind` | `MATMUL`: `NoOperationAttrs 2 -> 1` | `TensorMatmulExpressions` |
| `LossKind` | `MEAN_SQUARED_ERROR`: `MeanSquaredErrorAttrs 2 -> 1`; `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`: `DenseCategoricalCrossEntropyWithLogitsAttrs 2 -> 1`; `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`: `IndexCategoricalCrossEntropyWithLogitsAttrs 2 -> 1` | `TensorLossExpressions` |
| `BatchNormKind` | `BATCH_NORM_INFERENCE`: `BatchNormInferenceAttrs 5 -> 1`; `BATCH_NORM_TRAINING`: `BatchNormTrainingAttrs 5 -> 5` | inference/training helpers |
| `LayerNormKind` | `LAYER_NORM`: `LayerNormAttrs 1 -> 1` and `AffineLayerNormAttrs 3 -> 1` | `TensorLayerNormExpressions` |
| `RmsNormKind` | `RMS_NORM`: `RmsNormAttrs 1..2 -> 1` | `TensorRmsNormExpressions` |
| `SoftmaxKind` | `SOFTMAX, LOG_SOFTMAX`: `SoftmaxAttrs 1 -> 1` | `TensorSoftmaxExpressions` |
| `OrderingKind` | `SORT, ARGSORT`: `SortAttrs 1 -> 1` | `TensorSortExpressions` |
| `TopKKind` | `TOP_K`: `TopKAttrs 1 -> 2` | `TensorTopKExpressions` |
| `Pool2dKind` | `MAX_POOL2D`: `MaxPool2dAttrs 1 -> 1`; `AVERAGE_POOL2D`: `AveragePool2dAttrs 1 -> 1` | pool helpers |
| `DropoutKind` | `DROPOUT`: `DropoutAttrs 2 -> 3` | `TensorDropoutExpressions` |
| `GraphRngKind` | `INITIAL_STATE`: `GraphRngStateAttrs 0 -> 1` | public `GraphRngState.initial`, not a public Tensor method |
| `AggregateReductionKind` | `SUM`: `NoOperationAttrs`, `AxisReductionAttrs`, `MultiAxisReductionAttrs`, `MaskedReductionAttrs 2 -> 1`, `SumToShapeAttrs`; `MEAN`: first four except sum-to-Shape; `PROD, MIN, MAX, ALL, ANY`: ordinary/axis/multi-axis variants; `ARG_MAX, ARG_MIN`: `ArgExtremaAttrs`; `LOG_SUM_EXP, L1_NORM, L2_NORM`: `MultiAxisReductionAttrs`; `VARIANCE, STANDARD_DEVIATION`: `StatisticalReductionAttrs`. Unmarked variants are `1 -> 1`. | reduction, masked, multi-axis, arg-extrema, and sum-to-Shape helpers |
| `CumulativeScanKind` | `CUM_SUM, CUM_PROD`: `CumulativeScanAttrs 1 -> 1` | `TensorCumulativeScanExpressions` |

Every kind except `GraphRngKind.INITIAL_STATE` has a public Tensor construction path.
`INITIAL_STATE` deliberately has the public opaque `GraphRngState.initial` owner because exposing
its state lanes as a general numerical Tensor would weaken the selected state-threading boundary.
There is no unowned semantic-only kind.

## Multi-output occurrence inventory

These are all genuine multi-output forms. Each successful call creates one producer, ordered
descriptors, fresh wrapper identities for every slot, and provenance indices matching the table.

| Occurrence | Ordered roles and descriptors | Public boundary and sharing proof |
|---|---|---|
| `DROPOUT` | `0=output` (input floating type/Shape/eligibility), `1=keep mask` (BOOL, input Shape, non-gradient), `2=next state` (INT64 `Shape[2]`, non-gradient); all unresolved layout | [`TensorDropoutExpressions`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDropoutExpressions.java) creates all three under one producer. `DropoutResult` exposes slot 0 and opaque slot 2; slot 1 is hidden but producer-described for compiler-owned backward capture. |
| `BATCH_NORM_TRAINING` | `0=output` (input Shape), `1=next running mean`, `2=next running variance`, `3=saved batch mean`, `4=saved inverse standard deviation`; statistic slots use rank-one channel Shape and promoted floating type | [`TensorBatchNormTrainingExpressions`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormTrainingExpressions.java) creates five slots. `BatchNormTrainingResult` exposes 0–2; slots 3–4 remain hidden compiler-facing saved values. |
| `TOP_K` | `0=values` (input type/eligibility), `1=indices` (INT64/non-gradient); both share the Shape with selected axis replaced by static `k` | [`TensorTopKExpressions`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorTopKExpressions.java) and `TopKResult` expose both slots from one producer. |
| Attention with weights | `0=output` (promoted type, `[...,L,Ev]`, query/key/value eligibility OR), `1=weights` (promoted type, `[...,L,S]`, query/key eligibility OR) | [`TensorScaledDotProductAttentionExpressions`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressions.java) creates the explicit two-slot form; `ScaledDotProductAttentionResult` exposes both. The original attention methods create exactly one slot and no hidden weights. |

Public `unstack` is not in this table: it creates independent `SELECT` producers, each with output
index zero. No generic tuple carrier or false sibling provenance exists.

## Selected capability baseline assessment

The classification is deliberately cohesive; the [capability baseline](capabilities.md) remains
the concise selection record rather than a full library reference.

| Area | Classification and readiness conclusion |
|---|---|
| Construction, import, constants, ranges, explicit-source random leaves | **Current public primitive.** `TensorFactory` and `TensorRandoms` cover the selected eager boundary; `TensorRanges` is implementation-owned. Default/global random sources and test prefix population are rejected/deferred. |
| Elementwise numeric, comparison, logical, selection, cast | **Current public primitive**, with `clampMin`/`clampMax` as **current public conveniences**. The normalized seven-family arithmetic vocabulary, classifications, and explicit cast are representable. Unsupported mixed categories and integral DIV/POW/CLAMP remain **deliberately deferred**. |
| Reductions, statistics, scans, softmax | **Current public primitive.** Full/axis/multi-axis forms, masked sum/mean, sum-to-Shape, arg extrema, statistics, cumulative sum/product, softmax, and log-softmax cover the selected baseline. Additional statistical catalog breadth is deferred. |
| Layout, indexing, and composition | **Current public primitive**, with transpose, flip, embedding, and unstack as **current public conveniences**. Signed slicing, slice update/crop, Gather families, scatter families, pad/tile, concat/stack, and window transforms are representable for static and selected symbolic Shapes. `take` and first-class UNSTACK are **rejected from the core baseline**. |
| Linear algebra and attention | MATMUL and scaled dot-product attention are **current public primitives**; linear is a **current public convenience**. Broader decompositions, einsum, FFT, and attention dropout are **deliberately deferred**. |
| Convolution and pooling | Grouped NCHW Conv2d, max pool, and average pool are **current public primitives**. Additional layouts, dimensions, and adaptive/global variants are deferred. |
| Normalization | Layer, RMS, batch-inference, and pure batch-training/statistic-transition forms are **current public primitives**. Stateful module behavior belongs downstream. |
| Losses | Mean-squared error and dense/index-target categorical cross entropy from logits are **current public primitives**. Broader loss catalogs, weighting, smoothing, and probability-input forms are deferred. |
| Explicit graph RNG and dropout | `GraphRngState.initial` is **model semantic without public Tensor construction** through an opaque public owner; dropout is a **current public primitive**. Algorithm/bitstream selection is deliberately downstream-owned. |
| Graph and storage foundations | **Current public primitive/value contract** at the model boundary. Capture, compilation, publication planning, device storage, and execution are deliberately downstream-owned. |

The minimum selected inference/training baseline is therefore representable as model metadata.
The milestone remains open only because the current `GraphValue` Javadoc misstates the status of
that public model surface.

## Dynamic Shape and typed-scalar assessment

The Shape model covers the selected formulas without pretending to bind runtime values:

- named `DynamicDimension` preserves caller identity;
- `ExpressionDimension` represents canonical addition, non-negative multiplication, constant
  offset, floor/ceiling division, selected products of unresolved factors, and constrained
  unknowns;
- public construction derives exact symbolic Shapes where the selected algebra can prove them and
  records later binding obligations where it cannot;
- `sumToShape`, target-relative crop, dynamic window products, MATMUL/attention batch obligations,
  and normalization Shapes remain honest model metadata rather than hidden runtime extents.

[`DimensionExpressions`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DimensionExpressions.java)
and the Shape package tests are the primary evidence.

`ScalarValue` remains the sole exact scalar-attribute representation. Scalar arithmetic,
range clamp, padding, normalization epsilon/momentum, attention scale, and loss ignore values use
typed scalars. Source scans found no raw-double semantic alternative; retained `double` Tensor
overloads are exact-FLOAT64 public conveniences that construct `ScalarValue`. This boundary is
sufficient for all six selected data types and does not claim FLOAT16 or implicit conversion.

## Numerical-policy closure assessment

The selected model fixes forward meaning where metadata must distinguish results and names the
later owner where model-level bitwise execution policy is intentionally absent.

| Policy area | Closure assessment |
|---|---|
| Floating extrema/comparisons | Pairwise/reduction extrema specify NaN and signed-zero behavior; comparisons and arg extrema have explicit ordering/tie contracts where applicable. Backend conformance later proves execution. |
| Reduction empty domains and accumulation | Ordinary, masked, integral, statistical, norm, loss, pooling, and scan contracts state their applicable empty/identity/NaN or invalid-domain behavior and accumulation/result types. Floating reassociation/bitwise determinism remains backend-conformance policy where explicitly allowed. |
| Integral arithmetic | Selected ADD/SUB/MUL and reductions use exact-width modular semantics; signed MIN/MAX/comparisons are fixed. Integral DIV, POW, remainder, saturation, and unsigned arithmetic are rejected/deferred rather than ambiguous current operations. |
| Ordering/top-K | Stable sort/argsort and deterministic top-K specify axis, NaN placement, signed-zero order, ties, `k` bounds/deferred dynamic proof, and INT64 indices. |
| Gather/scatter/slice | Families have distinct Shape relations; signed slicing and dynamic bounds obligations are explicit. Index bounds are execution-time obligations. Scatter replacement/ADD and other reductions state duplicate/base participation; deterministic execution for repeated targets belongs to backend conformance. |
| Statistics/normalization | Correction, denominator validity, epsilon placement/type, population versus uncentered definitions, empty normalized regions, and promoted accumulation are fixed. Derivative behavior at singular/special points remains compiler-owned and non-blocking. |
| Transcendentals/activations | Each kind identifies a portable mathematical target; GELU approximation is fixed and named. Algorithm, tolerance, payload, and bitwise route are backend-conformance concerns, not additional model attributes. |
| Attention/losses | Mask/causal meaning, all-masked rows, stable logits formulation, class bounds/ignore ordering, reduction denominator, accumulation, and public weights roles are fixed. Backend determinism/tolerance remains downstream. |
| RNG/dropout | State words, modular counter interval, probability range, inverted scaling, one logical draw per element, mask role, and next-state role are fixed. No portable bitstream is selected; backend prepare/conformance owns the concrete algorithm and replay scope. |
| Adjoint boundaries | The adjoint matrix explicitly defers discontinuity, tie/subgradient, exceptional-value, and cross-floating-cast derivative policies to compiler work. Those are not forward-semantic ambiguities and need no backward-specific model kind. |

No selected forward occurrence has an unrecorded policy that prevents honest model construction.

## Legacy-cleanup and unusual-capability closure

Source, test, API, glossary, and capability scans confirm:

| Decision | Current closure evidence |
|---|---|
| Fast and inverse aliases | No `FAST_EXP`, `FAST_TANH`, `fastExp`, `fastTanh`, `INV`, or `inv` production symbol exists. `RECIPROCAL`/`reciprocal` is the sole spelling. |
| Arithmetic normalization | Floating Tensor/Tensor and Tensor/scalar use ADD, SUB, MUL, DIV, MIN, MAX, POW; `minimum`/`maximum` remain distinct from reduction `min`/`max`. Selected integral subsets remain coherent and mixed numeric categories require explicit cast. |
| Index taxonomy | Gather, Gather Elements, Gather-ND, Select, Slice, Scatter Add, Scatter Elements, and Scatter-ND remain distinct. No public `take` or first-class UNSTACK kind exists. |
| Unstack provenance | Public unstack remains ordered independent SELECT composition, never a shared producer. |
| Mask broadcasting | Masked sum/mean use ordinary right-aligned broadcasting and no heuristic axis mapper. |
| Prefix fixture | `fromStrictFlatPrefix`/`fromCyclicFlatPrefix` occur only in package-private test-source `TensorTestData`; `TensorFactory` exposes neither. |
| Construction ownership | TensorFactory, TensorRandoms, package-private TensorRanges, and constant/import paths remain separated. |
| Typed scalar | ScalarValue is the only supported exact scalar-attribute payload, including padding constants. |
| Signed slicing/layout | Public slice steps are signed non-zero; resolved view geometry remains limited to representable positive-stride cases. |
| Primitive take side effects | Primitive-array `take` and eager index-allocation/prevalidation behavior remain absent. |
| General overlap-add | `foldAxis` is public and general; no operation-specific backward kind accompanies it. |
| Symbolic products | Dimension expressions represent selected addition, product, and division-derived Shapes without model-time runtime binding. |
| Signature hardening | Family-owned typed signatures reject invalid kind/attributes pairs and validate local occurrence counts. |
| Multi-output identity | Dropout, batch training, top-K, and attention weights use exact shared producer identity and output indices with focused public carriers. |
| Backward taxonomy | No operation kind name contains `_BACKWARD`; `GraphPhase.BACKWARD` is a graph classification, not an operation family. |

## Adjoint-prerequisite closure

The completed [adjoint matrix](adjoint-expressibility-audit.md) selected six reusable public
prerequisites. Current source rechecks close all six:

| Prerequisite | Current evidence and conclusion |
|---|---|
| Binding-aware sum-to-Shape | `SUM` accepts `SumToShapeAttrs`; public `sumToShape(Shape)` exists. Closed. |
| Gather-compatible scatter-add | `SCATTER_ADD` and public `scatterAdd` preserve unresolved gathered extents and fixed duplicate ADD. Closed. |
| Slice placement and dynamic crop | `SLICE_UPDATE` plus `SLICE/CropToShapeAttrs` have public construction. Closed. |
| General fold and dynamic/configurable windows | Public `foldAxis`, typed-padding unfold2d, symbolic products, and dynamic fold/unfold Shapes exist. Closed. |
| Zero-safe product prerequisite | `CUM_PROD` and both public traversal forms exist. Closed. |
| Same-occurrence attention weights | Explicit two-output attention and `ScaledDotProductAttentionResult` exist without changing the one-output form. Closed. |

Existing Scatter Elements/Scatter-ND, typed zero/one expansion, top-K indices, dropout mask,
batch-training saved values, and recomputable max-pool selection remain sufficient exactly as the
matrix recorded. No `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP` and no operation-specific backward
family appeared. Compiler adoption remains downstream work.

## Documentation and terminology consistency

The [Tensor API](../../../api/tensor-api.md) and [Compile API](../../../api/compile-api.md) match the
200-method surface, current semantic construction, producer/provenance contracts, and planned
compiler boundary. The [Runtime API](../../../api/runtime-api.md) correctly remains conceptual.
The [glossary](../../../glossary.md) already defines the reusable terms and required no new entry
because this audit introduces no domain term or behavior.

Two documentation-only drifts were corrected within authorized paths:

- [Public API status](../../../api/public-api.md) no longer says the implemented public `Tensor`
  and model surface are absent.
- [Training API](../../../api/training-api.md) now keeps `Parameter` and `Buffer` in planned
  `extensions/nn`, while training owns parameter groups, optimizers, and orchestration.

Compile, Runtime, and glossary text required no modification because their current/planned and
ownership statements remain accurate. No Javadoc changed. Review found one blocking Javadoc drift
outside the authorized paths: `GraphValue` says the implemented public mutable Tensor is planned.
The final model Javadoc generation remains the rendering check; successful generation cannot make
that semantic status sentence accurate.

## Deferred, rejected, and downstream-owned capabilities

| Classification | Capabilities and owner |
|---|---|
| Deliberately deferred model breadth | FLOAT16, quantized/sparse types, broader integer arithmetic, einsum/FFT, additional layouts/convolutions/pooling/losses, attention dropout, and general library breadth; a future concrete model task must justify each. |
| Rejected from the selected core | Fast unary aliases, `inv`, ambiguous `take`, first-class UNSTACK, prefix-population factory APIs, raw-double semantic attributes, generic tuple outputs, backend/fused kinds, and operation-specific backward kinds. |
| Compiler-owned | Capture, graph-wide validation, symbolic binding proof, canonicalization, autograd rules, saved-output lifetime, accumulation, and deferred derivative policy. |
| Planning/prepare/backend-owned | Capability analysis, backend ownership, lowering, fusion, kernel choice, numerical algorithms, tolerances, determinism guarantees, and concrete RNG bitstreams. |
| Runtime-owned | Prepared schedules, physical memory, device residency, transfers, per-run state, execution, and publication. |
| Extension-owned | Module/parameter/buffer/train-eval behavior in `extensions/nn`; optimizers and training orchestration in `extensions/training`. |

These deferrals do not block model closure because the current model neither claims nor needs them
to represent the selected baseline honestly.

## Findings, severity, and disposition

Only the following labels are used.

| ID | Label | Evidence, impact, owner, and disposition |
|---|---|---|
| DOC-01 | `DOCUMENTATION_DRIFT` | `docs/api/public-api.md` contradicted implemented Tensor/source inventory. Corrected in this task; no behavior changed. |
| DOC-02 | `DOCUMENTATION_DRIFT` | `docs/api/training-api.md` assigned planned `Parameter` ownership to training. Corrected to the established nn-to-training boundary; no architecture change. |
| DEF-01 | `NON_BLOCKING_DEFERRED` | Compiler capture, autograd adoption, binding proof, and derivative boundary policies are not implemented. Their later owner and non-model impact are explicit. |
| DEF-02 | `NON_BLOCKING_DEFERRED` | Backend execution, numerical tolerances/determinism, and RNG bitstreams remain unimplemented. Model metadata does not claim them. |
| GAP-01 | `BLOCKING` | `GraphValue.java` describes the current public mutable Tensor API as planned. Evidence: current `Tensor.java`, the architecture invariant, and `GraphValue.java` lines 9–12. Impact: a core graph Javadoc contract misstates the status of the adjacent public model. Owner: one bounded model Javadoc correction with focused review and validation. Dependency: task 0024 audit evidence. First follow-up: concise Draft row 0024A; this task does not create its detailed specification or edit Java. |
| NC-01 | `NO_CHANGE_CONFIRMED` | Full production/test, kind/signature, public surface, cleanup, multi-output, and adjoint inventories found no behavioral model gap or architecture conflict. |

There is exactly one `BLOCKING` finding. It is documentation-only, requires no architecture
decision, and keeps the selected model milestone open until the bounded follow-up is complete.

## Checkpoint evidence and model-milestone decision

The single required combined Gradle checkpoint ran after the substantive audit and documentation
were stable:

```bash
./gradlew test :testing:architecture-tests:test :modules:model:javadoc
```

It completed successfully in two seconds. Gradle reported 39 actionable tasks: 2 executed and 37
up-to-date. The explicit architecture-test and model-Javadoc tasks were up-to-date and the root
test selection deduplicated them as expected. XML report evidence records 1,016 model tests across
127 suites and one architecture test, with zero failures, errors, or skips: 1,017 tests across
128 suites total.

The root aggregators and every project other than `modules:model` and
`testing:architecture-tests` had no tests: `backends`, `extensions`, `modules`, `testing`, `tools`,
`backends:openblas-provider`, `backends:cpu`, `backends:metal`, `backends:cuda`,
`extensions:onnx`, `extensions:training`, `modules:backend-contract`, `modules:compiler`,
`modules:config`, `modules:engine`, `modules:planning`, `modules:prepare`, `modules:runtime`,
`modules:trace`, `testing:backend-conformance`, `testing:integration-tests`, `tools:benchmarks`,
`tools:cli`, and `tools:tuning`.

Final non-Gradle validation covers repository Markdown, whitespace, the exact seven authorized
changed paths, absence of Java/architecture/build/cross-module changes, absence of a detailed
0024A or later task file, preserved completed history, and synchronized Complete/Open/Draft
status.

Model-milestone decision: **Open**. Task 0024 itself may become Complete because the requested
audit and checkpoint completed, but the selected `modules/model` milestone remains open. The first
bounded frontier is Draft task 0024A, a `GraphValue` Tensor-status Javadoc correction; no detailed
task specification is created here. `modules/trace` remains the next project area after that gap
is closed.

### Closure addendum — 2026-07-13

Task [0024A](tasks/0024a-graph-value-tensor-status-javadoc-correction.md) resolved GAP-01, the
audit's sole blocker. The `GraphValue` type Javadoc now calls `Tensor` the current public mutable
API object instead of planned. A direct source diff confirms that this one status word is the only
`GraphValue.java` change: imports, declaration, record components, constructor, accessors,
validation, messages, and surrounding contracts are unchanged.

Final `:modules:model:javadoc` generation passed, and inspection of the generated `GraphValue`
page confirmed the corrected current-status wording, the immutable graph-local value boundary,
and the absence of the stale phrase. Repository Markdown, links, anchors, fences, final newlines,
trailing whitespace, task-specific six-path scope, synchronized status, and `git diff --check`
also passed. Task 0024's successful 1,017-test repository checkpoint was reused because 0024A
changed no executable Java; no Java test suite was repeated.

The original `BLOCKING_GAP` verdict above remains the historical result at audit time. With its
only blocker resolved by 0024A, the selected `modules/model` capability milestone is now
**Complete**. No later detailed task specification was created.
