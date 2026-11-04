# [Model](../../../glossary.md#term-model) Capability and Contract Closure Audit

**Editorial reading key (later context, not a status update).**

The initial verdict and the closure addendum answer different historical checkpoints; the later navigation here changes neither. ‘Closed model representation’ means that selected meanings and metadata can be represented and constructed, not that every described operation executes. The application-facing Tensor can change its borrowed-storage association; GraphValue instead pairs a graph-local identity with an immutable description. Neither is a physical buffer. The audit's integration inventory checks those distinctions, while ownership of actual compilation, preparation and invocation resources stays with the downstream components named below.

Focused current reading: [Current Tensor state reference](../../../reference/tensor/identity.md#public-tensor-state); [Current graph-value reference](../../../reference/tensor/compiled-graph.md#graph-values-and-compiled-nodes). The original record below retains its own scope, decisions and evidence.


## Executive conclusion and closure verdict

**Verdict: `BLOCKING_GAP`.**

The audit completed and found one blocking documentation-contract gap in a review-only Java path:
[`GraphValue`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/GraphValue.java)
calls the already-implemented public [mutable](../../../glossary.md#term-mutable) [`Tensor`](../../../glossary.md#tensor) [API](../../../glossary.md#term-application-programming-interface-api) “planned.” The wording contradicts the
current source and architecture [invariant](../../../glossary.md#term-invariant). Task 0024 cannot correct Java/[Javadoc](../../../glossary.md#term-javadoc), so the selected
`modules/model` milestone remains open for one bounded follow-up.

Apart from that stale Javadoc sentence, current source provides a coherent [backend](../../../glossary.md#backend)-independent tensor [model](../../../glossary.md#term-model), exact [operation](../../../glossary.md#operation) [occurrence](../../../glossary.md#term-occurrence)
[signatures](../../../glossary.md#term-signature), public expression construction for the selected [inference](../../../glossary.md#term-inference)/training baseline, dynamic
[Shape](../../../glossary.md#shape) [metadata](../../../glossary.md#term-metadata), typed [scalar](../../../glossary.md#term-scalar) attributes, shared multi-output [provenance](../../../glossary.md#provenance), and [immutable](../../../glossary.md#term-immutable) [graph](../../../glossary.md#graph)
contracts. No behavioral model gap or architecture decision remains.

This verdict does not claim [graph capture](../../../glossary.md#capture-graph-ports-and-reconstruction), [automatic differentiation](../../../glossary.md#autograd) ([autograd](../../../glossary.md#autograd)), backend support,
[kernels](../../../glossary.md#kernel), numerical execution, prepared state, [runtime](../../../glossary.md#term-runtime) [residency](../../../glossary.md#residency), or general tensor-library
completeness. Those are later-layer responsibilities. The closure test is:

```text
selected model meaning + honest metadata + valid occurrence construction
  = closed model representation

compiler + planning + prepare + backend + runtime
  = downstream implementation, not implied by this verdict
```

The final [checkpoint](../../../glossary.md#term-checkpoint) evidence and the bounded follow-up appear in the last section.

## Authority, [scope](../../../glossary.md#term-scope), and method

[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative. This audit is a [planning](../../../glossary.md#term-planning) result,
not a new architecture rule or [API](../../../glossary.md#term-application-programming-interface-api) contract. It applied the General, Planning, and API/[Javadoc](../../../glossary.md#term-javadoc)
documentation profiles and task [0024](tasks/0024-model-capability-and-contract-closure-audit.md).

Current source and tests were primary evidence. Completed tasks and the
[adjoint-expressibility result](adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([adjoint glossary](../../../glossary.md#term-adjoint))</span> supplied decision history only.
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
for every foundation, [operation](../../../glossary.md#operation) family, expression helper, [producer](../../../glossary.md#term-producer), and result [carrier](../../../glossary.md#term-storage-carrier). The
reflection helper scanned every top-level compiled [model](../../../glossary.md#term-model) class rather than using a hand-maintained
kind [registry](../../../glossary.md#term-registry). Temporary helpers remained under `/tmp`.

The audit changed documentation/planning only. It did not change Java, tests, [Gradle](../../../glossary.md#term-gradle), architecture,
architecture tests, another [module](../../../glossary.md#term-module), [conformance](../../../glossary.md#term-conformance) tests, or integration tests.

## Architecture-boundary assessment

The source matches the [model](../../../glossary.md#term-model) boundary in the [architecture contract](../../../glossary.md#architecture-contract):

| Check | Evidence | Conclusion |
|---|---|---|
| Model semantics are [backend](../../../glossary.md#backend)-independent | [`Operation`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/Operation.java)<span data-glossary-navigation="true"> ([Operation glossary](../../../glossary.md#operation))</span>, [`OperationKind`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationKind.java)<span data-glossary-navigation="true"> ([OperationKind glossary](../../../glossary.md#operationkind))</span>, and an import [scan](../../../glossary.md#term-cumulative-scan) of all 176 production files | No [backend identity](../../../glossary.md#backend-identity--backendid), support query, [lowering](../../../glossary.md#lowering), [kernel](../../../glossary.md#kernel), [prepare](../../../glossary.md#prepare), or execution route occurs in model semantics. |
| [`Tensor`](../../../glossary.md#tensor) is public [mutable](../../../glossary.md#term-mutable) [API](../../../glossary.md#term-application-programming-interface-api) state, not [graph](../../../glossary.md#graph) [IR](../../../glossary.md#term-intermediate-representation-ir) | [`Tensor`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java), [`TensorProducer`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java)<span data-glossary-navigation="true"> ([TensorProducer glossary](../../../glossary.md#producer--tensorproducer))</span>, and [`CompiledNode`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledNode.java) | Tensor [identity](../../../glossary.md#term-identity-meanings)/storage association and pre-[capture](../../../glossary.md#term-capture-meanings) [provenance](../../../glossary.md#provenance) remain distinct from graph-local [node](../../../glossary.md#node)/value identity. |
| [Runtime](../../../glossary.md#term-runtime) hot-path types remain outside model | Production inventory and architecture-test source scan | No runtime, prepared-execution, [residency](../../../glossary.md#residency), physical-[buffer](../../../glossary.md#term-buffer), or workspace type appears under `modules/model`. |
| [Compiler](../../../glossary.md#term-compiler)/[autograd](../../../glossary.md#autograd) [ownership](../../../glossary.md#term-ownership-meanings) is preserved | [Compile API](../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../glossary.md#compile))</span>, [Training API](../../../api/training-api.md), and the [adjoint audit](adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([adjoint glossary](../../../glossary.md#term-adjoint))</span> | Model exposes forward semantics and reusable primitives; compiler remains the [owner](../../../glossary.md#term-owner) of capture, [gradient](../../../glossary.md#term-gradient) rules, saved-value [lifetime](../../../glossary.md#term-lifetime), and backward construction. |
| [Dependency direction](../../../glossary.md#java-platform-and-build-tools) is preserved | Model source-import scan plus the final architecture-test [checkpoint](../../../glossary.md#term-checkpoint) | `modules/model` does not import [planning](../../../glossary.md#term-planning), compiler, prepare, runtime, [engine](../../../glossary.md#term-engine), concrete backends, or extensions. |

The explanatory architecture pages and architecture tests required no change because no ownership,
[dependency](../../../glossary.md#term-dependency), or [lifecycle](../../../glossary.md#lifecycle) rule changed. The audit found no conflict requiring an [ADR](../../../glossary.md#term-architecture-decision-record-adr) or edit to
`ARCHITECTURE.md`.

## Foundation-contract inventory

The table records readiness rather than every accessor. Source links identify the owning contract;
the matching package tests are included in the 128-file test inventory.

| Foundation | Current contract and readiness |
|---|---|
| Data types and promotion | [`DataType`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataType.java)<span data-glossary-navigation="true"> ([DataType glossary](../../../glossary.md#data-type--datatype))</span> defines [FLOAT64](../../../glossary.md#term-float64), [FLOAT32](../../../glossary.md#term-float32), [BFLOAT16](../../../glossary.md#term-bfloat16), [INT32](../../../glossary.md#data-type--datatype), [INT64](../../../glossary.md#data-type--datatype), and [BOOL](../../../glossary.md#data-type--datatype); [`DataTypePromotion`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataTypePromotion.java) keeps floating and signed-integral promotion explicit. [Ready](../../../glossary.md#term-ready) for the selected baseline. |
| Typed [scalar](../../../glossary.md#term-scalar) values | [`ScalarValue`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/ScalarValue.java)<span data-glossary-navigation="true"> ([ScalarValue glossary](../../../glossary.md#typed-scalar-value--scalarvalue))</span> is the sole exact scalar-attribute [carrier](../../../glossary.md#term-storage-carrier) for all six types, preserving raw floating/BFLOAT16 bits and exact integral/BOOL values. Ready. |
| [Dimensions](../../../glossary.md#dimension) and [Shapes](../../../glossary.md#shape) | [`Dimension`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/Dimension.java), [`StaticDimension`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/StaticDimension.java), [`DynamicDimension`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DynamicDimension.java), [`ExpressionDimension`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/ExpressionDimension.java), and [`Shape`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/Shape.java) represent static, named, [canonical](../../../glossary.md#term-canonical-form) expression, and constrained-unknown extents without [runtime](../../../glossary.md#term-runtime) [binding](../../../glossary.md#term-binding). Ready. |
| [Layouts](../../../glossary.md#layout) and [descriptors](../../../glossary.md#term-descriptor) | [`LayoutDescriptor`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/layout/LayoutDescriptor.java), [`LayoutGeometry`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/layout/LayoutGeometry.java), and [`TensorDescriptor`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java) distinguish logical [metadata](../../../glossary.md#term-metadata) from optional representable physical geometry. Ready; negative-[stride](../../../glossary.md#term-stride) results honestly remain layout-unresolved. |
| Typed identifiers | [`TensorId`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorId.java)<span data-glossary-navigation="true"> ([TensorId glossary](../../../glossary.md#tensorid))</span>, [`NodeId`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/NodeId.java)<span data-glossary-navigation="true"> ([NodeId glossary](../../../glossary.md#nodeid))</span>, and [`ValueId`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/ValueId.java)<span data-glossary-navigation="true"> ([ValueId glossary](../../../glossary.md#logical-valueid-versus-physical-memory-slot))</span> remain separate domains. Ready. |
| [Tensor](../../../glossary.md#tensor) [identity](../../../glossary.md#term-identity-meanings) and [host storage](../../../glossary.md#host-storage) | [`Tensor`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java), [`HostTensorStorage`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/storage/HostTensorStorage.java), and [`MemorySegmentStorage`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/storage/MemorySegmentStorage.java) provide [mutable](../../../glossary.md#term-mutable) [borrowed](../../../glossary.md#ownership-borrowing-and-lease) [host](../../../glossary.md#term-host)-storage association without [device](../../../glossary.md#term-device) [residency](../../../glossary.md#residency). Ready. |
| [Eager](../../../glossary.md#term-eager-evaluation) construction | [`TensorFactory`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java) owns identity, allocation, imports, and [constants](../../../glossary.md#term-constant); [`TensorRandoms`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRandoms.java) owns explicit-source eager random leaves; package-private [`TensorRanges`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRanges.java) owns integer-range mechanics. Ready. |
| [Producer](../../../glossary.md#term-producer)/[provenance](../../../glossary.md#provenance) | [`TensorProducer`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java)<span data-glossary-navigation="true"> ([TensorProducer glossary](../../../glossary.md#producer--tensorproducer))</span> owns one pre-[capture](../../../glossary.md#term-capture-meanings) [occurrence](../../../glossary.md#term-occurrence) and ordered descriptors; [`TensorProvenance`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProvenance.java) selects one zero-based output. Ready. |
| [Immutable](../../../glossary.md#term-immutable) [graph](../../../glossary.md#graph) [model](../../../glossary.md#term-model) | [`GraphValue`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/GraphValue.java), [`CompiledNode`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledNode.java), [`CompiledGraphModel`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/CompiledGraphModel.java)<span data-glossary-navigation="true"> ([CompiledGraphModel glossary](../../../glossary.md#compiled-graph--compiledgraphmodel))</span>, [`GraphPhase`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/GraphPhase.java), and the current [`ForwardPublicationBinding`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/graph/ForwardPublicationBinding.java) provide structurally validated [compile](../../../glossary.md#compile)-time model values without implementing [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names). The later terminology correction from the historical `PublicationBinding` name does not change this audit conclusion. Behavior is ready; `GraphValue` has one stale public-Tensor status sentence that blocks documentation closure. |
| Public multi-output boundaries | [`DropoutResult`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/DropoutResult.java), [`BatchNormTrainingResult`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/BatchNormTrainingResult.java), [`TopKResult`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TopKResult.java), and [`ScaledDotProductAttentionResult`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/ScaledDotProductAttentionResult.java) expose [operation](../../../glossary.md#operation)-specific roles without a generic tuple hierarchy. Ready. |

## Public [Tensor](../../../glossary.md#tensor) and construction inventory

Java 26 reflection found exactly **200 declared public `Tensor` methods**. The following groups
contain every exact [signature](../../../glossary.md#term-signature); array [parameters](../../../glossary.md#term-parameter) shown below correspond to Java varargs where the
source declares them. Category subtotals sum to 200.

| Owning family or documented convenience | Exact signatures, grouped by method name | Count |
|---|---|---:|
| [Identity](../../../glossary.md#term-identity-meanings), [descriptor](../../../glossary.md#term-descriptor), [provenance](../../../glossary.md#provenance), storage, diagnostics | `id()`, `descriptor()`, `label()`, `provenance()`, `hostStorage()`, `replaceHostStorage(HostTensorStorage)`, `clearHostStorage()`, `toString()` | 8 |
| Seven [binary](../../../glossary.md#term-binary-descriptor-and-metadata)/[scalar](../../../glossary.md#term-scalar) arithmetic families | `add/sub/mul/div/minimum/maximum/pow` each with `(Tensor)`, `(ScalarValue)`, `(double)` | 21 |
| Range and one-bound clamp | `clamp(ScalarValue,ScalarValue)`, `clamp(double,double)`, `clampMin(ScalarValue)`, `clampMin(double)`, `clampMax(ScalarValue)`, `clampMax(double)` | 6 |
| [Unary](../../../glossary.md#term-unary) numeric and [activation](../../../glossary.md#term-activation) primitives | `abs()`, `neg()`, `reciprocal()`, `log()`, `log1p()`, `exp()`, `expm1()`, `erf()`, `sqrt()`, `rsqrt()`, `floor()`, `ceil()`, `sign()`, `relu()`, `sigmoid()`, `tanh()`, `gelu()`, `geluTanhApproximation()`, `silu()` | 19 |
| [Floating classification](../../../glossary.md#floating-classification) and cast | `isFinite()`, `isNaN()`, `isInf()`, `cast(DataType)` | 4 |
| Comparison, logical, and selection | Six `(Tensor)` comparisons; `logicalAnd(Tensor)`, `logicalOr(Tensor)`, `logicalNot()`; static `where(Tensor,Tensor,Tensor)` | 10 |
| Aggregate, statistical, and arg reductions | `sum` 6 forms; [`mean`](../../../glossary.md#term-mean) 6; `prod/min/max/all/any` 5 each; `sumToShape(Shape)`; `logSumExp` 2; [`variance`](../../../glossary.md#term-variance) 3; `standardDeviation` 3; `l1Norm/l2Norm` 2 each; `argMin/argMax` 3 each | 56 |
| Cumulative scans | `cumSum(int)`, `cumSum(int,boolean,boolean)`, `cumProd(int)`, `cumProd(int,boolean,boolean)` | 4 |
| [Softmax](../../../glossary.md#softmax--log-softmax) [normalization](../../../glossary.md#term-normalization) | `softmax(int)`, `logSoftmax(int)` | 2 |
| [Layout](../../../glossary.md#layout) and [rank](../../../glossary.md#term-rank) transformations | `contiguous()`; `reshape(long[])`, `reshape(Shape)`; `expand(long[])`, `expand(Shape)`; `expandDims(int)`, `squeeze(int)`, `permute(int[])`, `transpose()` | 9 |
| [Slice](../../../glossary.md#slice) and placement transformations | `slice(long[],long[],int[],long[])`; two `sliceAxis` forms; `flip(int[])`; `sliceUpdate(Tensor,long[],int[],long[])`; `cropToShape(Shape,Shape)` | 6 |
| Indexing and indexed update | `select`; [`gather`](../../../glossary.md#gather); [`embedding`](../../../glossary.md#embedding); `oneHot`; `gatherElements`; two `gatherNd`; three `scatterNd`; two `scatterElements`; `scatterAdd` | 13 |
| Composition, [padding](../../../glossary.md#padding), [tiling](../../../glossary.md#tiling), windows | Static `concat` and `stack`; `unstack`; two `pad`; `tile`; `unfold`; `foldAxis`; two `unfold2d`; `fold2d` | 11 |
| [Linear](../../../glossary.md#term-linear-meanings) algebra and linear convenience | `matmul(Tensor)`, `linear(Tensor)`, `linear(Tensor,Tensor)` | 3 |
| Ordering and top-K | Two `sort`; two `argsort`; two `topK` | 6 |
| [Convolution](../../../glossary.md#term-channels-first-convolution-layers) and [pooling](../../../glossary.md#term-pooling) | Two [`conv2d`](../../../glossary.md#convolution--conv2d); `maxPool2d`; `averagePool2d` | 4 |
| Normalization | Two `layerNorm`; two `rmsNorm`; `batchNormInference`; `batchNormTraining` | 6 |
| [Losses](../../../glossary.md#loss) | `meanSquaredError`; two `categoricalCrossEntropyWithLogits` | 3 |
| Attention | Four `scaledDotProductAttention`; four `scaledDotProductAttentionWithWeights` | 8 |
| Explicit [graph](../../../glossary.md#graph) [RNG](../../../glossary.md#graph-rng-state--graphrngstate)/[dropout](../../../glossary.md#dropout--inverted-dropout) | `dropout(double,GraphRngState)` | 1 |
| **Total** | Reflection over `Tensor.getDeclaredMethods()` with `public` filter | **200** |

Direct one-[occurrence](../../../glossary.md#term-occurrence) construction is package-owned by focused `Tensor*Expressions` helpers plus
[`GraphRngState`](../../../glossary.md#graph-rng-state--graphrngstate); all reach package-private
`TensorFactory.createDerived` or `createDerivedOutputs`, which creates [`TensorProducer`](../../../glossary.md#producer--tensorproducer) and
therefore validates selected signature counts before allocating result identities. Source search
found 58 `new Operation(...)` sites and 51 derived-factory call sites. Repeated family helpers
account for the difference.

Conveniences preserve visible primitive [ownership](../../../glossary.md#term-ownership-meanings):

| Convenience | Exact [producer](../../../glossary.md#term-producer) chain |
|---|---|
| `clampMin` / `clampMax` | Scalar `MAX` / scalar `MIN` |
| [`transpose`](../../../glossary.md#term-transpose) | `PERMUTE[1,0]` |
| `linear` | `PERMUTE -> MATMUL -> optional ADD` |
| `embedding` | `GATHER(axis=0)` |
| `flip` | One signed `SLICE` |
| `unstack` | Ordered independent `SELECT` occurrences |

No convenience hides a [fused](../../../glossary.md#specialization-and-fusion) or backward-only kind. All public [eager](../../../glossary.md#term-eager-evaluation) [leaves](../../../glossary.md#term-leaf) are provenance-free;
all direct expression results are storage-free and have validated provenance.

## [Operation](../../../glossary.md#operation) kind, attributes, and [signature](../../../glossary.md#term-signature) inventory

Reflection found **37 concrete kind enum types, 107 [constants](../../../glossary.md#term-constant), 47 concrete [`OperationAttrs`](../../../glossary.md#operationattrs)
implementations, and 127 signatures**. Every attributes implementation appears in at least one
signature. Every constant returns the same non-empty [immutable](../../../glossary.md#term-immutable) list across calls, each list has
one variant per exact attributes class, every range satisfies `0 <= minInputs <= maxInputs` and
`1 <= minOutputs <= maxOutputs`, and mutation attempts fail. `Operation.signatureFor` performs
exact-class matching, so no permissive kind/attributes pairing exists.

Notation in the table is `Attrs inputs -> outputs`; `1..N` means one through
`Integer.MAX_VALUE` inputs.

| Kind family | Constants and exact accepted signatures | Construction [owner](../../../glossary.md#term-owner) |
|---|---|---|
| `ScaledDotProductAttentionKind` | `SCALED_DOT_PRODUCT_ATTENTION`: `ScaledDotProductAttentionAttrs 3..4 -> 1..2` | `TensorScaledDotProductAttentionExpressions` |
| `Conv2dKind` | [`CONV2D`](../../../glossary.md#convolution--conv2d): `Conv2dAttrs 2..3 -> 1` | `TensorConv2dExpressions` |
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
| `AxisTransformKind` | `PERMUTE`: `PermutationAttrs 1 -> 1`; `EXPAND_DIMS, SQUEEZE`: `AxisTransformAttrs 1 -> 1` | [permutation](../../../glossary.md#permutation)/[rank](../../../glossary.md#term-rank)-edit helpers |
| `ContiguousKind` | [`CONTIGUOUS`](../../../glossary.md#term-contiguous): `NoOperationAttrs 1 -> 1` | `TensorContiguousExpressions` |
| `PadKind` | [`PAD`](../../../glossary.md#padding): `PadAttrs 1 -> 1` | `TensorPadTileExpressions` |
| `ShapeTransformKind` | `RESHAPE, EXPAND`: `TargetShapeAttrs 1 -> 1` | reshape/expand helpers |
| `SliceKind` | [`SLICE`](../../../glossary.md#slice): `SliceAttrs 1 -> 1` and `CropToShapeAttrs 1 -> 1`; `SLICE_UPDATE`: `SliceAttrs 2 -> 1` | slice/slice-placement helpers |
| `TensorCompositionKind` | `CONCAT, STACK`: `CompositionAxisAttrs 1..N -> 1` | `TensorCompositionExpressions` |
| `TileKind` | `TILE`: `TileAttrs 1 -> 1` | `TensorPadTileExpressions` |
| `WindowTransformKind` | `UNFOLD_AXIS`: `UnfoldAxisAttrs 1 -> 1`; `FOLD_AXIS`: `FoldAxisAttrs 1 -> 1`; `UNFOLD2D`: `Window2dAttrs 1 -> 1` and `Unfold2dAttrs 1 -> 1`; `FOLD2D`: `Fold2dAttrs 1 -> 1` | `TensorWindowExpressions` |
| `MatmulKind` | [`MATMUL`](../../../glossary.md#matrix-multiplication--matmul): `NoOperationAttrs 2 -> 1` | `TensorMatmulExpressions` |
| `LossKind` | `MEAN_SQUARED_ERROR`: `MeanSquaredErrorAttrs 2 -> 1`; `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`: `DenseCategoricalCrossEntropyWithLogitsAttrs 2 -> 1`; `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`: `IndexCategoricalCrossEntropyWithLogitsAttrs 2 -> 1` | `TensorLossExpressions` |
| `BatchNormKind` | `BATCH_NORM_INFERENCE`: `BatchNormInferenceAttrs 5 -> 1`; `BATCH_NORM_TRAINING`: `BatchNormTrainingAttrs 5 -> 5` | [inference](../../../glossary.md#term-inference)/training helpers |
| `LayerNormKind` | `LAYER_NORM`: `LayerNormAttrs 1 -> 1` and `AffineLayerNormAttrs 3 -> 1` | `TensorLayerNormExpressions` |
| `RmsNormKind` | `RMS_NORM`: `RmsNormAttrs 1..2 -> 1` | `TensorRmsNormExpressions` |
| `SoftmaxKind` | `SOFTMAX, LOG_SOFTMAX`: `SoftmaxAttrs 1 -> 1` | `TensorSoftmaxExpressions` |
| `OrderingKind` | `SORT, ARGSORT`: `SortAttrs 1 -> 1` | `TensorSortExpressions` |
| `TopKKind` | `TOP_K`: `TopKAttrs 1 -> 2` | `TensorTopKExpressions` |
| `Pool2dKind` | `MAX_POOL2D`: `MaxPool2dAttrs 1 -> 1`; `AVERAGE_POOL2D`: `AveragePool2dAttrs 1 -> 1` | pool helpers |
| `DropoutKind` | [`DROPOUT`](../../../glossary.md#dropout--inverted-dropout): `DropoutAttrs 2 -> 3` | `TensorDropoutExpressions` |
| `GraphRngKind` | `INITIAL_STATE`: `GraphRngStateAttrs 0 -> 1` | public `GraphRngState.initial`, not a public [Tensor](../../../glossary.md#tensor) method |
| `AggregateReductionKind` | `SUM`: [`NoOperationAttrs`](../../../glossary.md#nooperationattrs), `AxisReductionAttrs`, `MultiAxisReductionAttrs`, `MaskedReductionAttrs 2 -> 1`, `SumToShapeAttrs`; [`MEAN`](../../../glossary.md#term-mean): first four except [sum-to-Shape](../../../glossary.md#sum-to-shape); `PROD, MIN, MAX, ALL, ANY`: ordinary/[axis](../../../glossary.md#term-axis)/multi-axis variants; `ARG_MAX, ARG_MIN`: `ArgExtremaAttrs`; `LOG_SUM_EXP, L1_NORM, L2_NORM`: `MultiAxisReductionAttrs`; `VARIANCE, STANDARD_DEVIATION`: `StatisticalReductionAttrs`. Unmarked variants are `1 -> 1`. | reduction, masked, multi-axis, arg-[extrema](../../../glossary.md#saturation-pole-and-extremum), and sum-to-Shape helpers |
| `CumulativeScanKind` | `CUM_SUM, CUM_PROD`: `CumulativeScanAttrs 1 -> 1` | `TensorCumulativeScanExpressions` |

Every kind except `GraphRngKind.INITIAL_STATE` has a public Tensor construction path.
`INITIAL_STATE` deliberately has the public [opaque](../../../glossary.md#term-opaque) `GraphRngState.initial` owner because exposing
its state [lanes](../../../glossary.md#term-lane) as a general numerical Tensor would weaken the selected state-threading boundary.
There is no unowned semantic-only kind.

## Multi-output [occurrence](../../../glossary.md#term-occurrence) inventory

These are all genuine multi-output forms. Each successful call creates one [producer](../../../glossary.md#term-producer), ordered
[descriptors](../../../glossary.md#term-descriptor), fresh wrapper [identities](../../../glossary.md#term-identity-meanings) for every slot, and [provenance](../../../glossary.md#provenance) indices matching the table.

| Occurrence | Ordered roles and descriptors | Public boundary and sharing proof |
|---|---|---|
| [`DROPOUT`](../../../glossary.md#dropout--inverted-dropout) | `0=output` (input floating type/[Shape](../../../glossary.md#shape)/eligibility), `1=keep mask` ([BOOL](../../../glossary.md#data-type--datatype), input Shape, non-[gradient](../../../glossary.md#term-gradient)), `2=next state` ([INT64](../../../glossary.md#data-type--datatype) `Shape[2]`, non-gradient); all unresolved [layout](../../../glossary.md#layout) | [`TensorDropoutExpressions`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDropoutExpressions.java) creates all three under one producer. `DropoutResult` exposes slot 0 and [opaque](../../../glossary.md#term-opaque) slot 2; slot 1 is hidden but producer-described for [compiler](../../../glossary.md#term-compiler)-owned backward [capture](../../../glossary.md#term-capture-meanings). |
| `BATCH_NORM_TRAINING` | `0=output` (input Shape), `1=next running mean`, `2=next running variance`, `3=saved batch mean`, `4=saved inverse standard deviation`; [statistic](../../../glossary.md#term-statistic) slots use [rank-one](../../../glossary.md#axis-rank-scalar-vector-and-matrix) [channel](../../../glossary.md#term-channel) Shape and promoted floating type | [`TensorBatchNormTrainingExpressions`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormTrainingExpressions.java) creates five slots. `BatchNormTrainingResult` exposes 0–2; slots 3–4 remain hidden compiler-facing saved values. |
| `TOP_K` | `0=values` (input type/eligibility), `1=indices` (INT64/non-gradient); both share the Shape with selected [axis](../../../glossary.md#term-axis) replaced by static `k` | [`TensorTopKExpressions`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorTopKExpressions.java) and `TopKResult` expose both slots from one producer. |
| Attention with [weights](../../../glossary.md#term-weight) | `0=output` (promoted type, `[...,L,Ev]`, query/key/value eligibility OR), `1=weights` (promoted type, `[...,L,S]`, query/key eligibility OR) | [`TensorScaledDotProductAttentionExpressions`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressions.java) creates the explicit two-slot form; `ScaledDotProductAttentionResult` exposes both. The original attention methods create exactly one slot and no hidden weights. |

Public `unstack` is not in this table: it creates independent `SELECT` producers, each with output
index zero. No generic tuple [carrier](../../../glossary.md#term-storage-carrier) or false sibling provenance exists.

## Selected capability baseline assessment

The classification is deliberately cohesive; the [capability baseline](capabilities.md) remains
the concise selection record rather than a full library reference.

| Area | Classification and readiness conclusion |
|---|---|
| Construction, import, [constants](../../../glossary.md#term-constant), ranges, explicit-source random leaves | **Current public primitive.** `TensorFactory` and `TensorRandoms` cover the selected [eager](../../../glossary.md#term-eager-evaluation) boundary; `TensorRanges` is implementation-owned. Default/global random sources and test prefix population are rejected/deferred. |
| [Elementwise](../../../glossary.md#term-elementwise) numeric, comparison, logical, selection, cast | **Current public primitive**, with `clampMin`/`clampMax` as **current public conveniences**. The normalized seven-family arithmetic vocabulary, classifications, and explicit cast are representable. Unsupported mixed categories and integral DIV/POW/CLAMP remain **deliberately deferred**. |
| Reductions, statistics, scans, [softmax](../../../glossary.md#softmax--log-softmax) | **Current public primitive.** Full/[axis](../../../glossary.md#term-axis)/multi-axis forms, masked sum/[mean](../../../glossary.md#term-mean), [sum-to-Shape](../../../glossary.md#sum-to-shape), arg [extrema](../../../glossary.md#saturation-pole-and-extremum), statistics, cumulative sum/product, softmax, and [log-softmax](../../../glossary.md#softmax--log-softmax) cover the selected baseline. Additional statistical catalog breadth is deferred. |
| [Layout](../../../glossary.md#layout), indexing, and composition | **Current public primitive**, with [transpose](../../../glossary.md#term-transpose), flip, [embedding](../../../glossary.md#embedding), and unstack as **current public conveniences**. Signed slicing, [slice update](../../../glossary.md#slice-update)/crop, [Gather](../../../glossary.md#gather) families, scatter families, pad/tile, concat/stack, and window transforms are representable for static and selected symbolic [Shapes](../../../glossary.md#shape). `take` and first-class UNSTACK are **rejected from the core baseline**. |
| [Linear](../../../glossary.md#term-linear-meanings) algebra and attention | [MATMUL](../../../glossary.md#matrix-multiplication--matmul) and [scaled dot-product attention](../../../glossary.md#scaled-dot-product-attention) are **current public primitives**; linear is a **current public convenience**. Broader decompositions, einsum, [FFT](../../../glossary.md#alternative-convolution-algorithms), and attention [dropout](../../../glossary.md#dropout--inverted-dropout) are **deliberately deferred**. |
| [Convolution](../../../glossary.md#term-channels-first-convolution-layers) and [pooling](../../../glossary.md#term-pooling) | Grouped [NCHW](../../../glossary.md#term-nchw) [Conv2d](../../../glossary.md#convolution--conv2d), max pool, and average pool are **current public primitives**. Additional layouts, [dimensions](../../../glossary.md#dimension), and adaptive/global variants are deferred. |
| [Normalization](../../../glossary.md#term-normalization) | Layer, RMS, [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[inference](../../../glossary.md#term-inference), and pure batch-training/[statistic](../../../glossary.md#term-statistic)-transition forms are **current public primitives**. Stateful [module](../../../glossary.md#term-module) behavior belongs downstream. |
| [Losses](../../../glossary.md#loss) | [Mean-squared error](../../../glossary.md#mean-squared-error--mse) and dense/index-[target](../../../glossary.md#term-target) [categorical cross entropy](../../../glossary.md#categorical-cross-entropy) from logits are **current public primitives**. Broader loss catalogs, weighting, smoothing, and probability-input forms are deferred. |
| Explicit [graph](../../../glossary.md#graph) [RNG](../../../glossary.md#graph-rng-state--graphrngstate) and dropout | `GraphRngState.initial` is **[model](../../../glossary.md#term-model) semantic without public [Tensor](../../../glossary.md#tensor) construction** through an [opaque](../../../glossary.md#term-opaque) public [owner](../../../glossary.md#term-owner); dropout is a **current public primitive**. Algorithm/bitstream selection is deliberately downstream-owned. |
| Graph and storage foundations | **Current public primitive/value contract** at the model boundary. [Capture](../../../glossary.md#term-capture-meanings), [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names), [publication](../../../glossary.md#term-publication) [planning](../../../glossary.md#term-planning), [device](../../../glossary.md#term-device) storage, and execution are deliberately downstream-owned. |

The minimum selected inference/training baseline is therefore representable as model [metadata](../../../glossary.md#term-metadata).
The milestone remains open only because the current `GraphValue` [Javadoc](../../../glossary.md#term-javadoc) misstates the status of
that public model surface.

## Dynamic [Shape](../../../glossary.md#shape) and typed-[scalar](../../../glossary.md#term-scalar) assessment

The Shape [model](../../../glossary.md#term-model) covers the selected formulas without pretending to bind [runtime](../../../glossary.md#term-runtime) values:

- named `DynamicDimension` preserves caller [identity](../../../glossary.md#term-identity-meanings);
- `ExpressionDimension` represents canonical addition, non-negative multiplication, constant
  offset, floor/ceiling division, selected products of unresolved factors, and constrained
  unknowns;<span data-glossary-navigation="true"> ([canonical glossary](../../../glossary.md#term-canonical-form); [constant glossary](../../../glossary.md#term-constant))</span>
- public construction derives exact symbolic Shapes where the selected algebra can prove them and
  records later binding obligations where it cannot;<span data-glossary-navigation="true"> ([binding glossary](../../../glossary.md#term-binding))</span>
- `sumToShape`, target-relative crop, dynamic window products, MATMUL/attention batch obligations,
  and normalization Shapes remain honest model metadata rather than hidden runtime extents.<span data-glossary-navigation="true"> ([target-relative crop glossary](../../../glossary.md#target-relative-crop); [MATMUL glossary](../../../glossary.md#matrix-multiplication--matmul); [batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [metadata glossary](../../../glossary.md#term-metadata))</span>

[`DimensionExpressions`](../../../../modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DimensionExpressions.java)
and the Shape package tests are the primary evidence.

[`ScalarValue`](../../../glossary.md#typed-scalar-value--scalarvalue) remains the sole exact scalar-attribute representation. Scalar arithmetic,
range clamp, [padding](../../../glossary.md#padding), [normalization](../../../glossary.md#term-normalization) [epsilon](../../../glossary.md#epsilon)/[momentum](../../../glossary.md#term-momentum), attention scale, and [loss](../../../glossary.md#loss) ignore values use
typed scalars. Source scans found no raw-double semantic alternative; retained `double` [Tensor](../../../glossary.md#tensor)
overloads are exact-[FLOAT64](../../../glossary.md#term-float64) public conveniences that construct `ScalarValue`. This boundary is
sufficient for all six selected data types and does not claim [FLOAT16](../../../glossary.md#term-float16) or implicit conversion.

## Numerical-policy closure assessment

The selected [model](../../../glossary.md#term-model) fixes forward meaning where [metadata](../../../glossary.md#term-metadata) must distinguish results and names the
later [owner](../../../glossary.md#term-owner) where model-level bitwise execution policy is intentionally absent.

| Policy area | Closure assessment |
|---|---|
| Floating [extrema](../../../glossary.md#saturation-pole-and-extremum)/comparisons | Pairwise/reduction extrema specify [NaN](../../../glossary.md#term-nan) and [signed-zero](../../../glossary.md#nan-infinity-and-signed-zero) behavior; comparisons and arg extrema have explicit ordering/tie contracts where applicable. [Backend](../../../glossary.md#backend) [conformance](../../../glossary.md#term-conformance) later proves execution. |
| Reduction empty domains and accumulation | Ordinary, masked, integral, statistical, [norm](../../../glossary.md#term-norm), [loss](../../../glossary.md#loss), [pooling](../../../glossary.md#term-pooling), and [scan](../../../glossary.md#term-cumulative-scan) contracts state their applicable empty/[identity](../../../glossary.md#term-identity-meanings)/NaN or invalid-domain behavior and accumulation/result types. Floating [reassociation](../../../glossary.md#term-reassociation)/bitwise determinism remains backend-conformance policy where explicitly allowed. |
| Integral arithmetic | Selected ADD/SUB/MUL and reductions use exact-width modular semantics; signed MIN/MAX/comparisons are fixed. Integral DIV, POW, remainder, [saturation](../../../glossary.md#term-saturation), and unsigned arithmetic are rejected/deferred rather than ambiguous current operations. |
| Ordering/top-K | Stable sort/argsort and deterministic top-K specify [axis](../../../glossary.md#term-axis), NaN placement, signed-zero order, ties, `k` bounds/deferred dynamic proof, and [INT64](../../../glossary.md#data-type--datatype) indices. |
| [Gather](../../../glossary.md#gather)/scatter/[slice](../../../glossary.md#slice) | Families have distinct [Shape](../../../glossary.md#shape) relations; signed slicing and dynamic bounds obligations are explicit. Index bounds are execution-time obligations. Scatter replacement/ADD and other reductions state duplicate/base participation; deterministic execution for repeated [targets](../../../glossary.md#term-target) belongs to backend conformance. |
| Statistics/[normalization](../../../glossary.md#term-normalization) | Correction, denominator validity, [epsilon](../../../glossary.md#epsilon) placement/type, population versus uncentered definitions, empty normalized regions, and promoted accumulation are fixed. [Derivative](../../../glossary.md#term-derivative) behavior at singular/special points remains [compiler](../../../glossary.md#term-compiler)-owned and non-blocking. |
| Transcendentals/[activations](../../../glossary.md#term-activation) | Each kind identifies a portable mathematical target; [GELU](../../../glossary.md#gaussian-error-linear-unit--gelu) approximation is fixed and named. Algorithm, [tolerance](../../../glossary.md#term-tolerance), [payload](../../../glossary.md#artifact-envelope-and-atomic-publication), and bitwise route are backend-conformance concerns, not additional model attributes. |
| Attention/losses | [Mask](../../../glossary.md#term-boolean-mask)/causal meaning, all-masked rows, stable logits formulation, class bounds/ignore ordering, reduction denominator, accumulation, and public [weights](../../../glossary.md#term-weight) roles are fixed. Backend determinism/tolerance remains downstream. |
| [RNG](../../../glossary.md#graph-rng-state--graphrngstate)/[dropout](../../../glossary.md#dropout--inverted-dropout) | State words, modular counter interval, probability range, inverted scaling, one logical draw per [element](../../../glossary.md#term-element), mask role, and next-state role are fixed. No portable bitstream is selected; backend [prepare](../../../glossary.md#prepare)/conformance owns the concrete algorithm and replay [scope](../../../glossary.md#term-scope). |
| [Adjoint](../../../glossary.md#term-adjoint) boundaries | The adjoint [matrix](../../../glossary.md#term-matrix-meanings) explicitly defers discontinuity, tie/subgradient, exceptional-value, and cross-floating-cast derivative policies to compiler work. Those are not forward-semantic ambiguities and need no backward-specific model kind. |

No selected forward [occurrence](../../../glossary.md#term-occurrence) has an unrecorded policy that prevents honest model construction.

## Legacy-cleanup and unusual-capability closure

Source, test, [API](../../../glossary.md#term-application-programming-interface-api), glossary, and capability scans confirm:

| Decision | Current closure evidence |
|---|---|
| Fast and inverse [aliases](../../../glossary.md#aliasing-and-arena) | No `FAST_EXP`, `FAST_TANH`, `fastExp`, `fastTanh`, `INV`, or `inv` production [symbol](../../../glossary.md#term-symbol) exists. `RECIPROCAL`/`reciprocal` is the sole spelling. |
| Arithmetic [normalization](../../../glossary.md#term-normalization) | Floating [Tensor](../../../glossary.md#tensor)/Tensor and Tensor/[scalar](../../../glossary.md#term-scalar) use ADD, SUB, MUL, DIV, MIN, MAX, POW; `minimum`/`maximum` remain distinct from reduction `min`/`max`. Selected integral subsets remain coherent and mixed numeric categories require explicit cast. |
| Index taxonomy | [Gather](../../../glossary.md#gather), [Gather Elements](../../../glossary.md#gather-elements), [Gather-ND](../../../glossary.md#gather-nd), Select, [Slice](../../../glossary.md#slice), [Scatter Add](../../../glossary.md#scatter-add), [Scatter Elements](../../../glossary.md#scatter-elements), and [Scatter-ND](../../../glossary.md#scatter-nd) remain distinct. No public `take` or first-class UNSTACK kind exists. |
| Unstack [provenance](../../../glossary.md#provenance) | Public unstack remains ordered independent SELECT composition, never a shared [producer](../../../glossary.md#term-producer). |
| [Mask](../../../glossary.md#term-boolean-mask) [broadcasting](../../../glossary.md#broadcasting) | Masked sum/[mean](../../../glossary.md#term-mean) use ordinary right-aligned broadcasting and no [heuristic](../../../glossary.md#term-heuristic) [axis](../../../glossary.md#term-axis) mapper. |
| Prefix [fixture](../../../glossary.md#term-fixture) | `fromStrictFlatPrefix`/`fromCyclicFlatPrefix` occur only in package-private test-source `TensorTestData`; `TensorFactory` exposes neither. |
| Construction [ownership](../../../glossary.md#term-ownership-meanings) | TensorFactory, TensorRandoms, package-private TensorRanges, and [constant](../../../glossary.md#term-constant)/import paths remain separated. |
| Typed scalar | [ScalarValue](../../../glossary.md#typed-scalar-value--scalarvalue) is the only supported exact scalar-attribute [payload](../../../glossary.md#artifact-envelope-and-atomic-publication), including [padding](../../../glossary.md#padding) constants. |
| Signed slicing/[layout](../../../glossary.md#layout) | Public slice steps are signed non-zero; resolved [view](../../../glossary.md#term-view-meanings) geometry remains limited to representable positive-[stride](../../../glossary.md#term-stride) cases. |
| Primitive take side effects | Primitive-array `take` and [eager](../../../glossary.md#term-eager-evaluation) index-allocation/prevalidation behavior remain absent. |
| General overlap-add | `foldAxis` is public and general; no [operation](../../../glossary.md#operation)-specific backward kind accompanies it. |
| Symbolic products | [Dimension](../../../glossary.md#dimension) expressions represent selected addition, product, and division-derived [Shapes](../../../glossary.md#shape) without [model](../../../glossary.md#term-model)-time [runtime](../../../glossary.md#term-runtime) [binding](../../../glossary.md#term-binding). |
| [Signature](../../../glossary.md#term-signature) hardening | Family-owned typed signatures reject invalid kind/attributes pairs and validate local [occurrence](../../../glossary.md#term-occurrence) counts. |
| Multi-output [identity](../../../glossary.md#term-identity-meanings) | [Dropout](../../../glossary.md#dropout--inverted-dropout), [batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters) training, top-K, and attention [weights](../../../glossary.md#term-weight) use exact shared producer identity and output indices with focused public carriers. |
| Backward taxonomy | No operation kind name contains `_BACKWARD`; `GraphPhase.BACKWARD` is a [graph](../../../glossary.md#graph) classification, not an operation family. |

## [Adjoint](../../../glossary.md#term-adjoint)-prerequisite closure

The completed [adjoint matrix](adjoint-expressibility-audit.md) selected six reusable public
prerequisites. Current source rechecks close all six:

| Prerequisite | Current evidence and conclusion |
|---|---|
| [Binding](../../../glossary.md#term-binding)-aware [sum-to-Shape](../../../glossary.md#sum-to-shape) | `SUM` accepts `SumToShapeAttrs`; public `sumToShape(Shape)` exists. Closed. |
| [Gather](../../../glossary.md#gather)-compatible scatter-add | `SCATTER_ADD` and public `scatterAdd` preserve unresolved gathered extents and fixed duplicate ADD. Closed. |
| [Slice](../../../glossary.md#slice) placement and dynamic crop | `SLICE_UPDATE` plus `SLICE/CropToShapeAttrs` have public construction. Closed. |
| General fold and dynamic/configurable windows | Public `foldAxis`, typed-[padding](../../../glossary.md#padding) unfold2d, symbolic products, and dynamic fold/unfold [Shapes](../../../glossary.md#shape) exist. Closed. |
| Zero-safe product prerequisite | `CUM_PROD` and both public traversal forms exist. Closed. |
| Same-[occurrence](../../../glossary.md#term-occurrence) attention [weights](../../../glossary.md#term-weight) | Explicit two-output attention and `ScaledDotProductAttentionResult` exist without changing the one-output form. Closed. |

Existing [Scatter Elements](../../../glossary.md#scatter-elements)/[Scatter-ND](../../../glossary.md#scatter-nd), typed zero/one expansion, top-K indices, [dropout](../../../glossary.md#dropout--inverted-dropout) [mask](../../../glossary.md#term-boolean-mask),
[batch](../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-training saved values, and recomputable max-pool selection remain sufficient exactly as the
[matrix](../../../glossary.md#term-matrix-meanings) recorded. No `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP` and no [operation](../../../glossary.md#operation)-specific backward
family appeared. [Compiler](../../../glossary.md#term-compiler) adoption remains downstream work.

## Documentation and terminology consistency

The [Tensor API](../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../glossary.md#term-application-programming-interface-api))</span> and [Compile API](../../../api/compile-api.md) match the
200-method surface, current semantic construction, [producer](../../../glossary.md#term-producer)/[provenance](../../../glossary.md#provenance) contracts, and planned
[compiler](../../../glossary.md#term-compiler) boundary. The [Runtime API](../../../api/runtime-api.md) correctly remains conceptual.
The [glossary](../../../glossary.md) already defines the reusable terms and required no new entry
because this audit introduces no domain term or behavior.

Two documentation-only drifts were corrected within authorized paths:

- [Public API status](../../../api/public-api.md) no longer says the implemented public [`Tensor`](../../../glossary.md#tensor)
  and model surface are absent.
- [Training API](../../../api/training-api.md) now keeps [`Parameter`](../../../glossary.md#term-parameter) and [`Buffer`](../../../glossary.md#term-buffer) in planned
  `extensions/nn`, while training owns parameter groups, [optimizers](../../../glossary.md#term-optimizer), and orchestration.

[Compile](../../../glossary.md#compile), [Runtime](../../../glossary.md#term-runtime), and glossary text required no modification because their current/planned and
[ownership](../../../glossary.md#term-ownership-meanings) statements remain accurate. No [Javadoc](../../../glossary.md#term-javadoc) changed. Review found one blocking Javadoc drift
outside the authorized paths: `GraphValue` says the implemented public [mutable](../../../glossary.md#term-mutable) Tensor is planned.
The final [model](../../../glossary.md#term-model) Javadoc generation remains the rendering check; successful generation cannot make
that semantic status sentence accurate.

## Deferred, rejected, and downstream-owned capabilities

| Classification | Capabilities and [owner](../../../glossary.md#term-owner) |
|---|---|
| Deliberately deferred [model](../../../glossary.md#term-model) breadth | [FLOAT16](../../../glossary.md#term-float16), quantized/sparse types, broader integer arithmetic, einsum/[FFT](../../../glossary.md#alternative-convolution-algorithms), additional [layouts](../../../glossary.md#layout)/convolutions/[pooling](../../../glossary.md#term-pooling)/[losses](../../../glossary.md#loss), attention [dropout](../../../glossary.md#dropout--inverted-dropout), and general library breadth; a future concrete model task must justify each. |
| Rejected from the selected core | Fast [unary](../../../glossary.md#term-unary) [aliases](../../../glossary.md#aliasing-and-arena), `inv`, ambiguous `take`, first-class UNSTACK, prefix-population factory APIs, raw-double semantic attributes, generic tuple outputs, [backend](../../../glossary.md#backend)/[fused](../../../glossary.md#specialization-and-fusion) kinds, and [operation](../../../glossary.md#operation)-specific backward kinds. |
| [Compiler](../../../glossary.md#term-compiler)-owned | [Capture](../../../glossary.md#term-capture-meanings), [graph](../../../glossary.md#graph)-wide [validation](../../../glossary.md#term-validation), symbolic [binding](../../../glossary.md#term-binding) proof, [canonicalization](../../../glossary.md#occurrence-identity-and-canonical-form), [autograd](../../../glossary.md#autograd) rules, saved-output [lifetime](../../../glossary.md#term-lifetime), accumulation, and deferred [derivative](../../../glossary.md#term-derivative) policy. |
| [Planning](../../../glossary.md#term-planning)/[prepare](../../../glossary.md#prepare)/backend-owned | Capability analysis, [backend ownership](../../../glossary.md#backend-ownership), [lowering](../../../glossary.md#lowering), [fusion](../../../glossary.md#term-fusion), [kernel](../../../glossary.md#kernel) choice, numerical algorithms, tolerances, determinism guarantees, and concrete [RNG](../../../glossary.md#graph-rng-state--graphrngstate) bitstreams. |
| [Runtime](../../../glossary.md#term-runtime)-owned | Prepared schedules, physical memory, [device](../../../glossary.md#term-device) [residency](../../../glossary.md#residency), transfers, per-[run state](../../../glossary.md#run-state--runstate), execution, and [publication](../../../glossary.md#term-publication). |
| Extension-owned | [Module](../../../glossary.md#term-module)/[parameter](../../../glossary.md#term-parameter)/[buffer](../../../glossary.md#term-buffer)/train-eval behavior in `extensions/nn`; [optimizers](../../../glossary.md#term-optimizer) and training orchestration in `extensions/training`. |

These deferrals do not block model closure because the current model neither claims nor needs them
to represent the selected baseline honestly.

## Findings, severity, and disposition

Only the following labels are used.

| ID | Label | Evidence, impact, [owner](../../../glossary.md#term-owner), and disposition |
|---|---|---|
| DOC-01 | `DOCUMENTATION_DRIFT` | `docs/api/public-api.md` contradicted implemented [Tensor](../../../glossary.md#tensor)/source inventory. Corrected in this task; no behavior changed. |
| DOC-02 | `DOCUMENTATION_DRIFT` | `docs/api/training-api.md` assigned planned [`Parameter`](../../../glossary.md#term-parameter) [ownership](../../../glossary.md#term-ownership-meanings) to training. Corrected to the established nn-to-training boundary; no architecture change. |
| DEF-01 | `NON_BLOCKING_DEFERRED` | [Compiler](../../../glossary.md#term-compiler) [capture](../../../glossary.md#term-capture-meanings), [autograd](../../../glossary.md#autograd) adoption, [binding](../../../glossary.md#term-binding) proof, and [derivative](../../../glossary.md#term-derivative) boundary policies are not implemented. Their later owner and non-[model](../../../glossary.md#term-model) impact are explicit. |
| DEF-02 | `NON_BLOCKING_DEFERRED` | [Backend](../../../glossary.md#backend) execution, numerical tolerances/determinism, and [RNG](../../../glossary.md#graph-rng-state--graphrngstate) bitstreams remain unimplemented. Model [metadata](../../../glossary.md#term-metadata) does not claim them. |
| GAP-01 | `BLOCKING` | `GraphValue.java` describes the current public [mutable](../../../glossary.md#term-mutable) Tensor [API](../../../glossary.md#term-application-programming-interface-api) as planned. Evidence: current `Tensor.java`, the architecture [invariant](../../../glossary.md#term-invariant), and `GraphValue.java` lines 9–12. Impact: a core [graph](../../../glossary.md#graph) [Javadoc](../../../glossary.md#term-javadoc) contract misstates the status of the adjacent public model. Owner: one bounded model Javadoc correction with focused review and [validation](../../../glossary.md#term-validation). [Dependency](../../../glossary.md#term-dependency): task 0024 audit evidence. First follow-up: concise Draft row 0024A; this task does not create its detailed specification or edit Java. |
| NC-01 | `NO_CHANGE_CONFIRMED` | Full production/test, kind/[signature](../../../glossary.md#term-signature), public surface, cleanup, multi-output, and [adjoint](../../../glossary.md#term-adjoint) inventories found no behavioral model gap or architecture conflict. |

There is exactly one `BLOCKING` finding. It is documentation-only, requires no architecture
decision, and keeps the selected model milestone open until the bounded follow-up is complete.

## [Checkpoint](../../../glossary.md#term-checkpoint) evidence and [model](../../../glossary.md#term-model)-milestone decision

The single required combined [Gradle](../../../glossary.md#term-gradle) checkpoint ran after the substantive audit and documentation
were stable:

```bash
./gradlew test :testing:architecture-tests:test :modules:model:javadoc
```

It completed successfully in two seconds. Gradle reported 39 actionable tasks: 2 executed and 37
up-to-date. The explicit architecture-test and model-[Javadoc](../../../glossary.md#term-javadoc) tasks were up-to-date and the root
test selection deduplicated them as expected. XML report evidence records 1,016 model tests across
127 suites and one [architecture test](../../../glossary.md#term-architecture-test), with zero failures, errors, or skips: 1,017 tests across
128 suites total.

The root aggregators and every project other than `modules:model` and
`testing:architecture-tests` had no tests: [`backends`](../../../glossary.md#backend), `extensions`, [`modules`](../../../glossary.md#term-module), `testing`, `tools`,
`backends:openblas-provider`, `backends:cpu`, `backends:metal`, `backends:cuda`,
`extensions:onnx`, `extensions:training`, `modules:backend-contract`, `modules:compiler`,
`modules:config`, `modules:engine`, `modules:planning`, `modules:prepare`, `modules:runtime`,
`modules:trace`, `testing:backend-conformance`, `testing:integration-tests`, `tools:benchmarks`,
`tools:cli`, and `tools:tuning`.

Final non-Gradle [validation](../../../glossary.md#term-validation) covers repository Markdown, whitespace, the exact seven authorized
changed paths, absence of Java/architecture/build/cross-module changes, absence of a detailed
0024A or later task file, preserved completed history, and synchronized Complete/Open/Draft
status.

Model-milestone decision: **Open**. Task 0024 itself may become Complete because the requested
audit and checkpoint completed, but the selected `modules/model` milestone remains open. The first
bounded frontier is Draft task 0024A, a `GraphValue` [Tensor](../../../glossary.md#tensor)-status Javadoc correction; no detailed
task specification is created here. `modules/trace` remains the next project area after that gap
is closed.

### Closure addendum — 2026-07-13

Task [0024A](tasks/0024a-graph-value-tensor-status-javadoc-correction.md) resolved GAP-01, the
audit's sole blocker. The `GraphValue` type [Javadoc](../../../glossary.md#term-javadoc) now calls [`Tensor`](../../../glossary.md#tensor) the current public [mutable](../../../glossary.md#term-mutable)
[API](../../../glossary.md#term-application-programming-interface-api) object instead of planned. A direct source diff confirms that this one status word is the only
`GraphValue.java` change: imports, declaration, record components, constructor, accessors,
[validation](../../../glossary.md#term-validation), messages, and surrounding contracts are unchanged.

Final `:modules:model:javadoc` generation passed, and inspection of the generated `GraphValue`
page confirmed the corrected current-status wording, the [immutable](../../../glossary.md#term-immutable) [graph](../../../glossary.md#graph)-local value boundary,
and the absence of the stale phrase. Repository Markdown, links, anchors, fences, final newlines,
trailing whitespace, task-specific six-path [scope](../../../glossary.md#term-scope), synchronized status, and `git diff --check`
also passed. Task 0024's successful 1,017-test repository [checkpoint](../../../glossary.md#term-checkpoint) was reused because 0024A
changed no executable Java; no Java test suite was repeated.

The original `BLOCKING_GAP` verdict above remains the historical result at audit time. With its
only blocker resolved by 0024A, the selected `modules/model` capability milestone is now
**Complete**. No later detailed task specification was created.
