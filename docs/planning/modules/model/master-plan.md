# [Model](../../../glossary.md#term-model) Master Plan

**Editorial reading key (later context, not a status update).**

Use this map to distinguish responsibility for meaning from responsibility for integration and resources. Model defines Tensor descriptions, operations and immutable graph records. Compiler constructs/transforms graphs; downstream components turn admitted descriptions into executable work and invocation storage. A public Tensor is neither a graph-local ValueId nor its stored payload: its optional host-storage association borrows a wrapper. A leaf simply lacks a producing expression, even when no values have been allocated. The historical task rows and execution gates remain separate from the current reference pages linked here; a Complete Model row alone is not proof of numeric execution.

Focused current reading: [Current public Tensor state](../../../reference/tensor/identity.md#public-tensor-state); [Current graph representation](../../../reference/tensor/compiled-graph.md#graph-values-and-compiled-nodes). The original record below retains its own scope, decisions and evidence.


## Goal and authority

This is the current implementation map for [Model](../../../glossary.md#term-model) [identity](../../../glossary.md#term-identity-meanings), [descriptors](../../../glossary.md#term-descriptor), [provenance](../../../glossary.md#provenance), value types,
[operation](../../../glossary.md#operation) semantics, [host storage](../../../glossary.md#host-storage), and [immutable](../../../glossary.md#term-immutable) [graph](../../../glossary.md#graph) records.

```text
Model meaning and metadata -> Compiler graph work -> backend capability and execution
```

[ARCHITECTURE.md](../../../../ARCHITECTURE.md) is authoritative. Applicable headings are
[Core invariants](../../../../ARCHITECTURE.md#core-invariants),
[Fixed recurrent scan without graph regions](../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)<span data-glossary-navigation="true"> ([scan glossary](../../../glossary.md#term-cumulative-scan))</span>,
[Fixed family and current Model surface](../../../architecture/contracts/recurrent-scan.md#fixed-family-and-current-model-surface),
[Static Shape and runtime-value boundary](../../../architecture/contracts/recurrent-scan.md#static-shape-and-runtime-value-boundary)<span data-glossary-navigation="true"> ([Shape glossary](../../../glossary.md#shape); [runtime glossary](../../../glossary.md#term-runtime))</span>,
[Purity and lifecycle ownership](../../../architecture/contracts/recurrent-scan.md#purity-and-lifecycle-ownership)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../glossary.md#lifecycle); [ownership glossary](../../../glossary.md#term-ownership-meanings))</span>,
[`modules/model`](../../../architecture/contracts/foundational-modules.md#modulesmodel), and
[Dependency rules](../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../glossary.md#term-dependency))</span>. See the focused
[module](../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([module glossary](../../../glossary.md#term-module))</span>, [dependency](../../../architecture/dependency-rules.md),
and [capability](capabilities.md) guides.

## [Scope](../../../glossary.md#term-scope) and non-goals

[Model](../../../glossary.md#term-model) owns value types, logical [layouts](../../../glossary.md#layout), identifiers, [descriptors](../../../glossary.md#term-descriptor), public [Tensor](../../../glossary.md#tensor) state, [borrowed](../../../glossary.md#ownership-borrowing-and-lease)
[host storage](../../../glossary.md#host-storage), [operation](../../../glossary.md#operation) meanings, expression [provenance](../../../glossary.md#provenance), and [immutable](../../../glossary.md#term-immutable) [graph](../../../glossary.md#graph) records.

Model does not own [compilation](../../../glossary.md#compiler-engine-and-execution-stage-names), differentiation traversal, [backend capability](../../../glossary.md#backend-capability), physical/[device](../../../glossary.md#term-device)
storage, prepared/[runtime](../../../glossary.md#term-runtime) state, route selection, execution, training, or [persistence](../../../glossary.md#term-persistence).

## Stable invariants and boundaries

- [`Tensor`](../../../glossary.md#tensor) has immutable identity, descriptor, label, and expression provenance; it is not an
  intermediate-representation node. Its synchronized optional borrowed host-storage association
  is its only mutable field, and it owns no gradient, runtime, device, or graph-local state.<span data-glossary-navigation="true"> ([identity glossary](../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../glossary.md#term-descriptor); [provenance glossary](../../../glossary.md#provenance); [node glossary](../../../glossary.md#node); [borrowed glossary](../../../glossary.md#ownership-borrowing-and-lease); [host glossary](../../../glossary.md#term-host); [mutable glossary](../../../glossary.md#term-mutable); [gradient glossary](../../../glossary.md#term-gradient); [runtime glossary](../../../glossary.md#term-runtime); [device glossary](../../../glossary.md#term-device))</span>
- A derived [`TensorProducer`](../../../glossary.md#producer--tensorproducer) owns one immutable operation occurrence, ordered Tensor inputs,
  ordered output descriptors, and the canonical exact Tensor wrapper for every indexed output.
  Multi-output results share that producer; reconstruction of an equal output wrapper is invalid.<span data-glossary-navigation="true"> ([occurrence glossary](../../../glossary.md#term-occurrence); [canonical glossary](../../../glossary.md#term-canonical-form); [producer glossary](../../../glossary.md#term-producer))</span>
- [`Shape`](../../../glossary.md#shape) and [`Dimension`](../../../glossary.md#dimension) values support [rank-zero](../../../glossary.md#axis-rank-scalar-vector-and-matrix) and zero extents, non-negative static
  `long` extents, and canonical symbolic expressions. Locally provable incompatibilities fail
  during construction; binding-dependent obligations remain explicit for Compiler validation.<span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler); [validation glossary](../../../glossary.md#term-validation))</span>
- `TensorDescriptor` keeps data type, Shape, gradient eligibility, and resolved or unresolved
  logical layout together. Numeric layout geometry is resolved only when justified by known
  Shape/stride/offset facts; it is not a storage-alias, materialization, or execution promise.<span data-glossary-navigation="true"> ([data type glossary](../../../glossary.md#data-type--datatype); [layout glossary](../../../glossary.md#layout); [stride glossary](../../../glossary.md#term-stride); [alias glossary](../../../glossary.md#aliasing-and-arena); [materialization glossary](../../../glossary.md#materialization))</span>
- The current types are [`FLOAT64`](../../../glossary.md#term-float64), [`FLOAT32`](../../../glossary.md#term-float32), [`BFLOAT16`](../../../glossary.md#term-bfloat16), [`FLOAT16`](../../../glossary.md#term-float16), [`INT64`](../../../glossary.md#data-type--datatype), [`INT32`](../../../glossary.md#data-type--datatype), and [`BOOL`](../../../glossary.md#data-type--datatype).
  Exact typed scalar values preserve their represented bits. Axis-bearing APIs normalize and
  validate axes at their public boundary; numerical, accumulation, special-value, tie, and
  duplicate policies remain operation-family semantics.<span data-glossary-navigation="true"> ([scalar glossary](../../../glossary.md#term-scalar); [represented bits glossary](../../../glossary.md#term-represented-bits); [Axis glossary](../../../glossary.md#term-axis))</span>
- [`Operation`](../../../glossary.md#operation) owns backend-independent represented-value meaning and exact kind/attribute and
  cardinality signatures. It never owns backend support, lowering, algorithms, routes, or
  execution. Tensor expression construction validates descriptor-visible type, rank, axis, and
  Shape constraints without reading tensor values or executing an operation.<span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend); [cardinality glossary](../../../glossary.md#term-cardinality); [signatures glossary](../../../glossary.md#term-signature); [lowering glossary](../../../glossary.md#lowering); [rank glossary](../../../glossary.md#term-rank))</span>
- Public API stays minimal and domain-owned. Helpers are package-private unless a focused public
  namespace is needed; generic registries, service locators, and catch-all packages are absent.<span data-glossary-navigation="true"> ([API glossary](../../../glossary.md#term-application-programming-interface-api))</span>
- [Graph](../../../glossary.md#graph) records and [`CompiledGraphModel`](../../../glossary.md#compiled-graph--compiledgraphmodel) are [immutable](../../../glossary.md#term-immutable). [Publication](../../../glossary.md#term-publication) [binding](../../../glossary.md#term-binding) is a separate
  [`TensorId`](../../../glossary.md#tensorid)-to-[`ValueId`](../../../glossary.md#logical-valueid-versus-physical-memory-slot) value; graph capture, inference, optimization, gradients, and
  publication planning remain Compiler-owned.<span data-glossary-navigation="true"> ([graph capture glossary](../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../glossary.md#term-inference); [planning glossary](../../../glossary.md#term-planning))</span>
- The fixed recurrent family is exactly time-major RNN-tanh, reset-after GRU, and LSTM with one
  forward/reverse direction, fully static Shapes, runtime rank-one <span data-glossary-navigation="true"> ([time-major glossary](../../../glossary.md#term-time-major); [RNN glossary](../../../glossary.md#term-recurrent-neural-network-rnn); [tanh glossary](../../../glossary.md#term-tanh); [GRU glossary](../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../glossary.md#term-long-short-term-memory-lstm); [rank-one glossary](../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`INT64` valid lengths, dense
  outputs, explicit state/parameter inputs, and canonical multi-output provenance. Construction
  lives in stateless <span data-glossary-navigation="true"> ([parameter glossary](../../../glossary.md#term-parameter))</span>`model.tensor.RecurrentScan`, not on `Tensor`; there is no graph region,
  callback, hidden state, execution claim, or general scan body.<span data-glossary-navigation="true"> ([callback glossary](../../../glossary.md#term-callback); [general scan glossary](../../../glossary.md#general-differentiable-scan))</span>
- Model exposes forward algebra needed by differentiation but owns no backward kinds or lifecycle.
  Compiler owns gradient policy and traversal. Recurrent backpropagation through time and Conv3d
  gradients remain fail-closed downstream until their separate owners complete them.<span data-glossary-navigation="true"> ([owners glossary](../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Model glossary](../../../glossary.md#term-model); [lifecycle glossary](../../../glossary.md#lifecycle); [backpropagation through time glossary](../../../glossary.md#term-backpropagation-through-time-bptt); [Conv3d glossary](../../../glossary.md#conv3d--ncdhw); [fail-closed glossary](../../../glossary.md#term-fail-closed))</span>

## Dependencies and package map

[Model](../../../glossary.md#term-model) production uses only the [JDK](../../../glossary.md#term-java-development-kit-jdk). It must not depend on [Planning](../../../glossary.md#term-planning), [Compiler](../../../glossary.md#term-compiler), [Runtime](../../../glossary.md#term-runtime), [Prepare](../../../glossary.md#prepare),
[Engine](../../../glossary.md#term-engine), concrete [backends](../../../glossary.md#backend), or [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training. Packages remain acyclic: `datatype` and [`shape`](../../../glossary.md#shape) are
leaves; [`layout`](../../../glossary.md#layout) and `storage` use foundations; [`operation`](../../../glossary.md#operation) does not use [Tensor](../../../glossary.md#tensor)/[graph](../../../glossary.md#graph); `tensor`
and `graph` compose them.

| Package | [Ownership](../../../glossary.md#term-ownership-meanings) and public boundary |
|---|---|
| `model.datatype` | Data-type [metadata](../../../glossary.md#term-metadata), promotion, bit conversion, and exact [scalar](../../../glossary.md#term-scalar) values. |
| `model.shape` | Static/symbolic extents, [immutable](../../../glossary.md#term-immutable) Shapes, [axes](../../../glossary.md#term-axis), and local [broadcasting](../../../glossary.md#broadcasting). |
| `model.layout` | Resolved logical geometry and layout classification. |
| `model.storage` | [Borrowed](../../../glossary.md#ownership-borrowing-and-lease) [host](../../../glossary.md#term-host)-visible storage; no [device](../../../glossary.md#term-device) [residency](../../../glossary.md#residency) or allocation [lifecycle](../../../glossary.md#lifecycle). |
| `model.tensor` | Tensor [identity](../../../glossary.md#term-identity-meanings)/state, [descriptors](../../../glossary.md#term-descriptor), factories, [eager](../../../glossary.md#term-eager-evaluation) host initialization, expression construction, [producers](../../../glossary.md#term-producer)/results, and focused `RecurrentScan`. |
| `model.operation` | Operation/attribute/[signature](../../../glossary.md#term-signature) foundations and family-owned semantics. |
| `model.operation.elementwise.*` | [Binary](../../../glossary.md#term-binary-descriptor-and-metadata), [unary](../../../glossary.md#term-unary), classification, scalar, comparison, logical, selection, and cast families. |
| `model.operation.reduction`, `.scan`, `.normalization`, `.loss` | Reduction/statistics, cumulative scans, [softmax](../../../glossary.md#softmax--log-softmax)/[normalization](../../../glossary.md#term-normalization), and [loss](../../../glossary.md#loss) meanings. |
| `model.operation.linalg`, `.attention`, `.convolution`, `.pooling` | [Linear](../../../glossary.md#term-linear-meanings) algebra, attention, and [rank](../../../glossary.md#term-rank)-specific channels-first [spatial](../../../glossary.md#term-spatial-axes) meanings. |
| `model.operation.random`, `.ordering`, `.layout`, `.index`, `.recurrent` | Explicit graph [RNG](../../../glossary.md#graph-rng-state--graphrngstate), ordering, [view](../../../glossary.md#term-view-meanings)/layout, indexing/scatter, and fixed recurrent meanings. |
| `model.graph` | Immutable graph values/[nodes](../../../glossary.md#node)/model, phases, identifiers, and [publication](../../../glossary.md#term-publication) [binding](../../../glossary.md#term-binding). |

`linear`, [embedding](../../../glossary.md#embedding), [NCW](../../../glossary.md#term-ncw) Conv1d, and NCW Pool1d are visible compositions. [NCHW](../../../glossary.md#term-nchw)/[NCDHW](../../../glossary.md#conv3d--ncdhw) [convolution](../../../glossary.md#term-channels-first-convolution-layers)
and [pooling](../../../glossary.md#term-pooling) are rank-specific first-class families; no arbitrary-rank `ConvNd`, `PoolNd`, or
`WindowNd` is public. Eager host randomness and [graph RNG state](../../../glossary.md#graph-rng-state--graphrngstate) stay separate; normalization and
loss own no mode, session, or hidden [mutable](../../../glossary.md#term-mutable) statistics.

## Task list

The table is the ordered queue and status source. Tasks 0030, 0031, 0032A, and 0032 are Complete;
0026 is Superseded by the delivered low-type semantics consolidated in 0032. The [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) edge
`0032A -> 0032` is closed. The approved
named-arithmetic-site [DAZ](../../../glossary.md#term-daz)/[FTZ](../../../glossary.md#term-ftz) rule for [`FLOAT32`](../../../glossary.md#term-float32), [`BFLOAT16`](../../../glossary.md#term-bfloat16), and [`FLOAT16`](../../../glossary.md#term-float16) (forbidden for
[`FLOAT64`](../../../glossary.md#term-float64)) is implemented in the main worktree. The isolated full build, source-matched main
native rebuild, independent Class C review, and final documentation reconciliation passed.
[ADR](../../../glossary.md#term-architecture-decision-record-adr) 0026 is Accepted. [Model](../../../glossary.md#term-model) 0033 is Complete after the [owner](../../../glossary.md#term-owner)'s post-review decision to preserve
stable [finite](../../../glossary.md#term-finite) [softmax](../../../glossary.md#softmax--log-softmax), [CPU](../../../glossary.md#term-cpu)/[conformance](../../../glossary.md#term-conformance) implementation, and independent approval. The historical
common base was
`9fbd0cc36f2f187d1a658671cb55710ff13a5ed7`; the [0032 brief](tasks/0032-profile-free-numerical-semantics-reset.md)
records its serial integration and [validation](../../../glossary.md#term-validation). Git history records [publication](../../../glossary.md#term-publication) separately.

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [DataType model](tasks/0001-data-type-model.md)<span data-glossary-navigation="true"> ([DataType glossary](../../../glossary.md#data-type--datatype))</span> | Complete | - | Delivered. |
| 0002 | [Shape and dimension model](tasks/0002-shape-and-dimension-model.md)<span data-glossary-navigation="true"> ([Shape glossary](../../../glossary.md#shape); [dimension glossary](../../../glossary.md#dimension))</span> | Complete | - | Delivered. |
| 0003 | [Layout descriptor model](tasks/0003-layout-descriptor-model.md)<span data-glossary-navigation="true"> ([Layout glossary](../../../glossary.md#layout))</span> | Complete | 0002 | Delivered. |
| 0003A | [Data type package migration](tasks/0003a-data-type-package-migration.md)<span data-glossary-navigation="true"> ([Data type glossary](../../../glossary.md#data-type--datatype))</span> | Complete | 0001 | Delivered. |
| 0003B | [Shape package migration](tasks/0003b-shape-package-migration.md) | Complete | - | Delivered. |
| 0003C | [Layout package migration](tasks/0003c-layout-package-migration.md) | Complete | 0003B | Delivered. |
| 0004 | [Typed identifiers](tasks/0004-typed-identifiers.md) | Complete | 0003A–0003C | Delivered. |
| 0005 | [Operation semantic foundation](tasks/0005-operation-semantic-foundation.md) | Complete | - | Delivered. |
| 0006 | [Operation model](tasks/0006-operation-model.md) | Complete | 0005 | Delivered. |
| 0007 | [Tensor descriptor model](tasks/0007-tensor-descriptor-model.md)<span data-glossary-navigation="true"> ([Tensor descriptor glossary](../../../glossary.md#tensor-descriptor))</span> | Complete | 0001–0003, 0003A–0003C | Delivered. |
| 0008 | [Graph value and node model](tasks/0008-graph-value-and-node-model.md)<span data-glossary-navigation="true"> ([Graph value glossary](../../../glossary.md#graph-value); [node glossary](../../../glossary.md#node))</span> | Complete | 0004, 0006, 0007 | Delivered. |
| 0009 | [Compiled graph model](tasks/0009-compiled-graph-model.md)<span data-glossary-navigation="true"> ([Compiled graph glossary](../../../glossary.md#compiled-graph--compiledgraphmodel))</span> | Complete | 0008 | Delivered. |
| 0010 | [Host storage abstraction](tasks/0010-host-storage-abstraction.md)<span data-glossary-navigation="true"> ([Host storage glossary](../../../glossary.md#host-storage))</span> | Complete | 0001, 0003A | Delivered. |
| 0011 | [Public Tensor skeleton](tasks/0011-public-tensor-skeleton.md) | Complete | 0004, 0007, 0010 | Delivered. |
| 0012 | [Tensor factory foundation](tasks/0012-tensor-factory.md)<span data-glossary-navigation="true"> ([Tensor factory glossary](../../../glossary.md#tensor-factory))</span> | Complete | 0010, 0011 | Delivered. |
| 0012A | [JVM-managed heap host storage allocation](tasks/0012a-host-storage-allocation.md)<span data-glossary-navigation="true"> ([JVM glossary](../../../glossary.md#term-java-virtual-machine-jvm); [heap glossary](../../../glossary.md#term-heap))</span> | Complete | 0010, 0012 | Delivered. |
| 0012B | [Flat typed tensor import](tasks/0012b-flat-typed-tensor-import.md) | Complete | 0012A | Delivered. |
| 0012C | [Nested typed tensor import](tasks/0012c-nested-typed-tensor-import.md) | Complete | 0012B | Delivered. |
| 0012D | [Constant tensor creation](tasks/0012d-constant-tensor-creation.md)<span data-glossary-navigation="true"> ([Constant glossary](../../../glossary.md#term-constant))</span> | Complete | 0012B | Delivered. |
| 0012E | [Range and prefix population](tasks/0012e-range-and-prefix-population.md) | Complete | 0012B | Delivered. |
| 0012F | [Random tensor creation](tasks/0012f-random-tensor-creation.md) | Complete | 0012B | Delivered. |
| 0012G | [Uniform random tensor creation](tasks/0012g-uniform-random-tensor-creation.md) | Complete | 0012F | Delivered. |
| 0012H | [Integral random tensor creation](tasks/0012h-integral-random-tensor-creation.md) | Complete | 0012F | Delivered. |
| 0012I | [Bernoulli random tensor creation](tasks/0012i-bernoulli-random-tensor-creation.md)<span data-glossary-navigation="true"> ([Bernoulli glossary](../../../glossary.md#term-bernoulli-distribution))</span> | Complete | 0012F | Delivered. |
| 0013 | [Tensor provenance skeleton](tasks/0013-tensor-provenance-skeleton.md) | Complete | 0006, 0011, 0012 | Delivered. |
| 0013A | [Full-value and identity-matrix tensor creation](tasks/0013a-full-value-and-identity-matrix-tensor-creation.md)<span data-glossary-navigation="true"> ([matrix glossary](../../../glossary.md#term-matrix-meanings))</span> | Complete | 0012B, 0012D | Delivered. |
| 0014A | [Binary arithmetic semantic kinds](tasks/0014a-binary-arithmetic-semantic-kinds.md)<span data-glossary-navigation="true"> ([Binary glossary](../../../glossary.md#term-binary-descriptor-and-metadata))</span> | Complete | 0005, 0006 | Delivered. |
| 0014B | [Binary arithmetic Tensor expressions](tasks/0014b-binary-arithmetic-tensor-expressions.md) | Complete | 0013, 0014A | Delivered. |
| 0014C | [Unary elementwise semantic kinds](tasks/0014c-unary-elementwise-semantic-kinds.md)<span data-glossary-navigation="true"> ([Unary glossary](../../../glossary.md#term-unary); [elementwise glossary](../../../glossary.md#term-elementwise))</span> | Complete | 0005, 0006 | Delivered. |
| 0014D | [Unary elementwise Tensor expressions](tasks/0014d-unary-elementwise-tensor-expressions.md) | Complete | 0013, 0014C | Delivered. |
| 0014E | [Scalar arithmetic and clamp semantics](tasks/0014e-scalar-arithmetic-and-clamp-semantics.md)<span data-glossary-navigation="true"> ([Scalar glossary](../../../glossary.md#term-scalar))</span> | Complete | 0005, 0006 | Delivered. |
| 0014F | [Scalar arithmetic and clamp Tensor expressions](tasks/0014f-scalar-arithmetic-and-clamp-tensor-expressions.md) | Complete | 0013, 0014E | Delivered. |
| 0015A | [Binary comparison semantic kinds](tasks/0015a-binary-comparison-semantic-kinds.md) | Complete | 0005, 0006 | Delivered. |
| 0015B | [Binary comparison Tensor expressions](tasks/0015b-binary-comparison-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0015A | Delivered. |
| 0015C | [Boolean logical semantic kinds](tasks/0015c-boolean-logical-semantic-kinds.md)<span data-glossary-navigation="true"> ([Boolean glossary](../../../glossary.md#data-type--datatype))</span> | Complete | 0005, 0006 | Delivered. |
| 0015D | [Boolean logical Tensor expressions](tasks/0015d-boolean-logical-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0015C | Delivered. |
| 0015E | [Where selection semantic kind](tasks/0015e-where-selection-semantic-kind.md) | Complete | 0005, 0006 | Delivered. |
| 0015F | [Where selection Tensor expression](tasks/0015f-where-selection-tensor-expression.md) | Complete | 0001, 0002, 0013, 0015E | Delivered. |
| 0015G | [Cast semantic kind and attributes](tasks/0015g-cast-semantic-kind-and-attributes.md) | Complete | 0001, 0005, 0006 | Delivered. |
| 0015H | [Cast Tensor expression](tasks/0015h-cast-tensor-expression.md) | Complete | 0001, 0013, 0015G | Delivered. |
| 0016A | [Reduction semantic kinds and attributes](tasks/0016a-reduction-semantic-kinds-and-attributes.md) | Complete | 0005, 0006 | Delivered. |
| 0016B | [Sum, mean, and product Tensor expressions](tasks/0016b-sum-mean-and-product-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016A | Delivered. |
| 0016C | [Min and max Tensor reduction expressions](tasks/0016c-min-and-max-tensor-reduction-expressions.md) | Complete | 0001, 0002, 0013, 0014A, 0014B, 0016A, 0016B | Delivered. |
| 0016D | [Boolean all and any Tensor expressions](tasks/0016d-boolean-all-and-any-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0015C, 0015D, 0016A, 0016B, 0016C | Delivered. |
| 0016E | [Arg-max Tensor expressions](tasks/0016e-arg-max-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016A, 0016B, 0016C, 0016D | Delivered. |
| 0016F | [Masked reduction semantics and axis mapping](tasks/0016f-masked-reduction-semantics-and-axis-mapping.md)<span data-glossary-navigation="true"> ([Masked reduction glossary](../../../glossary.md#masked-reduction); [axis glossary](../../../glossary.md#term-axis))</span> | Complete | 0005, 0006, 0016A | Delivered. |
| 0016F1 | [Masked sum and mean Tensor expressions](tasks/0016f1-masked-sum-and-mean-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016B, 0016F | Delivered. |
| 0016G | [Cumulative-sum semantic kind and attributes](tasks/0016g-cumulative-sum-semantic-kind-and-attributes.md) | Complete | 0005, 0006 | Delivered. |
| 0016H | [Cumulative-sum Tensor expressions](tasks/0016h-cumulative-sum-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016G | Delivered. |
| 0016I | [Softmax semantic kinds and attributes](tasks/0016i-softmax-semantic-kinds-and-attributes.md) | Complete | 0005, 0006 | Delivered. |
| 0016J | [Softmax Tensor expressions](tasks/0016j-softmax-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016I | Delivered. |
| 0017A | [Contiguous semantic kind](tasks/0017a-contiguous-semantic-kind.md)<span data-glossary-navigation="true"> ([Contiguous glossary](../../../glossary.md#term-contiguous))</span> | Complete | 0005, 0006 | Delivered. |
| 0017B | [Contiguous Tensor expression](tasks/0017b-contiguous-tensor-expression.md) | Complete | 0002, 0003, 0007, 0011–0013, 0017A | Delivered. |
| 0017C | [Reshape and expand semantics](tasks/0017c-reshape-and-expand-semantics.md)<span data-glossary-navigation="true"> ([Reshape and expand semantics glossary](../../../glossary.md#reshape-and-expand-semantics))</span> | Complete | 0002, 0005, 0006 | Delivered. |
| 0017D | [Reshape Tensor expressions](tasks/0017d-reshape-tensor-expressions.md) | Complete | 0002, 0003, 0007, 0011–0013, 0017C | Delivered. |
| 0017D1 | [Expand Tensor expressions](tasks/0017d1-expand-tensor-expressions.md) | Complete | 0002, 0003, 0007, 0011–0013, 0017C | Delivered. |
| 0017E | [Axis-transform semantics](tasks/0017e-axis-transform-semantics.md) | Complete | 0002, 0005, 0006 | Delivered. |
| 0017F | [Permute and transpose Tensor expressions](tasks/0017f-permute-and-transpose-tensor-expressions.md)<span data-glossary-navigation="true"> ([transpose glossary](../../../glossary.md#term-transpose))</span> | Complete | 0002, 0003, 0013, 0017E | Delivered. |
| 0017F1 | [Expand-dimensions and squeeze Tensor expressions](tasks/0017f1-expand-dimensions-and-squeeze-tensor-expressions.md) | Complete | 0002, 0003, 0013, 0017E | Delivered. |
| 0017G | [Slice semantics](tasks/0017g-slice-semantics.md) | Complete | 0002, 0005, 0006 | Delivered. |
| 0017H | [Slice Tensor expressions](tasks/0017h-slice-tensor-expressions.md) | Complete | 0002, 0003, 0013, 0017G | Delivered. |
| 0017I | [Pad and tile semantics](tasks/0017i-pad-and-tile-semantics.md) | Complete | 0001, 0002, 0005, 0006 | Delivered. |
| 0017J | [Pad and tile Tensor expressions](tasks/0017j-pad-and-tile-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0017I | Delivered. |
| 0017K | [Tensor composition semantics](tasks/0017k-tensor-composition-semantics.md)<span data-glossary-navigation="true"> ([Tensor composition glossary](../../../glossary.md#tensor-composition))</span> | Complete | 0002, 0005, 0006 | Delivered. |
| 0017L | [Tensor composition expressions](tasks/0017l-tensor-composition-expressions.md) | Complete | 0001, 0002, 0013, 0017K | Delivered. |
| 0017M | [Unfold and fold semantics](tasks/0017m-unfold-and-fold-semantics.md)<span data-glossary-navigation="true"> ([Unfold and fold glossary](../../../glossary.md#window-transform))</span> | Complete | 0002, 0005, 0006 | Delivered. |
| 0017N | [Unfold and fold Tensor expressions](tasks/0017n-unfold-and-fold-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0017M | Delivered; 0018R removed and 0023D restored public `foldAxis`. |
| 0018A | [Scalar select semantics](tasks/0018a-scalar-select-semantics.md)<span data-glossary-navigation="true"> ([Scalar select glossary](../../../glossary.md#scalar-select))</span> | Complete | 0002, 0005, 0006 | Delivered. |
| 0018B | [Scalar select Tensor expression](tasks/0018b-scalar-select-tensor-expression.md) | Complete | 0002, 0003, 0013, 0018A | Delivered. |
| 0018C | [Axis gather semantics](tasks/0018c-axis-gather-semantics.md)<span data-glossary-navigation="true"> ([gather glossary](../../../glossary.md#gather))</span> | Complete | 0005, 0006 | Delivered. |
| 0018D | [Axis gather Tensor expressions](tasks/0018d-axis-gather-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0018C | Delivered. |
| 0018D1 | [Primitive take convenience](tasks/0018d1-primitive-take-convenience.md) | Complete | 0012B, 0018D | Delivered. |
| 0018E | [Gather-ND semantics](tasks/0018e-gather-nd-semantics.md)<span data-glossary-navigation="true"> ([Gather-ND glossary](../../../glossary.md#gather-nd))</span> | Complete | 0005, 0006 | Delivered. |
| 0018F | [Gather-ND Tensor expressions](tasks/0018f-gather-nd-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0018E | Delivered. |
| 0018G | [Axis scatter semantics](tasks/0018g-axis-scatter-semantics.md) | Complete | 0005, 0006, 0018C | Delivered. |
| 0018H | [Axis scatter Tensor expressions](tasks/0018h-axis-scatter-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0018G | Delivered. |
| 0018I | [Scatter-ND semantics](tasks/0018i-scatter-nd-semantics.md)<span data-glossary-navigation="true"> ([Scatter-ND glossary](../../../glossary.md#scatter-nd))</span> | Complete | 0005, 0006, 0018E | Delivered. |
| 0018J | [Scatter-ND Tensor expression](tasks/0018j-scatter-nd-tensor-expression.md) | Complete | 0001, 0002, 0013, 0018I | Delivered. |
| 0018K | [Operation signature and construction hardening](tasks/0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Operation signature glossary](../../../glossary.md#operation-signature))</span> | Complete | 0005, 0006, 0008 | Delivered. |
| 0018L | [Shared multi-output Tensor provenance](tasks/0018l-shared-multi-output-tensor-provenance.md) | Complete | 0007–0009, 0011–0013, 0018K | Delivered. |
| 0018M | [Symbolic extent expressions](tasks/0018m-symbolic-extent-expressions.md)<span data-glossary-navigation="true"> ([extent glossary](../../../glossary.md#term-extent))</span> | Complete | 0002, 0017C–0017N | Delivered. |
| 0018M1 | [Dynamic extent adoption in pad, tile, and concat](tasks/0018m1-dynamic-extent-adoption.md) | Complete | 0017J, 0017L, 0018M | Delivered. |
| 0018N | [Typed scalar value contract](tasks/0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([Typed scalar value glossary](../../../glossary.md#typed-scalar-value--scalarvalue))</span> | Complete | 0001, 0014E, 0014F, 0017I, 0017J, 0018K | Delivered. |
| 0018O | [Indexing taxonomy and unstack normalization](tasks/0018o-indexing-taxonomy-and-unstack-normalization.md)<span data-glossary-navigation="true"> ([normalization glossary](../../../glossary.md#term-normalization))</span> | Complete | 0017K–0017L, 0018A–0018J, 0018K–0018L | Finalized indexing taxonomy and repeated-`SELECT` unstack. |
| 0018P | [Elementwise semantic cleanup](tasks/0018p-elementwise-semantic-cleanup.md) | Complete | 0014C–0014F, 0018K, 0018N | Delivered. |
| 0018Q | [Masked reduction redesign](tasks/0018q-masked-reduction-redesign.md) | Complete | 0015E–0015F, 0016A–0016F1, 0018M–0018N | Delivered. |
| 0018R | [Slice and window public-contract cleanup](tasks/0018r-slice-and-window-public-contract-cleanup.md) | Complete | 0017G–0017N, 0018K–0018M | Delivered. |
| 0018S | [Tensor factory surface cleanup](tasks/0018s-tensor-factory-surface-cleanup.md) | Complete | 0012–0012I, 0013A, 0018N | Delivered. |
| 0018T | [Scalar arithmetic family normalization](tasks/0018t-scalar-arithmetic-family-normalization.md) | Complete | 0014A–0014B, 0014E–0014F, 0018K, 0018N, 0018P | Delivered. |
| 0018T1 | [Unary numeric gaps and floating diagnostics](tasks/0018t1-unary-numeric-gaps-and-floating-diagnostics.md) | Complete | 0014C–0014D, 0018K, 0018P, 0018T | Delivered. |
| 0018U | [Integral elementwise arithmetic and comparisons](tasks/0018u-integral-elementwise-arithmetic-and-comparisons.md) | Complete | 0001, 0014A–0015B, 0018K, 0018N, 0018T–0018T1 | Delivered. |
| 0018U1 | [Integral reductions and arg-min normalization](tasks/0018u1-integral-reductions-and-arg-min-normalization.md) | Complete | 0016A–0016E, 0018K, 0018U | Delivered. |
| 0018V | [Multi-axis and statistical reductions](tasks/0018v-multi-axis-and-statistical-reductions.md) | Complete | 0016A–0016J, 0018K, 0018M, 0018T1, 0018U1 | Delivered. |
| 0019 | [Matmul semantics and Tensor expression](tasks/0019-matmul-semantics-and-tensor-expression.md) | Complete | 0001–0002, 0005–0007, 0011–0013, 0018K–0018N, 0018T, 0018U–0018V | Delivered. |
| 0019A | [Modern activation semantics and Tensor expressions](tasks/0019a-modern-activation-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([activation glossary](../../../glossary.md#term-activation))</span> | Complete | 0014C–0014D, 0018K, 0018P | Delivered. |
| 0019A1 | [Embedding convenience](tasks/0019a1-embedding-convenience.md)<span data-glossary-navigation="true"> ([Embedding glossary](../../../glossary.md#embedding))</span> | Complete | 0018K, 0018O | Delivered. |
| 0019A2 | [One-hot encoding](tasks/0019a2-one-hot-encoding.md)<span data-glossary-navigation="true"> ([One-hot encoding glossary](../../../glossary.md#one-hot-encoding))</span> | Complete | 0001–0002, 0005–0007, 0011–0013, 0018K–0018O | Delivered. |
| 0019B | [Explicit graph RNG state foundation](tasks/0019b-explicit-graph-rng-state-foundation.md)<span data-glossary-navigation="true"> ([graph RNG state glossary](../../../glossary.md#graph-rng-state--graphrngstate))</span> | Complete | 0018K–0018L, 0018N, 0018S | Delivered. |
| 0019B1 | [Explicit graph dropout construction](tasks/0019b1-explicit-graph-dropout-construction.md)<span data-glossary-navigation="true"> ([graph glossary](../../../glossary.md#graph); [dropout glossary](../../../glossary.md#dropout--inverted-dropout))</span> | Complete | 0019B, 0018K–0018L | Delivered. |
| 0019C | [Sort and argsort](tasks/0019c-sort-and-argsort.md) | Complete | 0018K–0018L, 0018U–0018U1 | Delivered. |
| 0019C1 | [Top-K values and indices](tasks/0019c1-top-k-values-and-indices.md) | Complete | 0019C, 0018L | Delivered. |
| 0019D | [Linear convenience](tasks/0019d-linear-convenience.md)<span data-glossary-navigation="true"> ([Linear glossary](../../../glossary.md#term-linear-meanings))</span> | Complete | 0017F, 0014B, 0018K–0018N, 0018T, 0018U, 0019–0019C1 | Delivered. |
| 0019E | [Scaled dot-product attention](tasks/0019e-scaled-dot-product-attention.md)<span data-glossary-navigation="true"> ([Scaled dot-product attention glossary](../../../glossary.md#scaled-dot-product-attention))</span> | Complete | 0016I–0016J, 0018K–0018L, 0018N, 0018Q, 0019 | Delivered. |
| 0020 | [NCHW Conv2d semantics and Tensor expressions](tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([NCHW glossary](../../../glossary.md#term-nchw))</span> | Complete | 0018K–0018L, 0018M–0018M1, 0018N, 0018V, 0019 | Delivered. |
| 0020A | [NCHW Max Pool2d semantics and Tensor expression](tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md) | Complete | 0020, 0017M–0017N, 0018K–0018L, 0018M–0018M1, 0018N, 0018V | Delivered. |
| 0020A1 | [NCHW Average Pool2d semantics and Tensor expression](tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md) | Complete | 0020A | Delivered. |
| 0021 | [Layer normalization semantics and Tensor expressions](tasks/0021-layer-normalization-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([Layer normalization glossary](../../../glossary.md#layer-normalization))</span> | Complete | 0018K, 0018L, 0018N, 0018V | Delivered. |
| 0021A | [RMS normalization semantics and Tensor expressions](tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([RMS normalization glossary](../../../glossary.md#rms-normalization))</span> | Complete | 0018K, 0018L, 0018N, 0018V, 0021 | Delivered. |
| 0021B | [Batch-normalization inference](tasks/0021b-batch-normalization-inference.md)<span data-glossary-navigation="true"> ([Batch glossary](../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [inference glossary](../../../glossary.md#term-inference))</span> | Complete | 0018K, 0018L, 0018N, 0018V, 0021–0021A | Delivered. |
| 0021C | [Batch-normalization training and statistic transition](tasks/0021c-batch-normalization-training-and-statistic-transition.md)<span data-glossary-navigation="true"> ([statistic glossary](../../../glossary.md#term-statistic))</span> | Complete | 0021B, 0018L, 0018N, 0018V | Delivered. |
| 0022 | [Mean-squared-error loss](tasks/0022-mean-squared-error-loss.md)<span data-glossary-navigation="true"> ([loss glossary](../../../glossary.md#loss))</span> | Complete | 0018K, 0018N, 0018V | Delivered. |
| 0022A | [Dense-target categorical cross-entropy with logits](tasks/0022a-dense-target-categorical-cross-entropy-with-logits.md)<span data-glossary-navigation="true"> ([target glossary](../../../glossary.md#term-target))</span> | Complete | 0022, 0016I–0016J | Delivered. |
| 0022B | [Index-target categorical cross-entropy with logits](tasks/0022b-index-target-categorical-cross-entropy-with-logits.md) | Complete | 0022A, 0018O | Delivered. |
| 0023 | [Adjoint expressibility audit](tasks/0023-adjoint-expressibility-audit.md) | Complete | 0006, 0014A–0014F, 0015A–0015H, 0016A–0016J, 0017A–0017N including 0017D1 and 0017F1, 0018A–0019E including 0018D1, 0019A1, and 0019A2, 0020–0022B including 0020A–0020A1 and 0021A–0021C; post-0022B [checkpoint](../../../glossary.md#term-checkpoint) | Selected six generally useful public [adjoint](../../../glossary.md#term-adjoint) prerequisites. |
| 0023A | [Binding-aware sum-to-Shape](tasks/0023a-binding-aware-sum-to-shape.md)<span data-glossary-navigation="true"> ([sum-to-Shape glossary](../../../glossary.md#sum-to-shape))</span> | Complete | 0023 | Delivered. |
| 0023B | [Gather-compatible scatter-add](tasks/0023b-gather-compatible-scatter-add.md) | Complete | 0023, 0023A | Delivered. |
| 0023C | [Slice update and target-relative crop](tasks/0023c-slice-update-and-target-relative-crop.md)<span data-glossary-navigation="true"> ([Slice update glossary](../../../glossary.md#slice-update); [target-relative crop glossary](../../../glossary.md#target-relative-crop))</span> | Complete | 0023, 0023B | Delivered. |
| 0023D | [Public foldAxis and dynamic window transforms](tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md) | Complete | 0023, 0023C | Delivered. |
| 0023E | [Cumulative scan normalization and product](tasks/0023e-cumulative-scan-normalization-and-product.md)<span data-glossary-navigation="true"> ([Cumulative scan glossary](../../../glossary.md#cumulative-scan))</span> | Complete | 0016G–0016H, 0018K, 0018U–0018U1, 0023, 0023D | Delivered. |
| 0023F | [Scaled dot-product attention weights output](tasks/0023f-scaled-dot-product-attention-weights-output.md)<span data-glossary-navigation="true"> ([weights glossary](../../../glossary.md#term-weight))</span> | Complete | 0018K–0018L, 0019E, 0023, 0023E | Delivered. |
| 0024 | [Model capability and contract closure audit](tasks/0024-model-capability-and-contract-closure-audit.md) | Complete | 0001–0023F | Found one [Javadoc](../../../glossary.md#term-javadoc)-only gap, later closed by 0024A. |
| 0024A | [GraphValue Tensor-status Javadoc correction](tasks/0024a-graph-value-tensor-status-javadoc-correction.md) | Complete | 0024 | Closed the selected Model capability milestone. |
| 0025 | [Canonical TensorProducer outputs](tasks/0025-canonical-tensor-producer-outputs.md)<span data-glossary-navigation="true"> ([TensorProducer glossary](../../../glossary.md#producer--tensorproducer))</span> | Complete | 0018L, 0019B1, 0021C, 0023F, 0024A; prerequisite for [Compiler](../../../glossary.md#term-compiler) 0004 | Retained the [canonical](../../../glossary.md#term-canonical-form) [Tensor](../../../glossary.md#tensor) wrapper for every [producer](../../../glossary.md#term-producer) output. |
| 0025A | [Portable floating comparison, extrema, and clamp semantics](tasks/0025a-portable-floating-comparison-extrema-and-clamp-semantics.md) | Complete | 0018N, 0018T, 0018U, 0025; prerequisite for Compiler 0005A | Fixed portable floating comparison, [extrema](../../../glossary.md#saturation-pole-and-extremum), and clamp meanings. |
| 0025B | [Binding-aware expansion](tasks/0025b-binding-aware-expansion.md) | Complete | 0002, 0017C, 0017D1, 0018M, 0023A, 0025A; prerequisite for Compiler 0005B | Retained [binding](../../../glossary.md#term-binding)-dependent EXPAND obligations. |
| 0025C | [Portable functional-scatter reduction semantics](tasks/0025c-portable-functional-scatter-reduction-semantics.md) | Complete | 0018G–0018J, 0018U–0018U1, 0025A; prerequisite for Compiler 0005C | Fixed portable functional-scatter reduction meanings. |
| 0025D | [Dynamic-extent slice extraction and symbolic slice placement](tasks/0025d-dynamic-extent-slice-extraction-and-symbolic-slice-placement.md) | Complete | 0002, 0018M, 0018R, 0023C, 0025C; prerequisite for Compiler 0005C | Added dynamic-length [slice](../../../glossary.md#slice) and symbolic placement forms. |
| 0025E | [Fixed recurrent-scan semantic family and Tensor expressions](tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md) | Complete | accepted [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0021A architecture decision; 0018K–0018L, 0018N, 0019, 0025; current recurrent cell equations | Added fixed flat [RNN](../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../glossary.md#term-long-short-term-memory-lstm) [scan](../../../glossary.md#term-cumulative-scan) semantics and [provenance](../../../glossary.md#provenance). |
| 0025F | [Recurrent-scan expression namespace correction](tasks/0025f-recurrent-scan-expression-namespace-correction.md) | Complete | 0025E; before Compiler 0006A | Moved low-level scan construction to `RecurrentScan`. |
| 0025G | [NCW Conv1d composition](tasks/0025g-ncw-conv1d-composition.md)<span data-glossary-navigation="true"> ([Conv1d composition glossary](../../../glossary.md#conv1d-composition))</span> | Complete | 0020; 0017F1; current Compiler 0005D [convolution](../../../glossary.md#term-channels-first-convolution-layers) [gradients](../../../glossary.md#term-gradient) | Added [NCW](../../../glossary.md#term-ncw) Conv1d as visible [Conv2d](../../../glossary.md#convolution--conv2d) composition. |
| 0025H | [NCDHW Conv3d semantics and Tensor expressions](tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md) | Complete | 0020; 0018K–0018N; 0018V; 0019; 0025G | Added first-class grouped [NCDHW](../../../glossary.md#conv3d--ncdhw) [Conv3d](../../../glossary.md#conv3d--ncdhw). |
| 0025I | [NCW max/average Pool1d composition](tasks/0025i-ncw-max-average-pool1d-composition.md)<span data-glossary-navigation="true"> ([Pool1d composition glossary](../../../glossary.md#pool1d-composition))</span> | Complete | 0020A–0020A1; 0017F1; Compiler 0005D [pooling](../../../glossary.md#term-pooling) gradients | Added NCW Pool1d as visible Pool2d composition. |
| 0025J | [First-class NCDHW max/average Pool3d semantics](tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md) | Complete | 0025I; 0020A–0020A1; 0018K–0018N; 0018V | Added first-class NCDHW max/average [Pool3d](../../../glossary.md#pool3d--ncdhw-pooling). |
| 0025K | [Public NCDHW unfold3d and fold3d window transforms](tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md) | Complete | 0025J; 0023D | Added public NCDHW `unfold3d` and `fold3d`. |
| 0025L | [Cross-type CAST conversion semantics](tasks/0025l-cross-type-cast-conversion-semantics.md) | Complete | 0025K; 0001; 0003A; 0015G–0015H; 0018N; 0018U; owner-approved conversion policy | Fixed all 36 current-type CAST conversion meanings. |
| 0025M | [Tensor guide and API status reconciliation](tasks/0025m-tensor-guide-and-api-status-reconciliation.md) | Complete | Current Tensor/[host](../../../glossary.md#term-host)-storage source and [Engine](../../../glossary.md#term-engine)/public [API](../../../glossary.md#term-application-programming-interface-api); user-authorized drift remediation | Reconciled the Tensor guide and Tensor API opening with current Model and bounded CPU Engine behavior. |
| 0025N | [Recurrent Engine status reconciliation](tasks/0025n-recurrent-engine-status-reconciliation.md) | Complete | 0025E–0025F; Compiler 0006A; current Engine [lifecycle](../../../glossary.md#lifecycle) | Corrected the stale Engine status wording while preserving future recurrent execution and exception translation. |
| 0026 | [IEEE](../../../glossary.md#term-ieee) FLOAT16 and mixed-[precision](../../../glossary.md#term-precision) semantic contracts | Superseded | Replaced by delivered low-type semantics consolidated in Model 0032 | FLOAT16 is implemented. Future mixed-type capability requires a fresh [occurrence](../../../glossary.md#term-occurrence)-scoped brief, not revival of this obsolete add-type premise. |
| 0027 | [Explicit numerical-profile semantic authority](tasks/0027-explicit-numerical-profile-semantic-authority.md) | Complete | Completed [operation](../../../glossary.md#operation)-family semantics through 0025L; [Metal](../../../glossary.md#term-metal) 0014 and retained numerical blocker evidence; accepted Variant B | Established root/foundational authority, ADR 0019, bounded Model-owned result sets, affected Javadocs, and gross-error exclusions; implementation `ff86a302`, independent validation, and Class C review passed with zero failures, skips, or findings. |
| 0028 | [ACCELERATOR reduction exact-zero sign freedom](tasks/0028-accelerator-reduction-exact-zero-sign-freedom.md) | Complete | 0027; corrected Metal 0017 blocker evidence | Established final-result-only exact-zero sign freedom for arithmetic FLOAT32 SUM/[MEAN](../../../glossary.md#term-mean)/SUM_TO_SHAPE while preserving strict, terms, [identities](../../../glossary.md#term-identity-meanings), copies, and count division; implementation `fc003ab8`, proof and 23-task validation passed, and Class C review approved with zero findings. |
| 0029 | [ACCELERATOR MATMUL exact-zero sign freedom](tasks/0029-accelerator-matmul-exact-zero-sign-freedom.md) | Complete | 0027; 0028; retained Metal 0018 blocker evidence | Split the [contraction](../../../glossary.md#contraction-dimension) row and established only final-publication either-zero-sign freedom for a complete nonempty exact-zero FLOAT32 [ACCELERATOR](../../../glossary.md#term-accelerator) [MATMUL](../../../glossary.md#matrix-multiplication--matmul) result; implementation `30826783`, proof and 23-task validation passed, and Class C review approved with zero findings. |
| 0030 | [Total recursive ACCELERATOR numerical floor](tasks/0030-total-recursive-accelerator-numerical-floor.md) | Complete | 0031; 0027–0029; completed profile spine and [backend](../../../glossary.md#backend) realizations; retained Metal blocker evidence through 0040; approved minimal recursive redesign | Completed the documentation-only total recursive contract at remediation `2d95ab71` plus strict-baseline remediation `97cb9d11`; preserved the two-value API and all capability/[identity](../../../glossary.md#term-identity-meanings) behavior; independent Class C rereview after `97cb9d11` returned `APPROVE` with zero findings. |
| 0031 | [STRICT unary numerical baseline](tasks/0031-strict-unary-numerical-baseline.md) | Complete | 0018T1; 0019A; 0027–0029; completed CPU 0005H/0005I evidence | Owns exact-reference primitive bounds, special/domain rules, and recursive native/one-wider results for all nineteen kinds and every accepted BFLOAT16/FLOAT32/FLOAT64 type; remediation `97cb9d11` passed independent Class C rereview with zero findings and changed no executable behavior or capability. |
| 0032A | [Stored-subnormal provider regressions](tasks/0032a-stored-subnormal-provider-regressions.md) | Complete | 0030/0031, CPU 0017, Metal 0063 Complete; verified native package at execution | CPU/Metal backend-local stored-[subnormal](../../../glossary.md#term-subnormal) regressions passed; no stronger Model contract or behavior change. |
| 0032 | [Profile-free numerical-semantics reset](tasks/0032-profile-free-numerical-semantics-reset.md) | Complete | 0032A Complete; user-approved arithmetic-site DAZ/FTZ rule | Profile-free cutover, source-matched native rebuild, integrated validation, independent review, and documentation completed in main; historical ACCELERATOR true/false provider answers retained. |
| 0033 | [Stable softmax numerical contract](tasks/0033-stable-softmax-overflow-semantics.md) | Complete | Owner decision; 0032; review G1 | Stable finite meaning, CPU execution, conformance, bounded v2 generated/direct evidence, ADR 0029 and independent review complete. |
| 0034 | [General differentiable scan architecture decision/cutover](tasks/0034-general-differentiable-scan-decision.md)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../glossary.md#general-differentiable-scan))</span> | Complete | All 16 declared baseline predecessors Complete; actual concrete user approval and frontier verification recorded in brief | Authority-only cutover accepted by main 2026-10-21 after final independent Class C APPROVE, zero blockers; no Java. |
| 0035 | [General scan source and compiled body](tasks/0035-general-scan-source-and-compiled-body.md)<span data-glossary-navigation="true"> ([General scan glossary](../../../glossary.md#general-differentiable-scan))</span> | Complete | 0034 Complete | Bounded source/[IR](../../../glossary.md#term-intermediate-representation-ir) foundation accepted by main 2026-10-22 after independent authority/code/docs approval and compatibility checkpoint; no execution/[AD](../../../glossary.md#autograd). |
| 0036 | [Length-aware general scan design](tasks/0036-length-aware-general-scan-design.md) | Complete | Accepted general-scan/AD/cold/CPU/owned baseline; [planning](../../../glossary.md#term-planning)-only user approval | Proposal delivery only; independent final APPROVE and MAIN acceptance, no semantic/API/wire approval or implementation. |
| 0037 | [Length-aware Scan authority](tasks/0037-length-aware-general-scan-authority.md) | Complete | 0036 and accepted downstream baseline Complete; subsequent user design approval | Authority only; Hume final overall Class C APPROVE, zero open P1/P2; MAIN accepted 2026-10-24. |
| 0038 | [Length-aware Scan source/IR foundation](tasks/0038-length-aware-scan-source-foundation.md) | Complete | 0037 independent approval/MAIN acceptance Complete | MAIN accepted 2026-10-24 after independent APPROVE, zero open P1/P2, and combined [compile](../../../glossary.md#compile)/architecture/root retry PASS following one test-only ledger serializer repair; source/IR and guards only. |
| 0039 | [General Scan symbolic-batch design](tasks/0039-general-scan-symbolic-batch-design.md) | Complete | Complete Model0035/0038, Compiler0013, Prepare0013, CPU0037; user planning request | DESIGN DELIVERY ONLY; independent frozen APPROVE, zero open P1/P2, MAIN accepted 2026-10-24; no authority or implementation grant. |
| 0040 | [Same-geometry reuse design amendment](tasks/0040-same-geometry-reuse-design-amendment.md) | Complete | Complete Model0039 and Prepare0013; user planning-amendment request | DESIGN ONLY; independent APPROVE, zero open P1/P2, MAIN accepted 2026-10-25. Single-entry [immutable](../../../glossary.md#term-immutable) [metadata](../../../glossary.md#term-metadata) reuse retains per-call [admission](../../../glossary.md#term-admission); no authority or implementation grant. |
| 0041 | [Symbolic Scan and geometry-reuse authority](tasks/0041-symbolic-scan-authority-cutover.md)<span data-glossary-navigation="true"> ([Symbolic Scan glossary](../../../glossary.md#symbolic-scan))</span> | Complete | Complete Model0039/0040 and retained Model0038/Compiler0013/Prepare0013/CPU0037; user approval | AUTHORITY ONLY; independent APPROVE, zero open P1/P2; MAIN accepted 2026-10-25. Accepted future contracts/docs enacted, no implementation successor. |
| 0042 | [Static and symbolic Scan source foundation](tasks/0042-symbolic-scan-source-foundation.md) | Complete | Complete Model0041; retained Model0038/Compiler0013 | Static/named-[symbol](../../../glossary.md#term-symbol) Model source and consumer rejection complete; independent APPROVE, zero open P1/P2, configured root test/Javadoc PASS; MAIN accepted 2026-10-25. No downstream symbolic enablement. |
| 0043 | [Symbolic-batch/runtime-length Scan integration design](tasks/0043-symbolic-length-scan-integration-design.md)<span data-glossary-navigation="true"> ([runtime glossary](../../../glossary.md#term-runtime))</span> | Complete | Model0042; Compiler0013/0014; Prepare0011/0013/0014; CPU0037/0041/0042 Complete | DESIGN DELIVERY ONLY; Hume the 4th final independent Class C APPROVE, 0 P1/P2/no unresolved findings; MAIN accepted 2026-10-26. No new semantics, authority or implementation grant. |
| 0044 | [Static/one-B length Scan source foundation](tasks/0044-symbolic-length-scan-source-foundation.md) | Complete | 0043/0042/0038; Compiler0013/0014; Prepare0011/0014; CPU0042 Complete; user semantic/source approval | Source and two deliberate reject fixes complete; independent source/docs APPROVE, 0 P1/P2. MAIN accepted 2026-10-26 after configured root test/Javadoc PASS; evidence/counts in task Result. No downstream enablement or successor. |
| 0045 | [Per-value typed length-aware Scan source](tasks/0045-per-value-typed-length-scan-source.md) | Complete | 0044, Compiler0015, Prepare0015 Complete; authority/[Ready](../../../glossary.md#term-ready)/GO accepted 2026-11-01 | MAIN accepted source only 2026-11-01 after independent APPROVE, retained source-matched coverage and three approved Class A test-only recoveries, each 1 PASS/0 skip. Original commands remain FAILED, not a clean root PASS; [Result](tasks/0045-per-value-typed-length-scan-source.md#result) owns evidence/limits. |
| 0046 | [Empty-carry Scan and explicit length-mode source](tasks/0046-empty-carry-scan-and-explicit-length-mode.md#result)<span data-glossary-navigation="true"> ([carry glossary](../../../glossary.md#term-carry))</span> | Complete | Model0045/practical RNN Complete; integrated Complete CPU0078; accepted ADR0059 and coordinated source authority | Exact reviewed35-file source cutover; affected3,998pass/60existing skips, public96source/reject cases and28old CPU [artifact](../../../glossary.md#term-artifact) checks pass; independent final core/wire/evidence/docs APPROVE0. No typed execution/wire [widening](../../../glossary.md#term-widening). |
| 0047 | [Ordinary checked descriptor seam](tasks/0047-ordinary-checked-descriptor-seam.md#result) | Complete | Existing ordinary Model semantics; qualified Model0046 | Exact reviewed 88-file integration; affected 3,705 passes/60 existing skips, actual 84-[fixture](../../../glossary.md#term-fixture)/137-occurrence [descriptor](../../../glossary.md#term-descriptor) checks and 28 retained CPU [kernels](../../../glossary.md#kernel) pass. Final independent primitive/structured/contract evidence/docs APPROVE0; no row/AD/execution widening. |
| 0048 | [Shared per-output ordinary Scan semantics](tasks/0048-shared-ordinary-scan-value-semantics.md#result) | Complete | Complete0046/0047/Compiler0017A/CPU0078; final independent source/evidence/docs APPROVE0 | Fourteen paths integrated; affected3,261pass/60existing skips, actual public metadata/history [preflight](../../../glossary.md#term-preflight),85oldkernel files and48oldv4/v5archive bytes pass. No typed AD/physical/wire widening. |

## [General scan](../../../glossary.md#general-differentiable-scan) decision [DAG](../../../glossary.md#term-directed-acyclic-graph-dag) and frontier

Model0047 is Complete after final independent primitive, structured and contract/evidence/
documentation APPROVE0. No [descriptor](../../../glossary.md#term-descriptor)-source acceptance frontier remains.
Model0046 is Complete for bounded source after final independent core/wire/evidence/docs
APPROVE0. Its original failed full gate remains FAILED, with the [Compiler](../../../glossary.md#term-compiler) repair passed separately.
Model0047's original affected command likewise remains FAILED, with the repaired [Model](../../../glossary.md#term-model) gate
passed separately. General typed [inference](../../../glossary.md#term-inference)/[AD](../../../glossary.md#autograd)/[CPU](../../../glossary.md#term-cpu) execution and wire domains remain bounded.
Model0048 source/forward-[coordinate](../../../glossary.md#axis-rank-scalar-vector-and-matrix) semantics is Complete after final independent source/
evidence/documentation acceptance and actual consumer/[artifact](../../../glossary.md#term-artifact) [qualification](../../../glossary.md#term-qualification).
Typed AD, physical execution and [persistence](../../../glossary.md#term-persistence) remain independently bounded.
Current Model source frontier is only the generated-history header/declared-[axis](../../../glossary.md#term-axis) portion of
[Compiler0017 Ready](../compiler/tasks/0017-per-value-typed-length-scan-inference-and-ad.md),
under accepted coordinated authority and exact dirty source. Its [owner](../../../glossary.md#term-owner) must preserve independent
old cold/[codec](../../../glossary.md#term-codec) intersections. No new mathematical [guard](../../../glossary.md#term-guard)/coordinate transfer implementation,
standalone Model successor, persistence or numerical-policy grant is selected.

[Model0045](tasks/0045-per-value-typed-length-scan-source.md#result) remains Complete for source only,
with its retained review/coverage and three separate Class A recoveries. Its original commands
remain FAILED, not a clean root PASS; no rerun is authorized merely to relabel that evidence.
The new production prerequisite is separately scoped; MAIN owns shared authority/integration.
Model0044, Compiler0015 and Prepare0015 remain Complete, with no new downstream [admission](../../../glossary.md#term-admission). Typed CPU [bindings](../../../glossary.md#term-binding)/[dtypes](../../../glossary.md#data-type--datatype),
[feature](../../../glossary.md#term-feature) [SLICE](../../../glossary.md#slice), [layout](../../../glossary.md#layout) reads, cold/v5/legacy compatibility and [NN](../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) cutover stay coarse follow-up,
not source-step dependencies. Existing [unrolled](../../../glossary.md#backpropagation-through-time-and-recurrent-state)/static and fixed APIs remain intact; unchanged
unrolled higher-order support is not a blocker. Older closed-frontier statements below describe
their accepted [checkpoints](../../../glossary.md#term-checkpoint).

At Model0044 acceptance, Model [planning](../../../glossary.md#term-planning)/source/authority/implementation frontiers were None.
[Model0044](tasks/0044-symbolic-length-scan-source-foundation.md) is Complete. MAIN accepted bounded
authority/[Ready](../../../glossary.md#term-ready) after independent pre-Java APPROVE on 2026-10-26.
A different clean worker implemented the source/rejection [scope](../../../glossary.md#term-scope); independent source/docs APPROVE
has no open P1/P2. MAIN's configured root test/[Javadoc](../../../glossary.md#term-javadoc) checkpoint passed; MAIN accepted the exact
source-only task on 2026-10-26. Recorded Model0044 admission: PlanningAuthorized=true;
AuthorityAuthorized=true only for bounded Model-source wording;
ImplementationAuthorized=true only for the source foundation and deliberate rejection guards/tests.
The user explicitly accepted static-or-one-named-B length source with fixed T/[features](../../../glossary.md#term-software-feature)/[F32](../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../glossary.md#floating-point-precision-and-rounding) body,
not Compiler inference/AD/binding or [Prepare](../../../glossary.md#prepare)/CPU execution. No new production type/package/[API](../../../glossary.md#term-application-programming-interface-api);
existing [Scan](../../../glossary.md#term-cumulative-scan) owns source [validation](../../../glossary.md#term-validation); existing ScanGradientRules and [PrepareContext](../../../glossary.md#prepare-context-preparecontext) own early reject fixes.
Writes were serial, Parallel group None; MAIN completed shared integration and the configured root checkpoint.
The brief contains all seven integration fields and accepted dirty-base/[dependency](../../../glossary.md#term-dependency) evidence.
Other frontiers remain None; no standalone authority task or downstream successor is admitted.

[Model0043](tasks/0043-symbolic-length-scan-integration-design.md) is Complete for DESIGN DELIVERY ONLY;
Hume the 4th final independent Class C APPROVE, 0 P1/P2/no unresolved findings; MAIN accepted 2026-10-26.
The user's “ok, pust se do toho” selected only concise integration planning; MAIN verified Ready
admission/dependencies/base and confirmed launch. The [bounded proposal](../../length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration)
identifies the limited static-wording delta and real F64/control prerequisites, followed by coarse
source → Compiler AD/binding → Prepare/CPU/[Engine](../../../glossary.md#term-engine) capabilities. Its delivery created no successor;
the subsequent explicit source approval is now scoped only by Model0044 above.
Historical PlanningAuthorized=true; AuthorityAuthorized=false; ImplementationAuthorized=false.
MAIN's serial review/integration is closed; Parallel group None, all other gates remain held.

The user's “ok, pust se do prace” admitted [Model0042](tasks/0042-symbolic-scan-source-foundation.md), now Complete,
on 2026-10-25, including the explicit request to preserve static [dimensions](../../../glossary.md#dimension). ImplementationAuthorized=true
applies only to that compact source/test/documentation packet. MAIN verified authority/dependencies,
baseline and inactive conflicts. Independent APPROVE, zero open P1/P2, and MAIN's configured root
test/Javadoc gate passed; MAIN accepted 2026-10-25. Compiler0014 is Complete: scoped inference/
actual AD/binding, [module](../../../glossary.md#term-module)/Javadoc PASS (384 tests), independent APPROVE (0 P1/P2), and MAIN's
compatibility checkpoint PASS (68 tests); MAIN accepted 2026-10-25. Compiler frontier None;
Prepare/CPU/Engine frontiers remain None.
Authority frontier: None. The user's “ok, pokracuj tedy” authorized the now Complete
[Model0041](tasks/0041-symbolic-scan-authority-cutover.md) contract-first cutover after confirming
the original symbolic-B plan plus same-geometry reuse. AuthorityAuthorized=true;
ImplementationAuthorized=false. MAIN verified dependencies/base/conflicts on 2026-10-25.
[Model0040](tasks/0040-same-geometry-reuse-design-amendment.md) is Complete for DESIGN AMENDMENT ONLY,
authorized by the user's “ok, dopln to” on 2026-10-25; PlanningAuthorized=true,
ImplementationAuthorized=false. Independent frozen APPROVE, zero open P1/P2; MAIN accepted 2026-10-25.
Model0041 enacts the explicit [immutable](../../../glossary.md#term-immutable)-facts/[mutable](../../../glossary.md#term-mutable)-[run](../../../glossary.md#term-run)-array and prepared-meaning exception;
independent APPROVE, zero open P1/P2; MAIN accepted 2026-10-25. That authority task authorized no Java;
the separately admitted and completed Model0042 source foundation above did.
[Model0039](tasks/0039-general-scan-symbolic-batch-design.md) is Complete for DESIGN DELIVERY ONLY,
independent frozen APPROVE (zero open P1/P2), MAIN accepted 2026-10-24. PlanningAuthorized=true
records the completed admission from the user's “ok, pokracuj” planning request;
ImplementationAuthorized=false. The [focused proposal](general-scan-symbolic-batch-proposal.md)
separates Model [PURE](../../../glossary.md#term-pure)/dtype source validity, Compiler's bounded AD domain and initial F32 CPU
qualification; zero new/moved production types proposed, existing Scan/numeric owners extended.
Base HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted uncommitted
baseline; index unchanged, no competing writer, Parallel group None. MAIN owns integration;
the brief records all seven [metadata](../../../glossary.md#term-metadata) fields. Design review/MAIN acceptance → explicit user
semantic approval → separately admitted coordinated authority cutover → cohesive source/AD/
CPU vertical gates. User detailed semantic approval and authority admission are now supplied;
the sole [Scan semantic owner](../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan)
and [memo lifecycle owner](../../../architecture/contracts/runtime-prepare-engine.md#accepted-engine-geometry-memo)<span data-glossary-navigation="true"> ([memo glossary](../../../glossary.md#cache-and-memoization); [lifecycle glossary](../../../glossary.md#lifecycle))</span>
contain the accepted direction; only Model source is implemented and accepted through Model0042.
No new production type/[signature](../../../glossary.md#term-signature) or [atomic](../../../glossary.md#resource-transaction-and-rollback)
mechanism is selected. Independent authority review/MAIN acceptance are complete; separately Ready
implementation briefs remain required for downstream capabilities. The completed authority delivery's
ImplementationAuthorized=false is unchanged; Model0042's source authorization is completed, with no active successor.

At the completed length-authority checkpoint, authority frontier was None;
[Model0037](tasks/0037-length-aware-general-scan-authority.md) is Complete
after Hume's final overall independent Class C APPROVE, zero open P1/P2, and MAIN acceptance
2026-10-24. User “ok, pust se do prace” selects the reviewed design and first foundation.
Model source frontier: None; [Model0038](tasks/0038-length-aware-scan-source-foundation.md) Complete,
MAIN-accepted 2026-10-24 after independent review and successful combined integration retry.
Compiler0013 below is Complete, MAIN accepted 2026-10-24. Model0036 remains historical proposal-only Complete.
The recurrent-scan owner defines source/[IR](../../../glossary.md#term-intermediate-representation-ir) and the accepted downstream semantics. Compiler0013
implements bounded inference/first-order reverse construction/query/demand; cold, Prepare, CPU,
v5 and owned NN length capability remain unsupported.
Immediate DAG: Model0036 Complete + user approval → Model0037 review/MAIN acceptance → Model0038
Model source/IR with [fail-closed](../../../glossary.md#term-fail-closed) consumers → Compiler0013 implementation admission below. Beyond that immediate
packet, Prepare/CPU/cold-v5/owned qualification remain coarse dependencies, not detailed tasks.
The immediate Compiler-owned successor is now [Compiler0013](../compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md),
joining Complete Compiler0011 and Planning0008, is Complete; MAIN accepted 2026-10-24 after
independent code/docs APPROVE (0 P1/P2), worker module, compatibility and final docs/Javadoc PASS;
its [Result](../compiler/tasks/0013-length-aware-scan-capture-and-first-order-ad.md#result) owns evidence.
ImplementationAuthorized=true is historical admission; no new production type/Planning layer.
Ready frontier None; this is not a new Model task or successor authorization.
Cold [capture](../../../glossary.md#term-capture-meanings)/reconstruction, Prepare, CPU and codec remain individually closed owner gates.
NN0032 migration and its higher-order compatibility decision remain separately held.
MAIN owns shared integration; Parallel group None. Both new briefs record all seven metadata fields,
dependency/frontier checks and the uncommitted accepted common base. Conflicts: scan semantics/API,
role/count/AD/cold/[runtime](../../../glossary.md#term-runtime)/wire and shared documents, serialized; no competing writer active.
Completed integration order: authority acceptance → source/review → MAIN combined named [compile](../../../glossary.md#compile)/
architecture and existing-native root test/Javadoc. First attempt failed only test-ledger metadata;
bounded test-only repair and same-command retry passed, both performance flags false, no rebuild/
campaign, reusing successful/cached evidence. No automatic next task follows.
No production type/package additions: existing model.[operation](../../../glossary.md#operation).scan
ScanAttrs, model.[tensor](../../../glossary.md#tensor) Scan/ScanBody and model.[graph](../../../glossary.md#graph) CompiledNode/CompiledScanBody only; exact
bounded consumer rejection paths are in Model0038. Other owners' support remains held.

[Model0035](tasks/0035-general-scan-source-and-compiled-body.md)
is Complete, accepted by main 2026-10-22 after pre-code authority approval, final independent code/docs
APPROVE and the named compatibility checkpoint; its brief owns evidence and seven integration fields.
Model0034 remains Complete; its [decision brief](tasks/0034-general-differentiable-scan-decision.md)
records actual user approval and verified dependencies/base/inactive conflicts on 2026-10-21.
The [program](../../general-differentiable-scan.md) coordinates the serial DAG:
completed authority cutover → completed Model source/IR → Planning body query → Compiler body AD/capture
→ Prepare projection → bounded CPU F32 forward/automatic-reverse milestone → remaining declared
forms/closure + F64 → main low-level join. Planning0008 body query is Complete, accepted by main
2026-10-22 after final independent APPROVE and compatibility gates; Planning frontier None.
[Compiler0011](../compiler/tasks/0011-general-scan-capture-and-first-order-ad.md) is Complete, accepted by main
2026-10-22 after independent code/docs APPROVE and compatibility checkpoint; then Compiler frontier None.
[Prepare0010](../prepare/tasks/0010-general-scan-projection-and-resource-handoff.md) is Complete:
scoped projection/demand validation and resource [handoff](../../../glossary.md#term-handoff), main accepted 2026-10-22 after Huygens
final Class C code/docs APPROVE and the exact compatibility checkpoint.
At that earlier all-valid checkpoint CPU/Prepare/Compiler/Planning/Model/NN frontiers were None.
Current length-aware Model0037/0038 status is above. Bounded CPU F32 [pointwise](../../../glossary.md#term-pointwise) forward + actual
automatic reverse [CPU0030](../../backends/cpu/tasks/0030-general-scan-f32-pointwise-vertical.md)
is Complete, accepted by main 2026-10-22 after independent review and focused checkpoint PASS.
Remaining declared algebra/closure + F64 is a mandatory later Draft gate; no successor launch.
[Padding](../../../glossary.md#padding)/NN training and persistence retain separate gates; [BF16](../../../glossary.md#floating-point-precision-and-rounding)/[FP16](../../../glossary.md#term-fp16) qualification follows existing
semantics. Compiler0010 is Superseded; existing fixed/static [kernels](../../../glossary.md#kernel) and APIs remain.
Common base: main=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted
uncommitted authority/Model0034/0035, Planning0008, Compiler0011, Prepare0010 and CPU0030 work; Parallel group None.
Scan API/IR/[residual](../../../glossary.md#scan-residual-port-and-history-demand)/query/AD/projection/NN/authority
conflicts remain serialized. Main owns shared integration/docs and the later program root gate.
Implemented inventory: existing `model.tensor` Scan/ScanBody/ScanResult,
`model.operation.scan` ScanKind/ScanAttrs, `model.graph` CompiledScanBody.
This source/IR foundation grants no capture, AD, execution, training or persistence capability.
The program is non-authoritative coordination, not the semantic owner.

## Milestones and current frontier

- 0001–0013A established values, packages, immutable graph records, host storage, Tensor identity,
  factories, eager data creation, and provenance.<span data-glossary-navigation="true"> ([immutable glossary](../../../glossary.md#term-immutable); [graph glossary](../../../glossary.md#graph); [host storage glossary](../../../glossary.md#host-storage); [Tensor glossary](../../../glossary.md#tensor); [identity glossary](../../../glossary.md#term-identity-meanings); [eager glossary](../../../glossary.md#term-eager-evaluation); [provenance glossary](../../../glossary.md#provenance))</span>
- 0014A–0018V established operation families, then hardened signatures, provenance, symbolic
  Shapes, typed scalars, taxonomy, numeric coverage, and public boundaries.<span data-glossary-navigation="true"> ([operation glossary](../../../glossary.md#operation); [signatures glossary](../../../glossary.md#term-signature); [Shapes glossary](../../../glossary.md#shape); [scalars glossary](../../../glossary.md#term-scalar))</span>
- 0019–0022B added selected linear algebra, activations, explicit RNG/dropout, ordering, attention,
  channels-first Conv2d/Pool2d, normalization, and loss families.<span data-glossary-navigation="true"> ([linear glossary](../../../glossary.md#term-linear-meanings); [activations glossary](../../../glossary.md#term-activation); [RNG glossary](../../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../../glossary.md#dropout--inverted-dropout); [Conv2d glossary](../../../glossary.md#convolution--conv2d); [normalization glossary](../../../glossary.md#term-normalization); [loss glossary](../../../glossary.md#loss))</span>
- 0023–0024A closed the [adjoint](../../../glossary.md#term-adjoint)-expressibility prerequisites and selected [Model](../../../glossary.md#term-model) capability audit.
- 0025–0025D supplied focused Compiler prerequisites. 0025E–0025K added fixed recurrent and
  rank-specific spatial coverage; 0025L closed CAST semantics for the six current types.<span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler); [rank glossary](../../../glossary.md#term-rank); [spatial glossary](../../../glossary.md#term-spatial-axes))</span>
- Selected Model implementation [scope](../../../glossary.md#term-scope) is `Complete` through 0025L. Documentation-only tasks 0025M
  and 0025N are `Complete`.
- Task 0027 is `Complete` at implementation [commit](../../../glossary.md#term-commit) `ff86a302` after independent validation passed
  1,114 Model tests and nine architecture tests with zero failures or skips and Class C review
  returned <span data-glossary-navigation="true"> ([validation glossary](../../../glossary.md#term-validation))</span>`APPROVE` with zero findings. Config 0006, Engine 0018, CPU 0017, Metal 0015, exact
  both-profile Metal 0019, and Metal 0020 are also Complete.<span data-glossary-navigation="true"> ([Engine glossary](../../../glossary.md#term-engine); [CPU glossary](../../../glossary.md#term-cpu); [Metal glossary](../../../glossary.md#term-metal))</span>
- Task 0028 is `Complete` at implementation `fc003ab8` after its bounded reduction proof,
  23-task validation, and independent Class C approval passed. Metal 0017 remains Blocked under its
  historical reduction contract.
- Task 0029 is `Complete` at implementation `30826783`; its bounded proof, 23-task validation, and
  independent Class C review passed.
- Task 0030 is `Complete` after the original documentation-only recursive-contract cutover,
  remediation `2d95ab71683698753c9ac64d9373fde8301c9404`, and completed strict-baseline remediation
  `97cb9d116bd85ee6a0dfbf2e9b70d32604633c85`. It depends on the backend-independent all-type
  strict unary baseline, defines the inclusive accelerator ordered-binary32 distance-at-most-five
  rule, enumerates every composite primitive site, and reconciles current Metal reachability.
  Focused validation passed; independent Class C rereview after <span data-glossary-navigation="true"> ([backend glossary](../../../glossary.md#backend); [unary glossary](../../../glossary.md#term-unary); [accelerator glossary](../../../glossary.md#term-accelerator); [ordered-binary32 distance glossary](../../../glossary.md#ulp-and-numerical-tolerance); [primitive site glossary](../../../glossary.md#rounding-sites-fma-and-aggregate-floor))</span>`97cb9d11` returned `APPROVE` with
  zero findings.
- Task 0031 is `Complete`. Its documentation/Javadoc-only decision owns exact mathematical
  references and per-type bounds for all nineteen unary kinds, including BFLOAT16 without implying
  capability. Its initial baseline is <span data-glossary-navigation="true"> ([Javadoc glossary](../../../glossary.md#term-javadoc); [BFLOAT16 glossary](../../../glossary.md#term-bfloat16))</span>`a08db430f2d01f8250b74b65bea488bbc06b5860`; remediation
  `97cb9d116bd85ee6a0dfbf2e9b70d32604633c85` passed independent Class C rereview with zero
  findings. No executable statement or capability changed.
- Task 0032A is `Complete`: backend-local regressions record the current CPU/Metal stored-value
  subnormal realization, with focused CPU and verified-package Metal tests passed. This does not
  change Model semantics or provider behavior.<span data-glossary-navigation="true"> ([subnormal glossary](../../../glossary.md#term-subnormal))</span>
- Task 0032 is `Complete`: profile-free authority and implementation are in main, with the
  isolated full build (3,853 total, zero failures, 39 explained skips), architecture tests
  (9 total, zero failures/skips), source-matched main Metal package, independent Class C review,
  and final documentation reconciliation recorded in the brief. The P1 OpenBLAS ABI smoke and
  P2 documentation findings were corrected. ADR 0026 is Accepted; publication is separate from
  this validation result.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../glossary.md#term-openblas); [ABI glossary](../../../glossary.md#term-application-binary-interface-abi); [smoke glossary](../../../glossary.md#term-smoke-test); [ADR glossary](../../../glossary.md#term-architecture-decision-record-adr); [publication glossary](../../../glossary.md#term-publication))</span>
- [`FLOAT16`](../../../glossary.md#term-float16) is active through the later low-precision cutover; review 2026-10-13 closes the stale
  0026 planning row as Superseded, not as a newly implemented capability.<span data-glossary-navigation="true"> ([precision glossary](../../../glossary.md#term-precision); [planning glossary](../../../glossary.md#term-planning))</span>

## Live gates, decisions, and risks

- **Stable [softmax](../../../glossary.md#softmax--log-softmax) decision:** [0033](tasks/0033-stable-softmax-overflow-semantics.md) is Complete.
  The owner selected stable finite output after review G1; the coordinated contract, CPU
  conformance, bounded v2 performance diagnostic and independent re-review passed. G1 is closed.<span data-glossary-navigation="true"> ([owner glossary](../../../glossary.md#term-owner); [finite glossary](../../../glossary.md#term-finite); [CPU glossary](../../../glossary.md#term-cpu); [conformance glossary](../../../glossary.md#term-conformance))</span>

- **Numerical semantics:** Complete 0027–0031 and ADRs 0019/0021/0023 describe historical
  profile-era baselines, not the current rule. Complete 0032 and Accepted ADR 0026 record the
  main-worktree profile-free cutover; the root and incorporated Model contract are authoritative.
  Metal 0051 remains historically Blocked under its original premise. The MPSGraph scalar <span data-glossary-navigation="true"> ([ADR glossary](../../../glossary.md#term-architecture-decision-record-adr); [Model glossary](../../../glossary.md#term-model); [Metal glossary](../../../glossary.md#term-metal); [MPSGraph glossary](../../../glossary.md#term-mpsgraph); [scalar glossary](../../../glossary.md#term-scalar))</span>`+0`
  guard's performance effect is unmeasured, a nonblocking follow-up rather than a live cutover
  gate. Model 0033 was launched from Ready after its separate owner decision and dependency check
  and is now Complete. ADR 0029 records its distinct first-class softmax semantics.<span data-glossary-navigation="true"> ([guard glossary](../../../glossary.md#term-guard); [Ready glossary](../../../glossary.md#term-ready); [dependency glossary](../../../glossary.md#term-dependency))</span>
- **[FLOAT16](../../../glossary.md#term-float16) and [mixed precision](../../../glossary.md#term-mixed-precision):** Both low types are current and distinct. A shared two-byte
  carrier does not imply identical arithmetic or a mixed-low backend route; explicit casts form
  the current <span data-glossary-navigation="true"> ([carrier glossary](../../../glossary.md#term-storage-carrier); [backend route glossary](../../../glossary.md#backend-route))</span>[`FLOAT32`](../../../glossary.md#term-float32) boundary. Task 0026 is Superseded and is not a remaining prerequisite.
- **Recurrent execution:** Model 0025E–0025F and Compiler 0006A are Complete. Compiler captures
  each scan as one identity-distinct flat forward node, but recurrent differentiation remains
  fail-closed. CPU0028/CPU0029 and <span data-glossary-navigation="true"> ([captures glossary](../../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([Compiler glossary](../../../glossary.md#term-compiler); [scan glossary](../../../glossary.md#term-cumulative-scan); [identity glossary](../../../glossary.md#term-identity-meanings); [node glossary](../../../glossary.md#node); [fail-closed glossary](../../../glossary.md#term-fail-closed))</span>[NN0021B](../../extensions/nn/master-plan.md) are Complete:
  four scalar floating types and resolved nonnegative-stride input layouts are supported by CPU0029.
  Current fixed/static APIs remain unchanged. <span data-glossary-navigation="true"> ([stride glossary](../../../glossary.md#term-stride); [layouts glossary](../../../glossary.md#layout))</span>[0034](tasks/0034-general-differentiable-scan-decision.md)
  completed authority for generic all-valid body scan, not code, automatic padded rnnScan training or speed parity.
- **Other downstream [gradients](../../../glossary.md#term-gradient):** [Compiler 0006C](../compiler/master-plan.md) remains the Draft
  owner of Conv3d adjoint expressibility and gradient closure. It may request a separately
  selected minimal Model prerequisite, but this plan does not pre-authorize one.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../glossary.md#conv3d--ncdhw); [adjoint glossary](../../../glossary.md#term-adjoint))</span>
- **Capability separation:** a represented operation never implies Compiler gradient support,
  Planning ownership, backend execution, or performance. Exact capability filtering precedes
  route selection and tuning; current unsupported combinations fail closed.<span data-glossary-navigation="true"> ([operation glossary](../../../glossary.md#operation); [Planning glossary](../../../glossary.md#term-planning); [ownership glossary](../../../glossary.md#term-ownership-meanings); [backend glossary](../../../glossary.md#backend); [fail closed glossary](../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- **Primary risks:** treating Tensor/producers as graph IR; leaking runtime storage, backend
  capability, gradient lifecycle, or execution into Model; confusing logical layout with physical
  storage; deriving FLOAT16 policy from width; specializing recurrent topology to valid-length
  values; or widening public APIs and generic abstractions without a focused consumer.<span data-glossary-navigation="true"> ([producers glossary](../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../glossary.md#tensor); [graph glossary](../../../glossary.md#graph); [IR glossary](../../../glossary.md#term-intermediate-representation-ir); [runtime glossary](../../../glossary.md#term-runtime); [backend capability glossary](../../../glossary.md#backend-capability); [lifecycle glossary](../../../glossary.md#lifecycle); [topology glossary](../../../glossary.md#term-topology); [widening glossary](../../../glossary.md#term-widening))</span>

## History and update policy

Current source, contracts, this table, and each brief’s Status/Result override older prose.
Evidence, inventories, context IDs, and chronology stay in linked briefs or Git. The legacy
branch is read-only capability evidence.

Update this plan only when [Model](../../../glossary.md#term-model) [ownership](../../../glossary.md#term-ownership-meanings), package direction, ordered tasks, status, live gates,
risks, milestones, or frontier change. Do not retrospectively rewrite completed briefs. Create a
compact detailed brief only for the next selected frontier, after [dependency](../../../glossary.md#term-dependency) and roadmap
authorization; a `Draft` row alone is not implementation permission.
