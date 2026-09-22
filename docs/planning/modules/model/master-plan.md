# Model Master Plan

## Goal and authority

This is the current implementation map for Model identity, descriptors, provenance, value types,
operation semantics, host storage, and immutable graph records.

```text
Model meaning and metadata -> Compiler graph work -> backend capability and execution
```

[ARCHITECTURE.md](../../../../ARCHITECTURE.md) is authoritative. Applicable headings are
[Core invariants](../../../../ARCHITECTURE.md#core-invariants),
[Fixed recurrent scan without graph regions](../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions),
[Fixed family and current Model surface](../../../architecture/contracts/recurrent-scan.md#fixed-family-and-current-model-surface),
[Static Shape and runtime-value boundary](../../../architecture/contracts/recurrent-scan.md#static-shape-and-runtime-value-boundary),
[Purity and lifecycle ownership](../../../architecture/contracts/recurrent-scan.md#purity-and-lifecycle-ownership),
[`modules/model`](../../../architecture/contracts/foundational-modules.md#modulesmodel), and
[Dependency rules](../../../../ARCHITECTURE.md#dependency-rules). See the focused
[module](../../../architecture/module-boundaries.md), [dependency](../../../architecture/dependency-rules.md),
and [capability](capabilities.md) guides.

## Scope and non-goals

Model owns value types, logical layouts, identifiers, descriptors, public Tensor state, borrowed
host storage, operation meanings, expression provenance, and immutable graph records.

Model does not own compilation, differentiation traversal, backend capability, physical/device
storage, prepared/runtime state, route selection, execution, training, or persistence.

## Stable invariants and boundaries

- `Tensor` has immutable identity, descriptor, label, and expression provenance; it is not an
  intermediate-representation node. Its synchronized optional borrowed host-storage association
  is its only mutable field, and it owns no gradient, runtime, device, or graph-local state.
- A derived `TensorProducer` owns one immutable operation occurrence, ordered Tensor inputs,
  ordered output descriptors, and the canonical exact Tensor wrapper for every indexed output.
  Multi-output results share that producer; reconstruction of an equal output wrapper is invalid.
- `Shape` and `Dimension` values support rank-zero and zero extents, non-negative static
  `long` extents, and canonical symbolic expressions. Locally provable incompatibilities fail
  during construction; binding-dependent obligations remain explicit for Compiler validation.
- `TensorDescriptor` keeps data type, Shape, gradient eligibility, and resolved or unresolved
  logical layout together. Numeric layout geometry is resolved only when justified by known
  Shape/stride/offset facts; it is not a storage-alias, materialization, or execution promise.
- The current types are `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT64`, `INT32`, and `BOOL`.
  Exact typed scalar values preserve their represented bits. Axis-bearing APIs normalize and
  validate axes at their public boundary; numerical, accumulation, special-value, tie, and
  duplicate policies remain operation-family semantics.
- `Operation` owns backend-independent represented-value meaning and exact kind/attribute and
  cardinality signatures. It never owns backend support, lowering, algorithms, routes, or
  execution. Tensor expression construction validates descriptor-visible type, rank, axis, and
  Shape constraints without reading tensor values or executing an operation.
- Public API stays minimal and domain-owned. Helpers are package-private unless a focused public
  namespace is needed; generic registries, service locators, and catch-all packages are absent.
- Graph records and `CompiledGraphModel` are immutable. Publication binding is a separate
  `TensorId`-to-`ValueId` value; graph capture, inference, optimization, gradients, and
  publication planning remain Compiler-owned.
- The fixed recurrent family is exactly time-major RNN-tanh, reset-after GRU, and LSTM with one
  forward/reverse direction, fully static Shapes, runtime rank-one `INT64` valid lengths, dense
  outputs, explicit state/parameter inputs, and canonical multi-output provenance. Construction
  lives in stateless `model.tensor.RecurrentScan`, not on `Tensor`; there is no graph region,
  callback, hidden state, execution claim, or general scan body.
- Model exposes forward algebra needed by differentiation but owns no backward kinds or lifecycle.
  Compiler owns gradient policy and traversal. Recurrent backpropagation through time and Conv3d
  gradients remain fail-closed downstream until their separate owners complete them.

## Dependencies and package map

Model production uses only the JDK. It must not depend on Planning, Compiler, Runtime, Prepare,
Engine, concrete backends, or NN/Training. Packages remain acyclic: `datatype` and `shape` are
leaves; `layout` and `storage` use foundations; `operation` does not use Tensor/graph; `tensor`
and `graph` compose them.

| Package | Ownership and public boundary |
|---|---|
| `model.datatype` | Data-type metadata, promotion, bit conversion, and exact scalar values. |
| `model.shape` | Static/symbolic extents, immutable Shapes, axes, and local broadcasting. |
| `model.layout` | Resolved logical geometry and layout classification. |
| `model.storage` | Borrowed host-visible storage; no device residency or allocation lifecycle. |
| `model.tensor` | Tensor identity/state, descriptors, factories, eager host initialization, expression construction, producers/results, and focused `RecurrentScan`. |
| `model.operation` | Operation/attribute/signature foundations and family-owned semantics. |
| `model.operation.elementwise.*` | Binary, unary, classification, scalar, comparison, logical, selection, and cast families. |
| `model.operation.reduction`, `.scan`, `.normalization`, `.loss` | Reduction/statistics, cumulative scans, softmax/normalization, and loss meanings. |
| `model.operation.linalg`, `.attention`, `.convolution`, `.pooling` | Linear algebra, attention, and rank-specific channels-first spatial meanings. |
| `model.operation.random`, `.ordering`, `.layout`, `.index`, `.recurrent` | Explicit graph RNG, ordering, view/layout, indexing/scatter, and fixed recurrent meanings. |
| `model.graph` | Immutable graph values/nodes/model, phases, identifiers, and publication binding. |

`linear`, embedding, NCW Conv1d, and NCW Pool1d are visible compositions. NCHW/NCDHW convolution
and pooling are rank-specific first-class families; no arbitrary-rank `ConvNd`, `PoolNd`, or
`WindowNd` is public. Eager host randomness and graph RNG state stay separate; normalization and
loss own no mode, session, or hidden mutable statistics.

## Task list

The table is the ordered queue and status source. Evidence stays in linked briefs; 0026 has no
detailed brief.

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [DataType model](tasks/0001-data-type-model.md) | Complete | - | Delivered. |
| 0002 | [Shape and dimension model](tasks/0002-shape-and-dimension-model.md) | Complete | - | Delivered. |
| 0003 | [Layout descriptor model](tasks/0003-layout-descriptor-model.md) | Complete | 0002 | Delivered. |
| 0003A | [Data type package migration](tasks/0003a-data-type-package-migration.md) | Complete | 0001 | Delivered. |
| 0003B | [Shape package migration](tasks/0003b-shape-package-migration.md) | Complete | - | Delivered. |
| 0003C | [Layout package migration](tasks/0003c-layout-package-migration.md) | Complete | 0003B | Delivered. |
| 0004 | [Typed identifiers](tasks/0004-typed-identifiers.md) | Complete | 0003A–0003C | Delivered. |
| 0005 | [Operation semantic foundation](tasks/0005-operation-semantic-foundation.md) | Complete | - | Delivered. |
| 0006 | [Operation model](tasks/0006-operation-model.md) | Complete | 0005 | Delivered. |
| 0007 | [Tensor descriptor model](tasks/0007-tensor-descriptor-model.md) | Complete | 0001–0003, 0003A–0003C | Delivered. |
| 0008 | [Graph value and node model](tasks/0008-graph-value-and-node-model.md) | Complete | 0004, 0006, 0007 | Delivered. |
| 0009 | [Compiled graph model](tasks/0009-compiled-graph-model.md) | Complete | 0008 | Delivered. |
| 0010 | [Host storage abstraction](tasks/0010-host-storage-abstraction.md) | Complete | 0001, 0003A | Delivered. |
| 0011 | [Public Tensor skeleton](tasks/0011-public-tensor-skeleton.md) | Complete | 0004, 0007, 0010 | Delivered. |
| 0012 | [Tensor factory foundation](tasks/0012-tensor-factory.md) | Complete | 0010, 0011 | Delivered. |
| 0012A | [JVM-managed heap host storage allocation](tasks/0012a-host-storage-allocation.md) | Complete | 0010, 0012 | Delivered. |
| 0012B | [Flat typed tensor import](tasks/0012b-flat-typed-tensor-import.md) | Complete | 0012A | Delivered. |
| 0012C | [Nested typed tensor import](tasks/0012c-nested-typed-tensor-import.md) | Complete | 0012B | Delivered. |
| 0012D | [Constant tensor creation](tasks/0012d-constant-tensor-creation.md) | Complete | 0012B | Delivered. |
| 0012E | [Range and prefix population](tasks/0012e-range-and-prefix-population.md) | Complete | 0012B | Delivered. |
| 0012F | [Random tensor creation](tasks/0012f-random-tensor-creation.md) | Complete | 0012B | Delivered. |
| 0012G | [Uniform random tensor creation](tasks/0012g-uniform-random-tensor-creation.md) | Complete | 0012F | Delivered. |
| 0012H | [Integral random tensor creation](tasks/0012h-integral-random-tensor-creation.md) | Complete | 0012F | Delivered. |
| 0012I | [Bernoulli random tensor creation](tasks/0012i-bernoulli-random-tensor-creation.md) | Complete | 0012F | Delivered. |
| 0013 | [Tensor provenance skeleton](tasks/0013-tensor-provenance-skeleton.md) | Complete | 0006, 0011, 0012 | Delivered. |
| 0013A | [Full-value and identity-matrix tensor creation](tasks/0013a-full-value-and-identity-matrix-tensor-creation.md) | Complete | 0012B, 0012D | Delivered. |
| 0014A | [Binary arithmetic semantic kinds](tasks/0014a-binary-arithmetic-semantic-kinds.md) | Complete | 0005, 0006 | Delivered. |
| 0014B | [Binary arithmetic Tensor expressions](tasks/0014b-binary-arithmetic-tensor-expressions.md) | Complete | 0013, 0014A | Delivered. |
| 0014C | [Unary elementwise semantic kinds](tasks/0014c-unary-elementwise-semantic-kinds.md) | Complete | 0005, 0006 | Delivered. |
| 0014D | [Unary elementwise Tensor expressions](tasks/0014d-unary-elementwise-tensor-expressions.md) | Complete | 0013, 0014C | Delivered. |
| 0014E | [Scalar arithmetic and clamp semantics](tasks/0014e-scalar-arithmetic-and-clamp-semantics.md) | Complete | 0005, 0006 | Delivered. |
| 0014F | [Scalar arithmetic and clamp Tensor expressions](tasks/0014f-scalar-arithmetic-and-clamp-tensor-expressions.md) | Complete | 0013, 0014E | Delivered. |
| 0015A | [Binary comparison semantic kinds](tasks/0015a-binary-comparison-semantic-kinds.md) | Complete | 0005, 0006 | Delivered. |
| 0015B | [Binary comparison Tensor expressions](tasks/0015b-binary-comparison-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0015A | Delivered. |
| 0015C | [Boolean logical semantic kinds](tasks/0015c-boolean-logical-semantic-kinds.md) | Complete | 0005, 0006 | Delivered. |
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
| 0016F | [Masked reduction semantics and axis mapping](tasks/0016f-masked-reduction-semantics-and-axis-mapping.md) | Complete | 0005, 0006, 0016A | Delivered. |
| 0016F1 | [Masked sum and mean Tensor expressions](tasks/0016f1-masked-sum-and-mean-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016B, 0016F | Delivered. |
| 0016G | [Cumulative-sum semantic kind and attributes](tasks/0016g-cumulative-sum-semantic-kind-and-attributes.md) | Complete | 0005, 0006 | Delivered. |
| 0016H | [Cumulative-sum Tensor expressions](tasks/0016h-cumulative-sum-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016G | Delivered. |
| 0016I | [Softmax semantic kinds and attributes](tasks/0016i-softmax-semantic-kinds-and-attributes.md) | Complete | 0005, 0006 | Delivered. |
| 0016J | [Softmax Tensor expressions](tasks/0016j-softmax-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016I | Delivered. |
| 0017A | [Contiguous semantic kind](tasks/0017a-contiguous-semantic-kind.md) | Complete | 0005, 0006 | Delivered. |
| 0017B | [Contiguous Tensor expression](tasks/0017b-contiguous-tensor-expression.md) | Complete | 0002, 0003, 0007, 0011–0013, 0017A | Delivered. |
| 0017C | [Reshape and expand semantics](tasks/0017c-reshape-and-expand-semantics.md) | Complete | 0002, 0005, 0006 | Delivered. |
| 0017D | [Reshape Tensor expressions](tasks/0017d-reshape-tensor-expressions.md) | Complete | 0002, 0003, 0007, 0011–0013, 0017C | Delivered. |
| 0017D1 | [Expand Tensor expressions](tasks/0017d1-expand-tensor-expressions.md) | Complete | 0002, 0003, 0007, 0011–0013, 0017C | Delivered. |
| 0017E | [Axis-transform semantics](tasks/0017e-axis-transform-semantics.md) | Complete | 0002, 0005, 0006 | Delivered. |
| 0017F | [Permute and transpose Tensor expressions](tasks/0017f-permute-and-transpose-tensor-expressions.md) | Complete | 0002, 0003, 0013, 0017E | Delivered. |
| 0017F1 | [Expand-dimensions and squeeze Tensor expressions](tasks/0017f1-expand-dimensions-and-squeeze-tensor-expressions.md) | Complete | 0002, 0003, 0013, 0017E | Delivered. |
| 0017G | [Slice semantics](tasks/0017g-slice-semantics.md) | Complete | 0002, 0005, 0006 | Delivered. |
| 0017H | [Slice Tensor expressions](tasks/0017h-slice-tensor-expressions.md) | Complete | 0002, 0003, 0013, 0017G | Delivered. |
| 0017I | [Pad and tile semantics](tasks/0017i-pad-and-tile-semantics.md) | Complete | 0001, 0002, 0005, 0006 | Delivered. |
| 0017J | [Pad and tile Tensor expressions](tasks/0017j-pad-and-tile-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0017I | Delivered. |
| 0017K | [Tensor composition semantics](tasks/0017k-tensor-composition-semantics.md) | Complete | 0002, 0005, 0006 | Delivered. |
| 0017L | [Tensor composition expressions](tasks/0017l-tensor-composition-expressions.md) | Complete | 0001, 0002, 0013, 0017K | Delivered. |
| 0017M | [Unfold and fold semantics](tasks/0017m-unfold-and-fold-semantics.md) | Complete | 0002, 0005, 0006 | Delivered. |
| 0017N | [Unfold and fold Tensor expressions](tasks/0017n-unfold-and-fold-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0017M | Delivered; 0018R removed and 0023D restored public `foldAxis`. |
| 0018A | [Scalar select semantics](tasks/0018a-scalar-select-semantics.md) | Complete | 0002, 0005, 0006 | Delivered. |
| 0018B | [Scalar select Tensor expression](tasks/0018b-scalar-select-tensor-expression.md) | Complete | 0002, 0003, 0013, 0018A | Delivered. |
| 0018C | [Axis gather semantics](tasks/0018c-axis-gather-semantics.md) | Complete | 0005, 0006 | Delivered. |
| 0018D | [Axis gather Tensor expressions](tasks/0018d-axis-gather-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0018C | Delivered. |
| 0018D1 | [Primitive take convenience](tasks/0018d1-primitive-take-convenience.md) | Complete | 0012B, 0018D | Delivered. |
| 0018E | [Gather-ND semantics](tasks/0018e-gather-nd-semantics.md) | Complete | 0005, 0006 | Delivered. |
| 0018F | [Gather-ND Tensor expressions](tasks/0018f-gather-nd-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0018E | Delivered. |
| 0018G | [Axis scatter semantics](tasks/0018g-axis-scatter-semantics.md) | Complete | 0005, 0006, 0018C | Delivered. |
| 0018H | [Axis scatter Tensor expressions](tasks/0018h-axis-scatter-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0018G | Delivered. |
| 0018I | [Scatter-ND semantics](tasks/0018i-scatter-nd-semantics.md) | Complete | 0005, 0006, 0018E | Delivered. |
| 0018J | [Scatter-ND Tensor expression](tasks/0018j-scatter-nd-tensor-expression.md) | Complete | 0001, 0002, 0013, 0018I | Delivered. |
| 0018K | [Operation signature and construction hardening](tasks/0018k-operation-signature-and-construction-hardening.md) | Complete | 0005, 0006, 0008 | Delivered. |
| 0018L | [Shared multi-output Tensor provenance](tasks/0018l-shared-multi-output-tensor-provenance.md) | Complete | 0007–0009, 0011–0013, 0018K | Delivered. |
| 0018M | [Symbolic extent expressions](tasks/0018m-symbolic-extent-expressions.md) | Complete | 0002, 0017C–0017N | Delivered. |
| 0018M1 | [Dynamic extent adoption in pad, tile, and concat](tasks/0018m1-dynamic-extent-adoption.md) | Complete | 0017J, 0017L, 0018M | Delivered. |
| 0018N | [Typed scalar value contract](tasks/0018n-typed-scalar-value-contract.md) | Complete | 0001, 0014E, 0014F, 0017I, 0017J, 0018K | Delivered. |
| 0018O | [Indexing taxonomy and unstack normalization](tasks/0018o-indexing-taxonomy-and-unstack-normalization.md) | Complete | 0017K–0017L, 0018A–0018J, 0018K–0018L | Finalized indexing taxonomy and repeated-`SELECT` unstack. |
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
| 0019A | [Modern activation semantics and Tensor expressions](tasks/0019a-modern-activation-semantics-and-tensor-expressions.md) | Complete | 0014C–0014D, 0018K, 0018P | Delivered. |
| 0019A1 | [Embedding convenience](tasks/0019a1-embedding-convenience.md) | Complete | 0018K, 0018O | Delivered. |
| 0019A2 | [One-hot encoding](tasks/0019a2-one-hot-encoding.md) | Complete | 0001–0002, 0005–0007, 0011–0013, 0018K–0018O | Delivered. |
| 0019B | [Explicit graph RNG state foundation](tasks/0019b-explicit-graph-rng-state-foundation.md) | Complete | 0018K–0018L, 0018N, 0018S | Delivered. |
| 0019B1 | [Explicit graph dropout construction](tasks/0019b1-explicit-graph-dropout-construction.md) | Complete | 0019B, 0018K–0018L | Delivered. |
| 0019C | [Sort and argsort](tasks/0019c-sort-and-argsort.md) | Complete | 0018K–0018L, 0018U–0018U1 | Delivered. |
| 0019C1 | [Top-K values and indices](tasks/0019c1-top-k-values-and-indices.md) | Complete | 0019C, 0018L | Delivered. |
| 0019D | [Linear convenience](tasks/0019d-linear-convenience.md) | Complete | 0017F, 0014B, 0018K–0018N, 0018T, 0018U, 0019–0019C1 | Delivered. |
| 0019E | [Scaled dot-product attention](tasks/0019e-scaled-dot-product-attention.md) | Complete | 0016I–0016J, 0018K–0018L, 0018N, 0018Q, 0019 | Delivered. |
| 0020 | [NCHW Conv2d semantics and Tensor expressions](tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md) | Complete | 0018K–0018L, 0018M–0018M1, 0018N, 0018V, 0019 | Delivered. |
| 0020A | [NCHW Max Pool2d semantics and Tensor expression](tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md) | Complete | 0020, 0017M–0017N, 0018K–0018L, 0018M–0018M1, 0018N, 0018V | Delivered. |
| 0020A1 | [NCHW Average Pool2d semantics and Tensor expression](tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md) | Complete | 0020A | Delivered. |
| 0021 | [Layer normalization semantics and Tensor expressions](tasks/0021-layer-normalization-semantics-and-tensor-expressions.md) | Complete | 0018K, 0018L, 0018N, 0018V | Delivered. |
| 0021A | [RMS normalization semantics and Tensor expressions](tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md) | Complete | 0018K, 0018L, 0018N, 0018V, 0021 | Delivered. |
| 0021B | [Batch-normalization inference](tasks/0021b-batch-normalization-inference.md) | Complete | 0018K, 0018L, 0018N, 0018V, 0021–0021A | Delivered. |
| 0021C | [Batch-normalization training and statistic transition](tasks/0021c-batch-normalization-training-and-statistic-transition.md) | Complete | 0021B, 0018L, 0018N, 0018V | Delivered. |
| 0022 | [Mean-squared-error loss](tasks/0022-mean-squared-error-loss.md) | Complete | 0018K, 0018N, 0018V | Delivered. |
| 0022A | [Dense-target categorical cross-entropy with logits](tasks/0022a-dense-target-categorical-cross-entropy-with-logits.md) | Complete | 0022, 0016I–0016J | Delivered. |
| 0022B | [Index-target categorical cross-entropy with logits](tasks/0022b-index-target-categorical-cross-entropy-with-logits.md) | Complete | 0022A, 0018O | Delivered. |
| 0023 | [Adjoint expressibility audit](tasks/0023-adjoint-expressibility-audit.md) | Complete | 0006, 0014A–0014F, 0015A–0015H, 0016A–0016J, 0017A–0017N including 0017D1 and 0017F1, 0018A–0019E including 0018D1, 0019A1, and 0019A2, 0020–0022B including 0020A–0020A1 and 0021A–0021C; post-0022B checkpoint | Selected six generally useful public adjoint prerequisites. |
| 0023A | [Binding-aware sum-to-Shape](tasks/0023a-binding-aware-sum-to-shape.md) | Complete | 0023 | Delivered. |
| 0023B | [Gather-compatible scatter-add](tasks/0023b-gather-compatible-scatter-add.md) | Complete | 0023, 0023A | Delivered. |
| 0023C | [Slice update and target-relative crop](tasks/0023c-slice-update-and-target-relative-crop.md) | Complete | 0023, 0023B | Delivered. |
| 0023D | [Public foldAxis and dynamic window transforms](tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md) | Complete | 0023, 0023C | Delivered. |
| 0023E | [Cumulative scan normalization and product](tasks/0023e-cumulative-scan-normalization-and-product.md) | Complete | 0016G–0016H, 0018K, 0018U–0018U1, 0023, 0023D | Delivered. |
| 0023F | [Scaled dot-product attention weights output](tasks/0023f-scaled-dot-product-attention-weights-output.md) | Complete | 0018K–0018L, 0019E, 0023, 0023E | Delivered. |
| 0024 | [Model capability and contract closure audit](tasks/0024-model-capability-and-contract-closure-audit.md) | Complete | 0001–0023F | Found one Javadoc-only gap, later closed by 0024A. |
| 0024A | [GraphValue Tensor-status Javadoc correction](tasks/0024a-graph-value-tensor-status-javadoc-correction.md) | Complete | 0024 | Closed the selected Model capability milestone. |
| 0025 | [Canonical TensorProducer outputs](tasks/0025-canonical-tensor-producer-outputs.md) | Complete | 0018L, 0019B1, 0021C, 0023F, 0024A; prerequisite for Compiler 0004 | Retained the canonical Tensor wrapper for every producer output. |
| 0025A | [Portable floating comparison, extrema, and clamp semantics](tasks/0025a-portable-floating-comparison-extrema-and-clamp-semantics.md) | Complete | 0018N, 0018T, 0018U, 0025; prerequisite for Compiler 0005A | Fixed portable floating comparison, extrema, and clamp meanings. |
| 0025B | [Binding-aware expansion](tasks/0025b-binding-aware-expansion.md) | Complete | 0002, 0017C, 0017D1, 0018M, 0023A, 0025A; prerequisite for Compiler 0005B | Retained binding-dependent EXPAND obligations. |
| 0025C | [Portable functional-scatter reduction semantics](tasks/0025c-portable-functional-scatter-reduction-semantics.md) | Complete | 0018G–0018J, 0018U–0018U1, 0025A; prerequisite for Compiler 0005C | Fixed portable functional-scatter reduction meanings. |
| 0025D | [Dynamic-extent slice extraction and symbolic slice placement](tasks/0025d-dynamic-extent-slice-extraction-and-symbolic-slice-placement.md) | Complete | 0002, 0018M, 0018R, 0023C, 0025C; prerequisite for Compiler 0005C | Added dynamic-length slice and symbolic placement forms. |
| 0025E | [Fixed recurrent-scan semantic family and Tensor expressions](tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md) | Complete | accepted NN 0021A architecture decision; 0018K–0018L, 0018N, 0019, 0025; current recurrent cell equations | Added fixed flat RNN/GRU/LSTM scan semantics and provenance. |
| 0025F | [Recurrent-scan expression namespace correction](tasks/0025f-recurrent-scan-expression-namespace-correction.md) | Complete | 0025E; before Compiler 0006A | Moved low-level scan construction to `RecurrentScan`. |
| 0025G | [NCW Conv1d composition](tasks/0025g-ncw-conv1d-composition.md) | Complete | 0020; 0017F1; current Compiler 0005D convolution gradients | Added NCW Conv1d as visible Conv2d composition. |
| 0025H | [NCDHW Conv3d semantics and Tensor expressions](tasks/0025h-ncdhw-conv3d-semantics-and-tensor-expressions.md) | Complete | 0020; 0018K–0018N; 0018V; 0019; 0025G | Added first-class grouped NCDHW Conv3d. |
| 0025I | [NCW max/average Pool1d composition](tasks/0025i-ncw-max-average-pool1d-composition.md) | Complete | 0020A–0020A1; 0017F1; Compiler 0005D pooling gradients | Added NCW Pool1d as visible Pool2d composition. |
| 0025J | [First-class NCDHW max/average Pool3d semantics](tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md) | Complete | 0025I; 0020A–0020A1; 0018K–0018N; 0018V | Added first-class NCDHW max/average Pool3d. |
| 0025K | [Public NCDHW unfold3d and fold3d window transforms](tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md) | Complete | 0025J; 0023D | Added public NCDHW `unfold3d` and `fold3d`. |
| 0025L | [Cross-type CAST conversion semantics](tasks/0025l-cross-type-cast-conversion-semantics.md) | Complete | 0025K; 0001; 0003A; 0015G–0015H; 0018N; 0018U; owner-approved conversion policy | Fixed all 36 current-type CAST conversion meanings. |
| 0025M | [Tensor guide and API status reconciliation](tasks/0025m-tensor-guide-and-api-status-reconciliation.md) | Complete | Current Tensor/host-storage source and Engine/public API; user-authorized drift remediation | Reconciled the Tensor guide and Tensor API opening with current Model and bounded CPU Engine behavior. |
| 0026 | IEEE FLOAT16 and mixed-precision semantic contracts | Draft | 0001, 0018N, completed operation-family semantics; required before any backend advertises FLOAT16 | Preserve BFLOAT16, add distinct true IEEE-754 binary16 `FLOAT16`, and audit affected families for explicit input, accumulation/intermediate, and output types without adding backend support. |

## Milestones and current frontier

- 0001–0013A established values, packages, immutable graph records, host storage, Tensor identity,
  factories, eager data creation, and provenance.
- 0014A–0018V established operation families, then hardened signatures, provenance, symbolic
  Shapes, typed scalars, taxonomy, numeric coverage, and public boundaries.
- 0019–0022B added selected linear algebra, activations, explicit RNG/dropout, ordering, attention,
  channels-first Conv2d/Pool2d, normalization, and loss families.
- 0023–0024A closed the adjoint-expressibility prerequisites and selected Model capability audit.
- 0025–0025D supplied focused Compiler prerequisites. 0025E–0025K added fixed recurrent and
  rank-specific spatial coverage; 0025L closed CAST semantics for the six current types.
- Selected Model implementation scope is `Complete` through 0025L. Documentation-only task 0025M
  is also `Complete`: its clean implementation and independent targeted review reconciled the
  confirmed Tensor guide/API status drift. It changes no implementation scope, and no later Model
  task is authorized.
- Task 0026 remains `Draft`, has no detailed brief, and is selected only when IEEE-754 binary16
  `FLOAT16` and mixed-precision semantics become current. No other Model work is authorized.

## Live gates, decisions, and risks

- **FLOAT16 and mixed precision:** BFLOAT16 remains a distinct current type. Only 0026 may add true
  IEEE binary16 FLOAT16 and must audit each affected family’s input, accumulation/intermediate,
  and output types. A shared two-byte carrier does not imply arithmetic, Java Vector support, or a
  backend route. CPU explicitly forbids FLOAT16 work before 0026, and no backend may advertise it.
- **Recurrent execution:** Model 0025E–0025F and Compiler 0006A are Complete. Compiler captures
  each scan as one identity-distinct flat forward node, but recurrent differentiation remains
  fail-closed. No concrete backend currently advertises recurrent-scan execution, so
  [NN 0021B](../../extensions/nn/master-plan.md) remains Draft pending an honest route and
  lifecycle validation; current static NN sequence APIs remain unchanged.
- **Other downstream gradients:** [Compiler 0006C](../compiler/master-plan.md) remains the Draft
  owner of Conv3d adjoint expressibility and gradient closure. It may request a separately
  selected minimal Model prerequisite, but this plan does not pre-authorize one.
- **Capability separation:** a represented operation never implies Compiler gradient support,
  Planning ownership, backend execution, or performance. Exact capability filtering precedes
  route selection and tuning; current unsupported combinations fail closed.
- **Primary risks:** treating Tensor/producers as graph IR; leaking runtime storage, backend
  capability, gradient lifecycle, or execution into Model; confusing logical layout with physical
  storage; deriving FLOAT16 policy from width; specializing recurrent topology to valid-length
  values; or widening public APIs and generic abstractions without a focused consumer.

## History and update policy

Current source, contracts, this table, and each brief’s Status/Result override older prose.
Evidence, inventories, context IDs, and chronology stay in linked briefs or Git. The legacy
branch is read-only capability evidence.

Update this plan only when Model ownership, package direction, ordered tasks, status, live gates,
risks, milestones, or frontier change. Do not retrospectively rewrite completed briefs. Create a
compact detailed brief only for the next selected frontier, after dependency and roadmap
authorization; a `Draft` row alone is not implementation permission.
