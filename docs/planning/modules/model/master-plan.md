# Model Master Plan

## Goal

Define an intentionally selected backend-independent tensor API, immutable graph model, public
tensor state, and host storage contracts suitable for useful inference and training.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Model capability baseline](capabilities.md)

## Scope

- data type, dimension, shape, and layout value models
- typed tensor, node, and value identifiers
- backend-independent operations and immutable attributes
- tensor descriptors, immutable graph values and nodes, and compiled graph state
- public `Tensor`, host storage abstractions, factories, and minimal provenance
- model-level representation and public expression construction for the selected capability
  baseline, using legacy behavior as evidence rather than design authority

## Out of scope

- graph compilation and optimization
- backend ownership and partition planning
- prepared execution, runtime residency, and device buffers
- backend-specific storage, lowering, or kernels

## Module invariants

- `Tensor` is public model state and is not an IR node; its existing borrowed host-storage
  association is its only mutable field.
- `Tensor` has no gradient/backward lifecycle state.
- `Operation` owns semantics and never backend support.
- Compiled graph state is immutable.
- Host storage never represents runtime device residency.

## Allowed dependencies

- JDK standard library
- No project dependency until a focused task demonstrates an architecture-compliant need.

## Forbidden dependencies

- planning, compiler, runtime, prepare, and engine modules
- concrete backend modules
- kernel selection, device residency, runtime state, and prepared execution

## Package structure

The module root `io.github.pho001.synaptik.model` is a namespace boundary, not the default destination for new types. Model contracts are grouped by cohesive responsibility:

```text
io.github.pho001.synaptik.model.datatype
  Data type metadata, promotion, host-independent bit conversion, and exact typed scalar values.

io.github.pho001.synaptik.model.shape
  Static and symbolic extent expressions, immutable shapes, axes, and local broadcasting.

io.github.pho001.synaptik.model.layout
  Resolved logical layout geometry and layout classification.

io.github.pho001.synaptik.model.tensor
  Public Tensor state, TensorId, TensorDescriptor, TensorFactory, eager initialization helpers,
  immutable expression-producer identity, canonical producer output wrappers, and indexed
  provenance.

io.github.pho001.synaptik.model.storage
  Host-visible storage contracts and implementations.

io.github.pho001.synaptik.model.operation
  Backend-independent operation semantics, compact signatures, and immutable attributes that may
  consume foundational typed scalar values.

io.github.pho001.synaptik.model.operation.elementwise.binary
  Typed parameterless semantic kinds for tensor-to-tensor elementwise arithmetic.

io.github.pho001.synaptik.model.operation.elementwise.unary
  Typed parameterless semantic kinds for unary arithmetic, transcendental functions,
  and activations.

io.github.pho001.synaptik.model.operation.elementwise.classification
  Typed parameterless floating value classifications with fixed BOOL results.

io.github.pho001.synaptik.model.operation.elementwise.scalar
  Typed parameterized semantic kinds and immutable attributes for scalar arithmetic and clamps.

io.github.pho001.synaptik.model.operation.elementwise.comparison
  Typed parameterless semantic kinds for ordered Tensor-to-Tensor comparisons.

io.github.pho001.synaptik.model.operation.elementwise.logical
  Typed parameterless semantic kinds for elementwise boolean conjunction, disjunction,
  and negation.

io.github.pho001.synaptik.model.operation.elementwise.selection
  Typed parameterless semantics for elementwise conditional branch selection.

io.github.pho001.synaptik.model.operation.elementwise.cast
  Typed explicit data-type conversion semantics and immutable target-type attributes.

io.github.pho001.synaptik.model.operation.reduction
  Typed aggregate-reduction meanings, normalized single- and ordered multi-axis parameters,
  correction-bearing statistical parameters, target-Shape SUM parameters, and shared arg-extrema
  tie policy.

io.github.pho001.synaptik.model.operation.scan
  Typed shape-preserving ordered scan meanings and immutable scan parameters.

io.github.pho001.synaptik.model.operation.normalization
  Typed shape-preserving normalization meanings and immutable normalization parameters.

io.github.pho001.synaptik.model.operation.loss
  Typed backend-independent loss meanings, explicit loss-reduction policy, and immutable
  family-specific loss parameters; no optimizer, training-session, gradient, or backend state.

io.github.pho001.synaptik.model.operation.linalg
  Typed backend-independent linear-algebra meanings; currently contains parameterless MATMUL.

io.github.pho001.synaptik.model.operation.attention
  Typed backend-independent scaled dot-product attention meaning and immutable scale/causal
  parameters, including exact one-output and explicit output-plus-normalized-weights occurrences;
  no dropout, algorithm, or backend route.

io.github.pho001.synaptik.model.operation.convolution
  Typed backend-independent NCHW two-dimensional convolution meaning and immutable stride,
  symmetric-padding, dilation, and group parameters; kernels are derived from weight Shape.

io.github.pho001.synaptik.model.operation.pooling
  Typed backend-independent NCHW pooling meanings and operation-specific immutable window
  parameters; max and average policies do not share an attributes type.

io.github.pho001.synaptik.model.operation.random
  Explicit graph RNG-state initialization and state-consuming stochastic operation semantics;
  distinct from eager host-data population in the tensor package.

io.github.pho001.synaptik.model.operation.ordering
  Stable axis-wise sort, argsort, and later top-K semantic identities and immutable parameters;
  no sorting algorithm or backend route.

io.github.pho001.synaptik.model.operation.layout
  Typed backend-independent layout and view operation meanings and immutable parameters.

io.github.pho001.synaptik.model.operation.index
  Typed backend-independent indexing, gather, functional-scatter, and index-encoding meanings
  and immutable normalized parameters.

io.github.pho001.synaptik.model.graph
  NodeId, ValueId, graph values/nodes, graph phase, publication binding,
  and immutable compiled graph state.
```

Package dependencies remain acyclic. `datatype` and `shape` are foundational leaves; `layout` may depend on `shape`; `storage` may depend on `datatype`; `operation` may consume foundational value contracts but must not depend on public `Tensor` or compiled graph state; `tensor` may compose foundational values, host storage, and operation provenance; and `graph` may compose tensor descriptors and operation semantics. Package-private helpers live in the package whose contracts they implement.

Operation-family subpackages are introduced only when a focused operation task demonstrates a cohesive boundary. Generic `util`, `common`, `internal`, and `misc` packages are not part of the planned structure.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [DataType model](tasks/0001-data-type-model.md) | Complete | - | Define data type categories, metadata, floating promotion, and BFLOAT16 conversion. |
| 0002 | [Shape and dimension model](tasks/0002-shape-and-dimension-model.md) | Complete | - | Define static and symbolic dimensions, immutable shapes, checked element counts, axes, and broadcasting. |
| 0003 | [Layout descriptor model](tasks/0003-layout-descriptor-model.md) | Complete | 0002 | Define resolved layout kinds, checked element strides, offset/span, and view metadata. |
| 0003A | [Data type package migration](tasks/0003a-data-type-package-migration.md) | Complete | 0001 | Move completed data type contracts into `model.datatype` without changing behavior. |
| 0003B | [Shape package migration](tasks/0003b-shape-package-migration.md) | Complete | - | Move completed dimension and shape contracts into `model.shape` without changing behavior. |
| 0003C | [Layout package migration](tasks/0003c-layout-package-migration.md) | Complete | 0003B | Move completed layout contracts into `model.layout` and preserve their shape imports. |
| 0004 | [Typed identifiers](tasks/0004-typed-identifiers.md) | Complete | 0003A–0003C | Define TensorId, NodeId, and ValueId in their owning domain packages. |
| 0005 | [Operation semantic foundation](tasks/0005-operation-semantic-foundation.md) | Complete | - | Define the minimal operation-kind and typed-attribute contracts without family-specific semantics. |
| 0006 | [Operation model](tasks/0006-operation-model.md) | Complete | 0005 | Define the minimal immutable backend-independent kind-and-attributes descriptor. |
| 0007 | [Tensor descriptor model](tasks/0007-tensor-descriptor-model.md) | Complete | 0001–0003, 0003A–0003C | Define data type, shape, explicit resolved/unresolved layout, and requires-grad descriptors. |
| 0008 | [Graph value and node model](tasks/0008-graph-value-and-node-model.md) | Complete | 0004, 0006, 0007 | Define immutable graph value and node records. |
| 0009 | [Compiled graph model](tasks/0009-compiled-graph-model.md) | Complete | 0008 | Define immutable graph container, forward/backward phase, and standalone publication binding. |
| 0010 | [Host storage abstraction](tasks/0010-host-storage-abstraction.md) | Complete | 0001, 0003A | Define exact-size borrowed Java 26 memory-segment host storage without device buffers. |
| 0011 | [Public Tensor skeleton](tasks/0011-public-tensor-skeleton.md) | Complete | 0004, 0007, 0010 | Define stable public Tensor identity/descriptor/label and synchronized optional host-storage state without graph or runtime state. |
| 0012 | [Tensor factory foundation](tasks/0012-tensor-factory.md) | Complete | 0010, 0011 | Expose descriptor-based public construction, optional borrowed storage attachment, and JVM-wide tensor-ID allocation without allocating memory. |
| 0012A | [JVM-managed heap host storage allocation](tasks/0012a-host-storage-allocation.md) | Complete | 0010, 0012 | Add exact-span typed primitive-array allocation through the existing borrowed heap-segment storage contract. |
| 0012B | [Flat typed tensor import](tasks/0012b-flat-typed-tensor-import.md) | Complete | 0012A | Import copied flat primitive arrays into dense-contiguous tensors with exact carrier and logical-count validation. |
| 0012C | [Nested typed tensor import](tasks/0012c-nested-typed-tensor-import.md) | Complete | 0012B | Infer exact type and static dense shape from validated rectangular primitive arrays, then flatten and delegate to typed flat import. |
| 0012D | [Constant tensor creation](tasks/0012d-constant-tensor-creation.md) | Complete | 0012B | Add exact typed rank-zero scalars and independent dense zeros, ones, zeros-like, and ones-like tensors. |
| 0012E | [Range and prefix population](tasks/0012e-range-and-prefix-population.md) | Complete | 0012B | Add typed integer ranges plus strict and cyclic exact-carrier prefix population under explicit validation. |
| 0012F | [Random tensor creation](tasks/0012f-random-tensor-creation.md) | Complete | 0012B | Add normally distributed floating tensors from an explicit caller-owned random source with bounded reproducibility. |
| 0012G | [Uniform random tensor creation](tasks/0012g-uniform-random-tensor-creation.md) | Complete | 0012F | Add continuous uniform floating tensors with explicit half-open bounds and the existing caller-owned source policy. |
| 0012H | [Integral random tensor creation](tasks/0012h-integral-random-tensor-creation.md) | Complete | 0012F | Add exact INT32/INT64 overloads with exclusive bounds and unbiased JDK bounded sampling. |
| 0012I | [Bernoulli random tensor creation](tasks/0012i-bernoulli-random-tensor-creation.md) | Complete | 0012F | Add BOOL tensors sampled from an explicit probability using the existing caller-owned source policy. |
| 0013 | [Tensor provenance skeleton](tasks/0013-tensor-provenance-skeleton.md) | Complete | 0006, 0011, 0012 | Attach immutable operation-and-input origin metadata to Tensor for future compiler-owned graph capture. |
| 0013A | [Full-value and identity-matrix tensor creation](tasks/0013a-full-value-and-identity-matrix-tensor-creation.md) | Complete | 0012B, 0012D | Add typed full-value tensors through canonical `full` and dense rectangular identity matrices, with `eye` exactly aliasing the canonical `identityMatrix` semantics. |
| 0014A | [Binary arithmetic semantic kinds](tasks/0014a-binary-arithmetic-semantic-kinds.md) | Complete | 0005, 0006 | Define typed parameterless ADD, SUB, MUL, DIV, MIN, MAX, and POW kinds. |
| 0014B | [Binary arithmetic Tensor expressions](tasks/0014b-binary-arithmetic-tensor-expressions.md) | Complete | 0013, 0014A | Build locally validated floating broadcast-aware Tensor expressions with ordered provenance. |
| 0014C | [Unary elementwise semantic kinds](tasks/0014c-unary-elementwise-semantic-kinds.md) | Complete | 0005, 0006 | Define fifteen parameterless unary arithmetic, transcendental, activation, and fast-approximation kinds. |
| 0014D | [Unary elementwise Tensor expressions](tasks/0014d-unary-elementwise-tensor-expressions.md) | Complete | 0013, 0014C | Build floating shape-preserving Tensor expressions with exact one-input provenance. |
| 0014E | [Scalar arithmetic and clamp semantics](tasks/0014e-scalar-arithmetic-and-clamp-semantics.md) | Complete | 0005, 0006 | Define typed scalar elementwise kinds plus one-value and clamp-range attributes. |
| 0014F | [Scalar arithmetic and clamp Tensor expressions](tasks/0014f-scalar-arithmetic-and-clamp-tensor-expressions.md) | Complete | 0013, 0014E | Build floating shape-preserving parameterized Tensor expressions with exact one-input provenance. |
| 0015A | [Binary comparison semantic kinds](tasks/0015a-binary-comparison-semantic-kinds.md) | Complete | 0005, 0006 | Define six typed parameterless ordered comparison meanings. |
| 0015B | [Binary comparison Tensor expressions](tasks/0015b-binary-comparison-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0015A | Build floating broadcast-aware comparisons with BOOL results and ordered provenance. |
| 0015C | [Boolean logical semantic kinds](tasks/0015c-boolean-logical-semantic-kinds.md) | Complete | 0005, 0006 | Define parameterless AND, OR, and NOT semantic meanings. |
| 0015D | [Boolean logical Tensor expressions](tasks/0015d-boolean-logical-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0015C | Build BOOL-only broadcast-aware logical expressions. |
| 0015E | [Where selection semantic kind](tasks/0015e-where-selection-semantic-kind.md) | Complete | 0005, 0006 | Define parameterless ternary conditional selection semantics. |
| 0015F | [Where selection Tensor expression](tasks/0015f-where-selection-tensor-expression.md) | Complete | 0001, 0002, 0013, 0015E | Build condition/branch-validated broadcast selection with ordered provenance. |
| 0015G | [Cast semantic kind and attributes](tasks/0015g-cast-semantic-kind-and-attributes.md) | Complete | 0001, 0005, 0006 | Define typed target-data-type cast semantics. |
| 0015H | [Cast Tensor expression](tasks/0015h-cast-tensor-expression.md) | Complete | 0001, 0013, 0015G | Build fresh explicit cast expressions without eager conversion or model-time canonicalization. |
| 0016A | [Reduction semantic kinds and attributes](tasks/0016a-reduction-semantic-kinds-and-attributes.md) | Complete | 0005, 0006 | Define ordinary aggregate meanings, axis/full parameters, and arg-max tie policy. |
| 0016B | [Sum, mean, and product Tensor expressions](tasks/0016b-sum-mean-and-product-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016A | Build floating full and single-axis aggregate expressions. |
| 0016C | [Min and max Tensor reduction expressions](tasks/0016c-min-and-max-tensor-reduction-expressions.md) | Complete | 0001, 0002, 0013, 0014A, 0014B, 0016A, 0016B | Extend the shared floating aggregate boundary with full and single-axis extrema expressions. |
| 0016D | [Boolean all and any Tensor expressions](tasks/0016d-boolean-all-and-any-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0015C, 0015D, 0016A, 0016B, 0016C | Generalize the shared aggregate boundary with BOOL full and single-axis all/any expressions. |
| 0016E | [Arg-max Tensor expressions](tasks/0016e-arg-max-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016A, 0016B, 0016C, 0016D | Build numeric single-axis INT64 index expressions with explicit or FIRST_INDEX tie policy. |
| 0016F | [Masked reduction semantics and axis mapping](tasks/0016f-masked-reduction-semantics-and-axis-mapping.md) | Complete | 0005, 0006, 0016A | Define typed SUM/MEAN masked attributes with explicit mask-dimension-to-input-axis mapping. |
| 0016F1 | [Masked sum and mean Tensor expressions](tasks/0016f1-masked-sum-and-mean-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016B, 0016F | Resolve legacy-compatible mask alignment and build axis-removing public expressions. |
| 0016G | [Cumulative-sum semantic kind and attributes](tasks/0016g-cumulative-sum-semantic-kind-and-attributes.md) | Complete | 0005, 0006 | Define typed axis, exclusive, and reverse scan semantics. |
| 0016H | [Cumulative-sum Tensor expressions](tasks/0016h-cumulative-sum-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016G | Build shape-preserving numeric cumulative-sum expressions. |
| 0016I | [Softmax semantic kinds and attributes](tasks/0016i-softmax-semantic-kinds-and-attributes.md) | Complete | 0005, 0006 | Define typed softmax and log-softmax axis semantics. |
| 0016J | [Softmax Tensor expressions](tasks/0016j-softmax-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0016I | Build floating shape-preserving softmax expressions. |
| 0017A | [Contiguous semantic kind](tasks/0017a-contiguous-semantic-kind.md) | Complete | 0005, 0006 | Define the parameterless request for canonical dense row-major result geometry without materialization policy. |
| 0017B | [Contiguous Tensor expression](tasks/0017b-contiguous-tensor-expression.md) | Complete | 0002, 0003, 0007, 0011–0013, 0017A | Build the public contiguous request with explicit static/dynamic descriptor and provenance rules. |
| 0017C | [Reshape and expand semantics](tasks/0017c-reshape-and-expand-semantics.md) | Complete | 0002, 0005, 0006 | Define immutable target-shape meanings without public Tensor construction. |
| 0017D | [Reshape Tensor expressions](tasks/0017d-reshape-tensor-expressions.md) | Complete | 0002, 0003, 0007, 0011–0013, 0017C | Build raw-inferred and exact-Shape reshape expressions with locally provable view geometry. |
| 0017D1 | [Expand Tensor expressions](tasks/0017d1-expand-tensor-expressions.md) | Complete | 0002, 0003, 0007, 0011–0013, 0017C | Build right-aligned singleton/leading-axis expansion expressions and zero-stride view geometry. |
| 0017E | [Axis-transform semantics](tasks/0017e-axis-transform-semantics.md) | Complete | 0002, 0005, 0006 | Define permutation, dimension insertion, and dimension removal meanings. |
| 0017F | [Permute and transpose Tensor expressions](tasks/0017f-permute-and-transpose-tensor-expressions.md) | Complete | 0002, 0003, 0013, 0017E | Build arbitrary axis-reordering and rank-two transpose convenience with view geometry. |
| 0017F1 | [Expand-dimensions and squeeze Tensor expressions](tasks/0017f1-expand-dimensions-and-squeeze-tensor-expressions.md) | Complete | 0002, 0003, 0013, 0017E | Build singleton-axis insertion/removal with rank-editing view geometry. |
| 0017G | [Slice semantics](tasks/0017g-slice-semantics.md) | Complete | 0002, 0005, 0006 | Define immutable general positive-step slice parameters and single-axis convenience meaning. |
| 0017H | [Slice Tensor expressions](tasks/0017h-slice-tensor-expressions.md) | Complete | 0002, 0003, 0013, 0017G | Build general and single-axis slice views with local shape and layout rules. |
| 0017I | [Pad and tile semantics](tasks/0017i-pad-and-tile-semantics.md) | Complete | 0001, 0002, 0005, 0006 | Define constant-padding and axis-repeat meanings and immutable parameters. |
| 0017J | [Pad and tile Tensor expressions](tasks/0017j-pad-and-tile-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0017I | Build shape-validated constant-pad and tile expressions without eager execution. |
| 0017K | [Tensor composition semantics](tasks/0017k-tensor-composition-semantics.md) | Complete | 0002, 0005, 0006 | Define concat, stack, and unstack meanings and immutable axis parameters. |
| 0017L | [Tensor composition expressions](tasks/0017l-tensor-composition-expressions.md) | Complete | 0001, 0002, 0013, 0017K | Build concat, stack, and multi-result unstack public expression contracts. |
| 0017M | [Unfold and fold semantics](tasks/0017m-unfold-and-fold-semantics.md) | Complete | 0002, 0005, 0006 | Define single-axis and two-dimensional window transformation parameters. |
| 0017N | [Unfold and fold Tensor expressions](tasks/0017n-unfold-and-fold-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0017M | Historically built public shape-validated unfold, foldAxis, unfold2d, and fold2d expressions without execution or kernels; completed task 0018R later removed public foldAxis. |
| 0018A | [Scalar select semantics](tasks/0018a-scalar-select-semantics.md) | Complete | 0002, 0005, 0006 | Define scalar-index axis-selection meaning and immutable normalized axis/index attributes. |
| 0018B | [Scalar select Tensor expression](tasks/0018b-scalar-select-tensor-expression.md) | Complete | 0002, 0003, 0013, 0018A | Build public scalar-index selection with axis removal and locally provable view geometry. |
| 0018C | [Axis gather semantics](tasks/0018c-axis-gather-semantics.md) | Complete | 0005, 0006 | Define distinct gather, gather-axis/take, and take-along-axis meanings and normalized axis parameters. |
| 0018D | [Axis gather Tensor expressions](tasks/0018d-axis-gather-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0018C | Build index-type and Shape-validated public axis-gather expressions. |
| 0018D1 | [Primitive take convenience](tasks/0018d1-primitive-take-convenience.md) | Complete | 0012B, 0018D | Add legacy `take(int, int[])` by creating one copied dense INT32 index Tensor and delegating to tensor-index take. |
| 0018E | [Gather-ND semantics](tasks/0018e-gather-nd-semantics.md) | Complete | 0005, 0006 | Define gather-ND meaning and immutable batch-dimension parameters. |
| 0018F | [Gather-ND Tensor expressions](tasks/0018f-gather-nd-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0018E | Build public gather-ND construction with index-depth and batch validation. |
| 0018G | [Axis scatter semantics](tasks/0018g-axis-scatter-semantics.md) | Complete | 0005, 0006, 0018C | Define functional scatter-add, scatter-axis-add, and scatter-elements meanings and reduction policy. |
| 0018H | [Axis scatter Tensor expressions](tasks/0018h-axis-scatter-tensor-expressions.md) | Complete | 0001, 0002, 0013, 0018G | Build public Shape/type-validated functional axis-scatter expressions. |
| 0018I | [Scatter-ND semantics](tasks/0018i-scatter-nd-semantics.md) | Complete | 0005, 0006, 0018E | Define functional scatter-ND meaning, reduction policy, and batch-dimension parameters. |
| 0018J | [Scatter-ND Tensor expression](tasks/0018j-scatter-nd-tensor-expression.md) | Complete | 0001, 0002, 0013, 0018I | Build public Shape/type-validated functional scatter-ND construction. |
| 0018K | [Operation signature and construction hardening](tasks/0018k-operation-signature-and-construction-hardening.md) | Complete | 0005, 0006, 0008 | Prevent invalid kind/attribute pairings and define compact fixed/bounded/variadic input and output cardinality without a registry. |
| 0018L | [Shared multi-output Tensor provenance](tasks/0018l-shared-multi-output-tensor-provenance.md) | Complete | 0007–0009, 0011–0013, 0018K | Represent one immutable shared producer with ordered inputs/output descriptors and indexed Tensor results without turning Tensor into IR. |
| 0018M | [Symbolic extent expressions](tasks/0018m-symbolic-extent-expressions.md) | Complete | 0002, 0017C–0017N | Represent canonical checked linear combinations, signed constant offsets, floor/ceiling division, and constrained unknown extents without runtime binding. |
| 0018M1 | [Dynamic extent adoption in pad, tile, and concat](tasks/0018m1-dynamic-extent-adoption.md) | Complete | 0017J, 0017L, 0018M | Replace conservative identity-only dynamic results with exact model-owned extent expressions in three bounded Tensor shape transformations. |
| 0018N | [Typed scalar value contract](tasks/0018n-typed-scalar-value-contract.md) | Complete | 0001, 0014E, 0014F, 0017I, 0017J, 0018K | Preserve exact scalar values for the six current data types and atomically make scalar, clamp, padding attributes, and public expression boundaries data-type-safe. |
| 0018O | [Indexing taxonomy and unstack normalization](tasks/0018o-indexing-taxonomy-and-unstack-normalization.md) | Complete | 0017K–0017L, 0018A–0018J, 0018K–0018L | Align gather/scatter primitives with selected terminology, remove misleading axis `take`, make unstack repeated select, and demote specialized adjoints. |
| 0018P | [Elementwise semantic cleanup](tasks/0018p-elementwise-semantic-cleanup.md) | Complete | 0014C–0014F, 0018K, 0018N | Atomically rename `INV`/`inv` to `RECIPROCAL`/`reciprocal`, remove both fast variants without aliases, and preserve the typed scalar family unchanged. |
| 0018Q | [Masked reduction redesign](tasks/0018q-masked-reduction-redesign.md) | Complete | 0015E–0015F, 0016A–0016F1, 0018M–0018N | Remove heuristic mapping, require explicit right-aligned broadcast-to-input masks, retain minimal first-class two-input SUM/MEAN, and define all-false mean as NaN. |
| 0018R | [Slice and window public-contract cleanup](tasks/0018r-slice-and-window-public-contract-cleanup.md) | Complete | 0017G–0017N, 0018K–0018M | Normalize signed non-zero slices as start/length/step sequences, add one-SLICE flip, and remove the public Tensor foldAxis receiver while retaining its public Java semantic contracts; task 0023D now owns public restoration. |
| 0018S | [Tensor factory surface cleanup](tasks/0018s-tensor-factory-surface-cleanup.md) | Complete | 0012–0012I, 0013A, 0018N | Keep identity/construction/import/constants/range in TensorFactory, promote TensorRandoms as the focused public random owner, and move prefix population to test-only fixtures. |
| 0018T | [Scalar arithmetic family normalization](tasks/0018t-scalar-arithmetic-family-normalization.md) | Complete | 0014A–0014B, 0014E–0014F, 0018K, 0018N, 0018P | Complete parallel seven-operation Tensor/binary and Tensor/scalar arithmetic, distinguish pairwise `minimum`/`maximum` from reductions, and demote one-bound clamp kinds to conveniences. |
| 0018T1 | [Unary numeric gaps and floating diagnostics](tasks/0018t1-unary-numeric-gaps-and-floating-diagnostics.md) | Complete | 0014C–0014D, 0018K, 0018P, 0018T | Add floating-preserving `rsqrt`/`log1p`/`expm1` plus separately typed BOOL `isFinite`/`isNaN`/`isInf` classification semantics and public construction. |
| 0018U | [Integral elementwise arithmetic and comparisons](tasks/0018u-integral-elementwise-arithmetic-and-comparisons.md) | Complete | 0001, 0014A–0015B, 0018K, 0018N, 0018T–0018T1 | Add same-category promotion, selected modular INT32/INT64 elementwise arithmetic, exact scalar domains, and all six integral comparisons. |
| 0018U1 | [Integral reductions and arg-min normalization](tasks/0018u1-integral-reductions-and-arg-min-normalization.md) | Complete | 0016A–0016E, 0018K, 0018U | Added exact-type modular integral SUM/PROD, signed MIN/MAX with bounded empty identities, and shared `argMin`/`argMax` attributes, ordering, tie, and empty-axis contracts. |
| 0018V | [Multi-axis and statistical reductions](tasks/0018v-multi-axis-and-statistical-reductions.md) | Complete | 0016A–0016J, 0018K, 0018M, 0018T1, 0018U1 | Added ordered multi-axis ordinary reductions plus first-class floating log-sum-exp, corrected variance/standard deviation, and L1/L2 norm semantics and Tensor construction. |
| 0019 | [Matmul semantics and Tensor expression](tasks/0019-matmul-semantics-and-tensor-expression.md) | Complete | 0001–0002, 0005–0007, 0011–0013, 0018K–0018N, 0018T, 0018U–0018V | Added first-class vector, matrix, and batched MATMUL metadata with exact Shape, type, numerical, and provenance contracts. |
| 0019A | [Modern activation semantics and Tensor expressions](tasks/0019a-modern-activation-semantics-and-tensor-expressions.md) | Complete | 0014C–0014D, 0018K, 0018P | Added exact GELU, fixed tanh-approximation GELU, and canonical SiLU as first-class floating unary semantics. |
| 0019A1 | [Embedding convenience](tasks/0019a1-embedding-convenience.md) | Complete | 0018K, 0018O | Added rank-two floating `weights.embedding(indices)` as validated axis-zero Gather composition with no padding option or new kind. |
| 0019A2 | [One-hot encoding](tasks/0019a2-one-hot-encoding.md) | Complete | 0001–0002, 0005–0007, 0011–0013, 0018K–0018O | Added first-class trailing-axis, positive-static-depth BOOL one-hot semantics for INT32/INT64 indices with invalid-value execution boundaries and no broad configuration. |
| 0019B | [Explicit graph RNG state foundation](tasks/0019b-explicit-graph-rng-state-foundation.md) | Complete | 0018K–0018L, 0018N, 0018S | Added an opaque public key/counter graph-state value and zero-input Tensor producer without a hidden generator or selected bitstream. |
| 0019B1 | [Explicit graph dropout construction](tasks/0019b1-explicit-graph-dropout-construction.md) | Complete | 0019B, 0018K–0018L | Added floating training dropout construction with explicit state input, auxiliary mask, next-state output, and no hidden mutation. |
| 0019C | [Sort and argsort](tasks/0019c-sort-and-argsort.md) | Complete | 0018K–0018L, 0018U–0018U1 | Added stable values-only sort and indices-only argsort with fixed NaN-last ordering and exact all-type Shape/provenance contracts. |
| 0019C1 | [Top-K values and indices](tasks/0019c1-top-k-values-and-indices.md) | Complete | 0019C, 0018L | Added focused TOP_K semantics, deterministic largest/smallest selection, static/deferred `k` validation, and one shared two-output values/INT64-indices producer. |
| 0019D | [Linear convenience](tasks/0019d-linear-convenience.md) | Complete | 0017F, 0014B, 0018K–0018N, 0018T, 0018U, 0019–0019C1 | Add conventional weight-transposed MATMUL plus optional exact rank-one bias as fully prevalidated explicit public composition without a LINEAR kind. |
| 0019E | [Scaled dot-product attention](tasks/0019e-scaled-dot-product-attention.md) | Complete | 0016I–0016J, 0018K–0018L, 0018N, 0018Q, 0019 | Added one-output attention semantics and four receiver overloads with exact query/key/value, mask, causal, scale, numerical, and deferred-constraint contracts, excluding dropout. |
| 0020 | [NCHW Conv2d semantics and Tensor expressions](tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md) | Complete | 0018K–0018L, 0018M–0018M1, 0018N, 0018V, 0019 | Added grouped NCHW cross-correlation with optional bias, exact static/symbolic Shape, floating numerical policy, and ordered provenance. |
| 0020A | [NCHW Max Pool2d semantics and Tensor expression](tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md) | Complete | 0020, 0017M–0017N, 0018K–0018L, 0018M–0018M1, 0018N, 0018V | Added floating max pooling with exact static/symbolic geometry, literal ceil windows, excluded padding, and deterministic extrema semantics. |
| 0020A1 | [NCHW Average Pool2d semantics and Tensor expression](tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md) | Complete | 0020A | Added floating average pooling with a dedicated attrs type, literal floor/ceil grid, fixed count-padding divisor, selected accumulation, and exact special/empty-window policies. |
| 0021 | [Layer normalization semantics and Tensor expressions](tasks/0021-layer-normalization-semantics-and-tensor-expressions.md) | Complete | 0018K, 0018L, 0018N, 0018V | Added one-output trailing-Shape layer normalization with exact typed epsilon and explicit all-or-none affine inputs. |
| 0021A | [RMS normalization semantics and Tensor expressions](tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md) | Complete | 0018K, 0018L, 0018N, 0018V, 0021 | Added distinct one-output root-mean-square normalization with exact no-scale/scale-only inputs, typed epsilon, and no layer-attrs reuse or bias. |
| 0021B | [Batch-normalization inference](tasks/0021b-batch-normalization-inference.md) | Complete | 0018K, 0018L, 0018N, 0018V, 0021–0021A | Added stateless layout-neutral per-channel inference with mandatory explicit affine/running-statistic inputs, exact typed epsilon, one output, and no hidden mode or mutation. |
| 0021C | [Batch-normalization training and statistic transition](tasks/0021c-batch-normalization-training-and-statistic-transition.md) | Complete | 0021B, 0018L, 0018N, 0018V | Added one pure five-input/five-output training occurrence with explicit next running statistics and hidden saved mean/inverse-standard-deviation outputs, without cross-step state ownership. |
| 0022 | [Mean-squared-error loss](tasks/0022-mean-squared-error-loss.md) | Complete | 0018K, 0018N, 0018V | Added the minimal dense regression loss with exact-shape targets and explicit `NONE`/`SUM`/`MEAN` reduction and empty-domain policy. |
| 0022A | [Dense-target categorical cross-entropy with logits](tasks/0022a-dense-target-categorical-cross-entropy-with-logits.md) | Complete | 0022, 0016I–0016J | Added one exact-shape floating-target logits loss with normalized class axis, stable target-weighted log-softmax, sample-count mean, and no ignore-index or broad options surface. |
| 0022B | [Index-target categorical cross-entropy with logits](tasks/0022b-index-target-categorical-cross-entropy-with-logits.md) | Complete | 0022A, 0018O | Added INT32/INT64 class-index targets with class-axis removal, exact typed optional ignore index, non-ignored mean denominator, and deferred execution-time bounds obligations while preserving dense dispatch. |
| 0023 | [Adjoint expressibility audit](tasks/0023-adjoint-expressibility-audit.md) | Complete | 0006, 0014A–0014F, 0015A–0015H, 0016A–0016J, 0017A–0017N including 0017D1 and 0017F1, 0018A–0019E including 0018D1, 0019A1, and 0019A2, 0020–0022B including 0020A–0020A1 and 0021A–0021C; post-0022B checkpoint | Audited the exact adjoint matrix and selected only proven generally useful public prerequisites; see the [result artifact](adjoint-expressibility-audit.md). |
| 0023A | [Binding-aware sum-to-Shape](tasks/0023a-binding-aware-sum-to-shape.md) | Complete | 0023 | Adds one exact target-Shape variant of existing SUM plus one public transformation for deferred singleton-or-equal MATMUL/attention batch axes; no new or operation-specific unbroadcast kind. |
| 0023B | [Gather-compatible scatter-add](tasks/0023b-gather-compatible-scatter-add.md) | Complete | 0023, 0023A | Added final `SCATTER_ADD` and one public fixed-add expression whose updates have the exact current Gather result Shape, preserving unresolved gathered extents and duplicate accumulation. |
| 0023C | [Slice update and target-relative crop](tasks/0023c-slice-update-and-target-relative-crop.md) | Complete | 0023, 0023B | Added functional signed/multi-axis slice replacement plus exact target/prefix-Shape crop for unresolved Pad/Concat extents, without overlap addition or binding. |
| 0023D | [Public foldAxis and dynamic window transforms](tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md) | Complete | 0023, 0023C | Restored public general-axis overlap-add fold, added exact typed-padding UNFOLD2D, and retained dynamic canonical rank-three columns through a canonical symbolic Dimension product. |
| 0023E | [Cumulative scan normalization and product](tasks/0023e-cumulative-scan-normalization-and-product.md) | Complete | 0016G–0016H, 0018K, 0018U–0018U1, 0023, 0023D | Atomically normalized the sum-only scan types into one CUM_SUM/CUM_PROD family, preserved public cumulative sum, and added the general product scan needed for zero-safe product adjoints. |
| 0023F | [Scaled dot-product attention weights output](tasks/0023f-scaled-dot-product-attention-weights-output.md) | Complete | 0018K–0018L, 0019E, 0023, 0023E | Preserved fluent one-output attention and added an explicit two-output result whose output and normalized weights share one exact producer occurrence. |
| 0024 | [Model capability and contract closure audit](tasks/0024-model-capability-and-contract-closure-audit.md) | Complete | 0001–0023F | Audited the exact model surface and checkpoint with a `BLOCKING_GAP` verdict: behavior is coherent, but one `GraphValue` Javadoc sentence incorrectly calls the current public Tensor planned. |
| 0024A | [GraphValue Tensor-status Javadoc correction](tasks/0024a-graph-value-tensor-status-javadoc-correction.md) | Complete | 0024 | Corrected the one stale `GraphValue` Javadoc status sentence, preserved every declaration and behavior, recorded historical audit closure, and passed focused Javadoc/documentation validation. |
| 0025 | [Canonical TensorProducer outputs](tasks/0025-canonical-tensor-producer-outputs.md) | Complete | 0018L, 0019B1, 0021C, 0023F, 0024A; prerequisite for Compiler 0004 | Retains and retrieves the canonical exact Tensor wrapper for every producer output slot through one factory-atomic, safely published occurrence, without changing Tensor methods or ergonomic result carriers. |
| 0025A | [Portable floating comparison, extrema, and clamp semantics](tasks/0025a-portable-floating-comparison-extrema-and-clamp-semantics.md) | Complete | 0018N, 0018T, 0018U, 0025; prerequisite for Compiler 0005A | Fixed one represented-value contract for floating comparisons, pairwise/scalar MIN/MAX, and ordered first-class CLAMP without executable model evaluation or derivative policy. |

## Milestones

- Value foundations and package organization: tasks 0001–0004, including 0003A–0003C
- Operation and immutable graph model: tasks 0005–0009
- Public tensor and host storage: tasks 0010–0013 and factory follow-ups 0012A–0012I and 0013A
- Initial public operation families: tasks 0014A–0014F, 0015A–0015H, 0016A–0016J including
  0016F1, 0017A–0017N including 0017D1 and 0017F1, and 0018A–0018J including 0018D1
- Capability reset and foundation hardening: tasks 0018K–0018V
  - foundation-contract checkpoint after 0018N;
  - public-surface cleanup checkpoint after 0018S; and
  - completed capability-reset checkpoint after 0018V.
- Selected modern operation families: tasks 0019–0022B, including 0019A–0019A2, 0019B–0019E,
  0020A–0020A1, and 0021A–0021C;
  checkpoint after
  0022B before adjoint-expressibility and missing-public-primitive planning
- Adjoint expressibility, its six evidence-selected public-capability follow-ups, the model
  capability-and-contract closure audit, and its bounded Javadoc gap: tasks 0023–0024A, including
  0023A–0023F before 0024 and completed 0024A after the audit
- Compiler-enabling producer foundation: task 0025, reopened only for the exact hidden-output
  wrapper prerequisite discovered by the pre-capture autograd architecture decision
- Compiler-gradient numerical-semantics prerequisite: task 0025A, reopened only to make the
  portable floating comparison/extrema/clamp forward contract explicit before Compiler 0005A
  chooses separate derivative policies

Each listed checkpoint runs the full repository test suite, affected architecture tests, final
Javadoc and documentation validation, and the cross-task checks deferred by the preceding tasks.
Individual single-module tasks use the task-level validation defined in the planning guide.

## Current status

The historical selected model milestone and focused task 0025 remain Complete. Focused task 0025A
is also Complete and closes the bounded model interleave before Compiler 0005A.
Tasks 0014A through 0015H remain complete with the post-0014B vertical-slice reassessment
recorded. The broad former task 0016 is decomposed into 0016A–0016J. Tasks 0016A through 0016E are
complete. Tasks 0016F, 0016F1, 0016G, 0016H, 0016I, and 0016J are also complete. The broad former
task 0017 is decomposed into focused tasks 0017A–0017N. Tasks 0017A and 0017B are complete; 0017C
is also complete. The former combined 0017D is split into reshape task 0017D and expand task
0017D1; the former combined 0017F is split into permutation task 0017F and singleton-rank-edit task
0017F1. Tasks 0017D, 0017D1, 0017E, 0017F, 0017F1, 0017G, 0017H, 0017I, and 0017J are complete.
Tasks 0017K, 0017L, 0017M, and 0017N are complete. The former broad task 0018 is decomposed into
focused tasks 0018A–0018J plus primitive-convenience task 0018D1. Tasks 0018A through 0018D1 are
complete. Task 0018E is complete with tuple-index semantics and normalized batch attributes. Task
0018F is complete with public Gather-ND expression construction. Task 0018G is complete with the
three axis-scatter meanings, reusable reduction vocabulary, and explicit scatter-elements
attributes. Task 0018H is complete with public metadata-only axis-scatter expression construction.
Task 0018I is complete with functional tuple-index semantics and immutable batch-count plus
shared-reduction attributes. Task 0018J is complete with public metadata-only Scatter-ND
expression construction. The capability-reset audit then replaced legacy parity as the selection
rule and inserted tasks 0018K–0018V before linear algebra. Task 0018K is complete with exact
family-owned signatures and local occurrence-cardinality validation. Task 0018L is complete with
unified producer/output-index provenance. Task 0018M is complete with canonical symbolic extent
values and conservative Shape integration. Task 0018M1 is complete with canonical symbolic
padding, tiling, and concat Shape derivation. Task 0018N is complete with exact typed scalar
representation, migrated attributes, and receiver-aware Tensor validation. Task 0018O is complete
with the final indexing taxonomy and repeated-SELECT unstack. Task 0018P is complete with the
final thirteen-kind unary vocabulary. Tasks 0018Q, 0018R, 0018S, 0018T, and 0018T1 are complete.
Task 0018U is complete with same-category numeric promotion, selected signed-integral elementwise
arithmetic, exact scalar domains, and all six integral comparisons. Task 0018U1 is complete with
exact-type integral reductions and the normalized shared arg-extrema family. Task
Completed task [0018V](tasks/0018v-multi-axis-and-statistical-reductions.md) closed the
capability-reset frontier. Its cohesive 17-path scope kept shared ordered-axis normalization,
semantic signatures, Shape/result construction, numerical policy, tests, and public
documentation in one compilable state. The former broad task 0019 is now decomposed. Focused
[task 0019](tasks/0019-matmul-semantics-and-tensor-expression.md) completed MATMUL. Completed
[task 0019A](tasks/0019a-modern-activation-semantics-and-tensor-expressions.md) added exact GELU,
fixed tanh-approximation GELU, and SiLU semantics and Tensor expressions. Completed
[task 0019A1](tasks/0019a1-embedding-convenience.md) added embedding as direct axis-zero Gather.
Completed [task 0019A2](tasks/0019a2-one-hot-encoding.md) added first-class trailing-axis BOOL
one-hot semantics. The former broad 0019B frontier is now split: completed
[task 0019B](tasks/0019b-explicit-graph-rng-state-foundation.md) owns the explicit graph RNG state
foundation, while completed [task 0019B1](tasks/0019b1-explicit-graph-dropout-construction.md)
owns explicit-state dropout construction. The former 0019C row is now split: completed
[task 0019C](tasks/0019c-sort-and-argsort.md) owns full stable sort/argsort, while 0019C1 owns
genuine multi-output top-K and is Complete. Completed
[task 0019D](tasks/0019d-linear-convenience.md) adds explicit linear composition. Completed
[task 0019E](tasks/0019e-scaled-dot-product-attention.md) retains its established ID and delivers
semantics, attrs, public construction, API-locking tests, and documentation together in 17 paths;
no 0019E1 split was needed. The former broad 0020 frontier is split without renumbering later
tasks: focused [task 0020](tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md) is Complete
for NCHW convolution. The former combined pooling follow-up is now split: focused
[task 0020A](tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md) is Complete for max
pooling. Focused average-pooling follow-up
[0020A1](tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md) is also Complete and
was the only detailed specification after 0020A. The former broad 0021 row is now split without
renumbering 0022–0024: focused task 0021 is Complete for layer normalization, and focused task
[0021A](tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md) is Complete for RMS
normalization. Focused [task 0021B](tasks/0021b-batch-normalization-inference.md) is Complete.
Task [0021C](tasks/0021c-batch-normalization-training-and-statistic-transition.md) is Complete.
Completed [task 0022](tasks/0022-mean-squared-error-loss.md) establishes shared loss reduction.
[Task 0022A](tasks/0022a-dense-target-categorical-cross-entropy-with-logits.md) is Complete. Task
0022B is Complete. Task 0023 is Complete with its final
[audit matrix](adjoint-expressibility-audit.md). Task 0023A is Complete with its detailed
specification. It adds no kind: SUM alone appends exact `SumToShapeAttrs`, and one public
`sumToShape(Shape)` expression retains exact numeric metadata plus unresolved right-aligned
target-one-or-equal obligations. The focused 14-suite run passed 131 tests, the replacement final
model suite passed 977 tests, and the separate documentation pass validated model Javadoc,
examples, Markdown, exact 25-path scope, the 189-method public surface, and synchronized status.
Tasks 0023B, 0023C, 0023D, 0023E, and 0023F are Complete with their detailed specifications, and
[task 0024](tasks/0024-model-capability-and-contract-closure-audit.md) is Complete with a
`BLOCKING_GAP` verdict. The exact model behavior and semantic representation are coherent, but
that historical audit found `GraphValue` Javadoc incorrectly called the current public mutable
Tensor API planned. Focused
[task 0024A](tasks/0024a-graph-value-tensor-status-javadoc-correction.md) is Complete: the wording
now says current, every declaration and behavior remains unchanged, and the selected model
capability milestone is closed.
Task 0023B's
focused 15-suite run passed 124 tests, its
single final model suite passed 981 tests across 125 suites, and the separate documentation pass
validated model Javadoc, the executable example, Markdown, exact 26-path scope, the 190-method
public Tensor surface, and synchronized status.
Task 0023C adds exact `SLICE_UPDATE`/`SliceAttrs` functional replacement and
`SLICE`/`CropToShapeAttrs` target-relative extraction plus exactly two public Tensor methods. Its
focused 15-suite run passed 139 tests and its single final model suite passed 996 tests across 126
suites. The separate documentation pass validated model Javadoc, a runnable Java 26 update/crop
metadata example, Markdown and official references, exact 27-path scope, the 192-method public
Tensor surface, and synchronized Complete/Draft status.

Completed [task 0023D](tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md) selects the
existing canonical rank-three im2col/col2im representation rather than adding a second
non-flattened window format. It adds one canonical symbolic Dimension-product form for exact
`outputHeight * outputWidth`, restores the retained public `foldAxis`, extends existing
unfold2d/fold2d construction to dynamic channel and spatial Dimensions, and adds one exact typed
padding-value UNFOLD2D variant. Existing direct zero-padding producers, static behavior, kinds,
and architecture ownership remain unchanged.

Task 0023D's focused 17-suite run passed 175 tests, and its single final model suite passed 1,008
tests across 126 suites with no failures, errors, or skips. The independent documentation pass
finalized all nine affected production Javadocs, Tensor/Compile APIs, glossary and planning
records, then validated model Javadoc, a runnable Java 26 metadata example, generated API pages,
the 194-method public Tensor surface, Markdown, exact 33-path scope, status, and whitespace.

Completed [task 0023E](tasks/0023e-cumulative-scan-normalization-and-product.md) atomically
replaces the sum-only semantic/helper type names with the shared `CumulativeScanKind`,
`CumulativeScanAttrs`, and `TensorCumulativeScanExpressions` family. It preserves both public
`cumSum` overloads and adds exactly two `cumProd` overloads with selected integral modular,
floating special-value, multiplicative-positive-one boundary, and zero-length-axis meanings. Its
focused run passed 44 tests across five suites, and its single final model suite passed 1,008 tests
across 126 suites with no failures, errors, or skips. The independent documentation pass finalized
the affected Javadocs, Tensor/Compile APIs, glossary and planning records, then validated model
Javadoc, Java 26 API reflection, generated API pages, the 196-method public Tensor surface,
Markdown, exact 33-path scope, status, and whitespace without repeating executable Java tests.

Completed [task 0023F](tasks/0023f-scaled-dot-product-attention-weights-output.md) preserves all
four one-output attention methods and adds four explicit output-plus-normalized-weights methods,
one two-component public result record, and one shared two-output producer form under the existing
attention kind. Its focused run passed 40 tests across five suites, and its single final model
suite passed 1,016 tests across 127 suites with no failures, errors, or skips. The independent
documentation pass finalized the four affected production Javadocs, Tensor/Compile APIs, glossary,
and planning records, then validated model Javadoc, a runnable Java 26 metadata example, exact
reflection and generated API pages, the 200-method public Tensor surface, Markdown, exact 27-path
scope, synchronized Complete/Draft status, and whitespace without rerunning executable Java tests.

[Task 0023](tasks/0023-adjoint-expressibility-audit.md) executed after the completed post-0022B
capability checkpoint. Its [planning-only matrix](adjoint-expressibility-audit.md) finds no proven
compiler-only semantic gap and selects six general public prerequisites: binding-aware
sum-to-Shape, Gather-compatible axis scatter-add, slice placement plus dynamic crop, public
foldAxis plus redesigned dynamic/configurable 2D windows, cumulative product, and same-occurrence attention
weights. Current Scatter
Elements and Scatter-ND exactly serve Gather Elements and Gather-ND adjoints. Typed scalar leaves
expanded to a target Shape provide uncontaminated dynamic zeros and ones. Maximum-pool routing can
be recomputed through existing first-index arg-maximum semantics once dynamic windows exist, so it
needs no indices-output task. Tasks 0023A and 0023B are Complete with detailed specifications.
Tasks 0023C, 0023D, 0023E, and
[task 0023F](tasks/0023f-scaled-dot-product-attention-weights-output.md) are Complete with detailed
specifications.
Operation-specific backward kinds, compiler traversal, execution, backend/runtime behavior,
Gradle, dependencies, and architecture changes remain absent; 0023A–0023F are the completed
selected public prerequisites.

Task 0020 adds one `CONV2D` meaning, immutable geometry/group attributes, and two public receiver
methods for grouped NCHW cross-correlation with optional bias. It preserves exact batch and
output-channel Dimensions, derives checked static or canonical symbolic spatial extents, promotes
floating inputs in occurrence order, and records one fresh exact two- or three-input producer.
The implementation context passed 41 focused tests across seven suites and one final model run of
860 tests across 109 suites. Independent context
`/root/task_0020_implementation/task_0020_docs` finalized Javadocs, Tensor/Compile APIs, glossary,
and planning records, then passed model Javadoc, example, Markdown, exact-scope/surface/status,
and whitespace validation without repeating the successful Java suites.

Task 0019D adds conventional `[outFeatures, inFeatures]` weight-transposed MATMUL plus optional
exact rank-one bias as visible PERMUTE -> MATMUL -> optional ADD composition. Complete local
validation precedes intermediate IDs; no-bias allocates two wrappers and returns MATMUL, while
bias allocates three and returns ADD. The biased Shape is structurally equal to the product Shape
and reuses its exact Dimension references, but its outer Shape object may differ. The
implementation context passed 60 focused tests and one final 836-test/105-suite model run;
independent documentation review finalized Javadocs, Tensor/Compile APIs, glossary, planning,
the runnable producer-chain example, and all required surface, Markdown, scope, status, and
whitespace checks.

Task 0018U keeps the public Tensor surface at 127 methods and adds no operation kind. INT32/INT64
Tensor pairs promote within their category, exact scalar attributes do not promote, integral
ADD/SUB/MUL use fixed-width two's-complement modular meaning, and integral extrema/comparisons use
signed order. Integral DIV, POW, range CLAMP, reductions, and arg-min remain outside the completed
elementwise baseline.

Task 0018S leaves exactly 31 public TensorFactory construction/import/constant/range methods,
makes field-free `TensorRandoms` the sole public owner of five explicit caller-source random
entries, moves range mechanics to package-private `TensorRanges`, and removes prefix preparation
from production. Package-private test-source `TensorTestData` retains the fixture behavior through
public flat import. The implementation context passed 58 focused tests and the 715-test root
checkpoint; independent documentation review finalized Javadocs, Tensor API, glossary, planning
status, a runnable public example, and the required generated-Javadoc, Markdown, surface, scope,
status, terminology, and whitespace checks.

Task 0018T completes parallel `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, and `POW` binary/scalar
arithmetic, renames only pairwise public extrema to `minimum`/`maximum`, retains reduction
`min`/`max`, and makes one-bound clamps scalar MAX/MIN conveniences while range CLAMP stays first-
class. The implementation context passed the six-suite focused command and all 715 model tests
across 88 suites. Independent documentation review finalized five affected Javadocs, Tensor and
Compile APIs, glossary, capability/task/master/roadmap records, generated Javadoc, a compiled Java
26 surface example, Markdown links/anchors, removed-vocabulary, exact 18-path, status, formatting,
and whitespace validation. The original 17-path maximum was explicitly expanded only for stale
pairwise calls in `TensorNumericReductionTest`.

The capability baseline is documented and the ordered task queue covers its model-level
responsibilities. Tasks 0001 through 0007 and package migrations 0003A–0003C are complete. Task
0008, graph value and node model, task 0009, compiled graph model, and task 0010, host storage
abstraction, are complete. Task 0011, public Tensor skeleton, and task 0012, the bounded Tensor
factory foundation, are also complete. Task 0012A, JVM-managed heap host storage allocation, is
complete. Task 0012B, flat typed tensor import, is also complete. Task 0012C, nested typed tensor
import, is complete. Task 0012D, constant tensor creation, and task 0012E, deterministic range and
prefix population, are also complete. Normal population task 0012F, uniform population task 0012G,
and integral population task 0012H are complete. Bernoulli task 0012I and provenance task 0013 are
also complete. Full-value and identity-matrix factory task 0013A is complete. The foundation
checkpoint selected continued sequential model operation-family work. Task 0014A, binary
arithmetic semantic kinds, and task 0014B, public binary arithmetic Tensor expressions, are
complete. The explicitly authorized Compile API correction now distinguishes current public
Tensor expressions from the still-planned compiler capture lifecycle.

## Open questions

- Exact public overloads and operation-attribute record boundaries remain local to the applicable
  focused task.
- Task 0022A fixes the dense-target categorical-loss contract. Completed task 0022B fixes the index,
  bounds, ignore, and non-ignored denominator details.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- ADR 0009 reopens the model queue for exactly one compiler-enabling prerequisite. Task 0025
  changes the completed 0018L producer contract only by retaining the canonical exact wrapper for
  every output slot and exposing the smallest indexed retrieval surface. It adds no derivative
  rule, Tensor method, graph identity, registry, or runtime state.
- The post-Compiler-0005 reassessment reopens the model queue for focused task 0025A before
  Compiler 0005A. The task clarifies existing Javadocs and APIs only: floating comparisons use
  ordinary ordered numeric relations and numeric equality, MIN/MAX propagate NaN and select the
  directional signed zero, and first-class CLAMP is ordered
  `MIN(MAX(input, minValue), maxValue)`. It adds no evaluator, policy object, operation, Tensor
  method, data type, backend behavior, or derivative convention.
- The selected capability baseline is defined by semantic coherence and a useful
  inference/training target, not by blanket legacy parity.
- The former broad loss row is split without renumbering established tasks 0023–0024. Completed
  task 0022 establishes one exact-shape mean-squared-error operation and shared explicit loss
  reduction. Completed task 0022A owns one exact-shape dense floating categorical cross-entropy from
  logits with normalized class axis, stable target weighting, and sample-count mean; completed task
  0022B owns index-target logits cross-entropy, class-axis removal, ignore index, and its
  non-ignored denominator. Standalone probability-input cross entropy, standalone log-probability
  negative-log-likelihood, weights, masks, label smoothing, binary cross entropy, and additional
  losses are not selected for this minimal frontier.
- Task 0023 is an audit-only adjoint-expressibility frontier, not a catalog of operation-specific
  backward kinds. Its [completed matrix](adjoint-expressibility-audit.md) distinguishes
  exact current composition, existing auxiliary outputs, reusable primitive gaps,
  non-differentiable roles, and deferred derivative policy. Formula complexity, performance, and
  fusion never justify model semantics.
- The audit selects exactly six generally useful public-capability rows: completed task 0023A
  binding-aware sum-to-Shape, completed task 0023B final Gather-compatible `SCATTER_ADD`, completed
  task 0023C functional signed slice update plus target-relative symbolic crop, 0023D public
  foldAxis and dynamic/configurable 2D windows, 0023E cumulative product, and 0023F same-occurrence
  attention weights. Each rejects
  a narrower backward-only spelling and depends on 0023. Task 0024 depends on all six and is
  Complete with its detailed audit and `BLOCKING_GAP` verdict. It is the planning-only model exit gate, not another
  capability redesign: it inventories the completed public and semantic contracts, checks them
  against the intentional baseline and exclusions, runs the repository/architecture/documentation
  checkpoint, and reports any bounded follow-up without implementing it. Completed task 0023C
  selects one appended `SLICE_UPDATE`
  identity paired with existing normalized `SliceAttrs`, one target/prefix-Shape attributes
  variant of `SLICE`, and exactly two public transformations. It uses functional replacement,
  preserves all existing extraction behavior, and retains unresolved crop-bound obligations
  without changing Shape or compiler ownership.
- Current Scatter Elements and Scatter-ND exactly express Gather Elements and Gather-ND adjoints.
  Positive-static-depth Gather additionally composes through one-hot selection and reduction;
  unresolved gathered depth is the 0023B gap. Exact typed scalar leaves expanded to a target Shape
  provide dynamic zero/one expressions.
  Max-pool selection can be recomputed from first-index arg-maximum after 0023D. These conclusions
  remove three speculative follow-ups.
- Completed task 0023A adds no kind: it extends existing `AggregateReductionKind.SUM` with exact
  `SumToShapeAttrs`, one public `sumToShape(Shape)` method, and right-aligned target-one-or-equal
  validation. It supplies the general transformation selected for current deferred
  singleton-or-equal batch obligations in MATMUL and attention while remaining independently
  usable. Ordinary binary, `where`, linear, and `EXPAND` reversal uses statically known axes.
  Task 0023D must choose additional Shape expressibility or a non-flattened dynamic window
  contract because the current rank-three columns require a product of two unresolved extents.
- No `GENUINELY_NON_EXPRESSIBLE_SEMANTIC_GAP` remains after the general-primitive test, so the audit
  adds no compiler-only semantic row. Saved statistics, dropout masks, and top-K indices continue
  to use indexed shared-producer outputs. No gradient, traversal, capture, execution, backend,
  runtime, Gradle, dependency, architecture, or public API implementation is claimed. Completed
  task 0024A was limited to the audit's one stale `GraphValue` Javadoc status sentence.
- The former broad task 0019 is split without renumbering established 0019A–0019C. Task 0019 is
  the cohesive MATMUL primitive; 0019D owns `linear`; 0019E owns scaled dot-product attention.
- The former 0019A umbrella is split without changing 0019B–0019E. Completed task 0019A owns exact
  GELU, fixed tanh-approximation GELU, and canonical SiLU as first-class parameterless unary kinds.
  Literal primitive compositions are insufficient because their infinity-times-zero intermediate
  does not preserve the selected negative-infinity continuous extension. Completed task 0019A1
  adds rank-two `weights.embedding(indices)` as axis-zero Gather composition with no padding
  option.
  Completed task 0019A2 adds trailing-axis BOOL one-hot with positive static depth, invalid
  negative/out-of-range execution values, and no configurable axis, result type, or on/off values.
  Existing `RELU`/`relu()` remains complete under tasks 0014C–0014D and is not duplicated. Completed
  task 0019B owns explicit graph RNG state, while completed task 0019B1 is the sole dropout owner.
- MATMUL is one first-class `MATMUL` kind with no attributes, two inputs, one output, and one
  public `matmul(Tensor)` method. It follows rank-one promotion/removal and right-aligned batch
  broadcasting across vector-vector, matrix-vector, vector-matrix, matrix-matrix, and batched
  inputs. Locally provable incompatibilities fail; contraction equality may defer because it does
  not affect output Shape, while batch deferral is accepted only when the exact output extent is
  still derivable.
- MATMUL accepts only same-category floating or signed-integral inputs through current promotion.
  BFLOAT16 results use FLOAT32 accumulation; other floating results accumulate in their promoted
  type without a bitwise-order guarantee. Integral results are the promoted-width modular sum of
  products. Layout remains unresolved and gradient eligibility is input-request OR.
- `linear` is not a model operation kind. Completed task 0019D adds `linear(weight)` and
  `linear(weight, bias)` as explicit composition `input.matmul(weight.transpose())` plus optional
  ADD. Weight Shape is `[outFeatures, inFeatures]`; input rank is at least one with final extent
  `inFeatures`; optional bias is exactly rank one `[outFeatures]` with structural Dimension
  equality. It inherits current floating and signed-integral MATMUL/ADD promotion and numerical
  policies. Complete validation precedes intermediate IDs. No-bias provenance ends at MATMUL after
  two wrappers; bias provenance ends at ADD after three. Transpose may retain resolved view layout,
  while MATMUL and ADD are unresolved. No LINEAR kind, layer state, gradient rule, compiler pass,
  or backend behavior is added.
- Scaled dot-product attention remains a distinct first-class high-level semantic operation in
  task 0019E because mask, causal, scale, softmax, and all-masked-row meaning must survive compiler
  inspection. Completed task 0019E originally used one `SCALED_DOT_PRODUCT_ATTENTION` kind, input
  range three to four, exactly one output, ordered inputs `[query, key, value]` or
  `[query, key, value, mask]`, and
  immutable `ScaledDotProductAttentionAttrs(Optional<ScalarValue> scale, boolean causal)`.
  The public receiver methods are `scaledDotProductAttention(key, value)`,
  `scaledDotProductAttention(key, value, attrs)`,
  `scaledDotProductAttention(key, value, mask)`, and
  `scaledDotProductAttention(key, value, mask, attrs)`. This public operation-specific attrs value
  is inspectable semantic state, not a broad options framework. Absent scale selects the default,
  while a present scale must be finite, positive, floating, and exactly match the promoted
  query/key/value type. Query `[...batch, L, E]`, key
  `[...batch, S, E]`, and value
  `[...batch, S, Ev]` use broadcast batch/head prefixes and produce
  `[...broadcastBatch, L, Ev]`. Inputs are floating; BOOL mask `true` participates and `false`
  masks, broadcasting exactly to score Shape `[..., L, S]`; causal masking additionally retains
  key positions `j <= i`; the default scale is `1 / sqrt(E)` and explicit scale is finite and
  positive; softmax is over final key axis `S`; an all-masked row has zero weights and zero output.
- Completed task 0023F preserves all four initial one-output methods without a hidden weights
  descriptor and widens the same kind's sole signature to one through two outputs. Four explicit
  `scaledDotProductAttentionWithWeights` forms return output slot zero and normalized weights slot
  one from one exact shared producer. Attention still has no dropout parameter or RNG ownership
  and no technical dependency on task 0019B, although table order places it afterward. Task 0019B
  owns graph RNG state and completed task 0019B1 owns dropout; a later attention-dropout extension
  must consume that explicit state rather than hide a generator.
- The former broad 0019B frontier is split without changing established tasks 0019C–0019E.
  Completed task 0019B adds public opaque `GraphRngState`, zero-input/one-output
  `GraphRngKind.INITIAL_STATE`, and exact `GraphRngStateAttrs(long key, long counter)`. Both words
  are unsigned 64-bit bit patterns; the state Tensor is fixed `INT64 Shape[2]`, unresolved,
  non-gradient, unlabeled, storage-free, and producer output zero. State objects use expression
  identity equality. The model selects no PRNG algorithm or cross-backend bitstream.
- Completed task 0019B1 owns `Tensor.dropout(double, GraphRngState)` and public
  `DropoutResult(output, nextState)`. One producer consumes `[input, state]` and produces
  `[output, auxiliaryMask, nextState]`. The hidden same-Shape BOOL mask supports compiler-owned
  backward construction. Drop probability is finite in `[0,1)`, kept values use inverted scaling,
  every element consumes one draw including probability zero, empty tensors consume none, and
  dynamic Shapes advance by their bound execution count. Inference bypasses dropout and state
  advancement. Its detailed specification fixes three wrapper/ID outputs because the current
  factory seam constructs one indexed Tensor per producer slot.
- The current unconstrained `Operation(kind, attrs)` pairing is not an acceptable stable contract.
  Task 0018K adds compact family-owned signature validation, including occurrence cardinality,
  without a global registry.
- Task 0018K uses one exact-attribute-class `OperationSignature` value with inclusive input and
  output bounds. Each kind family owns a stable non-empty variant list; `Operation` resolves its
  pair immediately and `CompiledNode` validates only local occurrence counts.
- Task 0018K completed that contract across every current production family. Its independent
  documentation review finalized affected Javadocs, Tensor API, Compile API, glossary, task
  evidence, master plan, and roadmap after the final 743-test/86-suite model run, model Javadoc,
  413-link/110-anchor Markdown, fence/final-newline, scope, and whitespace checks passed.
  Operand-aware and graph-wide validation, compiler/backend/runtime behavior, and shared
  multi-output Tensor provenance remain outside this task.
- Genuine multi-output operations require shared producer provenance. Unstack is instead repeated
  scalar select and does not retain a distinct UNSTACK semantic primitive.
- Shared provenance uses one identity-bearing `TensorProducer` with exact operation, ordered input
  Tensors, and ordered output descriptors. Each result records that exact producer and its output
  index. Output count is derived from descriptors, and single-output expressions use the same
  model at index zero. Completed task 0025 intentionally supersedes only the historical
  no-output-wrapper constraint: the producer retains and retrieves each canonical exact output
  Tensor so pre-capture compiler rules can use hidden auxiliaries without reconstruction.
- Completed task 0019C keeps full `SORT` and `ARGSORT` as distinct one-output occurrences. It selects
  unconditional stability, deterministic logical-index ties, NaNs last for both directions,
  negative-zero-before-positive-zero ascending, all six current input types, values-only sort,
  and INT64 indices-only argsort. Completed 0019C1 separately owns focused `TopKKind.TOP_K`, top-K's
  `k`-replaced Shape, and shared `[values, indices]` producer, preserving the existing multi-output
  foundation without a registry or service locator. Its final 827-test/104-suite model run,
  independent documentation review, clean Javadoc, runnable construction example, 565-link/
  154-anchor Markdown checks, exact 18-path audit, repository checkpoint, synchronized status,
  and whitespace checks passed.
- Task 0018L completed that contract and atomically migrated every current expression helper.
  Independent documentation review finalized affected Javadocs, Tensor API, Compile API,
  glossary, task evidence, master plan, and roadmap after the final 749-test/87-suite model run,
  model Javadoc, compiled example, Markdown, scope, status, and whitespace checks passed. Current
  unstack remains independent one-output producers; no production multi-output operation or
  compiler capture was added.
- Dynamic convolution and pooling require symbolic extent expressions for addition, constant
  multiplication, and floor/ceiling division before those families become Ready.
- Max and average pooling are separate tasks. They share NCHW window-coordinate vocabulary but
  not one attributes contract: max owns excluded padding and extrema ordering/ties, while average
  owns divisor, padding-count, accumulation, and invalid-divisor semantics. Combining both would
  exceed the established 18-path capability scope.
- Task 0020A selects one `MAX_POOL2D` kind, one-input/one-output `MaxPool2dAttrs`, and
  `Tensor.maxPool2d(attrs)`. Floor and ceil output extents use the literal symmetric padded grid;
  ceil mode does not remove a terminal window whose start lies in trailing padding. Padding
  samples are excluded, an all-padding window returns negative infinity, NaNs dominate, positive
  zero is greater than negative zero, and equal candidates select the first height-major kernel
  sample.
- Completed task 0020A1 adds exactly `averagePool2d(AveragePool2dAttrs)` and extends `Pool2dKind` with
  `AVERAGE_POOL2D` without changing max semantics. It reuses the literal floor/ceil coordinate
  grid but fixes count-padding as operation meaning: every kernel position contributes to the
  divisor, padding contributes conceptual positive zero, and the divisor is always the positive
  mathematical kernel-height-times-kernel-width product. There is no count flag, divisor
  override, or valid-sample mode. BFLOAT16/FLOAT32 accumulate and divide in FLOAT32, FLOAT64 in
  FLOAT64; an all-padding result is positive zero, while an exact-zero finite mean is negative zero
  only when every divisor contribution is an in-bounds negative zero. NaN propagates, opposing
  infinities produce NaN, one infinity sign is retained, and conforming reassociation is allowed
  without bitwise or cross-backend rounding identity.
- The former broad normalization row is split by semantic and lifecycle boundary without
  renumbering tasks 0022–0024. Completed task 0021 owns deterministic one-output layer normalization;
  completed task 0021A owns the distinct RMS formula; completed task 0021B owns stateless one-output
  batch-normalization inference with mandatory explicit affine and running-statistic inputs; and
  completed task 0021C owns training-time batch statistics, explicit running-stat transition,
  saved statistics, and genuine multi-output provenance. No model-level training/evaluation flag
  or hidden mutable state is selected.
- Task 0021 selects one `LAYER_NORM` kind with exact no-affine `[input]` and affine
  `[input, scale, bias]` one-output variants. Separate `LayerNormAttrs` and
  `AffineLayerNormAttrs` preserve those disjoint cardinalities under the current exact-class
  signature contract. Both retain a non-empty trailing normalized Shape and exact typed finite
  positive epsilon. The affine result promotes floating inputs in occurrence order; population
  variance uses correction zero; BFLOAT16/FLOAT32 accumulate in FLOAT32 and FLOAT64 in FLOAT64.
  Saved mean/inverse-standard-deviation outputs remain compiler concerns rather than public task-
  0021 results.
- Task 0021A selects one `RMS_NORM` kind, one `RmsNormAttrs` value, and a signature whose inclusive
  one-to-two input range safely represents exact `[input]` and `[input, scale]` occurrences. It
  adds `rmsNorm(normalizedShape, epsilon)` and
  `rmsNorm(normalizedShape, scale, epsilon)`, requires scale Shape exactly equal to the non-empty
  trailing normalized Shape, and adds no bias. The formula is the uncentered
  `x / sqrt(mean(x * x) + epsilon)` with population divisor `N`, no correction option, and exact
  result-typed finite positive epsilon. No-scale results retain input type; scaled results promote
  input and scale in occurrence order. BFLOAT16/FLOAT32 accumulate in FLOAT32 and FLOAT64 in
  FLOAT64. Empty, symbolic, NaN, infinity, signed-zero, overflow, reassociation, freshness, and
  exact one-output provenance policies are fixed without specifying execution.
- Task 0021A's implementation context passed the exact focused command and final 908-test model
  suite. Independent documentation review finalized four production Javadocs, Tensor/Compile
  APIs, glossary, capability/task/master/roadmap records, generated Javadoc, 607 local links with
  165 anchors, official references, exact 183-method surface, exact 19-path scope, synchronized
  status, fences/newlines/whitespace, and `git diff --check` without repeating Java tests.
- Task 0018M completed canonical positive-coefficient linear combinations with signed constant
  offsets, explicit floor/ceiling division, identity-based bounded unknowns, non-static Shape
  inspection, readable diagnostics, and structurally conservative broadcasting. It owns only the
  shape value foundation; completed task 0018M1 adopts that foundation in pad, tile, and concat
  while keeping the foundational implementation within the normal task-size guardrail.
- The independent task-0018M documentation review finalized the affected shape Javadocs, Tensor
  API, glossary, task evidence, capability baseline, master plan, and roadmap after the reused
  765-test model result, model Javadoc, runnable symbolic-extent example, public-surface, Markdown,
  scope, status, and whitespace checks passed. Compile API, Training API, related Tensor and
  operation contracts, architecture/ADRs/tests, conformance/integration, Java 26 Gradle
  configuration, dependencies, and other modules remain accurate unchanged because the task adds
  only model-owned shape values without Tensor-operation adoption, binding/evaluation, gradients,
  compiler/prepare/runtime/backend behavior, or execution.
- Task 0018M1 completed canonical before-then-after padding addition, one-step tiling
  multiplication, and encounter-order concat addition through `DimensionExpressions`. Its
  independent documentation review finalized helper and public Tensor Javadocs, Tensor API,
  glossary, task evidence, capability baseline, master plan, and roadmap after the reused
  766-test/88-suite model result, model Javadoc, Markdown, eleven-path, status, and whitespace
  checks passed. Explicit authorization expanded the original ten-path cap solely to correct
  public `Tensor` Javadocs that still stated the removed dynamic rejection rules; declarations and
  executable behavior remained unchanged. Compile API, Training API, shape foundations, semantic
  attributes, architecture/ADRs/tests, conformance/integration, Java 26 Gradle configuration,
  dependencies, and other modules remain accurate unchanged because this task changes only
  model-owned result-Shape formulas and their documentation.
- Semantic scalar attributes become data-type-safe. Raw binary64 attributes are not sufficient for
  exact INT64, BOOL, FLOAT32, or BFLOAT16 constants.
- Task 0018N uses one final `model.datatype.ScalarValue` backed by exact `DataType` plus primitive
  bits. It preserves FLOAT64/FLOAT32 signed zero and NaN payloads, raw BFLOAT16 patterns, signed
  INT32/INT64 including values above `2^53`, and canonical BOOL without boxing, a subtype
  hierarchy, registry, service, or general conversion API.
- Task 0018N atomically migrates `ScalarValueAttrs`, `ClampRangeAttrs`, and `PadAttrs`, adds exact
  typed Tensor overloads, and retains existing double overloads only as exact-FLOAT64
  conveniences. Receiver-aware Tensor helpers own exact parameter/input DataType matching;
  operation signatures keep their attribute-class and occurrence-cardinality role.
- Existing primitive TensorFactory scalar/full APIs and explicit `BFloat16Bits` conversion remain
  unchanged in 0018N. Completed task 0018S keeps those exact primitive eager-storage entries and does
  not add `ScalarValue` factory overloads because semantic attributes and storage carriers remain
  distinct responsibilities.
- Task 0018N completed with 57 focused tests and the final 770-test model suite passing in the
  implementation context. Independent documentation review finalized the seven affected
  Javadocs, Tensor and Compile API references, glossary, capability baseline, task evidence, and
  synchronized planning status without changing executable Java after that model run.
- Completed task 0018O normalizes public indexing primitives to GATHER, GATHER_ELEMENTS, GATHER_ND,
  SCATTER_ELEMENTS, SCATTER_ND, SELECT, and SLICE. It removes every `take` spelling, the current
  reduced-rank gather, fixed-add public scatter adjoints, and first-class UNSTACK semantics rather
  than retaining transitional aliases. Public unstack remains an ordered repeated-SELECT
  convenience whose results have independent one-output producers and provenance output index
  zero.
- Its implementation context passed the exact focused command and the 725-test/88-suite model
  suite. Independent documentation review finalized affected Javadocs, Tensor and Compile APIs,
  glossary, capability baseline, task evidence, master plan, and roadmap after model Javadoc,
  exact surface/absence, 469-link/anchor, fence/newline/whitespace, exact 29-path, status, and
  `git diff --check` validation passed.
- `fastExp` and `fastTanh` leave the public semantic baseline; approximation route selection
  belongs to backend prepare unless a future operation specifies portable accuracy.
- Completed task 0018P owns that atomic cleanup. Its final unary order is `ABS`, `NEG`, `RECIPROCAL`,
  `LOG`, `EXP`, `ERF`, `SQRT`, `FLOOR`, `CEIL`, `SIGN`, `RELU`, `SIGMOID`, `TANH`; the matching
  public methods use the same vocabulary, with no `INV`, fast variant, alias, or deprecated
  bridge. `EXP` and `TANH` remain portable mathematical requests without an algorithm, bitwise,
  approximation-bound, or backend-route promise. The typed scalar family remains unchanged by
  0018P. Completed task 0018T owns the complete seven-operation scalar arithmetic family and
  pairwise-extrema naming. Completed task 0018T1 separately owns floating-preserving `rsqrt`, `log1p`,
  and `expm1` plus fixed-BOOL floating classifications in a distinct semantic family.
- Its implementation context passed the focused 50-test contract set and the 725-test/88-suite
  model suite. Independent documentation review finalized the unary Javadocs, Tensor and Compile
  APIs, glossary, capability baseline, task evidence, master plan, and roadmap after model
  Javadoc, the runnable reciprocal example, generated-page and removed-vocabulary checks,
  Markdown structure, exact thirteen-path scope, synchronized status, and whitespace validation.
- Task 0018T1 extends that cleaned unary family to sixteen kinds with first-class `RSQRT`, `LOG1P`,
  and `EXPM1`, and adds separate `IS_FINITE`, `IS_NAN`, and `IS_INF` classifications whose public
  results are fixed non-differentiable BOOL metadata. Its implementation context passed the exact
  focused command and final model suite. Independent documentation review finalized all affected
  Javadocs and seven documentation/planning files after model Javadoc, a runnable transform-plus-
  classification metadata example, generated-page, 127-method/helper/alias, 493-link/139-anchor,
  exact eighteen-path, status, formatting, and whitespace checks passed.
- Completed task 0018Q removes heuristic masked-reduction axis mapping and simplifies
  `MaskedReductionAttrs` to one normalized axis. Ordinary right-aligned broadcasting of masks
  must produce exactly the input Shape, so callers make other axis intent visible with reshape or
  expansion. The two public overloads remain first-class two-input SUM/MEAN occurrences rather
  than misleading primitive compositions; all-false sum is zero and all-false mean is NaN.
- Its implementation context passed the exact focused contract command and all 720 model tests
  across 88 suites. Independent documentation review finalized the four affected Javadocs,
  Tensor and Compile APIs, glossary, capability baseline, task evidence, master plan, and roadmap
  after model Javadoc, the explicit-alignment Java 26 example, generated-page, Markdown,
  official-URL, exact thirteen-path, synchronized-status, terminology, and whitespace checks
  passed.
- Public `inv` becomes `reciprocal`; completed task 0018R removes public `foldAxis` while retaining
  `WindowTransformKind.FOLD_AXIS` and `FoldAxisAttrs` as public Java semantic contracts without a
  public Tensor receiver/construction method. Task 0023 selected follow-up 0023D, which later
  restored that public primitive and separately generalized 2D windows.
  It also selects
  normalized start/length/signed-step slice attributes, one explicit-step `sliceAxis` overload,
  and `flip(int... axes)` as one `SLICE` occurrence without negative-stride layout. Strict and
  cyclic prefix population moves to test/data utilities.
- Task 0018R completed that cleanup with normalized finite start/length/signed-step slice
  sequences, directional raw half-open normalization, positive-step-only resolved logical views,
  and explicit step-aware single-axis and one-producer flip conveniences. Public `foldAxis` and
  its helper path are absent; public `unfold`, `unfold2d`, and `fold2d` remain unchanged, while
  `FOLD_AXIS` and `FoldAxisAttrs` remain public Java semantic contracts but have no public Tensor
  receiver/construction method pending task 0023D's later completed public restoration.
  The implementation
  context passed 78 focused tests and all 715 model tests across 88 suites. Independent
  documentation review finalized seven Javadocs, Tensor/Compile APIs, glossary, capability/task/
  master/roadmap synchronization, the runnable Java 26 example, generated Javadoc, Markdown,
  official-link, exact eighteen-path, public-surface, status, terminology, and whitespace checks.
- FLOAT16 is important before accelerator mixed-precision support is claimed, but it is not a
  prerequisite for the linear-algebra model task.
- The initial data type baseline is `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, and `BOOL`.
- Static dimensions use non-negative `long` sizes; dynamic dimensions use explicit canonical symbols rather than negative sentinels.
- Scalar shape is rank zero, and zero-sized static dimensions are supported.
- Local broadcasting is right-aligned and conservative for symbolic dimensions; graph-wide symbolic constraints remain a compiler responsibility.
- Numeric layout descriptors are resolved only for fully static shapes and use non-negative `long` element strides and offsets.
- Layout geometry distinguishes dense, offset-dense, strided, and broadcast views; planning remains responsible for materialization decisions.
- Model contracts are organized by `datatype`, `shape`, `layout`, `tensor`, `storage`, `operation`, and `graph` responsibilities; the module root is not a flat type container.
- Java 26 is the project baseline. Stable Java 26 APIs require no preview opt-in; preview and incubator features remain disabled unless a focused owning-module task explicitly configures and validates them.
- The broad indexing/scatter frontier is split into select, axis-gather, gather-ND, axis-scatter,
  and scatter-ND semantic/expression pairs so each task owns one concept and one validation model.
- Scalar select accepts a negative public index only when the selected static extent can normalize
  it locally. A non-negative index on a dynamic selected extent remains representable with
  deferred bounds validation; a negative dynamic index is not locally representable.
- Completed tasks 0018C–0018D originally kept `GATHER`, `GATHER_AXIS`, and
  `TAKE_ALONG_AXIS` distinct. The capability reset supersedes that provisional naming while
  preserving the completed implementation history: task 0018O normalized the primitives to
  GATHER, GATHER_ELEMENTS, and GATHER_ND and removed axis `take` from the stable API.
- Completed tasks 0018D–0018D1 record how Tensor-index and primitive-array `take` were originally
  implemented. Completed task 0018O removes both public `take` overloads and the eager primitive-array
  helper without a compatibility alias; the final indexing surface accepts an index Tensor through
  `gather`, `gatherElements`, or `gatherNd`.
- Gather-ND stores only a normalized non-negative batch-dimension count; tuple depth remains the
  final indices Shape dimension and all input-dependent rank/Shape validation belongs to its
  Tensor-expression task.
- Gather-ND public construction requires INT32/INT64 indices, statically known positive tuple
  depth, structurally equal shared batch Dimensions, and unresolved result layout; index values
  and bounds remain outside model metadata construction.
- Completed tasks 0018G–0018H record the distinct provisional `SCATTER_ADD`,
  `SCATTER_AXIS_ADD`, and `SCATTER_ELEMENTS` contracts. Completed task 0018O retains only public
  `SCATTER_ELEMENTS` with explicit reduction attributes. Task 0023 confirms that it and public
  `SCATTER_ND` exactly serve Gather Elements and Gather-ND adjoints, while completed task 0023B
  owns final `SCATTER_ADD` with the missing rank-changing Gather-compatible updates Shape.
- Completed task 0023B appends `SCATTER_ADD` after the existing enum constant, pairs it with the
  existing normalized `IndexAxisAttrs`, and adds exactly one `scatterAdd(indices, updates, axis)`
  method. Its updates Shape is data prefix plus the complete indices Shape plus data suffix. The
  final name does not restore historical `SCATTER_AXIS_ADD` or the removed reduced-rank meaning;
  it is the exact additive functional counterpart of final `GATHER`.
- Scatter reduction is one reusable typed vocabulary in exact `NONE`, `ADD`, `MUL`, `MAX`, and
  `MIN` order. `NONE` represents unambiguous replacement and rejects duplicate targets in later
  value-aware validation rather than defining traversal-order-dependent overwrite behavior.
- The final axis-scatter public construction uses two `scatterElements` overloads and one
  field-free helper. It permits `NONE` for every current type and arithmetic reductions for
  floating/integral types, requires INT32/INT64 indices, preserves exact data Shape/type with
  unresolved layout, and never inspects index values.
- Scatter-ND uses one semantic kind plus immutable normalized batch count and the existing shared
  reduction vocabulary. Tuple depth remains the final indices Dimension; updates use the
  Gather-ND result Shape and the functional result keeps exact data Shape.
- Public Scatter-ND uses three overloads: `NONE` with zero batch count, explicit reduction with
  zero batch count, and fully explicit reduction plus batch count. One shared helper owns exact
  type/rank/batch/tuple/updates-Shape validation without reading values.
- Task 0018J completed exactly those three overloads and one final package-private, field-free,
  eleven-method helper. Valid requests preserve exact data Shape/type, combine data/update
  eligibility, leave layout unresolved, and record fresh exact `[data, indices, updates]`
  provenance. Independent documentation review finalized the two temporal semantic Javadocs,
  Tensor API, Compile API, glossary, task evidence, master plan, and roadmap after focused
  10-test, Tensor API 14-test, all 735-model-test/85-suite, model-Javadoc, root-test,
  javap/reflection/import/source/generated-page, Java 26 example, 442-link/134-anchor Markdown,
  fence/whitespace/newline,
  exact twelve-path, synchronized-status, semantic-bytecode-equivalence, and no-task-0019-spec
  checks passed. Training API, capabilities, related contracts, architecture/ADRs/tests,
  conformance/integration, Java 26 Gradle/dependencies, other modules, and later tasks remain
  accurate unchanged for their recorded ownership reasons.
- Task 0018I completed exactly `ScatterNdKind.SCATTER_ND` and
  `ScatterNdAttrs(batchDimensions, reduction)` while reusing unchanged `ScatterReduction`.
  Independent documentation review retained both complete production Javadocs and finalized
  Tensor API, glossary, task evidence, master plan, and roadmap after focused 9-test, all
  725-model-test/84-suite, model-Javadoc, root-test, javap/reflection/import/source/generated-page,
  425-link/131-anchor, fence/whitespace/newline, exact eight-path, synchronized-status, and
  no-0018J-spec checks passed. Compile API, Training API, capabilities, related contracts,
  architecture/ADRs/tests, conformance/integration, Java 26 Gradle/dependencies, other modules,
  and later tasks remain accurate unchanged for the recorded reasons.
- Task 0018G completed exactly `AxisScatterKind`, `ScatterReduction`, and
  `ScatterElementsAttrs`, plus unchanged reuse of `IndexAxisAttrs` for fixed-add kinds. Ordered
  `[data, indices, updates]`, functional data-shaped results, all three Shape relationships, every
  reduction meaning, and value-aware `NONE` duplicate rejection are semantic contracts only;
  completed task 0018H owns public type/Shape/axis validation and Tensor construction.
- Task 0018H completed exactly four public Tensor methods and one field-free eleven-method helper.
  Fixed-add paths require matching floating data/updates; scatter-elements permits `NONE` for every
  current type and arithmetic reductions for floating/integral types. Results retain exact data
  Shape/type, data/update eligibility OR, unresolved layout, and exact ordered provenance without
  value access, writes, reductions, mutation, gradients, compiler behavior, or execution.
- Independent task-0018G documentation review retained all three new production Javadocs,
  clarified only `IndexAxisAttrs` ownership wording, and finalized Tensor API, glossary, task
  evidence, master plan, and roadmap. Focused 12-test, all 706-model-test/82-suite, model-Javadoc,
  root-test, javap/reflection/import/source/generated-page, 392-link/100-anchor,
  fence/whitespace/newline, exact ten-path, synchronized-status, `IndexAxisAttrs` bytecode-
  equivalence, and no-0018H-spec checks passed. Compile API, Training API, capabilities, related
  contracts, architecture/ADRs/tests, conformance/integration, Java 26 Gradle/dependencies, other
  modules, and later tasks remain accurate unchanged for the recorded reasons.
- Task 0018F completed exactly two public `Tensor.gatherNd` methods and one field-free eight-method
  helper. It validates index metadata, ranks, normalized batch count, structurally equal shared
  batch prefixes, and static positive tuple depth before deriving the exact indices-prefix plus
  data-suffix Shape. Results preserve data type/eligibility, remain unresolved and storage-free,
  and record fresh exact `GATHER_ND`/attributes/`[data, indices]` provenance without value access.
- Independent task-0018F documentation review finalized Tensor/helper and the two authorized
  semantic temporal Javadocs, Tensor and Compile API references, glossary, task evidence, master
  plan, and roadmap. Focused 10-test and 14-test suites, all 694 model tests across 81 suites,
  model Javadoc, root tests, executable example, bytecode/reflection/import/source/generated-page,
  417-link/121-anchor, fence/whitespace/newline, exact twelve-path, synchronized-status,
  semantic-bytecode-equivalence, and no-0018G-spec checks passed. Training API, capabilities,
  architecture/ADRs/tests, conformance/integration, Java 26 Gradle/dependencies, other modules, and
  later tasks remain accurate unchanged for the recorded reasons.
- Task 0018E completed exactly `GatherNdKind.GATHER_ND` and
  `GatherNdAttrs(batchDimensions)`. The semantic contract defines ordered `[data, indices]`, final
  indices-Dimension tuple depth, shared batch prefix, indexed data axes, untouched suffix, and the
  conceptual result formula without storing occurrence-specific Shape facts or adding public
  Tensor, gradient, compiler, backend, or execution behavior.
- Independent task-0018E documentation review retained both complete production Javadocs and
  finalized Tensor API, glossary, task evidence, master plan, and roadmap after focused 9-test,
  all 684-model-test/80-suite, model-Javadoc, root-test, javap/reflection/import/source/generated-
  page, 403-link/119-anchor, fence/whitespace/newline, exact eight-path, synchronized-status, and
  no-0018F-spec checks passed. Compile API, Training API, capabilities, related contracts,
  architecture/ADRs/tests, conformance/integration, Java 26 Gradle configuration, dependencies,
  other modules, and later tasks remain accurate unchanged for the recorded reasons.
- Task 0018A completed exactly `SelectKind.SELECT`, normalized non-negative `SelectAttrs(axis,
  index)`, and focused structural/validation/composition coverage without public Tensor, Shape,
  layout, provenance, compiler, backend, or execution behavior.
- Its independent documentation review retained both complete production Javadocs and finalized
  Tensor API, glossary, task evidence, master plan, and roadmap after focused 9-test, all 638 model
  tests across 75 suites, model-Javadoc, root-test, javap/reflection/import/generated-page,
  link/anchor/fence/whitespace, exact eight-path, synchronized-status, and no-0018B-spec checks
  passed. Compile API, Training API, capabilities, related model contracts,
  architecture/ADRs/tests, conformance/integration, Java 26 Gradle, dependencies, and other modules
  remain accurate unchanged because the task adds only model-owned scalar-select semantics.
- Task 0018B completed exactly public `Tensor.select(int, long)` plus one field-free five-method
  package-private helper. The expression normalizes one raw axis, normalizes and bounds-checks a
  coordinate for a static selected extent, accepts a non-negative dynamic coordinate with its
  upper bound deferred, removes the selected axis, and preserves exact unaffected Dimension
  references.
- Resolved input geometry with a non-empty result receives one new checked logical view descriptor
  with selected-stride removal and offset advancement; unresolved input and empty results remain
  unresolved. Every result preserves exact type/eligibility, records `SELECT`/normalized
  `SelectAttrs`/`[input]`, and is fresh, unlabeled, and storage-free without a physical-alias,
  value, gradient, compiler, backend, or execution promise.
- Independent task-0018B documentation review finalized Tensor/helper Javadocs, Tensor API,
  Compile API, glossary, task evidence, master plan, and roadmap. Training API, capabilities,
  semantic/foundational/adjacent contracts, architecture/ADRs/tests, conformance/integration,
  Java 26 Gradle configuration, dependencies, and other modules remain accurate unchanged because
  this task adds only model-owned scalar-select expression metadata.
- Task 0018C completed exactly `AxisGatherKind.GATHER`, `GATHER_AXIS`, and `TAKE_ALONG_AXIS` plus
  shared normalized non-negative `IndexAxisAttrs(axis)`. Ordered `[data, indices]` roles, three
  distinct result-Shape relationships, and public tensor-index `take` as an exact `GATHER_AXIS` alias
  are semantic documentation rather than stored inputs, validation, or result construction.
- Independent task-0018C documentation review finalized both production Javadocs, Tensor API,
  glossary, task evidence, master plan, and roadmap after focused 9-test, all 657-model-test/
  77-suite, model-Javadoc, root-test, javap/reflection/source/import/generated-page,
  363-link/91-anchor, fence/whitespace/newline, exact eight-path, synchronized-status, and
  no-0018D-spec checks passed. Compile API, Training API, capabilities, related contracts,
  architecture/ADRs/tests, conformance/integration, Java 26 Gradle configuration, dependencies,
  and other modules remain accurate unchanged because the task adds only model-owned semantic
  vocabulary without Tensor input validation, result metadata, provenance, gradients, compiler,
  backend, or execution behavior.
- Task 0018D completed exactly four public Tensor-index methods and one field-free nine-method
  helper. It accepts only INT32/INT64 indices, normalizes one data axis, implements the distinct
  reduced, inserted, and aligned Shape rules, leaves every result layout unresolved, and creates
  fresh exact `[data, indices]` provenance without value, bounds, storage, gradient, compiler,
  backend, or execution behavior. Tensor-index `take` delegates exactly to GATHER_AXIS.
- Independent task-0018D documentation review finalized Tensor/helper and two explicitly
  authorized semantic Javadoc corrections, Tensor API, Compile API, glossary, task evidence,
  master plan, and roadmap after focused tests, all model tests, model Javadoc, root tests,
  bytecode/reflection/import/source/generated-page review, an executable example, Markdown and
  exact twelve-path checks passed. Training API, capabilities, architecture/ADRs/tests,
  conformance/integration, Java 26 Gradle configuration, dependencies, related foundational
  behavior, and other modules remain accurate unchanged for the recorded reasons.
- Task 0018D1 completed exactly one public `Tensor.take(int, int[])` overload and one field-free
  two-method helper. It validates a non-null, non-empty primitive source, clones it once, creates
  one dense rank-one non-differentiable INT32 leaf Tensor through existing flat import, and
  delegates once to completed tensor-index take. Every signed value is copied unchanged without
  bounds inspection; final provenance is exact `[data, generatedIndices]` with GATHER_AXIS
  semantics.
- Independent task-0018D1 documentation review finalized Tensor/helper Javadocs, Tensor and
  Compile API references, glossary, task evidence, master plan, and roadmap after focused/model/
  root tests, model Javadoc, bytecode/reflection/import/source/generated-page review, an executable
  Java 26 example, Markdown and exact ten-path checks passed. Training API, capabilities, related
  foundational and axis-gather contracts, architecture/ADRs/tests, conformance/integration,
  Gradle configuration, dependencies, and other modules remain accurate unchanged because the
  task adds only model-owned primitive input adaptation and existing expression composition.
- Typed identifiers live with their domains. The current plan includes `TensorId`, `NodeId`, and `ValueId`; `OperationId` is deferred unless a focused task demonstrates identity distinct from `NodeId`.
- Host storage contracts precede the public `Tensor`, and `Tensor` reuses `TensorDescriptor` rather than duplicating descriptor validation.
- `HostTensorStorage` is a sealed model boundary with one final identity-based
  `MemorySegmentStorage` implementation. The wrapper borrows an exact-size live segment, exposes
  raw segment/read-only/liveness facts, uses checked `long` capacity sizing, and owns no arena,
  allocation, close behavior, typed access, alignment, byte order, tensor geometry, or runtime
  residency policy.
- Operation-family table order coordinates delivery; dependencies record only real contract prerequisites rather than the preceding row.
- Every intentionally selected public or compiler-only operation must remain representable without
  backend knowledge. Legacy-only capabilities may instead be redesigned, demoted, or excluded.
- Model capability selection and end-to-end executable completion are tracked separately.
- Fusion is not a model-level mathematical operation capability.
- The compiled graph container stores ordered values, topological nodes, explicit input/output
  boundaries, and an exact node-to-forward/backward-phase mapping. It stores no derived indexes.
- Publication binding remains a standalone `TensorId`-to-`ValueId` model DTO for a later
  compiler-owned publication plan; it is not part of `CompiledGraphModel`.
- The current graph-phase vocabulary is exactly forward and backward. Optimizer-update graph work
  remains a future architecture change, not a task-0009 phase.
- The task-0011 Tensor skeleton is one final public identity object with package-private
  construction. It retains one stable `TensorId`, immutable `TensorDescriptor`, and normalized
  optional label; task 0012 owns the public factory and ID allocation policy.
- Tensor's only mutable state is a synchronized optional borrowed `HostTensorStorage`
  association. Matching data type is always required; resolved layouts require capacity at least
  their referenced element span, while unresolved layouts do not invent physical geometry.
- Tensor accepts read-only storage, rejects storage already dead at attachment, continues to expose
  storage that dies later, owns no arena, retains object identity equality/hashing, and stores no
  graph-local ID, gradient/trainable/publication, runtime, or backend state. Task 0013 adds only
  final optional provenance metadata and does not change host storage's sole-mutation role.
- Task 0012 is a non-instantiable static `TensorFactory` with exactly descriptor-only and
  descriptor/optional-label/optional-storage creation methods. It delegates semantic label and
  storage validation to the package-private Tensor constructor and performs no storage allocation,
  import, population, descriptor construction, layout resolution, or provenance work.
- Factory-assigned tensor IDs are unique across factory calls in one JVM, including concurrent
  calls. A hidden `AtomicLong`/`AtomicBoolean` allocator issues non-negative candidates from zero
  through `Long.MAX_VALUE`, permits the final value once, never wraps or reuses a value, and then
  fails permanently. Numeric order and gaplessness are not public caller contracts.
- Factory argument-container null failures occur before ID allocation. Tensor label/storage
  failures occur after allocation and consume the candidate so the factory does not duplicate
  canonical validation or attempt unsafe concurrent rollback.
- The broad factory baseline is split into completed task 0012A for JVM-managed heap allocation,
  completed task 0012B for flat typed import, completed task 0012C for nested typed import,
  completed task 0012D for constant tensors, completed task 0012E for deterministic range/prefix
  population, completed task 0012F for normal random tensors, completed task 0012G for uniform
  random tensors, completed task 0012H for integral tensors, and completed task 0012I for
  Bernoulli tensors. These rows remain before completed provenance task 0013 and completed
  full-value/identity task 0013A. The completed model foundation checkpoint selected task 0014A
  as the next implementation frontier.
- Task 0012A adds only JVM-managed heap allocation to `TensorFactory`. It allocates one typed
  primitive array whose length is the resolved layout's referenced element span, wraps the
  `MemorySegment.ofArray(...)` result in the existing `MemorySegmentStorage`, and delegates to the
  existing `create(...)` path.
- Java 26 heap segments use an automatic scope that keeps the primitive-array heap base reachable
  and is always accessible from any thread. Task 0012A therefore adds no owning wrapper, arena,
  close behavior, external owner, or storage-contract change.
- Task 0012A requires resolved layout, rejects span above `Integer.MAX_VALUE`, and keeps allocation
  separate from the imports in completed tasks 0012B and 0012C, constant creation in completed task
  0012D, deterministic population in completed task 0012E, and random population in completed task
  0012F–0012I.
- Task 0012B adds six typed flat-array overloads for `double[]`, `float[]`, raw BFLOAT16 `short[]`,
  `int[]`, `long[]`, and BOOL `byte[]`. It accepts only resolved dense-contiguous layout, validates
  source length against logical element count, copies all input data, and normalizes BOOL bytes to
  canonical zero or one without retaining caller arrays.
- Offset, strided, and broadcast layouts are rejected by flat import because mapping independent
  row-major source values into aliased or sparse physical geometry is a distinct scatter/view
  policy. Task 0012B reuses task-0012A allocation and the existing factory identity path.
- Task 0012C accepts exactly rank-two-or-greater Java arrays whose ultimate component is one of the
  six primitive host carriers. One `Object` method is used because arbitrary array rank has no
  finite overload family; runtime class metadata must still prove declared rank and exact carrier.
- Nested import validates the full reachable structure for rectangular lengths and non-null
  subarrays, rejects empty non-final axes whose trailing extents are unobservable, accepts an empty
  final leaf axis, and flattens row-major into a fresh matching carrier. It synthesizes only a
  fully static dense-contiguous descriptor and delegates final creation to task 0012B.
- Task 0012D defines exact primitive-carrier rank-zero scalars, including an explicitly named
  BFLOAT16 conversion, plus all-data-type zeros and ones over fully static shapes. Zeros reuse
  default-zero allocation; scalars and ones reuse typed flat import.
- Constant `*Like` methods copy only template shape and data type. They take explicit label and
  gradient intent and create new dense-contiguous descriptor, storage, and identity without
  observing or preserving template layout, label, storage, liveness, or ID.
- Task 0012E keeps deterministic population type-exact: `int` and `long` ranges produce only
  `INT32` and `INT64`, while strict and cyclic prefixes use six primitive-carrier overloads with no
  implicit conversion. All results synthesize canonical dense layout and reuse flat import.
- Integer ranges are eager non-differentiable leaf data with inclusive start, exclusive end,
  positive or negative non-zero step, exact overflow-safe count, and no automatic label. Prefixes
  require fully static shape, copy source values, preserve raw BFLOAT16 bits, and reuse downstream
  BOOL normalization. Empty cyclic input is valid only for an empty result.
- Task 0012F uses one transient caller-owned `RandomGenerator` and stores no random service, source,
  seed, or algorithm. It consumes exactly one `nextGaussian()` per logical element, applies an
  explicit binary64 normal transformation, converts only to FLOAT64/FLOAT32/BFLOAT16, and delegates
  completed carriers to flat import.
- Random reproducibility is bounded to equivalent generator implementation/state and identical
  arguments without interfering use. No cross-algorithm/provider/Java-version promise, default
  source, synchronization, or seed-only convenience is introduced.
- User-approved random initialization expansion remained sequential: completed task 0012G adds
  floating uniform sampling, completed task 0012H adds typed bounded integral sampling, and
  completed task 0012I adds BOOL Bernoulli sampling. Each reuses the caller-owned source policy
  without changing task 0012F.
- Task 0012H uses two `randomInt` overloads: int bounds infer INT32 and long bounds infer INT64.
  Both use strict half-open bounds, false gradient intent, and the matching unbiased JDK bounded
  generator method without project-owned modulo arithmetic.
- Task 0012I uses one BOOL-only `randomBernoulli` method with finite probability in `[0,1]`. It
  consumes one unbounded binary64 draw per element even at probability endpoints and stores the
  strict `draw < probability` result as canonical zero/one bytes before BOOL flat import.
- Random factory methods and package-private helpers remain in `model.tensor`. A `randoms` package
  would break useful package-private collaboration or require a public implementation surface and
  is not justified without independent public random-domain types.
- Task 0013 adds one immutable `TensorProvenance` value containing a backend-independent
  `Operation` and an ordered immutable snapshot of input Tensor identities. Tensor retains it as
  optional final metadata; it receives no graph-local identity and does not become IR.
- Existing public factory paths remain provenance-free leaves. One package-private derived-
  construction seam reuses the existing TensorFactory allocator, attaches exact provenance, and
  creates no storage, inference, graph, compiler, runtime, or backend state.
- Task 0013A is model-owned eager tensor creation, not a training initializer, graph operation, or
  runtime/backend capability. `full` and `identityMatrix` are the canonical factory names, and
  `eye` is the exact convenience alias for `identityMatrix` semantics.
- Task 0013A added six primitive-carrier `full` methods, including explicitly converted
  `fullBFloat16`, plus one all-data-type rectangular `identityMatrix`. `eye` delegates only to the
  canonical method. All results are dense provenance-free leaves created through flat import.
- The post-foundation checkpoint selected continued model work instead of an immediate
  cross-module vertical slice because no production concrete OperationKind existed yet for
  compiler capture or backend capability work.
- The broad former task 0014 is decomposed into semantic-vocabulary and public-expression pairs:
  binary arithmetic 0014A–0014B, unary/activation 0014C–0014D, and scalar/clamp 0014E–0014F.
- Task 0014A implements one parameterless `BinaryArithmeticKind` enum in
  `model.operation.elementwise.binary` with exact constants ADD, SUB, MUL, DIV, MIN, MAX, and POW.
  Broadcast geometry, dtype rules, provenance, and Tensor methods remain in task 0014B.
- Task 0014B adds seven fluent Tensor methods through one package-private helper. It promotes only
  floating data types, delegates right-aligned shape algebra to `ShapeBroadcast`, leaves result
  layout unresolved, propagates gradient eligibility as input OR, and records exact ordered
  provenance without storage access, execution, canonicalization, or graph capture.
- The post-0014B vertical-slice reassessment keeps `modules/model` as the active frontier. Trace,
  backend-contract, config, planning, and compiler still contain only placeholder production types
  and broad master plans, so crossing those boundaries now would hide several foundational tasks
  rather than form one bounded implementation session. This changes order only, not dependencies.
- Task 0014C uses one parameterless `UnaryElementwiseKind` enum in
  `model.operation.elementwise.unary` for ABS, NEG, INV, LOG, EXP, ERF, SQRT, FLOOR, CEIL, SIGN,
  RELU, SIGMOID, TANH, FAST_EXP, and FAST_TANH. Task 0014D implements their Tensor expression
  behavior.
- Task 0014D adds the matching fifteen zero-argument Tensor methods through one package-private
  helper. It accepts only floating inputs, retains exact data type and Shape, leaves result layout
  unresolved, propagates gradient eligibility unchanged, and records exact one-input provenance
  without value/storage access, domain checks, canonicalization, gradient rules, or execution.
- Task 0014E defines `ScalarElementwiseKind` for MUL, POW, CLAMP, CLAMP_MIN, and CLAMP_MAX plus
  immutable `ScalarValueAttrs` and `ClampRangeAttrs`. Parameters remain exact binary64 semantic
  values without Tensor/DataType coupling, alternate-precision caches, execution, or family
  registries. Task 0014F supplies their public Tensor expression behavior.
- Task 0014F adds five matching Tensor overloads through one package-private helper. It accepts only
  floating inputs, preserves exact caller binary64 attributes and input type/Shape, leaves layout
  unresolved, propagates gradient eligibility unchanged, records exact one-input provenance, and
  represents range clamp as one first-class operation without conversion or canonicalization.
- The broad former task 0015 is decomposed into four semantic/expression pairs: binary comparisons
  0015A–0015B, BOOL logic 0015C–0015D, ternary `where` 0015E–0015F, and cast 0015G–0015H.
- Task 0015A defines parameterless `BinaryComparisonKind` in
  `model.operation.elementwise.comparison` for GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN,
  LESS_OR_EQUAL, EQUAL, and NOT_EQUAL. Tensor inputs, broadcasting, and BOOL results remain in
  task 0015B.
- Task 0015B implements six fluent floating-only Tensor methods through one package-private
  helper. It validates the common floating comparison domain, delegates right-aligned shape
  algebra to `ShapeBroadcast`, creates an unresolved non-differentiable BOOL descriptor, and
  records exact ordered provenance without storage access, numerical comparison, graph capture,
  or execution.
- Task 0015C implements one parameterless `BooleanLogicalKind` enum in
  `model.operation.elementwise.logical` with exact constants AND, OR, and NOT. Binary/unary input
  roles are documented family context, not arity metadata; BOOL validation, broadcasting, result
  descriptors, provenance, and Tensor methods remain in task 0015D.
- Task 0015D implements `logicalAnd`, `logicalOr`, and `logicalNot` through one package-private
  helper with explicit binary/unary entries. It enforces exact BOOL inputs and family arity,
  broadcasts only binary inputs, preserves unary shape, creates unresolved non-differentiable BOOL
  descriptors, and records exact provenance without storage access or execution.
- The independent task-0015D documentation review finalized one helper Javadoc clarification,
  Tensor API, Compile API, glossary, task evidence, master plan, and roadmap. Training API,
  capabilities, architecture/ADRs/tests, conformance and integration tests, Java 26 build
  configuration, foundational contracts, and existing expression families remain accurate
  unchanged because this task adds only model-owned logical expression construction without
  compiler, training, dependency, truth-value, or executable behavior.
- Task 0015E implements one parameterless `WhereSelectionKind` enum in
  `model.operation.elementwise.selection` with the sole `WHERE` identity. Exact ordered condition,
  true-branch, and false-branch roles are documented ternary family context, not stored arity or
  input state. Task 0015F now owns public BOOL/floating validation, branch promotion, three-way
  broadcasting, result construction, and provenance; gradients and executable behavior remain in
  later owning layers.
- The independent task-0015E documentation review found the enum and constant Javadocs complete,
  then finalized Tensor API, glossary, task evidence, master plan, and roadmap. Compile API,
  Training API, capabilities, architecture/ADRs/tests, conformance and integration tests, Java 26
  build configuration, foundational operation contracts, existing kind/expression families, and
  Tensor remain accurate unchanged because the task adds only model-owned conditional-selection
  semantic vocabulary without public expressions, dependencies, inference, provenance, training,
  indexing behavior, or execution.
- Task 0015F adds static public `Tensor.where(condition, ifTrue, ifFalse)` through one
  package-private single-entry helper. It requires an exact BOOL condition, promotes floating
  branches through the shared contract, composes two pairwise broadcasts in branch-first order,
  creates an unresolved descriptor with branch-only gradient eligibility OR, and records exact
  `[condition, ifTrue, ifFalse]` provenance without value access, gradient rules, graph capture, or
  execution.
- The independent task-0015F documentation review found the Tensor and helper Javadocs complete,
  then finalized Tensor API, Compile API, glossary, task evidence, master plan, and roadmap.
  Training API, capabilities, architecture/ADRs/tests, conformance and integration tests, Java 26
  build configuration, foundational contracts, and existing expression families remain accurate
  unchanged because the task composes existing model-owned validation and provenance without
  value selection, gradient routing, graph capture, ONNX mapping, backend behavior, dependencies,
  or execution.
- Task 0015G adds `CastKind.CAST` and immutable `CastAttrs(targetDataType)` in
  `model.operation.elementwise.cast`. The target is required and may be any current DataType;
  source type, same-type behavior, result descriptors, gradient eligibility, provenance, and
  public `Tensor.cast` were assigned to task 0015H; conversion policy, gradient rules, and
  execution remain in later owning layers.
- Task 0015H implements fluent `Tensor.cast(DataType)` through one package-private helper. Every
  current source/target pair creates a fresh explicit storage-free expression, including same-type
  requests; the result retains the exact Shape, leaves layout unresolved, and retains gradient
  eligibility only for an already-eligible floating-to-floating cast. Legacy same-type input
  return is deliberately replaced by compiler-owned redundant-cast canonicalization.
- The independent task-0015H documentation review found the Tensor and helper Javadocs complete,
  then finalized the Tensor API, Compile API current-expression inventory, glossary, task evidence,
  master plan, and roadmap. Training API, capabilities, architecture/ADRs/tests, conformance and
  integration tests, Java 26 build configuration, foundational contracts, and existing expression
  families remain accurate unchanged because this task adds only model-owned cast expression
  construction without numerical conversion, gradient rules, compiler canonicalization, backend
  behavior, dependencies, or execution.
- The broad reduction-and-scan frontier is decomposed into aggregate semantics 0016A, focused
  ordinary aggregate Tensor expression tasks 0016B–0016E, masked semantic task 0016F and expression
  task 0016F1, cumulative-sum semantics/expressions 0016G–0016H, and softmax semantics/expressions
  0016I–0016J. This preserves the semantic/expression split and prevents public API, shape
  inference, masked reductions, scan options, and normalization from becoming one oversized task.
- Task 0016G introduced the cohesive `model.operation.scan` package with exactly the
  `CUM_SUM` semantic identity and immutable normalized-axis, exclusive, and reverse attributes.
  Its four scan modes are documented as semantic meaning only; Tensor construction, input type
  validation, Shape retention, provenance, numerical behavior, and execution remain outside this
  task and are not inferred by the generic Operation contract.
- Task 0016H adds exactly two public `Tensor.cumSum` overloads through one dedicated
  package-private helper. It accepts all five current numeric types, rejects BOOL, normalizes one
  axis, retains exact Shape/type/gradient-eligibility metadata in an unresolved descriptor, and
  records one-input CUM_SUM provenance without inspecting storage, accumulating values, defining
  a gradient rule, capturing a graph, or executing backend work.
- The independent task-0016H documentation review finalized the Tensor and helper Javadocs, Tensor
  API, Compile API current-expression inventory, glossary, task evidence, master plan, and roadmap.
  Training API, capabilities, foundational and cumulative-sum semantic contracts, other expression
  families, focused architecture/ADRs/tests, conformance and integration tests, Gradle
  configuration, and other modules remain accurate unchanged because this task adds only
  model-owned cumulative-sum expression metadata without value accumulation, gradient rules,
  compiler capture, dependencies, backend behavior, or execution.
- Task 0016I completed the cohesive `model.operation.normalization` package with exact SOFTMAX and
  LOG_SOFTMAX identities plus one immutable normalized-axis attributes record. Its Javadocs,
  Tensor API, and glossary explain probability and log-probability slice semantics and their
  mathematical relationship while leaving Tensor construction, floating eligibility, Shape
  retention, provenance, numerical algorithms, gradients, compiler decomposition, backend
  behavior, and execution to later owners. The independent documentation pass found the submitted
  Javadocs complete unchanged and validated focused/model/root tests, generated Javadoc, bytecode,
  imports, Markdown, exact eight-path scope, and synchronized status.
- Task 0016J adds exactly `Tensor.softmax(axis)` and `Tensor.logSoftmax(axis)` through one
  dedicated package-private helper. It accepts floating inputs, normalizes one axis, retains exact
  Shape/type/gradient-eligibility metadata in an unresolved descriptor, and records the requested
  first-class normalization kind with one-input provenance without reading values, selecting an
  algorithm, defining gradients, decomposing the operation, or executing work.
- The independent task-0016J documentation review found the Tensor and helper Javadocs complete,
  then finalized Tensor API, Compile API, glossary, task evidence, master plan, and roadmap.
  Training API, capabilities, softmax semantic/foundational and other expression contracts,
  architecture/ADRs/tests, backend conformance and integration tests, Gradle configuration, and
  other modules remain accurate unchanged because this task adds only model-owned softmax
  expression metadata without numerical evaluation, gradient rules, compiler capture or
  decomposition, dependencies, backend behavior, or execution.
- The broad layout/view frontier is decomposed into tasks 0017A–0017N. Each semantic group is
  separated from public Tensor expression construction so immutable vocabulary, local shape/layout
  rules, provenance, and future materialization boundaries do not become one oversized task.
  Contiguous semantics/expressions are 0017A–0017B; reshape/expand are 0017C–0017D plus 0017D1;
  axis transforms are 0017E–0017F plus 0017F1; slicing is 0017G–0017H; pad/tile are 0017I–0017J;
  concat/stack/unstack are 0017K–0017L; and unfold/fold are 0017M–0017N.
- Task 0017K is complete with first-class CONCAT, STACK, and individually indexed UNSTACK-output
  semantics. Concat/stack share one normalized axis value; each unstack output carries normalized
  axis plus outputIndex so current one-provenance-per-Tensor metadata remains unambiguous without
  producer grouping or graph-contract changes. Public APIs and result construction are implemented
  by completed task 0017L.
- Task 0017L is complete with ordered varargs CONCAT/STACK and immutable-list UNSTACK APIs. It applies
  exact type/Shape rules, limited provable dynamic concat, insertion/removal Shape derivation,
  eligibility OR/preservation, unresolved layouts, and individually indexed unstack provenance
  without grouped producers, values, storage, gradients, or cross-layer behavior.
- Task 0017M is complete with distinct UNFOLD_AXIS, FOLD_AXIS, UNFOLD2D, and FOLD2D semantic
  identities; normalized long-valued axis/window geometry; and explicit fold target extents.
  FOLD_AXIS was the semantic basis for the historical public task-0017N Tensor expression; task
  0018R later removed that public method while retaining the public Java semantic contracts.
  [Task 0023](tasks/0023-adjoint-expressibility-audit.md) selected task 0023D to restore the
  generally useful public overlap-add primitive before compiler use; that task is now Complete.
  Task 0017M defines NCHW im2col/col2im
  meaning and overlap summation without Tensor construction, Shape arithmetic, provenance,
  gradients, materialization, compiler behavior, backend behavior, or execution.
- Task 0017N is complete with the four public storage-free Tensor expressions that existed at its
  historical completion, checked long-valued local Shape derivation, conservative dynamic-
  dimension preservation, unresolved result layouts, and exact one-input provenance. Completed
  task 0018R later removed public `foldAxis`; completed task 0023D has now restored it. Neither
  historical task implements scatter-add execution or gradient behavior.
- The independent task-0017N documentation review finalized Tensor/helper and temporal semantic
  Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap. Focused
  16-test, all 629 model-test across 74 suites, model-Javadoc, root-test, executable-example,
  javap/reflection/bytecode/import/source/generated-page, 370-link/108-anchor, fence/whitespace,
  exact fifteen-path, synchronized-status, and no-0018-spec checks passed. Training API,
  capabilities, related model declarations, architecture/ADRs/tests, conformance/integration,
  Java 26 Gradle/dependencies, other modules, and task 0023 remain accurate unchanged.
- The independent task-0017M documentation review finalized all five new Javadocs, Tensor API,
  glossary, task evidence, master plan, and roadmap. Focused 12-test, all 613 model-test across 73
  suites, generated-Javadoc, root-test, javap/reflection/import/source, link/anchor/fence/
  whitespace, exact twelve-path, synchronized-status, and no-0017N-spec checks passed. Compile
  API, Training API, related model contracts, architecture/ADRs/tests, conformance/integration,
  Java 26 Gradle configuration, dependencies, and other modules remain accurate unchanged because
  this task adds only backend-neutral semantic values without Tensor construction, gradients, or
  cross-layer behavior.
- The independent task-0017L documentation review finalized Tensor/helper Javadocs, Tensor API,
  Compile API, glossary, task evidence, master plan, and roadmap. Focused 27-test, all 601 model-
  test, generated-Javadoc, root-test, executable-example, bytecode/reflection/import/source,
  link/anchor/fence/whitespace, exact ten-path, synchronized-status, and no-0017M-spec checks
  passed. Training API, capabilities, semantic/provenance/graph/foundational contracts,
  architecture/ADRs/tests, conformance/integration, Java 26 Gradle configuration, dependencies,
  and other modules remain accurate unchanged because this task adds only model-owned composition
  expression metadata without grouped producers, values, gradients, or cross-layer behavior.
- The independent task-0017K documentation review retained all three complete production
  Javadocs, then finalized Tensor API, glossary, task evidence, master plan, and roadmap. The
  focused 11-test suite, all 588 model tests, generated Javadoc, root tests, javap/reflection/
  import/source checks, link/anchor/fence/whitespace checks, exact nine-path scope, synchronized
  statuses, and no-0017L-spec check passed. Compile API, Training API, capabilities, related model
  contracts, architecture/ADRs/tests, conformance/integration, Java 26 Gradle configuration,
  dependencies, and other modules remain accurate unchanged because this task adds only
  model-owned composition semantic values without Tensor construction, provenance attachment,
  graph grouping, gradients, compiler/backend behavior, or execution.
- Task 0017G is complete with one `SLICE` identity and one immutable normalized `SliceAttrs` value.
  Four equal-size parallel lists carry long half-open bounds and positive steps plus distinct int
  axes. Single-axis slicing remains one entry with step one rather than a second operation kind;
  Tensor request normalization, Shape/layout derivation, and provenance are implemented by task
  0017H.
- Task 0017H is complete with explicit long-bound/step arrays and one-axis convenience. It normalizes
  and clamps against selected static dimensions, permits zero-extent results, preserves unaffected
  Dimension references, derives non-empty resolved view geometry, and records fresh SLICE
  provenance without values, storage aliases, gradients, or cross-layer behavior.
- Task 0017I is complete with separate `PAD` and `TILE` semantic identities. Immutable `PadAttrs`
  carries ordered non-negative long before/after widths plus an uninterpreted double constant;
  `TileAttrs` carries ordered positive long complete-pattern repeat counts. Empty lists are scalar
  identity parameters, extreme longs and every double are retained structurally, and rank, Shape
  arithmetic, constant conversion, Tensor construction, provenance, gradients, and execution are
  separated into task 0017J or later owners.
- Task 0017J is complete with exact long-array pad and varargs tile Tensor methods. Its initial
  implementation derived checked static Shapes and preserved only identity-transformed dynamic
  dimensions; completed task 0018M1 later replaced those conservative derivation rules with
  canonical symbolic formulas. Result layouts remain unresolved, every current DataType and raw
  double constant remains accepted, and fresh typed provenance records no values, storage,
  gradients, compiler behavior, or execution.
- The independent task-0017J documentation review finalized Tensor/helper Javadocs, Tensor API,
  Compile API, glossary, task evidence, master plan, and roadmap. Focused/model/root tests, generated
  Javadoc, bytecode/reflection/import/source checks, the executable documentation example,
  link/anchor/fence/whitespace checks, exact ten-path scope, synchronized statuses, and the
  no-0017K-spec check passed. Training API, capabilities, semantic/foundational/adjacent contracts,
  architecture/ADRs/tests, conformance/integration, Java 26 Gradle configuration, dependencies, and
  other modules remain accurate unchanged because the task adds only model-owned pad/tile
  expression metadata without values, gradients, compiler/backend behavior, or execution.
- The independent task-0017I documentation review retained PadKind/TileKind Javadocs, tightened
  exact constructor-failure wording for PadAttrs/TileAttrs, and finalized Tensor API, glossary,
  task evidence, master plan, and roadmap. Focused/model/root tests, generated Javadoc,
  bytecode/reflection/import/source, link/anchor/fence/whitespace, exact ten-path scope, and status
  checks passed. Compile API, Training API, capabilities, related contracts, architecture/ADRs/
  tests, conformance/integration, Java 26 Gradle configuration, dependencies, and other modules
  remain accurate unchanged because this task adds only model-owned semantic values.
- The independent task-0017H documentation review corrected the public Javadoc's non-empty-result
  wording and arithmetic-failure detail, then finalized Tensor API, Compile API, glossary, task
  evidence, master plan, and roadmap. Training API, capabilities, semantic/foundational/completed
  task contracts, architecture/ADRs/tests, backend conformance and integration tests, Java 26
  Gradle configuration, dependencies, and other modules remain accurate unchanged because the
  task adds only model-owned slice expression metadata without values, physical storage aliases,
  gradient rules, compiler behavior, backend/ONNX behavior, or execution.
- Task 0017A adds the sole parameterless `CONTIGUOUS` identity in the new
  `model.operation.layout` package. It describes a request for logically equivalent canonical
  dense row-major zero-offset output, while remaining distinct from resolved
  `LayoutKind.DENSE_CONTIGUOUS` geometry and from planning/backend/runtime materialization.
- The independent task-0017A documentation review found the enum and constant Javadocs complete,
  then finalized Tensor API, glossary, task evidence, master plan, and roadmap. Operation and
  attribute foundations, resolved layout values, current operation families and Tensor
  expressions, capabilities, Compile API, Training API, architecture/ADRs/tests, backend
  conformance and integration tests, Java 26 Gradle configuration, and other modules remain
  accurate unchanged because this task adds only model-owned semantic vocabulary without public
  Tensor construction, descriptor derivation, input inspection, provenance, materialization,
  gradients, compiler behavior, dependencies, backend behavior, or execution.
- Task 0017B defines exactly one public `Tensor.contiguous()` method and one single-method
  package-private helper. It creates fresh storage-free provenance for every request. Fully static
  results receive newly resolved canonical dense row-major zero-offset layout; dynamic results
  retain exact Shape with unresolved layout. Input layout, storage, label, provenance, and values
  are not inspected, and compiler/planning/prepare/backend layers retain canonicalization and
  materialization responsibilities.
- Task 0017B completed that exact public method and helper. Independent documentation review
  finalized the Tensor and Compile APIs, glossary, Javadocs, task evidence, master plan, and
  roadmap after focused/model/root tests, generated Javadoc, bytecode/reflection/source/import,
  example, link/anchor/fence/whitespace, scope, and status checks passed. Training API,
  capabilities, related model contracts, architecture and architecture tests, conformance,
  integration, Gradle, dependencies, and other modules remain accurate without modification.
- Task 0017C adds `ShapeTransformKind` with exact `RESHAPE` and `EXPAND` identities plus one shared
  immutable `TargetShapeAttrs`. The stored Shape is already normalized model semantics and accepts
  scalar, zero-extent, static, and dynamic dimensions; public numeric `-1` inference, input count
  compatibility, right-aligned singleton expansion, descriptors, layouts, and provenance remain
  in task 0017D.
- The independent task-0017C documentation review found both production Javadocs complete, then
  finalized Tensor API, glossary, task evidence, master plan, and roadmap. Operation/attribute
  foundations, Shape/Dimension, contiguous semantics, resolved layout values, Tensor descriptors
  and expressions, capabilities, Compile API, Training API, focused architecture/ADRs/tests,
  conformance and integration tests, Java 26 Gradle configuration, and other modules remain
  accurate unchanged because this task adds only model-owned target-shape semantic vocabulary
  without public Tensor construction, inference, provenance, gradients, dependencies, or
  execution.
- The former combined reshape/expand expression task is split into 0017D and 0017D1. Reshape owns
  raw `long...` and exact Shape requests, `-1` inference, known element-count validation, and
  conditional contiguous-input alias-view geometry. Expand independently owns right-aligned
  singleton/leading-axis validation and zero-stride geometry. This avoids mixing two different
  validation and layout algebras in one implementation session.
- Task 0017D completed exactly `Tensor.reshape(long...)` and `Tensor.reshape(Shape)` plus one
  bounded package-private helper. Known counts must match, dynamic equality is deferred, and
  resolved view layout is published only for contiguous input plus static target. Other geometry
  remains unresolved without implicit materialization.
- Independent task-0017D documentation review finalized Tensor and helper Javadocs, Tensor API,
  Compile API, glossary, task evidence, master plan, and roadmap after focused/model/root tests,
  generated Javadoc, Java 26 example, bytecode/reflection/import/source, link/anchor/fence/
  whitespace, exact-scope, and status checks passed. Training API, capabilities, related model
  contracts, architecture/ADRs/tests, conformance, integration, Gradle, dependencies, and other
  modules remain accurate unchanged because the task adds only model-owned reshape expression
  metadata without values, gradients, compiler behavior, materialization, backend behavior, or
  execution.
- Task 0017D1 completed exactly `Tensor.expand(long...)` and `Tensor.expand(Shape)` plus one
  bounded package-private helper. It requires directional right-aligned equality or input-
  singleton compatibility, accepts new leading target axes, and rejects unprovable dynamic
  combinations without binding symbols.
- With static target geometry and any resolved input layout, task 0017D1 derives a new logical
  view with exact source offset, preserved aligned strides, and zero strides for repeated or new
  leading axes. Dynamic or unresolved geometry remains unresolved; storage aliasing,
  materialization, gradients, compiler behavior, lowering, and execution remain deferred.
- Independent task-0017D1 documentation review finalized Tensor and helper Javadocs, Tensor API,
  Compile API, glossary, task evidence, master plan, and roadmap after focused/model/root tests,
  generated Javadoc, Java 26 example execution, bytecode/reflection/import/source, link/anchor/
  fence/whitespace, exact-scope, and status checks passed. Training API, capabilities, related
  model contracts, architecture/ADRs/tests, conformance, integration, Gradle, dependencies, and
  other modules remain accurate unchanged because the task adds only model-owned expand
  expression metadata without values, gradients, compiler behavior, materialization, backend
  behavior, or execution.
- Task 0017E completed exact `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE` semantic identities,
  immutable complete output-to-input permutation attributes, and one normalized non-negative axis
  attribute shared by insertion/removal meanings. Rank-two transpose remains a later convenience
  over `PERMUTE [1, 0]`, not a fourth kind.
- Task 0017E owns semantic vocabulary only. Raw negative-axis normalization, input-rank and
  singleton validation, Shape/layout derivation, Tensor construction, provenance, gradients,
  compiler behavior, materialization, lowering, and execution remain deferred to their owning
  tasks and layers.
- Independent task-0017E documentation review found all three submitted production Javadocs
  complete unchanged, then finalized Tensor API, glossary, task evidence, master plan, and roadmap
  after focused/model/root tests, generated Javadoc, bytecode/reflection/import/source, link/
  anchor/fence/whitespace, exact nine-path scope, and status checks passed. Operation foundations,
  existing layout-operation semantics, capabilities, Compile API, Training API, architecture/
  ADRs/tests, conformance/integration material, Java 26 Gradle configuration, dependencies, and
  other modules remain accurate unchanged because the task adds only model-owned semantic values.
- The former combined task 0017F is split into permutation/transpose task 0017F and singleton
  insertion/removal task 0017F1. Permutation reorders existing Dimension and stride entries,
  whereas expand-dimensions and squeeze change rank and require different singleton and inserted-
  stride rules.
- Task 0017F completed exactly `Tensor.permute(int...)` and rank-two `Tensor.transpose()` plus one
  bounded package-private helper. It owns raw negative-axis normalization, complete permutation
  validation, exact Dimension/stride reordering, PERMUTE attributes, and provenance. Resolved view
  layout remains logical metadata without attached storage or an execution guarantee.
- Independent task-0017F documentation review finalized Tensor/helper Javadocs, Tensor API,
  Compile API, glossary, task evidence, master plan, and roadmap after focused/model/root tests,
  generated Javadoc, Java 26 example execution, bytecode/reflection/import/source, link/anchor/
  fence/whitespace, exact ten-path scope, and status checks passed. Training API, capabilities,
  related model contracts, architecture/ADRs/tests, conformance/integration, Gradle, dependencies,
  and other modules remain accurate unchanged because this task adds only model-owned permutation
  expression metadata without values, gradients, compiler behavior, materialization, backend
  behavior, or execution.
- Task 0017F1 completed exactly `Tensor.expandDims(int)` and `Tensor.squeeze(int)` plus one bounded
  package-private rank-editing helper. It owns insertion-axis normalization, static singleton
  proof, exact Dimension insertion/removal, resolved stride insertion/removal, semantic attributes,
  and provenance.
- Dynamic selected dimensions are not assumed to be singleton. Resolved results preserve offset
  and view metadata without storage attachment; gradients, compiler canonicalization, planning,
  lowering, and execution remain deferred.
- Independent task-0017F1 documentation review retained the complete production Javadocs and
  finalized Tensor API, Compile API, glossary, task evidence, master plan, and roadmap after
  focused/model/root tests, generated Javadoc, Java 26 example execution, bytecode/reflection/
  import/source, link/anchor/fence/whitespace, exact ten-path scope, and status checks passed.
  Training API, capabilities, semantic/foundational/completed permutation contracts,
  architecture/ADRs/tests, conformance/integration, Gradle, dependencies, and other modules remain
  accurate unchanged because this task adds only model-owned rank-editing expression metadata.
- The independent task-0016G documentation review found both production Javadocs complete, then
  finalized Tensor API, glossary, task evidence, master plan, and roadmap. Operation foundations,
  aggregate/masked reduction contracts, capabilities, Compile API, Training API, focused
  architecture/ADRs/tests, conformance and integration tests, and Java 26 build configuration
  remain accurate unchanged because this task adds only model-owned scan semantic vocabulary
  without Tensor construction, inference, provenance, gradients, dependencies, or execution.
- Task 0016A specifies `AggregateReductionKind` for SUM, MEAN, PROD, MIN, MAX, ALL, ANY, and
  ARG_MAX; `AxisReductionAttrs` for normalized single-axis ordinary reductions; explicit
  `NoOperationAttrs.INSTANCE` full forms; and `ArgMaxAttrs` plus FIRST_INDEX/LAST_INDEX tie policy.
  Tensor/Shape/data-type/result behavior remains in later tasks.
- The independent task-0016A documentation review found all four production Javadocs complete,
  then finalized Tensor API, glossary, task evidence, master plan, and roadmap. Compile API,
  Training API, capabilities, architecture/ADRs/tests, conformance and integration tests, Java 26
  build configuration, Shape axis normalization, operation foundations, existing concrete
  families, and Tensor expression contracts remain accurate unchanged because this task adds only
  model-owned reduction semantic vocabulary without public expressions, inference, provenance,
  gradients, dependencies, or execution.
- Task 0016B implements nine fluent full/axis/retained-axis `sum`, `mean`, and `prod` methods through
  one package-private helper. It accepts only floating inputs, normalizes axes through Shape,
  produces canonical rank-zero full results, preserves unaffected static/dynamic Dimension
  references, retains input type and gradient eligibility, and records exact one-input provenance
  without aggregation, numerical policy, gradient rules, graph capture, or execution.
- The independent task-0016B documentation review found the Tensor and helper Javadocs complete,
  then finalized Tensor API, Compile API current-expression inventory, glossary, task evidence,
  master plan, and roadmap. Training API, capabilities, architecture/ADRs/tests, conformance and
  integration tests, Java 26 build configuration, foundational contracts, and reduction semantic
  contracts remain accurate unchanged because this task adds only model-owned aggregate
  expression construction without numerical aggregation, gradient rules, compiler behavior,
  dependencies, or execution.
- Task 0016C adds six full/axis/retained-axis reduction `min` and `max` overloads by
  extending the existing `TensorReductionExpressions` helper and focused test rather than
  duplicating their validation and Shape derivation. Aggregate `MIN/MAX` stay typed separately
  from binary elementwise `MIN/MAX`; all five floating aggregate kinds share canonical rank-zero
  full results, normalized single-axis structure, exact input type and eligibility, unresolved
  storage-free results, and one-input provenance. Empty-domain, NaN, signed-zero, comparison,
  extrema-tie gradient, compiler, and execution behavior remain deferred.
- The independent task-0016C documentation review found the Tensor and helper Javadocs complete,
  finalized Tensor API, Compile API, glossary, task evidence, master plan, and roadmap, and caught
  a focused-test regression before closure. A separate constrained implementation turn restored
  the prior SUM/PROD freshness and nesting assertions alongside MIN/MAX coverage. Training API,
  capabilities, architecture/ADRs/tests, conformance and integration tests, Java 26 build
  configuration, and related foundational/operation contracts remain accurate unchanged because
  the task adds only model-owned extrema expression construction without numerical comparison,
  tie-gradient, compiler, dependency, or executable behavior.
- Task 0016D adds six full/axis/retained-axis `all` and `any` overloads by generalizing
  the existing six-method `TensorReductionExpressions` helper rather than duplicating its Shape
  logic. One kind-aware private validator preserves floating validation for all five numeric kinds
  and requires exact BOOL for `ALL/ANY`. Boolean results retain the shared canonical rank-zero/full
  and normalized one-axis structure, are fixed non-differentiable BOOL expressions, and record one
  input. Aggregate `ALL/ANY` remain typed separately from elementwise `AND/OR`; truth evaluation,
  empty-domain identities, gradients, compiler behavior, and execution remain deferred.
- The independent task-0016D documentation review finalized the new Tensor axis-overload and
  generalized-helper Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and
  roadmap. Training API, capabilities, architecture/ADRs/tests, conformance and integration tests,
  Java 26 build configuration, and related foundational, boolean-logical, and reduction contracts
  remain accurate unchanged because this task adds only model-owned BOOL aggregate expression
  construction without truth evaluation, empty-domain policy, gradients, compiler behavior,
  dependencies, or execution.
- Task 0016E adds three axis-only `argMax` overloads through a dedicated four-method
  package-private helper. Convenience forms explicitly use `FIRST_INDEX`; the complete form
  retains a non-null caller policy in `ArgMaxAttrs`. All floating and integral inputs are accepted,
  BOOL is rejected, and every result is fresh unresolved INT64 with false gradient eligibility and
  one-input provenance. The ordinary reduction helper remains unchanged because arg-max has no
  full form and owns different attributes/result semantics. Actual comparison, NaN/equality,
  empty-axis behavior, compiler work, and execution remain deferred.
- The independent task-0016E documentation review identified and returned an initial focused-test
  coverage gap before closure, then finalized the corrected seven-test suite's Tensor/helper
  Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap. Training
  API, capabilities, architecture/ADRs/tests, conformance and integration tests, Java 26 build
  configuration, ordinary reductions, and related foundational/reduction contracts remain
  accurate unchanged because this task adds only model-owned index-expression metadata without
  comparison, empty-axis policy, gradients, compiler behavior, dependencies, or execution.
- Completed task 0018U1 later broadened ordinary SUM/PROD/MIN/MAX to exact signed-integral input,
  selected their modular/signed/empty-domain semantics, and replaced the historical arg-max-only
  types/helper with shared arg-extrema contracts. It also fixed floating/integral arg-extrema
  ordering and static-empty-selected-axis rejection. The 0016A–0016E bullets above remain
  historical completion records rather than descriptions of the current public API.
- Task 0016F adds `MaskedReductionAttrs(axis, maskInputAxes)` and documents SUM/MEAN pairing
  without public Tensor construction. The immutable strictly increasing mapping records
  which input axis receives each mask dimension, so masks such as `[batch, time]` for
  `[batch, time, features]` remain representable without hiding reshape/expand behavior in a
  backend. False positions are excluded; all-false masked sum and masked mean both produce zero.
  Completion of 0016F left task 0016F1 Draft for later Shape-based mapping resolution and public
  axis-removing expressions.
- The independent task-0016F documentation review finalized exact constructor-failure Javadocs,
  the Tensor API semantic reference, glossary terminology, task evidence, master plan, and
  roadmap. Compile API, Training API, capabilities, focused architecture/ADRs/tests, conformance
  and integration tests, Gradle configuration, operation foundations, ordinary/arg-max reduction
  attributes, other aggregate constants, and existing Tensor expressions remain accurate
  unchanged because this task adds only model-owned masked semantic attributes without public
  Tensor construction, Shape resolution, gradients, dependencies, compiler behavior, or
  execution.
- Task 0016F1 adds exactly `Tensor.sum(axis, mask)` and `mean(axis, mask)` plus one
  package-private construction helper. It resolves an ordered injective mapping from mask
  dimensions to input axes using locally provable Dimension equality or mask-side singleton
  expansion. Candidate preference is explicit: cover the reduced axis when possible, then
  minimize positional displacement, then choose lexicographically. The result removes the axis,
  preserves input type and eligibility, and records one masked operation with exact ordered
  `[input, mask]` provenance. Values, storage alignment, gradients, compiler behavior, and
  execution remain deferred.
- The independent task-0016F1 documentation review found the Tensor and helper Javadocs complete,
  then finalized Tensor API, Compile API, glossary, task evidence, master plan, and roadmap.
  Training API, capabilities, architecture/ADRs/tests, backend conformance and integration tests,
  Gradle configuration, foundational contracts, ordinary reductions, and other modules remain
  accurate unchanged because the task adds only model-owned masked expression metadata without
  value work, gradient rules, compiler behavior, dependencies, or execution.
- The independent task-0015G documentation review found the enum and record Javadocs complete,
  then finalized Tensor API, glossary, task evidence, master plan, and roadmap. Compile API,
  Training API, capabilities, architecture/ADRs/tests, conformance and integration tests, Java 26
  build configuration, foundational and existing concrete operation contracts, and Tensor
  expression contracts remain accurate unchanged because the task adds only model-owned cast
  identity and target attributes without public expressions, inference, numerical conversion,
  gradients, provenance, dependencies, or execution.
- The independent task-0015C documentation review found the enum and every constant Javadoc
  complete, then finalized Tensor API, glossary, task evidence, master plan, and roadmap. Compile
  API, Training API, capabilities, architecture/ADRs/tests, conformance and integration tests,
  Java 26 build configuration, foundational operation contracts, existing kind families, and
  Tensor expression contracts remain accurate unchanged because the task adds only model-owned
  semantic vocabulary without public expressions, dependencies, inference, provenance, training,
  or execution behavior.
- The independent task-0015B documentation review found the Tensor and helper Javadocs complete,
  then finalized Tensor API, Compile API current-expression inventory, glossary, task evidence,
  master plan, and roadmap. Training API, capabilities, architecture/ADRs/tests, conformance and
  integration tests, Java 26 build configuration, foundational contracts, and existing expression
  families remain accurate unchanged because the task adds only model-owned comparison expression
  construction without compiler, training, dependency, numerical, or executable behavior.
- The independent task-0015A documentation review found the enum and every constant Javadoc
  complete, then finalized Tensor API, glossary, task evidence, master plan, and roadmap. Compile
  API, Training API, capabilities, architecture/ADRs/tests, build configuration, existing
  operation/Tensor/expression contracts, and other modules remain accurate unchanged because the
  task adds only model-owned semantic vocabulary without public expressions, dependencies,
  inference, provenance, training, or execution behavior.
- The independent task-0014F documentation review finalized Tensor API, the pre-authorized Compile
  API current-status wording, glossary, task evidence, master plan, and roadmap. Existing Javadocs
  were already complete. Training API, capabilities, architecture/ADRs/tests, build configuration,
  existing scalar semantics, binary/unary expressions, and other modules remain accurate
  unchanged because the task adds only model-owned expression construction.
- The independent task-0014E documentation review finalized only the two new record Javadocs,
  Tensor API, glossary, task evidence, master plan, and roadmap. Existing operation, Tensor,
  Compile API, Training API, capabilities, architecture, test, and build contracts remain accurate
  unchanged because 0014E adds semantic values without public expressions or cross-layer behavior.
- The independent task-0014B documentation review found stale public-Tensor status in the Compile
  API. Explicit authorization expanded the task to ten paths solely to correct that statement.
  The updated page describes public Tensor and binary expression construction as current while
  preserving compiler entry, capture, inference, optimization, artifacts, and engine APIs as
  planned.

## Risks

- Accidentally treating public `Tensor` as compiled IR.
- Leaking runtime storage or backend support into the model.
- Expanding the public API before value-model invariants are stable.
- Creating cycles between `operation`, `tensor`, and `graph` packages.
- Enabling preview or incubator features globally instead of containing them in the module that requires them.
- Treating the operation inventory as permission to move graph inference, autograd rules, fallback, or execution into model.
- Reproducing accidental legacy behavior instead of specifying and testing the intended contract.
- Growing a registry or abstraction hierarchy while fixing kind/attribute and cardinality
  validation instead of keeping the contract small and family-owned.
- Treating shared public provenance as graph IR or giving Tensor graph-local producer identity.
- Adding convolution, pooling, top-K, or graph randomness before symbolic extents and multi-output
  provenance can represent their results honestly.
- Letting a global identity counter wrap, collide under concurrency, or become a runtime service
  registry rather than remaining hidden model-only allocation state.
- Treating completed JVM heap allocation as import/population or native/runtime allocation parity
  before the applicable typed-population and deterministic-resource contracts exist.

## Notes

Execute tasks in table order, including package migrations 0003A through 0003C before task 0004.
Completed task
[0018K](tasks/0018k-operation-signature-and-construction-hardening.md) was an explicitly
documented atomic-migration exception to the usual file-count guardrail because partial signature
enforcement would either break valid current families or retain a permissive unsafe fallback.
Tasks 0018L, 0018M, 0018M1, 0018N, 0018O, 0018P, 0018Q, 0018R, 0018S, 0018T, and 0018T1 are
complete. Task 0018U, task 0018U1, and linked task 0018V are also complete. Focused MATMUL task
0019, 0019A, and 0019A1 are complete.
Task 0019A2, task 0019B, task 0019B1, task 0019C, task 0019C1, and task 0019D are complete. Task
0019E, task 0020, task 0020A, task 0020A1, and task 0021 are complete. Task
[0021A](tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md) is Complete. Task
0021B is Complete. Task
[0021C](tasks/0021c-batch-normalization-training-and-statistic-transition.md) is Complete. Task
0022, 0022A, and 0022B are Complete. Task 0023 is Complete with its detailed audit
specification and result artifact. Tasks 0023A and 0023B are Complete with their detailed
specifications. Tasks 0023C, 0023D, 0023E, and 0023F are Complete with detailed specifications,
while task 0024 is Complete with its closure artifact and task 0024A is Complete. The selected
model capability milestone remains historically closed. Accepted ADR 0009 exposed one later
compiler-enabling foundation gap, and task 0025 is Complete. The post-Compiler-0005 reassessment
selected task 0025A for the remaining portable floating comparison/extrema/clamp forward-contract
prerequisite, and that focused task is Complete.
Other operation-family rows are not permission for oversized
implementations; apply the normal limits in the
[planning guide](../../planning-guide.md).

The 0019A–0019C suffixes are established sequential rows after task 0019. Decimal follow-ups
0019A1–0019A2 split the original 0019A scope without changing established 0019B–0019E or
renumbering 0020–0024. Follow-up 0019B1 similarly splits dropout from the reusable 0019B state
foundation without changing 0019C–0019E. Follow-up 0019C1 splits genuine multi-output top-K from
0019C full sort/argsort without changing established 0019D–0019E. These rows are independent frontiers, not hidden
subtasks of 0019; their
`Depends on` entries list technical
prerequisites, while table order remains the default execution order.

The former broad 0020 row is split into focused convolution task 0020 and pooling follow-ups
0020A–0020A1 without renumbering established tasks 0021–0024. Convolution is first because it
fixes the weight-derived kernel, grouped-channel, optional-bias, promotion, and dynamic spatial-
expression precedents. Max pooling is next because its excluded-padding, extrema, tie, and literal
ceil-grid contract is cohesive. Average pooling follows because divisor and padding-count
semantics are independent. This is a cohesion and dependency split, not authorization to share
an operation attribute whose meaning differs.

Package migrations 0003A–0003C and tasks 0004–0009 are complete. Task 0008 added the two local
immutable graph element records, and task 0009 added the structurally closed graph container,
forward/backward node phases, and standalone publication binding. Task 0010 added the sealed raw
host-storage boundary and exact-size borrowed Java 26 memory-segment wrapper. Task 0011 added the
completed public Tensor skeleton with stable metadata and a synchronized borrowed host-storage
association. Task 0012 completed public descriptor-based construction, optional borrowed storage
attachment, and JVM-scoped ID allocation only. Task 0012A completed exact-span typed primitive-
array heap allocation through automatic-scope segments without changing the borrowed storage
contract. Task 0012B completed copied flat typed import for all six data types with
dense-contiguous/count validation and BOOL normalization. Task 0012C completed validated
rectangular nested primitive-array import, exact carrier/static-shape inference, and row-major
delegation to flat import. Task 0012D completed exact typed scalars and independent dense zero/one
constants. Task 0012E completed deterministic typed range and strict/cyclic prefix population, and
task 0012F completed explicit-source normal-random population, task 0012G completed bounded
continuous-uniform floating population, task 0012H completed bounded-integral population, and task
0012I completed BOOL Bernoulli population, task 0013 completed immutable Tensor provenance and
the package-private derived-construction seam, and task 0013A completed type-safe full-value and
rectangular identity creation. The completed post-foundation checkpoint selected sequential model
operation-family work, and task 0014A completed the first concrete parameterless kind family.
Task 0014B completed the matching public expression construction and all authorized documentation,
including the focused Compile API status correction. The post-0014B reassessment selected
continued ordered model work. Task 0014C completed the fifteen parameterless unary elementwise
semantic kinds, and task 0014D completed their public floating Tensor expression construction.
Task 0014E completed the scalar arithmetic/clamp semantic kinds and exact immutable attributes,
task 0014F completed their public floating Tensor expression construction, task 0015A completed the
six parameterless ordered binary comparison semantic kinds, and task 0015B completed their public
floating comparison Tensor construction. Task 0015C completed the three parameterless boolean
logical semantic kinds, and task 0015D completed their public BOOL-only binary/unary Tensor
expression construction. Task 0015E completed the sole parameterless `WHERE` conditional-selection
semantic identity, and task 0015F completed its public static Tensor expression construction.
Task 0015G completed the exact `CAST` semantic identity and immutable target-data-type attributes.
Task 0015H completed its public storage-free Tensor expression construction. The former broad task
0016 is decomposed into tasks 0016A–0016J. Tasks 0016A through 0016J, including 0016F1, are
complete. The former broad task 0017 is decomposed into tasks 0017A–0017N. Task 0017A is complete;
task 0017B is complete, task 0017C is complete, task 0017D is complete, and task 0017D1 is
complete. Task 0017E, task 0017F, task 0017F1, task 0017G, task 0017H, and task 0017I are also
complete. Tasks 0017J, 0017K, 0017L, 0017M, and 0017N are also complete. The former broad task
0018 is decomposed into 0018A–0018J. Task 0018A is complete with first-class scalar-select
semantics and normalized axis/index attributes. Task 0018B is also complete with public scalar-
select expression construction. Task 0018C is complete with the three exact axis-gather meanings
and their shared normalized-axis attributes. Task 0018D is complete with the four public
Tensor-index expressions. Task 0018D1 is complete with the primitive-take convenience. Task 0018E
is complete with `GATHER_ND` and normalized batch-dimension attributes. Task 0018F is complete with
public Gather-ND expression construction. Task 0018G is complete with functional axis-scatter
semantic values. Task 0018H is complete with public functional axis-scatter expression
construction. Task 0018I is complete with functional Scatter-ND semantic values. Task 0018J is
complete with public functional Scatter-ND expression construction. The capability reset inserted
0018K–0018V as the new foundation frontier. Tasks 0018K through 0018T1 and task 0018U are complete;
0018U1 and linked task 0018V are complete. Task 0019 is complete with its detailed MATMUL
specification. Tasks 0019A, 0019A1, 0019A2, 0019B, 0019B1, 0019C, 0019C1, and 0019D are complete.
Tasks 0019E, 0020, 0020A, 0020A1, 0021, and 0021A are complete. Task 0021B is Complete. Task
[0021C](tasks/0021c-batch-normalization-training-and-statistic-transition.md) is Complete. Task
0022, 0022A, and 0022B are Complete. Task 0023 is Complete with its detailed audit
specification and result artifact. Tasks 0023A and 0023B are Complete with their detailed
specifications. Tasks 0023C, 0023D, 0023E, and 0023F are Complete with detailed specifications,
while task 0024 is Complete with its closure artifact and task 0024A is Complete. The selected
model capability milestone remains historically closed. Task 0025 is the completed focused
compiler prerequisite. Task 0025A is the completed later detailed model prerequisite. Compiler
0005A remains Draft and may be promoted only through a separate planning decision.
The legacy branch must be consulted read-only for capability and test evidence when preparing each
applicable capability task.
