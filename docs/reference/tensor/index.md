<a id="tensor-api"></a>
# Tensor reference: values, expressions and geometry

A Tensor has stable identity and a logical descriptor (type, shape, optional layout, gradient eligibility), plus an optional mutable association with borrowed host storage. An expression result describes an operation occurrence; constructing it does not execute its arithmetic. Start with the family that matches your question, not a complete inventory of methods.

[Reference goals](../index.md) · [Tensor source](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java) · [Generated Tensor Javadoc, build required](../../../modules/model/build/docs/javadoc/io/github/pho001/synaptik/model/tensor/Tensor.html) ([obtain it](../javadoc.md))

**New to tensor terminology?** A tensor is an array with one coordinate per **axis**.
Its **shape** lists the axis sizes; its **rank** counts them. Shape `[2, 3]` means two rows
and three columns. A **descriptor** describes that array without containing its numbers.
An **expression** records how a result should be calculated; **execution** performs the
calculation. **Gradient eligibility** requests that selected derivatives may be needed; it
does not itself establish derivative support. Start with [shapes](shapes.md) for coordinates
and [layouts](layouts.md) for how coordinates map to storage.

## Foundations and creation

| Question | Canonical topics |
|---|---|
| What does a descriptor mean? | [Types and scalar bits](data-types.md), [shapes and broadcasting](shapes.md), [resolved layouts](layouts.md), [descriptors](descriptors.md) |
| Which identity is retained? | [Tensor identity and canonical producer outputs](identity.md), [typed IDs](identifiers.md) |
| How do values enter host storage? | [Allocation/import contracts](creation.md), [constants](constants.md), [flat and nested imports](import.md) |
| Who owns the data? | [Mutable storage association](storage-association.md), [host-storage lifetimes](host-storage.md), [failure and ownership summary](failures.md) |
| Which random state is consumed? | [Eager ranges and random initialization](random-initialization.md), [explicit graph RNG and dropout](graph-randomness.md) |

## Operation families

| Intent | Canonical topics |
|---|---|
| Calculate elementwise | [Binary arithmetic/comparison/logic](arithmetic.md), [selection and casts](selection-casts.md), [unary/scalar/clamp](elementwise.md) |
| Reduce values or produce indices | [Numeric and statistical reductions](reductions.md), [masked/Boolean/index reductions](masked-index-reductions.md), [sort and Top-K](sort-topk.md) |
| Normalize or describe a loss | [Normalization](normalization.md), [training statistics and losses](statistics-losses.md) |
| Describe recurrence | [Cumulative/fixed scans](fixed-scans.md), [general Scan source](general-scan.md) |
| Change coordinate geometry | [Contiguous/reshape/expand](reshape-expand.md), [axis transforms](axis-transforms.md), [slicing and functional updates](slicing.md) |
| Index or compose tensors | [Select/gather/one-hot](gather.md), [Gather-ND and scatter](scatter.md), [pad/tile/concat/stack](composition.md) |
| Project or attend | [MATMUL and linear projection](matmul-linear.md), [attention](attention.md) |
| Use spatial operators | [Window transforms](windows.md), [convolution](convolution.md), [pooling](pooling.md) |

The first learning step needs only creation, a descriptor, linear projection, subtraction, multiplication and SUM. It uses no RNG or Scan. Input `[1,3]` with shape `[2,1]` times transposed weight `0` produces `[0,0]`; squared error against `[2,6]` sums to `40`. Execution and gradient construction belong to [Engine](../engine/lifecycle.md) and [Compiler](../compiler/autograd.md), not Tensor construction itself.

## Compiler-facing semantic data

Use these when reading a captured graph or implementing an inward collaboration, not as an alternate execution API: [operation signatures](operation-foundation.md), [arithmetic and cast kinds](operation-arithmetic.md), [reduction kinds](operation-reductions.md), [geometry kinds](operation-geometry.md), [indexing/composition kinds](operation-indexing.md), [window kinds](operation-windows.md), [elementwise kinds](operation-elementwise.md), and [graph values/nodes](compiled-graph.md).

## Qualification and history

Tensor metadata construction does not perform downstream stages. Compiler inference, selected differentiation and backend preparation/execution are implemented in their own owners; their occurrence guards remain independent. Use the [current production matrix](../../backend-guide/production-support.md), not earlier task-era absence lists, to determine executable occurrences and derivatives.
