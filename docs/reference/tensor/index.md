<a id="tensor-api"></a>
# [Tensor](../../glossary.md#tensor) reference: values, expressions and geometry

A Tensor has stable [identity](../../glossary.md#term-identity-meanings) and a logical [descriptor](../../glossary.md#term-descriptor) (type, [shape](../../glossary.md#shape), optional [layout](../../glossary.md#layout), [gradient](../../glossary.md#term-gradient) eligibility), plus an optional [mutable](../../glossary.md#term-mutable) association with [borrowed](../../glossary.md#ownership-borrowing-and-lease) [host storage](../../glossary.md#host-storage). An expression result describes an [operation](../../glossary.md#operation) [occurrence](../../glossary.md#term-occurrence); constructing it does not execute its arithmetic. Start with the family that matches your question, not a complete inventory of methods.

[Reference goals](../index.md) · [Tensor source](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java) · [Generated Tensor Javadoc, build required](../../../modules/model/build/docs/javadoc/io/github/pho001/synaptik/model/tensor/Tensor.html)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc))</span> ([obtain it](../javadoc.md))

**New to tensor terminology?** A tensor is an array with one [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) per **[axis](../../glossary.md#term-axis)**.
Its **shape** lists the axis sizes; its **[rank](../../glossary.md#term-rank)** counts them. Shape `[2, 3]` means two rows
and three columns. A **descriptor** describes that array without containing its numbers.
An **expression** records how a result should be calculated; **execution** performs the
calculation. **Gradient eligibility** requests that selected [derivatives](../../glossary.md#term-derivative) may be needed; it
does not itself establish derivative support. Start with [shapes](shapes.md) for coordinates
and [layouts](layouts.md) for how coordinates map to storage.

## Follow the data boundary

The caller is the application or framework component invoking a method. For example, an NN layer
can call `Tensor.linear` while building an expression on the application's behalf.

- `TensorFactory.create` can make a **leaf**: a Tensor with no producing expression, possibly
  without any stored values.
- [Import](import.md) copies the application's array values into independent storage.
- [Attaching storage](storage-association.md) instead retains an existing wrapper without copying;
  the memory's owner must keep it alive.
- An arithmetic Tensor method retains its input Tensor references and returns a description of a
  future result. Compiler and Engine perform the separate capture, preparation and execution work.

Ownership is not a rule that every Tensor must be closed. Read
[host-storage lifetimes](host-storage.md#borrowed-lifetime-example) before using arena-backed memory.

## Foundations and creation

| Question | [Canonical](../../glossary.md#term-canonical-form) topics |
|---|---|
| What does a [descriptor](../../glossary.md#term-descriptor) mean? | [Types and scalar bits](data-types.md)<span data-glossary-navigation="true"> ([scalar glossary](../../glossary.md#term-scalar))</span>, [shapes and broadcasting](shapes.md)<span data-glossary-navigation="true"> ([shapes glossary](../../glossary.md#shape); [broadcasting glossary](../../glossary.md#broadcasting))</span>, [resolved layouts](layouts.md)<span data-glossary-navigation="true"> ([layouts glossary](../../glossary.md#layout))</span>, [descriptors](descriptors.md) |
| Which [identity](../../glossary.md#term-identity-meanings) is retained? | [Tensor identity and canonical producer outputs](identity.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../glossary.md#tensor); [producer glossary](../../glossary.md#term-producer))</span>, [typed IDs](identifiers.md) |
| How do values enter [host storage](../../glossary.md#host-storage)? | [Allocation/import contracts](creation.md), [constants](constants.md)<span data-glossary-navigation="true"> ([constants glossary](../../glossary.md#term-constant))</span>, [flat and nested imports](import.md) |
| Who owns the data? | [Mutable storage association](storage-association.md)<span data-glossary-navigation="true"> ([Mutable glossary](../../glossary.md#term-mutable))</span>, [host-storage lifetimes](host-storage.md)<span data-glossary-navigation="true"> ([host glossary](../../glossary.md#term-host))</span>, [failure and ownership summary](failures.md)<span data-glossary-navigation="true"> ([ownership glossary](../../glossary.md#term-ownership-meanings))</span> |
| Which random state is consumed? | [Eager ranges and random initialization](random-initialization.md)<span data-glossary-navigation="true"> ([Eager glossary](../../glossary.md#term-eager-evaluation))</span>, [explicit graph RNG and dropout](graph-randomness.md)<span data-glossary-navigation="true"> ([graph glossary](../../glossary.md#graph); [RNG glossary](../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../glossary.md#dropout--inverted-dropout))</span> |

## [Operation](../../glossary.md#operation) families

| Intent | [Canonical](../../glossary.md#term-canonical-form) topics |
|---|---|
| Calculate [elementwise](../../glossary.md#term-elementwise) | [Binary arithmetic/comparison/logic](arithmetic.md)<span data-glossary-navigation="true"> ([Binary glossary](../../glossary.md#term-binary-descriptor-and-metadata))</span>, [selection and casts](selection-casts.md), [unary/scalar/clamp](elementwise.md)<span data-glossary-navigation="true"> ([unary glossary](../../glossary.md#term-unary); [scalar glossary](../../glossary.md#term-scalar))</span> |
| Reduce values or produce indices | [Numeric and statistical reductions](reductions.md), [masked/Boolean/index reductions](masked-index-reductions.md)<span data-glossary-navigation="true"> ([Boolean glossary](../../glossary.md#data-type--datatype))</span>, [sort and Top-K](sort-topk.md) |
| Normalize or describe a [loss](../../glossary.md#loss) | [Normalization](normalization.md)<span data-glossary-navigation="true"> ([Normalization glossary](../../glossary.md#term-normalization))</span>, [training statistics and losses](statistics-losses.md) |
| Describe recurrence | [Cumulative/fixed scans](fixed-scans.md), [general Scan source](general-scan.md)<span data-glossary-navigation="true"> ([general Scan glossary](../../glossary.md#general-differentiable-scan))</span> |
| Change [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) geometry | [Contiguous/reshape/expand](reshape-expand.md)<span data-glossary-navigation="true"> ([Contiguous glossary](../../glossary.md#term-contiguous))</span>, [axis transforms](axis-transforms.md)<span data-glossary-navigation="true"> ([axis glossary](../../glossary.md#term-axis))</span>, [slicing and functional updates](slicing.md) |
| Index or compose [tensors](../../glossary.md#tensor) | [Select/gather/one-hot](gather.md)<span data-glossary-navigation="true"> ([gather glossary](../../glossary.md#gather))</span>, [Gather-ND and scatter](scatter.md)<span data-glossary-navigation="true"> ([Gather-ND glossary](../../glossary.md#gather-nd))</span>, [pad/tile/concat/stack](composition.md) |
| Project or attend | [MATMUL and linear projection](matmul-linear.md)<span data-glossary-navigation="true"> ([MATMUL glossary](../../glossary.md#matrix-multiplication--matmul))</span>, [attention](attention.md) |
| Use [spatial](../../glossary.md#term-spatial-axes) operators | [Window transforms](windows.md), [convolution](convolution.md)<span data-glossary-navigation="true"> ([convolution glossary](../../glossary.md#term-channels-first-convolution-layers))</span>, [pooling](pooling.md)<span data-glossary-navigation="true"> ([pooling glossary](../../glossary.md#term-pooling))</span> |

The first learning step needs creation, descriptors, linear projection, subtraction, multiplication
and SUM, but no RNG or Scan. For two finite real input values $x=[1,3]$ shaped `[2,1]`,
one scalar weight $w=0$ gives predictions $p_i=x_iw=0$ for $i=0,1$.
Against targets $t=[2,6]$, squared-error SUM is
$(p_0-t_0)^2+(p_1-t_1)^2=(0-2)^2+(0-6)^2=40$.
This explains the intended objective, not construction-time evaluation or permission to reorder
rounded arithmetic. [Engine](../engine/lifecycle.md) evaluates the request and
[Compiler](../compiler/autograd.md) constructs supported gradients.

## [Compiler](../../glossary.md#term-compiler)-facing semantic data

Use these when reading a captured [graph](../../glossary.md#graph) or implementing an inward collaboration, not as an alternate execution [API](../../glossary.md#term-application-programming-interface-api): [operation signatures](operation-foundation.md)<span data-glossary-navigation="true"> ([operation glossary](../../glossary.md#operation); [signatures glossary](../../glossary.md#term-signature))</span>, [arithmetic and cast kinds](operation-arithmetic.md), [reduction kinds](operation-reductions.md), [geometry kinds](operation-geometry.md), [indexing/composition kinds](operation-indexing.md), [window kinds](operation-windows.md), [elementwise kinds](operation-elementwise.md)<span data-glossary-navigation="true"> ([elementwise glossary](../../glossary.md#term-elementwise))</span>, and [graph values/nodes](compiled-graph.md)<span data-glossary-navigation="true"> ([nodes glossary](../../glossary.md#node))</span>.

## [Qualification](../../glossary.md#term-qualification) and history

[Tensor](../../glossary.md#tensor) [metadata](../../glossary.md#term-metadata) construction does not perform downstream stages. [Compiler](../../glossary.md#term-compiler) [inference](../../glossary.md#term-inference-meanings), selected differentiation and [backend](../../glossary.md#backend) preparation/execution are implemented in their own [owners](../../glossary.md#term-owner); their [occurrence](../../glossary.md#term-occurrence) guards remain independent. Use the [current production matrix](../../backend-guide/production-support.md)<span data-glossary-navigation="true"> ([matrix glossary](../../glossary.md#term-matrix-meanings))</span>, not earlier task-era absence lists, to determine executable occurrences and [derivatives](../../glossary.md#term-derivative).
