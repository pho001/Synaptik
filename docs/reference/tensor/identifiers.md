# Typed identifiers

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Read the name in its owning context

The application obtains a `TensorId` from a factory-created Tensor. The Compiler separately
assigns graph-local `NodeId` and `ValueId` names when capturing expressions. Passing those IDs
to a graph record connects descriptions, not memory addresses, and copies no numeric values.
For example, value `7` in graph A and value `7` in graph B do not name shared storage.
See [compiled graph construction](compiled-graph.md#graph-values-and-compiled-nodes) for how
the graph checks references and [producer identity](identity.md) for the objects before capture.

**[Reader model](../../glossary.md#term-reader-model).** A typed identifier is a number labeled with what it identifies.
`TensorId(7)`, `NodeId(7)`, and `ValueId(7)` are different kinds of names, not interchangeable
references to the same entity. A [node](../../glossary.md#node) names a calculation [occurrence](../../glossary.md#term-occurrence); a value names data entering
or leaving calculations. **[Graph](../../glossary.md#graph)-local** means a name is interpreted only inside its owning
graph, like a local variable inside its function. A public [Tensor](../../glossary.md#tensor) can participate in more than
one graph without adopting one graph's local numbering.

<a id="typed-identifiers"></a>
## Typed identifiers

The [model](../../glossary.md#term-model) separates public [tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings) from [graph](../../glossary.md#graph)-local computation and data identity:

- `io.github.pho001.synaptik.model.tensor.TensorId` identifies public [mutable](../../glossary.md#term-mutable) tensor state;
- `io.github.pho001.synaptik.model.graph.NodeId` identifies a computation [occurrence](../../glossary.md#term-occurrence) within an owning graph; and
- `io.github.pho001.synaptik.model.graph.ValueId` identifies an input, intermediate, or output logical value within an owning graph.

Each identifier is an [immutable](../../glossary.md#term-immutable) record over one non-negative `long` value. Zero is valid, negative
sentinels are rejected, and different identifier types cannot be interchanged. The records
themselves do not generate or guarantee uniqueness. `TensorFactory` now allocates [`TensorId`](../../glossary.md#tensorid)
values unique among its own calls in one [JVM](../../glossary.md#term-java-virtual-machine-jvm); manually constructed values remain outside that
guarantee. Future graph builders and [compiler](../../glossary.md#term-compiler) sessions own graph-local allocation within their
lifecycles.

Graph-local numeric identifiers may be reused by different graph containers. [`NodeId`](../../glossary.md#nodeid) identifies a computation, whereas [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) identifies data flowing between computations. A value can exist without a producing [node](../../glossary.md#node), one node can produce multiple values, and one value can have multiple consumers.

An implemented `ForwardPublicationBinding` associates a public `TensorId` with a `ValueId`. The
[binding](../../glossary.md#term-binding) is standalone and cannot prove that the value belongs to a particular graph; current
compiler-owned `PublicationPlan` provides that context and separates its forward bindings from
Compiler-owned [gradient](../../glossary.md#term-gradient) bindings.
The implemented `Tensor` stores no graph-local IDs
because the same tensor may participate in multiple separately compiled graphs. `OperationId` is
not currently defined: [operation](../../glossary.md#operation) semantics occur through graph nodes, and no independent
operation-identity [lifecycle](../../glossary.md#lifecycle) has been established.


---

[Previous topic](host-storage.md) · [Next topic](operation-foundation.md)
