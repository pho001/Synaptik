# Typed identifiers

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**Reader model.** A typed identifier is a number labeled with what it identifies.
`TensorId(7)`, `NodeId(7)`, and `ValueId(7)` are different kinds of names, not interchangeable
references to the same entity. A node names a calculation occurrence; a value names data entering
or leaving calculations. **Graph-local** means a name is interpreted only inside its owning
graph, like a local variable inside its function. A public Tensor can participate in more than
one graph without adopting one graph's local numbering.

<a id="typed-identifiers"></a>
## Typed identifiers

The model separates public tensor identity from graph-local computation and data identity:

- `io.github.pho001.synaptik.model.tensor.TensorId` identifies public mutable tensor state;
- `io.github.pho001.synaptik.model.graph.NodeId` identifies a computation occurrence within an owning graph; and
- `io.github.pho001.synaptik.model.graph.ValueId` identifies an input, intermediate, or output logical value within an owning graph.

Each identifier is an immutable record over one non-negative `long` value. Zero is valid, negative
sentinels are rejected, and different identifier types cannot be interchanged. The records
themselves do not generate or guarantee uniqueness. `TensorFactory` now allocates `TensorId`
values unique among its own calls in one JVM; manually constructed values remain outside that
guarantee. Future graph builders and compiler sessions own graph-local allocation within their
lifecycles.

Graph-local numeric identifiers may be reused by different graph containers. `NodeId` identifies a computation, whereas `ValueId` identifies data flowing between computations. A value can exist without a producing node, one node can produce multiple values, and one value can have multiple consumers.

An implemented `ForwardPublicationBinding` associates a public `TensorId` with a `ValueId`. The
binding is standalone and cannot prove that the value belongs to a particular graph; current
compiler-owned `PublicationPlan` provides that context and separates its forward bindings from
Compiler-owned gradient bindings.
The implemented `Tensor` stores no graph-local IDs
because the same tensor may participate in multiple separately compiled graphs. `OperationId` is
not currently defined: operation semantics occur through graph nodes, and no independent
operation-identity lifecycle has been established.


---

[Previous topic](host-storage.md) · [Next topic](operation-foundation.md)
