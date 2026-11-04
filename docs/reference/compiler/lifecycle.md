# Internal [lifecycle](../../glossary.md#lifecycle), failures and interpretation

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

Read this lifecycle as a sequence of [admission](../../glossary.md#term-admission) checks, not successive attempts to [run](../../glossary.md#term-run) the
computation. An **[occurrence](../../glossary.md#term-occurrence)** is one particular [operation](../../glossary.md#operation) in the [graph](../../glossary.md#graph), even when another operation
has the same kind and [shapes](../../glossary.md#shape). [Owner](../../glossary.md#term-owner) selection names its responsible [backend](../../glossary.md#backend); later **route**
selection chooses how that backend implements it.

<a id="current-internal-lifecycle-and-planned-public-failures"></a>
## Current internal [lifecycle](../../glossary.md#lifecycle) and planned public failures

```text
forward Tensor outputs
  -> if the internal mode requests first-order gradients:
     preflight -> Tensor formulas -> combined expression
  -> one phase-aware capture
  -> inference and validation
  -> canonicalization and one-shot exact whole-graph optimization
  -> published-constant, final convolution, and static-result logical-descriptor closure
  -> publication, backend ownership, partitions, and logical memory
  -> CompileArtifacts
```

The current internal entry rejects invalid [shapes](../../glossary.md#shape), data types, operations, [graph](../../glossary.md#graph) structure,
[publication](../../glossary.md#term-publication) roles, [planning](../../glossary.md#term-planning) composition, unsatisfied hard capability, partitions, memory, source
roles, or diagnostics before returning an [artifact](../../glossary.md#term-artifact). A no-hard-eligible [occurrence](../../glossary.md#term-occurrence) becomes a
[compiler](../../glossary.md#term-compiler)-owned `IllegalStateException` with [node](../../glossary.md#node) index, ID, and [operation](../../glossary.md#operation)-kind context, while the
Planning failure remains its cause. Provider [runtime](../../glossary.md#term-runtime) failures otherwise propagate unchanged.
The [integration port](../../glossary.md#term-integration-port) preserves the current compiler and Planning failure classes and [identities](../../glossary.md#term-identity-meanings);
it performs no [Engine](../../glossary.md#term-engine) exception translation. An ordinary-user Engine failure taxonomy remains
planned.

<a id="example-interpretation"></a>
## Example interpretation

For an explicitly composed [graph](../../glossary.md#graph) containing a [matrix multiplication](../../glossary.md#matrix-multiplication--matmul) and a small [elementwise operation](../../glossary.md#elementwise-unary-binary-and-interval-vocabulary),
capability analysis may find both [CPU](../../glossary.md#term-cpu) and [Metal](../../glossary.md#term-metal) valid. [Backend](../../glossary.md#backend)-neutral scoring may assign both
[nodes](../../glossary.md#node) to Metal to avoid a transfer boundary. The [artifact](../../glossary.md#term-artifact) records only `owner = Metal`; it does
not record [MPSGraph](../../glossary.md#term-mpsgraph) or a custom Metal [kernel](../../glossary.md#kernel). Metal [prepare](../../glossary.md#prepare) makes that later choice.

Both CPU and Metal [lifecycle](../../glossary.md#lifecycle) integrations are current through explicit `Engine.builder()`
composition and exact [occurrence](../../glossary.md#term-occurrence) [ownership](../../glossary.md#term-ownership-meanings); no discovery, retry or CPU fallback is implied.
Current `GraphCompilationPort` can invoke package-private `GraphCompiler` to construct and
validate its bounded forward-only or combined first-order graph, query explicitly supplied
providers, select [backend ownership](../../glossary.md#backend-ownership), derive partitions and logical memory, and return
`CompileArtifacts`. `AdvancedEngine` uses that port but keeps the artifacts [opaque](../../glossary.md#term-opaque). The ordinary
`Engine.standard()` [facade](../../glossary.md#term-facade) also keeps artifacts opaque while providing its fixed forward-only and
explicitly seeded first-order [compile](../../glossary.md#compile) overloads; typed [binding](../../glossary.md#term-binding), [publication](../../glossary.md#term-publication) [metadata](../../glossary.md#term-metadata), and explicit
[detached](../../glossary.md#detached-model-results) [host](../../glossary.md#term-host) [materialization](../../glossary.md#materialization) continue through the
[Runtime API](../runtime/engine-boundary.md#current-ordinary-engine-boundary)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>.

<a id="related-contracts"></a>
## Related contracts

- [Current Tensor and graph-model API](../tensor/index.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../glossary.md#tensor); [graph glossary](../../glossary.md#graph); [model glossary](../../glossary.md#term-model); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- [Lifecycle](../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Partition scoring](../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../glossary.md#partition))</span>
- [Compiling graphs user guide](../../user-guide/compiling-graphs.md)
- [Roadmap](../../planning/roadmap.md)

---

[Previous topic](memory-output.md)
