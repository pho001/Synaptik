# Internal lifecycle, failures and interpretation

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

<a id="current-internal-lifecycle-and-planned-public-failures"></a>
## Current internal lifecycle and planned public failures

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

The current internal entry rejects invalid shapes, data types, operations, graph structure,
publication roles, planning composition, unsatisfied hard capability, partitions, memory, source
roles, or diagnostics before returning an artifact. A no-hard-eligible occurrence becomes a
compiler-owned `IllegalStateException` with node index, ID, and operation-kind context, while the
Planning failure remains its cause. Provider runtime failures otherwise propagate unchanged.
The integration port preserves the current compiler and Planning failure classes and identities;
it performs no Engine exception translation. An ordinary-user Engine failure taxonomy remains
planned.

<a id="example-interpretation"></a>
## Example interpretation

For an explicitly composed graph containing a matrix multiplication and a small elementwise operation,
capability analysis may find both CPU and Metal valid. Backend-neutral scoring may assign both
nodes to Metal to avoid a transfer boundary. The artifact records only `owner = Metal`; it does
not record MPSGraph or a custom Metal kernel. Metal prepare makes that later choice.

Both CPU and Metal lifecycle integrations are current through explicit `Engine.builder()`
composition and exact occurrence ownership; no discovery, retry or CPU fallback is implied.
Current `GraphCompilationPort` can invoke package-private `GraphCompiler` to construct and
validate its bounded forward-only or combined first-order graph, query explicitly supplied
providers, select backend ownership, derive partitions and logical memory, and return
`CompileArtifacts`. `AdvancedEngine` uses that port but keeps the artifacts opaque. The ordinary
`Engine.standard()` facade also keeps artifacts opaque while providing its fixed forward-only and
explicitly seeded first-order compile overloads; typed binding, publication metadata, and explicit
detached host materialization continue through the
[Runtime API](../runtime/engine-boundary.md#current-ordinary-engine-boundary).

<a id="related-contracts"></a>
## Related contracts

- [Current Tensor and graph-model API](../tensor/index.md)
- [Lifecycle](../../architecture/lifecycle.md)
- [Partition scoring](../../architecture/partition-scoring.md)
- [Compiling graphs user guide](../../user-guide/compiling-graphs.md)
- [Roadmap](../../planning/roadmap.md)

---

[Previous topic](memory-output.md)
