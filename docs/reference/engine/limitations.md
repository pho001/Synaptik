# [CPU](../../glossary.md#term-cpu) limitations and development compatibility

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)

Read support at the exact workflow boundary: a valid expression can still fail preparation if
the selected [backend](../../glossary.md#backend) cannot implement it. A **resolved [layout](../../glossary.md#layout)** describes how logical [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix)
map to stored [elements](../../glossary.md#term-element); **[contiguous](../../glossary.md#term-contiguous)** means those elements follow a compact ordering. Neither
word alone proves every upstream [operation](../../glossary.md#operation) has a supported implementation.

<a id="current-cpu-limitations-and-planned-conveniences"></a>
## Current [CPU](../../glossary.md#term-cpu) limitations and planned conveniences

The fixed CPU-only and advanced CPU compositions [prepare](../../glossary.md#prepare) exactly one non-empty maximal CPU
[partition](../../glossary.md#partition). Zero-[node](../../glossary.md#node) pass-through and direct multi-partition [artifacts](../../glossary.md#term-artifact) are rejected. Inputs and
every CPU [operation](../../glossary.md#operation) [occurrence](../../glossary.md#term-occurrence) require fully static [Shapes](../../glossary.md#shape) and resolved compatible [layouts](../../glossary.md#layout);
adding `contiguous()` after an operation does not retroactively resolve that operation's inputs or
output. A [compile](../../glossary.md#compile) success therefore does not guarantee CPU preparation or execution success.

The focused supported examples use [`CONTIGUOUS`](../../glossary.md#term-contiguous) directly over resolved [`FLOAT32`](../../glossary.md#term-float32) leaves. The
seeded example reuses `seedLeaf.contiguous()` as the explicit [seed](../../glossary.md#term-seed) for two outputs. It does not use
`ADD`, whose current public expression result has unresolved layout. The current CPU integration
does support a fully static [canonical](../../glossary.md#term-canonical-form) source-only published splat when it accompanies the required
non-empty CPU partition: CPU supplies its physical declaration, shared Prepare assigns it, and
each [run](../../glossary.md#term-run) initializes a distinct run-owned representation before [binding](../../glossary.md#term-binding) and schedule traversal.
Pure [constant](../../glossary.md#term-constant) [graphs](../../glossary.md#graph) remain unsupported because they have no non-empty partition. See the
[Runtime API ordinary examples](../runtime/engine-boundary.md#current-ordinary-engine-boundary)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [API glossary](../../glossary.md#term-application-programming-interface-api))</span> for the complete
setup.

Ordinary [Engine](../../glossary.md#term-engine) [host](../../glossary.md#term-host) [materialization](../../glossary.md#materialization) is current through the CPU integration's canonical copy.
Engine selects and authenticates the exact [publication occurrence](../../glossary.md#publication-occurrence), serializes the [synchronous](../../glossary.md#term-synchronous)
copy against result closure, and returns [detached](../../glossary.md#detached-model-results) [immutable](../../glossary.md#term-immutable) canonical bytes. The CPU operation
itself still neither selects a [publication](../../glossary.md#term-publication) nor obtains a representation, performs a transfer,
synchronizes result closure, constructs a [Tensor](../../glossary.md#tensor) or `HostTensorStorage`, streams a [payload](../../glossary.md#artifact-envelope-and-atomic-publication), or
generalizes the format across [backends](../../glossary.md#backend). Engine-owned [one-shot forward execution](../../glossary.md#one-shot-forward-execution) is current through
the four ordinary `Engine.compute(...)` overloads described above; it does not add an
`output.execute()` Tensor method. Engine-owned [one-shot scalar-objective backward execution](../../glossary.md#one-shot-scalar-objective-backward-execution) is
current through the explicit-[target](../../glossary.md#term-target) `Engine.backward(...)` method described above. It adds no
no-[argument](../../glossary.md#term-argument) backward contract, implicit targets, or Tensor mutation.

<a id="compatibility-expectations-during-development"></a>
## Compatibility expectations during development

- Treat [Javadoc](../../glossary.md#term-javadoc) and implemented tests as the contract for code that exists.
- Treat architecture snippets as conceptual unless a reference page explicitly marks them current.
- Do not depend on draft [planning](../../glossary.md#term-planning) types or package names in external code.
- Check the [roadmap](../../planning/roadmap.md) before assuming a planned [module](../../glossary.md#term-module) is available.

<a id="related-documentation"></a>
## Related documentation

- [Getting started](../../getting-started.md)
- [Architecture overview](../../architecture/overview.md)
- [Implementation roadmap](../../planning/roadmap.md)

---

[Previous topic](composition.md)
