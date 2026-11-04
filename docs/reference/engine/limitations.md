# [CPU](../../glossary.md#term-cpu) limitations and development compatibility

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)

Read support at the exact workflow boundary: a valid expression can still fail preparation if
the selected [backend](../../glossary.md#backend) cannot implement it. A **resolved [layout](../../glossary.md#layout)** describes how logical [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix)
map to stored [elements](../../glossary.md#term-element); **[contiguous](../../glossary.md#term-contiguous)** means those elements follow a compact ordering. Neither
word alone proves every upstream [operation](../../glossary.md#operation) has a supported implementation.

<a id="current-cpu-limitations-and-planned-conveniences"></a>
## Current [CPU](../../glossary.md#term-cpu) limitations and planned conveniences

The fixed CPU-only and advanced CPU compositions prepare one non-empty maximal CPU partition.
Pure zero-node graphs remain rejected. Final operation geometry must satisfy the backend's
actual guards; qualified symbolic CPU execution is a separate
[named-batch workflow](lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle).

The focused examples use resolved input leaves and CONTIGUOUS to isolate binding and cleanup.
They are not an operation-support ceiling: Compiler now closes eligible fully static materialized
results, including ADD, before capability planning. An initially unresolved expression result
therefore need not remain unresolved in the compiled graph. Unproved geometry still rejects,
and adding `contiguous()` is not a general repair for unsupported upstream work. See
[final logical layout closure](../compiler/optimization.md) for that distinction and the
[current production matrix](../../backend-guide/production-support.md) for actual backend domains.

Source-only published callers/constants can accompany a nonempty partition under the existing
Prepare handoff; they do not make a zero-node graph executable. CPU supplies physical geometry,
Prepare assigns slots, and per-run creators initialize constants. See the
[ordinary Runtime examples](../runtime/engine-boundary.md#current-ordinary-engine-boundary)
for separate description, execution and detached-copy lifetimes.

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
