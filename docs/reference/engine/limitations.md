# CPU limitations and development compatibility

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="current-cpu-limitations-and-planned-conveniences"></a>
## Current CPU limitations and planned conveniences

The fixed CPU-only and advanced CPU compositions prepare exactly one non-empty maximal CPU
partition. Zero-node pass-through and direct multi-partition artifacts are rejected. Inputs and
every CPU operation occurrence require fully static Shapes and resolved compatible layouts;
adding `contiguous()` after an operation does not retroactively resolve that operation's inputs or
output. A compile success therefore does not guarantee CPU preparation or execution success.

The focused supported examples use `CONTIGUOUS` directly over resolved `FLOAT32` leaves. The
seeded example reuses `seedLeaf.contiguous()` as the explicit seed for two outputs. It does not use
`ADD`, whose current public expression result has unresolved layout. The current CPU integration
does support a fully static canonical source-only published splat when it accompanies the required
non-empty CPU partition: CPU supplies its physical declaration, shared Prepare assigns it, and
each run initializes a distinct run-owned representation before binding and schedule traversal.
Pure constant graphs remain unsupported because they have no non-empty partition. See the
[Runtime API ordinary examples](../runtime/engine-boundary.md#current-ordinary-engine-boundary) for the complete
setup.

Ordinary Engine host materialization is current through the CPU integration's canonical copy.
Engine selects and authenticates the exact publication occurrence, serializes the synchronous
copy against result closure, and returns detached immutable canonical bytes. The CPU operation
itself still neither selects a publication nor obtains a representation, performs a transfer,
synchronizes result closure, constructs a Tensor or `HostTensorStorage`, streams a payload, or
generalizes the format across backends. Engine-owned one-shot forward execution is current through
the four ordinary `Engine.compute(...)` overloads described above; it does not add an
`output.execute()` Tensor method. Engine-owned one-shot scalar-objective backward execution is
current through the explicit-target `Engine.backward(...)` method described above. It adds no
no-argument backward contract, implicit targets, or Tensor mutation.

<a id="compatibility-expectations-during-development"></a>
## Compatibility expectations during development

- Treat Javadoc and implemented tests as the contract for code that exists.
- Treat architecture snippets as conceptual unless a reference page explicitly marks them current.
- Do not depend on draft planning types or package names in external code.
- Check the [roadmap](../../planning/roadmap.md) before assuming a planned module is available.

<a id="related-documentation"></a>
## Related documentation

- [Getting started](../../getting-started.md)
- [Architecture overview](../../architecture/overview.md)
- [Implementation roadmap](../../planning/roadmap.md)

---

[Previous topic](composition.md)
