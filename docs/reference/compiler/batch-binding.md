<a id="compile-api"></a>
# Named-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [metadata](../../glossary.md#term-metadata) [binding](../../glossary.md#term-binding)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

A named [batch dimension](../../glossary.md#batch-dimension), written B, leaves the number of examples unspecified in a template.
**Metadata binding** supplies actual input descriptions and checks that every use of B agrees,
without reading the input's numbers. This differs from [runtime](../../glossary.md#term-runtime) input binding, which associates
usable stored values with the compiled input positions.

## Who binds the named dimension?

Engine's run path supplies actual **descriptors** to Compiler's `BatchBindingPlan`; Compiler
checks their shared batch size and deferred shape relationships before Engine borrows storage.
The plan retains immutable logical facts, not batches, arrays or a cached actual size. A new call
can bind another admitted size without mutating the original template. The number of examples
is a non-negative integer $B$; for fixed time $T=3$ and feature width $F=2$, a sequence descriptor
$[T,B,F]$ becomes $[3,40,2]$ at $B=40$. This is dimension substitution, not a copy or a transpose.
See [changing-batch execution](../engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle)
for the independently qualified numerical route.

<a id="ordinary-named-batch-metadata-binding"></a>
## Ordinary named-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [metadata](../../glossary.md#term-metadata) [binding](../../glossary.md#term-binding)

`BatchBindingPlan.prepare(artifacts)` indexes final compiled caller positions, typed [Shape](../../glossary.md#shape)
obligations and logical [layout](../../glossary.md#layout) intent without adding a component to `CompileArtifacts`.
`bind(actualDescriptors)` accepts fully static [descriptors](../../glossary.md#term-descriptor) in final caller-input order, derives
one named B from every declared [occurrence](../../glossary.md#term-occurrence), and checks all obligations before storage is touched.
An output-only/unknown [symbol](../../glossary.md#dimension), inconsistent B, negative derived [extent](../../glossary.md#term-extent) or arithmetic [overflow](../../glossary.md#term-overflow)
rejects. The plan and each returned `Binding` are [immutable](../../glossary.md#term-immutable) and storage-free.

`Binding.batchSize()` is actual non-negative B; `valueCount()` counts final value-list positions.
`descriptor(valueIndex)` lazily resolves only that value, preserving generated [gradient](../../glossary.md#term-gradient) eligibility
and source-[view](../../glossary.md#term-view-meanings) offsets/[strides](../../glossary.md#term-stride). Fixed caller descriptors remain exact. Symbolic caller positions
require present [canonical](../../glossary.md#term-canonical-form), zero-offset dense actual layouts; dense read views are legal.

Metadata binding does not grant [backend](../../glossary.md#backend) support for every representable expression. The ordinary
[Engine](../../glossary.md#term-engine)/[CPU](../../glossary.md#term-cpu) implementation is [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32), one named B, [rank](../../glossary.md#term-rank) at most two and fixed [features](../../glossary.md#term-feature)
and [parameters](../../glossary.md#term-parameter), including actual first-order [MATMUL](../../glossary.md#matrix-multiplication--matmul)/[MEAN](../../glossary.md#term-mean) backward. See the
[public workflow](../engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle) and
[CPU qualification](../../backend-guide/cpu-backend.md#ordinary-named-batch-cpu-execution).
[Symbolic Scan](../../glossary.md#symbolic-scan) metadata [admission](../../glossary.md#term-admission) is implemented by Compiler0014, independently reviewed and
MAIN accepted after compatibility [validation](../../glossary.md#term-validation). The separately implemented CPU execution slice is
homogeneous FLOAT32, generated [scalar](../../glossary.md#term-scalar) and single-range, with fixed time/features/state widths;
[Prepare0014](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md) tracks final [qualification](../../glossary.md#term-qualification).
[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training reuse, cold acquisition and symbolic wire remain separate gates.

For example, a compiled all-valid [Scan](../../glossary.md#general-differentiable-scan) with X `[3,B,2]` and state `[B,3]` binds actual B=40
to X `[3,40,2]` and state `[40,3]`. Every final/emission/[residual port](../../glossary.md#scan-residual-port-and-history-demand), even a dead sibling,
gets canonical non-view actual geometry. The original symbolic templates remain unresolved.
Each call checks every scoped declaration and typed obligation afresh, including repeated B.
Input-backed publications retain that call's supplied descriptor and `isView`; no binding is cached.


---

[Next topic](index.md)
