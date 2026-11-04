<a id="compile-api"></a>
# Named-batch metadata binding

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="ordinary-named-batch-metadata-binding"></a>
## Ordinary named-batch metadata binding

`BatchBindingPlan.prepare(artifacts)` indexes final compiled caller positions, typed Shape
obligations and logical layout intent without adding a component to `CompileArtifacts`.
`bind(actualDescriptors)` accepts fully static descriptors in final caller-input order, derives
one named B from every declared occurrence, and checks all obligations before storage is touched.
An output-only/unknown symbol, inconsistent B, negative derived extent or arithmetic overflow
rejects. The plan and each returned `Binding` are immutable and storage-free.

`Binding.batchSize()` is actual non-negative B; `valueCount()` counts final value-list positions.
`descriptor(valueIndex)` lazily resolves only that value, preserving generated gradient eligibility
and source-view offsets/strides. Fixed caller descriptors remain exact. Symbolic caller positions
require present canonical, zero-offset dense actual layouts; dense read views are legal.

Metadata binding does not grant backend support for every representable expression. The ordinary
Engine/CPU implementation is homogeneous FLOAT32, one named B, rank at most two and fixed features
and parameters, including actual first-order MATMUL/MEAN backward. See the
[public workflow](../engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle) and
[CPU qualification](../../backend-guide/cpu-backend.md#ordinary-named-batch-cpu-execution).
Symbolic Scan metadata admission is implemented by Compiler0014, independently reviewed and
MAIN accepted after compatibility validation. The separately implemented CPU execution slice is
homogeneous FLOAT32, generated scalar and single-range, with fixed time/features/state widths;
[Prepare0014](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md) tracks final qualification.
NN/Training reuse, cold acquisition and symbolic wire remain separate gates.

For example, a compiled all-valid Scan with X `[3,B,2]` and state `[B,3]` binds actual B=40
to X `[3,40,2]` and state `[40,3]`. Every final/emission/residual port, even a dead sibling,
gets canonical non-view actual geometry. The original symbolic templates remain unresolved.
Each call checks every scoped declaration and typed obligation afresh, including repeated B.
Input-backed publications retain that call's supplied descriptor and `isView`; no binding is cached.


---

[Next topic](index.md)
