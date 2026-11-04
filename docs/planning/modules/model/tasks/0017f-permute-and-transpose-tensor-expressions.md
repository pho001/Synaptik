# Task 0017F: Permute and [Transpose](../../../../glossary.md#term-transpose) [Tensor](../../../../glossary.md#tensor) Expressions

## Status

Complete

## Goal

Add public, storage-free [Tensor](../../../../glossary.md#tensor) expressions for arbitrary complete [axis](../../../../glossary.md#term-axis) [permutation](../../../../glossary.md#permutation) and [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
[transpose](../../../../glossary.md#term-transpose) convenience using the completed `PERMUTE` semantics.

Permutation normalizes caller axes against the input [rank](../../../../glossary.md#term-rank), reorders exact [Dimension](../../../../glossary.md#dimension) references and
resolved [strides](../../../../glossary.md#term-stride) in output-to-input order, preserves logical values and [storage offset](../../../../glossary.md#term-storage-offset) [metadata](../../../../glossary.md#term-metadata),
and records one-input [provenance](../../../../glossary.md#provenance). `transpose()` is exactly a rank-two convenience for permutation
`[1, 0]`, not a separate semantic kind.

This task does not implement [singleton](../../../../glossary.md#term-singleton)-axis insertion/removal. Those distinct rank-editing rules
remain in task 0017F1.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly `Tensor.permute(int... requestedAxes)` and parameterless `Tensor.transpose()`.
- Make each public method delegate exactly once to its package-private helper entry.
- Add one package-private final field-free `TensorPermutationExpressions` helper with exactly the
  six methods specified under Required contract and one private constructor.
- Defensively copy the caller's raw [permutation](../../../../glossary.md#permutation) and never retain or mutate the caller array.
- Require raw [axis](../../../../glossary.md#term-axis) count to equal input [rank](../../../../glossary.md#term-rank).
- Normalize each negative raw axis by adding input rank once.
- Require the normalized axes to be an exact complete permutation with no duplicate.
- Accept empty axes for [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar) [identity](../../../../glossary.md#term-identity-meanings) permutation.
- Require [transpose](../../../../glossary.md#term-transpose) input rank exactly two and use normalized `[1, 0]` PERMUTE semantics.
- Build result [Shape](../../../../glossary.md#shape) by reordering the exact [immutable](../../../../glossary.md#term-immutable) input [Dimension](../../../../glossary.md#dimension) references.
- For any resolved input layout, build one new view layout by reordering exact input strides and
  preserving storage offset.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [strides glossary](../../../../glossary.md#term-stride); [storage offset glossary](../../../../glossary.md#term-storage-offset))</span>
- Leave [layout](../../../../glossary.md#layout) unresolved when input layout is unresolved, including dynamic input Shapes.
- Preserve exact input [data type](../../../../glossary.md#data-type--datatype) and [gradient](../../../../glossary.md#term-gradient) eligibility.
- Construct exact `AxisTransformKind.PERMUTE`, immutable normalized `PermutationAttrs`, and ordered
  provenance <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>`[input]`.
- Delegate once to `TensorFactory.createDerived` with no label or [host storage](../../../../glossary.md#host-storage).
- Return a fresh [Tensor](../../../../glossary.md#tensor) for identity, transpose, repeated, inverse, and nested requests.
- Add one focused same-package test and update Tensor's exact public method inventory.
- Finalize affected Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and
  roadmap through the mandatory independent documentation pass.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- `expandDims`, `squeeze`, singleton insertion/removal, inserted-axis stride policy, or task-0017F1
  implementation/specification<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [stride glossary](../../../../glossary.md#term-stride))</span>
- another Tensor method, static facade, factory, builder, permutation list overload, two-axis
  transpose overload, matrix argument, or <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [facade glossary](../../../../glossary.md#term-facade); [permutation glossary](../../../../glossary.md#permutation); [transpose glossary](../../../../glossary.md#term-transpose); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [argument glossary](../../../../glossary.md#term-argument))</span>`transpose(int, int)`
- another operation kind/attributes type, TRANSPOSE kind, modification of AxisTransformKind,
  PermutationAttrs, AxisTransformAttrs, or generic Operation validation<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [validation glossary](../../../../glossary.md#term-validation))</span>
- changing Shape, Dimension, LayoutDescriptor, LayoutKind, TensorDescriptor, TensorFactory,
  TensorProvenance, Tensor fields/constructor/equality/storage behavior, or completed expressions<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension))</span>
- named axes, partial permutations, repeated axes, ellipsis, automatic missing axes, sorting,
  inverse creation, or multi-output behavior
- returning input for identity, collapsing inverse/nested permutations, caching, interning, CSE,
  or canonicalization<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- reading/copying values, looking up or attaching host storage, physical aliasing, allocation,
  materialization, mutation, capacity/lifetime validation, or zero-copy execution guarantees<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [aliasing glossary](../../../../glossary.md#term-aliasing); [materialization glossary](../../../../glossary.md#materialization); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- gradient inverse, autograd, training behavior, graph capture, compiler passes, planning,
  prepare, backend lowering, runtime, execution, ONNX mapping, or conformance<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- another module, dependency, Gradle/build change, preview feature, architecture change, or later
  task specification<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [preview feature glossary](../../../../glossary.md#term-preview-feature))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0017E](0017e-axis-transform-semantics.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes arbitrary `Tensor.permute(int...)` with negative
[axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization) and complete-[permutation](../../../../glossary.md#permutation) [validation](../../../../glossary.md#term-validation). Its parameterless `transpose()` requires
[rank](../../../../glossary.md#term-rank) two and calls `permute(1, 0)`. Permutation reorders [Shape](../../../../glossary.md#shape) and [stride](../../../../glossary.md#term-stride) entries, preserves [storage
offset](../../../../glossary.md#term-storage-offset), produces a logical [view](../../../../glossary.md#term-view-meanings), supports [identity](../../../../glossary.md#term-identity-meanings) permutations, and defines inverse permutation
for [gradients](../../../../glossary.md#term-gradient).

This task preserves the selected [model](../../../../glossary.md#term-model)-level capability: raw normalization, result Shape and
logical [layout](../../../../glossary.md#layout) [metadata](../../../../glossary.md#term-metadata), exact PERMUTE attributes, data-type/eligibility retention, and [provenance](../../../../glossary.md#provenance).
It excludes legacy immediate storage [aliasing](../../../../glossary.md#term-aliasing), [graph](../../../../glossary.md#graph) builders, gradient callbacks, [mutable](../../../../glossary.md#term-mutable) arrays,
[operation](../../../../glossary.md#operation) traits, [compiler](../../../../glossary.md#term-compiler)/[lowering](../../../../glossary.md#lowering) code, [kernels](../../../../glossary.md#kernel), and [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) execution.

## Architecture constraints

- [Tensor](../../../../glossary.md#tensor) is public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and not an [IR](../../../../glossary.md#term-intermediate-representation-ir) [node](../../../../glossary.md#node).
- [Operation](../../../../glossary.md#operation) owns [backend](../../../../glossary.md#backend)-neutral meaning and no support, cost, route, or executable behavior.
- Shape and LayoutDescriptor are immutable model metadata; resolved view layout does not prove
  physical storage aliasing or executable zero-copy support.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [immutable glossary](../../../../glossary.md#term-immutable); [metadata glossary](../../../../glossary.md#term-metadata); [view glossary](../../../../glossary.md#term-view-meanings); [layout glossary](../../../../glossary.md#layout); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- [Model](../../../../glossary.md#term-model) performs only local [rank](../../../../glossary.md#term-rank)/[permutation](../../../../glossary.md#permutation) [normalization](../../../../glossary.md#term-normalization) and [descriptor](../../../../glossary.md#term-descriptor) derivation.
- [Compiler](../../../../glossary.md#term-compiler) owns [graph capture](../../../../glossary.md#capture-graph-ports-and-reconstruction) and permutation [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).
- [Planning](../../../../glossary.md#term-planning)/backend [prepare](../../../../glossary.md#prepare) own [materialization](../../../../glossary.md#materialization) and concrete [alias](../../../../glossary.md#aliasing-and-arena)/copy [lowering](../../../../glossary.md#lowering).
- [Runtime](../../../../glossary.md#term-runtime) executes prepared schedules and manages physical storage/[residency](../../../../glossary.md#residency).
- No runtime [service locator](../../../../glossary.md#term-service-locator), backend [lookup](../../../../glossary.md#term-lookup), [dependency](../../../../glossary.md#term-dependency), or architecture change is authorized.

## Package impact

No package is added or moved.

- Public methods remain in `io.github.pho001.synaptik.model.tensor.Tensor`.
- Package-private `TensorPermutationExpressions` lives in `model.tensor` beside existing
  expression helpers.
- Focused `TensorPermutationExpressionTest` mirrors that package for helper inspection.
- Existing shape, layout, operation-layout, descriptor, provenance, and factory packages are only
  consumed and remain unchanged.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>

## Required contract

### Public [Tensor](../../../../glossary.md#tensor) surface

Add exactly:

```java
public Tensor permute(int... requestedAxes) {
    return TensorPermutationExpressions.apply(this, requestedAxes);
}

public Tensor transpose() {
    return TensorPermutationExpressions.transpose(this);
}
```

Do not add another [permutation](../../../../glossary.md#permutation)/[transpose](../../../../glossary.md#term-transpose) overload or [alias](../../../../glossary.md#aliasing-and-arena).

### Helper structure

`TensorPermutationExpressions` is package-private, final, field-free, and non-instantiable. Apart
from one private zero-[argument](../../../../glossary.md#term-argument) constructor, it declares exactly:

```java
static Tensor apply(Tensor input, int[] requestedAxes)
static Tensor transpose(Tensor input)
private static int[] normalizePermutation(int rank, int[] requestedAxes)
private static Shape permuteShape(Shape inputShape, int[] normalizedAxes)
private static Optional<LayoutDescriptor> resolveViewLayout(
        TensorDescriptor inputDescriptor, Shape resultShape, int[] normalizedAxes)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape resultShape,
        int[] normalizedAxes,
        Optional<LayoutDescriptor> resultLayout)
```

Do not add fields, nested types, another constructor/helper, or hidden overload.

### Permute [validation](../../../../glossary.md#term-validation) and [normalization](../../../../glossary.md#term-normalization)

`apply` performs exactly this order:

1. null-check input with exact message `input`;
2. null-check requestedAxes with exact message `requestedAxes`;
3. read exact input [descriptor](../../../../glossary.md#term-descriptor) once and exact input [Shape](../../../../glossary.md#shape) once;
4. normalize and validate a private copy of the raw [axes](../../../../glossary.md#term-axis);
5. derive result Shape;
6. resolve or defer result [layout](../../../../glossary.md#layout);
7. perform common construction once.

`normalizePermutation` first checks axis count. On mismatch throw `IllegalArgumentException`:

```text
permutation axis count <count> must equal input rank <rank>
```

Clone `requestedAxes` exactly once after the count check. Inspect copied entries in ascending index
order. For each raw axis, calculate normalization in `long`: a negative value adds [rank](../../../../glossary.md#term-rank) once; a
non-negative value is unchanged. If normalized value is outside `[0, rank)`, throw:

```text
permutation axis <rawValue> at index <index> is outside rank <rank>
```

Then reject the first duplicate [normalized axis](../../../../glossary.md#normalized-axis) with:

```text
permutation contains duplicate normalized axis <axis> at index <index>
```

Store normalized values into the private copy and return it. [Rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) plus empty axes is valid.
Do not mutate/retain caller input, sort, fill missing axes, normalize repeatedly, or create an
inverse.

### [Transpose](../../../../glossary.md#term-transpose) [validation](../../../../glossary.md#term-validation)

`transpose` null-checks input with message `input`, reads its [rank](../../../../glossary.md#term-rank), and requires exactly two. Any
other rank throws `IllegalStateException` with exact message:

```text
transpose() requires rank-2 tensor, got rank=<rank>
```

For rank two, delegate to the common `apply` path with private fixed [axes](../../../../glossary.md#term-axis) `[1, 0]`. The resulting
[operation](../../../../glossary.md#operation) is PERMUTE with `PermutationAttrs(List.of(1, 0))`; no transpose kind exists.

### Result [Shape](../../../../glossary.md#shape)

Allocate one `Dimension[]` with input [rank](../../../../glossary.md#term-rank). For each output [axis](../../../../glossary.md#term-axis) `i`, store the exact [immutable](../../../../glossary.md#term-immutable)
reference `inputShape.dimensions().get(normalizedAxes[i])`. Construct through
`Shape.ofDimensions(resultDimensions)`.

This preserves [Dimension](../../../../glossary.md#dimension) [identity](../../../../glossary.md#term-identity-meanings)/order and supports static, zero, mixed dynamic, fully dynamic,
and [scalar](../../../../glossary.md#term-scalar) Shapes. Identity [permutation](../../../../glossary.md#permutation) remains an explicit new Shape for non-scalar ranks; scalar
construction may return [canonical](../../../../glossary.md#term-canonical-form) `Shape.scalar()`.

### Resolved and unresolved [layout](../../../../glossary.md#layout)

Read `inputDescriptor.layout()` exactly once. If absent, return `Optional.empty()` without guessing
geometry. A present layout implies static input/result geometry.

For a present layout, allocate one `long[]` with result [rank](../../../../glossary.md#term-rank) and assign:

```text
resultStrides[i] = inputLayout.stride(normalizedAxes[i])
```

Create exactly one:

```java
LayoutDescriptor.of(
        resultShape,
        resultStrides,
        inputLayout.storageOffset(),
        true)
```

Accept every resolved input layout kind, preserve exact offset and raw [stride](../../../../glossary.md#term-stride) values in permuted
order, and always mark result as a [view](../../../../glossary.md#term-view-meanings). Let LayoutDescriptor derive kind/span from result geometry.
Never reuse the input layout object. Resolved geometry attaches no storage and promises no physical
[alias](../../../../glossary.md#aliasing-and-arena) or zero-copy execution.

### Result construction

Common construction builds an [immutable](../../../../glossary.md#term-immutable) boxed list from [normalized axis](../../../../glossary.md#normalized-axis) values in exact order and
then creates:

```java
PermutationAttrs attrs = new PermutationAttrs(normalizedAxesList);
TensorDescriptor descriptor = new TensorDescriptor(
        inputDescriptor.dataType(),
        resultShape,
        resultLayout,
        inputDescriptor.requiresGrad());
Operation operation = new Operation(AxisTransformKind.PERMUTE, attrs);
TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
```

Use no production factory/helper beyond the exact class. Every successful call consumes one fresh
[Tensor](../../../../glossary.md#tensor) ID and has absent label/storage. [Validation](../../../../glossary.md#term-validation)/[layout](../../../../glossary.md#layout) failures consume no ID; exhaustion occurs
only at final factory delegation.

## Affected files

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPermutationExpressions.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorPermutationExpressionTest.java`
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/tasks/0017f-permute-and-transpose-tensor-expressions.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most the ten paths above may change. Production is limited to [Tensor](../../../../glossary.md#tensor) and one package-private
helper; tests to Tensor inventory and one focused suite; documentation to public [API](../../../../glossary.md#term-application-programming-interface-api)/status and
[planning](../../../../glossary.md#term-planning) evidence.

If another type, helper, overload, existing contract edit, [dependency](../../../../glossary.md#term-dependency), build/architecture change,
or eleventh path is needed, stop and report. Do not create task 0017F1 or 0017G.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Fully document both public methods and the helper type/constructor/all six methods.
- Explain output-to-input [axis](../../../../glossary.md#term-axis) order and negative raw-axis [normalization](../../../../glossary.md#term-normalization).
- Document caller-array ownership, scalar empty permutation, full-permutation constraints, exact
  failures, and transpose rank-two behavior.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [permutation glossary](../../../../glossary.md#permutation); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Explain Dimension-reference preservation, stride reordering, offset/view metadata, unresolved
  dynamic geometry, and all accepted resolved layout kinds.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [stride glossary](../../../../glossary.md#term-stride); [layout glossary](../../../../glossary.md#layout))</span>
- Document result type, Shape, layout, eligibility, label, storage, operation, attributes,
  provenance, freshness, failures, and identifier exhaustion.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
- State that [transpose](../../../../glossary.md#term-transpose) is PERMUTE `[1, 0]`, not another kind.
- Explain why resolved [view](../../../../glossary.md#term-view-meanings) [metadata](../../../../glossary.md#term-metadata) is neither attached storage nor an execution guarantee.
- Independently review related axis semantic, Shape, layout, descriptor, factory, provenance,
  reshape/expand/contiguous Javadocs; stop on an out-of-scope inconsistency.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [contiguous glossary](../../../../glossary.md#term-contiguous); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- [Tensor](../../../../glossary.md#tensor) exposes exactly `permute(int...)` and parameterless [`transpose`](../../../../glossary.md#term-transpose); public method inventory
  rises from 81 to 83.
- Each public method has one helper invocation; helper has exact class/constructor/six-method [shape](../../../../glossary.md#shape).
- Raw array ownership, validation order, negative normalization, messages, duplicate handling, and
  scalar identity match this specification.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Transpose accepts exactly [rank](../../../../glossary.md#term-rank) two and records normalized `[1, 0]` PERMUTE semantics.
- Result Shape reorders exact [Dimension](../../../../glossary.md#dimension) references for static/dynamic/zero/[scalar](../../../../glossary.md#term-scalar) inputs.
- Every resolved layout kind derives correct permuted strides, exact offset, view flag, kind, and
  referenced span; unresolved input stays unresolved.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings); [referenced span glossary](../../../../glossary.md#referenced-element-span))</span>
- All six data types and valid eligibility states retain exact [metadata](../../../../glossary.md#term-metadata).
- Result has exact PERMUTE/PermutationAttrs/[input], no label/storage, and fresh [identity](../../../../glossary.md#term-identity-meanings).
- Identity, repeated, inverse, and nested calls remain explicit and uncanonicalized.
- Early failures consume no ID; final exhaustion behavior remains intact.
- No values/storage/cross-layer behavior, new dependencies, or architecture change.
- Independent documentation review and all validation/status synchronization complete before
  marking Complete.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorPermutationExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact public/helper [API](../../../../glossary.md#term-application-programming-interface-api); all data types/eligibility; caller [ownership](../../../../glossary.md#term-ownership-meanings); [scalar](../../../../glossary.md#term-scalar),
static, zero, mixed/fully dynamic [Shapes](../../../../glossary.md#shape); [identity](../../../../glossary.md#term-identity-meanings)/general/negative permutations; every exact
failure/precedence including extreme integers and normalized duplicates; [transpose](../../../../glossary.md#term-transpose) success/error;
[Dimension](../../../../glossary.md#dimension) identity; dense/offset/strided/broadcast [layouts](../../../../glossary.md#layout); permuted [strides](../../../../glossary.md#term-stride)/offset/kind/[view](../../../../glossary.md#term-view-meanings)/span;
unresolved geometry; [operation](../../../../glossary.md#operation)/attributes/[provenance](../../../../glossary.md#provenance); absent label/storage; dead-storage
non-interference; freshness; ID side effects; and exhaustion.

Inspect `javap -p -c -s`, reflection, imports, and source. Confirm two one-call public methods,
exact six-method stateless helper, one [normalization](../../../../glossary.md#term-normalization)/copy path, one Shape reorder, one layout
reorder, one common construction, and no expandDims/squeeze/value/storage/cross-layer behavior.
Validate generated [Javadoc](../../../../glossary.md#term-javadoc), executable examples, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) API and glossary, links/anchors/
fences/whitespace, exact ten paths, synchronized status, and no task-0017F1/0017G spec.

## Dependencies

- 0002 supplies [immutable](../../../../glossary.md#term-immutable) static/dynamic [Dimension](../../../../glossary.md#dimension) and [Shape](../../../../glossary.md#shape) contracts.
- 0003 supplies resolved [stride](../../../../glossary.md#term-stride)/offset/[view](../../../../glossary.md#term-view-meanings) geometry and classification.
- 0007 supplies resolved-or-unresolved TensorDescriptor.
- 0011–0013 supply [Tensor](../../../../glossary.md#tensor), centralized derived [identity](../../../../glossary.md#term-identity-meanings), and [provenance](../../../../glossary.md#provenance).
- 0017E supplies PERMUTE and immutable normalized PermutationAttrs.
- Completed contiguous/reshape/expand tasks provide adjacent view-expression patterns but no hard
  production dependency.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Follow-up tasks

- 0017F1 remains Draft for expandDims/squeeze expressions, axis normalization, singleton checks,
  rank-editing Shape derivation, and inserted/removed stride geometry.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [singleton glossary](../../../../glossary.md#term-singleton); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [stride glossary](../../../../glossary.md#term-stride))</span>
- 0017G remains Draft for [slice](../../../../glossary.md#slice) semantics.
- [Compiler](../../../../glossary.md#term-compiler) later owns [identity](../../../../glossary.md#term-identity-meanings)/inverse/nested [permutation](../../../../glossary.md#permutation) [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form).
- [Planning](../../../../glossary.md#term-planning)/[backend](../../../../glossary.md#backend) [prepare](../../../../glossary.md#prepare) later own [materialization](../../../../glossary.md#materialization) and concrete [view](../../../../glossary.md#term-view-meanings)/copy [lowering](../../../../glossary.md#lowering).
- Training/compiler-generated semantics later own inverse permutation [gradients](../../../../glossary.md#term-gradient).

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. [Model](../../../../glossary.md#term-model) already owns [Tensor](../../../../glossary.md#tensor) expression construction, [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout) [metadata](../../../../glossary.md#term-metadata),
[operation](../../../../glossary.md#operation) semantics, [descriptors](../../../../glossary.md#term-descriptor), and [provenance](../../../../glossary.md#provenance). Later [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings) is unchanged.

Stop if implementation needs storage [aliasing](../../../../glossary.md#term-aliasing), [gradients](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler)/[planning](../../../../glossary.md#term-planning)/[backend](../../../../glossary.md#backend) behavior,
another [dependency](../../../../glossary.md#term-dependency), or architecture change.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0017E/0017F, Tensor API,
Compile API, Training API, glossary, current Dimension/Shape/LayoutDescriptor/LayoutKind/
TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/AxisTransformKind/
PermutationAttrs contracts and focused expression tests, and Java 26 Gradle configuration.

Implement task 0017F exactly. Modify Tensor.java and add package-private final
TensorPermutationExpressions.java. Update TensorTest only for exact two-method API expansion and
add TensorPermutationExpressionTest. Add exactly permute(int...) and transpose().

The helper has exactly six specified methods. Permute defensively copies raw axes, requires count
equal rank, normalizes each negative once, rejects out-of-range/duplicates, and accepts empty scalar
permutation. Transpose requires rank two and uses PERMUTE [1,0]. Reorder exact Dimension references
and, for any resolved input layout, exact strides while preserving offset in a new view; unresolved
input stays unresolved. Preserve type/eligibility, create exact PermutationAttrs/PERMUTE/[input],
and call createDerived once with no label/storage. Every call is fresh.

Do not implement expandDims/squeeze, modify semantic/foundational contracts, inspect/copy values or
storage, attach physical aliases, canonicalize, add APIs/helpers/types, define gradients, capture
graphs, or add compiler/planning/prepare/runtime/backend behavior, dependencies, build/architecture
changes, or later specs. Stop beyond ten paths or on architecture uncertainty.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record no-change conclusions, and rerun validation.

Update task 0017F, model master plan, and roadmap only for planning status/evidence. Do not mark
Complete until both passes succeed. Leave 0017F1 Draft without a specification. Do not commit/push.
```

## Local decisions

- The former combined expression row is split: 0017F owns permutation/transpose; 0017F1 owns
  singleton insertion/removal. Their normalization and layout algebra differ materially.<span data-glossary-navigation="true"> ([permutation glossary](../../../../glossary.md#permutation); [singleton glossary](../../../../glossary.md#term-singleton); [normalization glossary](../../../../glossary.md#term-normalization); [layout glossary](../../../../glossary.md#layout))</span>
- [Transpose](../../../../glossary.md#term-transpose) is a [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) convenience over the same PERMUTE construction path and exact `[1, 0]`
  attributes, not a separate operation identity.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Negative [axes](../../../../glossary.md#term-axis) are normalized once in long arithmetic to avoid integer [overflow](../../../../glossary.md#term-overflow) ambiguity.
- Result [Shape](../../../../glossary.md#shape) preserves exact [Dimension](../../../../glossary.md#dimension) references in normalized output-to-input order.
- Every resolved layout kind supports direct stride permutation; no contiguity requirement or
  materialization is inserted.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [contiguity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [materialization glossary](../../../../glossary.md#materialization))</span>
- Resolved result is [view](../../../../glossary.md#term-view-meanings) [metadata](../../../../glossary.md#term-metadata) without attached storage or zero-copy execution promise.
- Identity and inverse/nested permutations remain fresh explicit expressions; compiler owns
  canonicalization.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>

## Known limitations

- No expandDims/squeeze until task 0017F1.
- No storage alias, gradient inverse, compiler capture/canonicalization, planning materialization,
  backend lowering, runtime execution, ONNX mapping, or conformance behavior.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [planning glossary](../../../../glossary.md#term-planning); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed architecture/focused boundary docs; documentation/planning rules; roadmap; [model](../../../../glossary.md#term-model)
capabilities/master plan; prerequisite tasks; current [Dimension](../../../../glossary.md#dimension)/[Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), [descriptor](../../../../glossary.md#term-descriptor),
[Tensor](../../../../glossary.md#tensor)/factory/[provenance](../../../../glossary.md#provenance), [Operation](../../../../glossary.md#operation), AxisTransformKind/PermutationAttrs source/tests; neighboring
expression helpers/tests; Tensor/[Compile](../../../../glossary.md#compile)/Training APIs, glossary, and Java 26 [Gradle](../../../../glossary.md#term-gradle).

The read-only legacy branch confirms arbitrary and negative-[axis](../../../../glossary.md#term-axis) permutations, [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [transpose](../../../../glossary.md#term-transpose)
delegation to `[1, 0]`, Shape/[stride](../../../../glossary.md#term-stride) reordering, offset-preserving [view](../../../../glossary.md#term-view-meanings) behavior, [identity](../../../../glossary.md#term-identity-meanings)
[permutation](../../../../glossary.md#permutation), and inverse-[gradient](../../../../glossary.md#term-gradient) capability. Coupled storage, [graph](../../../../glossary.md#graph), gradient, [compiler](../../../../glossary.md#term-compiler), [kernel](../../../../glossary.md#kernel),
[runtime](../../../../glossary.md#term-runtime), and [backend](../../../../glossary.md#backend) design is excluded.

Planning split the broad expression row because permutation/transpose and [singleton](../../../../glossary.md#term-singleton) [rank](../../../../glossary.md#term-rank) editing
have different [normalization](../../../../glossary.md#term-normalization), Shape, layout, and validation rules. Existing contracts support 0017F
without another public type, package, [dependency](../../../../glossary.md#term-dependency), foundational edit, or architecture decision.
Planning validation is recorded after [synchronization](../../../../glossary.md#term-synchronization); implementation/documentation evidence is
empty until execution.

Planning validation after synchronization:

- `git diff --check` passed, and targeted trailing-whitespace scans found no matches.
- Exact planning scope is three paths: this task, model master plan, and roadmap. No Java, Gradle,
  architecture, API, glossary, completed-task, or other-module file changed during planning.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>
- All 20 [canonical](../../../../glossary.md#term-canonical-form) task sections are present.
- Markdown backtick fences are balanced: twenty-two in this task, two in the master plan, and zero
  in the roadmap.
- All 206 local Markdown file links across the three changed planning files resolve.
- Every changed file ends with a newline.
- Task, master plan, and roadmap consistently identify 0017F as Ready; 0017F1 and 0017G remain
  Draft, and neither has a detailed specification.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- The model task sequence contains 74 ordered rows with no duplicate order number after inserting
  0017F1.
- Package review confirms no new package: public methods and helper remain in `model.tensor`, and
  the focused test mirrors that package.
- Scope review confirms exactly ten permitted implementation paths and no need for another public
  type, dependency, foundational edit, or architecture decision.
- Granularity review records why permutation/transpose is isolated from singleton rank editing;
  task 0017F1 remains a concise future row only.

Implementation and independent documentation validation:

- The implementation context added exactly the two public methods, one package-private final
  field-free helper with the specified private constructor and six methods, the focused ten-test
  suite, and the two Tensor API inventory checks. Initial focused tests, all 518 model tests,
  generated model Javadoc, and the root test lifecycle passed before documentation review.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Clean documentation-focused context `/root/implement_model_0017f/docs_review_0017f` applied
  General style, API and Javadoc style, Planning style, and Example format after reading the
  architecture contract and focused boundary docs, documentation workflow, planning guide and
  roadmap, model capabilities/master plan, tasks 0002/0003/0007/0011/0012/0013/0017E/0017F,
  final source/tests, Tensor/Compile/Training APIs, glossary, generated Javadoc, related model
  contracts, neighboring expression contracts, and Java 26 build configuration. It independently
  inspected behavior and tests rather than relying on the implementation handoff.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Javadoc review found the two public methods and helper type, constructor, and all six methods
  complete for output-to-input order, one-time negative normalization, ownership, scalar empty
  permutation, failures, Shape/stride/offset rules, unresolved layout, result metadata,
  provenance, freshness, identifier exhaustion, and cross-layer boundaries. The pass made only a
  formatting correction in the affected Tensor type Javadoc; logic and signatures were unchanged.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [metadata glossary](../../../../glossary.md#term-metadata); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Tensor API now documents current permute/transpose construction and includes a complete Java 26
  example. `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-permute-doc-example /tmp/PermuteExpressionExample.java` passed, and `java -cp
  modules/model/build/classes/java/main:/tmp/synaptik-permute-doc-example
  PermuteExpressionExample` printed the documented Shape `[3, 4, 2]`, strides `[4, 1, 12]`,
  offset `5`, `STRIDED`, span `29`, normalized axes `[1, 2, 0]`, exact-input and storage-free
  booleans, transpose Shape `[3, 2]`, axes `[1, 0]`, and unresolved-layout [boolean](../../../../glossary.md#data-type--datatype).
- Compile API now lists permutation and transpose among current expression inputs while preserving
  capture, inference, canonicalization, materialization planning, backend lowering, and execution
  as planned. Glossary axis-transform, permutation, provenance, Tensor, and status distinctions
  now reflect the same current-versus-planned boundary without adding an unnecessary term.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering))</span>
- `./gradlew --no-daemon :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorPermutationExpressionTest --tests
  io.github.pho001.synaptik.model.tensor.TensorTest` passed after documentation review; focused XML
  records 24 tests total (10 permutation-expression tests and 14 Tensor tests), with zero failures,
  errors, or skips.
- `./gradlew --no-daemon :modules:model:test` passed after documentation review; XML aggregation
  records 518 tests with zero failures, errors, or skips.
- `./gradlew --no-daemon :modules:model:javadoc` passed. Generated Tensor Javadoc contains both
  methods, complete parameters/returns/failures, output-to-input order, and physical-alias/
  zero-copy boundaries.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- `./gradlew --no-daemon test` passed for the complete repository; the final run reported 36
  actionable tasks and no failure.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `javap -p -c -s` confirmed the exact field-free helper, private zero-argument constructor, six
  static methods, one long-arithmetic normalization path, exact Shape/stride reordering, and one
  final <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>`TensorFactory.createDerived` call. `javap -p -s` confirmed exactly `permute(int...)` and
  parameterless `transpose()` on Tensor. Reflection tests confirmed the same public/helper shape.
- Import and source inspection confirmed only JDK and local model contracts, no forbidden
  compiler/planning/prepare/runtime/backend/training dependency, no value or storage access, and
  no expand-dimensions or squeeze behavior. Existing task-0017F1 and task-0017G specifications do
  not exist.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [prepare glossary](../../../../glossary.md#prepare))</span>
- A targeted Ruby check resolved every local Markdown file link and heading anchor in the six
  changed documentation/planning files. Fence counts are even, trailing-whitespace scans found no
  matches, generated Javadoc inspection passed, every changed file ends with a newline, and
  `git diff --check` passed including explicit checks of both new Java files.
- Final scope inspection found exactly the ten permitted paths: two production files, two tests,
  Tensor API, Compile API, glossary, this task, model master plan, and roadmap. Task 0017F is
  `Complete` in all three planning locations; task 0017F1 and task 0017G remain `Draft` without
  specifications.
- Training API remains accurate unchanged because no trainable state, autograd rule, gradient
  publication, optimizer, or prepared execution was added. Capabilities remains accurate because
  it already lists permute/transpose at the model/public-API layer and distinguishes that layer
  from compiler and executable parity.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [optimizer glossary](../../../../glossary.md#term-optimizer); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- `AxisTransformKind`, `PermutationAttrs`, `Operation`, Shape/Dimension, LayoutDescriptor/
  LayoutKind, TensorDescriptor, TensorFactory, and TensorProvenance remain accurate unchanged:
  the helper composes their existing semantic, immutable metadata, allocation, and provenance
  contracts without changing them. Contiguous, reshape, and expand contracts also remain accurate
  because permutation uses a distinct complete-axis validation and stride-reordering algebra.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- Architecture and ADRs, architecture tests, backend conformance, integration tests, Gradle Java
  26 configuration, dependencies, and other modules remain accurate unchanged because this task
  changes no ownership, dependency, build, backend behavior, or end-to-end execution contract.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- Added exactly `Tensor.permute(int...)` and `Tensor.transpose()` with complete public Javadocs and
  one helper call each.
- Added the exact six-method package-private construction helper with defensive normalization,
  exact Dimension/stride reordering, conditional same-offset view layout, preserved type and
  eligibility, and exact PERMUTE provenance through one derived-factory delegation.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [Dimension glossary](../../../../glossary.md#dimension); [stride glossary](../../../../glossary.md#term-stride); [view glossary](../../../../glossary.md#term-view-meanings); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance))</span>
- Added focused coverage for API/helper shape, every data type and valid eligibility, ownership,
  scalar/dynamic/zero Shapes, exact failures and precedence, transpose, every resolved layout kind,
  unresolved geometry, freshness, storage/value non-interference, and identity side effects.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [data type glossary](../../../../glossary.md#data-type--datatype); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [transpose glossary](../../../../glossary.md#term-transpose); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finalized Tensor API, Compile API, glossary, task/master/roadmap status, generated Javadoc, and
  the runnable Java 26 example without changing another contract or module.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Javadoc glossary](../../../../glossary.md#term-javadoc); [module glossary](../../../../glossary.md#term-module))</span>

## Completion summary

- Completed changes: Implemented and documented arbitrary complete Tensor permutation and rank-two
  transpose as fresh storage-free PERMUTE expressions with conditional logical view geometry.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [permutation glossary](../../../../glossary.md#permutation); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [transpose glossary](../../../../glossary.md#term-transpose); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Files changed or created: Tensor, the new TensorPermutationExpressions helper, Tensor inventory
  test, the new focused expression test, Tensor API, Compile API, glossary, this task, model master
  plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused 24 tests, all 518 model tests, model Javadoc, full repository tests,
  runnable Java 26 example, bytecode/reflection/import/source inspection, link/anchor/fence/
  whitespace checks, exact-scope review, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0017f/docs_review_0017f` independently finalized all affected Javadocs,
  explanatory documentation, glossary impact, planning evidence, and status synchronization.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Documentation impact: Permute/transpose are current model expression construction; compiler
  capture/canonicalization, materialization, backend lowering, gradients, and execution remain
  planned in their owning layers.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [materialization glossary](../../../../glossary.md#materialization); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Javadoc review: Affected Tensor/helper contracts are complete; related foundational and
  neighboring expression Javadocs remain accurate unchanged for the reasons recorded above.
- Glossary impact: Existing axis-transform, permutation, provenance, Tensor, and implementation-
  status entries were updated; no new reusable domain term was needed.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [provenance glossary](../../../../glossary.md#provenance))</span>
- Unresolved issues: None.
- Follow-up required: None. Task 0017F1 remains the next Draft planning frontier without a detailed
  specification; task 0017G also remains Draft without a specification.

Status: Complete
