# [Contiguous](../../glossary.md#term-contiguous), reshape and expand expressions

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Keep regrouping separate from repetition

Application or layer code calls these methods on an input Tensor. The result records the exact
input reference; it does not copy or borrow an output buffer during construction. Raw shape
arrays are not retained, while an immutable target `Shape` is retained exactly.
Reshaping `[2, 3]` to `[3, 2]` preserves $2\times3=3\times2=6$ positions.
Expanding `[1, 3]` to `[2, 3]` instead describes two logical uses of each original column.
Logical view geometry does not extend an input arena's lifetime or promise physical aliasing.
Compare [axis reordering](axis-transforms.md) and [layout positions](layouts.md#resolved-layouts).

**[Reader model](../../glossary.md#term-reader-model).** These requests change different aspects of representation.
`contiguous` requests dense [row-major](../../glossary.md#term-row-major) geometry: last-[axis](../../glossary.md#term-axis) neighbors are adjacent, with no gaps
between rows. **Reshape** regroups the same ordered [element](../../glossary.md#term-element) sequence under new [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix),
so `[2, 3]` can become `[3, 2]` but cannot gain another element.
**Expand** logically repeats [singleton](../../glossary.md#term-singleton) positions or introduces leading repeated axes, such as
`[1, 3]` becoming `[2, 3]`. That repetition can be described by zero [strides](../../glossary.md#term-stride).
None of these expression constructors alone guarantees physical copying or shared memory;
a logical **[view](../../glossary.md#term-view-meanings)** describes coordinates, while **[materialization](../../glossary.md#materialization)** actually produces stored data.

<a id="contiguous-expressions"></a>
### [Contiguous](../../glossary.md#term-contiguous) expressions

The current parameterless `Tensor.contiguous()` method requests [canonical](../../glossary.md#term-canonical-form) dense [row-major](../../glossary.md#term-row-major) result
geometry for one logical input. It accepts every [Model](../../glossary.md#term-model) [data type](../../glossary.md#data-type--datatype) and preserves the exact input [Shape](../../glossary.md#shape)
reference, data type, and `requiresGrad` value. Every valid call creates a fresh unlabeled,
storage-free [Tensor](../../glossary.md#tensor) with `CONTIGUOUS`, `NoOperationAttrs.INSTANCE`, and exact ordered [provenance](../../glossary.md#provenance)
`[input]`.

For a fully static Shape, construction creates a new resolved `LayoutDescriptor` with canonical
row-major [strides](../../glossary.md#term-stride), logical [storage offset](../../glossary.md#term-storage-offset) zero, `isView() == false`, and the checked [referenced
element span](../../glossary.md#referenced-element-span). A [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../glossary.md#term-scalar) therefore has no strides and span one; any zero-[extent](../../glossary.md#term-extent) Shape has
span zero. For a Shape containing a dynamic [dimension](../../glossary.md#dimension), [layout](../../glossary.md#layout) remains unresolved because numeric
strides and span cannot yet be calculated.

Construction does not inspect the input layout, label, provenance, storage, [liveness](../../glossary.md#term-liveness), or values.
An unresolved, already dense, offset-contiguous, strided, or broadcast input therefore follows the
same Shape-based result rule. Repeated requests and a request on an already-contiguous expression
remain distinct until a later [compiler](../../glossary.md#term-compiler) proves a legal [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form). Resolved result geometry
describes the requested logical representation; it does not prove allocation, copying, distinct
physical storage, [runtime](../../glossary.md#term-runtime) [residency](../../glossary.md#residency), [backend](../../glossary.md#backend) support, or execution.

Current package-private compiler [autograd](../../glossary.md#autograd) passes a [cotangent](../../glossary.md#cotangent) through a floating
`CONTIGUOUS` [occurrence](../../glossary.md#term-occurrence) with exact matching input/output type and Shape. This logical formula
does not choose physical [aliasing](../../glossary.md#term-aliasing), copying, or [materialization](../../glossary.md#materialization).

<a id="complete-expression-construction-example-1"></a>
#### Complete expression-construction example

<a id="goal-and-inputs-24"></a>
##### Goal and inputs

Construct [contiguous](../../glossary.md#term-contiguous) expressions from a static strided [`FLOAT32`](../../glossary.md#term-float32) [tensor](../../glossary.md#tensor) and a dynamic [`INT64`](../../glossary.md#data-type--datatype)
tensor. Neither input needs [host storage](../../glossary.md#host-storage) because this boundary creates expression [metadata](../../glossary.md#term-metadata) only.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class ContiguousExpressionExample {
    public static void main(String[] args) {
        Shape staticShape = Shape.of(2, 3);
        LayoutDescriptor strided = LayoutDescriptor.of(
                staticShape, new long[] {1, 2}, 4, true);
        Tensor staticInput = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, staticShape, Optional.of(strided), true));

        Shape dynamicShape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));
        Tensor dynamicInput = TensorFactory.create(new TensorDescriptor(
                DataType.INT64, dynamicShape, Optional.empty(), false));

        // Canonical result geometry is a request, not proof that any copy has occurred.
        Tensor staticResult = staticInput.contiguous();
        Tensor dynamicResult = dynamicInput.contiguous();
        LayoutDescriptor resultLayout = staticResult.descriptor().layout().orElseThrow();

        System.out.println(staticResult.descriptor().shape() == staticShape);
        System.out.println(Arrays.toString(resultLayout.strides()));
        System.out.println(resultLayout.storageOffset());
        System.out.println(resultLayout.isView());
        System.out.println(resultLayout.referencedElementSpan());
        System.out.println(dynamicResult.descriptor().layout().isEmpty());
        System.out.println(staticResult.provenance().orElseThrow().operation().kind());
        System.out.println(staticResult.provenance().orElseThrow().inputs().getFirst()
                == staticInput);
        System.out.println(staticResult.hostStorage().isEmpty());
    }
}
```

<a id="meaningful-lines-and-intermediate-results-9"></a>
##### Meaningful lines and intermediate results

- The input uses non-[canonical](../../glossary.md#term-canonical-form) [strides](../../glossary.md#term-stride) `[1, 2]`, offset `4`, and [view](../../glossary.md#term-view-meanings) [metadata](../../glossary.md#term-metadata), but
  `staticInput.contiguous()` derives result geometry only from `staticShape`.
- The static result retains the exact [Shape](../../glossary.md#shape), [`FLOAT32`](../../glossary.md#term-float32), and true gradient-eligibility metadata,
  but receives newly constructed canonical strides <span data-glossary-navigation="true"> ([gradient glossary](../../glossary.md#term-gradient))</span>`[3, 1]`, offset zero, non-view metadata, and
  span six.
- The dynamic result retains its exact Shape and [`INT64`](../../glossary.md#data-type--datatype) metadata while leaving [layout](../../glossary.md#layout) unresolved.
- [Provenance](../../glossary.md#provenance) records exactly `staticInput`; neither result receives label or storage.

<a id="result-and-interpretation-24"></a>
##### Result and interpretation

The program prints:

```text
true
[3, 1]
0
false
6
true
CONTIGUOUS
true
true
```

The result demonstrates static and dynamic [descriptor](../../glossary.md#term-descriptor) derivation plus one-input [provenance](../../glossary.md#provenance). It
does not demonstrate a copy, [materialization](../../glossary.md#materialization), [compiler](../../glossary.md#term-compiler) [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), [backend route](../../glossary.md#backend-route), or
execution.

<a id="failures-and-useful-variations-21"></a>
##### Failures and useful variations

- [Canonical](../../glossary.md#term-canonical-form) [stride](../../glossary.md#term-stride) or span [overflow](../../glossary.md#term-overflow) for a fully static [Shape](../../glossary.md#shape) throws `ArithmeticException` before
  a Tensor identity is consumed. Dynamic Shapes do not perform that numeric layout calculation.<span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings); [layout glossary](../../glossary.md#layout))</span>
- Exhausted [Tensor](../../glossary.md#tensor) identifier space throws `IllegalStateException` only after local immutable
  descriptor, operation, and provenance metadata has been constructed.<span data-glossary-navigation="true"> ([immutable glossary](../../glossary.md#term-immutable); [descriptor glossary](../../glossary.md#term-descriptor); [operation glossary](../../glossary.md#operation); [provenance glossary](../../glossary.md#provenance); [metadata glossary](../../glossary.md#term-metadata))</span>
- Scalar and zero-extent Shapes are valid, as are all current data types. Repeated or nested
  requests always return fresh expression identities.<span data-glossary-navigation="true"> ([Scalar glossary](../../glossary.md#term-scalar); [extent glossary](../../glossary.md#term-extent))</span>

<a id="reshape-expressions"></a>
### Reshape expressions

The current `Tensor.reshape(long...)` and `Tensor.reshape(Shape)` methods create fresh [model](../../glossary.md#term-model)
expressions that preserve the input's ordered logical [element](../../glossary.md#term-element) sequence under new [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix).
Both overloads accept every Model [data type](../../glossary.md#data-type--datatype), retain the exact input data type and
`requiresGrad` value, and return an unlabeled, storage-free [Tensor](../../glossary.md#tensor). [Provenance](../../glossary.md#provenance) contains exactly
`Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(targetShape))` and ordered input
`[input]`. Same-[shape](../../glossary.md#shape), repeated, and nested requests remain explicit fresh expressions.

Current package-private [compiler](../../glossary.md#term-compiler) [autograd](../../glossary.md#autograd) reverses a supported floating reshape with
`gradient.reshape(input.descriptor().shape())` after [preflight](../../glossary.md#term-preflight) verifies the current element-count
contract. The formula constructs [metadata](../../glossary.md#term-metadata) only and does not select physical [view](../../glossary.md#term-view-meanings) or copy
behavior.

The raw `long...` overload treats its array as caller-owned input: it neither mutates nor retains
the array. An empty request means the [canonical](../../glossary.md#term-canonical-form) [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../glossary.md#term-scalar) Shape. Every ordinary [extent](../../glossary.md#term-extent) must
be non-negative, so zero extents are valid. At most one exact `-1` may request [inference](../../glossary.md#term-inference-meanings); every
value below `-1` is invalid. The sentinel is normalized before `Shape` and `TargetShapeAttrs`
construction and therefore never appears in stored semantic state.

Inference requires a fully known input element count and a non-zero checked product of all other
requested extents. The input count must be exactly divisible by that product. A zero input count
can therefore infer zero when the other requested product is non-zero, such as reshaping `[0, 3]`
with `[-1, 2]` to `[0, 2]`. A request such as `[0, -1]` is rejected because infinitely many
inferred extents satisfy a zero product. Once a zero requested extent is present, that ambiguity is
reported without performing irrelevant multiplication that could [overflow](../../glossary.md#term-overflow) elsewhere in the raw
request.

The exact-Shape overload accepts a non-null already normalized scalar, zero-extent, static, mixed
dynamic, or fully dynamic `Shape` and retains that exact [immutable](../../glossary.md#term-immutable) reference in both the result
[descriptor](../../glossary.md#term-descriptor) and `TargetShapeAttrs`. For either overload, known input and [target](../../glossary.md#term-target) element counts must
match. If either Shape has a dynamic [dimension](../../glossary.md#dimension), the equality constraint is accepted and deferred
to later compiler [validation](../../glossary.md#term-validation); this method neither binds symbols nor assumes that the counts match.

Result [layout](../../glossary.md#layout) is resolved only when two facts are locally available: the input descriptor has
resolved [contiguous](../../glossary.md#term-contiguous) geometry, including offset-contiguous geometry, and the target Shape is fully
static. That branch creates a new view-marked layout with canonical target [row-major](../../glossary.md#term-row-major) [strides](../../glossary.md#term-stride) and
the exact input element offset. Offset zero classifies as `DENSE_CONTIGUOUS`; non-zero offset
classifies as `DENSE_WITH_OFFSET`. Unresolved, strided, or broadcast input geometry, or a dynamic
target, produces unresolved result layout. Reshape does not insert `contiguous()`, access or attach
storage, copy values, or decide an executable [alias](../../glossary.md#aliasing-and-arena)-versus-[materialization](../../glossary.md#materialization) route. Even resolved
view metadata is only a logical model fact and does not promise zero-copy execution.

<a id="complete-expression-construction-example-2"></a>
#### Complete expression-construction example

<a id="goal-and-inputs-25"></a>
##### Goal and inputs

Normalize one inferred raw request against a static offset-[contiguous](../../glossary.md#term-contiguous) input, then retain one exact
dynamic [target](../../glossary.md#term-target) [Shape](../../glossary.md#shape). Both calls observe expression [metadata](../../glossary.md#term-metadata) only; no storage is needed.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class ReshapeExpressionExample {
    public static void main(String[] args) {
        Shape inputShape = Shape.of(2, 3);
        LayoutDescriptor inputLayout = LayoutDescriptor.of(
                inputShape, new long[] {3, 1}, 5, true);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, inputShape, Optional.of(inputLayout), true));

        long[] request = {3, -1};
        // Infer only the missing extent; mutating the raw array later cannot edit the result.
        Tensor staticResult = input.reshape(request);
        request[0] = 99;

        Shape dynamicTarget = Shape.ofDimensions(
                new DynamicDimension("items"), new StaticDimension(2));
        Tensor dynamicResult = input.reshape(dynamicTarget);
        LayoutDescriptor resultLayout = staticResult.descriptor().layout().orElseThrow();

        System.out.println(staticResult.descriptor().shape());
        System.out.println(Arrays.toString(resultLayout.strides()));
        System.out.println(resultLayout.storageOffset());
        System.out.println(resultLayout.isView());
        System.out.println(staticResult.provenance().orElseThrow().operation().kind());
        System.out.println(staticResult.provenance().orElseThrow().inputs().getFirst() == input);
        System.out.println(dynamicResult.descriptor().shape() == dynamicTarget);
        System.out.println(dynamicResult.descriptor().layout().isEmpty());
        System.out.println(staticResult.hostStorage().isEmpty());
    }
}
```

<a id="meaningful-lines-and-intermediate-results-10"></a>
##### Meaningful lines and intermediate results

- Input [Shape](../../glossary.md#shape) `[2, 3]` has six [elements](../../glossary.md#term-element). In raw request `[3, -1]`, the known product is three, so
  the inferred extent is <span data-glossary-navigation="true"> ([extent glossary](../../glossary.md#term-extent))</span>`6 / 3 = 2` and the normalized [target](../../glossary.md#term-target) is `[3, 2]`.
- Mutating `request` after the call cannot change the result because raw array ownership remains
  with the caller and the normalized Shape owns its immutable dimensions.<span data-glossary-navigation="true"> ([ownership glossary](../../glossary.md#term-ownership-meanings); [normalized Shape glossary](../../glossary.md#normalized-shape); [immutable glossary](../../glossary.md#term-immutable); [dimensions glossary](../../glossary.md#dimension))</span>
- The input layout is contiguous at element offset five. The static result therefore receives new
  canonical target strides <span data-glossary-navigation="true"> ([layout glossary](../../glossary.md#layout); [contiguous glossary](../../glossary.md#term-contiguous); [canonical glossary](../../glossary.md#term-canonical-form); [strides glossary](../../glossary.md#term-stride))</span>`[2, 1]`, retains offset five, and is explicitly marked as a [view](../../glossary.md#term-view-meanings).
- The exact dynamic target is accepted because its element count is not locally known. Its
  reference is retained, while layout remains unresolved.
- Both results retain [`FLOAT32`](../../glossary.md#term-float32) and true [gradient](../../glossary.md#term-gradient) eligibility, record exact one-input `RESHAPE`
  provenance, and have no label or host storage.<span data-glossary-navigation="true"> ([provenance glossary](../../glossary.md#provenance); [host storage glossary](../../glossary.md#host-storage))</span>

<a id="result-and-interpretation-25"></a>
##### Result and interpretation

The program prints:

```text
Shape[3, 2]
[2, 1]
5
true
RESHAPE
true
true
true
true
```

The result demonstrates raw [inference](../../glossary.md#term-inference-meanings), defensive [ownership](../../glossary.md#term-ownership-meanings), exact dynamic-[Shape](../../glossary.md#shape) retention,
conditional same-offset [view](../../glossary.md#term-view-meanings) geometry, and [provenance](../../glossary.md#provenance). It does not demonstrate value movement,
storage [aliasing](../../glossary.md#term-aliasing), [compiler](../../glossary.md#term-compiler) constraint solving or [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), [gradients](../../glossary.md#term-gradient), [materialization](../../glossary.md#materialization),
[backend](../../glossary.md#backend) [lowering](../../glossary.md#lowering), or execution.

<a id="failures-and-useful-variations-22"></a>
##### Failures and useful variations

- A null raw array or exact [Shape](../../glossary.md#shape) fails with `NullPointerException` naming `requestedShape` or
  `targetShape`, respectively.
- A raw [extent](../../glossary.md#term-extent) below `-1`, a second `-1`, dynamic-input inference, zero-product inference,
  non-divisible inference, or unequal known counts fails with <span data-glossary-navigation="true"> ([inference glossary](../../glossary.md#term-inference-meanings))</span>`IllegalArgumentException`.
- Checked element-count, non-zero requested-product, canonical-stride, and referenced-span
  overflow propagates as <span data-glossary-navigation="true"> ([canonical glossary](../../glossary.md#term-canonical-form); [stride glossary](../../glossary.md#term-stride); [overflow glossary](../../glossary.md#term-overflow))</span>`ArithmeticException` before a [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings) is consumed.
- Exhausted Tensor identifier space fails with `IllegalStateException` only at final derived
  construction after local validation and metadata construction.<span data-glossary-navigation="true"> ([validation glossary](../../glossary.md#term-validation); [metadata glossary](../../glossary.md#term-metadata))</span>
- `reshape()` requests a scalar and succeeds only when the input has one known element or when
  compatibility is dynamic. <span data-glossary-navigation="true"> ([scalar glossary](../../glossary.md#term-scalar))</span>`reshape(-1, 2)` on a zero-[element](../../glossary.md#term-element) input infers `[0, 2]`, while
  `reshape(0, -1)` is ambiguous and fails.

<a id="expand-expressions"></a>
### Expand expressions

The current `Tensor.expand(long...)` and `Tensor.expand(Shape)` methods create fresh [model](../../glossary.md#term-model)
expressions that logically repeat compatible input positions into an exact [target](../../glossary.md#term-target) [Shape](../../glossary.md#shape). Both
overloads accept every current [data type](../../glossary.md#data-type--datatype), retain the exact input data type and `requiresGrad`
value, and return an unlabeled, storage-free [Tensor](../../glossary.md#tensor). [Provenance](../../glossary.md#provenance) contains exactly
`Operation(ShapeTransformKind.EXPAND, new TargetShapeAttrs(targetShape))` and ordered input
`[input]`. [Identity](../../glossary.md#term-identity-meanings)-like, repeated, and nested calls remain explicit fresh expressions.

Current package-private [compiler](../../glossary.md#term-compiler) [autograd](../../glossary.md#autograd) reverses a compiler-admitted floating expand with
`gradient.sumToShape(input.descriptor().shape())`, using the existing public [binding](../../glossary.md#term-binding)-aware
reduction. Compiler verification represents a non-equal unresolved aligned pair as the
source-one-or-source-equal obligation, proves or retains it, and [preflight](../../glossary.md#term-preflight)/[AD](../../glossary.md#autograd) use the same
binding-aware `sumToShape` semantics. Concrete binding must discharge retained obligations
within its qualified [slice](../../glossary.md#slice). This does not promise general dynamic execution, select storage
[aliasing](../../glossary.md#term-aliasing) or choose [materialization](../../glossary.md#materialization).

Expansion is directional rather than symmetric [broadcasting](../../glossary.md#broadcasting). Input and target [axes](../../glossary.md#term-axis) align from
the right, and extra leading target axes are valid because the missing input axes behave as
logical singletons. [Structural equality](../../glossary.md#term-structural-equality) and a static source [singleton](../../glossary.md#term-singleton) are immediately compatible.
An unequal fully static pair is rejected when the source [extent](../../glossary.md#term-extent) is not `1`; zero is an ordinary
static extent, so `0` cannot expand to `1`. When either member of an aligned pair is unresolved,
Model construction accepts the expression and retains the later requirement
`source extent == 1 || source extent == target extent`. This includes unequal named [dimensions](../../glossary.md#dimension),
expressions, constrained unknowns, static/unresolved, unresolved/static, and
unresolved/unresolved pairs. Model construction does not bind, unify, or evaluate a dimension and
does not create a [deferred graph constraint](../../glossary.md#deferred-graph-constraint).

The raw `long...` overload passes the caller-owned dimensions through `Shape.of(long...)`; the
array is not retained or mutated. Every value is a literal non-negative extent, including zero,
and an empty request denotes [scalar](../../glossary.md#term-scalar) Shape. Unlike reshape, numeric `-1` has no [inference](../../glossary.md#term-inference-meanings) meaning
and is rejected. The `Shape` overload instead retains the exact [immutable](../../glossary.md#term-immutable) target reference,
including scalar, zero-extent, mixed dynamic, and fully dynamic Shapes. The exact reference is
retained even when aligned compatibility depends on later bindings.

For a fully static target and resolved input geometry, expand publishes a new logical [view](../../glossary.md#term-view-meanings)
[layout](../../glossary.md#layout). It preserves the input [element](../../glossary.md#term-element) offset and the exact aligned [stride](../../glossary.md#term-stride) for unchanged axes,
including an existing zero or non-[canonical](../../glossary.md#term-canonical-form) stride. New leading axes and changed input-singleton
axes receive stride zero. A dynamic target, unresolved input geometry, or accepted
binding-dependent pair leaves result layout unresolved. The derived view [metadata](../../glossary.md#term-metadata) does not attach
or [alias](../../glossary.md#aliasing-and-arena) [host storage](../../glossary.md#host-storage), repeat values, choose materialization, or promise zero-copy execution.

<a id="complete-expression-construction-example-3"></a>
#### Complete expression-construction example

<a id="goal-and-inputs-26"></a>
##### Goal and inputs

Expand an offset-[contiguous](../../glossary.md#term-contiguous) `[1, 3]` input to `[2, 4, 3]`, then retain a [binding](../../glossary.md#term-binding)-dependent exact
[target](../../glossary.md#term-target) for a separate dynamic input. The example observes [descriptors](../../glossary.md#term-descriptor) and [provenance](../../glossary.md#provenance) only.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class ExpandExpressionExample {
    public static void main(String[] args) {
        Shape inputShape = Shape.of(1, 3);
        LayoutDescriptor inputLayout = LayoutDescriptor.of(
                inputShape, new long[] {3, 1}, 5, true);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, inputShape, Optional.of(inputLayout), true));

        long[] request = {2, 4, 3};
        // Added and repeated axes can receive zero strides without allocating repeated data.
        Tensor staticResult = input.expand(request);
        request[0] = 99;
        LayoutDescriptor resultLayout = staticResult.descriptor().layout().orElseThrow();

        Shape dynamicInputShape = Shape.ofDimensions(
                new DynamicDimension("sourceItems"), new StaticDimension(3));
        Tensor dynamicInput = TensorFactory.create(new TensorDescriptor(
                DataType.INT64, dynamicInputShape, Optional.empty(), false));
        Shape dynamicTarget = Shape.ofDimensions(
                new DynamicDimension("targetItems"), new StaticDimension(3));
        Tensor dynamicResult = dynamicInput.expand(dynamicTarget);

        System.out.println(staticResult.descriptor().shape());
        System.out.println(Arrays.toString(resultLayout.strides()));
        System.out.println(resultLayout.storageOffset());
        System.out.println(resultLayout.kind());
        System.out.println(resultLayout.referencedElementSpan());
        System.out.println(resultLayout.isView());
        System.out.println(staticResult.provenance().orElseThrow().operation().kind());
        System.out.println(staticResult.provenance().orElseThrow().inputs().getFirst() == input);
        System.out.println(dynamicResult.descriptor().shape() == dynamicTarget);
        System.out.println(dynamicResult.descriptor().layout().isEmpty());
        System.out.println(staticResult.hostStorage().isEmpty());
    }
}
```

<a id="meaningful-lines-and-intermediate-results-11"></a>
##### Meaningful lines and intermediate results

- Pairing axes from the right maps input `[1, 3]` to the final two target axes of `[2, 4, 3]`.
  The new leading extent `2` and expanded singleton extent `4` receive zero strides; the
  unchanged extent `3` preserves stride one. Result strides are `[0, 0, 1]`.
- The input offset five is preserved. The greatest referenced element index is seven, so the
  referenced element span is eight even though the result has 24 logical positions.<span data-glossary-navigation="true"> ([element glossary](../../glossary.md#term-element); [referenced element span glossary](../../glossary.md#referenced-element-span))</span>
- Mutating the raw request cannot change the normalized target [Shape](../../glossary.md#shape).
- `[sourceItems, 3]` to `[targetItems, 3]` is accepted as a binding-dependent Model expression.
  Its later aligned condition is <span data-glossary-navigation="true"> ([binding glossary](../../glossary.md#term-binding); [Model glossary](../../glossary.md#term-model))</span>`sourceItems == 1 || sourceItems == targetItems`; construction
  creates no binding or deferred compiler constraint. The exact target reference is retained and
  layout remains unresolved.<span data-glossary-navigation="true"> ([compiler glossary](../../glossary.md#term-compiler); [layout glossary](../../glossary.md#layout))</span>
- Both results record one-input `EXPAND` [provenance](../../glossary.md#provenance) and receive no label or [host storage](../../glossary.md#host-storage).

<a id="result-and-interpretation-26"></a>
##### Result and interpretation

The program prints:

```text
Shape[2, 4, 3]
[0, 0, 1]
5
BROADCAST_ZERO_STRIDE
8
true
EXPAND
true
true
true
true
```

The result demonstrates directional compatibility, [binding](../../glossary.md#term-binding)-dependent exact-[Shape](../../glossary.md#shape) retention,
defensive raw-array [ownership](../../glossary.md#term-ownership-meanings), preserved and zero [strides](../../glossary.md#term-stride), same-offset [view](../../glossary.md#term-view-meanings) geometry, and
[provenance](../../glossary.md#provenance). It does not prove the retained binding obligation or demonstrate value repetition,
storage [aliasing](../../glossary.md#term-aliasing), [gradients](../../glossary.md#term-gradient), [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings) or constraints, [materialization](../../glossary.md#materialization), [backend](../../glossary.md#backend) [lowering](../../glossary.md#lowering),
or execution.

<a id="failures-and-useful-variations-23"></a>
##### Failures and useful variations

- A null raw array or exact [Shape](../../glossary.md#shape) fails with `NullPointerException` naming `requestedShape` or
  `targetShape`, respectively.
- A negative raw [extent](../../glossary.md#term-extent) or lower-[rank](../../glossary.md#term-rank) [target](../../glossary.md#term-target) fails with `IllegalArgumentException`. An unequal
  fully static aligned pair also fails when its source extent is not one, including `0` to `1`.
  A pair containing an unresolved dimension is retained instead of failing locally.<span data-glossary-navigation="true"> ([dimension glossary](../../glossary.md#dimension))</span>
- Resolved [layout](../../glossary.md#layout) arithmetic [overflow](../../glossary.md#term-overflow) propagates as `ArithmeticException` before a Tensor identity
  is consumed.<span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings))</span>
- Exhausted [Tensor](../../glossary.md#tensor) identifier space fails with `IllegalStateException` only at final derived
  construction after local validation and metadata construction.<span data-glossary-navigation="true"> ([validation glossary](../../glossary.md#term-validation); [metadata glossary](../../glossary.md#term-metadata))</span>
- Scalar input may expand to any target Shape. Static or dynamic identity-like requests remain
  fresh expressions rather than returning the input.<span data-glossary-navigation="true"> ([Scalar glossary](../../glossary.md#term-scalar))</span>


---

[Previous topic](statistics-losses.md) · [Next topic](axis-transforms.md)
