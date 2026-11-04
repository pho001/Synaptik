# Contiguous, reshape and expand expressions

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="contiguous-expressions"></a>
### Contiguous expressions

The current parameterless `Tensor.contiguous()` method requests canonical dense row-major result
geometry for one logical input. It accepts every Model data type and preserves the exact input Shape
reference, data type, and `requiresGrad` value. Every valid call creates a fresh unlabeled,
storage-free Tensor with `CONTIGUOUS`, `NoOperationAttrs.INSTANCE`, and exact ordered provenance
`[input]`.

For a fully static Shape, construction creates a new resolved `LayoutDescriptor` with canonical
row-major strides, logical storage offset zero, `isView() == false`, and the checked referenced
element span. A rank-zero scalar therefore has no strides and span one; any zero-extent Shape has
span zero. For a Shape containing a dynamic dimension, layout remains unresolved because numeric
strides and span cannot yet be calculated.

Construction does not inspect the input layout, label, provenance, storage, liveness, or values.
An unresolved, already dense, offset-contiguous, strided, or broadcast input therefore follows the
same Shape-based result rule. Repeated requests and a request on an already-contiguous expression
remain distinct until a later compiler proves a legal canonicalization. Resolved result geometry
describes the requested logical representation; it does not prove allocation, copying, distinct
physical storage, runtime residency, backend support, or execution.

Current package-private compiler autograd passes a cotangent through a floating
`CONTIGUOUS` occurrence with exact matching input/output type and Shape. This logical formula
does not choose physical aliasing, copying, or materialization.

<a id="complete-expression-construction-example-1"></a>
#### Complete expression-construction example

<a id="goal-and-inputs-24"></a>
##### Goal and inputs

Construct contiguous expressions from a static strided `FLOAT32` tensor and a dynamic `INT64`
tensor. Neither input needs host storage because this boundary creates expression metadata only.

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

- The input uses non-canonical strides `[1, 2]`, offset `4`, and view metadata, but
  `staticInput.contiguous()` derives result geometry only from `staticShape`.
- The static result retains the exact Shape, `FLOAT32`, and true gradient-eligibility metadata,
  but receives newly constructed canonical strides `[3, 1]`, offset zero, non-view metadata, and
  span six.
- The dynamic result retains its exact Shape and `INT64` metadata while leaving layout unresolved.
- Provenance records exactly `staticInput`; neither result receives label or storage.

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

The result demonstrates static and dynamic descriptor derivation plus one-input provenance. It
does not demonstrate a copy, materialization, compiler canonicalization, backend route, or
execution.

<a id="failures-and-useful-variations-21"></a>
##### Failures and useful variations

- Canonical stride or span overflow for a fully static Shape throws `ArithmeticException` before
  a Tensor identity is consumed. Dynamic Shapes do not perform that numeric layout calculation.
- Exhausted Tensor identifier space throws `IllegalStateException` only after local immutable
  descriptor, operation, and provenance metadata has been constructed.
- Scalar and zero-extent Shapes are valid, as are all current data types. Repeated or nested
  requests always return fresh expression identities.

<a id="reshape-expressions"></a>
### Reshape expressions

The current `Tensor.reshape(long...)` and `Tensor.reshape(Shape)` methods create fresh model
expressions that preserve the input's ordered logical element sequence under new coordinates.
Both overloads accept every Model data type, retain the exact input data type and
`requiresGrad` value, and return an unlabeled, storage-free Tensor. Provenance contains exactly
`Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(targetShape))` and ordered input
`[input]`. Same-shape, repeated, and nested requests remain explicit fresh expressions.

Current package-private compiler autograd reverses a supported floating reshape with
`gradient.reshape(input.descriptor().shape())` after preflight verifies the current element-count
contract. The formula constructs metadata only and does not select physical view or copy
behavior.

The raw `long...` overload treats its array as caller-owned input: it neither mutates nor retains
the array. An empty request means the canonical rank-zero scalar Shape. Every ordinary extent must
be non-negative, so zero extents are valid. At most one exact `-1` may request inference; every
value below `-1` is invalid. The sentinel is normalized before `Shape` and `TargetShapeAttrs`
construction and therefore never appears in stored semantic state.

Inference requires a fully known input element count and a non-zero checked product of all other
requested extents. The input count must be exactly divisible by that product. A zero input count
can therefore infer zero when the other requested product is non-zero, such as reshaping `[0, 3]`
with `[-1, 2]` to `[0, 2]`. A request such as `[0, -1]` is rejected because infinitely many
inferred extents satisfy a zero product. Once a zero requested extent is present, that ambiguity is
reported without performing irrelevant multiplication that could overflow elsewhere in the raw
request.

The exact-Shape overload accepts a non-null already normalized scalar, zero-extent, static, mixed
dynamic, or fully dynamic `Shape` and retains that exact immutable reference in both the result
descriptor and `TargetShapeAttrs`. For either overload, known input and target element counts must
match. If either Shape has a dynamic dimension, the equality constraint is accepted and deferred
to later compiler validation; this method neither binds symbols nor assumes that the counts match.

Result layout is resolved only when two facts are locally available: the input descriptor has
resolved contiguous geometry, including offset-contiguous geometry, and the target Shape is fully
static. That branch creates a new view-marked layout with canonical target row-major strides and
the exact input element offset. Offset zero classifies as `DENSE_CONTIGUOUS`; non-zero offset
classifies as `DENSE_WITH_OFFSET`. Unresolved, strided, or broadcast input geometry, or a dynamic
target, produces unresolved result layout. Reshape does not insert `contiguous()`, access or attach
storage, copy values, or decide an executable alias-versus-materialization route. Even resolved
view metadata is only a logical model fact and does not promise zero-copy execution.

<a id="complete-expression-construction-example-2"></a>
#### Complete expression-construction example

<a id="goal-and-inputs-25"></a>
##### Goal and inputs

Normalize one inferred raw request against a static offset-contiguous input, then retain one exact
dynamic target Shape. Both calls observe expression metadata only; no storage is needed.

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

- Input Shape `[2, 3]` has six elements. In raw request `[3, -1]`, the known product is three, so
  the inferred extent is `6 / 3 = 2` and the normalized target is `[3, 2]`.
- Mutating `request` after the call cannot change the result because raw array ownership remains
  with the caller and the normalized Shape owns its immutable dimensions.
- The input layout is contiguous at element offset five. The static result therefore receives new
  canonical target strides `[2, 1]`, retains offset five, and is explicitly marked as a view.
- The exact dynamic target is accepted because its element count is not locally known. Its
  reference is retained, while layout remains unresolved.
- Both results retain `FLOAT32` and true gradient eligibility, record exact one-input `RESHAPE`
  provenance, and have no label or host storage.

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

The result demonstrates raw inference, defensive ownership, exact dynamic-Shape retention,
conditional same-offset view geometry, and provenance. It does not demonstrate value movement,
storage aliasing, compiler constraint solving or canonicalization, gradients, materialization,
backend lowering, or execution.

<a id="failures-and-useful-variations-22"></a>
##### Failures and useful variations

- A null raw array or exact Shape fails with `NullPointerException` naming `requestedShape` or
  `targetShape`, respectively.
- A raw extent below `-1`, a second `-1`, dynamic-input inference, zero-product inference,
  non-divisible inference, or unequal known counts fails with `IllegalArgumentException`.
- Checked element-count, non-zero requested-product, canonical-stride, and referenced-span
  overflow propagates as `ArithmeticException` before a Tensor identity is consumed.
- Exhausted Tensor identifier space fails with `IllegalStateException` only at final derived
  construction after local validation and metadata construction.
- `reshape()` requests a scalar and succeeds only when the input has one known element or when
  compatibility is dynamic. `reshape(-1, 2)` on a zero-element input infers `[0, 2]`, while
  `reshape(0, -1)` is ambiguous and fails.

<a id="expand-expressions"></a>
### Expand expressions

The current `Tensor.expand(long...)` and `Tensor.expand(Shape)` methods create fresh model
expressions that logically repeat compatible input positions into an exact target Shape. Both
overloads accept every current data type, retain the exact input data type and `requiresGrad`
value, and return an unlabeled, storage-free Tensor. Provenance contains exactly
`Operation(ShapeTransformKind.EXPAND, new TargetShapeAttrs(targetShape))` and ordered input
`[input]`. Identity-like, repeated, and nested calls remain explicit fresh expressions.

Current package-private compiler autograd reverses a compiler-admitted floating expand with
`gradient.sumToShape(input.descriptor().shape())`, using the existing public binding-aware
reduction. That compiler path currently admits only the locally established equal-dimension or
static-source-singleton cases. It does not yet adopt the wider Model acceptance described below:
an EXPAND occurrence with a non-equal unresolved aligned pair still fails current compiler
inference. Compiler task 0005B is planned to translate that occurrence into the corresponding
deferred proof and to make preflight and gradient construction consistent. This logical formula
does not select storage aliasing or materialization.

Expansion is directional rather than symmetric broadcasting. Input and target axes align from
the right, and extra leading target axes are valid because the missing input axes behave as
logical singletons. Structural equality and a static source singleton are immediately compatible.
An unequal fully static pair is rejected when the source extent is not `1`; zero is an ordinary
static extent, so `0` cannot expand to `1`. When either member of an aligned pair is unresolved,
Model construction accepts the expression and retains the later requirement
`source extent == 1 || source extent == target extent`. This includes unequal named dimensions,
expressions, constrained unknowns, static/unresolved, unresolved/static, and
unresolved/unresolved pairs. Model construction does not bind, unify, or evaluate a dimension and
does not create a deferred graph constraint.

The raw `long...` overload passes the caller-owned dimensions through `Shape.of(long...)`; the
array is not retained or mutated. Every value is a literal non-negative extent, including zero,
and an empty request denotes scalar Shape. Unlike reshape, numeric `-1` has no inference meaning
and is rejected. The `Shape` overload instead retains the exact immutable target reference,
including scalar, zero-extent, mixed dynamic, and fully dynamic Shapes. The exact reference is
retained even when aligned compatibility depends on later bindings.

For a fully static target and resolved input geometry, expand publishes a new logical view
layout. It preserves the input element offset and the exact aligned stride for unchanged axes,
including an existing zero or non-canonical stride. New leading axes and changed input-singleton
axes receive stride zero. A dynamic target, unresolved input geometry, or accepted
binding-dependent pair leaves result layout unresolved. The derived view metadata does not attach
or alias host storage, repeat values, choose materialization, or promise zero-copy execution.

<a id="complete-expression-construction-example-3"></a>
#### Complete expression-construction example

<a id="goal-and-inputs-26"></a>
##### Goal and inputs

Expand an offset-contiguous `[1, 3]` input to `[2, 4, 3]`, then retain a binding-dependent exact
target for a separate dynamic input. The example observes descriptors and provenance only.

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

- Right alignment maps input `[1, 3]` to the final two target axes in `[2, 4, 3]`. The new leading
  extent `2` and expanded singleton extent `4` receive zero strides; the unchanged extent `3`
  preserves stride one. The result strides are therefore `[0, 0, 1]`.
- The input offset five is preserved. The greatest referenced element index is seven, so the
  referenced element span is eight even though the result has 24 logical positions.
- Mutating the raw request cannot change the normalized target Shape.
- `[sourceItems, 3]` to `[targetItems, 3]` is accepted as a binding-dependent Model expression.
  Its later aligned condition is `sourceItems == 1 || sourceItems == targetItems`; construction
  creates no binding or deferred compiler constraint. The exact target reference is retained and
  layout remains unresolved.
- Both results record one-input `EXPAND` provenance and receive no label or host storage.

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

The result demonstrates directional compatibility, binding-dependent exact-Shape retention,
defensive raw-array ownership, preserved and zero strides, same-offset view geometry, and
provenance. It does not prove the retained binding obligation or demonstrate value repetition,
storage aliasing, gradients, compiler capture or constraints, materialization, backend lowering,
or execution.

<a id="failures-and-useful-variations-23"></a>
##### Failures and useful variations

- A null raw array or exact Shape fails with `NullPointerException` naming `requestedShape` or
  `targetShape`, respectively.
- A negative raw extent or lower-rank target fails with `IllegalArgumentException`. An unequal
  fully static aligned pair also fails when its source extent is not one, including `0` to `1`.
  A pair containing an unresolved dimension is retained instead of failing locally.
- Resolved layout arithmetic overflow propagates as `ArithmeticException` before a Tensor identity
  is consumed.
- Exhausted Tensor identifier space fails with `IllegalStateException` only at final derived
  construction after local validation and metadata construction.
- Scalar input may expand to any target Shape. Static or dynamic identity-like requests remain
  fresh expressions rather than returning the input.


---

[Previous topic](statistics-losses.md) · [Next topic](axis-transforms.md)
