# Describe tensor metadata

## Outcome

You can use the implemented model API to describe an element type, shape, and resolved layout. You cannot yet create a public mutable `Tensor` or attach host storage; those contracts remain planned.

## Prerequisites

Use JDK 26 and add `implementation(project(":modules:model"))` to the consuming module's Gradle dependencies. See [Getting started](../getting-started.md) for the complete setup and interpretation.

## Create logical metadata

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;

DataType dataType = DataType.FLOAT32;
Shape shape = Shape.of(2, 3);
LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);

long elementCount = shape.knownElementCount().orElseThrow();
long referencedSpan = layout.referencedElementSpan();
```

`FLOAT32` describes 32-bit floating values but does not allocate them. `Shape.of(2, 3)` creates two rows of three logical positions, so `elementCount` is `2 × 3 = 6`. The contiguous strides are `[3, 1]`, and the layout references six storage positions, so `referencedSpan` is also `6`.

## Expected result

The example produces immutable metadata for six logical values. It proves that the current model can validate shape and layout geometry. It does not create storage, a tensor expression, a compiled graph, or backend work.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `knownElementCount()` is empty | The shape contains a symbolic dynamic dimension. | Bind or resolve the dimension in a later compiler contract; do not invent a negative size. |
| Layout creation rejects a shape | Numeric layouts require a fully static shape. | Keep only the `Shape` until dimensions are resolved. |
| `Tensor` or `TensorFactory` cannot be imported | Their ordered model tasks are not complete. | Use metadata types only and check the [roadmap](../planning/roadmap.md). |

## Limitations

Host storage, mutable tensor state, factories, provenance, and expression-building operations are planned. The [Tensor API](../api/tensor-api.md) is the precise reference for what exists now.
