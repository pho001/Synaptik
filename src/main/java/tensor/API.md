# Tensor API Reference

This document describes the current public surface of `tensor.Tensor`.

Primary class:
- [src/main/java/tensor/Tensor.java](../tensor/Tensor.java)

This reference focuses on:
- what each public operation does
- what parameters it accepts
- what it returns
- short usage examples

## Conventions

- Axis indices are zero-based.
- Negative axes are accepted only where the underlying implementation explicitly supports them.
- Shapes are written as arrays such as `[2, 3, 4]`.
- `BOOL` tensors are logical tensors.
- Comparison ops and logical bool ops are nondifferentiable.
- `where(condition, x, y)` is differentiable only in the data branches.

## Preferred Execution API

### `prepare(ExecutionProfile profile)`

Builds a prepared execution object for the current tensor graph using an execution profile.

Parameters:
- `profile`: execution profile containing optimizer, runtime config, and execution mode defaults

Returns:
- `PreparedExecution`

Example:
```java
ExecutionProfile profile = ExecutionProfile.inferenceDefaults();
PreparedExecution execution = y.prepare(profile);
```

### `compute(ExecutionProfile profile)`

Builds and runs the graph using the supplied execution profile.

Parameters:
- `profile`: execution profile to use

Returns:
- nothing; the tensor graph is executed in place

Example:
```java
ExecutionProfile profile = ExecutionProfile.trainingDefaults();
loss.compute(profile);
```

### `compute(PreparedExecution execution, ExecutionMode mode)`

Runs a previously prepared execution.

Parameters:
- `execution`: prepared execution object
- `mode`: `FORWARD` or `FORWARD_BACKWARD`

Returns:
- nothing; the prepared execution is run in place

Example:
```java
PreparedExecution execution = y.prepare(profile);
y.compute(execution, ExecutionMode.FORWARD);
```

## Static Factories

### `scalar(double value)`
### `scalar(double value, DataType dataType)`

Creates a scalar tensor with shape `[1]`.

Parameters:
- `value`: scalar value
- `dataType`: optional output dtype

Returns:
- scalar tensor

Examples:
```java
Tensor a = Tensor.scalar(3.0);
Tensor b = Tensor.scalar(1.5, DataType.FLOAT64);
```

### `onesLike(Tensor other)`

Creates a dense numeric tensor filled with ones and matching `other.shape`.

Parameters:
- `other`: reference tensor

Returns:
- tensor with the same shape and dtype as `other`

Example:
```java
Tensor mask = Tensor.onesLike(x);
```

### `zerosLike(Tensor other)`

Creates a dense numeric tensor filled with zeros and matching `other.shape`.

Parameters:
- `other`: reference tensor

Returns:
- tensor with the same shape and dtype as `other`

Example:
```java
Tensor zeros = Tensor.zerosLike(x);
```

## Layout / Shape Operations

### `contiguous()`

Materializes the tensor into a dense contiguous layout.

Parameters:
- none

Returns:
- tensor with the same logical shape and values, but contiguous storage

Examples:
```java
Tensor dense = view.contiguous();
Tensor materialized = expanded.contiguous();
```

### `reshape(int... newShape)`

Changes the logical shape while preserving element count.

Parameters:
- `newShape`: requested output shape; may include one inferred `-1`

Returns:
- reshaped tensor

Examples:
```java
Tensor y = x.reshape(3, 2);
Tensor z = x.reshape(3, -1);
```

### `expand(int... newShape)`

Expands singleton dimensions using broadcast semantics.
Current implementation is a zero-stride alias view, not a dense materialization.

Parameters:
- `newShape`: target shape; rank may stay the same or increase

Returns:
- expanded view tensor

Examples:
```java
Tensor y = x.expand(2, 3);
Tensor z = bias.expand(batch, features);
```

### `permute(int... axes)`

Reorders tensor axes.

Parameters:
- `axes`: permutation of all axes

Returns:
- permuted view tensor

Examples:
```java
Tensor y = x.permute(1, 0);
Tensor z = x.permute(0, 2, 1);
```

### `transpose()`

Convenience wrapper for rank-2 transpose.

Parameters:
- none

Returns:
- transposed rank-2 tensor

Examples:
```java
Tensor y = matrix.transpose();
```

### `expandDims(int axis)`

Inserts a singleton dimension at `axis`.

Parameters:
- `axis`: insertion position

Returns:
- tensor with one additional dimension of size `1`

Examples:
```java
Tensor y = x.expandDims(0);
Tensor z = x.expandDims(2);
```

### `squeeze(int axis)`

Removes a singleton dimension at `axis`.

Parameters:
- `axis`: axis that must currently have size `1`

Returns:
- tensor with that dimension removed

Examples:
```java
Tensor y = x.squeeze(0);
Tensor z = expanded.squeeze(1);
```

## Binary Arithmetic Operations

These operations follow the standard binary broadcasting contract:
- ranks align from the right
- missing leading dimensions behave as `1`
- dimensions are compatible if equal or one side is `1`
- gradients are reduced back to original operand shapes in backward execution

### `add(Tensor second)`

Element-wise addition.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise sum

Examples:
```java
Tensor y = a.add(b);
Tensor z = matrix.add(bias);
```

### `sub(Tensor second)`

Element-wise subtraction.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise difference

Examples:
```java
Tensor y = a.sub(b);
Tensor z = prediction.sub(target);
```

### `mul(Tensor second)`

Element-wise multiplication.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise product

Examples:
```java
Tensor y = a.mul(b);
Tensor z = x.mul(mask);
```

### `div(Tensor second)`

Element-wise division.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise quotient

Examples:
```java
Tensor y = a.div(b);
Tensor z = x.div(scale);
```

### `min(Tensor second)`

Element-wise minimum.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise minimum

Examples:
```java
Tensor y = a.min(b);
Tensor z = logits.min(cap);
```

### `max(Tensor second)`

Element-wise maximum.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise maximum

Examples:
```java
Tensor y = a.max(b);
Tensor z = x.max(floor);
```

### `matmul(Tensor second)`

Matrix multiplication for rank-2 tensors.

Parameters:
- `second`: right-hand matrix

Returns:
- matrix product

Examples:
```java
Tensor y = a.matmul(b);
Tensor logits = input.matmul(weights);
```

## Comparison Operations

Comparison ops:
- require numeric inputs
- use the same binary broadcasting contract as arithmetic binary ops
- return `BOOL` tensors
- are nondifferentiable

### `greaterThan(Tensor second)`

Element-wise `left > right`.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this > second`

Examples:
```java
Tensor mask = a.greaterThan(b);
Tensor active = scores.greaterThan(threshold);
```

### `greaterOrEqual(Tensor second)`

Element-wise `left >= right`.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this >= second`

Examples:
```java
Tensor mask = a.greaterOrEqual(b);
```

### `lessThan(Tensor second)`

Element-wise `left < right`.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this < second`

Examples:
```java
Tensor mask = a.lessThan(b);
```

### `lessOrEqual(Tensor second)`

Element-wise `left <= right`.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this <= second`

Examples:
```java
Tensor mask = a.lessOrEqual(b);
```

### `equalTo(Tensor second)`

Element-wise exact equality.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this == second`

Examples:
```java
Tensor mask = a.equalTo(b);
```

### `notEqualTo(Tensor second)`

Element-wise exact inequality.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this != second`

Examples:
```java
Tensor mask = a.notEqualTo(b);
```

## Select Operation

### `where(Tensor condition, Tensor ifTrue, Tensor ifFalse)` (static)

Element-wise branch selection.

Parameters:
- `condition`: `BOOL` tensor
- `ifTrue`: numeric tensor used where condition is true
- `ifFalse`: numeric tensor used where condition is false

Returns:
- numeric tensor with common broadcasted output shape and promoted branch dtype

Behavior:
- `condition`, `ifTrue`, and `ifFalse` are broadcast to a common output shape
- output dtype is the promoted numeric dtype of `ifTrue` and `ifFalse`
- gradient flows only through selected data branches
- `condition` itself has no gradient

Examples:
```java
Tensor y = Tensor.where(mask, x, yFallback);
Tensor clipped = Tensor.where(x.greaterThan(cap), cap, x);
```

## Logical Bool Operations

Logical bool ops:
- require `BOOL` tensors
- return `BOOL` tensors
- are nondifferentiable

### `logicalAnd(Tensor second)`

Element-wise logical conjunction.

Parameters:
- `second`: right-hand `BOOL` operand

Returns:
- `BOOL` tensor with `true` where both operands are true

Examples:
```java
Tensor mask = a.logicalAnd(b);
Tensor combined = gtMask.logicalAnd(eqMask);
```

### `logicalOr(Tensor second)`

Element-wise logical disjunction.

Parameters:
- `second`: right-hand `BOOL` operand

Returns:
- `BOOL` tensor with `true` where at least one operand is true

Examples:
```java
Tensor mask = a.logicalOr(b);
```

### `logicalNot()`

Element-wise logical negation.

Parameters:
- none

Returns:
- `BOOL` tensor with inverted logical values

Examples:
```java
Tensor inverted = mask.logicalNot();
```

## Unary / Scalar Operations

### `neg()`

Arithmetic negation.

Parameters:
- none

Returns:
- tensor with all elements multiplied by `-1`

Example:
```java
Tensor y = x.neg();
```

### `log()`

Natural logarithm applied element-wise.

Parameters:
- none

Returns:
- tensor of `ln(x)` values

Example:
```java
Tensor y = x.log();
```

### `exp()`

Natural exponential applied element-wise.

Parameters:
- none

Returns:
- tensor of `e^x` values

Example:
```java
Tensor y = x.exp();
```

### `fastExp()`

Approximation-oriented exponential op.

Parameters:
- none

Returns:
- tensor with fast approximate `exp`

Example:
```java
Tensor y = x.fastExp();
```

### `tanh()`

Hyperbolic tangent applied element-wise.

Parameters:
- none

Returns:
- tensor of `tanh(x)` values

Example:
```java
Tensor y = x.tanh();
```

### `fastTanh()`

Approximation-oriented hyperbolic tangent op.

Parameters:
- none

Returns:
- tensor with fast approximate `tanh`

Example:
```java
Tensor y = x.fastTanh();
```

### `pow(double exponent)`

Raises each element to a scalar exponent.

Parameters:
- `exponent`: scalar exponent

Returns:
- tensor of `x^exponent`

Examples:
```java
Tensor y = x.pow(2.0);
Tensor z = x.pow(0.5);
```

### `mul(double scalar)`

Multiplies each element by a scalar.

Parameters:
- `scalar`: scalar multiplier

Returns:
- scaled tensor

Examples:
```java
Tensor y = x.mul(0.5);
Tensor z = x.mul(2.0);
```

### `inv()`

Element-wise reciprocal.

Parameters:
- none

Returns:
- tensor of `1 / x`

Example:
```java
Tensor y = x.inv();
```

### `sqrt()`

Element-wise square root.

Parameters:
- none

Returns:
- tensor of `sqrt(x)`

Example:
```java
Tensor y = x.sqrt();
```

### `sigmoid()`

Element-wise logistic sigmoid.

Parameters:
- none

Returns:
- tensor of `1 / (1 + exp(-x))`

Example:
```java
Tensor y = logits.sigmoid();
```

## Reduction Operations

Reductions:
- reduce one axis or all elements
- optionally preserve the reduced axis via `keepDims`

### `sum()`

Reduces all elements to a tensor of shape `[1]`.

Parameters:
- none

Returns:
- scalar-shaped sum tensor

Example:
```java
Tensor y = x.sum();
```

### `sum(int dimension)`

Reduces one axis and removes it from the output shape.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced tensor with one fewer dimension

Example:
```java
Tensor y = x.sum(1);
```

### `sum(int dimension, boolean keepDims)`

Reduces one axis and optionally preserves it as size `1`.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis

Returns:
- reduced tensor

Examples:
```java
Tensor y = x.sum(1, true);
Tensor z = x.sum(0, false);
```

### `mean()`

Reduces all elements to shape `[1]` and divides by element count.

Parameters:
- none

Returns:
- scalar-shaped mean tensor

Example:
```java
Tensor y = x.mean();
```

### `mean(int dimension)`

Reduces one axis by arithmetic mean.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced mean tensor

Example:
```java
Tensor y = x.mean(1);
```

### `mean(int dimension, boolean keepDims)`

Axis mean with optional reduced-axis preservation.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis as size `1`

Returns:
- reduced mean tensor

Example:
```java
Tensor y = x.mean(1, true);
```

### `min()`

Reduces all elements to the global minimum.

Parameters:
- none

Returns:
- scalar-shaped minimum tensor

Example:
```java
Tensor y = x.min();
```

### `min(int dimension)`

Reduces one axis by minimum.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced minimum tensor

Example:
```java
Tensor y = x.min(1);
```

### `min(int dimension, boolean keepDims)`

Axis minimum with optional reduced-axis preservation.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis as size `1`

Returns:
- reduced minimum tensor

Example:
```java
Tensor y = x.min(1, true);
```

### `max()`

Reduces all elements to the global maximum.

Parameters:
- none

Returns:
- scalar-shaped maximum tensor

Example:
```java
Tensor y = x.max();
```

### `max(int dimension)`

Reduces one axis by maximum.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced maximum tensor

Example:
```java
Tensor y = x.max(1);
```

### `max(int dimension, boolean keepDims)`

Axis maximum with optional reduced-axis preservation.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis as size `1`

Returns:
- reduced maximum tensor

Example:
```java
Tensor y = x.max(1, true);
```

Reduction gradient notes:
- `sum` and `mean` broadcast gradients back to input shape
- `mean` additionally scales by reduced-size reciprocal
- `min` and `max` route gradients only to winning elements
- ties split gradient evenly across winners

## Execution Anchor / Autodiff Helpers

### `forwardOutput()`

Creates a forward anchor node used as the execution sink of the current graph.

Parameters:
- none

Returns:
- tensor that aliases the current tensor as forward output anchor

Example:
```java
Tensor out = y.forwardOutput();
```

### `buildBackwardGraph()`

Runs the stored backward graph builder for this tensor.
This is internal graph wiring machinery, not a typical user-facing call.

Parameters:
- none

Returns:
- nothing

Example:
```java
loss.buildBackwardGraph();
```

### `setBackwardFunction(Runnable backwardFunction)`

Registers the backward graph builder for this tensor.
This is internal wiring API.

Parameters:
- `backwardFunction`: callback that builds or accumulates backward graph nodes

Returns:
- nothing

Example:
```java
t.setBackwardFunction(() -> { /* internal backward wiring */ });
```

## Metadata and Data Access

### Shape / layout

#### `getShape()`
Returns a defensive copy of the tensor shape.

Example:
```java
int[] shape = x.getShape();
```

#### `getShapeUnsafe()`
Returns the internal shape reference.
Use only in internal/runtime code.

#### `getStrides()`
Returns a defensive copy of strides.

#### `getStridesUnsafe()`
Returns the internal stride reference.
Use only in internal/runtime code.

#### `getStride(int index)`
Returns stride for one axis.

#### `getDimensionAt(int index)`
Returns dimension size at one axis.

#### `getFlatDataSize()`
Returns logical element count.

#### `isContiguous()`
Returns whether the tensor layout is contiguous.

#### `getFlatIndex(int[] indices)`
Maps multi-dimensional indices to flat index according to current layout.

#### `getSpatialIndex(int index)`
Maps a flat index back to logical coordinates.

#### `computeStrides(int[] shape)`
Static-like helper on the instance for dense strides of `shape`.

#### `computeStrides()`
Returns a copy of current strides.

#### `calculateSize(int[] dimensions)`
Returns total element count for a shape.

### Data / dtype / storage

#### `getDataType()`
Returns tensor dtype.

#### `setDataType(DataType dataType)`
Changes dtype within numeric families.
`BOOL <-> numeric` implicit conversion is not supported.

#### `getStorage()`
Returns backing storage object.
This is mainly internal/runtime oriented.

#### `getFloat64Data()`
Returns raw `double[]` storage when dtype is `FLOAT64`, otherwise `null`.

#### `getFloat32Data()`
Returns raw `float[]` storage when dtype is `FLOAT32`, otherwise `null`.

#### `getFloat16Data()`
Returns raw `short[]` storage when dtype is `FLOAT16`, otherwise `null`.

#### `getBoolData()`
Returns raw `byte[]` storage when dtype is `BOOL`, otherwise `null`.

#### `getData()`
Legacy F64-only raw accessor.
Supported only for `FLOAT64`.

#### `setData(double[] data)`
Replaces tensor data for numeric tensors using `double[]` input.

#### `setData(float[] data)`
Replaces tensor data using `float[]` input.

#### `setData(short[] data)`
Replaces tensor data using `FLOAT16` raw storage input.

#### `setData(byte[] data)`
Replaces tensor data using raw `BOOL` storage input.

#### `setFloat32Data(float[] data)`
F32-only convenience raw setter.

#### `getByFlatIndex(int index)`
Reads one logical element as `double`.

Example:
```java
double v = x.getByFlatIndex(3);
```

#### `setDataAt(int flatIndex, double value)`
Writes one logical element for numeric tensors.
Broadcast views are not writable through this API.

#### `toDoubleArrayCopy()`
Returns logical tensor contents as `double[]`.

#### `toBooleanArrayCopy()`
Returns logical tensor contents as `boolean[]` for `BOOL` tensors.

#### `scalarAsDouble()`
Returns scalar value for numeric scalar tensors.

### Labels / grad / graph wiring

#### `getLabel()`
Returns tensor label.

#### `setLabel(String label)`
Sets tensor label.

#### `getRequiresGrad()`
Returns whether this tensor participates in gradient tracking.

#### `setRequiresGrad(boolean requiresGrad)`
Enables or disables gradient tracking.

#### `getGradient()`
Returns gradient tensor if present.

#### `setGradient(Tensor gradient)`
Assigns gradient tensor.

#### `getPrevTensors()`
Returns predecessor tensors in the graph.

#### `setPrevTensors(List<Tensor> prevTensors)`
Replaces predecessor list.

#### `getOperation()`
Returns operation descriptor for this node.

#### `setOperation(Operation operation)`
Assigns operation descriptor.

#### `isBackward()`
Returns whether this node belongs to backward graph execution.

#### `setBackward(boolean backward)`
Marks or unmarks node as backward-stage node.

#### `topologicalSort()`
Returns topological order rooted at this tensor.

#### `markDataViewStale()`
Internal runtime helper; currently effectively a no-op.

#### `aliasRuntimeFrom(Tensor source)`
Internal runtime helper that aliases storage from another tensor.

#### `copyDataFrom(Tensor source)`
Internal/runtime-oriented typed data copy between tensors of same shape and dtype.

## Backend Selection

### `setBackend(ComputeBackend backend)`

Forces a backend for this tensor node.

### `resolveBackend()`

Returns the effective backend:
- forced backend if set
- otherwise `CPU`

## Debug Formatting

### `toStructString()`

Returns a string representation of shape, strides, and logical values.

Example:
```java
System.out.println(x.toStructString());
```
