# Tensor API Reference

This document describes the current public surface of `tensor.Tensor`.

Primary class:
- [src/main/java/tensor/Tensor.java](../tensor/Tensor.java)

The focus here is practical API usage:
- what each public operation does
- what parameters it accepts
- what it returns
- short commented examples

## Contents

- [Conventions](#conventions)
- [Execution API](#execution-api)
- [Static Factories](#static-factories)
- [Layout / Shape Operations](#layout--shape-operations)
- [Binary Arithmetic Operations](#binary-arithmetic-operations)
- [Comparison Operations](#comparison-operations)
- [Select Operation](#select-operation)
- [Logical Bool Operations](#logical-bool-operations)
- [Unary / Scalar Operations](#unary--scalar-operations)
- [Reduction Operations](#reduction-operations)
- [Execution Anchor / Autodiff Helpers](#execution-anchor--autodiff-helpers)
- [Metadata and Data Access](#metadata-and-data-access)
- [Backend Selection](#backend-selection)
- [Debug Formatting](#debug-formatting)

## Conventions

- Axis indices are zero-based.
- Shapes are written as arrays such as `[2, 3, 4]`.
- `BOOL` tensors are logical tensors.
- Comparison ops and logical bool ops are nondifferentiable.
- `where(condition, x, y)` is differentiable only in the data branches.
- `all` / `any` are `BOOL`-only reductions and are nondifferentiable.

## Execution API

### `prepare(ExecutionProfile profile)`

Builds a prepared execution for the graph rooted at this tensor.

Parameters:
- `profile`: execution profile containing optimizer, runtime config, and execution mode defaults

Returns:
- `PreparedExecution`

Example:
```java
// Builds a prepared execution object for the graph rooted at y.
ExecutionProfile profile = ExecutionProfile.inferenceDefaults();
PreparedExecution execution = y.prepare(profile);
// Returns: a PreparedExecution bound to the selected optimizer/runtime profile.
```

### `compute(ExecutionProfile profile)`

Builds and executes the graph using the supplied profile.

Parameters:
- `profile`: execution profile to use

Returns:
- nothing; the graph is executed in place

Example:
```java
// Compiles and executes the graph rooted at loss using the training profile.
ExecutionProfile profile = ExecutionProfile.trainingDefaults();
loss.compute(profile);
// Returns: nothing; loss and all dependent tensors are executed in place.
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
// Reuses a prepared execution and runs only the forward pass.
PreparedExecution execution = y.prepare(profile);
y.compute(execution, ExecutionMode.FORWARD);
// Returns: nothing; the prepared execution is run in place.
```

## Static Factories

### `scalar(double value)`
### `scalar(double value, DataType dataType)`

Creates a scalar tensor with shape `[1]`.

Parameters:
- `value`: scalar value
- `dataType`: optional dtype

Returns:
- scalar tensor

Example:
```java
// Creates a default scalar tensor with value 3.0 and shape [1].
Tensor a = Tensor.scalar(3.0);
// Creates a FLOAT64 scalar tensor with value 1.5 and shape [1].
Tensor b = Tensor.scalar(1.5, DataType.FLOAT64);
// Returns: scalar tensors.
```

### `onesLike(Tensor other)`

Creates a dense tensor of ones matching `other.shape` and `other.dtype`.

Parameters:
- `other`: reference tensor

Returns:
- tensor filled with ones

Example:
```java
// Creates a tensor of ones with the same shape and dtype as x.
Tensor mask = Tensor.onesLike(x);
// Returns: a tensor full of ones.
```

### `zerosLike(Tensor other)`

Creates a dense tensor of zeros matching `other.shape` and `other.dtype`.

Parameters:
- `other`: reference tensor

Returns:
- tensor filled with zeros

Example:
```java
// Creates a tensor of zeros with the same shape and dtype as x.
Tensor zeros = Tensor.zerosLike(x);
// Returns: a tensor full of zeros.
```

## Layout / Shape Operations

### `contiguous()`

Materializes the tensor into a dense contiguous layout.

Parameters:
- none

Returns:
- tensor with the same logical shape and values, but contiguous storage

Example:
```java
// Materializes a non-contiguous view into dense contiguous storage.
Tensor dense = view.contiguous();
// Materializes an expanded zero-stride view into dense contiguous storage.
Tensor materialized = expanded.contiguous();
// Returns: contiguous tensors with the same logical values.
```

### `reshape(int... newShape)`

Changes the logical shape while preserving element count.

Parameters:
- `newShape`: requested output shape; may include one inferred `-1`

Returns:
- reshaped tensor

Example:
```java
// Reshapes x to shape [3, 2] without changing element count.
Tensor y = x.reshape(3, 2);
// Reshapes x to [3, inferred] using one -1 placeholder.
Tensor z = x.reshape(3, -1);
// Returns: reshaped tensors.
```

### `expand(int... newShape)`

Expands singleton dimensions using broadcast semantics.
Current implementation is a zero-stride alias view, not a dense materialization.

Parameters:
- `newShape`: target shape; rank may stay the same or increase

Returns:
- expanded view tensor

Example:
```java
// Expands a singleton dimension to produce a broadcast view of shape [2, 3].
Tensor y = x.expand(2, 3);
// Broadcast-expands a bias tensor across the batch dimension.
Tensor z = bias.expand(batch, features);
// Returns: zero-stride broadcast views.
```

### `permute(int... axes)`

Reorders tensor axes.

Parameters:
- `axes`: permutation of all axes

Returns:
- permuted view tensor

Example:
```java
// Swaps the two axes of a rank-2 tensor.
Tensor y = x.permute(1, 0);
// Reorders a rank-3 tensor from [N, C, H] to [N, H, C].
Tensor z = x.permute(0, 2, 1);
// Returns: permuted view tensors.
```

### `transpose()`

Convenience wrapper for rank-2 transpose.

Parameters:
- none

Returns:
- transposed rank-2 tensor

Example:
```java
// Convenience rank-2 transpose.
Tensor y = matrix.transpose();
// Returns: a transposed rank-2 tensor.
```

### `expandDims(int axis)`

Inserts a singleton dimension at `axis`.

Parameters:
- `axis`: insertion position

Returns:
- tensor with one additional dimension of size `1`

Example:
```java
// Inserts a leading singleton axis, e.g. [3, 4] -> [1, 3, 4].
Tensor y = x.expandDims(0);
// Inserts a singleton axis at position 2.
Tensor z = x.expandDims(2);
// Returns: tensors with one extra size-1 axis.
```

### `squeeze(int axis)`

Removes a singleton dimension at `axis`.

Parameters:
- `axis`: axis that must currently have size `1`

Returns:
- tensor with that dimension removed

Example:
```java
// Removes a leading singleton axis.
Tensor y = x.squeeze(0);
// Removes the singleton axis produced earlier by expandDims.
Tensor z = expanded.squeeze(1);
// Returns: tensors with one fewer axis.
```

## Binary Arithmetic Operations

These operations:
- use standard binary broadcasting
- align ranks from the right
- treat missing leading dimensions as `1`
- reduce broadcasted gradients back to original operand shapes during backward execution

### `add(Tensor second)`

Element-wise addition.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise sum

Example:
```java
// Element-wise add with matching shapes.
Tensor y = a.add(b);
// Broadcast-adds bias across matrix rows.
Tensor z = matrix.add(bias);
// Returns: element-wise sums.
```

### `sub(Tensor second)`

Element-wise subtraction.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise difference

Example:
```java
// Element-wise subtract with matching shapes.
Tensor y = a.sub(b);
// Computes residual error between prediction and target.
Tensor z = prediction.sub(target);
// Returns: element-wise differences.
```

### `mul(Tensor second)`

Element-wise multiplication.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise product

Example:
```java
// Element-wise multiply with matching shapes.
Tensor y = a.mul(b);
// Multiplies x by a broadcast-compatible mask tensor.
Tensor z = x.mul(mask);
// Returns: element-wise products.
```

### `div(Tensor second)`

Element-wise division.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise quotient

Example:
```java
// Element-wise divide with matching shapes.
Tensor y = a.div(b);
// Divides x by a broadcast-compatible scale tensor.
Tensor z = x.div(scale);
// Returns: element-wise quotients.
```

### `min(Tensor second)`

Element-wise minimum.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise minimum

Example:
```java
// Element-wise minimum with matching shapes.
Tensor y = a.min(b);
// Clamps logits from above using an element-wise cap tensor.
Tensor z = logits.min(cap);
// Returns: element-wise minima.
```

### `max(Tensor second)`

Element-wise maximum.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise maximum

Example:
```java
// Element-wise maximum with matching shapes.
Tensor y = a.max(b);
// Clamps x from below using an element-wise floor tensor.
Tensor z = x.max(floor);
// Returns: element-wise maxima.
```

### `matmul(Tensor second)`

Matrix multiplication for rank-2 tensors.

Parameters:
- `second`: right-hand matrix

Returns:
- matrix product

Example:
```java
// Multiplies two rank-2 tensors.
Tensor y = a.matmul(b);
// Typical dense layer projection.
Tensor logits = input.matmul(weights);
// Returns: matrix products.
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

Example:
```java
// Builds a BOOL mask where a is strictly greater than b.
Tensor mask = a.greaterThan(b);
// Builds an activation mask above the given threshold.
Tensor active = scores.greaterThan(threshold);
// Returns: BOOL tensors.
```

### `greaterOrEqual(Tensor second)`

Element-wise `left >= right`.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this >= second`

Example:
```java
// Builds a BOOL mask where a is greater than or equal to b.
Tensor mask = a.greaterOrEqual(b);
// Returns: a BOOL tensor.
```

### `lessThan(Tensor second)`

Element-wise `left < right`.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this < second`

Example:
```java
// Builds a BOOL mask where a is strictly less than b.
Tensor mask = a.lessThan(b);
// Returns: a BOOL tensor.
```

### `lessOrEqual(Tensor second)`

Element-wise `left <= right`.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this <= second`

Example:
```java
// Builds a BOOL mask where a is less than or equal to b.
Tensor mask = a.lessOrEqual(b);
// Returns: a BOOL tensor.
```

### `equalTo(Tensor second)`

Element-wise exact equality.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this == second`

Example:
```java
// Builds a BOOL mask of exact equality.
Tensor mask = a.equalTo(b);
// Returns: a BOOL tensor.
```

### `notEqualTo(Tensor second)`

Element-wise exact inequality.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this != second`

Example:
```java
// Builds a BOOL mask of exact inequality.
Tensor mask = a.notEqualTo(b);
// Returns: a BOOL tensor.
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

Example:
```java
// Selects x where mask is true, otherwise yFallback.
Tensor y = Tensor.where(mask, x, yFallback);
// Builds a simple upper clamp through compare/select composition.
Tensor clipped = Tensor.where(x.greaterThan(cap), cap, x);
// Returns: numeric tensors with the promoted branch dtype.
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

Example:
```java
// Logical conjunction of two BOOL tensors.
Tensor mask = a.logicalAnd(b);
// Combines two BOOL masks into one.
Tensor combined = gtMask.logicalAnd(eqMask);
// Returns: BOOL tensors.
```

### `logicalOr(Tensor second)`

Element-wise logical disjunction.

Parameters:
- `second`: right-hand `BOOL` operand

Returns:
- `BOOL` tensor with `true` where at least one operand is true

Example:
```java
// Logical disjunction of two BOOL tensors.
Tensor mask = a.logicalOr(b);
// Returns: a BOOL tensor.
```

### `logicalNot()`

Element-wise logical negation.

Parameters:
- none

Returns:
- `BOOL` tensor with inverted logical values

Example:
```java
// Logical negation of a BOOL tensor.
Tensor inverted = mask.logicalNot();
// Returns: a BOOL tensor.
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
// Arithmetic negation.
Tensor y = x.neg();
// Returns: a tensor with all values multiplied by -1.
```

### `log()`

Natural logarithm applied element-wise.

Parameters:
- none

Returns:
- tensor of `ln(x)` values

Example:
```java
// Natural logarithm applied element-wise.
Tensor y = x.log();
// Returns: a tensor of ln(x) values.
```

### `exp()`

Natural exponential applied element-wise.

Parameters:
- none

Returns:
- tensor of `e^x` values

Example:
```java
// Natural exponential applied element-wise.
Tensor y = x.exp();
// Returns: a tensor of exp(x) values.
```

### `fastExp()`

Approximation-oriented exponential op.

Parameters:
- none

Returns:
- tensor with fast approximate `exp`

Example:
```java
// Fast approximate exponential applied element-wise.
Tensor y = x.fastExp();
// Returns: a tensor of approximate exp(x) values.
```

### `tanh()`

Hyperbolic tangent applied element-wise.

Parameters:
- none

Returns:
- tensor of `tanh(x)` values

Example:
```java
// Hyperbolic tangent applied element-wise.
Tensor y = x.tanh();
// Returns: a tensor of tanh(x) values.
```

### `fastTanh()`

Approximation-oriented hyperbolic tangent op.

Parameters:
- none

Returns:
- tensor with fast approximate `tanh`

Example:
```java
// Fast approximate hyperbolic tangent applied element-wise.
Tensor y = x.fastTanh();
// Returns: a tensor of approximate tanh(x) values.
```

### `pow(double exponent)`

Raises each element to a scalar exponent.

Parameters:
- `exponent`: scalar exponent

Returns:
- tensor of `x^exponent`

Example:
```java
// Squares every element of x.
Tensor y = x.pow(2.0);
// Computes square roots through exponent 0.5.
Tensor z = x.pow(0.5);
// Returns: tensors of x^exponent.
```

### `mul(double scalar)`

Multiplies each element by a scalar.

Parameters:
- `scalar`: scalar multiplier

Returns:
- scaled tensor

Example:
```java
// Scales x by one half.
Tensor y = x.mul(0.5);
// Doubles every element of x.
Tensor z = x.mul(2.0);
// Returns: scaled tensors.
```

### `inv()`

Element-wise reciprocal.

Parameters:
- none

Returns:
- tensor of `1 / x`

Example:
```java
// Element-wise reciprocal.
Tensor y = x.inv();
// Returns: a tensor of 1 / x values.
```

### `sqrt()`

Element-wise square root.

Parameters:
- none

Returns:
- tensor of `sqrt(x)`

Example:
```java
// Element-wise square root.
Tensor y = x.sqrt();
// Returns: a tensor of sqrt(x) values.
```

### `sigmoid()`

Element-wise logistic sigmoid.

Parameters:
- none

Returns:
- tensor of `1 / (1 + exp(-x))`

Example:
```java
// Logistic sigmoid applied element-wise.
Tensor y = logits.sigmoid();
// Returns: a tensor of sigmoid(logits) values.
```

### `clamp(double minValue, double maxValue)`

Clamps every numeric element into the closed interval `[minValue, maxValue]`.

Parameters:
- `minValue`: lower bound
- `maxValue`: upper bound

Returns:
- clamped numeric tensor

Behavior:
- values below `minValue` become `minValue`
- values above `maxValue` become `maxValue`
- values already inside the interval stay unchanged

Example:
```java
// Clips x into the interval [0.0, 1.0].
Tensor y = x.clamp(0.0, 1.0);
// Returns: a tensor whose values are limited to [0.0, 1.0].
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
// Sums all elements of x into a scalar-shaped tensor [1].
Tensor y = x.sum();
// Returns: a scalar-shaped sum tensor.
```

### `sum(int dimension)`

Reduces one axis and removes it from the output shape.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced tensor with one fewer dimension

Example:
```java
// Reduces axis 1 and removes it from the result shape.
Tensor y = x.sum(1);
// Returns: an axis-reduced tensor.
```

### `sum(int dimension, boolean keepDims)`

Reduces one axis and optionally preserves it as size `1`.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis

Returns:
- reduced tensor

Example:
```java
// Reduces axis 1 and keeps it as size 1.
Tensor y = x.sum(1, true);
// Reduces axis 0 and removes it.
Tensor z = x.sum(0, false);
// Returns: reduced tensors.
```

### `mean()`

Reduces all elements to shape `[1]` and divides by element count.

Parameters:
- none

Returns:
- scalar-shaped mean tensor

Example:
```java
// Mean over all elements of x.
Tensor y = x.mean();
// Returns: a scalar-shaped mean tensor.
```

### `mean(int dimension)`

Reduces one axis by arithmetic mean.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced mean tensor

Example:
```java
// Mean reduction over axis 1.
Tensor y = x.mean(1);
// Returns: an axis-reduced mean tensor.
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
// Mean reduction over axis 1 while keeping the axis.
Tensor y = x.mean(1, true);
// Returns: a reduced mean tensor.
```

### `min()`

Reduces all elements to the global minimum.

Parameters:
- none

Returns:
- scalar-shaped minimum tensor

Example:
```java
// Global minimum of x.
Tensor y = x.min();
// Returns: a scalar-shaped minimum tensor.
```

### `min(int dimension)`

Reduces one axis by minimum.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced minimum tensor

Example:
```java
// Minimum reduction over axis 1.
Tensor y = x.min(1);
// Returns: an axis-reduced minimum tensor.
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
// Minimum reduction over axis 1 while keeping the axis.
Tensor y = x.min(1, true);
// Returns: a reduced minimum tensor.
```

### `max()`

Reduces all elements to the global maximum.

Parameters:
- none

Returns:
- scalar-shaped maximum tensor

Example:
```java
// Global maximum of x.
Tensor y = x.max();
// Returns: a scalar-shaped maximum tensor.
```

### `max(int dimension)`

Reduces one axis by maximum.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced maximum tensor

Example:
```java
// Maximum reduction over axis 1.
Tensor y = x.max(1);
// Returns: an axis-reduced maximum tensor.
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
// Maximum reduction over axis 1 while keeping the axis.
Tensor y = x.max(1, true);
// Returns: a reduced maximum tensor.
```

### `all()`

Reduces all elements of a `BOOL` tensor with logical conjunction.

Parameters:
- none

Returns:
- scalar-shaped `BOOL` tensor that is `true` only if all elements are `true`

Example:
```java
// Returns true only if every element in mask is true.
Tensor y = mask.all();
// Returns: a scalar-shaped BOOL tensor.
```

### `all(int dimension)`

Reduces one axis of a `BOOL` tensor with logical conjunction.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced `BOOL` tensor

Example:
```java
// Reduces axis 1 with logical AND.
Tensor y = mask.all(1);
// Returns: an axis-reduced BOOL tensor.
```

### `all(int dimension, boolean keepDims)`

Reduces one axis of a `BOOL` tensor with logical conjunction and optional axis preservation.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis as size `1`

Returns:
- reduced `BOOL` tensor

Example:
```java
// Reduces axis 1 and keeps it as size 1.
Tensor y = mask.all(1, true);
// Returns: a reduced BOOL tensor.
```

### `any()`

Reduces all elements of a `BOOL` tensor with logical disjunction.

Parameters:
- none

Returns:
- scalar-shaped `BOOL` tensor that is `true` if any element is `true`

Example:
```java
// Returns true if at least one element in mask is true.
Tensor y = mask.any();
// Returns: a scalar-shaped BOOL tensor.
```

### `any(int dimension)`

Reduces one axis of a `BOOL` tensor with logical disjunction.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced `BOOL` tensor

Example:
```java
// Reduces axis 1 with logical OR.
Tensor y = mask.any(1);
// Returns: an axis-reduced BOOL tensor.
```

### `any(int dimension, boolean keepDims)`

Reduces one axis of a `BOOL` tensor with logical disjunction and optional axis preservation.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis as size `1`

Returns:
- reduced `BOOL` tensor

Example:
```java
// Reduces axis 1 and keeps it as size 1.
Tensor y = mask.any(1, true);
// Returns: a reduced BOOL tensor.
```

Reduction gradient notes:
- `sum` and `mean` broadcast gradients back to input shape
- `mean` additionally scales by reduced-size reciprocal
- `min` and `max` route gradients only to winning elements
- ties split gradient evenly across winners
- `all` and `any` are nondifferentiable

## Execution Anchor / Autodiff Helpers

### `forwardOutput()`

Creates a forward anchor node used as the execution sink of the current graph.

Parameters:
- none

Returns:
- tensor that aliases the current tensor as forward output anchor

Example:
```java
// Creates the explicit forward sink node for y.
Tensor out = y.forwardOutput();
// Returns: a tensor used as the forward execution anchor.
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
// Executes the internal backward graph builder attached to loss.
loss.buildBackwardGraph();
// Returns: nothing; this is internal graph wiring.
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
// Installs the internal backward graph builder for tensor t.
t.setBackwardFunction(() -> { /* internal backward wiring */ });
// Returns: nothing; this is internal graph wiring.
```

## Metadata and Data Access

### Shape / layout

#### `getShape()`
Returns a defensive copy of the tensor shape.

Example:
```java
// Reads the logical shape of x.
int[] shape = x.getShape();
// Returns: a copy of the shape array.
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
Returns dense strides for a requested shape.

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
Returns the backing storage object.
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
// Reads one logical element from x.
double v = x.getByFlatIndex(3);
// Returns: the value at flat logical index 3 as double.
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
// Formats x for debugging, including shape, strides, and logical values.
System.out.println(x.toStructString());
// Returns: a human-readable debug string.
```
