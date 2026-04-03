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
- [Broadcasting Contract](#broadcasting-contract)
- [Contiguous and Materialization Contract](#contiguous-and-materialization-contract)
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

## Broadcasting Contract

The core broadcasting contract used by tensor operations is:

- ranks align from the right
- missing leading dimensions behave as `1`
- two dimensions are compatible if:
  - they are equal
  - or one side is `1`
- the output dimension size is the maximum of the two compatible dimensions

Examples:
- `[2, 3, 4]` with `[3, 4]` produces `[2, 3, 4]`
- `[3, 4]` with `[2, 1, 4]` produces `[2, 3, 4]`
- `[1, 1, 2, 4]` with `[2, 3, 1, 4]` produces `[2, 3, 2, 4]`

This contract is used by:
- binary arithmetic ops
- comparison ops
- logical binary bool ops
- `where(condition, x, y)` via common broadcast shape resolution across all three inputs

Backward note:
- if an operand was broadcast in forward execution, its gradient is reduced back to the original operand shape

## Contiguous and Materialization Contract

`Tensor` supports both:

- dense contiguous tensors
- non-contiguous or view-like tensors based on shape/stride metadata

Important current behavior:

- `permute(...)` creates a view-like tensor with reordered strides
- `expand(...)` creates a zero-stride broadcast alias view
- `reshape(...)` is a layout-level transform that preserves element count
- `contiguous()` is the canonical explicit materialization path

`contiguous()` is useful when:

- a later kernel performs better on dense layout
- you want to materialize an expanded zero-stride broadcast view
- you want a stable dense tensor independent of the source view layout

Example:
```java
// expanded is a zero-stride broadcast view, not a dense copy.
Tensor expanded = bias.expand(batch, features);
// contiguous() materializes it into dense storage.
Tensor dense = expanded.contiguous();
// Returns: a dense tensor with the same logical values as expanded.
```

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

Use this when you have a view-like tensor whose logical values are correct, but whose storage layout is non-contiguous or zero-stride.
`contiguous()` creates a dense tensor with the same logical values in row-major order.

Parameters:
- none

Returns:
- tensor with the same logical shape and values, but contiguous storage

Example:
```java
Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base");
// base logically represents:
// [[1, 2, 3],
//  [4, 5, 6]]

Tensor permuted = base.permute(1, 0);
// permuted is a view of shape [3, 2] with logical values:
// [[1, 4],
//  [2, 5],
//  [3, 6]]

Tensor dense = permuted.contiguous();
// dense keeps the same logical shape [3, 2]
// and the same logical values:
// [[1, 4],
//  [2, 5],
//  [3, 6]]
// but stores them in contiguous row-major order.
//
// Returns: a dense materialized tensor with the same logical values as permuted.
```

### `reshape(int... newShape)`

Changes the logical shape while preserving element count.

`reshape(...)` does not change values, only how the same flat element sequence is interpreted as dimensions.
The total number of elements before and after reshape must match.

Parameters:
- `newShape`: requested output shape; may include one inferred `-1`

Returns:
- reshaped tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
// x logically represents:
// [[1, 2, 3],
//  [4, 5, 6]]

Tensor y = x.reshape(3, 2);
// y logically represents:
// [[1, 2],
//  [3, 4],
//  [5, 6]]

Tensor z = x.reshape(3, -1);
// -1 is inferred, so z is also shape [3, 2]
// with the same logical values as y.
//
// Returns: tensors with the same elements but a different shape interpretation.
```

### `expand(int... newShape)`

Expands singleton dimensions using broadcast semantics.
Current implementation is a zero-stride alias view, not a dense materialization.

`expand(...)` is valid only when every expanded axis was size `1` in the source or already matches the target size.
It does not create new independent values; it creates a logical broadcast view over the original storage.

Parameters:
- `newShape`: target shape; rank may stay the same or increase

Returns:
- expanded view tensor

Example:
```java
Tensor bias = new Tensor(new double[]{10, 20, 30}, new int[]{1, 3}, null, "bias");
// bias logically represents:
// [[10, 20, 30]]

Tensor y = bias.expand(2, 3);
// y logically represents:
// [[10, 20, 30],
//  [10, 20, 30]]
//
// Important:
// y is not a dense copy yet
// it is a zero-stride broadcast view over bias

Tensor dense = y.contiguous();
// dense has the same logical values as y
// but is now explicitly materialized.
//
// Returns: zero-stride broadcast views, optionally materialized later via contiguous().
```

### `permute(int... axes)`

Reorders tensor axes.

`permute(...)` changes axis order.
It does not change numeric values, but it changes which coordinate maps to which value.

Parameters:
- `axes`: permutation of all axes

Returns:
- permuted view tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
// x logically represents:
// [[1, 2, 3],
//  [4, 5, 6]]

Tensor y = x.permute(1, 0);
// y has shape [3, 2]
// and logically represents:
// [[1, 4],
//  [2, 5],
//  [3, 6]]

Tensor z = y.contiguous();
// z keeps the same logical values as y
// but stores them densely in row-major order.
//
// Returns: permuted view tensors.
```

### `transpose()`

Convenience wrapper for rank-2 transpose.

For rank-2 tensors, `transpose()` is equivalent to `permute(1, 0)`.

Parameters:
- none

Returns:
- transposed rank-2 tensor

Example:
```java
Tensor matrix = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "matrix");
// matrix logically represents:
// [[1, 2, 3],
//  [4, 5, 6]]

Tensor y = matrix.transpose();
// y has shape [3, 2]
// and logically represents:
// [[1, 4],
//  [2, 5],
//  [3, 6]]
//
// Returns: a rank-2 transposed view tensor.
```

### `expandDims(int axis)`

Inserts a singleton dimension at `axis`.

This is a pure shape transform.
It does not change values, only adds an axis of size `1`.

Parameters:
- `axis`: insertion position

Returns:
- tensor with one additional dimension of size `1`

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "x");
// x logically represents:
// [[1, 2],
//  [3, 4]]

Tensor y = x.expandDims(0);
// y has shape [1, 2, 2]
// and logically represents:
// [[[1, 2],
//   [3, 4]]]

Tensor z = x.expandDims(2);
// z has shape [2, 2, 1]
// and logically represents:
// [[[1], [2]],
//  [[3], [4]]]
//
// Returns: tensors with one extra size-1 axis.
```

### `squeeze(int axis)`

Removes a singleton dimension at `axis`.

This is the inverse of `expandDims(...)` when the chosen axis has size `1`.

Parameters:
- `axis`: axis that must currently have size `1`

Returns:
- tensor with that dimension removed

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{1, 2, 2}, null, "x");
// x logically represents:
// [[[1, 2],
//   [3, 4]]]

Tensor y = x.squeeze(0);
// y has shape [2, 2]
// and logically represents:
// [[1, 2],
//  [3, 4]]

Tensor expanded = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2, 1}, null, "expanded");
Tensor z = expanded.squeeze(2);
// z has shape [2, 2]
// and keeps the same logical values.
//
// Returns: tensors with one fewer size-1 axis.
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
Tensor a = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b");

Tensor y = a.add(b);
// y has shape [3] and values [11, 22, 33].

Tensor matrix = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "matrix");
Tensor bias = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "bias");
Tensor z = matrix.add(bias);
// z has shape [2, 3] and values:
// [[11, 22, 33],
//  [14, 25, 36]]
//
// Returns: element-wise sums with optional broadcasting.
```

### `sub(Tensor second)`

Element-wise subtraction.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise difference

Example:
```java
Tensor a = new Tensor(new double[]{5, 7, 9}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "b");

Tensor y = a.sub(b);
// y has shape [3] and values [4, 5, 6].

Tensor prediction = new Tensor(new double[]{0.9, 0.1}, new int[]{2}, null, "prediction");
Tensor target = new Tensor(new double[]{1.0, 0.0}, new int[]{2}, null, "target");
Tensor z = prediction.sub(target);
// z has shape [2] and values [-0.1, 0.1].
//
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
Tensor a = new Tensor(new double[]{2, 3, 4}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b");

Tensor y = a.mul(b);
// y has shape [3] and values [20, 60, 120].

Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor mask = new Tensor(new double[]{1, 0, 1}, new int[]{3}, null, "mask");
Tensor z = x.mul(mask);
// z has shape [2, 3] and values:
// [[1, 0, 3],
//  [4, 0, 6]]
//
// Returns: element-wise products with optional broadcasting.
```

### `div(Tensor second)`

Element-wise division.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise quotient

Example:
```java
Tensor a = new Tensor(new double[]{8, 9, 10}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{2, 3, 5}, new int[]{3}, null, "b");

Tensor y = a.div(b);
// y has shape [3] and values [4, 3, 2].

Tensor x = new Tensor(new double[]{2, 4, 6, 8, 10, 12}, new int[]{2, 3}, null, "x");
Tensor scale = new Tensor(new double[]{2, 2, 3}, new int[]{3}, null, "scale");
Tensor z = x.div(scale);
// z has shape [2, 3] and values:
// [[1, 2, 2],
//  [4, 5, 4]]
//
// Returns: element-wise quotients with optional broadcasting.
```

### `min(Tensor second)`

Element-wise minimum.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise minimum

Example:
```java
Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{2, 4, 3}, new int[]{3}, null, "b");

Tensor y = a.min(b);
// y has shape [3] and values [1, 4, 3].

Tensor logits = new Tensor(new double[]{1, 9, 3, 7}, new int[]{2, 2}, null, "logits");
Tensor cap = new Tensor(new double[]{4}, new int[]{1}, null, "cap");
Tensor z = logits.min(cap);
// z has shape [2, 2] and values:
// [[1, 4],
//  [3, 4]]
//
// Returns: element-wise minima with optional broadcasting.
```

### `max(Tensor second)`

Element-wise maximum.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise maximum

Example:
```java
Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{2, 4, 3}, new int[]{3}, null, "b");

Tensor y = a.max(b);
// y has shape [3] and values [2, 5, 3].

Tensor x = new Tensor(new double[]{-2, 1, -1, 3}, new int[]{2, 2}, null, "x");
Tensor floor = new Tensor(new double[]{0}, new int[]{1}, null, "floor");
Tensor z = x.max(floor);
// z has shape [2, 2] and values:
// [[0, 1],
//  [0, 3]]
//
// Returns: element-wise maxima with optional broadcasting.
```

### `matmul(Tensor second)`

Matrix multiplication for rank-2 tensors.

Parameters:
- `second`: right-hand matrix

Returns:
- matrix product

Example:
```java
Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "a");
Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "b");

Tensor y = a.matmul(b);
// y has shape [2, 2] and values:
// [[19, 22],
//  [43, 50]]
//
// Returns: matrix products for rank-2 tensors.
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
Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{4, 4, 4}, new int[]{3}, null, "b");

Tensor mask = a.greaterThan(b);
// mask has shape [3] and values [false, true, false].

Tensor scores = new Tensor(new double[]{0.2, 0.8, 0.4}, new int[]{3}, null, "scores");
Tensor threshold = new Tensor(new double[]{0.5}, new int[]{1}, null, "threshold");
Tensor active = scores.greaterThan(threshold);
// active has shape [3] and values [false, true, false].
//
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
Tensor a = new Tensor(new double[]{1, 4, 4}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{1, 5, 4}, new int[]{3}, null, "b");

Tensor mask = a.greaterOrEqual(b);
// mask has shape [3] and values [true, false, true].
//
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
Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{4, 4, 4}, new int[]{3}, null, "b");

Tensor mask = a.lessThan(b);
// mask has shape [3] and values [true, false, true].
//
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
Tensor a = new Tensor(new double[]{1, 4, 4}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{1, 5, 4}, new int[]{3}, null, "b");

Tensor mask = a.lessOrEqual(b);
// mask has shape [3] and values [true, true, true].
//
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
Tensor a = new Tensor(new double[]{1, 2, 2}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{1, 0, 2}, new int[]{3}, null, "b");

Tensor mask = a.equalTo(b);
// mask has shape [3] and values [true, false, true].
//
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
Tensor a = new Tensor(new double[]{1, 2, 2}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{1, 0, 2}, new int[]{3}, null, "b");

Tensor mask = a.notEqualTo(b);
// mask has shape [3] and values [false, true, false].
//
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
Tensor mask = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "mask", DataType.BOOL);
Tensor x = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "x");
Tensor yFallback = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "yFallback");

Tensor y = Tensor.where(mask, x, yFallback);
// y has shape [3] and values [10, 2, 30].

Tensor cap = new Tensor(new double[]{15}, new int[]{1}, null, "cap");
Tensor clipped = Tensor.where(x.greaterThan(cap), cap, x);
// clipped has shape [3] and values [10, 15, 15].
//
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
Tensor a = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "a", DataType.BOOL);
Tensor b = new Tensor(new byte[]{1, 1, 0, 0}, new int[]{2, 2}, null, "b", DataType.BOOL);

Tensor mask = a.logicalAnd(b);
// mask has shape [2, 2] and values:
// [[true,  false],
//  [false, false]]
//
// Returns: a BOOL tensor.
```

### `logicalOr(Tensor second)`

Element-wise logical disjunction.

Parameters:
- `second`: right-hand `BOOL` operand

Returns:
- `BOOL` tensor with `true` where at least one operand is true

Example:
```java
Tensor a = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "a", DataType.BOOL);
Tensor b = new Tensor(new byte[]{1, 1, 0, 0}, new int[]{2, 2}, null, "b", DataType.BOOL);

Tensor mask = a.logicalOr(b);
// mask has shape [2, 2] and values:
// [[true,  true],
//  [true,  false]]
//
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
Tensor mask = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "mask", DataType.BOOL);

Tensor inverted = mask.logicalNot();
// inverted has shape [2, 2] and values:
// [[false, true],
//  [false, true]]
//
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
Tensor x = new Tensor(new double[]{1, -2, 3}, new int[]{3}, null, "x");
Tensor y = x.neg();
// y has shape [3] and values [-1, 2, -3].
//
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
Tensor x = new Tensor(new double[]{1, Math.E, Math.E * Math.E}, new int[]{3}, null, "x");
Tensor y = x.log();
// y has shape [3] and values [0, 1, 2].
//
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
Tensor x = new Tensor(new double[]{0, 1, 2}, new int[]{3}, null, "x");
Tensor y = x.exp();
// y has shape [3] and values [1, e, e^2].
//
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
Tensor x = new Tensor(new double[]{0, 1, 2}, new int[]{3}, null, "x");
Tensor y = x.fastExp();
// y has shape [3] and returns an approximate exp(x) result.
//
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
Tensor x = new Tensor(new double[]{-1, 0, 1}, new int[]{3}, null, "x");
Tensor y = x.tanh();
// y has shape [3] and values [tanh(-1), 0, tanh(1)].
//
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
Tensor x = new Tensor(new double[]{-1, 0, 1}, new int[]{3}, null, "x");
Tensor y = x.fastTanh();
// y has shape [3] and returns an approximate tanh(x) result.
//
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
Tensor x = new Tensor(new double[]{4, 9, 16}, new int[]{3}, null, "x");
Tensor y = x.pow(2.0);
// y has values [16, 81, 256].
Tensor z = x.pow(0.5);
// z has values [2, 3, 4].
//
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
Tensor x = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "x");
Tensor y = x.mul(0.5);
// y has values [0.5, 1.0, 1.5].
Tensor z = x.mul(2.0);
// z has values [2, 4, 6].
//
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
Tensor x = new Tensor(new double[]{2, 4, 8}, new int[]{3}, null, "x");
Tensor y = x.inv();
// y has values [0.5, 0.25, 0.125].
//
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
Tensor x = new Tensor(new double[]{4, 9, 16}, new int[]{3}, null, "x");
Tensor y = x.sqrt();
// y has values [2, 3, 4].
//
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
Tensor logits = new Tensor(new double[]{-2, 0, 2}, new int[]{3}, null, "logits");
Tensor y = logits.sigmoid();
// y has values [sigmoid(-2), 0.5, sigmoid(2)].
//
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
Tensor x = new Tensor(new double[]{-2.0, -0.5, 0.5, 3.0}, new int[]{4}, null, "x");
Tensor y = x.clamp(0.0, 1.0);
// y has values [0.0, 0.0, 0.5, 1.0].
//
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
Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "x");
Tensor y = x.sum();
// y has shape [1] and value [10].
//
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
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.sum(1);
// y has shape [2] and values [6, 15].
//
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
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.sum(1, true);
// y has shape [2, 1] and values [6, 15].
Tensor z = x.sum(0, false);
// z has shape [3] and values [5, 7, 9].
//
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
Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "x");
Tensor y = x.mean();
// y has shape [1] and value [2.5].
//
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
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.mean(1);
// y has shape [2] and values [2.0, 5.0].
//
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
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.mean(1, true);
// y has shape [2, 1] and values [2.0, 5.0].
//
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
Tensor x = new Tensor(new double[]{4, 1, 9, 2}, new int[]{4}, null, "x");
Tensor y = x.min();
// y has shape [1] and value [1].
//
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
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.min(1);
// y has shape [2] and values [1, 4].
//
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
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.min(1, true);
// y has shape [2, 1] and values [1, 4].
//
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
Tensor x = new Tensor(new double[]{4, 1, 9, 2}, new int[]{4}, null, "x");
Tensor y = x.max();
// y has shape [1] and value [9].
//
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
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.max(1);
// y has shape [2] and values [3, 6].
//
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
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.max(1, true);
// y has shape [2, 1] and values [3, 6].
//
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
Tensor mask = new Tensor(new byte[]{1, 1, 0, 1}, new int[]{4}, null, "mask", DataType.BOOL);
Tensor y = mask.all();
// y has shape [1] and value [false].
//
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
Tensor mask = new Tensor(new byte[]{1, 1, 0, 1, 1, 1}, new int[]{2, 3}, null, "mask", DataType.BOOL);
Tensor y = mask.all(1);
// y has shape [2] and values [false, true].
//
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
Tensor mask = new Tensor(new byte[]{1, 1, 0, 1, 1, 1}, new int[]{2, 3}, null, "mask", DataType.BOOL);
Tensor y = mask.all(1, true);
// y has shape [2, 1] and values [false, true].
//
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
Tensor mask = new Tensor(new byte[]{0, 0, 0, 1}, new int[]{4}, null, "mask", DataType.BOOL);
Tensor y = mask.any();
// y has shape [1] and value [true].
//
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
Tensor mask = new Tensor(new byte[]{1, 1, 0, 1, 1, 1}, new int[]{2, 3}, null, "mask", DataType.BOOL);
Tensor y = mask.any(1);
// y has shape [2] and values [true, true].
//
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
Tensor mask = new Tensor(new byte[]{1, 1, 0, 1, 1, 1}, new int[]{2, 3}, null, "mask", DataType.BOOL);
Tensor y = mask.any(1, true);
// y has shape [2, 1] and values [true, true].
//
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
