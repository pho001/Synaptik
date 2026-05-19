<!-- generated-by: codex-docs-update -->
# Sequence Tensor Primitives

Navigation: [Index](index.md#recommended-reading-paths) | [Quickstart](quickstart.md#sequence-shaped-tensors) | [Tensor API](tensor-api.md#layout-and-view-operations) | [Public API](public-api.md#tensor) | [Examples](examples.md#sequence-shaped-tensor)

Chapters: [Scope](#scope) | [Shape Vocabulary](#shape-vocabulary) | [Factory And Shape Helpers](#factory-and-shape-helpers) | [N-D Linear](#n-d-linear) | [Stack And Unstack](#stack-and-unstack) | [Axis Indexing](#axis-indexing) | [Masked Reductions](#masked-reductions) | [Masked Cross Entropy](#masked-cross-entropy) | [Autograd Contract](#autograd-contract) | [Implementation Map](#implementation-map) | [Common Mistakes](#common-mistakes)

This document explains the general N-D tensor operations that make sequence-shaped workloads practical in Synaptik.

The key idea is simple:

```text
Use one tensor with shape [batch, time, features]
instead of a Java Tensor[] where each array entry is one timestep.
```

Synaptik still remains a tensor/autograd engine. It does not introduce high-level neural-network concepts such as `Layer`, `Model`, `RNN`, `LSTM`, or `GRU`. A consumer framework can build those abstractions above Synaptik by calling the primitive tensor operations documented here.

## Scope

This feature set is about general tensor mechanics:

- creating common tensors without low-level constructor boilerplate
- projecting the last dimension of an N-D tensor with `linear`
- converting several same-shaped tensors into one higher-rank tensor with `stack`
- splitting a tensor along an axis with `unstack`
- slicing or taking positions along one axis
- reducing padded sequence data with a BOOL validity mask
- computing masked cross entropy for padded classification targets

It is not about sequence-specific layers.

That distinction matters because these operations are useful outside recurrent networks:

- token classification: `[batch, time, classes]`
- sensor windows: `[batch, time, channels]`
- spectrogram-like data: `[batch, frames, bins]`
- batched trajectories: `[batch, step, state]`
- generic N-D features: `[..., features]`

## Shape Vocabulary

Synaptik writes shapes as integer arrays:

```text
[2, 3, 4]
```

For sequence examples, this often means:

```text
[batch, time, features]
```

Concrete meaning:

```text
batch    = how many independent examples are processed together
time     = how many sequence positions each example has
features = how many numbers describe each position
```

So shape `[2, 3, 4]` means:

```text
2 sequences
3 timesteps per sequence
4 feature values per timestep
```

Synaptik uses row-major logical order for dense examples. Row-major means the last dimension changes fastest. For shape `[2, 3, 2]`, flat values are grouped like this:

```text
flat values:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

logical tensor [batch=2, time=3, features=2]:
batch 0:
  time 0: [1, 2]
  time 1: [3, 4]
  time 2: [5, 6]

batch 1:
  time 0: [7, 8]
  time 1: [9, 10]
  time 2: [11, 12]
```

## Factory And Shape Helpers

The public `Tensor` API includes factory helpers for common tensors:

```java
Tensor.zeros(int[] shape)
Tensor.zeros(int[] shape, DataType dataType)
Tensor.zeros(int[] shape, DataType dataType, String label)

Tensor.ones(int[] shape)
Tensor.ones(int[] shape, DataType dataType)
Tensor.ones(int[] shape, DataType dataType, String label)

Tensor.randn(int[] shape)
Tensor.randn(int[] shape, double mean, double stdDev, DataType dataType, String label)

Tensor.arange(int start, int end, int step, DataType dataType)
```

Use them when the tensor values are regular and the constructor would only add noise.

Example:

```java
Tensor weights = Tensor.randn(new int[]{4, 8}, 0.0, 0.02, DataType.FLOAT64, "weights");
Tensor bias = Tensor.zeros(new int[]{8}, DataType.FLOAT64, "bias");
Tensor mask = Tensor.ones(new int[]{2, 3}, DataType.BOOL, "validMask");
Tensor timeIds = Tensor.arange(0, 3, 1, DataType.INT32);
```

Results:

```text
weights shape = [4, 8]
bias shape    = [8]
mask shape    = [2, 3]
timeIds       = [0, 1, 2]
```

Important dtype rules:

- `zeros` and `ones` support floating dtypes, integer dtypes, and `BOOL`.
- `randn` supports floating dtypes only: `FLOAT64`, `FLOAT32`, and `BFLOAT16`.
- `arange` supports numeric non-BOOL dtypes and rejects empty ranges.
- `onesLike` and `zerosLike` preserve dtype, including `BOOL`.

The shape helper methods reduce boilerplate in consumer code:

```java
int rank = x.rank();
int elements = x.size();
int features = x.lastDim();
boolean isSequenceBatch = x.shapeEquals(2, 3, 4);
int[] shape = x.shapeCopy();
```

Meanings:

```text
rank()       = number of dimensions
size()       = total logical element count
lastDim()    = size of the final dimension
shapeEquals  = exact shape comparison
shapeCopy()  = defensive copy of the logical shape
```

`shapeCopy()` is intentionally safe for application code. `getShapeUnsafe()` still exists for low-level infrastructure, but it exposes the internal shape array and should not be used as ordinary modeling API.

## N-D Linear

`linear` projects the last dimension of the input and preserves every leading dimension.

Shape contract:

```text
input  shape [..., inFeatures]
weight shape [inFeatures, outFeatures]
bias   shape [outFeatures] or [1, outFeatures]
output shape [..., outFeatures]
```

The `...` means any number of leading dimensions. This is useful for sequences because `[batch, time]` is just a leading prefix.

### Rank-2 Example

```java
Tensor x = new Tensor(new double[]{
        1, 2,
        3, 4
}, new int[]{2, 2}, null, "x", DataType.FLOAT64);

Tensor w = new Tensor(new double[]{
        10, 20,
        30, 40
}, new int[]{2, 2}, null, "w", DataType.FLOAT64);

Tensor b = new Tensor(new double[]{1, 2}, new int[]{2}, null, "b", DataType.FLOAT64);

Tensor y = x.linear(w, b).compute();
```

Calculation:

```text
row 0 = [1, 2] @ [[10, 20], [30, 40]] + [1, 2]
      = [70, 100] + [1, 2]
      = [71, 102]

row 1 = [3, 4] @ [[10, 20], [30, 40]] + [1, 2]
      = [150, 220] + [1, 2]
      = [151, 222]
```

Output:

```text
y shape  = [2, 2]
y values = [71, 102, 151, 222]
```

### Sequence Example

```java
Tensor x = new Tensor(new double[]{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,

        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
}, new int[]{2, 3, 4}, null, "x", DataType.FLOAT64);

Tensor w = Tensor.randn(new int[]{4, 5}, 0.0, 0.02, DataType.FLOAT64, "w");
Tensor b = Tensor.zeros(new int[]{5}, DataType.FLOAT64, "b");

Tensor y = x.linear(w, b);
```

Shapes:

```text
x = [batch, time, inFeatures] = [2, 3, 4]
w = [inFeatures, outFeatures] = [4, 5]
b = [outFeatures] = [5]
y = [batch, time, outFeatures] = [2, 3, 5]
```

The operation is equivalent to applying the same weight matrix to every `[features]` vector at every batch and time position. You do not need to loop over timesteps in Java.

### Bias Broadcasting

Both bias shapes below are valid:

```java
Tensor b1 = Tensor.zeros(new int[]{5}, DataType.FLOAT64, "biasVector");
Tensor b2 = Tensor.zeros(new int[]{1, 5}, DataType.FLOAT64, "biasRow");
```

Both broadcast over all prefix dimensions:

```text
input shape = [2, 3, 4]
linear output before bias = [2, 3, 5]
bias [5]      broadcasts as [1, 1, 5]
bias [1, 5]   broadcasts as [1, 1, 5]
```

## Stack And Unstack

`stack` converts several same-shaped tensors into one tensor with a new axis.

Signature:

```java
static Tensor Tensor.stack(int axis, Tensor... inputs)
```

Shape rule:

```text
each input shape = [B, F]
stack(0, inputs) -> [T, B, F]
stack(1, inputs) -> [B, T, F]
stack(2, inputs) -> [B, F, T]
```

Example:

```java
Tensor t0 = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "t0", DataType.FLOAT64);
Tensor t1 = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "t1", DataType.FLOAT64);
Tensor t2 = new Tensor(new double[]{9, 10, 11, 12}, new int[]{2, 2}, null, "t2", DataType.FLOAT64);

Tensor seq = Tensor.stack(1, t0, t1, t2).compute();
```

Output shape:

```text
seq shape = [2, 3, 2]
```

Logical values:

```text
batch 0:
  time 0: [1, 2]
  time 1: [5, 6]
  time 2: [9, 10]

batch 1:
  time 0: [3, 4]
  time 1: [7, 8]
  time 2: [11, 12]
```

`unstack` goes the other way:

```java
Tensor[] timesteps = seq.unstack(1);
```

Shape rule:

```text
seq shape = [B, T, F]
seq.unstack(1) returns T tensors
each returned tensor shape = [B, F]
```

`stack` validates that every input has the same shape and dtype. `unstack` validates that the selected axis exists. Both are gradient-safe because they are built from existing `expandDims`, `concat`, and `select` primitives.

## Axis Indexing

Sequence code usually needs two kinds of axis selection:

1. a contiguous range, such as "first 10 timesteps"
2. explicit positions, such as "first and last timestep"

### Contiguous Range: `sliceAxis`

```java
Tensor firstTwo = seq.sliceAxis(1, 0, 2);
```

If `seq.shape = [2, 3, 2]`, then:

```text
axis = 1
fromInclusive = 0
toExclusive = 2
output shape = [2, 2, 2]
```

This is the ergonomic one-axis wrapper over the more general `slice(starts, ends, axes, steps)` API. It uses positive step `1`.

### Explicit Positions: `take`

```java
Tensor endpoints = seq.take(1, new int[]{0, 2});
```

`take` is a convenience wrapper over ONNX-style `gatherAxis`.

Shape rule:

```text
input shape  = dataShape
indices shape = indicesShape
axis = selected input axis
output shape = dataShape[:axis] + indicesShape + dataShape[axis + 1:]
```

For `seq.shape = [2, 3, 2]` and `indices.shape = [2]`:

```text
output shape = [2] + [2] + [2] = [2, 2, 2]
```

Example with concrete values:

```java
Tensor x = new Tensor(new double[]{
        1, 2, 3, 4,
        5, 6, 7, 8
}, new int[]{2, 4}, null, "x", DataType.FLOAT64);

Tensor y = x.take(1, new int[]{3, 1}).compute();
```

Output:

```text
x = [[1, 2, 3, 4],
     [5, 6, 7, 8]]

take axis 1 with indices [3, 1]

y = [[4, 2],
     [8, 6]]
```

Use `takeAlongAxis` when your index tensor already has the desired output shape and must match the input on every non-axis dimension. Use `take`/`gatherAxis` when you want ONNX Gather semantics where the index tensor shape is inserted at the selected axis.

## Masked Reductions

Padded sequences need a way to ignore padding.

Synaptik uses `BOOL` masks:

```text
true  = valid data
false = padding / ignored data
```

Public API:

```java
Tensor sum(int dimension, Tensor mask)
Tensor mean(int dimension, Tensor mask)
```

Example:

```java
Tensor values = new Tensor(new double[]{
        1, 2,
        3, 4,
        5, 6,

        7, 8,
        9, 10,
        11, 12
}, new int[]{2, 3, 2}, null, "values", DataType.FLOAT64);

Tensor mask = new Tensor(new byte[]{
        1, 1, 0,
        1, 0, 0
}, new int[]{2, 3}, null, "mask", DataType.BOOL);

Tensor sum = values.sum(1, mask).compute();
Tensor mean = values.mean(1, mask).compute();
```

Shape interpretation:

```text
values shape = [batch, time, features] = [2, 3, 2]
mask shape   = [batch, time] = [2, 3]
mask expands to [2, 3, 1]
then broadcasts to [2, 3, 2]
```

Logical input:

```text
batch 0:
  time 0: [1, 2]   valid
  time 1: [3, 4]   valid
  time 2: [5, 6]   ignored

batch 1:
  time 0: [7, 8]   valid
  time 1: [9, 10]  ignored
  time 2: [11, 12] ignored
```

Results:

```text
sum  = [[4, 6], [7, 8]]
mean = [[2, 3], [7, 8]]
```

The mean denominator is the number of valid timesteps, not the padded length. If all positions for an output element are masked out, the denominator is clamped to `1`, so the result is `0` rather than NaN or infinity.

## Masked Cross Entropy

Masked cross entropy is useful for padded per-position classification.

Dense target API:

```java
Tensor crossEntropyLoss(Tensor targets, int classDimension, Tensor mask)
```

Index target API:

```java
Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, Tensor mask)
```

The mask applies after the class axis is reduced.

For logits shaped `[batch, time, classes]`:

```text
logits shape = [2, 3, 10]
class axis   = 2
sample shape = [2, 3]
mask shape   = [2, 3]
```

Dense example:

```java
Tensor logits = new Tensor(new double[]{
        2, 0, 0,
        0, 2, 0,
        0, 0, 2,

        2, 0, 0,
        0, 2, 0,
        0, 0, 2
}, new int[]{2, 3, 3}, null, "logits", DataType.FLOAT64);

Tensor targets = new Tensor(new double[]{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,

        1, 0, 0,
        0, 1, 0,
        0, 0, 1
}, new int[]{2, 3, 3}, null, "targets", DataType.FLOAT64);

Tensor mask = new Tensor(new byte[]{
        1, 1, 0,
        1, 0, 0
}, new int[]{2, 3}, null, "mask", DataType.BOOL);

Tensor loss = logits.crossEntropyLoss(targets, 2, mask).compute();
```

Only three positions contribute:

```text
batch 0, time 0
batch 0, time 1
batch 1, time 0
```

The mean denominator is `3`, not `6`.

Index example:

```java
Tensor targetIds = new Tensor(new int[]{
        0, 1, 2,
        0, 1, 2
}, new int[]{2, 3}, null, "targetIds", DataType.INT32);

Tensor loss = logits.crossEntropyLossFromIndices(targetIds, 2, mask).compute();
```

This is equivalent to the dense example when `targets` is one-hot encoded.

## Autograd Contract

The sequence-friendly API stays inside Synaptik's ordinary reverse-mode autodiff model.

Gradient behavior:

- `linear` computes gradients for input, weight, and bias.
- `stack` routes gradient slices back to each input tensor through `concat` backward.
- `unstack` routes gradients through `select` backward.
- `sliceAxis` scatters gradients back into sliced positions.
- `take` / `gatherAxis` scatter gradients back into gathered source positions.
- masked reductions route gradients only through valid mask positions.
- masked cross entropy gives zero gradient for masked-out sample positions.

Masks and index tensors are not differentiable. They control data movement or selection; they do not receive gradients.

Example:

```java
Tensor x = Tensor.randn(new int[]{2, 3, 4}, 0.0, 1.0, DataType.FLOAT64, "x");
Tensor w = Tensor.randn(new int[]{4, 5}, 0.0, 0.02, DataType.FLOAT64, "w");
Tensor b = Tensor.zeros(new int[]{5}, DataType.FLOAT64, "b");
x.setRequiresGrad(true);
w.setRequiresGrad(true);
b.setRequiresGrad(true);

Tensor mask = new Tensor(new byte[]{
        1, 1, 0,
        1, 0, 0
}, new int[]{2, 3}, null, "mask", DataType.BOOL);

Tensor loss = x.linear(w, b).mean(1, mask).sum();
loss.compute(CompileMode.TRAINING);

Tensor dx = x.getGradient();
Tensor dw = w.getGradient();
Tensor db = b.getGradient();
```

Expected gradient publication:

```text
dx shape = [2, 3, 4]
dw shape = [4, 5]
db shape = [5]
```

The invalid timesteps in `mask` contribute no data gradient to `x`.

## Implementation Map

| Public API | Implementation | Primitive strategy |
|---|---|---|
| `Tensor.zeros`, `Tensor.ones`, `Tensor.randn`, `Tensor.arange` | `TensorDataFactory` | Leaf tensor construction |
| `rank`, `size`, `lastDim`, `shapeEquals`, `shapeCopy` | `Tensor` metadata helpers | No graph node |
| `linear` | `tensor.ops.linalg.LinearOp`, `LinearSpec`, CPU `LinearExecutor` | Primitive `LINEAR`; N-D over leading axes |
| `stack` | `tensor.ops.layout.StackOp` | Composes `expandDims` plus `concat` |
| `unstack` | `tensor.ops.layout.UnstackOp` | Composes `select` |
| `sliceAxis` | `TensorOps.sliceAxis` -> `SliceOp` | Static positive-step slice |
| `take` | `GatherOp.take` -> `gatherAxis` | Primitive `GATHER_AXIS` |
| masked `sum` / `mean` | `SumOp` / `MeanOp` | Composes `where`, `sum`, `div`, `clampMin` |
| masked cross entropy | `DenseCrossEntropyLossOp` / `CrossEntropyLossFromIndicesOp` | Composes `logSoftmax`, class reduction, `where`, reduction |

The composition-first rows are intentional. They avoid adding sequence-specific primitive kernels when existing tensor primitives already express the semantics.

## Common Mistakes

### Treating Synaptik As The Sequence Framework

Do not add `RnnLayer`, `LstmCell`, or `SequenceBatch` to Synaptik core just because sequence workloads need these primitives.

Preferred split:

```text
Synaptik: Tensor, autograd, graph compile/execute, primitive operations
Consumer framework: Layer, Model, RNN/LSTM/GRU, dataset conventions
```

### Using `Tensor[]` As The Long-Term Sequence Contract

`Tensor[]` is fine as an import or transitional form, but it loses shape information. Prefer:

```text
[batch, time, features]
```

or, when the consumer framework chooses time-major layout:

```text
[time, batch, features]
```

`linear` works for both because it only cares about the last dimension.

### Confusing `take` With `takeAlongAxis`

Use `take` when the index shape should be inserted at the selected axis:

```java
x.take(1, new int[]{0, 2});
```

Use `takeAlongAxis` when the index tensor already has the full output rank:

```java
x.takeAlongAxis(indicesWithOutputShape, 1);
```

### Normalizing Masked Loss By Padded Length

For padded sequences, the correct denominator is usually the valid count:

```text
loss = sum(valid per-position losses) / count(valid positions)
```

Synaptik's masked cross entropy follows that rule. It does not divide by the full padded `[batch * time]` size.

### Mutating Shape Arrays From Unsafe Accessors

Use:

```java
int[] safe = tensor.shapeCopy();
```

Avoid application-level mutation through:

```java
int[] unsafe = tensor.getShapeUnsafe();
```

`getShapeUnsafe()` is for low-level infrastructure that intentionally works with internal metadata.
