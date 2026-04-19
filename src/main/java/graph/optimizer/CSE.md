# CSE Stage

`CSE` is the structural common subexpression elimination stage.

Its job is simple:

- detect tensors that compute the same thing
- keep one representative
- rewrite the rest to point at the existing tensor

This is purely a graph-structure pass. It does not evaluate tensors, and it does not use runtime profiling.

## Entry Point

- implementation: [rules/CommonSubexpressionEliminationRule.java](./rules/CommonSubexpressionEliminationRule.java)
- config: [../../config/optimizer/CseConfig.java](../../config/optimizer/CseConfig.java)

Default presets:

- training defaults use `strictSafety = true`
- inference defaults use `strictSafety = false`

## Execution Model

The pass walks the sorted graph once from left to right.

For each tensor:

1. rewrite already replaced inputs through `OptimizerGraphSupport.rewriteInputs(...)`
2. build a structural signature for the current tensor
3. if an identical signature has already been seen, replace the tensor with the existing one
4. otherwise remember the tensor as the canonical representative of that signature

At the end, it rebuilds the topological closure so the final graph contains only reachable tensors.

## What Counts As "The Same"

The core data structure is a structural signature with these fields:

- `opType`
- `backward` flag
- `requiresGrad` when `strictSafety = true`
- resolved backend when `strictSafety = true`
- output shape when `strictSafety = true`
- op-specific parameter signature
- recursively computed input signatures

That means the pass is not "value based". It is graph-structure based.

Two tensors are considered equal only if their operation, parameters, and recursively their inputs all match under the current safety mode.

## Leaf Tensor Semantics

Leafs are treated carefully because that is where accidental over-merging can break autograd semantics.

### Trainable leaf tensors

If a tensor:

- has no operation, and
- `requiresGrad == true`

then it is signatured by object identity, not by value.

So two different trainable parameters with the same shape and same values are not merged.

### Scalar constant leaf tensors

If a tensor:

- has no operation
- `requiresGrad == false`
- has exactly one scalar value

then it is signatured structurally by:

- scalar bit pattern
- shape

So repeated scalar constants may be merged.

### Other leaf tensors

All other leafs are identity-based.

This includes:

- non-scalar constants
- leaf inputs
- views that are already leaves in the optimizer graph

## Commutativity Handling

Only two ops are currently treated as commutative:

- `ADD`
- `MUL`

For those ops, input signature order is sorted before the final signature is built.

That means:

```text
add(a, b)
```

and

```text
add(b, a)
```

are considered the same subexpression.

The same is true for `mul`.

No other operation gets this treatment today.

## Which Nodes Are Skipped Entirely

The pass does not attempt to merge the following:

- `noop`
- existing fused operations
- anything whose `opType()` is already `FUSED`
- operations whose class name contains `"random"`
- operations whose class name contains `"dropout"`
- nodes with no inputs

The reason is straightforward:

- `noop` and fused nodes are structural/control artifacts
- random and dropout-like ops are intentionally effectful or stochastic
- nodes without inputs are covered by leaf handling

## Parameter-Aware Signatures

The pass does not just compare op types. Many operations carry parameters that materially change semantics, so `CSE` builds dedicated parameter keys.

### Scalar parameter ops

- `POW` -> exponent
- `MUL_SCALAR` -> scalar
- `CLAMP_MIN` -> min value
- `CLAMP_MAX` -> max value

### Axis and reduction ops

- `SUM` -> `(dimension, keepDims)`
- `MEAN` -> `(dimension, keepDims)`
- `SOFTMAX` -> axis
- `SOFTMAX_GRAD` -> axis
- `LOG_SOFTMAX` -> axis
- `LOG_SOFTMAX_GRAD` -> axis
- `REDUCE_MIN` -> `(dimension, keepDims)`
- `REDUCE_MAX` -> `(dimension, keepDims)`
- `REDUCE_MIN_GRAD` -> axis
- `REDUCE_MAX_GRAD` -> axis
- `MIN_GRAD` -> whether this is the first-input gradient branch
- `MAX_GRAD` -> whether this is the first-input gradient branch

### Normalization ops

- `LAYER_NORM` -> `(normalizedRank, epsilonBits)`
- `RMS_NORM` -> `(normalizedRank, epsilonBits)`

### Loss ops

- `NLL_LOSS` -> class dimension
- `CROSS_ENTROPY_LOSS` -> class dimension
- `CROSS_ENTROPY_LOSS_INDICES` -> class dimension, reduction kind, ignore-index settings
- `CROSS_ENTROPY_LOSS_INDICES_GRAD` -> class dimension

### Layout and indexing ops

- `RESHAPE` -> target shape
- `PERMUTE` -> axes
- `EXPAND` -> target shape
- `SELECT` -> `(dimension, index)`
- `EXPAND_DIMS` -> axis
- `SQUEEZE` -> axis
- `GATHER` -> dimension
- `GATHER_GRAD` -> dimension
- `TAKE_ALONG_AXIS` -> dimension
- `TAKE_ALONG_AXIS_GRAD` -> dimension
- `SCATTER_ADD` -> dimension

### Higher-level structured ops

- `SCALED_DOT_PRODUCT_ATTENTION` -> scale bits and `hasMask`
- `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD` -> output kind
- `LINEAR` -> `hasBias`

### Conv and pooling ops

`CSE` also includes specialized signature packing for:

- `CONV2D`
- `CONV2D_GEMM`
- `CONV2D_BACKWARD_INPUT`
- `CONV2D_BACKWARD_WEIGHT`
- `MAX_POOL2D`
- `MAX_POOL2D_BACKWARD_INPUT`
- `AVG_POOL2D`
- `AVG_POOL2D_BACKWARD_INPUT`

Those signatures include the relevant stride, padding, dilation, group, kernel, and shape parameters so only semantically identical kernels are merged.

## Strict Safety Mode

When `strictSafety = true`, the structural signature also includes:

- `requiresGrad`
- resolved backend
- output shape

This is the safer mode and is the training default.

When `strictSafety = false`, those fields are omitted. That allows more aggressive merging in inference-style graphs where autograd identity concerns matter less.

## Examples

### Example 1: repeated affine-free subexpression

```text
t1 = add(x, y)
t2 = add(x, y)
```

Result:

- `t2` is replaced by `t1`

### Example 2: commutative merge

```text
t1 = mul(x, y)
t2 = mul(y, x)
```

Result:

- `t2` is replaced by `t1`

### Example 3: trainable parameters are not merged

```text
w1 = leaf(requiresGrad = true)
w2 = leaf(requiresGrad = true)
t1 = add(x, w1)
t2 = add(x, w2)
```

Even if `w1` and `w2` happen to have the same values, they are different leaves by identity, so `t1` and `t2` do not merge.

### Example 4: scalar constants may merge

```text
c1 = scalar(2.0)
c2 = scalar(2.0)
t1 = mulScalar(x, c1)
t2 = mulScalar(x, c2)
```

The scalar leaves may share the same scalar-leaf signature.

## Important Limits

- `CSE` is not symbolic algebra. It will not prove semantic equivalence across different formulas.
- It does not reorder arbitrary operands except for `ADD` and `MUL`.
- It does not attempt to merge stochastic ops.
- It is intentionally conservative around trainable leaves.

## Practical Meaning

`CSE` is the stage that turns duplicated graph construction into shared computation. It is especially useful when:

- the same backward helper expression is built multiple times
- imported graphs contain repeated subgraphs
- earlier rewrites produced repeated structured subexpressions

If two nodes look visually identical in the graph and you expect them to share storage and execution, `CSE` is the first place to inspect.

## Realistic Before/After Example

A common backward-style pattern is repeated helper work:

```text
t1 = sub(outGrad, mean(outGrad, -1, keepDims = true))
t2 = mul(t1, inv(std))
t3 = sub(outGrad, mean(outGrad, -1, keepDims = true))
t4 = mul(t3, inv(std))
```

After `CSE`, this can become conceptually:

```text
t1 = sub(outGrad, mean(outGrad, -1, keepDims = true))
t2 = inv(std)
t3 = mul(t1, t2)
t4 = mul(t1, t2)
```

That matters because fewer repeated graph nodes means:

- fewer executions
- fewer temporaries
- more opportunity for later fusion and memory reuse

## Why Structural Equality Stays Strict

These two merge:

```text
add(x, y)
add(y, x)
```

because `ADD` is explicitly handled as commutative.

These do not merge until earlier canonicalization rewrites them:

```text
sub(x, neg(y))
add(x, y)
```

That division of labor is intentional:

- `AR` canonicalizes
- `CSE` deduplicates

## Debug Checklist

If you expected a merge and it did not happen, inspect:

1. whether `AR` ran before `CSE`
2. whether the two subgraphs are structurally identical after canonicalization
3. whether `strictSafety` is enabled
4. whether one of the differing inputs is a distinct leaf by identity
5. whether an op parameter differs subtly but materially

Typical subtle blockers:

- different reduction axes
- different `keepDims`
- `linear(..., hasBias=true)` vs no bias
- same values but different trainable leaves
