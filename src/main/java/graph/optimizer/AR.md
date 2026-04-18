# AR Stage

`AR` is the composite rewrite stage. Despite the short name, it is not just a small algebraic simplifier. It is the place where the optimizer:

- canonicalizes imported or manually decomposed graphs
- performs local algebraic cleanup
- lowers decomposed patterns into structured primitives

In other words, `AR` is the "graph semantics cleanup and lowering" family.

## Entry Points

- stage wrapper: [rewrite/RewriteRule.java](./rewrite/RewriteRule.java)
- shared base class for most subpasses: [rewrite/AbstractRewriteRule.java](./rewrite/AbstractRewriteRule.java)
- config root: [../../config/optimizer/RewriteConfig.java](../../config/optimizer/RewriteConfig.java)

## Execution Order

The current delegate order is:

1. optional `PiecewiseLoweringRewrite`
2. `AlgebraicRewrite`
3. `LinearLoweringRewrite`
4. `LossLoweringRewrite`
5. `ReductionLoweringRewrite`
6. `AttentionLoweringRewrite`
7. `AttentionBackwardLoweringRewrite`
8. optional `Conv2dLoweringRewrite`

That order is intentional.

- Piecewise lowering runs first because it canonicalizes decomposed expressions into known unary primitives.
- Algebraic cleanup runs before larger pattern lowerings so trivial noise is removed early.
- Structured lowerings such as `linear`, loss, reduction, and attention run after the graph has already been locally simplified.
- `conv2d` lowering is deliberately policy-controlled and sits at the end of the family.

## Generic Rewrite Mechanics

Every `AbstractRewriteRule` subpass follows the same skeleton:

1. remember the original sink tensors
2. walk the graph in topological order
3. rewrite already-replaced inputs before visiting the current tensor
4. decide whether the current tensor should be replaced
5. preserve backward flags
6. repair gradient references through the replacement map
7. rebuild the topological closure from the resolved sinks

This is why the subpasses can be simple peephole matchers while still keeping the final graph valid.

## PiecewiseLoweringRewrite

File: [rewrite/PiecewiseLoweringRewrite.java](./rewrite/PiecewiseLoweringRewrite.java)

Purpose:

- repair imported graphs
- canonicalize manually decomposed expressions
- reduce decomposed piecewise logic back to semantic primitives

This pass is fully optional. The config lives in [../../config/optimizer/PiecewiseLoweringConfig.java](../../config/optimizer/PiecewiseLoweringConfig.java).

Defaults:

- `canonicalSigmoid = false`
- `reluLikeWhere = false`
- `clampLikeWhere = false`

So by default this pass is effectively off unless the config explicitly enables at least one branch.

### Implemented Piecewise Patterns

1. Canonical sigmoid

Pattern:

```text
inv(add(1, exp(neg(x))))
```

Equivalent source forms also allow the negation to appear as `mulScalar(x, -1)`.

Rewrite:

```text
sigmoid(x)
```

2. ReLU-like `where`

Pattern:

```text
where(gt(x, 0), x, zeros_like(x))
```

Requirements:

- condition must be `x > 0`
- `ifTrue` must be the same tensor `x`
- `ifFalse` must be a leaf zero tensor with the same dtype and shape as `x`

Rewrite:

```text
relu(x)
```

3. Clamp-min like `where`

Pattern:

```text
where(lt(x, t), t, x)
```

Requirements:

- `t` must be a scalar leaf constant
- both constants must be dtype-compatible

Rewrite:

```text
clampMin(x, t)
```

4. Clamp-max like `where`

Pattern:

```text
where(gt(x, t), t, x)
```

Rewrite:

```text
clampMax(x, t)
```

### What This Pass Does Not Do

- It does not invent new semantics.
- It does not repair arbitrary `where` expressions.
- It does not replace normal forward `relu()` calls, because those should already have been created as `relu` primitives by the `Tensor` API.

## AlgebraicRewrite

File: [rewrite/AlgebraicRewrite.java](./rewrite/AlgebraicRewrite.java)

Purpose:

- local peephole simplification
- scalar canonicalization
- removal of obvious algebraic noise

It only rewrites a narrow set of op families and explicitly skips:

- leaf tensors
- already fused ops

The handled op types are:

- `ADD`
- `SUB`
- `MUL`
- `MUL_SCALAR`
- `DIV`
- `POW`
- `NEG`
- `LOG`
- `EXP`
- `INV`
- `SQRT`
- `WHERE`
- `CLAMP_MIN`
- `CLAMP_MAX`

Important current detail:

- `WHERE`, `CLAMP_MIN`, and `CLAMP_MAX` currently have placeholder methods only and perform no rewrite.

### Implemented Algebraic Rewrites

#### `ADD`

Implemented rules:

- `a + 0 -> a`
- `0 + a -> a`
- `a + a -> a * 2`
- `a + (-a) -> zerosLike(a)`
- `(-a) + a -> zerosLike(a)`
- `(c * a) + a -> (1 + c) * a`
- `a + (c * a) -> (1 + c) * a`
- `(-a) + (-b) -> -(a + b)`
- `log(a) + log(b) -> log(a * b)`

Example:

```text
add(mulScalar(x, 3.0), x)
=> mulScalar(x, 4.0)
```

#### `SUB`

Implemented rules:

- `a - 0 -> a`
- `0 - a -> -a`
- `a - a -> zerosLike(a)`
- `a - (-b) -> a + b`
- `a - (c * a) -> (1 - c) * a`
- `(c * a) - a -> (c - 1) * a`

Example:

```text
sub(x, neg(y))
=> add(x, y)
```

#### `MUL`

Implemented rules:

- `a * 0 -> zerosLike(result)`
- `0 * a -> zerosLike(result)`
- `a * 1 -> a`
- `1 * a -> a`
- `a * -1 -> -a`
- `-1 * a -> -a`
- `inv(a) * a -> onesLike(a)`
- `a * inv(a) -> onesLike(a)`
- `(-a) * (-b) -> a * b`
- `exp(a) * exp(b) -> exp(a + b)`

#### `MUL_SCALAR`

Implemented rules:

- `mulScalar(x, 0) -> zerosLike(x)`
- `mulScalar(x, 1) -> x`
- `mulScalar(x, -1) -> neg(x)`
- `mulScalar(mulScalar(x, c1), c2) -> mulScalar(x, c1 * c2)`
- `mulScalar(neg(x), c) -> mulScalar(x, -c)`
- scalar leaf constant folding

Example:

```text
mulScalar(mulScalar(x, 0.5), 4.0)
=> mulScalar(x, 2.0)
```

#### `DIV`

Implemented rules:

- `0 / a -> zerosLike(result)`
- `a / 1 -> a`
- `a / -1 -> -a`
- `a / inv(b) -> a * b`
- `mulScalar(a, c) / c -> a`
- `mulScalar(a, c) / k -> mulScalar(a, c / k)` for scalar constant `k`
- `a / k -> mulScalar(a, 1 / k)` for scalar constant `k`

#### `POW`

Implemented rules:

- `pow(a, 0) -> onesLike(a)`
- `pow(a, 1) -> a`
- `pow(a, 2) -> a * a`
- `pow(inv(a), e) -> pow(a, -e)`
- `pow(pow(a, e1), e2) -> pow(a, e1 * e2)`

#### `NEG`

Implemented rules:

- `neg(0) -> 0`
- `neg(neg(a)) -> a`
- `neg(a - b) -> b - a`
- `neg(mulScalar(a, c)) -> mulScalar(a, -c)`

#### `LOG`

Implemented rules:

- `log(pow(a, e)) -> log(a) * e`
- `log(inv(a)) -> -log(a)`
- `log(sqrt(a)) -> log(a) * 0.5`

#### `EXP`

Implemented rules:

- `exp(log(a)) -> a`

#### `INV`

Implemented rules:

- `inv(inv(a)) -> a`
- `inv(pow(a, e)) -> pow(a, -e)`
- `inv(exp(a)) -> exp(-a)`
- `inv(neg(a)) -> -inv(a)`
- `inv(sigmoid(x)) -> 1 + exp(-x)`

That last rewrite is intentionally explicit in the current code:

```text
inv(sigmoid(x))
=> onesLike(x) + exp(-x)
```

#### `SQRT`

Implemented rules:

- `sqrt(0) -> 0`
- `sqrt(1) -> 1`

### System Properties For AlgebraicRewrite

The pass has a global kill switch:

- `cg.optimizer.ar.disableAllTransforms`

It also has a separate toggle for topological-closure rebuilding:

- `cg.optimizer.ar.disableRebuildTopologicalClosure`

And it has individual switches for specific transforms:

- `cg.optimizer.ar.disablePow2ToMul`
- `cg.optimizer.ar.disableAddSelfToMul2`
- `cg.optimizer.ar.disableAddNegToZero`
- `cg.optimizer.ar.disableAddNegNegToNegAdd`
- `cg.optimizer.ar.disableAddLogLogToLogMul`
- `cg.optimizer.ar.disableSubNegToAdd`
- `cg.optimizer.ar.disableDivConstToMulRecip`
- `cg.optimizer.ar.disableDivMulScalarByConst`
- `cg.optimizer.ar.disableDivInvToMul`
- `cg.optimizer.ar.disableMulScalarAssoc`
- `cg.optimizer.ar.disableMulScalarNegPush`
- `cg.optimizer.ar.disableMulScalarConstFold`
- `cg.optimizer.ar.disableAddSubFactorize`
- `cg.optimizer.ar.disableMulInvToOne`
- `cg.optimizer.ar.disableMulNegNegToMul`
- `cg.optimizer.ar.disableMulExpExpToExpAdd`
- `cg.optimizer.ar.disableNegSubSwap`
- `cg.optimizer.ar.disableNegMulScalarPush`
- `cg.optimizer.ar.disablePowPowFlatten`
- `cg.optimizer.ar.disablePowInvToNegExp`
- `cg.optimizer.ar.disableLogPowToMulLog`
- `cg.optimizer.ar.disableLogInvToNegLog`
- `cg.optimizer.ar.disableLogSqrtToHalfLog`
- `cg.optimizer.ar.disableExpLogCancel`
- `cg.optimizer.ar.disableInvSigmoidPattern`
- `cg.optimizer.ar.disableInvPowToNegExp`
- `cg.optimizer.ar.disableInvExpToExpNeg`
- `cg.optimizer.ar.disableInvNegPush`

These flags are useful when benchmarking the impact of a single rewrite family or when investigating a suspicious regression.

## LinearLoweringRewrite

File: [rewrite/LinearLoweringRewrite.java](./rewrite/LinearLoweringRewrite.java)

Purpose:

- replace decomposed affine layers with the explicit `linear` primitive

Recognized pattern:

```text
add(matmul(input, weight), bias)
```

The matcher tries both operand orders, so the following also works:

```text
add(bias, matmul(input, weight))
```

Requirements:

- first candidate must be `matmul`
- second candidate must be a 1D numeric bias tensor
- `input.shape[-1] == weight.shape[0]`
- `bias.shape[0] == weight.shape[1]`
- output rank must match input rank
- output batch prefix must match input batch prefix

Rewrite:

```text
linear(input, weight, bias)
```

This pass is config-gated:

- file: [../../config/optimizer/LinearLoweringConfig.java](../../config/optimizer/LinearLoweringConfig.java)
- default: enabled

## LossLoweringRewrite

File: [rewrite/LossLoweringRewrite.java](./rewrite/LossLoweringRewrite.java)

Purpose:

- recognize decomposed loss graphs
- lower them to loss-specific primitives the backend can target directly

### Forward Cross-Entropy From Indices

Recognized per-sample form:

```text
neg(gather(logSoftmax(logits), targetIndices))
```

The gather axis must match the `logSoftmax` class dimension.

Supported reduced wrappers:

- `sum(..., dimension = -1)` -> reduction `SUM`
- `mean(..., dimension = -1)` -> reduction `MEAN`

Rewrites:

- no wrapper -> `crossEntropyLossIndices(..., reduction = NONE)`
- `sum(-1)` wrapper -> `crossEntropyLossIndices(..., reduction = SUM)`
- `mean(-1)` wrapper -> `crossEntropyLossIndices(..., reduction = MEAN)`

Example:

```text
mean(
  neg(
    gather(
      logSoftmax(logits, classDim),
      targetIndices,
      classDim
    )
  ),
  -1
)
=> crossEntropyLossIndices(logits, targetIndices, classDim, MEAN)
```

### Backward Cross-Entropy From Indices

This subpass only acts on backward tensors.

Recognized shape:

```text
sub(
  mul(softmax(logits), sampleScaleBroadcast),
  scatterAdd(zeros_like(logits), targetIndices, sampleScale)
)
```

The matcher accepts either operand ordering in the `SUB`.

Important current requirements:

- `softmax` class dimension must match `scatterAdd` dimension
- the `sampleScale` broadcast must be represented as:
  - `expandDims(sampleScale, classDimension)`, or
  - `expand(expandDims(sampleScale, classDimension), targetShape)`
- the scatter base must be a leaf tensor labeled exactly `"zeros_like"`
- the zero base and logits must have the same shape and dtype
- both branches must use the exact same `sampleScale` tensor object

Rewrite:

```text
crossEntropyLossIndicesGrad(logits, targetIndices, sampleScale)
```

## ReductionLoweringRewrite

File: [rewrite/ReductionLoweringRewrite.java](./rewrite/ReductionLoweringRewrite.java)

Purpose:

- replace decomposed backward formulas of reduction-like operations with dedicated gradient primitives

### Softmax Backward

Recognized pattern:

```text
mul(
  softmaxOut,
  sub(outGrad, sum(mul(outGrad, softmaxOut), dim, keepDims = true))
)
```

The pass checks:

- one input is a `softmax` output
- the other input is a `SUB`
- the inner `sum` uses the same dimension as the `softmax`
- `keepDims` must be `true`
- `softmaxOut` and `outGrad` must have the same shape and dtype

Rewrite:

```text
softmaxGrad(softmaxOut, outGrad, dim)
```

### LogSoftmax Backward

Recognized pattern:

```text
sub(
  outGrad,
  mul(
    exp(logSoftmaxOut),
    sum(outGrad, dim, keepDims = true)
  )
)
```

Checks:

- the `exp` input must be a `logSoftmax` output
- the summed branch must sum `outGrad` over the same dimension
- `keepDims` must be `true`
- `logSoftmaxOut` and `outGrad` must have the same shape and dtype

Rewrite:

```text
logSoftmaxGrad(logSoftmaxOut, outGrad, dim)
```

## AttentionLoweringRewrite

File: [rewrite/AttentionLoweringRewrite.java](./rewrite/AttentionLoweringRewrite.java)

Purpose:

- recognize decomposed scaled dot-product attention forward graphs
- replace them with `scaledDotProductAttention`

Recognized forward structure:

```text
matmul(
  softmax(scores),
  value
)
```

where `scores` is one of:

```text
matmul(query, permute(key, ..., last, secondLast))
```

or

```text
mulScalar(
  matmul(query, permute(key, ..., last, secondLast)),
  positiveScale
)
```

or masked:

```text
where(mask, keptScores, fillScalar)
```

where `keptScores` is one of the two forms above.

Requirements:

- the attention weights input must be a `softmax`
- softmax axis must be the last dimension of the weight tensor
- key must be represented as a swap of the last two axes through `permute`
- output shape must match `query.shape[:-1] + [value.shape[-1]]`
- mask dtype must be `BOOL`
- mask fill scalar must match the hardcoded dtype-specific sentinel

Current mask fill sentinel values:

- `FLOAT64` -> `-1.0e30`
- `FLOAT32` -> `-1.0e9`
- `BFLOAT16` -> `-1.0e30`

Rewrite:

```text
scaledDotProductAttention(query, key, value, [mask], scale)
```

## AttentionBackwardLoweringRewrite

File: [rewrite/AttentionBackwardLoweringRewrite.java](./rewrite/AttentionBackwardLoweringRewrite.java)

Purpose:

- recognize decomposed backward graphs around scaled dot-product attention
- replace raw backward branches with `scaledDotProductAttentionBackward`

This pass does not inherit from `AbstractRewriteRule`. It uses a custom full-graph pass because it needs:

- an index of already-lowered forward attention nodes
- a notion of which tensors are reachable from the forward output
- final sink repair after multi-node replacement

### Supported Dtypes

Current lowered backward support is limited to:

- `FLOAT32`
- `FLOAT64`

`BFLOAT16` is not lowered here today.

### Forward Attention Index

Before matching backward nodes, the pass builds an index of existing forward attention primitives:

- key: `(query, key, canonicalMask, scaleBits)`
- value: all matching `scaledDotProductAttention` tensors

Mask canonicalization currently strips chains of `expand(...)` so the same logical mask can still be recognized.

### Why Forward Reachability Is Tracked

The pass collects tensors reachable from the special forward output anchor labeled `Tensor.SYSTEM_FORWARD_OUTPUT_LABEL`.

This is used to avoid accidentally treating the actual forward softmax in the forward graph as a backward-only attention-weights candidate.

### Value Gradient Lowering

Recognized raw pattern:

```text
matmul(permute(attentionWeights), outGrad)
```

where `attentionWeights` can be resolved back to exactly one matching attention forward node.

Rewrite:

```text
scaledDotProductAttentionBackward(attentionOut, outGrad, VALUE)
```

### Query Gradient Lowering

Recognized raw pattern:

```text
matmul(dScores, keyLikeOperand)
```

where:

- `dScores` may be optionally scaled by `mulScalar`
- `dScores` may be optionally masked by `where(mask, softmaxGrad(...), zero)`
- the inner gradient core must be `softmaxGrad(weights, dWeights)`
- `dWeights` must match `matmul(outGrad, permute(value))`
- `keyLikeOperand` must be either the exact attention key tensor or the recognized permuted key form
- the scale and mask must match the original forward attention node

Rewrite:

```text
scaledDotProductAttentionBackward(attentionOut, outGrad, QUERY)
```

### Key Gradient Lowering

Recognized raw pattern:

```text
permute(
  matmul(
    permute(query),
    dScores
  )
)
```

with the same `dScores` matcher as query backward.

Rewrite:

```text
scaledDotProductAttentionBackward(attentionOut, outGrad, KEY)
```

### Important Current Constraints

- if multiple forward attention nodes share the same `(query, key, mask, scale)` signature, the decomposed backward softmax form is not lowered through that path because the forward match would be ambiguous
- masked backward matching requires the masking branch to be `where(mask, kept, zeroTensor)`
- the scale must be positive
- mask fill checking uses the same dtype-specific sentinel values as forward attention lowering

## Conv2dLoweringRewrite

Files:

- [rewrite/Conv2dLoweringRewrite.java](./rewrite/Conv2dLoweringRewrite.java)
- [rewrite/Conv2dLoweringHeuristics.java](./rewrite/Conv2dLoweringHeuristics.java)

Purpose:

- replace generic `conv2d` with `conv2dGemm` when policy allows it

Config:

- [../../config/optimizer/Conv2dLoweringConfig.java](../../config/optimizer/Conv2dLoweringConfig.java)
- modes: `OFF`, `ALWAYS`, `HEURISTIC`

This pass rewrites forward `conv2d` and the dedicated backward convolution primitives when policy allows it.

Rewrite:

```text
conv2d(...) -> conv2dGemm(...)
conv2dBackwardInput(...) -> conv2dBackwardInputGemm(...)
conv2dBackwardWeight(...) -> conv2dBackwardWeightGemm(...)
```

### Heuristic Mode Details

Current heuristic preconditions:

- input, weight, and output must all be rank-4 tensors
- `groups == 1`
- `dilationH == 1`
- `dilationW == 1`

Current pointwise lowering heuristic:

- kernel `1x1`
- stride `1x1`
- padding `0x0`
- `inChannels >= 128`
- `outChannels >= 64`
- `outChannels <= inChannels * 2`
- `batch * outH * outW >= 256`

Current standard `3x3` lowering heuristic:

- kernel `3x3`
- stride `1x1`
- padding `1x1`
- `inChannels >= 64`
- `outChannels >= 64`
- `batch * outH * outW >= 512`

Everything else remains on the regular `conv2d` primitive.

## Summary

`AR` is the semantic cleanup stage. Its current responsibilities are:

- optional canonicalization of decomposed piecewise expressions
- local algebraic simplification
- lowering of `matmul + bias` into `linear`
- lowering of cross-entropy-from-indices forward and backward
- lowering of softmax and log-softmax backward formulas
- lowering of decomposed attention forward and selected backward branches
- optional lowering of `conv2d` into `conv2dGemm`

If a new pattern is semantically recognizable in the graph and should become a stable backend primitive, `AR` is usually the first place to consider.
