# AR Stage

`AR` is the simplification rewrite stage.

Despite the short name, it is not "just algebraic simplification".
It is the place where the compiler:

- canonicalizes decomposed patterns
- removes local algebraic noise
- leaves policy-sensitive lowering to the optional lowering stage

In practice, `AR` is the semantic simplification stage of the compiler. Optional lowering is configured from the same
`RewriteConfig`, but `OptimizerFactory` wires it as a separate stage after simplification.

## Entry Points

- algebraic rule:
  - [rewrite/algebraic/AlgebraicSimplificationRule.java](./rewrite/algebraic/AlgebraicSimplificationRule.java)
- shared rewrite base:
  - [rewrite/LocalTensorRewriteRule.java](./rewrite/LocalTensorRewriteRule.java)
- configuration root:
  - [../../config/optimizer/RewriteConfig.java](../../config/optimizer/RewriteConfig.java)

## Current Rule Order

The current simplification rewrite order is:

1. optional `PiecewiseCanonicalizationRule`
2. `AlgebraicSimplificationRule`

The optional lowering order is:

1. `LinearLoweringRule`
2. `LossForwardLoweringRule`
3. optional `Conv2dDagLoweringRule`

That order is intentional:

- piecewise repair first
- local algebraic simplification second
- structural lowerings in the separate lowering stage after simplification
- conv2d lowering last because it is more policy-sensitive

## Generic Rewrite Mechanics

Most subpasses inherit the same shape from `LocalTensorRewriteRule`:

1. remember observable roots
2. walk the graph in topological order
3. rewrite already replaced inputs
4. optionally replace the current tensor
5. preserve important flags and gradient references
6. rebuild a clean topological closure

This lets each subpass stay a local matcher instead of re-implementing whole-graph repair every time.

## Piecewise Lowering

File:

- [rewrite/canonical/PiecewiseCanonicalizationRule.java](./rewrite/canonical/PiecewiseCanonicalizationRule.java)

This pass is optional and disabled by default unless its config enables specific branches.

### Implemented patterns today

#### 1. Canonical sigmoid

Pattern:

```text
inv(add(1, exp(neg(x))))
```

Equivalent negation form also accepted:

```text
inv(add(1, exp(mulScalar(x, -1))))
```

Lowering:

```text
sigmoid(x)
```

Worked value example:

- `x = 2.0`
- `neg(x) = -2.0`
- `exp(-2.0) ≈ 0.1353`
- `1 + exp(-2.0) ≈ 1.1353`
- `inv(...) ≈ 0.8808`
- lowered `sigmoid(2.0)` gives the same result

#### 2. ReLU-like `where`

Pattern:

```text
where(gt(x, 0), x, zeros_like(x))
```

Lowering:

```text
relu(x)
```

Worked value example:

- `x = [-2, 3]`
- condition = `[false, true]`
- result = `[0, 3]`
- same as `relu(x)`

#### 3. Clamp-like `where`

Patterns:

```text
where(lt(x, t), t, x)   -> clampMin(x, t)
where(gt(x, t), t, x)   -> clampMax(x, t)
```

Worked value example:

- `x = [1, 5, 9]`, `t = 4`
- `where(lt(x, 4), 4, x) = [4, 5, 9]`
- lowered `clampMin(x, 4) = [4, 5, 9]`

## Algebraic Rewrite

File:

- [rewrite/algebraic/AlgebraicSimplificationRule.java](./rewrite/algebraic/AlgebraicSimplificationRule.java)

This pass contains the highest number of local identities.
It is heavily feature-gated with system properties for targeted experiments, but the defaults enable the normal rule set.

### Add rules

Currently implemented:

- `x + 0 -> x`
- `0 + x -> x`
- `x + x -> x * 2`
- `x + (-x) -> 0`
- `(-a) + (-b) -> -(a + b)`
- `log(a) + log(b) -> log(a * b)`
- factorization of `x + c*x` style forms through `mulScalar`

Worked example:

- `x = 3`
- `x + x = 6`
- rewritten `x * 2 = 6`

### Sub rules

- `x - 0 -> x`
- `0 - x -> neg(x)`
- `x - x -> 0`
- `x - (-y) -> x + y`
- factorization of `x - c*x` patterns

### Mul rules

- `x * 0 -> 0`
- `x * 1 -> x`
- `x * -1 -> neg(x)`
- `x * inv(x) -> 1`
- `(-a) * (-b) -> a * b`
- `exp(a) * exp(b) -> exp(a + b)`

### `mulScalar` rules

- `x * 0 -> 0`
- `x * 1 -> x`
- `x * -1 -> neg(x)`
- nested scalar fold:
  - `(x * 3) * 2 -> x * 6`
- push through negation:
  - `neg(x) * 4 -> x * -4`
- constant fold for scalar leaves

### Div and inv rules

- `0 / x -> 0`
- `1 / x -> inv(x)`
- `x / 1 -> x`
- `x / -1 -> neg(x)`
- `x / inv(y) -> x * y`
- `(x * c) / c -> x`
- `(x * c) / d -> x * (c / d)`
- `x / c -> x * (1/c)` for scalar constant `c`
- `inv(inv(x)) -> x`
- `inv(pow(x, e)) -> pow(x, -e)`
- `inv(exp(x)) -> exp(-x)`
- `inv(neg(x)) -> neg(inv(x))`
- `inv(sigmoid(x)) -> 1 + exp(-x)`

### Pow / log / exp / sqrt / neg rules

- `pow(x, 0) -> 1`
- `pow(x, 1) -> x`
- `pow(x, -1) -> inv(x)`
- `pow(x, 2) -> x * x`
- `pow(inv(x), e) -> pow(x, -e)`
- `pow(pow(x, a), b) -> pow(x, a*b)`
- `log(pow(x, e)) -> log(x) * e`
- `log(inv(x)) -> -log(x)`
- `log(sqrt(x)) -> log(x) * 0.5`
- `exp(log(x)) -> x`
- `neg(neg(x)) -> x`
- `neg(sub(a, b)) -> sub(b, a)`
- `neg(mulScalar(x, c)) -> mulScalar(x, -c)`
- `sqrt(0) -> 0`
- `sqrt(1) -> 1`

### Clamp rules

- `clampMin(x, -inf) -> x`
- `clampMax(x, +inf) -> x`
- `clampMin(clampMin(x, a), b) -> clampMin(x, max(a, b))`
- `clampMax(clampMax(x, a), b) -> clampMax(x, min(a, b))`

## Linear Lowering

File:

- [rewrite/lowering/LinearLoweringRule.java](./rewrite/lowering/LinearLoweringRule.java)

Pattern:

```text
add(matmul(input, weight), bias)
```

Requirements:

- `weight` rank is exactly 2
- `bias` rank is exactly 1
- output shape matches linear semantics

Lowering:

```text
linear(input, weight, bias)
```

Worked shape example:

- `input`: `[4, 16]`
- `weight`: `[16, 32]`
- `bias`: `[32]`
- `matmul(input, weight)` -> `[4, 32]`
- `add(..., bias)` broadcasts `[32]` over batch -> `[4, 32]`
- lowered primitive: `linear(input, weight, bias)`

## Loss Lowering

Files:

- [rewrite/lowering/LossForwardLoweringRule.java](./rewrite/lowering/LossForwardLoweringRule.java)

### Forward pattern

Current forward lowering recognizes the decomposed negative-log-likelihood shape:

```text
neg(gather(logSoftmax(logits), targetIndices))
```

and reduced variants:

```text
sum(neg(gather(logSoftmax(logits), targetIndices)))
mean(neg(gather(logSoftmax(logits), targetIndices)))
```

Lowering target:

```text
crossEntropyLossFromIndices(logits, targetIndices, classDimension, reduction)
```

Backward loss DAGs remain in their canonical primitive form. The default graph
optimizer must not rewrite public Tensor/autograd backward DAGs into legacy
gradient descriptors such as `CROSS_ENTROPY_LOSS_INDICES_GRAD`.

## Attention

Attention is intentionally not lowered by this graph rewrite stage. The public
`scaledDotProductAttention` API builds a primitive DAG from matmul, permute,
stable softmax, optional masking, and value projection. Backend-specific SDPA
recognition belongs in accelerator partition lowering, where a backend may lower
that primitive partition to an SDPA DAG primitive without changing the compiled
semantic graph.
- `QUERY`
- `KEY`

The matcher is dtype-limited to:

- `FLOAT32`
- `FLOAT64`

## Conv2d Lowering

File:

- [rewrite/lowering/Conv2dDagLoweringRule.java](./rewrite/lowering/Conv2dDagLoweringRule.java)

This pass is policy-controlled through `Conv2dLoweringConfig`.
`HEURISTIC` mode reads its thresholds from `Conv2dDagLoweringProfile`; the built-in profile keeps conservative defaults, and future calibration should write those profile fields instead of changing the lowering rule.

Modes:

- `OFF`
- `ALWAYS`
- `HEURISTIC`

Currently it can lower semantic `conv2d` to canonical Tensor primitives:
`UNFOLD2D`, `MATMUL`, `RESHAPE`, and optional bias add.

The lowering itself is semantic.
Runtime selection between Java and BLAS for the lowered GEMM primitives still happens later in backend preparation.

## What AR Does Not Do

`AR` does not:

- decide vector widths
- choose BLAS thresholds
- pick matmul microkernels
- decide thread counts
- run memory reuse

Those are downstream prepare/planning concerns.

`AR` should be read as:

- clean the graph
- recover or introduce better primitives
- leave runtime policy to later layers
