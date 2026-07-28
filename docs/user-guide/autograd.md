# Understand the current autograd boundary

## Outcome

This guide explains what Synaptik's current internal automatic differentiation (autograd) can
construct and what a user still cannot invoke. Autograd derives gradient expressions from a
forward Tensor expression.

Compiler tasks 0004, 0004A, 0004B, and 0005A implement a bounded package-private first-order
graph stage, its exact-composition and shared-algebra extensions, and the exact current 48-kind
elementwise/activation policy.
There is no public compile request for an objective, targets, or seed; no gradient publication;
and no prepared or executable training workflow. The current `CompileMode` enum is standalone
declarative configuration, not a public compiler entry point:

```java
import io.github.pho001.synaptik.config.compile.CompileMode;

CompileMode graphScope = CompileMode.FORWARD_AND_BACKWARD;
```

Constructing `graphScope` does not capture a graph, construct or publish a gradient, prepare a
schedule, select a backend, or execute training.

## Current internal flow

The package-private compiler follows this sequence:

```text
ordered forward Tensor outputs
  -> validate one scalar floating objective and ordered targets
  -> preflight the complete selected objective-to-target slice
  -> construct formulas through ordinary public Tensor operations
  -> combine forward outputs and gradient roots
  -> one phase-aware graph capture
  -> inference, validation, and one-shot exact whole-graph optimization
  -> package-private GraphCompilation
```

`GraphCompiler` is the internal entry owner. It takes direct parameters rather than a public
request aggregate and returns internal immutable `GraphCompilation`. `FORWARD_ONLY` requires no
first-order request, contains no `BACKWARD` nodes, and returns no gradient roles.
`FORWARD_AND_BACKWARD` and the current internal `TRAINING_STEP` path perform the same first-order
combined construction; `TRAINING_STEP` adds no optimizer update.

The objective must be one exact requested forward-output Tensor with scalar Shape, a floating
data type, and gradient eligibility. Targets are a non-empty ordered list of exact-object-
identity-unique Tensors in that objective's ancestry. Each target must have a floating type,
request gradients, and lie on a selected differentiable route. A target may have a different
floating type from the objective and may be a leaf, an intermediate, or the objective itself.

The only current seed is an implicit rank-zero positive one with the objective's exact type.
Generated BFLOAT16, FLOAT32, and FLOAT64 scalar bases are storage-free leaves registered
explicitly as logical splats: one scalar value repeated at every logical coordinate. One
request-local cache keys zero, one, and extrema/clamp bounds by exact data type and represented
bits and preserves deterministic first-use order. BFLOAT16 uses exact zero/one bits `0x0000` and
`0x3F80`. Storage, labels, layout, Shape, factory history, and missing provenance never imply a
constant.

## Supported formulas

The closed current matrix contains only:

| Family | Supported variants |
|---|---|
| Elementwise | All seven promoted floating binary arithmetic kinds; all eight exact-type floating scalar kinds, including first-class `CLAMP`; promoted branch-only `WHERE`; floating-to-floating `CAST`; and all nineteen floating unary kinds |
| Reduction and scan | Floating ordinary full, single-axis, and multi-axis `SUM`/`MEAN`; masked floating `SUM`/`MEAN`; locally invertible floating `SUM_TO_SHAPE`; floating `CUM_SUM` |
| Linear algebra | Every floating `MATMUL` vector/matrix rank pairing, including mixed-floating selected operands |
| Logical layout and selection | Floating `CONTIGUOUS`, `RESHAPE`, `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, `PERMUTE`, normalized `SLICE`, both normalized `SLICE_UPDATE` data roles, `SELECT`, `PAD`, `TILE`, `CONCAT`, and `STACK` |

Forward expressions and generated gradients use the same ordinary Tensor operations, model
numerical semantics, inference, validation, and exact optimization rules. A mixed-floating
contribution first uses `sumToShape` when broadcasting must be reversed and then uses ordinary
`cast` only when its type differs from the selected input. The contribution therefore reaches
accumulation with that input's exact Shape and floating type.

Binary DIV uses `g / right` for its left input and the exact ordered expression
`-(g * left) / (right * right)` for its right input; scalar DIV uses `g / scalar`. These formulas
have no gradient-only rule for singularities, NaNs, infinities, signed zeros, overflow,
underflow, rounding, rewriting, or folding. `FLOOR`, `CEIL`, and `SIGN` instead use an explicit
first-order local convention: a direct exact positive-zero cotangent, without `g * 0` or a
floating comparison.

MIN and MAX split an exact numeric tie equally between Tensor inputs; scalar extrema give the
Tensor receiver one half. CLAMP applies that convention to the ordered composition
`MIN(MAX(input, min), max)`, so a normal endpoint receives one half and a simultaneous two-stage
tie receives one quarter. Opposite signed zeros are ties. Unordered NaN makes the comparisons
false, so extrema, CLAMP, ABS, and RELU return exact positive zero at NaN positions.

POW, reciprocal, logarithm, square-root, and inverse-square-root rules use their direct analytic
Tensor formulas without compiler-inserted domain masks. Scalar POW subtracts one exactly once in
the represented exponent type. ABS returns zero at both signed zeros; RELU returns `g` only above
positive zero. Exact GELU, fixed tanh-approximation GELU, and SiLU use their analytic formula for
finite and NaN inputs, `g` at positive infinity, and exact positive zero at negative infinity.
Their coefficients use fixed BFLOAT16/FLOAT32/FLOAT64 bit patterns rather than host
transcendental calculation.

Ordinary MEAN restores and expands `g`, reduces an input-shaped logical-one expression over the
same axes to obtain a count, expands that count, and divides. This represents static, dynamic,
expression, zero-sized, and empty-axis-list domains without host count calculation. Masked MEAN
counts true positions with ordinary `WHERE` and `SUM`, divides, and applies a final ordinary
`WHERE`; an all-false slice receives zero at every input coordinate. This convention makes no
promise about evaluation of the unselected quotient branch.

Ordinary SUM restores removed axes before expansion; masked SUM additionally routes the expanded
cotangent through the original mask. CUM_SUM retains exclusivity while reversing scan direction,
and PERMUTE applies the inverse permutation. WHERE routes no cotangent through its BOOL
condition, and masked reductions route none through their mask. A `SUM_TO_SHAPE` route is accepted
only when each aligned input/target Dimension is exactly equal or the target extent is statically
one.

MATMUL supports all four vector/matrix rank pairings. Each selected contribution reverses batch
broadcasting and then casts once when its promoted type differs from the operand. SLICE and SELECT
scatter into typed zeros; SLICE_UPDATE routes separately to its base and update roles; PAD crops;
TILE sums interleaved repeat axes; CONCAT crops by ordered input prefixes; and STACK selects its
inserted axis. The SLICE_UPDATE update-role rule needs static selected base extents. Repeated
operand positions remain repeated contributions.

An unlisted operation on a selected route fails before the compiler creates the seed or any
formula Tensor. Product/reduction-extrema/softmax/statistics/normalization, remaining
layout/indexing/stochastic work, attention, convolution, pooling, and losses remain later
Compiler 0005B–0005D families. Binding-dependent `SUM_TO_SHAPE`, target-relative crop, and other
unselected layout cases retain their existing fail-closed guards. Comparisons, BOOL
logic/classification, the `WHERE` condition, scalar attributes and bounds, non-floating casts,
`ALL`, `ANY`, arg-extrema outputs, indices, masks, and graph RNG state are non-differentiable.

## Conceptual example

### Goal and inputs

Consider a floating Tensor `x` with Shape `[2]` and the scalar objective:

```text
y = sum(x * x)
```

The requested target is the exact `x` object. This is a conceptual compiler example because no
public objective/target compile API exists yet.

### Meaningful steps

Preflight first verifies that `y` is a requested scalar forward output and that the selected path
contains only supported `SUM` and `MUL` occurrences. Only then does construction add the implicit
scalar-one seed. Reverse traversal produces two ordered contributions for the repeated `x`
operand:

```text
dy/dx = x + x
```

The compiler captures `y` and the gradient root together. Original producers receive phase
`FORWARD`; generated addition receives phase `BACKWARD`. Graph-local IDs are assigned once.

### Result and interpretation

Internal `GraphCompilation` retains the final forward graph value plus one role from `x`'s
`TensorId` to the final gradient `ValueId`. If several targets resolve to that same gradient
value, every target role remains ordered while the graph-output boundary lists the value only at
its first occurrence. No identity node is manufactured.

This result proves expression construction and graph-stage bookkeeping only. It does not compute
the numerical gradient for an input value, publish data, update a parameter, select a backend,
prepare storage, or execute a graph.

## Common errors

| Symptom | Likely cause | Correction |
|---|---|---|
| A public call site cannot provide objective and targets | The current request and `GraphCompiler` are package-private. | Wait for the planned public compile/publication contract; do not depend on internal compiler types. |
| Preflight rejects an operation that has a public Tensor method | Public expression construction does not imply a selected derivative rule or policy. | Keep the selected route inside the closed matrix or wait for its owning follow-up. |
| Preflight rejects a mixed-floating route | Mixed types are accepted only for the exact binary, `WHERE`, floating `CAST`, and `MATMUL` rows with a complete `sumToShape`-then-`cast` normalization path. | Make every selected role floating and keep the route inside those current promotion and Shape guards. |
| A BOOL condition or comparison is requested as a target | BOOL roles are non-differentiable. | Request a floating target reached through selected differentiable input roles. |
| `TRAINING_STEP` produces no optimizer update | The current internal mode covers only the same combined forward/backward graph stage as `FORWARD_AND_BACKWARD`. | Keep optimizer behavior in the planned training lifecycle. |

## Limitations

The current compiler path has no public objective/target/seed request, non-scalar objective,
caller-supplied seed, disconnected-target zero policy, vector-Jacobian product, higher derivative,
gradient publication, optimizer update, training session, preparation, runtime execution, or
backend-specific behavior. Public Tensors remain expression model state with no gradient field or
`backward()` method.

## Related documentation

- [Compile API status and exact internal matrix](../api/compile-api.md#current-package-private-pre-capture-autograd)
- [Tensor API](../api/tensor-api.md)
- [Autograd strategy note](../design/notes/autograd-strategy.md)
- [Training graph](../architecture/training-graph.md)
- [Training API status](../api/training-api.md)
