# Understand the current autograd boundary

## Outcome

This guide explains what Synaptik's current internal automatic differentiation (autograd) can
construct and what a user still cannot invoke. Autograd derives gradient expressions from a
forward Tensor expression.

Compiler tasks 0004 and 0004A implement a bounded package-private first-order graph stage and its
first policy-free exact-composition rule extension. There is no public compile request for an
objective, targets, or seed; no gradient publication; and no prepared or executable training
workflow. The current `CompileMode` enum is standalone declarative configuration, not a public
compiler entry point:

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
identity-unique Tensors in that objective's ancestry. Each target must have the objective's exact
floating type, request gradients, and lie on a selected differentiable route. A target may be a
leaf, an intermediate, or the objective itself.

The only current seed is an implicit rank-zero positive one with the objective's exact type.
Generated BFLOAT16, FLOAT32, and FLOAT64 zero/one bases are storage-free leaves registered
explicitly as logical splats: one scalar value repeated at every logical coordinate. BFLOAT16
uses exact zero/one bits `0x0000` and `0x3F80`. Storage, labels, layout, Shape, factory history,
and missing provenance never imply a constant.

## Supported formulas

The closed current matrix contains only:

| Family | Supported variants |
|---|---|
| Elementwise | Same-floating-type binary and exact-scalar `ADD`/`SUB`/`MUL`; same-type branch-only `WHERE`; same-type floating `CAST`; `NEG`/`EXP`/`EXPM1`/`SIGMOID`/`TANH`/`ERF` |
| Reduction and scan | Floating ordinary full, single-axis, and multi-axis `SUM`; masked floating `SUM`; locally invertible floating `SUM_TO_SHAPE`; floating `CUM_SUM` |
| Linear algebra | Every floating `MATMUL` vector/matrix rank pairing, with role-aware selected-operand type checks |
| Logical layout and selection | Floating `CONTIGUOUS`, `RESHAPE`, `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, `PERMUTE`, normalized `SLICE`, both normalized `SLICE_UPDATE` data roles, `SELECT`, `PAD`, `TILE`, `CONCAT`, and `STACK` |

Broadcasted contributions use `sumToShape`. Ordinary SUM restores removed axes before expansion;
masked SUM additionally routes the expanded cotangent through the original mask. CUM_SUM retains
exclusivity while reversing scan direction, and PERMUTE applies the inverse permutation. WHERE
routes no cotangent through its BOOL condition. A `SUM_TO_SHAPE` route is accepted only when each
aligned input/target Dimension is exactly equal or the target extent is statically one.

MATMUL supports all four vector/matrix rank pairings. A selected operand must have the output
floating type; an unselected narrower operand may differ. SLICE and SELECT scatter into typed
zeros; SLICE_UPDATE routes separately to its base and update roles; PAD crops; TILE sums
interleaved repeat axes; CONCAT crops by ordered input prefixes; and STACK selects its inserted
axis. The SLICE_UPDATE update-role rule needs static selected base extents. Repeated operand
positions remain repeated contributions.

An unlisted operation on a selected route fails before the compiler creates the seed or any
formula Tensor. Later work retains mixed-floating cotangent conversion and rules needing tie,
endpoint, discontinuity, singularity, empty-domain, NaN/infinity, or other exceptional-value
policies. Current exclusions therefore include division/power, extrema/clamp and other nonsmooth
operations, reciprocal/log/root families, product/mean/statistical reductions, `CUM_PROD`,
softmax and normalization, indexing outside the listed layout rules, random/dropout, losses, and
other unlisted families.

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
