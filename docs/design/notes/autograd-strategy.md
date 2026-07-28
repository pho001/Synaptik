# Autograd strategy

> [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) is authoritative; this note explains its
> compiler-owned pre-capture automatic-differentiation design.

## Purpose and status

Automatic differentiation (autograd) derives gradient computations from a forward expression.
The design is accepted and its first bounded package-private implementation is current. Model task
0025 lets every producer return its canonical exact Tensor wrapper for each output position.
[Compiler task 0004](../../planning/modules/compiler/tasks/0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
is Complete with the scalar-objective, implicit-unit-seed first-order graph-stage contract.
[Compiler task 0004A](../../planning/modules/compiler/tasks/0004a-exact-composition-gradient-rule-extensions.md)
is Complete with the first policy-free exact-composition rule extension.
[Compiler task 0004B](../../planning/modules/compiler/tasks/0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md)
is Complete with mixed-floating cotangent normalization, division, direct-zero local conventions,
and ordinary or masked mean.
[Compiler task 0005A](../../planning/modules/compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
is Complete with the exact 48-kind elementwise/activation classification and formulas.
[Compiler task 0005B](../../planning/modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
is Complete with binding-aware expansion plus reduction, scan, softmax, statistics, norm, and
Layer/RMS/batch-normalization rules. Compiler 0005 already provides current immutable compile
artifacts, while public gradient requests, higher derivatives, optimizer updates, preparation,
and execution remain planned.

## Mental model

```text
original forward Tensor expression DAG
  -> fail-closed compiler preflight
  -> reverse traversal using named compiler gradient rules
  -> ordinary Tensor expressions for contributions and accumulation
  -> combined forward + gradient Tensor expression DAG
  -> one phase-aware capture
  -> immutable combined compiler graph
  -> inference, validation, exact combined optimization, final validation
  -> publication and planning
```

Tensor expression objects are the single construction language before capture.
`CompiledGraphModel` is the immutable graph state after capture. The compiler does not convert
captured `ValueId` values back into placeholder Tensors and does not maintain a second algebra.

Compiler task 0004 implements the general package-private entry owner `GraphCompiler` and its
mode-neutral graph-stage result `GraphCompilation`, without adding a request aggregate.
`FORWARD_ONLY` produces no BACKWARD nodes and empty gradient results. Backward-capable modes may
produce the combined forward/backward graph described below. `GraphCompilation` is distinct from
the later `CompileArtifacts` aggregate.

## Current first-order request

One backward-capable internal request contains:

- one exact objective Tensor that is also a requested forward output, has scalar Shape, one
  floating data type, and gradient eligibility;
- one non-empty ordered list of exact-object-identity-unique target Tensors in the objective
  ancestry, each with a floating type, gradient eligibility, and a selected differentiable route;
  a target may have a different floating type from the objective; and
- one implicit rank-zero positive-one seed with the objective's exact type.

A target may be a leaf, an intermediate, or the objective itself. Target order becomes
gradient-result-role order; it does not stop traversal needed by another upstream target. There
is no current caller-supplied seed, non-scalar objective, disconnected-target zero policy,
vector-Jacobian product, public target/publication request, or higher derivative.

## Ownership and reverse accumulation

- `modules/model` owns public Tensor operations, exact producer occurrence identity, canonical
  output wrappers, and immutable indexed provenance. It owns no derivative rule.
- `modules/compiler` owns preflight, output seeds, differentiation targets, rule dispatch,
  deterministic reverse traversal, contribution accumulation, phase-aware capture, validation,
  and combined optimization.
- Planning assigns backend ownership after expansion. Concrete backends prepare assigned regions.
  Runtime executes prepared work and never derives gradients.

Named compiler components such as `ElementwiseGradientRules` call only existing public methods
such as `mul`, `add`, `sumToShape`, and `transpose`. During one compile request, identity-based
maps associate exact Tensor objects with ordered contributions and accumulated gradients. The
compiler processes selected occurrences in reverse producer postorder, ascending selected output
slot for one producer, and ascending input position, then combines contributions with ordinary
left-associated `Tensor.add`. These maps are temporary bookkeeping, not graph IR, public Tensor
state, a tape, or a registry.

## Preflight and construction failures

Before constructing a backward expression, the compiler inventories every backward-reachable
producer occurrence, output role, exact attributes, and required derivative policy. Unsupported
or ambiguous work fails closed. This prevents a known incomplete rule matrix from creating a
partial backward expression.

The closed implemented matrix through Compiler 0005B is:

| Family | Current exact variants |
|---|---|
| Elementwise | All seven promoted floating binary arithmetic kinds; all eight exact-type floating scalar kinds, including first-class `CLAMP`; promoted branch-only `WHERE`; floating-to-floating `CAST`; and all nineteen floating unary kinds |
| Reduction/scan/softmax | Floating ordinary full, single-axis, and ordered multi-axis `SUM`/`MEAN`/`PROD`/`MIN`/`MAX`; masked floating `SUM`/`MEAN`; binding-aware floating `SUM_TO_SHAPE`; floating `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM`, `CUM_SUM`, `CUM_PROD`, `SOFTMAX`, and `LOG_SOFTMAX` |
| Normalization | No-affine and affine floating Layer normalization; no-scale and scaled floating RMS normalization; all five floating batch-inference inputs; and batch-training public output slots zero through two with their exact input-role matrices |
| Linear algebra | Every current floating `MATMUL` vector/matrix rank pairing, with role-aware mixed-floating normalization and batch unbroadcasting |
| Logical layout/selection | Floating `CONTIGUOUS`, `RESHAPE`, binding-aware `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, `PERMUTE`, normalized `SLICE`, both normalized `SLICE_UPDATE` data roles, `SELECT`, `PAD`, `TILE`, `CONCAT`, and `STACK` |

Forward and generated expressions use one model-owned Tensor algebra, inference/validation
contract, numerical-semantics contract, and exact optimization pipeline. For a selected mixed-
floating input, cotangent normalization first reverses ordinary broadcasting with public
`sumToShape` when needed and then calls ordinary `cast` exactly when the contribution type
differs. The contribution has the selected input's exact Shape and floating data type before it
enters deterministic accumulation. `WHERE` applies this rule independently to its branches;
MATMUL applies it after the rank-specific formula and batch unbroadcasting. Same-type rows add no
cast.

Binary DIV constructs `g / right` for the left input and the exact ordered expression
`-(g * left) / (right * right)` for the right input. Scalar DIV constructs `g / scalar`. These
ordinary operations retain the shared arithmetic, exceptional-value, validation, rewrite, fold,
DCE, and phase-local CSE contracts. There is no gradient-only singularity, NaN, infinity,
signed-zero, overflow, underflow, rounding, or optimization policy.

`FLOOR`, `CEIL`, and `SIGN` use a direct exact positive-zero first-order cotangent at every
represented input, independent of `g`; the rule constructs neither `g * 0` nor a floating
comparison. Ordinary MEAN derives its denominator by reducing an input-shaped exact logical-one
expression over the selected full, single-axis, or ordered multi-axis domain, then divides the
restored cotangent by the expanded count. This formula represents static, dynamic, expression,
zero-sized, and empty-axis-list extents without a host count or exact-integer threshold.

Masked SUM restores the reduced axis, expands the cotangent, and routes it with the original mask
while placing an exact typed zero elsewhere. Masked MEAN constructs its true count with ordinary
`WHERE` and `SUM`, divides the restored cotangent, and uses a final ordinary `WHERE`. The selected
all-false-slice convention returns zero at every input coordinate; it neither suppresses the
intermediate quotient nor defines branch evaluation. BOOL conditions and masks receive no
cotangent. `SUM_TO_SHAPE` is invertible only when every aligned input/target Dimension is either
exactly equal, the static target extent is one, or the binding-dependent inverse uses the exact
forward target-one-or-target-equal-source predicate. Binding-dependent `EXPAND` similarly emits
`source == 1 || source == target` for each unresolved aligned pair. These obligations remain
deferred compile constraints; preflight proves the inverse relationship without binding a symbol.
CUM_SUM retains exclusivity and reverses scan direction; PERMUTE uses the inverse permutation.

Product reduction is division-free: sequential keep-dimension reductions are reversed with
exclusive prefix and suffix products. Cumulative product replaces zeros with one for its safe
product, tracks zero-prefix counts, and accumulates in the opposite direction. Reduction extrema
compare with the exact saved output and divide among all matching positions. A NaN output matches
nothing and therefore selects exact zero.

Softmax and log-softmax consume their exact forward output rather than recomputing it:

```text
softmax:     y * (g - sum(g * y, axis, true))
logSoftmax:  g - exp(y) * sum(g, axis, true)
```

Log-sum-exp uses its restored saved output. Variance and standard deviation derive count and
correction through exact logical-one Tensor expressions. Standard deviation and L2 norm select
zero when their saved result is not strictly positive; L1 norm selects zero at signed zero and
NaN. Other exceptional values flow through ordinary Tensor operations in formula order.

Layer and RMS normalization reconstruct their population statistics over exact trailing axes.
Affine operands are visibly aligned, mixed-floating calculation uses the selected forward-output
type, and each completed contribution is cast back to its selected input type. Batch inference
supports all five floating Tensor inputs. Batch training selects input/scale/bias from output slot
zero, input/running mean from slot one, and input/running variance from slot two. Its formulas
consume the canonical saved mean and inverse-standard-deviation wrappers at slots three and four;
those slots cannot be independent cotangent roots.

ERF constructs `g * exp(-(x * x)) * (2 / sqrt(pi))`. Its coefficient is exact scalar-operation
metadata with fixed BFLOAT16/FLOAT32/FLOAT64 bits `0x3F90`, `0x3F906EBB`, and
`0x3FF20DD750429B6D`; the rule does not evaluate host floating arithmetic.

MIN and MAX split an exact represented-numeric tie equally between Tensor inputs; scalar extrema
give the Tensor receiver one half at a tie. First-class CLAMP applies that rule to the exact
ordered composition `MIN(MAX(x, min), max)`, producing one half at an ordinary endpoint and one
quarter when both stages tie. Opposite signed zeros compare equal. Unordered NaN comparisons
make piecewise extrema, CLAMP, ABS, and RELU return exact positive zero.

Binary POW constructs `g * right * pow(left, right - 1)` and `g * output * log(left)`. Scalar POW
derives exponent-minus-one with exactly one subtraction in the exponent's represented floating
type. RECIPROCAL, LOG, LOG1P, SQRT, and RSQRT use their direct analytic formulas. ABS returns
zero at signed zero and NaN; RELU returns `g` only when its input is strictly greater than
positive zero.

Exact GELU, fixed tanh-approximation GELU, and SiLU use fixed-coefficient analytic formulas for
finite and NaN inputs. They return `g` at positive infinity and exact positive zero at negative
infinity. Other analytic rows add no domain, finite, singularity, or continuous-extension mask;
ordinary Tensor operations determine their exceptional behavior in formula order. Compiler
0005A uses fixed BFLOAT16/FLOAT32/FLOAT64 coefficient bits for one half, negative one half, two,
the inverse-square-root terms, and `0.044715`/`0.134145`; compilation evaluates no host
transcendental coefficient.

MATMUL handles vector-vector, vector-matrix, matrix-vector, and matrix-matrix rank promotion.
Each selected result reverses batch broadcasting and then casts once if the operand type differs
from the promoted type. Integral MATMUL is rejected.
SLICE scatters into an input-shaped typed zero. SLICE_UPDATE masks the base cotangent or extracts
the update cotangent; only the update role requires all selected base extents to be static.
SELECT scatters at one restored axis coordinate. PAD crops its before-width prefix. TILE reshapes
to interleaved repeat/input axes and sums the repeat axes. CONCAT crops by ordered symbolic input
prefixes. STACK selects the corresponding inserted-axis coordinate. Repeated input positions
remain repeated contributions and therefore accumulate deterministically.

Everything outside this table fails closed on a selected route. Compiler 0005C retains remaining
layout/indexing/ordering/stochastic families; Compiler 0005D retains attention, convolution,
pooling, and losses; Compiler 0005E owns the complete first-order closure audit. Comparisons, BOOL
logic/classification, the `WHERE` condition, scalar attributes and bounds, non-floating casts,
`ALL`, `ANY`, arg-extrema outputs, batch saved auxiliary roots, one-hot and other index roles,
masks, and graph RNG state remain non-differentiable.

Preflight is not full graph inference. The compiler performs authoritative inference and
validation after the one combined capture. A later Tensor construction, capture, inference,
validation, or optimization failure can therefore consume temporary `TensorId` values. This is
compatible with the existing opaque, monotonic, non-reusable ID contract.

## Constants and hidden outputs

The implicit unit seed, routing zeros, MEAN logical ones, and scalar values needed as Tensor
comparison operands are storage-free Tensor leaves or expressions. One request-local cache keys
each base by exact `ScalarValue` data type and represented bits, registers it explicitly as one
logical splat, and preserves deterministic first-use order. BFLOAT16 zero/one use exact bits
`0x0000`/`0x3F80`; FLOAT32 and FLOAT64 use exact positive zero/one. Shape-specific values are
ordinary `expand` expressions. Arithmetic-only coefficients remain exact scalar-operation
metadata rather than splat leaves. Host storage, labels, factory history, Shape, layout, and
provenance absence never imply constant status. Only generated bases reachable from returned
gradient expressions remain in combined-capture ingress.

Some formulas need producer outputs omitted from a public ergonomic result. Dropout, for example,
returns the public result and next RNG state while its same-occurrence keep mask is hidden.
Batch-normalization training similarly hides saved batch statistics. Model task 0025 makes each
producer retain the canonical Tensor wrapper for every slot and exposes the smallest indexed
retrieval contract needed by compiler. Compiler 0005B now consumes the exact batch saved mean and
inverse-standard-deviation wrappers from slots three and four without reconstructing a wrapper or
recomputing the statistics.

This creates an intentional reference cycle:

```text
Tensor -> TensorProvenance -> TensorProducer -> canonical outputs -> Tensor
```

The cycle is immutable expression metadata. Factory construction finishes all final fields before
publishing any output, and ordinary garbage collection can reclaim the whole unreachable
occurrence. There is no global registry, weak-reference protocol, graph membership, or runtime
resource ownership.

## One phase-aware capture

Capture receives:

- ordered forward outputs;
- ordered gradient roots with target-specific roles;
- the identity set of original forward producers; and
- explicit constant-splat facts.

It traverses the combined expression once, assigns `NodeId` and `ValueId` once, and gives every
producer occurrence a per-node `FORWARD` or `BACKWARD` phase. A single positional
`backwardStartIndex` cannot replace this phase map.

Multiple targets may share the exact same accumulated-gradient Tensor and therefore one captured
gradient `ValueId`. Result roles still map those targets independently. The graph output boundary
lists each distinct gradient value once; no manufactured identity node is needed.

## Combined optimization

The immutable combined graph, not a forward-only prefix, enters optimization. Compiler task 0004
adapts the completed task-0003, 0003A, and 0003B orchestration:

- canonicalization remains mandatory;
- the already selected exact rewrites and constant folds may apply in either phase only when
  their existing guards remain valid;
- dead-code elimination sees the whole graph;
- common-subexpression elimination remains phase-local initially; and
- every changed candidate returns through Compiler 0002 validation.

The sequence runs once: canonicalize/validate, exact rewrite, exact fold, whole-graph dead-code
elimination, phase-local common-subexpression elimination, then whole-graph cleanup dead-code
elimination. This migration authorizes no new rewrite, fixed point, relaxed arithmetic, floating
evaluation, or physical constant materialization.

## Compile modes and future derivatives

`FORWARD_ONLY` skips autograd. `FORWARD_AND_BACKWARD` and the initial `TRAINING_STEP` build the
combined expression before capture. `TRAINING_STEP` does not add optimizer updates yet.

Generated gradients are ordinary differentiable Tensor expressions, preserving a route to higher
derivatives. Higher derivatives are not part of the current first-order implementation. A later
task must define an
explicit create-graph or derivative-order lifecycle contract, provide rules for every operation
used in gradient formulas, and represent derivative order in addition to graph phase.

The design adds no `Tensor.gradient`, `Tensor.backward`, mutable gradient field, ThreadLocal
compilation scope, model-owned derivative rule, public compiler registry, physical saved buffer,
or backend-owned global autograd.

See [Training graph](../../architecture/training-graph.md),
[ADR 0009](../decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md), the
[model master plan](../../planning/modules/model/master-plan.md), and the
[compiler master plan](../../planning/modules/compiler/master-plan.md).
