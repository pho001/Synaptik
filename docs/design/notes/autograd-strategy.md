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
Public gradient requests, publication, compile artifacts, higher derivatives, optimizer updates,
preparation, and execution remain planned.

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
  ancestry, each with the objective's exact type, gradient eligibility, and a selected
  differentiable route; and
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
compiler combines contributions with ordinary `Tensor.add`. These maps are temporary bookkeeping,
not graph IR, public Tensor state, a tape, or a registry.

## Preflight and construction failures

Before constructing a backward expression, the compiler inventories every backward-reachable
producer occurrence, output role, exact attributes, and required derivative policy. Unsupported
or ambiguous work fails closed. This prevents a known incomplete rule matrix from creating a
partial backward expression.

The closed union of `SUPPORTED_0004` and `SUPPORTED_0004A` is:

| Family | Current exact variants |
|---|---|
| Elementwise | Same-floating-type binary and exact-scalar `ADD`/`SUB`/`MUL`; same-type branch-only `WHERE`; same-type floating `CAST`; `NEG`/`EXP`/`EXPM1`/`SIGMOID`/`TANH`/`ERF` |
| Reduction/scan | Floating ordinary full, single-axis, and multi-axis `SUM`; masked floating `SUM`; locally invertible floating `SUM_TO_SHAPE`; floating `CUM_SUM` |
| Linear algebra | Every current floating `MATMUL` vector/matrix rank pairing, with role-aware selected-operand type checks and batch unbroadcasting |
| Logical layout/selection | Floating `CONTIGUOUS`, `RESHAPE`, `EXPAND`, `EXPAND_DIMS`, `SQUEEZE`, `PERMUTE`, normalized `SLICE`, both normalized `SLICE_UPDATE` data roles, `SELECT`, `PAD`, `TILE`, `CONCAT`, and `STACK` |

Broadcast reversal uses public `sumToShape`; ordinary SUM restores removed axes before expansion;
CUM_SUM retains exclusivity and reverses scan direction; and PERMUTE uses the inverse permutation.
WHERE's BOOL condition is non-differentiable. Masked SUM restores the reduced axes, expands the
cotangent, and routes it with the original mask while placing an exact typed zero elsewhere.
`SUM_TO_SHAPE` is invertible only when every aligned input/target Dimension is either exactly
equal or the static target extent is one.

ERF constructs `g * exp(-(x * x)) * (2 / sqrt(pi))`. Its coefficient is exact scalar-operation
metadata with fixed BFLOAT16/FLOAT32/FLOAT64 bits `0x3F90`, `0x3F906EBB`, and
`0x3FF20DD750429B6D`; the rule does not evaluate host floating arithmetic.

MATMUL handles vector-vector, vector-matrix, matrix-vector, and matrix-matrix rank promotion.
Each selected operand must have the output floating type; an unselected narrower operand is
permitted because no cotangent conversion is needed for that role. Integral MATMUL is rejected.
SLICE scatters into an input-shaped typed zero. SLICE_UPDATE masks the base cotangent or extracts
the update cotangent; only the update role requires all selected base extents to be static.
SELECT scatters at one restored axis coordinate. PAD crops its before-width prefix. TILE reshapes
to interleaved repeat/input axes and sums the repeat axes. CONCAT crops by ordered symbolic input
prefixes. STACK selects the corresponding inserted-axis coordinate. Repeated input positions
remain repeated contributions and therefore accumulate deterministically.

Everything outside this table fails closed on a selected route. Task 0004B retains
mixed-floating cotangent conversion and tie/endpoint/discontinuity/singularity/empty-domain/
NaN/infinity policies. Current exclusions include division and power; extrema, clamp, and other
nonsmooth families; reciprocal/log/root families; product, mean, extrema, and statistical
reductions; `CUM_PROD`; softmax and normalization; indexing families outside the listed exact
layout rules; random/dropout; losses; and other unlisted operation families.

Preflight is not full graph inference. The compiler performs authoritative inference and
validation after the one combined capture. A later Tensor construction, capture, inference,
validation, or optimization failure can therefore consume temporary `TensorId` values. This is
compatible with the existing opaque, monotonic, non-reusable ID contract.

## Constants and hidden outputs

The implicit unit seed, WHERE routing zeros, and other derivative constants are storage-free
Tensor leaves or expressions. The compiler registers each BFLOAT16, FLOAT32, or FLOAT64 zero/one
base explicitly with one exact logical-splat fact for combined capture. BFLOAT16 zero/one use
exact bits `0x0000`/`0x3F80`; FLOAT32 and FLOAT64 use exact positive zero/one. The ERF coefficient
is instead exact scalar-operation metadata and is not registered as a splat leaf. Host storage,
labels, factory history, Shape, layout, and provenance absence never imply constant status.

Some formulas need producer outputs omitted from a public ergonomic result. Dropout, for example,
returns the public result and next RNG state while its same-occurrence keep mask is hidden.
Batch-normalization training similarly hides saved batch statistics. Model task 0025 makes each
producer retain the canonical Tensor wrapper for every slot and exposes the smallest indexed
retrieval contract needed by compiler. It never reconstructs an equal wrapper.

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
derivatives. Higher derivatives are not part of Compiler 0004. A later task must define an
explicit create-graph or derivative-order lifecycle contract, provide rules for every operation
used in gradient formulas, and represent derivative order in addition to graph phase.

The design adds no `Tensor.gradient`, `Tensor.backward`, mutable gradient field, ThreadLocal
compilation scope, model-owned derivative rule, public compiler registry, physical saved buffer,
or backend-owned global autograd.

See [Training graph](../../architecture/training-graph.md),
[ADR 0009](../decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md), the
[model master plan](../../planning/modules/model/master-plan.md), and the
[compiler master plan](../../planning/modules/compiler/master-plan.md).
