# Task 0004: Compiler-owned Pre-capture Autograd and Graph Compilation

## Status

Complete

## Goal

Implement the first bounded compiler-owned reverse-mode automatic-differentiation path over the
original Tensor expression DAG, then capture the requested forward results and generated
first-order gradients together exactly once.

The task must fail closed before allocating any derivative Tensor identity when the requested
backward slice contains an unsupported occurrence, output role, attributes variant, data-type
relationship, or derivative policy. A successful backward-capable request constructs formulas
only with ordinary public `Tensor` operations, explicitly registers storage-free logical-splat
constants, assigns graph-local identities once during one phase-aware combined capture, validates
the combined graph, and applies only the exact transformations already selected by Compiler
0003–0003B.

This task also establishes the smallest package-private scalar-objective and gradient-target
contract needed to prove the pipeline. It does not establish the future public compile,
publication, arbitrary-seed, or higher-derivative APIs.

## Scope

### Mandatory obsolete-source cleanup

Before creating or modifying any implementation Java file, verify that these exact six paths exist
and are reported as untracked (`??`) by `git status --short -- <paths>`:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradExpansion.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradGraph.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BackwardGraphBuilder.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseAdjoints.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/IndexingRandomAdjoints.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionLayoutAdjoints.java`

They are obsolete pre-planning prototypes located under the compiler production source root, so
leaving them present would make Gradle compile unauthorized code. After the exact path/status
check succeeds, delete all six as the first implementation cleanup step and verify their absence
before compiling or testing.

The cleanup is removal-only. Do not read them for design evidence, copy, rename, move, stage,
adapt, salvage, quote, or reintroduce any content from them. They are not repository history,
legacy capability evidence, architecture, accepted design, or an implementation starting point.
If any path is missing, tracked, modified in Git rather than untracked, or replaced by a different
filesystem type at implementation start, stop and report the status mismatch before touching
implementation files.

### Package-private compilation request

Add one package-private stateless `GraphCompiler` entry point with this exact parameter list:

```java
static GraphCompilation compile(
        CompileMode mode,
        List<Tensor> forwardOutputs,
        Optional<AutogradPreflight.FirstOrderRequest> firstOrderRequest,
        CompileTimeConstantGraph.Ingress forwardConstants,
        GraphOptimizationConfig optimizationConfig)
```

Do not wrap these parameters in a request aggregate or add another entry type. Their types and
order are fixed by this task. The implementation must not add a public facade or aggregate.
Validation is deterministic and checks the arguments in declaration order before graph
construction.

`forwardOutputs` is a non-null, non-empty ordered snapshot. Its Tensor references must be
identity-unique and must resolve to distinct logical forward values, preserving Compiler 0001
behavior. The complete original producer inventory starts from every forward output, not only the
backward objective, because capture phase classification covers the complete mode-selected
request.

The mode matrix is exact:

| `CompileMode` | First-order request | Behavior |
|---|---|---|
| `FORWARD_ONLY` | Must be absent. | Inventory and phase-aware capture the requested forward DAG, perform no derivative preflight or construction, then validate and optimize the forward-only graph-stage result. |
| `FORWARD_AND_BACKWARD` | Must be present. | Preflight and construct the first-order backward slice, then use the combined pipeline. |
| `TRAINING_STEP` | Must be present. | Perform exactly the same work as `FORWARD_AND_BACKWARD`; add no optimizer, optimizer-update node, session, schedule, preparation, or execution state. |

No mode is inferred from Tensor metadata. A mode/request mismatch fails before derivative
construction.

### First-order objective, target, and seed contract

`AutogradPreflight.FirstOrderRequest` is a package-private immutable request containing:

```java
record FirstOrderRequest(Tensor objective, List<Tensor> targets)
```

Its exact contract is:

- `objective` is one exact Tensor reference present in `forwardOutputs`;
- the objective has scalar `Shape`, one floating data type, and `requiresGrad == true`;
- `targets` is a non-null, non-empty ordered immutable snapshot of identity-unique exact Tensor
  references;
- each target occurs in the original objective ancestry, has the same floating data type as the
  objective along the selected supported path, and has `requiresGrad == true`;
- a target may be a provenance-free leaf or an intermediate original Tensor;
- the objective itself may be a target;
- each target must be reachable through at least one differentiable input-role path selected by
  this task's matrix; an unrelated or condition-only target is rejected rather than assigned an
  invented zero;
- target order determines result-role order and is otherwise not a graph traversal order; and
- targets are not traversal stop points, because another requested target may be upstream of an
  intermediate target.

The only current seed is an implicit rank-zero positive one with the objective's exact floating
data type. There is no caller-supplied seed Tensor. The compiler creates that seed only after
preflight succeeds as a provenance-free, storage-free, non-gradient Tensor leaf and explicitly
registers its exact logical-splat value. BFLOAT16 one uses exact bits `0x3F80`; FLOAT32 and FLOAT64
use exact positive `1.0` in their respective types.

Arbitrary objectives, non-scalar objectives, explicit seeds, vector-Jacobian products,
disconnected-target zero policy, public target/publication requests, and multiple derivative
orders remain Compiler 0006 work.

### Original forward inventory and fail-closed preflight

Add `AutogradPreflight` as one package-private stateless owner of:

- a deterministic iterative inventory of exact original Tensor and `TensorProducer` identities;
- producer postorder and reverse-postorder views;
- exact producer inputs, every canonical output wrapper and output index, exact `Operation`,
  attributes runtime type and value, and descriptor facts;
- objective-to-target differentiable-role reachability; and
- the closed rule-selection decision for every occurrence/output/input role required by the
  request.

Traversal must use object identity, not `TensorId`, record equality, operation equality, structural
hashing, labels, storage, or captured graph IDs. It must be iterative so a deep expression chain
does not consume the Java call stack. Structurally equal but separately invoked producers remain
distinct occurrences. Repeated exact input positions remain repeated.

Preflight proceeds in this order:

1. validate the top-level mode, forward-output, optional request, ingress, and optimization
   arguments without creating a Tensor;
2. inventory the complete original forward DAG and validate the request's objective/target
   membership;
3. derive, by exact Tensor identity, which objective ancestry branches can reach at least one
   requested target through differentiable roles;
4. inspect every required producer occurrence, exact output role, input role, kind, attributes
   variant and value, descriptor, data-type relationship, Shape relationship, and required
   derivative policy;
5. reject the first unsupported fact with deterministic occurrence, output-role, input-role,
   operation-kind, attributes, and reason context; and
6. only after the complete selected slice succeeds, create the implicit seed, derivative
   constants, or any formula Tensor.

An unsupported branch that cannot reach any requested target is not differentiated and does not
require a rule. A non-differentiable role stops that route. An unsupported occurrence on a
selected objective-to-target route rejects the complete request. Preflight is not a replacement
for Compiler 0002 inference and validation; post-capture verification remains authoritative.

Known preflight rejection must consume no new `TensorId`. Once construction begins, a later public
Tensor-construction, capture, inference, validation, or optimization failure may consume IDs.
Identifiers are never rolled back or reused.

### Closed `SUPPORTED_0004` derivative matrix

Only the rows below are supported. Each selected occurrence must have its current exact
family-owned signature, output count, attributes variant, descriptor relationship, and a floating
gradient path. `g` denotes the accumulated cotangent of the selected output, `x`/`left`/`right`
denote exact original inputs, `y` denotes the producer's canonical exact output wrapper, and
`B_input(v)` is `v.sumToShape(input.descriptor().shape())`.

| Family and exact variant | Additional preflight guard | Formula for selected differentiable input roles |
|---|---|---|
| `BinaryArithmeticKind.ADD` with `NoOperationAttrs.INSTANCE` | Left, right, and output have one exact equal floating type; mixed floating promotion is rejected. | `B_left(g)`, `B_right(g)` |
| `BinaryArithmeticKind.SUB` with `NoOperationAttrs.INSTANCE` | Same-type guard as ADD. | `B_left(g)`, `B_right(g.neg())` |
| `BinaryArithmeticKind.MUL` with `NoOperationAttrs.INSTANCE` | Same-type guard as ADD. | `B_left(g.mul(right))`, `B_right(g.mul(left))` |
| `ScalarElementwiseKind.ADD` or `SUB` with exact `ScalarValueAttrs` | Input, output, and scalar attribute have one exact equal floating type. | `g` |
| `ScalarElementwiseKind.MUL` with exact `ScalarValueAttrs` | Same exact-type guard as scalar ADD/SUB. | `g.mul(attrs.value())` |
| `WhereSelectionKind.WHERE` with `NoOperationAttrs.INSTANCE` | Condition is exact BOOL; both branches and output have one exact equal floating type, so no branch promotion/cotangent cast is required. | True branch: `B_true(Tensor.where(condition, g, Z_y))`; false branch: `B_false(Tensor.where(condition, Z_y, g))`; condition is non-differentiable. |
| `CastKind.CAST` with exact `CastAttrs` | Source and target are the same exact floating type. | `g` |
| `UnaryElementwiseKind.NEG` with `NoOperationAttrs.INSTANCE` | Input/output exact same floating type and Shape. | `g.neg()` |
| `UnaryElementwiseKind.EXP` with `NoOperationAttrs.INSTANCE` | Input/output exact same floating type and Shape. | `g.mul(y)` |
| `UnaryElementwiseKind.EXPM1` with `NoOperationAttrs.INSTANCE` | Input/output exact same floating type and Shape. | `g.mul(y.add(O_y))` |
| `UnaryElementwiseKind.SIGMOID` with `NoOperationAttrs.INSTANCE` | Input/output exact same floating type and Shape. | `g.mul(y).mul(O_y.sub(y))` |
| `UnaryElementwiseKind.TANH` with `NoOperationAttrs.INSTANCE` | Input/output exact same floating type and Shape. | `g.mul(O_y.sub(y.mul(y)))` |
| `AggregateReductionKind.SUM` with `NoOperationAttrs.INSTANCE` | One exact floating input/output type; this is full reduction only. | Restore every removed axis as extent one, then expand to `x.shape()`. |
| `AggregateReductionKind.SUM` with exact `AxisReductionAttrs` | One exact floating input/output type and the recorded normalized axis/keep-dimensions facts. | If the axis was removed, `expandDims` at that axis; then expand to `x.shape()`. |
| `AggregateReductionKind.SUM` with exact `MultiAxisReductionAttrs` | One exact floating input/output type and ordered distinct normalized axes; an empty list is a point reduction. | If dimensions were removed, insert the selected axes in ascending axis order; then expand to `x.shape()`. Empty axes return `g`. |
| `CumulativeScanKind.CUM_SUM` with exact `CumulativeScanAttrs(axis, exclusive, reverse)` | Input/output exact same floating type and Shape. | `g.cumSum(axis, exclusive, !reverse)` |
| `ContiguousKind.CONTIGUOUS` with `NoOperationAttrs.INSTANCE` | Input/output exact same floating type and Shape. | `g` |
| `ShapeTransformKind.RESHAPE` with exact `TargetShapeAttrs` | Input/output exact same floating type and the current forward element-count contract. | `g.reshape(x.shape())` |
| `ShapeTransformKind.EXPAND` with exact `TargetShapeAttrs` | Input/output exact same floating type and current locally proved expand relationship. | `B_x(g)` |
| `AxisTransformKind.EXPAND_DIMS` with exact `AxisTransformAttrs` | Input/output exact same floating type and recorded normalized result axis. | `g.squeeze(axis)` |
| `AxisTransformKind.SQUEEZE` with exact `AxisTransformAttrs` | Input/output exact same floating type and recorded normalized input axis. | `g.expandDims(axis)` |
| `AxisTransformKind.PERMUTE` with exact `PermutationAttrs` | Input/output exact same floating type and a complete permutation. | `g.permute(inversePermutation)` |

`O_y` and `Z_y` are the compiler's explicit exact typed logical one and positive-zero expressions
expanded to `y.shape()`. They are not host-storage values, eager factory constants, descriptor
inference, labels, or scalar literals hidden from capture.

The formulas may reuse exact forward Tensor wrappers, including `producer.output(outputIndex)`.
They must never reconstruct an equal output wrapper from a descriptor. The selected first matrix
does not need a hidden auxiliary output, but the traversal and capture contracts must retain every
output role so later named rule families can consume dropout masks and saved batch-normalization
statistics without another pipeline redesign.

### Explicitly deferred and non-differentiable work

Preflight rejects every operation or variant not listed in `SUPPORTED_0004`; it does not fall
through to a generic rule. The following ownership split is explicit:

| Category | Decision |
|---|---|
| Mixed-floating binary arithmetic or WHERE branch promotion | Deferred. Backward would require a selected cross-floating cotangent conversion. |
| Cross-floating `CAST` | Deferred to Compiler 0004B's explicit conversion policy. Floating-to-integral, integral-to-floating, integral, and BOOL roles are non-differentiable. |
| Binary/scalar `DIV` and `POW`; `RECIPROCAL`, `LOG`, `LOG1P`, `SQRT`, `RSQRT` | Deferred to 0004B because zero, invalid-domain, singular, infinity, NaN, and exceptional-value derivative behavior is unresolved. |
| Binary/scalar `MIN`/`MAX`, `CLAMP`, `ABS`, `RELU`, `FLOOR`, `CEIL`, `SIGN` | Deferred to 0004B because tie, endpoint, discontinuity, signed-zero, NaN, or subgradient behavior is unresolved. |
| `ERF`, GELU variants, `SILU`, and other exact-composition unary additions | Deferred to 0004A; this task does not expand its closed core matrix or select coefficient handling. |
| SUM with `MaskedReductionAttrs` or `SumToShapeAttrs`; MEAN, PROD, reduction extrema, advanced/statistical/norm reductions, softmax/log-softmax, and normalization | Deferred to 0004A when policy-free and otherwise 0004B. Empty-domain, tie, zero-norm, singular, and exceptional policies are not selected here. |
| `CUM_PROD` | Deferred. This task selects no product special-value derivative behavior. |
| Matrix multiplication, attention, convolution, pooling, losses, indexing/scatter, slicing/pad/tile/composition/windows, ordering/top-k, dropout, and batch normalization | Deferred to later exact-rule or policy tasks. Current public primitives and canonical auxiliary outputs remain evidence, not authorization to widen 0004. |
| Comparisons, BOOL logic and classification, BOOL reductions, arg-extrema/index outputs, one-hot indices, and graph RNG state | Non-differentiable roles. They may occur on an unselected route or serve as fixed conditions, but no cotangent propagates through them. |
| Unknown/custom `OperationKind`, wrong attributes class, wrong cardinality, missing canonical output, or descriptor contradiction | Deterministic preflight failure. |

Compiler 0004A may add only separately specified exact-composition formulas after this pipeline is
complete. Compiler 0004B must select explicit derivative policies before adding any
tie/discontinuity/singularity/cross-floating-dependent rule. Neither follow-up may silently widen
this matrix.

### Named gradient rules and deterministic reverse accumulation

Add exactly these named stateless package-private rule owners:

- `ElementwiseGradientRules` — binary/scalar arithmetic, WHERE, same-type cast, and the selected
  unary formulas;
- `ReductionGradientRules` — ordinary SUM and CUM_SUM; and
- `LayoutGradientRules` — contiguous, reshape, expand, rank edits, and permutation.

Dispatch is by typed `OperationKind`/attributes tests and exhaustive closed switches. Do not use
strings, reflection, service loading, a public or mutable registry, `Map<OperationKind, ...>`,
generic symbolic algebra, direct `Operation`/`CompiledNode` construction, or backend dispatch.
Rules call only current public `Tensor` methods. They do not perform graph capture, inference,
numerical execution, storage access, or lowering.

`FirstOrderAutograd` owns one deterministic reverse traversal over the successful preflight plan.
During one request it may retain only ephemeral identity-based maps for:

- exact Tensor to ordered incoming contribution list;
- exact Tensor to accumulated gradient Tensor; and
- request-local generated zero/one leaves and their explicit splat bindings.

The traversal starts with the implicit seed contribution to the objective. It processes producers
in deterministic reverse postorder after all downstream contributions for the selected output
role are known. For each processed role it accumulates contributions with left-associated
ordinary `Tensor.add` in insertion order. It visits differentiable input positions in operation
order and appends each contribution immediately, so a repeated operand receives repeated
contributions in position order. Consumer ordering follows the deterministic original producer
postorder. It does not sort by Tensor ID, kind name, hash, label, or structural equality.

Distinct targets may resolve to the exact same accumulated gradient Tensor. The expansion returns
one ordered target-to-gradient role per request target and does not manufacture identity,
reshape, add-zero, or copy expressions to make roles distinct.

All maps and Tensor references used for backward construction are discarded after combined
capture. They
are not stored in `Tensor`, `TensorProvenance`, `TensorProducer`, `CompiledGraphModel`,
`GraphCompilation`, runtime state, or a global cache.

### Derivative constants

Generated positive zero and positive one are exact BFLOAT16, FLOAT32, or FLOAT64 values matching
the formula type. Each base constant is:

- a provenance-free Tensor made with public `TensorFactory.create`;
- rank zero with unresolved layout and `requiresGrad == false`;
- free of label and host storage; and
- bound explicitly through `CompileTimeConstantGraph.Binding` and `Splat`.

Within one expansion, a request-local helper may reuse at most one base zero and one base one per
floating data type, in deterministic first-use order. Shape-specific constants are public
`expand(Shape)` expressions from those base leaves. Equal caller-supplied and generated splats are
not inferred to be the same source. Merge forward ingress bindings first in caller order and
generated derivative bindings second in deterministic creation order, rejecting an identity
collision.

No constant is inferred from factory history, host storage, descriptor/layout facts, label,
provenance absence, Shape, zero extent, or numerical evaluation. The compiler reads no Tensor
payload.

### One phase-aware combined capture

Extend `GraphCapture` with one package-private combined entry that receives:

- ordered forward outputs;
- ordered target/gradient Tensor roles;
- the identity-based inventory of every original forward producer;
- the merged explicit constant ingress; and
- no captured graph or placeholder Tensor.

The combined traversal is iterative and assigns each `NodeId` and `ValueId` exactly once in
deterministic input-first/depth-first postorder. Every declared producer output position becomes
a graph value, including hidden or opaque positions. A producer whose exact identity is in the
original inventory receives `GraphPhase.FORWARD`; every generated producer receives
`GraphPhase.BACKWARD`. Leaf values have no phase. A positional `backwardStartIndex` is forbidden.

The graph-output boundary is:

1. every distinct requested forward value in forward request order; then
2. each gradient value not already present, at its first target-role occurrence.

Duplicate gradient roots and a gradient already present at the forward boundary are not repeated.
The combined capture retains, for each target role, the exact graph-output ordinal to which it
resolved. This positional role metadata, not a Tensor retained after capture, remaps roles through
later canonicalization and optimization.

Existing `GraphCapture.capture` entries remain package-private and behaviorally compatible for
Compiler 0001/0003B callers and tests. They may delegate to a shared implementation, but they
continue to classify every producer as `FORWARD` and preserve their current duplicate-output
failure contract.

### Immutable mode-neutral graph-stage result

Add package-private immutable `GraphCompilation` containing:

```java
record GraphCompilation(
        CompileMode mode,
        ValidatedGraph validatedGraph,
        List<ValueId> forwardOutputs,
        List<GradientResultRole> gradientResults)
```

`GradientResultRole` is an immutable nested record containing the exact requested target's
`TensorId` and its final gradient `ValueId`. The list order equals request target order. Distinct
target IDs may name the same gradient value. No Tensor reference is retained.

The constructor snapshots lists and validates:

- every forward and gradient value exists and is a graph output;
- forward outputs are identity-unique by `ValueId` and preserve the requested prefix;
- the graph output boundary equals the forward prefix followed by stable first-occurrence
  de-duplication of gradient values not already present;
- no gradient role exists in `FORWARD_ONLY`;
- backward-capable modes have the non-empty role list produced by their request; and
- each role has non-null target and gradient IDs.

`GraphCompilation` is the mode-neutral package-private result of this internal graph stage.
`FORWARD_ONLY` contains a forward-only validated graph, no `GraphPhase.BACKWARD` nodes, and an
empty `gradientResults` list. `FORWARD_AND_BACKWARD` and `TRAINING_STEP` may contain the combined
forward/backward graph and non-empty target-specific gradient roles.

The exact `ValidatedGraph` retains the final `CompileTimeConstantGraph`, constraints, constants,
bindable inputs, structural graph, graph boundaries, and per-node phase facts. `GraphCompilation`
is not the later public or cross-package aggregate `CompileArtifacts`: it adds no publication
binding, publication plan, logical-memory/partition plan, physical constant plan, diagnostics
bundle, backend choice, schedule, prepared state, or executable artifact.

### Mode-selected graph inference and exact optimization

The immutable graph captured for the selected mode is the sole optimization unit. In
backward-capable modes that unit is the combined forward/backward graph. Preserve this exact
one-shot order:

1. one phase-aware capture, combined for backward-capable modes;
2. `CapturedGraphInference.inferAndValidate` on the captured constant sidecar;
3. mandatory dense `GraphCanonicalization` and validation;
4. if optional optimization is disabled, return;
5. existing seven-rule exact arithmetic rewriting once;
6. existing bounded BOOL/signed-integral logical-splat folding once;
7. whole-graph sidecar-aware DCE once;
8. exact phase-local CSE once;
9. whole-graph sidecar-aware cleanup DCE once; and
10. return the final validated result, remapping forward/gradient roles by preserved graph-output
    ordinal after every ID rebuild.

Every changed candidate is revalidated through Compiler 0002 before the next pass consumes it.
An unchanged helper result is not redundantly validated. No fixed point or extra scan is added.

The completed package-private helper names may remain `Forward*` to keep this already large
pipeline migration bounded, but their Javadocs and behavior must no longer use the name as a
phase filter:

- arithmetic rewriting may consider FORWARD or BACKWARD nodes while retaining every existing
  0003A guard: the closed seven rules, exact descriptor equality, `requiresGrad == false`,
  one non-graph-output result, and no new algebra;
- constant folding may consider either phase while retaining the exact 0003B operation/type
  matrix, all graph-output/multi-output exclusions, exact typed evaluation, and bounded work;
- DCE derives liveness only from the complete selected graph output boundary and retained
  dependencies; a BACKWARD node is not automatically live merely because of its phase;
- CSE may consider both phases, but `GraphPhase` remains in the complete expression key and
  candidates never merge across phases; graph-output producers remain excluded; and
- constant sidecar roles, graph-output order, complete multi-output occurrences, exact operation
  and descriptor references, and phase facts survive every rebuild.

This task authorizes no new rewrite, floating/BFLOAT16 evaluation, cast folding, reassociation,
relaxed arithmetic, approximate rule, constant materialization, or cross-phase CSE.

## Out of scope

- any public compiler, gradient, compile-request, compile-result, publication, or artifact type
- `CompileArtifacts`, `CompileConfig`, public objective/target/seed selection, gradient
  publication bindings, or engine integration
- non-scalar objectives, explicit cotangent seeds, Jacobian construction, higher-order
  differentiation, create-graph flags, derivative-order state, or differentiating generated
  gradients
- every derivative rule or attributes variant outside `SUPPORTED_0004`
- selecting a tie, endpoint, discontinuity, singularity, invalid-domain, NaN/infinity,
  zero-count/zero-norm, or cross-floating conversion derivative policy
- optimizer semantics, optimizer-update graph nodes, parameter mutation, `TrainingSession`, or
  training execution
- `Tensor.gradient`, `Tensor.backward`, mutable gradient fields, model-owned rules, a runtime
  tape, ThreadLocal compilation state, global caches, registries, service locators, plugins, or
  reflection/string dispatch
- placeholder Tensors for `ValueId`, conversion from captured graph values back to Tensor
  expressions, a second algebra, or direct compiler construction of model IR nodes for formulas
- reconstructing a canonical producer output wrapper, changing `TensorProducer`,
  `TensorFactory`, provenance, Tensor methods, result carriers, model operation semantics, or
  model tests
- reading or materializing host/device storage, evaluating Tensor values, buffers, planning,
  lowering, backend work, prepare, runtime, engine, execution, or trace events
- new arithmetic rewrites, new constant-fold rows, pass iteration, cross-phase CSE, or approximate
  math
- architecture, ADR, module dependency, Gradle/build, Java-version, architecture-test,
  backend-conformance, or integration-test changes
- implementing Compiler 0004A, 0004B, 0005, or 0006 or creating their detailed specifications
- reading, copying, adapting, moving, renaming, staging, or treating the six removal-only obsolete
  prototype Java files as design authority; their required deletion is the sole permitted action
  on those paths

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially compiler ownership,
  pre-capture autograd, combined capture, phases, constant facts, validation, and lifecycle
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [Superseded ADR 0005](../../../../design/decisions/0005-training-combined-forward-backward-graph.md)
- [Autograd strategy](../../../../design/notes/autograd-strategy.md)
- [Planning guide](../../../planning-guide.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0001](0001-tensor-expression-graph-capture.md)
- [Compiler 0002](0002-captured-graph-inference-and-validation.md)
- [Compiler 0003](0003-canonicalization-and-forward-optimization.md)
- [Compiler 0003A](0003a-exact-arithmetic-rewriting.md)
- [Compiler 0003B](0003b-compile-time-constants-and-constant-folding.md)
- [Model master plan](../../model/master-plan.md)
- [Model capabilities](../../model/capabilities.md)
- [Adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)
- [Model contract-closure audit](../../model/model-capability-contract-closure-audit.md)
- [Model 0025](../../model/tasks/0025-canonical-tensor-producer-outputs.md)
- [Compile API](../../../../api/compile-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative. The accepted pre-capture design already covers this task;
  no architecture or ADR change is expected.
- Compiler owns reverse traversal, derivative-rule selection, accumulation, phase-aware capture,
  validation, and exact combined optimization.
- Model owns Tensor operations, producer occurrence identity, canonical output wrappers,
  descriptors, operation vocabulary, and immutable compiled graph values. It owns no derivative
  rule or backward lifecycle.
- Compiler formulas use only public Tensor operations and explicit storage-free leaves. Captured
  graph IDs are assigned once, after formulas exist.
- Compiler result state is immutable, backend-neutral, and contains no runtime, prepare, concrete
  backend, physical storage, or execution object.
- Existing dependency direction remains unchanged. No architecture test is required because no
  dependency rule changes.
- If implementation needs a model API, public compiler API, additional derivative policy,
  different graph phase model, second algebra, or path outside the maximum scope, stop and report
  the exact conflict instead of inventing a contract.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — the cohesive package-private compiler front-end boundary.

Packages added or changed:

- No package is added. No public package surface changes.

Type placement:

- `io.github.pho001.synaptik.compiler.AutogradPreflight` — owns the nested package-private
  first-order request, original identity inventory, target-path analysis, and complete preflight
  plan.
- `io.github.pho001.synaptik.compiler.ElementwiseGradientRules` — owns the selected typed
  elementwise formulas.
- `io.github.pho001.synaptik.compiler.ReductionGradientRules` — owns the selected typed reduction
  and scan formulas.
- `io.github.pho001.synaptik.compiler.LayoutGradientRules` — owns the selected typed logical
  layout-transform formulas.
- `io.github.pho001.synaptik.compiler.FirstOrderAutograd` — owns reverse traversal,
  identity-keyed contribution accumulation, implicit seed, derivative constants, and ordered
  target/gradient construction results.
- `io.github.pho001.synaptik.compiler.GraphCapture` — remains the sole Tensor/producer
  identity-to-graph-ID boundary and gains combined phases, role ordinals, and output
  de-duplication.
- `io.github.pho001.synaptik.compiler.GraphCompilation` — owns the mode-neutral immutable
  validated graph-stage result, forward boundary, and optional target-specific gradient roles;
  it is distinct from later `CompileArtifacts`.
- `io.github.pho001.synaptik.compiler.GraphCompiler` — owns exact mode routing and the
  capture/infer/optimize/result orchestration without adding a request aggregate.
- Existing `ForwardGraphOptimization`, exact rewriting/folding, DCE, and CSE helpers remain
  package-private and are generalized only as specified for a backward-capable combined graph.

Tests mirror the production package to exercise package-private contracts.

## Affected files

Expected production files:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompilation.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCapture.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimization.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardExactArithmeticRewriting.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardConstantFolding.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardDeadCodeElimination.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardCommonSubexpressionElimination.java`

Expected test files:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCaptureTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimizationTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardExactArithmeticRewritingTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardConstantFoldingTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardDeadCodeEliminationTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardCommonSubexpressionEliminationTest.java`

Expected documentation and planning files:

- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`
- `docs/api/training-api.md`
- `docs/design/notes/autograd-strategy.md`
- `docs/user-guide/autograd.md`
- `docs/glossary.md`
- `docs/planning/modules/compiler/tasks/0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Required removal-only pre-existing untracked paths:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradExpansion.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradGraph.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BackwardGraphBuilder.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseAdjoints.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/IndexingRandomAdjoints.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionLayoutAdjoints.java`

## Maximum scope

This task has a 38 touched-path ceiling:

- at most the 32 expected tracked create/modify paths listed above; plus
- exactly the six pre-existing untracked production-source paths listed under required
  removal-only paths, which must only be verified and deleted.

The higher-than-normal path count is justified by one indivisible invariant explicitly assigned to
Compiler 0004 by the roadmap: preflight must cover the complete selected backward slice before any
formula Tensor exists; formula results and generated splats must enter the same identity-aware
capture; combined output roles must survive every ID-rebuilding pass; and the completed
forward-only optimizer must become exact for a backward-capable combined graph before a result is
returned.
Splitting combined optimization into a 0004C would leave an invalid intermediate contract and is
explicitly rejected by the compiler master plan and roadmap. The task remains bounded by one
module, one closed rule matrix, package-private types, and the exact path inventory above.

If another tracked file, type, test, documentation path, operation rule, model change, or build
change is required, stop and propose a follow-up or task-spec revision before implementation.
If any cleanup path would require content modification, migration, staging, or preservation rather
than deletion, stop instead of exceeding its removal-only authorization.

## Acceptance criteria

### Request and preflight

- The package-private compile entry validates the exact mode/request matrix and preserves
  declaration-order null/failure behavior.
- The first-order request supports exactly one scalar floating objective, one implicit exact
  typed unit seed, and an ordered identity-unique non-empty target list.
- Inventory covers every original forward producer and every canonical output role by exact
  object identity without recursion.
- Target reachability prunes branches unrelated to every requested target and stops
  non-differentiable roles.
- The complete selected slice is preflighted before the first seed, constant, or formula Tensor
  is created.
- Every unsupported kind/attrs/type/role/policy failure reports deterministic contextual
  diagnostics and consumes no new Tensor ID.
- Unknown operation families fail closed; no fallback or registry exists.

### Rule matrix and accumulation

- Tests exercise every `SUPPORTED_0004` row, attributes variant, input role, exact same-type guard,
  broadcasting reversal, rank restoration, inverse permutation, and CUM_SUM mode combination.
- Tests cover BFLOAT16, FLOAT32, and FLOAT64 seed/zero/one bit/type behavior where applicable.
- Mixed-floating binary/WHERE, cross-floating cast, SumToShape-as-forward-SUM, masked SUM, and
  representative 0004A/0004B/non-differentiable families fail or stop exactly as specified.
- Repeated operands receive repeated contributions in input-position order.
- Shared subexpressions and multiple consumers accumulate in deterministic reverse traversal and
  left-associated `Tensor.add` order.
- A target equal to the objective receives the implicit unit seed.
- An intermediate target does not stop propagation needed by an upstream target.
- Distinct targets can return the exact same gradient Tensor without an identity node.
- Rule owners are exactly the three named `*GradientRules` components and use only current public
  Tensor methods.

### Constants, capture, and result

- Every derivative zero/one base is storage-free, non-gradient, provenance-free, explicitly
  registered as an exact logical splat, and never inferred.
- Forward ingress order and generated constant order are stable, and all fixed sources are absent
  from `bindableInputs()`.
- One combined capture for a backward-capable request assigns graph-local IDs once, preserves
  every producer output slot, classifies original producers FORWARD and generated producers
  BACKWARD by exact identity, and retains no Tensor/producer reference in the immutable final
  result.
- Forward outputs preserve request order. Gradient graph outputs use stable first-occurrence
  de-duplication, while every target retains its own ordered role.
- Multiple targets may map to one final `ValueId`, and no manufactured identity/copy node appears.
- `GraphCompilation` retains exact mode, final `ValidatedGraph`, forward boundary,
  target-`TensorId`/gradient-`ValueId` roles, constants, bindable inputs, constraints, and phase
  facts through its components.
- `GraphCompilation` remains package-private, mode-neutral graph-stage state and is neither named
  nor shaped as the later `CompileArtifacts` aggregate.
- `FORWARD_ONLY` creates no seed, gradient constant, gradient role, or BACKWARD node, and its
  `GraphCompilation.gradientResults()` is empty.
- `FORWARD_AND_BACKWARD` and `TRAINING_STEP` produce equal graph semantics for equal requests; the
  latter contains no optimizer work.

### Combined optimization

- Disabled optional optimization still canonicalizes and validates the graph selected by the mode
  and remaps every applicable result role correctly.
- Enabled optimization uses exactly rewrite, fold, whole-graph DCE, phase-local CSE, whole-graph
  DCE once each in the specified order.
- Existing exact rewrite and fold matrices are unchanged and may consider either phase only under
  all existing non-phase guards.
- Whole-graph DCE removes an unused BACKWARD occurrence and keeps every dependency of a live
  forward or gradient output, including multi-output producers and constant roles.
- CSE merges equal eligible occurrences within FORWARD and within BACKWARD, never across phases,
  and never merges graph-output producers.
- Every changed candidate is revalidated through Compiler 0002; unchanged candidates are not.
- Graph output order, role ordinals, exact constants, constraints, operations, descriptors,
  phases, and immutable snapshots survive all rebuilds.
- No new algebra, folding row, fixed point, approximate math, storage read, or physical
  materialization is present.

### Boundaries, documentation, and completion

- No public declaration is added to `modules/compiler`; reflection/source tests lock the intended
  package-private surface.
- `GraphCompiler` exposes only the specified direct package-private parameter list, returns
  `GraphCompilation`, and adds no request aggregate, alternate entry/result owner, or alias with
  the rejected combined-prefixed names.
- No model, config, planning, trace, runtime, prepare, backend, engine, Gradle, architecture,
  conformance, or integration source/test changes occur.
- The six obsolete prototype paths were confirmed present and untracked before implementation,
  deleted before any implementation Java edit or compiler invocation, and are absent from the
  source tree at final status.
- No prototype content was copied, adapted, renamed, moved, staged, cited as design evidence, or
  reintroduced under another path.
- All affected production Javadocs fully document ownership, inputs, results, failure modes,
  identity, ordering, immutability, constants, phases, and no-side-effect boundaries.
- Compile API, Tensor API, Training API, autograd strategy, user guide, glossary, task, compiler
  master plan, and roadmap consistently distinguish current implementation from planned public
  publication, policy extensions, artifacts, and higher derivatives.
- Documentation explicitly records reviewed no-change conclusions for `ARCHITECTURE.md`, focused
  architecture documents, ADRs, model/config APIs, dependency/architecture tests,
  backend-conformance, integration tests, Gradle, and other modules.
- A separate clean-context documentation-focused agent pass finalizes affected Javadocs,
  explanatory documentation, glossary, planning status/evidence, terminology, links, examples,
  and no-change conclusions in the same overall change without repeating successful Java tests
  unless it changes executable behavior or records a concrete reason.

## Tests / validation

Before the first implementation edit, record:

```bash
git status --short -- \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradExpansion.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradGraph.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BackwardGraphBuilder.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseAdjoints.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/IndexingRandomAdjoints.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionLayoutAdjoints.java
```

The recorded output must contain exactly six `??` entries for those exact paths. Delete them
without reading or migrating their contents. Before the first compiler invocation and again at
final status, verify:

```bash
for path in \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradExpansion.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradGraph.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BackwardGraphBuilder.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseAdjoints.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/IndexingRandomAdjoints.java \
  modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionLayoutAdjoints.java
do
  test ! -e "$path"
done
```

Implementation-focused validation:

```bash
./gradlew :modules:compiler:test
git diff --check
```

The final compiler suite must include focused evidence for:

- zero-ID side effects across every preflight rejection class;
- deep iterative inventory, deep reverse traversal, and deep backward-capable combined capture;
- complete supported/deferred/non-differentiable matrix locks;
- repeated operands, shared consumers, intermediate and duplicate-gradient targets;
- exact hidden-output inventory and producer identity classification;
- generated splat bits/types, source roles, and no storage inference;
- all three modes and mode/request mismatch ordering;
- exact `GraphCompiler`/`GraphCompilation` package-private signatures and absence of a request
  aggregate or alternate entry/result type;
- per-node phases and graph-output/role de-duplication;
- inference/constraint regeneration after backward-capable combined capture and every changed
  pass;
- rewrite/fold phase eligibility without matrix widening;
- whole-graph DCE and phase-local CSE; and
- immutability, deterministic IDs/order, no mutation, and no public compiler declarations.

Documentation-focused pass:

```bash
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

If the repository Markdown validator is not present, recreate or use the established equivalent
that checks every local Markdown link and anchor, balanced fences, final newlines, and trailing
whitespace, and record the exact command.

The documentation pass also inspects generated compiler Javadoc, source/API terminology, exact
task/master/roadmap statuses, the 38 touched-path ceiling (32 tracked create/modify plus six
removal-only), the initial cleanup evidence, final prototype absence, and the no-change
boundaries.

Repository-wide validation is deferred to the compiler transformation/autograd capability
checkpoint after Compiler 0004B and to CI. This task changes one module and no dependency,
architecture, build, or public cross-module contract.

## Dependencies

- Model 0025 — Complete; supplies exact canonical Tensor wrappers for every producer output.
- Compiler 0001 — Complete; supplies deterministic Tensor-expression graph capture.
- Compiler 0002 — Complete; supplies authoritative captured-graph inference and validation.
- Compiler 0003 — Complete; supplies canonicalization and bounded optimization orchestration.
- Compiler 0003A — Complete; supplies the closed seven-rule exact rewriting matrix.
- Compiler 0003B — Complete; supplies explicit logical-splat ingress, exact bounded folding, and
  source-role sidecars.
- Config 0002 — Complete; supplies `CompileMode` and `GraphOptimizationConfig`.
- ADR 0009 — Accepted; selects compiler-owned pre-capture Tensor-expression autograd.
- The current public Tensor operations named in `SUPPORTED_0004` — Complete and source-verified.

No unresolved dependency blocks this task.

## Follow-up tasks

- Compiler 0004A — add separately specified policy-free exact-composition gradient rules without
  changing the one-capture lifecycle.
- Compiler 0004B — select explicit tie, discontinuity, singularity, exceptional-value, empty
  domain, and cross-floating conversion policies before adding policy-dependent rules.
- Compiler 0005 — add publication, planning orchestration, and immutable public/cross-package
  compile artifacts while transporting the exact mode-selected graph-stage result, applicable
  gradient roles, constraints, and constant facts.
- Compiler 0006 — define public functional objective/target/seed requests and higher-order
  derivative lifecycle/order representation.

Do not create detailed specifications for these tasks until the frontier advances.

## Architecture impact

Expected impact: None.

The authoritative architecture, focused architecture documents, and accepted ADR 0009 already
specify this ownership and lifecycle. Dependencies and graph phase vocabulary do not change.

If implementation requires architecture, ADR, dependency, model graph, phase, public API, or
cross-module changes, stop and report the issue.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository.

Read:
- AGENTS.md
- ARCHITECTURE.md
- docs/planning/planning-guide.md
- docs/planning/roadmap.md
- docs/planning/modules/compiler/master-plan.md
- docs/planning/modules/compiler/tasks/0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md

Implement Compiler task 0004 exactly as specified. Before any implementation Java edit, verify the
six exact obsolete compiler-source prototype paths are present and untracked, then delete them as
the first cleanup step. Do not read, copy, adapt, rename, move, stage, preserve, or treat their
contents as design authority, and verify their absence before compiling or testing. Do not
implement out-of-scope rules or public APIs. Use the exact planned package-private names
`GraphCompiler`, `GraphCompilation`, and `GraphCompilerTest`; retain the specified compile
parameter list and do not add a request aggregate or another entry/result type. Keep
`GraphCompilation` distinct from later `CompileArtifacts`. Stop and report any prototype-status,
architecture, policy, or maximum-scope conflict. Do not commit or push.

After executable code and the compiler module test pass are stable, hand the same working tree and
recorded test evidence to a separate clean-context documentation-focused agent. That pass must
follow docs/developer-guide/documentation-rules.md, finalize all affected Javadocs, API/design/user
documentation, glossary, planning status/evidence, and no-change conclusions in this same overall
change, and run compiler Javadoc plus documentation validation. It must not repeat the successful
Java suite unless it changes executable behavior or records a concrete reason.

At the end, update this task with local decisions, limitations, validation evidence,
implementation notes, completion summary, and final status. Mark Complete only after every
acceptance criterion and the documentation pass succeed.
```

## Local decisions

- Keep the general entry and graph-stage result package-private as the specified `GraphCompiler`
  and `GraphCompilation`; retain the direct compile parameter list rather than adding a request
  aggregate, facade, registry, or alternate result type.
- Separate a Tensor-allocation-free inventory/preflight plan from formula expansion. Preflight
  uses the closed `SUPPORTED_0004` matrix, and successful expansion uses only ordinary public
  Tensor operations plus explicitly registered derivative splats.
- Preserve deterministic identity semantics with compile-local identity maps and ordered
  contribution lists. Accumulate repeated and shared contributions with ordinary `Tensor.add`.
- Capture requested forward outputs and gradient roots in one phase-aware traversal. Preserve
  every requested target role while de-duplicating only the combined graph output boundary by
  stable first occurrence.
- Run inference/validation, canonicalization, exact rewrite/fold, whole-graph DCE, phase-local
  CSE, and final DCE as one bounded mode-selected graph pipeline. Revalidate each changed
  candidate.

## Known limitations

- The supported derivative matrix is exactly `SUPPORTED_0004`; all other operations, output
  roles, attribute variants, mixed-floating cotangent conversions, and unselected derivative
  policies fail closed.
- Requests are limited to one requested scalar floating objective, ordered identity-unique
  targets, one implicit exact typed rank-zero unit seed, and first-order derivatives.
- `TRAINING_STEP` currently constructs the same combined graph as `FORWARD_AND_BACKWARD`; it adds
  no optimizer update, training state, prepare behavior, backend work, or runtime execution.
- Public compile requests, publication/planning orchestration, `CompileArtifacts`, explicit seeds,
  higher derivatives, and policy-dependent formulas remain with Compiler 0004A, 0004B, 0005, and
  0006 as specified.

## Validation evidence

- Before any implementation Java read or edit, the implementation context ran the exact
  six-path `git status --short -- ...` command and observed exactly six `??` entries. It deleted
  those obsolete prototypes without reading, copying, adapting, moving, or staging their
  contents. The exact absence loop passed before the first Gradle invocation and again during the
  final documentation audit.
- The focused ten-class compiler suite passed. The final
  `./gradlew :modules:compiler:test` run passed 118 tests with zero failures, errors, or skips;
  Gradle reported 13 actionable tasks, one executed and 12 up-to-date.
- The separate clean-context documentation pass
  `/root/implement_compiler_0004_precapture/compiler_0004_docs` changed no executable Java
  tokens and deliberately did not repeat the successful Java suite. Comment-stripped hashes for
  all 13 authorized production Java paths matched the hashes recorded before the Javadoc pass.
- `./gradlew :modules:compiler:javadoc` passed, and the generated compiler Javadoc was inspected.
- `python3 /tmp/validate_synaptik_markdown.py` passed every repository-local Markdown link and
  anchor, balanced fence, final newline, and trailing-whitespace check.
- `git diff --check` passed. The final path audit found exactly 32 tracked create/modify paths
  plus the six required removal-only prototype paths, for the specified 38 touched-path ceiling,
  with no out-of-scope path.
- Repository-wide validation remains deferred to the Compiler 0004B transformation/autograd
  capability checkpoint and CI, as specified by the validation tiers.

## Implementation notes

- Added seven focused package-private production types for preflight, the three named rule
  families, formula expansion, compilation ownership, and the immutable graph-stage result.
- Extended phase-aware capture and the existing exact optimization components so one
  backward-capable request is captured and optimized as a complete combined graph. DCE is
  whole-graph; CSE remains phase-local.
- Added or extended ten compiler test classes covering the closed matrix, preflight side-effect
  boundary, deep iterative traversal, modes, identities, constants, phases, output/role
  de-duplication, and combined optimization.
- Finalized Javadocs for all 13 affected production paths and synchronized the Compile, Tensor,
  and Training API documents, autograd design note and user guide, glossary, task, compiler master
  plan, and roadmap.
- Reviewed `ARCHITECTURE.md`, focused architecture documents, ADR 0009, model/config APIs,
  architecture and dependency tests, backend conformance, integration tests, Gradle
  configuration, and other modules. No change was needed: the implementation realizes the
  already accepted ownership and lifecycle, changes no dependency or public cross-module
  contract, and adds no backend/runtime/execution behavior.

## Completion summary

Completed compiler-owned bounded first-order pre-capture autograd and one phase-aware combined
graph compilation path. The change preserves the exact package-private boundary, closed derivative
matrix, exact typed splats, deterministic identity/order behavior, and one-shot exact optimization
contract. All specified source, test, Javadoc, API, design, user, glossary, and planning work is
complete, and all required implementation and documentation validation passed.

Status: Complete
