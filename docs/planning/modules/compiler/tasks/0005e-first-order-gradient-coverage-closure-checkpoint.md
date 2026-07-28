# Task 0005E: First-order Gradient Coverage Closure Checkpoint

## Status

Complete

## Goal

Close the current first-order differentiation milestone with one source-backed, fail-closed
coverage checkpoint. Prove that every production Model `OperationKind` constant, accepted
`OperationSignature` variant, legal output slot, and ordered Tensor input role has exactly one
current disposition:

- `D` — an implemented first-order contribution for the selected occurrence and role;
- `ND` — intentionally non-differentiable, with no cotangent route; or
- `FC` — deliberately fail-closed, with one exact structural or expressibility reason.

Eliminate the concrete drift risk between preflight role selection, occurrence validation, and
formula-family dispatch with one small package-private Compiler coverage checker. Verify the
checker against production Model classes discovered from the compiled source output, not only
against a hand-maintained family list. Exercise every current role through preflight and formula
dispatch, verify every operation emitted by the formulas, and run a bounded nested first-order
probe wherever the generated first gradient remains connected to the original target.

This task is a closure audit and checkpoint, not another formula-family task. Current source has
37 production kind enum families, 107 constants, and 128 signature variants. The historical
127-signature Model closure inventory predates Model 0025D's
`SLICE_UPDATE/CropToShapeAttrs` variant; Compiler 0005C already implements both of that variant's
data roles. No missing representable first-order formula was found during planning.

The task adds no public gradient request, derivative-order or `createGraph` option, disconnected-
result policy, Tensor gradient state, second algebra, backward operation kind, runtime tape,
backend behavior, numerical execution, or task 0006 behavior.

## Scope

### Closure decision and bounded mechanism

Add package-private `FirstOrderGradientCoverage` in the existing Compiler root package. It is a
closed current-capability checker, not a public registry or extensibility surface.

The checker must:

- classify the semantic row for an exact producer occurrence, output slot, and input position as
  conditional `D`, `ND`, or direct `FC`;
- retain a deterministic nonblank reason for every `ND` and `FC` result;
- map each `D` occurrence to exactly one existing family-rule owner used by
  `FirstOrderAutograd`;
- return `FC` for an unknown `OperationKind`, unknown kind/attributes pairing, illegal
  cardinality or slot, and a newly discovered but unclassified production kind;
- include the role facts needed to select the existing Compiler 0005A–0005D row, including data
  type, output count, reduction/scatter form, and canonical auxiliary availability;
- remain pure, stateless, immutable, package-private, and free of Tensor allocation, capture,
  graph IDs, reflection, storage reads, bindings, execution, and backend state; and
- add no per-element or runtime hot-path work. It is used only during compile-time preflight and
  package-private dispatch.

`AutogradPreflight` must delegate semantic role selection to the checker and retain its existing
exact occurrence/prerequisite validation and failure order. A conditional `D` becomes an accepted
`D` only after that validation proves every exact Shape, DataType, cardinality, normalization, and
formula-construction prerequisite; a failed prerequisite is the final `FC` result with the
existing deterministic reason. Do not move or duplicate the large typed validation matrix merely
to make the checker appear self-contained. `FirstOrderAutograd` must dispatch successful selected
occurrences through the same checker-owned family-owner result. Family formula classes remain
unchanged unless the closure tests expose a real contradiction; a new formula is outside this
task and requires stopping for clarification.

The checker does not replace Model family-owned `signatures()`, `Operation.signature()`,
`CapturedGraphInference`, typed constraints, or the family rule classes. It does not become a
general operation registry, gradient-rule registry, service loader, plugin boundary, public
facade, or serialized capability format.

### Source-backed baseline and discovery

At implementation time, discover the production inventory from the Model module's compiled main
output:

1. locate the code-source root that contains `OperationKind`;
2. walk top-level production classes under `io.github.pho001.synaptik.model.operation`;
3. load classes without initializing unrelated types;
4. retain every concrete enum implementing `OperationKind`;
5. enumerate every enum constant and each immutable `signatures()` entry; and
6. compare the discovered exact set with the coverage checker's current rows.

This narrow reflection/classpath scan belongs only in `FirstOrderGradientCoverageTest`. It is
justified by the open `OperationKind` interface and the demonstrated 127-to-128 signature drift.
Production code must not scan classes or use reflection. The test must fail with an exact missing
or extra family, constant, attributes type, cardinality, output slot, or role rather than merely
asserting baseline counts.

Also record the source command:

```text
rg -l 'implements OperationKind' modules/model/src/main/java | sort
```

The expected current checkpoint is 37 enum families, 107 constants, and 128 signatures. Counts
are corroborating evidence only; equality of the complete discovered rows is the invariant.

### Exact signature, output, and role inventory

Notation is `Attrs inputs -> outputs`. `D` applies only when the exact current data-type and
occurrence guards pass. `ND` includes false-gradient BOOL, index, state, mask, and non-floating
data roles. `FC` is a selected structurally gradient-eligible route that preflight must reject
before allocation.

| Family and exact signature variants | Signature count | Output and ordered-input disposition |
|---|---:|---|
| `ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION`: `ScaledDotProductAttentionAttrs 3..4 -> 1..2` | 1 | One-output slot 0: query/key/value `FC` because canonical weights do not exist; optional mask `ND`. Two-output slot 0: query/key/value `D`, mask `ND`. Slot 1 weights: query/key `D`, value/mask `ND`. |
| `Conv2dKind.CONV2D`: `Conv2dAttrs 2..3 -> 1` | 1 | Slot 0 input/weight/optional bias are floating `D`; invalid or unproved group/window/channel relations are `FC`. |
| `BinaryArithmeticKind`: seven `NoOperationAttrs 2 -> 1` variants | 7 | Slot 0 floating left/right `D`; non-floating roles `ND`. |
| `CastKind.CAST`: `CastAttrs 1 -> 1` | 1 | Slot 0 input is `D` only for floating-to-floating conversion; every other conversion is `ND`. |
| `FloatingClassificationKind`: three `NoOperationAttrs 1 -> 1` variants | 3 | BOOL slot 0 and its numeric input route are `ND`. |
| `BinaryComparisonKind`: six `NoOperationAttrs 2 -> 1` variants | 6 | BOOL slot 0 and both ordered input routes are `ND`. |
| `BooleanLogicalKind`: `AND/OR NoOperationAttrs 2 -> 1`; `NOT NoOperationAttrs 1 -> 1` | 3 | BOOL output and every BOOL input route are `ND`. |
| `ScalarElementwiseKind`: seven `ScalarValueAttrs 1 -> 1`; `CLAMP ClampRangeAttrs 1 -> 1` | 8 | Floating receiver is `D`; non-floating receiver is `ND`. Scalar values/bounds are attrs, never Tensor roles. |
| `WhereSelectionKind.WHERE`: `NoOperationAttrs 3 -> 1` | 1 | Condition is `ND`; floating true/false branches are `D`; non-floating branches are `ND`. |
| `UnaryElementwiseKind`: nineteen `NoOperationAttrs 1 -> 1` variants | 19 | Floating input is `D`, including direct-zero FLOOR/CEIL/SIGN conventions; non-floating input is `ND`. |
| `AxisGatherKind.GATHER/GATHER_ELEMENTS`: `IndexAxisAttrs 2 -> 1` | 2 | Floating data is `D`; index is `ND`. |
| `AxisScatterKind.SCATTER_ELEMENTS`: `ScatterElementsAttrs 3 -> 1`; `SCATTER_ADD`: `IndexAxisAttrs 3 -> 1` | 2 | Floating base and updates are `D` for every adopted replacement/add/mul/min/max policy; indices are `ND`; unproved duplicate/Shape policy is `FC`. |
| `GatherNdKind.GATHER_ND`: `GatherNdAttrs 2 -> 1` | 1 | Floating data is `D`; indices are `ND`. |
| `OneHotKind.ONE_HOT`: `OneHotAttrs 1 -> 1` | 1 | BOOL output and integral indices are `ND`. |
| `ScatterNdKind.SCATTER_ND`: `ScatterNdAttrs 3 -> 1` | 1 | Floating base and updates are `D` for every adopted replacement/add/mul/min/max policy; indices are `ND`; unproved occurrence policy is `FC`. |
| `SelectKind.SELECT`: `SelectAttrs 1 -> 1` | 1 | Floating source is `D`; non-floating source is `ND`; axis/index attrs are configuration. |
| `AxisTransformKind`: `PERMUTE PermutationAttrs 1 -> 1`; `EXPAND_DIMS/SQUEEZE AxisTransformAttrs 1 -> 1` | 3 | Floating source is `D`; non-floating source is `ND`; axes/permutation are configuration. |
| `ContiguousKind.CONTIGUOUS`: `NoOperationAttrs 1 -> 1` | 1 | Floating source is `D`; non-floating source is `ND`. |
| `PadKind.PAD`: `PadAttrs 1 -> 1` | 1 | Floating source is `D`; non-floating source is `ND`; widths/value attrs are configuration. |
| `ShapeTransformKind.RESHAPE/EXPAND`: `TargetShapeAttrs 1 -> 1` | 2 | Floating source is `D`; non-floating source is `ND`; binding-dependent EXPAND without the retained inverse proof is `FC`. |
| `SliceKind.SLICE`: `SliceAttrs 1 -> 1`, `CropToShapeAttrs 1 -> 1`; `SLICE_UPDATE`: both attrs forms `2 -> 1` | 4 | Floating extraction source is `D`; floating update base/update are `D`; non-floating data roles are `ND`; unproved bounds/placement are `FC`. |
| `TensorCompositionKind.CONCAT/STACK`: `CompositionAxisAttrs 1..N -> 1` | 2 | Every floating variadic input position is `D`; non-floating positions are `ND`; axis is configuration. |
| `TileKind.TILE`: `TileAttrs 1 -> 1` | 1 | Floating source is `D`; non-floating source is `ND`; repeats are configuration. |
| `WindowTransformKind`: `UNFOLD_AXIS UnfoldAxisAttrs 1 -> 1`; `FOLD_AXIS FoldAxisAttrs 1 -> 1`; `UNFOLD2D Window2dAttrs/Unfold2dAttrs 1 -> 1`; `FOLD2D Fold2dAttrs 1 -> 1` | 5 | Floating source is `D`; non-floating source is `ND`; unproved dynamic/window placement is `FC`. |
| `MatmulKind.MATMUL`: `NoOperationAttrs 2 -> 1` | 1 | Floating left/right are `D`; signed-integral left/right are `ND`; unproved rank/broadcast restoration is `FC`. |
| `LossKind`: three exact two-input/one-output attrs variants | 3 | MSE prediction/target `D`; dense categorical logits/target `D`; index categorical logits `D`, integral target `ND`. Dynamic or zero class depth for index categorical logits is `FC`. |
| `BatchNormKind.BATCH_NORM_INFERENCE`: `BatchNormInferenceAttrs 5 -> 1`; `BATCH_NORM_TRAINING`: `BatchNormTrainingAttrs 5 -> 5` | 2 | Inference slot 0: input/scale/bias/running mean/running variance `D`. Training slot 0: input/scale/bias `D`, running statistics `ND`; slot 1: input/running mean `D`, other roles `ND`; slot 2: input/running variance `D`, other roles `ND`; slots 3/4 are canonical auxiliaries and `FC` as independent cotangent roots. |
| `LayerNormKind.LAYER_NORM`: `LayerNormAttrs 1 -> 1`, `AffineLayerNormAttrs 3 -> 1` | 2 | Floating input and present scale/bias are `D`; configuration attrs are not Tensor roles. |
| `RmsNormKind.RMS_NORM`: `RmsNormAttrs 1..2 -> 1` | 1 | Floating input and optional scale are `D`; epsilon/axes are configuration. |
| `SoftmaxKind.SOFTMAX/LOG_SOFTMAX`: `SoftmaxAttrs 1 -> 1` | 2 | Floating input is `D`; axis is configuration. |
| `OrderingKind.SORT/ARGSORT`: `SortAttrs 1 -> 1` | 2 | Floating SORT source is `D`; non-floating SORT and every ARGSORT route are `ND`. |
| `TopKKind.TOP_K`: `TopKAttrs 1 -> 2` | 1 | Slot 0 values routes floating input as `D`; slot 1 indices and non-floating input routes are `ND`. Canonical slot-1 indices remain a same-occurrence auxiliary. |
| `Pool2dKind.MAX_POOL2D/AVERAGE_POOL2D`: exact attrs `1 -> 1` | 2 | Floating input is `D`; unproved window/output geometry is `FC`. |
| `DropoutKind.DROPOUT`: `DropoutAttrs 2 -> 3` | 1 | Slot 0 routes floating input as `D` and state as `ND`; slot 1 mask and slot 2 next state are `ND`. Canonical mask is an auxiliary used by slot-0 formula construction. |
| `GraphRngKind.INITIAL_STATE`: `GraphRngStateAttrs 0 -> 1` | 1 | Opaque state output has no Tensor cotangent route and is `ND`; there are no inputs. |
| `AggregateReductionKind`: SUM accepts `NoOperationAttrs 1 -> 1`, `AxisReductionAttrs 1 -> 1`, `MultiAxisReductionAttrs 1 -> 1`, `MaskedReductionAttrs 2 -> 1`, and `SumToShapeAttrs 1 -> 1`; MEAN accepts the first four; PROD/MIN/MAX/ALL/ANY accept the first three ordinary variants; ARG_MIN/ARG_MAX accept `ArgExtremaAttrs 1 -> 1`; LOG_SUM_EXP/L1_NORM/L2_NORM accept `MultiAxisReductionAttrs 1 -> 1`; VARIANCE/STANDARD_DEVIATION accept `StatisticalReductionAttrs 1 -> 1` | 31 | Floating SUM/MEAN/PROD/MIN/MAX/LOG_SUM_EXP/VARIANCE/STANDARD_DEVIATION/L1_NORM/L2_NORM data is `D`; masked SUM/MEAN mask is `ND`; non-floating numeric routes and every ALL/ANY/ARG_MIN/ARG_MAX route are `ND`; missing inverse/binding proof is `FC`. |
| `CumulativeScanKind.CUM_SUM/CUM_PROD`: `CumulativeScanAttrs 1 -> 1` | 2 | Floating source is `D`; integral source is `ND`; axis/traversal attrs are configuration. |
| **Total** | **128** | Every legal output/input role must match exactly one row. |

The three exact loss attributes are `MeanSquaredErrorAttrs`,
`DenseCategoricalCrossEntropyWithLogitsAttrs`, and
`IndexCategoricalCrossEntropyWithLogitsAttrs`. The two exact pool attributes are
`MaxPool2dAttrs` and `AveragePool2dAttrs`.

The inventory must also test minimum and maximum legal cardinality where a signature is ranged:
three/four attention inputs, one/two attention outputs, two/three convolution inputs, one/two RMS
inputs, one and multiple composition inputs, and every fixed multi-output slot. It must not try to
materialize `Integer.MAX_VALUE` composition operands; the checker represents the same role pattern
for every legal variadic position.

### Existing policy and pipeline closure

For every `D`, `ND`, and `FC` row, verify the relevant current 0005A–0005D contract without
choosing a new derivative policy:

- extrema ties, clamp endpoints, discontinuities, NaN/infinity/signed-zero and raw-domain
  decisions from 0005A;
- product zero-count, extrema, scan, softmax/log-softmax, statistics, norm and normalization
  decisions from 0005B;
- slice/window placement, scatter duplicate/reduction, ordering tie, top-K, and dropout decisions
  from 0005C; and
- attention, grouped convolution, pooling winner/divisor, and loss/ignore/reduction decisions from
  0005D.

Reprove these cross-cutting invariants:

- exact Shape and DataType cotangent normalization uses ordinary `sumToShape` then `cast` when
  required;
- contributions accumulate by exact Tensor identity in deterministic selected-output-slot and
  input-position order;
- repeated operand positions, shared producers, distinct wrappers, and multiple target roles do
  not collapse by equality, label, descriptor, or graph ID;
- two targets may retain distinct ordered roles while sharing one generated Tensor and later one
  captured `ValueId`;
- canonical attention weights, batch statistics, top-K indices, dropout masks, and original
  maximum-pool outputs come from the exact original producer/output index;
- generated logical splats retain exact type/bits, request-local first-use order, sidecar ingress,
  and unreachable-constant pruning;
- preflight completes for the whole selected ancestry before the seed, a splat, matching ARGSORT,
  formula occurrence, capture, `NodeId`, or `ValueId` is created;
- unknown/new/malformed/unsupported rows consume no Tensor identifier and produce no partial
  formula or sidecar state;
- one phase-aware combined capture still assigns graph-local IDs once;
- every original producer is `FORWARD`, every generated producer is `BACKWARD`, and canonical
  auxiliaries retain original-producer phase;
- exact mandatory canonicalization and bounded rewrite/fold/DCE/phase-local-CSE/DCE cleanup remain
  unchanged and revalidate changed candidates; and
- forward outputs, gradient roles, publication roles, constants, and diagnostics remain stable
  through the existing Compiler 0005 artifact path.

### Transitive formula-operation closure

Build representative valid scalar objectives for every `D` row. For non-scalar selected outputs,
use an ordinary public SUM solely to create the current scalar-objective request. Expand the first
gradient through the existing package-private preflight and `FirstOrderAutograd` path.

For every generated gradient root:

1. walk exact reachable Tensor/producer identities;
2. inventory every generated operation, attributes variant, output slot, and ordered input edge;
3. classify every edge through `FirstOrderGradientCoverage`;
4. fail if any emitted occurrence or edge is unknown, ambiguously classified, or `FC`;
5. confirm that every edge which carries a cotangent toward the original target is `D`;
6. confirm every comparison, logical, classification, index, mask, state, one-hot, arg-extrema,
   or selection-condition edge used only for a fixed policy is explicitly `ND`; and
7. compare the discovered emitted set with the exact allowed shared-algebra boundary accumulated
   by Compiler 0004–0005D.

Then form `firstGradient.sum()` and run a second package-private preflight/expansion against the
same original target whenever exact identity ancestry proves that target remains connected. This
is a nested use of the existing first-order primitive for checkpoint evidence only. It adds no
public higher-order request, `createGraph`, derivative-order field, graph-phase order, or
disconnected-result behavior.

When a first gradient is structurally constant or disconnected from the target, record the exact
formula/role as `SECOND_PASS_NOT_APPLICABLE` and prove by identity ancestry that no target path
exists. Linear rules, direct-zero conventions, and constant first derivatives are not failures;
task 0006 owns the public disconnected-result and derivative-order policy needed to return their
higher derivatives.

### Documentation impact

After executable Java and final Java test evidence stabilize, a separate documentation-focused
clean context must:

- finalize meaningful Javadocs for `FirstOrderGradientCoverage`,
  `AutogradPreflight`, and `FirstOrderAutograd`;
- update the Compile API from family-complete first-order support to a source-backed closed
  37-family/107-kind/128-signature checkpoint;
- update the glossary's pre-capture autograd/first-order request explanation only where the
  internal coverage closure changes current status;
- keep the Tensor API unchanged unless a direct contradiction is found, because no public Tensor
  method, kind, signature, result carrier, or forward behavior changes;
- keep the Training API unchanged because no optimizer, training request, session, publication,
  preparation, or execution contract changes;
- record reasoned no-change conclusions for Public/Runtime APIs, model capabilities/master/tasks,
  architecture/ADR/tests, backend conformance, integration, Gradle, and unrelated modules; and
- synchronize this task, the Compiler master plan, and the roadmap only after all evidence passes.

Do not add a glossary entry for a task name, test mechanism, or package-private checker. Reuse
existing terms and update current-status prose only.

## Out of scope

- a new or changed first-order derivative formula, subgradient, exceptional-value policy, or
  representable role
- changes under `modules/model`, including `OperationKind`, `OperationSignature`, Tensor methods,
  producer outputs, attributes, signatures, or result carriers
- a public or reusable gradient registry, dispatch SPI, service loader, plugin mechanism, facade,
  request, result, or serialization format
- production reflection/classpath scanning or global mutable coverage state
- public explicit objectives, targets, seeds, vector-Jacobian products, disconnected-result
  behavior, `createGraph`, derivative order, phase order, or higher-order API behavior
- Tensor gradient/backward lifecycle state, mutable gradient storage, optimizer state, tape,
  physical saved values, buffers, schedules, or execution units
- new operation kinds, hidden outputs, backward-specific kinds, or a second algebra
- architecture, dependency, module, Gradle, config, planning, prepare, trace, runtime, backend,
  engine, training, conformance, integration, or numerical execution changes
- unrelated refactors, performance work, formula rewrites, or public documentation examples
- creating a Compiler 0006 task specification or advancing 0006 from Draft

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Overview](../../../../architecture/overview.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Model master plan](../../model/master-plan.md)
- [Model capabilities](../../model/capabilities.md)
- [Model capability closure audit](../../model/model-capability-contract-closure-audit.md)
- [Model adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)
- [Model 0023 adjoint audit task](../../model/tasks/0023-adjoint-expressibility-audit.md)
- [Model 0025 canonical outputs](../../model/tasks/0025-canonical-tensor-producer-outputs.md)
- [Model 0025A floating policies](../../model/tasks/0025a-portable-floating-comparison-extrema-and-clamp-semantics.md)
- [Model 0025B binding-aware expansion](../../model/tasks/0025b-binding-aware-expansion.md)
- [Model 0025C scatter policies](../../model/tasks/0025c-portable-functional-scatter-reduction-semantics.md)
- [Model 0025D dynamic slice placement](../../model/tasks/0025d-dynamic-extent-slice-extraction-and-symbolic-slice-placement.md)
- [Compiler 0001](0001-tensor-expression-graph-capture.md)
- [Compiler 0002](0002-captured-graph-inference-and-validation.md)
- [Compiler 0003](0003-canonicalization-and-forward-optimization.md)
- [Compiler 0003A](0003a-exact-arithmetic-rewriting.md)
- [Compiler 0003B](0003b-compile-time-constants-and-constant-folding.md)
- [Compiler 0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Compiler 0004A](0004a-exact-composition-gradient-rule-extensions.md)
- [Compiler 0004B](0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md)
- [Compiler 0005](0005-publication-planning-orchestration-and-compile-artifacts.md)
- [Compiler 0005A](0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
- [Compiler 0005B](0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [Compiler 0005C](0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
- [Compiler 0005D](0005d-attention-convolution-pooling-and-loss-gradient-completion.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative.
- Compiler owns preflight, derivative policy, formula dispatch, reverse accumulation, and combined
  capture; Model retains only backend-independent forward semantics and family-owned signatures.
- Forward and generated expressions use the same public Tensor algebra, inference, validation,
  numerical contract, and exact optimization pipeline.
- The coverage checker is package-private compile-time implementation, not a public registry,
  model contract, graph artifact, or extension boundary.
- Canonical auxiliaries are exact original producer outputs, never physical saved buffers or a
  runtime tape.
- Tensor identity maps remain request-local implementation state and never become graph or
  artifact representation.
- One combined phase-aware capture assigns graph-local IDs once.
- No dependency direction, module boundary, public API, lifecycle, backend, or execution contract
  changes.
- If the audit finds a missing formula, conflicting derivative policy, required Model change,
  architecture change, or need for task 0006 semantics, stop and request clarification.

## Package impact

Existing package used and changed:

- `io.github.pho001.synaptik.compiler` — retains the cohesive package-private compiler front-end,
  preflight, family dispatch, formula owners, accumulation, capture, and checkpoint test seam.

No package is added.

Type placement:

- `io.github.pho001.synaptik.compiler.FirstOrderGradientCoverage` — package-private current
  first-order role/family checker colocated with its only production consumers,
  `AutogradPreflight` and `FirstOrderAutograd`.

The discovery helper remains private test code inside `FirstOrderGradientCoverageTest`; do not add
a production scanner or reusable test utility package.

## Affected files

Production:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverage.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`

Tests:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverageTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`

Documentation/planning:

- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/compiler/tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless a contradiction requires stopping: `ARCHITECTURE.md`, focused architecture and
ADR 0009; Tensor/Public/Training/Runtime APIs; model capabilities/master/audits/tasks and all
current Model operation source/tests; every completed Compiler task and family rule/test; compiler
package Javadoc; architecture/backend-conformance/integration tests; Gradle/build files; Config,
Planning, Prepare, Trace, Runtime, backends, Engine, training, and tools.

## Maximum scope

At most 12 paths: three production, four tests, and five documentation/planning paths listed
above.

The mechanism may centralize role classification and family-owner dispatch only. If a family
formula, another existing family-rule class/test, Model/API path, thirteenth path, another module,
architecture/dependency/build change, or different derivative decision is required, stop and
report the exact gap. Do not raise the ceiling during implementation to hide a formula or
architecture problem.

A separate documentation-focused clean context finalizes the authorized Javadocs,
documentation, glossary impact, planning evidence, and status without repeating successful Java
tests unless it changes executable Java behavior or records a concrete reason.

## Acceptance criteria

### Inventory and checker

- Production discovery finds exactly the current 37 enum families, 107 constants, and 128
  signature variants, including both `SLICE_UPDATE` attrs variants.
- The full discovered set, not only counts, equals the checker inventory.
- Every signature fingerprint includes exact kind identity, attrs class, input bounds, and output
  bounds.
- Every legal output slot and ordered input position has exactly one `D`, `ND`, or `FC`
  disposition and deterministic reason where required.
- Every `D` maps to exactly one existing family owner; no `ND` or `FC` maps to a formula owner.
- Unknown custom kinds, new/unclassified production kinds, malformed signatures, illegal slots,
  and unsupported prerequisites fail closed.
- Production code contains no reflection, classpath scan, public registry/facade, mutable global
  state, service loader, or new dependency.

### First-order and fail-closed behavior

- Representative BFLOAT16/FLOAT32/FLOAT64 and applicable integral/BOOL rows exercise every
  signature/output/input disposition through real preflight.
- Every current 0005A–0005D derivative policy and occurrence guard remains unchanged.
- All `FC` and unknown cases fail before any seed, constant, ARGSORT, formula Tensor, sidecar
  binding, capture, or graph ID allocation; exact Tensor ID deltas prove no allocation.
- Canonical auxiliary output indices and exact producer identities are preserved.
- Shape/type normalization, identity accumulation, repeated positions, shared producers, shared
  gradients, deterministic constants, one combined capture, phases, publication, and optimization
  remain exact.
- No formula-family source change, new formula, new kind, hidden output, second algebra, or public
  gradient behavior is present.

### Transitive closure and checkpoint

- Every actually emitted formula operation/attrs/output/input edge is classified; none is unknown
  or `FC`.
- Every target-carrying generated edge is `D`; every policy-only condition/index/mask/state edge is
  explicitly `ND`.
- Every connected first gradient passes the nested scalar-sum second preflight and expansion.
- Every skipped second pass has an identity-proven disconnected/constant reason recorded as
  `SECOND_PASS_NOT_APPLICABLE`.
- Tests do not claim public higher-order support or require disconnected-result/order semantics.
- Focused tests, the full Compiler module, Compiler Javadoc, and the repository/architecture
  capability checkpoint pass.

### Documentation and scope

- A separate documentation-focused clean context finalizes affected Javadocs, Compile API,
  glossary status, task evidence/summary, Compiler master plan, and roadmap.
- Tensor, Public, Training, Runtime, architecture/ADR/tests, model, conformance/integration,
  Gradle, and other-module no-change conclusions are recorded with reasons.
- Exactly the authorized paths change, no Compiler 0006 task file exists, and 0006 remains Draft.
- Links, heading anchors, fences, final newlines, trailing whitespace, status synchronization, and
  `git diff --check` pass.

## Tests / validation

Focused executable validation:

```text
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderAutogradTest \
  --tests io.github.pho001.synaptik.compiler.GraphCompilerTest
```

After executable Java stabilizes, run the affected module once:

```text
./gradlew :modules:compiler:test
```

This task is the recorded first-order capability checkpoint. Run:

```text
./gradlew test :testing:architecture-tests:test
```

The documentation-focused pass reuses successful Java evidence unless it changes executable Java.
After final Javadocs and documentation:

```text
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

If `/tmp/validate_synaptik_markdown.py` is absent, create an equivalent temporary validator
outside the repository. It must check every changed Markdown link target and heading anchor,
balanced fences, final newlines, and trailing whitespace.

Required source/manual evidence:

- `rg -l 'implements OperationKind' modules/model/src/main/java | sort`;
- discovered 37-family/107-kind/128-signature exact fingerprints and complete coverage-set
  equality;
- representative legal ranged cardinalities and every multi-output slot;
- complete `D`/`ND`/`FC` reason and family-owner uniqueness scans;
- no-ID-delta evidence for known `FC`, unknown kind, new/unclassified row, malformed signature,
  illegal slot, and failed prerequisite;
- emitted operation/attrs/output/input inventory, connected nested-pass results, and exact
  disconnected reasons;
- exact auxiliary producer/output-index, repeated-input, shared-producer, shared-gradient,
  constant-sidecar, one-capture, phase, publication, and optimization inspection;
- production bytecode/source surface proving the checker and consumers remain package-private and
  production contains no reflection/registry/service-loader/public API;
- exact 12-path ceiling and no unlisted path;
- no Java outside `modules/compiler`, no Gradle/architecture/model/other-module change;
- task 0005E Complete status synchronization, 0006 Draft, and no 0006 task file; and
- General/API-Javadoc/Planning/Example profile review, documentation no-change conclusions, links,
  anchors, fences, whitespace, and final newline checks.

Do not run backend conformance or integration suites separately: no backend or end-to-end behavior
changes, and the root checkpoint is sufficient. Do not repeat successful Java commands in the
documentation context without changed executable behavior or a recorded concrete risk.

## Dependencies

- Model 0023 and 0023A–0023F adjoint expressibility prerequisites — Complete.
- Model 0025 and 0025A–0025D current producer, policy, expansion, scatter, and dynamic-slice
  prerequisites — Complete.
- Compiler 0001–0003B capture, validation, and exact optimization — Complete.
- Compiler 0004–0004B pre-capture autograd, combined capture, and shared normalization — Complete.
- Compiler 0005 compile artifacts and publication/planning orchestration — Complete.
- Compiler 0005A–0005D complete current family formulas and policies — Complete.
- Stable current Model `OperationKind`, `OperationSignature`, Tensor producer/output, Shape,
  DataType, and public-expression contracts.

## Follow-up tasks

- Compiler 0006 may receive a detailed specification only after this task is Complete. It owns
  public explicit objectives/targets/seeds, disconnected-result behavior, `createGraph` or
  derivative order, and phase/order representation.
- A future focused Model/Compiler task may add a new operation/signature or reconsider a current
  `FC` row only with an independently justified public capability and formula/policy contract.
  Such work must update this closure checker and checkpoint.

Do not create either follow-up specification in this task.

## Architecture impact

Expected impact: None.

The existing architecture already assigns preflight, derivative policy, formula dispatch, reverse
accumulation, and combined capture to Compiler. The package-private checker makes one current
compile-time invariant executable without changing ownership, dependencies, lifecycle, graph
representation, or public API.

If implementation requires Model-owned derivative metadata, a public registry, another module
dependency, production reflection, a new formula/policy, public higher-order semantics, runtime
state, or a changed architecture rule, stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, focused architecture and ADR 0009,
documentation rules/profiles, compiler/model master plans and capabilities, and
docs/planning/modules/compiler/tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md.
Read every affected/review-only source, test, API, glossary, and completed prerequisite named by
that task.

Implement Compiler 0005E exactly within its 12 authorized paths. Add only the package-private
coverage checker, centralize preflight role selection and formula-family dispatch through it, and
prove the complete source-discovered 37-family/107-kind/128-signature D/ND/FC inventory plus
transitive and connected nested-pass formula closure. Do not add or change a derivative formula,
Model/public API, production reflection, registry/SPI, second algebra, higher-order request,
disconnected policy, runtime/backend behavior, or Compiler 0006 specification. Stop on any
formula, policy, architecture, package, or scope conflict.

Run the focused tests, full Compiler tests once, repository/architecture checkpoint, and required
source/surface checks. Then hand the actual diff and successful Java evidence to a separate
documentation-focused clean context. That pass must follow documentation-rules.md; finalize the
authorized Javadocs, Compile API, glossary impact, task evidence/summary, master plan, roadmap,
links, status, scope, and no-change conclusions; and not repeat Java tests unless executable
behavior changes or a concrete risk is recorded.

Return both context IDs, exact paths, commands/results/counts, inventory and nested-pass evidence,
no-change conclusions, unresolved issues, required follow-up, and the repository completion
status format. Mark Complete only after every acceptance criterion and documentation gate passes.
```

## Local decisions

- Planning selected a small package-private checker rather than tests/audit/documentation only.
  `OperationKind` is an open interface, while differentiable-role selection, occurrence
  validation, and formula-family dispatch are currently separate hand-maintained branches. A
  test-only hand list would detect row behavior but would not remove production dispatch drift.
- The checker centralizes only role disposition and existing family-owner selection.
  `AutogradPreflight` retains exact occurrence/prerequisite validation and family rule owners
  retain formulas. This is the smallest mechanism that makes the current closure invariant
  executable without creating a registry or refactoring formulas.
- Test-only classpath discovery is justified by a concrete recurring risk. The historical Model
  closure audit counted 127 signatures; current source contains 128 after Model 0025D appended
  `SLICE_UPDATE/CropToShapeAttrs`. Compiler 0005C already covers both data roles, so the difference
  is evidence drift rather than a formula gap.
- The nested second pass is an internal checkpoint probe over the existing first-order primitive.
  It is not task 0006. Identity-proven constant/disconnected first gradients are recorded rather
  than forcing a public disconnected-result policy into this task.
- No architecture, Model, public API, Gradle, backend, runtime, conformance, or integration change
  is planned.

## Known limitations

- One-output attention remains `FC` because its producer has no canonical weights output.
- Index-target categorical loss remains `FC` for dynamic or zero class depth.
- Batch-normalization saved-statistic slots remain canonical formula auxiliaries and cannot seed
  independent cotangent roots.
- Occurrence-dependent dynamic/binding formulas remain `FC` when their exact retained proof is
  unavailable.
- The current public surface still has no explicit gradient request, seed, disconnected-result
  policy, `createGraph`, derivative order, or higher-order result. Compiler 0006 owns those
  decisions after this checkpoint.
- The test-only discovery assumes Gradle runs tests with the Model production class output
  available as a walkable directory or archive code source. The implementation must support the
  actual Gradle test classpath form without adding a dependency or production scanner.

## Validation evidence

Planning evidence:

- Read repository instructions, authoritative/focused architecture and ADR 0009, documentation
  rules/profiles, planning guide/roadmap, compiler/model master plans and capabilities, completed
  Compiler 0001–0005D tasks, referenced Model adjoint/prerequisite tasks, current compiler/model
  source/tests, Compile/Tensor/Training APIs, and glossary.
- Confirmed a clean worktree and no pre-existing detailed 0005E or 0006 specification.
- Source/class-output discovery found 37 production `OperationKind` enum families, 107 constants,
  and 128 signatures. The 128th current row is Model 0025D's
  `SLICE_UPDATE/CropToShapeAttrs`; Compiler 0005C covers base and update.
- Confirmed all current representable first-order roles are implemented by Compiler 0005A–0005D.
  The concrete remaining gap is automated source-wide coverage/dispatch and transitive closure
  evidence, not a missing formula.
- Selected the package-private checker and test-only discovery/nested-pass design recorded above.

Implementation evidence:

- Implementation context `/root/implement_compiler_0005e` added package-private
  `FirstOrderGradientCoverage`, an immutable 128-row signature inventory, closed `D`/`ND`/`FC`
  decisions with deterministic rejection reasons, and one existing formula-family owner for
  every conditional `D`.
- `AutogradPreflight` now delegates role disposition to that checker while retaining its exact
  typed occurrence/prerequisite validation and failure order. Successful selected occurrences
  carry the checker-owned family. `FirstOrderAutograd` dispatches only through that retained
  family and no longer maintains a second kind-family branch.
- Test-only compiled-production discovery supports directory and archive code sources and proved
  complete set equality for 37 top-level enum families, 107 kind constants, and 128 exact
  kind/attributes/input-range/output-range fingerprints, including both `SLICE_UPDATE`
  attributes variants. The recorded source command
  `rg -l 'implements OperationKind' modules/model/src/main/java | sort` returned the same 37
  production family source files.
- Exact independent expectations checked every legal signature/ranged-cardinality/output/input
  boundary as `D`, `ND`, or `FC`, including owner/reason invariants, non-floating policy roles,
  unknown kinds, malformed cardinality, and illegal output/input positions. One global
  Tensor-ID-delta assertion around all four rejected explicit-fact classifications proved that
  those checker calls created no Tensor identity.
- The bounded real-expansion closure probe covered 22 representative cases across all 12 formula
  owners and high-complexity formulas. It classified 64 exact generated
  kind/attributes/output/input-edge fingerprints with no `FC` edge, completed 10
  `D`-edge-identity-connected nested expansions, and recorded 12 identity-proven
  `SECOND_PASS_NOT_APPLICABLE` disconnected/constant first gradients. This is bounded checkpoint
  evidence, not a claim that every `D` row received a nested pass. Existing Compiler 0005A–0005D
  suites remain the per-family formula evidence.
- Replacement focused validation after the documentation review identified and the implementation
  context closed an evidence gap:
  `./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest --tests io.github.pho001.synaptik.compiler.FirstOrderAutogradTest --tests io.github.pho001.synaptik.compiler.GraphCompilerTest`
  passed 43 tests across four suites: 5 coverage, 12 preflight, 9 autograd, and 17 compiler tests;
  0 skipped, failures, or errors.
- Replacement `./gradlew :modules:compiler:test` passed 29 suites and 194 tests with 0 skipped,
  failures, or errors. Replacement `./gradlew test :testing:architecture-tests:test` passed with
  all 48 actionable tasks up-to-date immediately after the full Compiler run. The replacement
  runs were required because the documentation review caused a material executable test
  expansion; production Java behavior did not change.

Documentation evidence:

- Documentation context `/root/implement_compiler_0005e/docs_finalize_0005e` applied the General,
  API/Javadoc, Planning, and Example profiles and independently reviewed the final implementation,
  coverage/preflight/autograd/compiler tests, architecture/ADR boundaries, Compile/Tensor/Public/
  Training/Runtime APIs, glossary, planning status, and affected Javadocs.
- Finalized the three production Javadocs, including the 37/107/128 source-closed contract,
  conditional disposition versus occurrence validation, retained family-owner dispatch, and the
  distinction between no Tensor/graph construction and ordinary `Decision` allocation.
- Updated the Compile API and the existing glossary pre-capture-autograd entry with the current
  source-backed closure. No new checker, task, or test-mechanism glossary entry was added.
- `./gradlew :modules:compiler:javadoc` passed after final Javadoc edits.
- `python3 /tmp/validate_synaptik_markdown.py` passed changed-Markdown local link targets and
  anchors, balanced fences, final newlines, and trailing whitespace.
- `git diff --check` passed. Manual scope/status/surface checks found exactly nine changed paths:
  three Compiler production files, one Compiler test, and five documentation/planning files; no
  unlisted path, Java outside Compiler, Model, Gradle, architecture/ADR/test, conformance,
  integration, backend, runtime, training, or unrelated-module change. Production contains no
  reflection, classpath scan, registry, service loader, public checker/facade, or mutable global
  coverage state. Compiler 0006 remains Draft and no 0006 task file exists.
- Tensor API and Model capabilities/master/tasks remain unchanged because no Model kind,
  signature, Tensor method, descriptor, result carrier, provenance, or forward semantic changed.
  Public, Training, and Runtime APIs remain unchanged because the checkpoint is package-private
  compile-time validation/dispatch and adds no public request, artifact, publication, optimizer,
  preparation, execution, or runtime behavior. Architecture, focused architecture pages, ADR
  0009, and architecture tests remain unchanged because ownership, dependency direction,
  lifecycle, and graph representation did not change. Backend conformance and integration tests
  required no focused change or separate run because no backend or end-to-end behavior changed;
  the repository/architecture checkpoint supplied the required capability evidence.

## Implementation notes

- The checker remains a pure package-private compile-time component. Production uses direct typed
  kind checks and an immutable list; only its test discovers compiled Model classes.
- The implementation centralized only role disposition and formula-family selection. Existing
  family owners, formulas, exact occurrence validation, combined capture, optimization,
  publication, planning, and diagnostics remain unchanged.
- The documentation pass initially held completion after finding that the first coverage test
  version did not independently lock exact dispositions or provide the intended bounded
  cross-family closure evidence. The implementation context expanded that test and reran the
  focused, full-module, and checkpoint commands before documentation status synchronization.

## Completion summary

- Completed changes: added the source-backed closed first-order coverage checker, unified
  preflight role selection and formula-family dispatch, and added exact inventory/disposition plus
  bounded transitive/nested closure evidence.
- Files changed or created: three Compiler production files, one Compiler test, the Compile API,
  glossary, this task, Compiler master plan, and roadmap.
- Tests and validation: replacement focused 43-test command, full Compiler 29-suite/194-test
  command, repository/architecture checkpoint, Compiler Javadoc, Markdown validation, source/
  surface/scope/status checks, and `git diff --check` all passed.
- Documentation-agent review: completed in
  `/root/implement_compiler_0005e/docs_finalize_0005e`.
- Documentation impact: Compile API and current glossary status synchronized; Tensor, Public,
  Training, Runtime, architecture/ADR, Model, conformance/integration, Gradle, backend, and
  unrelated-module no-change conclusions are recorded above.
- Javadoc review: finalized for `FirstOrderGradientCoverage`, `AutogradPreflight`, and
  `FirstOrderAutograd`; no executable Java behavior changed in the documentation context.
- Glossary impact: updated only the existing pre-capture Tensor-expression autograd entry; no new
  implementation-mechanism term was added.
- Unresolved issues: None.
- Follow-up required: None for this task. Compiler 0006 remains a separate Draft frontier without
  a detailed task specification.

Status: Complete
