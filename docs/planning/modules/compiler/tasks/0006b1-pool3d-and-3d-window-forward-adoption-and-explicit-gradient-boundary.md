# Task 0006B1: Pool3d and 3D-Window Forward Adoption and Explicit Gradient Boundary

## Status

Complete

## Goal

Restore a complete and truthful Compiler forward inventory after completed Model tasks 0025J and
0025K by adopting all five newly absent signatures, not only the two pooling signatures:

```text
Pool3dKind.MAX_POOL3D         + MaxPool3dAttrs   1 -> 1
Pool3dKind.AVERAGE_POOL3D     + AveragePool3dAttrs 1 -> 1
WindowTransformKind.UNFOLD3D  + Window3dAttrs    1 -> 1
WindowTransformKind.UNFOLD3D  + Unfold3dAttrs    1 -> 1
WindowTransformKind.FOLD3D    + Fold3dAttrs      1 -> 1
```

Each occurrence remains one ordinary flat, functionally pure `CompiledNode`. Compiler must
independently infer and final-validate its complete descriptor and every unresolved
depth-height-width domain obligation from captured graph state, preserve exact operation values,
ordered inputs, canonical output descriptors, optimization behavior, diagnostics, publication,
and ordinary Planning handoff, and add no lowering or execution behavior.

`FORWARD_ONLY` must accept all five signatures. `FORWARD_AND_BACKWARD` and `TRAINING_STEP` must
reject any complete original forward inventory containing any of the five signatures during
allocation-free autograd preflight, before request-stage validation, seed normalization,
derivative constants, formula Tensors, Tensor identifiers, or combined capture. Exact Pool3d and
three-dimensional window-transform gradients remain owned only by Draft Compiler task 0006B2.

After this task, the closed forward inventory is exactly 40 operation-kind enum families, 115
constants, and 137 signatures. The production first-order gradient inventory remains unchanged at
37 families, 107 constants, and 128 signatures. Its exact disjoint deferred partition becomes
nine signatures: three recurrent, one Conv3d, two Pool3d, and three 3D-window signatures.

## Scope

- Route `Pool3dKind.MAX_POOL3D` and `Pool3dKind.AVERAGE_POOL3D` through the existing structured
  operation inference boundary.
- Extend existing layout/window inference for the two `UNFOLD3D` attribute variants and the one
  `FOLD3D` variant.
- Independently validate exact kind/attributes pairing, one-input/one-output cardinality,
  BFLOAT16/FLOAT32/FLOAT64 eligibility, rank, Shape, exact typed padding, target Shape,
  structural column compatibility, result type, unresolved layout, and `requiresGrad` metadata.
- Independently derive Pool3d result Shape `[N, C, D_out, H_out, W_out]`, preserving the exact
  captured input batch and channel Dimension references.
- Independently derive canonical volumetric columns
  `[N, C * kD * kH * kW, D_out * H_out * W_out]` and exact `fold3d` target descriptors without
  trusting stored Model outputs.
- Use checked signed-`long` arithmetic and the existing canonical Dimension-expression vocabulary
  for floor or literal-ceiling depth, height, and width geometry.
- Prove, reject, or retain exact domain obligations in deterministic depth, height, width order.
- Preserve direct `UNFOLD3D + Window3dAttrs` and typed
  `UNFOLD3D + Unfold3dAttrs` as distinct exact operation values; retain the exact typed padding
  scalar and exact fold target Shape references already held by the attributes.
- Preserve ordinary exact CSE, whole-node dead-code elimination, canonical reindexing, final
  validation, publication, diagnostics, and `OperationCapabilityQuery` handoff. Equal eligible
  internal occurrences may merge only under the existing complete expression key; graph-output
  producers remain ineligible. No new identity exclusion is added.
- Extend the complete-original-forward-inventory preflight guard so the first Pool3d or 3D-window
  occurrence in deterministic producer postorder fails before derivative allocation in both
  backward-capable modes, including when the requested gradient lies on an unrelated branch.
- Make `AutogradPreflight`'s exact window-kind/attributes validation exhaustive for the current
  six `WindowTransformKind` constants without treating the three new signatures as differentiable.
- Keep `FirstOrderGradientCoverage.SIGNATURES`, production family-owner vocabulary, Pool2d rules,
  and layout gradient rules unchanged. Update source-backed tests to partition all 137 Model
  signatures into 128 supported and exactly nine deferred signatures.
- Finalize directly affected Compiler Javadocs, API explanations, glossary/capability wording, and
  planning in a mandatory separate clean documentation-focused context.

## Out of scope

- Pool3d formulas, `PoolingGradientRules` changes, saved maximum indices, winner reconstruction,
  fixed-count average cotangent distribution, or adding Pool3d to production first-order coverage
- `UNFOLD3D` or `FOLD3D` gradients, `LayoutGradientRules` changes, or any higher-order closure;
  all three exact transform signatures belong only to Draft task 0006B2 together with the two
  Pool3d signatures
- any detailed specification or implementation for 0006B2, 0006C, 0007, or CPU 0008G1
- changing Pool1d visible composition, Pool2d inference/gradients, Conv3d behavior, recurrence,
  existing general-axis/2D window behavior, or any completed Compiler task history
- Model semantics, attributes, Tensor methods, descriptors, factory/provenance behavior, public
  method counts, numerical policies, source, tests, or public Javadocs
- decomposition into unfold/fold or other operations, lowering, backend capability advertisement,
  ownership-policy changes, preparation, runtime work, kernels, generated code, storage,
  numerical evaluation, or performance claims
- new graph predicates, a generic PoolNd/WindowNd/ConvNd abstraction, another IR, graph region,
  public Compiler API, registry, facade, package, dependency, Gradle change, or architecture rule
- Planning, Prepare, Runtime, Engine, Config, Trace, backend-contract, concrete backend, extension,
  architecture-test, conformance-test, or integration-test implementation

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially Model/Compiler ownership,
  the flat graph model, compiler-owned pre-capture autograd, compile lifecycle, and forbidden
  Compiler dependencies
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0001: Tensor expression graph capture](0001-tensor-expression-graph-capture.md)
- [Compiler 0002: Captured-graph inference and validation](0002-captured-graph-inference-and-validation.md)
- [Compiler 0003: Canonicalization and forward optimization](0003-canonicalization-and-forward-optimization.md)
- [Compiler 0005: Publication, Planning orchestration, and compile artifacts](0005-publication-planning-orchestration-and-compile-artifacts.md)
- [Compiler 0005D: Attention, convolution, pooling, and loss gradients](0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
- [Compiler 0005E: First-order gradient coverage closure](0005e-first-order-gradient-coverage-closure-checkpoint.md)
- [Compiler 0006: Explicit functional gradient requests](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
- [Compiler 0006A: Fixed recurrent-scan forward adoption](0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md)
- [Compiler 0006B: Conv3d forward adoption](0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md)
- [Model 0025J: First-class NCDHW Pool3d](../../model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md)
- [Model 0025K: Public NCDHW unfold3d and fold3d](../../model/tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)

## Architecture constraints

- Model remains the sole owner of Pool3d and 3D-window meaning, immutable attributes, public
  Tensor construction, numerical policy, descriptor metadata, and canonical provenance.
- Compiler independently derives and validates captured descriptors and logical obligations; it
  must not trust stored Model descriptors merely because Model construction produced them.
- Every selected occurrence remains one ordinary flat `CompiledNode`. Compiler adds no pooling or
  window IR, nested graph, region, decomposition, executable unit, backend payload, physical
  buffer, or runtime state.
- Unresolved spatial-domain obligations use only existing `DeferredGraphConstraint` and
  `GraphPredicate` contracts. Compiler does not bind Dimensions or defer descriptor validity to
  Runtime.
- Compiler-owned autograd fails closed before derivative Tensor allocation for every unsupported
  occurrence in the complete original forward inventory. Forward inference and derivative
  support remain separate closed inventories.
- CSE remains phase- and derivative-order-local and compares exact operation value, ordered
  remapped inputs, and complete ordered output descriptors. Pool3d and 3D-window transforms are
  pure and receive no recurrent-style identity exclusion.
- Planning receives the exact surviving operation and descriptors through its existing query. No
  provider is changed and no backend support is implied.
- Compiler keeps its existing allowed dependencies and adds no public API or package.
- If implementation needs a public contract, new predicate, another module, architecture change,
  gradient formula, or file outside the authorized ceiling, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.compiler` — existing package-private capture, inference, validation,
  optimization, autograd preflight, inventory, publication, and Planning-orchestration boundary.
- `io.github.pho001.synaptik.model.operation.pooling` and
  `io.github.pho001.synaptik.model.operation.layout` — completed immutable kinds and attributes,
  consumed without modification.

Packages added or changed:

- no package is added;
- no public package surface is widened.

Type placement:

- `io.github.pho001.synaptik.compiler.CapturedGraphInference` — adds only closed dispatch for the
  existing `Pool3dKind`; `WindowTransformKind` already routes to layout inference.
- `io.github.pho001.synaptik.compiler.StructuredOperationInference` — owns Pool3d descriptor and
  ordered domain derivation beside current Pool2d/Conv3d structured inference.
- `io.github.pho001.synaptik.compiler.LayoutInference` — owns 3D-window descriptor and domain
  derivation beside current axis/2D window inference.
- `io.github.pho001.synaptik.compiler.AutogradPreflight` — owns exact current window-pairing
  validation and deterministic complete-forward rejection before derivative allocation.
- `io.github.pho001.synaptik.compiler.FirstOrderGradientCoverage` — remains unchanged; only its
  same-package source-backed test expands the exact deferred set.

No generic pooling/window helper, public diagnostic type, gradient registry, or new package is
authorized.

## Forward inference and validation contract

### Pool3d signatures

Accept only the exact two one-input/one-output signatures listed in Goal. Ordered inputs are
`[input]`. Inference must:

1. validate exact kind/attribute pairing and cardinality;
2. require BFLOAT16, FLOAT32, or FLOAT64 rank-five NCDHW input;
3. derive depth, height, then width geometry from the captured input descriptor and exact attrs;
4. retain the exact input type, batch Dimension, channel Dimension, and `requiresGrad` value;
5. produce unresolved layout; and
6. compare the independently derived descriptor with the stored output descriptor through the
   existing node-context mismatch boundary.

For each axis with input Dimension `X`, kernel `k`, symmetric padding per side `p`, dilation `d`,
stride `s`, and exact `ceilMode`:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = X + 2 * p - effectiveKernel
floor output    = floor(numerator / s) + 1
ceil output     = ceil(numerator / s) + 1
```

All static arithmetic is checked. A static negative numerator fails. An unresolved numerator is
retained through existing canonical Dimension expressions and an ordered
`DimensionAtLeast(numerator, 0)` request. Constraint subjects are exactly:

```text
pool3d depth numerator non-negative
pool3d height numerator non-negative
pool3d width numerator non-negative
```

The same geometry applies to maximum and average kinds. Compiler performs no value evaluation and
does not reinterpret excluded-padding maximum, first-winner, fixed-divisor average, accumulation,
rounding, NaN, infinity, or signed-zero semantics.

### Three-dimensional window signatures

Accept only the exact three one-input/one-output signatures listed in Goal.

Direct and typed unfold both require BFLOAT16, FLOAT32, or FLOAT64 rank-five input. Typed unfold
also requires exact `paddingValue.dataType() == input.dataType()`. Both independently derive:

```text
[N, C * kernelDepth * kernelHeight * kernelWidth,
    outputDepth * outputHeight * outputWidth]
```

Products use successive existing `DimensionExpressions.multiply` calls in the Model-defined
order; no host `long` kernel-volume product is introduced. The exact input batch Dimension is
retained. Depth, height, and width use the Pool3d formula above and emit ordered subjects:

```text
unfold3d depth domain
unfold3d height domain
unfold3d width domain
```

`FOLD3D + Fold3dAttrs` requires floating rank-three canonical columns and a rank-five NCDHW
target Shape. It requires exact structural equality for batch, complete channel-kernel product,
and complete flattened window-count product, matching Model 0025K; unrelated unresolved symbols
fail rather than creating unnamed equality constraints. It derives target depth, height, then
width domains and emits:

```text
fold3d output depth domain
fold3d output height domain
fold3d output width domain
```

The inferred result retains the exact `Fold3dAttrs.outputShape()` reference, input type,
`requiresGrad`, and unresolved layout. Compiler neither reads the padding scalar nor interprets
coordinate order, padding exclusion, overlap addition, or numerical accumulation during
inference; those are immutable operation semantics for later backend execution and gradient
construction.

### Repeated validation and ordinary graph behavior

`CapturedGraphInference.inferAndValidate` must perform these checks before canonicalization and
after canonicalization or every changed optional optimization candidate, exactly like current
families. Final diagnostics contain only obligations for the final graph in deterministic node
and depth-height-width rule order.

Capture remains generic and creates one flat node per Model producer. DCE remains whole-node.
Pool3d and 3D-window occurrences use ordinary exact CSE: equal eligible internal occurrences may
merge; different kinds, attributes including padding bits/target Shape/window geometry, inputs,
phases, derivative orders, or output descriptors do not; graph-output producers do not merge.

Existing publication and Planning orchestration carry the exact surviving operation, ordered
input descriptors, output descriptor, and diagnostics. A recording test provider may accept a
query only to prove the unchanged handoff; no production provider or capability table changes.

## Autograd and inventory boundary

`GraphCompiler` continues to validate top-level arguments first. For either backward-capable
mode, `AutogradPreflight.preflight` then inventories the complete original forward boundary and
calls the forward-only guard before reading the first request stage, validating outputs/targets,
inspecting explicit/default seeds or ingress, selecting routes, or constructing any derivative
Tensor.

The guard scans deterministic producer postorder and rejects the first occurrence whose kind is
recurrent, Conv3d, Pool3d, `UNFOLD3D`, or `FOLD3D`. Existing recurrent and Conv3d diagnostics stay
unchanged. New diagnostics contain producer-postorder index, exact kind class/name, exact
attributes class, and one stable explanation:

```text
Pool3d is forward-only until Compiler task 0006B2 closes its gradients
three-dimensional window transforms are forward-only until Compiler task 0006B2 closes their gradients
```

The Pool3d explanation applies to either Pool3d kind. The window explanation applies to all three
exact 3D-window signatures. Rejection applies even when the occurrence is a separate forward root
and the requested target lies on an unrelated supported branch. It leaves the next Tensor ID
unchanged and creates no seed, typed splat, local formula, gradient accumulator, matching
auxiliary, second-stage expression, combined graph, or partial backward graph.

The exact inventory after implementation is:

```text
Model and Compiler forward inventory: 40 families, 115 constants, 137 signatures
Compiler supported gradient inventory: 37 families, 107 constants, 128 signatures
Deferred exact signatures:              9
  recurrent:                             3
  Conv3d:                                1
  Pool3d:                                2
  UNFOLD3D/FOLD3D:                       3
Supported intersection deferred:        empty
Supported union deferred:               exact complete Model inventory
```

All five new signatures continue to classify every output/input boundary as `FC` with the existing
reason `unknown or unclassified operation kind/attributes pairing`. Do not add a production
family owner or formula row in this task.

## Affected files

Expected Compiler production source (4):

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CapturedGraphInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StructuredOperationInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`

Expected Compiler tests (5):

- add `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/Pool3dAndWindow3dCompilerTest.java`
- update `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/StructuredOperationInferenceTest.java`
- update `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LayoutInferenceTest.java`
- update `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- update `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverageTest.java`

Expected documentation and planning (8):

- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/master-plan.md`
- this task specification
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged and record a reasoned conclusion unless evidence requires stopping:
`GraphCapture`, `ForwardCommonSubexpressionElimination`, `ForwardDeadCodeElimination`,
`GraphCompiler`, `PublicationPlan`, `CompileArtifacts`, `FirstOrderGradientCoverage`,
`PoolingGradientRules`, `LayoutGradientRules`, their focused tests, Model Pool3d/3D-window
source/tests/Javadocs, architecture and focused architecture pages, Runtime and Training APIs,
Planning/Prepare/Runtime/Engine, backend plans/providers, architecture/conformance/integration
tests, Gradle/dependencies, and every other module.

## Maximum scope

Hard ceiling: 17 paths total — four Compiler production files, five Compiler test files, and eight
documentation/planning files listed above.

The 17-path ceiling is justified because one atomic Compiler inventory closure must update two
existing inference owners, the common dispatch, the pre-allocation guard, family-focused tests,
one vertical handoff test, source-backed inventory evidence, and the documentation that currently
states all five signatures are absent. Splitting Pool3d from the 3D-window signatures would leave
the Compiler source uncompilable against the completed enum and would publish a knowingly
incomplete forward inventory.

Do not use the ceiling for unrelated cleanup. Stop before changing another path, production
first-order coverage, a gradient-rule owner, Model Java, another module, public API, dependency,
build file, architecture contract, provider capability, conformance, or integration behavior.

## Acceptance criteria

- Compiler builds against the completed six-value `WindowTransformKind`; the exhaustive
  preflight pairing switch accepts only the exact three new 3D attribute pairings and does not
  grant derivative support.
- `FORWARD_ONLY` accepts all five signatures as one ordinary flat node each with exact operation,
  attributes, input, output, type, Shape, layout, and gradient-metadata preservation.
- Pool3d inference independently validates exact pairings/cardinality/floating rank-five input,
  derives exact floor/literal-ceil NCDHW geometry, retains batch/channel references, and emits
  ordered depth-height-width domain obligations.
- 3D-window inference independently validates both unfold variants and fold, exact padding type,
  ranks, structural column compatibility, target Shape, canonical products, floor/literal-ceil
  geometry, and ordered depth-height-width domain obligations.
- Static valid, zero batch/channel, literal terminal all-padding, and exact symbolic cases pass;
  wrong pairing/cardinality/type/rank/padding/target/column/geometry and checked-overflow cases fail
  with deterministic node context.
- Stored-descriptor mismatches are caught during initial/final validation with existing node,
  output, ValueId, expected, and stored context.
- Ordinary CSE/DCE/canonicalization behavior is unchanged and explicitly tested for equal internal,
  distinct operation-value/input, and graph-output cases across Pool3d and 3D-window families.
- Publication, diagnostics, and a recording Planning provider receive exact surviving operation
  attributes and ordered descriptors; no production provider advertises these kinds.
- Both backward-capable modes reject the first new-family occurrence in complete producer
  postorder before any derivative Tensor/ID allocation, including unrelated-branch cases, for all
  five exact signatures.
- The source-backed inventory is exactly 40/115/137 forward, 37/107/128 supported gradients, and
  nine exact deferred signatures with empty intersection and complete union.
- `FirstOrderGradientCoverage`, `PoolingGradientRules`, `LayoutGradientRules`, existing Pool2d,
  Conv3d, recurrent, and 2D-window formulas remain unchanged.
- No decomposition, execution, capability advertisement, public API, Model Java/test, dependency,
  Gradle, architecture, backend, conformance, or integration change is present.
- Focused Compiler tests, focused unchanged Model boundary tests, one final Compiler suite,
  Compiler Javadoc, documentation validation, exact 17-path scope, package/status/order/task-file
  checks, empty staging, and `git diff --check` pass.
- A separate clean documentation-focused context independently finalizes affected source Javadocs,
  APIs, glossary/capability impact, planning evidence/status, and no-change conclusions in the same
  overall change.

## Tests / validation

Focused Compiler matrix during implementation:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.Pool3dAndWindow3dCompilerTest \
  --tests io.github.pho001.synaptik.compiler.StructuredOperationInferenceTest \
  --tests io.github.pho001.synaptik.compiler.LayoutInferenceTest \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest \
  --tests io.github.pho001.synaptik.compiler.ForwardCommonSubexpressionEliminationTest \
  --tests io.github.pho001.synaptik.compiler.CompileDiagnosticsTest
```

Focused unchanged Model boundary once:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.pooling.Pool3dSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMaxPool3dExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorAveragePool3dExpressionTest \
  --tests io.github.pho001.synaptik.model.operation.layout.WindowTransformSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorWindowExpressionTest
```

After executable Compiler code stabilizes, run exactly one final affected-module suite:

```bash
./gradlew :modules:compiler:test
```

Record Gradle outcomes and JUnit XML suite/test/skip/failure/error counts. Do not run the full
Model or repository suite: Model executable behavior, dependencies, architecture, shared build,
backend behavior, and end-to-end behavior do not change. Architecture, backend-conformance, and
integration suites are review-only unless implementation reveals a boundary conflict, in which
case stop rather than expanding this task.

The implementation context hands the actual diff and exact executable evidence to a distinct
clean documentation-focused context. That context reuses successful Java evidence unless it
changes executable behavior or identifies a concrete stale-evidence risk, then runs:

```bash
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --cached --name-only
git status --short
```

The documentation pass must also inspect generated public Compiler Javadocs and all four changed
package-private source Javadocs; validate local Markdown links/anchors, unique headings, balanced
fences, final newlines, terminology, and current-versus-planned claims; confirm exact operation,
constraint, diagnostic, inventory, and failure-order wording; confirm exactly 17 authorized paths
and matching package placement; confirm 0006B1 is Complete only after all gates; keep 0006B2,
0006C, and 0007 Draft with no detailed files; preserve the cross-plan order
`0025I -> 0025J -> 0025K -> 0006B1 -> 0006B2 -> CPU 0008G1 -> CPU 0008H`; and confirm an empty
Git index.

## Dependencies

Hard prerequisites, all Complete:

- Model 0018K–0018N, 0018V, and 0025 provide immutable operation signatures, canonical producer
  outputs, symbolic Dimensions, descriptors, and exact inventory foundations.
- [Model 0025J](../../model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md)
  provides exact Pool3d kinds, attributes, geometry, numerical meaning, and one-input provenance.
- [Model 0025K](../../model/tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md)
  provides the two additional window constants and three exact signatures whose omission would
  leave Compiler's forward inventory and enum handling incomplete.
- Compiler 0001–0003B provide generic flat capture, independent inference/final validation,
  canonicalization, exact optimization, DCE/CSE, and constant sidecars.
- Compiler 0004–0005E provide allocation-free autograd preflight, combined capture, publication/
  Planning orchestration, Pool2d/layout gradient precedent, and the source-backed 128-signature
  first-order closure.
- Compiler 0006 provides the current one/two-stage functional requests and both backward-capable
  modes.
- Compiler 0006A and 0006B provide complete-original-forward rejection and the intentional split
  between adopted forward inference and deferred derivative inventories.

Existing Compiler dependencies on Model, Config, Planning, Backend Contract, and Trace are
sufficient. No architecture decision or new dependency is required.

## Follow-up tasks

- Draft Compiler 0006B2 remains the sole owner of all five gradient signatures deferred here. It
  must add exact Pool3d gradients through public `unfold3d`/`fold3d` and close the three 3D-window
  adjoints required for transitive and higher-order formula closure. Average uses the logical
  `kernelDepth * kernelHeight * kernelWidth` divisor; maximum reconstructs the first eligible
  depth-height-width winner with exact NaN, signed-zero, padding, and occurrence matching.
- Draft Compiler 0006C remains the separate Conv3d adjoint-expressibility and gradient-closure
  owner.
- Draft CPU 0008G1 remains the execution owner after 0006B2; it does not alter this Compiler task.
- Compiler 0007 remains the separate exact/permission-aware algebra owner.

Do not create a detailed 0006B2 or other follow-up specification while implementing this task.

## Architecture impact

Expected impact: None.

This task closes Compiler's forward inventory through existing flat capture, inference,
validation, optimization, diagnostics, publication, and Planning handoff and preserves the
architecture's pre-capture fail-closed autograd boundary. It changes no ownership, dependency,
public API, build, or execution contract. If implementation reveals a need for any such change,
stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate clean implementation context:

```text
Work in /Users/phujka/IdeaProjects/Synaptik. Do not commit, stage, or push. Do not use a GSD
workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, docs/planning/roadmap.md, the
Compiler master plan, and task
docs/planning/modules/compiler/tasks/0006b1-pool3d-and-3d-window-forward-adoption-and-explicit-gradient-boundary.md.
Read its directly referenced Model 0025J/0025K, completed Compiler capture/inference/validation/
CSE/publication/autograd/inventory precedents, and affected source/tests/docs.

Implement task 0006B1 exactly within its 17 authorized paths. Adopt all five missing signatures
for ordinary flat forward inference/final validation and reject all five from every
backward-capable complete forward inventory before derivative allocation. Preserve the exact
40/115/137 forward inventory, unchanged 37/107/128 supported-gradient inventory, and exact nine
deferred signatures. Do not implement 0006B2 gradients, modify production gradient coverage or
rules, add decomposition/execution/capability/public API, change another module or architecture,
or create a later task specification. Stop on any architecture, inventory, failure-order, or
scope conflict.

After executable implementation and recorded focused Model/Compiler plus final Compiler
validation, hand the actual diff and evidence to a distinct clean documentation-focused context.
That pass must follow docs/developer-guide/documentation-rules.md, independently finalize the
four changed source Javadocs and eight documentation/planning paths, record glossary and reasoned
no-change conclusions, and run documentation/scope/status/hygiene gates without repeating
successful Java tests unless executable behavior changes or a concrete stale-evidence risk is
recorded.

Update this task with local decisions, exact validation evidence, implementation notes,
completion summary, and final status. Do not mark it Complete before every acceptance criterion
and the mandatory documentation pass finish.
```

Mandatory documentation-agent handoff content:

- the exact task goal and this task file;
- the stabilized implementation diff and exact Java test evidence;
- affected Pool3d/3D-window inference, validation, autograd-boundary, inventory, publication, and
  Planning-handoff behavior;
- the architecture and 17-path constraints;
- the four production source Javadocs and eight documentation/planning paths to finalize; and
- the documentation/Javadoc/scope/status/hygiene commands above.

## Local decisions

- 0006B1 adopts all five missing signatures. Adopting only the two Pool3d signatures would leave
  `UNFOLD3D`/`FOLD3D` outside a claimed complete forward inventory and leaves the current
  exhaustive `AutogradPreflight` enum switch uncompilable against completed Model 0025K.
- Pool3d stays in `StructuredOperationInference`, and 3D windows stay in `LayoutInference`, beside
  their existing rank-two family owners. A new PoolNd/WindowNd helper or package is not justified.
- Pool3d and 3D-window occurrences use ordinary CSE because their Model semantics are pure and do
  not have recurrent occurrence identity.
- Dynamic spatial validity is explicit typed Compiler state in depth-height-width order. Exact
  fold column compatibility remains structural equality, preserving Model 0025K's deliberate
  rejection of unrelated unresolved symbols.
- The first-order production inventory remains unchanged. The complete source-backed deferred set
  expands from four to nine exact signatures; forward adoption is not gradient support.
- All three 3D-window gradients belong only to 0006B2 because Pool3d formulas use those operations
  and bounded higher-order differentiation requires their transitive adjoints to close with the
  two Pool3d signatures.
- Documentation uses General, API/Javadoc, and Planning profiles. No architecture document or ADR
  changes because ownership and lifecycle boundaries remain unchanged.

## Known limitations

- The five signatures compile only as forward metadata. No production backend is required to
  accept them, so Compiler adoption alone does not make them executable.
- Only the current BFLOAT16/FLOAT32/FLOAT64 NCDHW and canonical-column contracts are included.
- Unresolved spatial domains remain deferred compile constraints; this task adds no binding API.
- Pool3d and 3D-window gradients remain fail-closed until 0006B2. Conv3d and recurrent gradients
  remain separately deferred.
- Repository-wide validation remains deferred to continuous integration or a later named
  capability checkpoint.

## Validation evidence

- Implementation-focused Compiler matrix: `./gradlew :modules:compiler:test --tests
  io.github.pho001.synaptik.compiler.Pool3dAndWindow3dCompilerTest --tests
  io.github.pho001.synaptik.compiler.StructuredOperationInferenceTest --tests
  io.github.pho001.synaptik.compiler.LayoutInferenceTest --tests
  io.github.pho001.synaptik.compiler.AutogradPreflightTest --tests
  io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest --tests
  io.github.pho001.synaptik.compiler.ForwardCommonSubexpressionEliminationTest --tests
  io.github.pho001.synaptik.compiler.CompileDiagnosticsTest` — `BUILD SUCCESSFUL`; seven JUnit
  XML suites record 46 tests with zero skipped, failures, or errors.
- Focused unchanged Model matrix covering `Pool3dSemanticsTest`,
  `TensorMaxPool3dExpressionTest`, `TensorAveragePool3dExpressionTest`,
  `WindowTransformSemanticsTest`, and `TensorWindowExpressionTest` — `BUILD SUCCESSFUL`; five
  JUnit XML suites record 54 tests with zero skipped, failures, or errors.
- Final implementation module run: `./gradlew :modules:compiler:test` — `BUILD SUCCESSFUL`;
  Compiler JUnit XML records 237 tests in 35 suites with zero skipped, failures, or errors.
- Implementation-pass Javadoc verification: `./gradlew :modules:compiler:javadoc` —
  `BUILD SUCCESSFUL` without warnings.
- `git diff --check` passed. The implementation changed only the four authorized production and
  five authorized test paths; the pre-existing planning diff remains for documentation review.

- Planning context reviewed the current Model inventory and completed 0025J/0025K contracts,
  confirming exactly 40 families, 115 constants, and 137 signatures.
- Planning context reviewed current Compiler dispatch, structured/layout inference, autograd
  preflight, first-order coverage, CSE, publication, Planning handoff, and focused tests. At that
  pre-implementation point, Compiler forward coverage remained 39/111/132 and production gradient
  coverage remained 37/107/128.
- `./gradlew :modules:compiler:compileJava :modules:compiler:compileTestJava` currently fails at
  the exhaustive `AutogradPreflight` `WindowTransformKind` switch because completed Model 0025K
  added `UNFOLD3D` and `FOLD3D`. This is planning evidence for the five-signature atomic scope, not
  implementation evidence.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests
  io.github.pho001.synaptik.model.operation.layout.WindowTransformSemanticsTest --tests
  io.github.pho001.synaptik.model.operation.pooling.Pool3dSemanticsTest` passed in the planning
  context: three suites, 23 tests, zero skips, failures, or errors.
- Planning-only Markdown, link, status/order, task-file-absence, exact changed-path, and
  `git diff --check` results were recorded by the initial planning completion summary before
  implementation.

## Implementation notes

- Added closed Pool3d dispatch and independent maximum/average rank-five inference with checked
  floor/literal-ceil geometry, exact input batch/channel retention, and deterministic
  depth-height-width numerator constraints.
- Added independent direct/typed `UNFOLD3D` and `FOLD3D` inference, including typed padding,
  canonical column products, exact fold target retention, structural batch/channel/grid checks,
  and ordered spatial-domain constraints.
- Extended allocation-free complete-forward preflight to reject all five signatures before seed
  validation or derivative Tensor allocation in both backward-capable modes. Production
  first-order support remains exactly 37/107/128; the source-backed deferred partition is exactly
  nine signatures and the complete Model inventory is exactly 40/115/137.
- Focused tests cover static and symbolic inference, malformed manually constructed descriptors,
  exact attribute variants, stored-descriptor validation, capture, CSE/DCE, publication,
  diagnostics, Planning queries, and fail-closed ordering/allocation.
- The mandatory clean documentation pass finalized the four affected production Javadocs,
  Compile API, Tensor API, glossary, capability wording, and planning state. It reran Compiler
  Javadoc successfully, inspected the rendered/source contracts, confirmed the compiled
  package-private method surface with `javap`, validated imports and the exact five-signature
  source partition, checked local Markdown links/anchors/fences across all eight changed
  documentation/planning files, confirmed the absent 0006B2 task file and exact 17-path ceiling,
  and passed `git diff --check`. No executable Java changed during that pass, so the successful
  Java test suites were not repeated.

## Completion summary

- Completed changes: all implementation, final Javadocs, focused tests, API/glossary/capability
  documentation, planning synchronization, and prescribed validation for Compiler 0006B1.
- Files changed or created: exactly four authorized Compiler production paths, five authorized
  Compiler test paths, and eight authorized documentation/planning paths.
- Tests and validation: focused Compiler 46/46, focused Model 54/54, and full Compiler 237/237
  passed with zero skips, failures, or errors. Final Compiler Javadoc passed without warnings;
  rendered/source inspection, `javap`, import/signature checks, Markdown links/anchors/fences,
  task status/order/absence checks, exact 17-path scope, and `git diff --check` passed.
- Documentation and Javadoc impact: the four production Javadocs now state the complete Pool3d
  and 3D-window forward-only boundary. Compile/Tensor API and glossary document current inference,
  final validation, ordinary graph handling, fail-closed backward ordering, exact inventories,
  and deferred execution/gradient ownership. Compiler/Model planning and roadmap state are
  synchronized to Complete 0006B1 and Draft 0006B2 without a detailed task file.
- Reasoned no-change conclusions: generic capture, CSE/DCE, compiler orchestration, publication,
  and Planning handoff required no source changes beyond the existing dispatch/inference owners;
  production coverage and Pooling/Layout gradient rules remain unchanged because preflight rejects
  the five signatures before derivative construction. Model source/tests/public Javadocs remain
  accurate because Model semantics did not change. Architecture and focused architecture pages,
  Runtime and Training APIs, Planning/Prepare/Runtime/Engine implementation, provider plans and
  capabilities, Gradle/dependencies, architecture tests, backend conformance, integration tests,
  and other modules require no change because this task adds no public API, dependency, ownership,
  lowering, backend capability, execution, materialization, numerical result, or performance
  contract.
- Unresolved issues: none within task 0006B1.
- Required follow-up: Draft Compiler 0006B2 remains responsible for the two Pool3d gradients and
  three 3D-window adjoints; it requires a future detailed task specification before execution.

Status: Complete
