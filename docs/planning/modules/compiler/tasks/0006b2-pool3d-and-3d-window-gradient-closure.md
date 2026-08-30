# Task 0006B2: Pool3d and 3D-Window Gradient Closure

## Status

Complete

## Goal

Close the five first-order gradient signatures deliberately deferred by completed Compiler task
0006B1, without changing the complete Model or Compiler forward inventory:

```text
Pool3dKind.MAX_POOL3D         + MaxPool3dAttrs      1 -> 1
Pool3dKind.AVERAGE_POOL3D     + AveragePool3dAttrs  1 -> 1
WindowTransformKind.UNFOLD3D  + Window3dAttrs       1 -> 1
WindowTransformKind.UNFOLD3D  + Unfold3dAttrs       1 -> 1
WindowTransformKind.FOLD3D    + Fold3dAttrs         1 -> 1
```

Implement their exact adjoints solely as ordinary public `Tensor` expressions constructed by
Compiler before graph capture. The formulas must themselves remain valid inputs to the existing
two-stage functional-gradient machinery. No saved maximum index, backward-only operation,
direct graph-node construction, hidden execution, or pooling-backward kind is permitted.

Forward coverage remains exactly 40 operation-kind enum families, 115 constants, and 137
signatures. Production first-order support moves from 37 families, 107 constants, and 128
signatures to exactly 38 families, 111 constants, and 133 signatures. The exact disjoint deferred
partition shrinks from nine to four signatures: `RNN`, `GRU`, `LSTM`, and `CONV3D`.

## Scope

- Add the two Pool3d and three 3D-window signatures to the closed production first-order coverage
  inventory and route `Pool3dKind` through the existing pooling rule owner.
- Remove only these five signatures from complete-forward-inventory rejection. Preserve
  allocation-free fail-closed rejection for the three recurrent signatures and `CONV3D`.
- Extend `PoolingGradientRules` with exact NCDHW maximum- and average-pooling formulas.
- Extend `LayoutGradientRules` with both `UNFOLD3D` adjoints and the `FOLD3D` adjoint.
- Preserve exact BFLOAT16, FLOAT32, and FLOAT64 types, ordinary public-operation accumulation and
  narrowing, canonical captured Shape/Dimension references, and exact same-occurrence provenance.
- Prove formula constructibility, shape compatibility, exact attributes, structural kernel
  volume, and every existing symbolic obligation during preflight before derivative allocation.
- Add source-backed inventory, preflight, first-order formula, transitive-closure, and explicit
  two-stage semantic coverage for all five signatures and their exceptional-value policies.
- Finalize affected Javadocs, API explanations, glossary impact, and synchronized planning in a
  mandatory independent clean documentation-focused context after a clean implementation context.

## Out of scope

- recurrent BPTT, `CONV3D` adjoints, or changing their deferred/fail-closed behavior
- Pool1d composition, Pool2d formulas, existing general-axis or 2D-window formulas, or any
  completed task's semantics
- a saved-index Pool3d result, a maximum-pooling backward operation, a generic PoolNd/WindowNd
  abstraction, direct `Operation`/graph-node construction, or a private formula IR
- Model kinds, attributes, signatures, Tensor methods, Shape/Dimension contracts, provenance,
  numerical semantics, source, tests, public method count, or API surface
- lowering, Planning capability ownership, Prepare, Runtime, Engine, backend contracts, CPU
  execution, generated code, storage, numerical kernels, or performance claims
- changing forward Pool3d accumulation, narrowing, excluded-padding, NaN, signed-zero, first-winner,
  or fixed-divisor semantics
- more than two derivative stages, distributional derivatives at discontinuities, or a new
  differentiability policy for comparisons, classification, Boolean expressions, `ARG_MAX`,
  `ONE_HOT`, or `WHERE` conditions
- architecture, dependency, module, Gradle, architecture-test, conformance-test, or integration-test
  changes
- a detailed task file for CPU 0008G1, Compiler 0006C, or Compiler 0007

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially Model/Compiler ownership,
  compiler-owned pre-capture autograd, the flat graph model, and dependency direction
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0005C](0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
- [Compiler 0005D](0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
- [Compiler 0005E](0005e-first-order-gradient-coverage-closure-checkpoint.md)
- [Compiler 0006](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
- [Compiler 0006B1](0006b1-pool3d-and-3d-window-forward-adoption-and-explicit-gradient-boundary.md)
- [Model capabilities](../../model/capabilities.md)
- [Model master plan](../../model/master-plan.md)
- [Model 0025J](../../model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md)
- [Model 0025K](../../model/tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)

## Architecture constraints

- Model remains the sole owner of operation meaning, public Tensor construction, immutable
  attributes, Shapes, scalar values, and numerical policy. Compiler owns pre-capture formula
  generation and closed derivative coverage.
- Every adjoint is composed through public `Tensor` algebra. It may refer to exact captured input,
  exact same-occurrence producer output, exact immutable attrs, and incoming cotangent, but never
  instantiate a graph node or operation directly.
- Generated formulas are ordinary Tensor expression graphs. Capture, inference, validation,
  optimization, publication, and Planning handoff remain unchanged.
- Complete-forward preflight must prove every current prerequisite for every requested stage before
  seed normalization, constants, formula Tensors, Tensor identifiers, or combined capture.
- Comparison, classification, Boolean, arg-extrema, one-hot, and selection-condition expressions
  keep the completed 0005A/0006 non-differentiable-condition policy. Differentiable `WHERE`
  branches remain ordinary expressions.
- Compiler adds no dependency and no public API. If exact semantics require a new Model operation,
  new predicate, direct node construction, saved state, architecture change, or a path outside the
  authorized ceiling, stop and keep the affected signature deferred.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.compiler` — closed first-order inventory, preflight, and existing
  pooling/layout rule owners only.

Existing Model packages are read-only inputs:

- `io.github.pho001.synaptik.model.operation.pooling`
- `io.github.pho001.synaptik.model.operation.layout`
- `io.github.pho001.synaptik.model`

No package, public type, public method, operation kind, attribute type, registry, facade, or module
is added.

## Exact adjoint contract

For NCDHW input `[N,C,D,H,W]`, kernel `[kD,kH,kW]`, stride `[sD,sH,sW]`, dilation
`[dD,dH,dW]`, symmetric padding `[pD,pH,pW]`, and output grid `[DOut,HOut,WOut]`, canonical
unfold columns have Shape `[N,C*kD*kH*kW,DOut*HOut*WOut]`. Their exact coordinate order is:

```text
id = od*sD - pD + kd*dD
ih = oh*sH - pH + kh*dH
iw = ow*sW - pW + kw*dW
q  = (((c*kD + kd)*kH + kh)*kW + kw)
r  = ((od*HOut + oh)*WOut + ow)
```

Depth precedes height, height precedes width, and width is fastest within both kernel and output
coordinates. All formulas must use the exact captured original Shape and immutable window attrs;
they must not rederive an equivalent but identity-different symbolic Shape when the public method
accepts the original reference.

### Three-dimensional window adjoints

- For either direct `UNFOLD3D + Window3dAttrs` or typed
  `UNFOLD3D + Unfold3dAttrs`, the input cotangent is:

  ```text
  dInput = incoming.fold3d(originalInput.shape(), window)
  ```

  Typed padding is immutable configuration and has no cotangent. `fold3d` excludes geometrically
  padded and ceil-tail samples, so their constants contribute nothing to the input cotangent.
- For `FOLD3D + Fold3dAttrs`, the columns-input cotangent is:

  ```text
  dColumns = incoming.unfold3d(window)
  ```

  This is the direct positive-zero-padding overload. It is not an arbitrary typed-padding
  reconstruction: columns discarded by forward fold because they address padding or a ceil tail
  must receive exact positive zero.
- Linear closure alternates these public operations. Generated `UNFOLD3D` differentiates through
  `FOLD3D`; generated `FOLD3D` differentiates through direct positive-zero `UNFOLD3D`.

### Average Pool3d gradient

Let `K = kD*kH*kW` be the logical kernel cardinality and `L = DOut*HOut*WOut`. Restore incoming
cotangent to `[N,C,L]`. Construct the fixed divisor as a Tensor expression by expanding an exact
typed one over Shape `[kD,kH,kW]` and reducing all three axes. Do not count in-bounds samples and
do not use a host-computed `long K` as the numerical divisor. Then:

```text
perOutput = reshape(incoming, [N,C,L]) / logicalKernelDivisor
columns   = expand/reshape(perOutput, [N,C*kD*kH*kW,L])
dInput    = columns.fold3d(originalInput.shape(), window)
```

Every logical kernel position receives `incoming/K`; padded and ceil-tail positions are discarded
by fold. Overlapping valid positions accumulate through public `fold3d`. The result retains the
input floating type through the existing cotangent normalization boundary. BFLOAT16, FLOAT32, and
FLOAT64 use their existing public-operation promotion, accumulation, and narrowing contracts; no
gradient-only cast, host arithmetic, alternative accumulator, or changed forward rounding rule is
introduced.

In particular, every emitted scalar and arithmetic expression stays in the original floating
type under the existing public promotion table. `fold3d` accumulates contributions in canonical
column order: FLOAT64 performs sequential binary64 additions, FLOAT32 performs sequential binary32
additions, and BFLOAT16 widens the represented accumulator and operand to FLOAT32, adds once, then
narrows to BFLOAT16 after every contribution. The logical fixed divisor is represented through the
ordinary Tensor reduction/division rules. These derivative-expression rules neither reinterpret
nor change forward Average Pool3d's FLOAT32 accumulation/division for BFLOAT16/FLOAT32 inputs,
FLOAT64 accumulation/division for FLOAT64 input, or BFLOAT16 single result narrowing.

### Maximum Pool3d gradient

Winner reconstruction must use the exact same-occurrence forward output `producer.output(0)` and
must not re-run Pool3d. With `K` and `L` as above:

1. Unfold input with an exact typed negative-infinity padding scalar, reshape to `[N,C,K,L]`, and
   permute to `[N,C,L,K]` candidates.
2. Unfold an exact typed one-like input with direct positive-zero padding and the same window;
   reshape/permute identically and compare with zero to form the in-bounds mask.
3. Reshape the same-occurrence output to `[N,C,L,1]`. An eligible candidate is in bounds and either
   both candidate and output are NaN, or they are numerically equal with exact signed-zero
   agreement. Signed-zero agreement is proved by reciprocal signs (`1/candidate == 1/output`) when
   both values compare equal to zero.
4. Convert eligibility to exact typed zero/one, take `argMax(-1, false, FIRST_INDEX)`, construct a
   public one-hot mask of static depth `K`, and intersect it with eligibility. The intersection is
   mandatory so an all-padding output selects nothing even though first-index arg-max returns zero.
5. Route reshaped incoming cotangent through `where(selected, incoming, +0)`, inverse-permute and
   reshape to canonical columns, then fold into `originalInput.shape()`.

This selects the first exact eligible occurrence in depth-height-width order, excludes padding
even when a real input is negative infinity, treats NaN as dominant through same-output matching,
distinguishes `+0.0` from `-0.0`, chooses the first equal occurrence, produces zero for all-padding
outputs, and accumulates overlapping winners through fold. Exact occurrence matching, not merely
numeric equality, is required.

## Higher-order and preflight contract

- Average Pool3d and the three window transforms are linear in their differentiable input. Their
  generated formulas remain ordinary public Tensor graphs; a second derivative with respect to
  that input is disconnected and follows the request's existing `ERROR` or exact typed `ZERO`
  policy.
- Maximum Pool3d's reconstructed selector is a non-differentiable constant under the existing
  comparison/selection policy. Away from winner changes, its derivative with respect to input is
  zero. At equal-value ties, signed-zero boundaries, NaN selection changes, and other
  discontinuities, the same selected zero/almost-everywhere convention applies; no distributional
  derivative is represented. `DisconnectedPolicy.ZERO` returns exact typed zero and `ERROR`
  rejects the disconnected second derivative. A differentiable incoming branch still propagates
  through the selected `WHERE` branch normally.
- Preflight validates exact signatures, descriptors, floating eligibility, attrs, Shapes, output
  alignment, and public-formula availability for every reachable requested stage.
- Checked host multiplication may prove only the positive, statically representable structural
  kernel volume required by reshape and one-hot depth. It must not become average's numerical
  divisor. An overflow, non-positive/unrepresentable one-hot depth, unprovable Shape equality,
  malformed manual occurrence, or unavailable public expression fails before allocation.
- Existing canonical Dimension expressions and forward inference constraints remain authoritative.
  Compiler does not bind symbols, materialize data, or defer gradient-formula legality to Runtime.

## Affected files

Production, exactly these existing files when implementation requires their owned change:

1. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
2. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverage.java`
3. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/PoolingGradientRules.java`
4. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LayoutGradientRules.java`

Tests, exactly these existing files:

5. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
6. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverageTest.java`
7. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ConvolutionAndPoolingGradientRulesTest.java`
8. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LayoutWindowGradientRulesTest.java`
9. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/Pool3dAndWindow3dCompilerTest.java`

Documentation and synchronized planning:

10. `docs/api/compile-api.md`
11. `docs/api/tensor-api.md`
12. `docs/glossary.md`
13. `docs/planning/modules/compiler/tasks/0006b2-pool3d-and-3d-window-gradient-closure.md`
14. `docs/planning/modules/compiler/master-plan.md`
15. `docs/planning/modules/model/capabilities.md`
16. `docs/planning/modules/model/master-plan.md`
17. `docs/planning/roadmap.md`

Review without modification unless a documented conflict forces the task to stop:
`FirstOrderAutograd`, `GraphCompiler`, `CapturedGraphInference`,
`StructuredOperationInference`, `LayoutInference`, public Tensor Pool3d/window methods and
Javadocs, Compile and Training APIs, architecture documents, Model tests, architecture tests,
backend conformance, integration tests, Gradle files, and other modules.

## Maximum scope

The hard ceiling is 17 paths: four existing production files, five existing test files, and eight
documentation/planning files. No new Java type or test file is authorized. Fewer paths are
preferred when existing Javadocs or prose remain exact, but every no-change conclusion must be
recorded. Crossing the ceiling, changing Model or public API, or needing a new operation is a stop
condition requiring a new planning decision.

## Acceptance criteria

- The exact five signatures are production-supported and owner-dispatched; forward remains
  40/115/137, support is exactly 38/111/133, and deferred is exactly the four recurrent/Conv3d
  signatures with disjoint union equality against the reflection-derived Model inventory.
- Both backward-capable modes accept the five signatures when all prerequisites hold, including on
  unrelated requested branches, while recurrent and Conv3d still fail before any derivative
  allocation.
- Both unfold variants and fold produce the exact adjoints above, including positive-zero padded
  column cotangents for fold and overlap accumulation for unfold.
- Average Pool3d divides by fixed logical `kD*kH*kW`, never valid-sample count, and proves padding,
  dilation, ceil-tail, overlap, symbolic Shape, all-padding, and all three floating-type cases.
- Maximum Pool3d proves same-occurrence output use, excluded padding, real negative infinity,
  all-padding zero, NaN dominance, signed-zero ordering, first equal depth-height-width occurrence,
  exact occurrence matching, overlap accumulation, and all three floating types.
- Formula graphs contain only public Tensor operations and captured references. Tests prove there
  is no saved index, backward-only kind, direct graph-node construction, hidden helper dispatch, or
  forward Pool3d reconstruction.
- Two-stage tests prove linear second-derivative `ERROR`/`ZERO` behavior and maximum's fixed-mask
  almost-everywhere convention, including ties/discontinuities and propagation through a
  differentiable incoming branch.
- Every unprovable structural or semantic prerequisite fails in preflight before seed/constants,
  formula Tensor, Tensor-ID, or combined-capture allocation.
- Directly affected Javadocs are meaningful and complete. Compile/Tensor API and glossary remain
  synchronized with current-versus-planned boundaries; unchanged documentation has an explicit
  reasoned no-change conclusion.
- No public API, Model behavior, forward inventory, dependency, architecture, backend capability,
  Gradle configuration, or CPU task file changes.

## Tests / validation

Implementation context, focused affected Compiler suites:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest \
  --tests io.github.pho001.synaptik.compiler.ConvolutionAndPoolingGradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.LayoutWindowGradientRulesTest \
  --tests io.github.pho001.synaptik.compiler.Pool3dAndWindow3dCompilerTest
```

Focused unchanged Model boundary evidence:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.pooling.Pool3dSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMaxPool3dExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorAveragePool3dExpressionTest \
  --tests io.github.pho001.synaptik.model.operation.layout.WindowTransformSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorWindowExpressionTest
```

Run the final affected-module suite exactly once after focused evidence:

```bash
./gradlew :modules:compiler:test
```

Documentation context, after reviewing the implementation evidence and changing no executable
Java unless it discovers a concrete defect that returns to implementation context:

```bash
./gradlew :modules:compiler:javadoc :modules:model:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Inventory/reflection evidence must enumerate all current Model declarations, prove 40/115/137,
prove exact supported 38/111/133 and deferred four, verify owner dispatch, and verify the public
Tensor method surface is unchanged. The source-backed/reflection-driven assertions in
`FirstOrderGradientCoverageTest` and existing Model signature/surface tests are authoritative;
record their exact results rather than hand-counting.

Also validate Markdown links and anchors, balanced fences, exact 17-path ceiling, no unexpected
files, synchronized statuses/order, and absence of
`docs/planning/backends/cpu/tasks/0008g1-*.md`. Do not rerun the repository-wide suite: this task
changes one module, no dependency or shared build boundary, and the planning guide assigns the
full repository tier to a later recorded checkpoint or CI.

## Dependencies

- Complete [Model 0025J](../../model/tasks/0025j-first-class-ncdhw-max-average-pool3d-semantics.md)
- Complete [Model 0025K](../../model/tasks/0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md)
- Complete [Compiler 0005D](0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
- Complete [Compiler 0006B1](0006b1-pool3d-and-3d-window-forward-adoption-and-explicit-gradient-boundary.md)
- Existing completed Compiler 0005A/0005C/0005E/0006 policies and public Tensor algebra

## Follow-up tasks

- Draft Compiler 0006C remains the separate Conv3d adjoint-expressibility/closure decision.
- Draft Compiler 0007 remains later exact algebra work.
- Draft CPU 0008G1 follows this task and owns Pool1d composition validation plus Pool3d execution;
  it remains without a detailed task file.
- Draft CPU 0008H remains after CPU 0008G1. This task must not reorder or detail either CPU task.

## Architecture impact

No architecture change is expected. This task fills five already assigned Compiler-owned gradient
slots using existing public Model algebra and the established pre-capture autograd boundary. It
adds no module edge, public surface, graph layer, runtime behavior, or backend contract. Any need
to alter `ARCHITECTURE.md`, architecture tests, or an ADR is a stop condition, not implied scope.

## Implementation prompt

Execute this task in a separate clean implementation context. Do not use a GSD workflow, commit,
or push. Read `AGENTS.md`, `ARCHITECTURE.md`, the current architecture plan, planning guide,
roadmap, this task, its dependencies, the directly affected source/tests, and public Tensor
Pool3d/window contracts in full. Implement exactly the five formulas and inventory/preflight
transition specified here within the 17-path ceiling. Use only public Tensor expressions and exact
captured references. Preserve all unrelated behavior and stop on architectural uncertainty.

Run the focused Compiler suites, focused unchanged Model boundary suites, and one final full
Compiler suite. Record exact commands/results and all reasoned no-change conclusions. Then hand
the same uncommitted change to a distinct clean documentation-focused context. That context must
read the documentation rules and General/API-Javadoc/Planning profiles, independently inspect the
source and evidence, finalize affected Javadocs/API/glossary/planning, run Javadoc/Markdown/scope/
hygiene checks without repeating successful Java suites, and leave the overall task Complete only
when implementation, documentation, and validation all pass.

## Local decisions

- The task is one cohesive closure because Pool3d gradients require the same public 3D
  unfold/fold algebra whose adjoints are needed for transitive and higher-order differentiation.
- `FOLD3D`'s adjoint uses direct positive-zero unfold; typed unfold padding is configuration, not a
  differentiable input.
- Average uses a Tensor-computed fixed logical divisor. Host kernel volume is structural only.
- Maximum reconstructs the selected occurrence from the exact same-occurrence output and an
  explicit in-bounds mask. Same numeric value alone is insufficient for NaN and signed zero.
- The first-winner mask is non-differentiable under the already accepted policy. Higher-order
  behavior is the fixed-mask, almost-everywhere convention with existing disconnected handling.
- The production inventory transition is exactly 37/107/128 -> 38/111/133; forward 40/115/137 is
  unchanged; only recurrent ×3 and Conv3d remain deferred.
- No repository-wide validation is required because there is no boundary/build/dependency change.

## Known limitations

- Maximum Pool3d derivatives are not distributional at ties, NaN boundaries, or winner changes.
- No saved maximum indices are exposed, so exact reconstruction intentionally uses public algebra
  and the same-occurrence output.
- Recurrent and Conv3d gradients remain unavailable and fail closed.
- This task adds no CPU execution. Successful formula capture is not a backend capability claim.

## Validation evidence

Planning preparation established from current source and completed task evidence:

- complete forward inventory: 40 families / 115 constants / 137 signatures;
- current support: 37 / 107 / 128;
- this exact addition: Pool3d `+1/+2/+2`, 3D-window `+0/+2/+3`;
- resulting support: 38 / 111 / 133, with exactly four deferred signatures;
- the five public formulas are expressible through existing unfold3d/fold3d, reshape, permute,
  expand, reduction, comparison/classification, arg-max, one-hot, Boolean, and where operations;
- no CPU 0008G1 detailed task file exists.

Implementation and documentation evidence follows.

Implementation-context evidence on 2026-08-30:

- `FirstOrderGradientCoverageTest` passed its reflection-derived partition assertion: complete
  Model forward inventory remains exactly 40 families / 115 constants / 137 signatures;
  production first-order coverage is exactly 38 / 111 / 133; the disjoint deferred set is exactly
  `RNN_TANH`, `GRU_RESET_AFTER`, `LSTM`, and `CONV3D`.
- The exact focused Compiler command passed 34 tests across the five requested suites with zero
  failures, errors, or skips: `AutogradPreflightTest` 14,
  `FirstOrderGradientCoverageTest` 6, `ConvolutionAndPoolingGradientRulesTest` 6,
  `LayoutWindowGradientRulesTest` 3, and `Pool3dAndWindow3dCompilerTest` 5.
- The exact focused unchanged Model command was up-to-date and its retained JUnit XML reports 54
  tests with zero failures, errors, or skips: `Pool3dSemanticsTest` 4,
  `TensorMaxPool3dExpressionTest` 6, `TensorAveragePool3dExpressionTest` 6,
  `WindowTransformSemanticsTest` 14, and `TensorWindowExpressionTest` 24.
- The single final `./gradlew :modules:compiler:test` run passed 241 tests across 35 suites with
  zero failures, errors, or skips.
- Formula-structure tests prove both unfold attribute variants fold into the exact captured input
  Shape, fold uses direct `UNFOLD3D + Window3dAttrs`, average constructs its logical divisor by
  expanding one across `[kD,kH,kW]` and reducing, and maximum uses the same-occurrence output,
  typed negative-infinity candidates, a direct-positive-zero in-bounds probe, NaN and reciprocal
  signed-zero matching, first-index arg-max, static-depth one-hot, eligibility intersection,
  public `WHERE`, inverse permutation/reshape, and overlap `FOLD3D`. No generated Pool3d
  reconstruction occurs.
- Two-stage tests prove disconnected linear `UNFOLD3D` uses the existing `ERROR`/typed-`ZERO`
  policy and a maximum fixed mask still permits a differentiable incoming branch to close through
  a connected second stage.
- `javap -public` still reports exactly 213 public `Tensor` declarations. Neither rule owner
  imports or constructs `Operation`, `TensorProducer`, or graph capture state. Existing central
  dispatch remains `POOLING -> PoolingGradientRules` and `LAYOUT -> LayoutGradientRules`.
- The exact changed-path audit contains 14 paths, all within the authorized 17-path ceiling;
  `git diff --check` passes; no `docs/planning/backends/cpu/tasks/0008g1-*.md` exists.
- Reasoned no-change conclusions: Model source/tests and public Tensor/Shape/Dimension contracts
  are unchanged because all derivatives compose existing public operations; `FirstOrderAutograd`,
  `GraphCompiler`, inference owners, Compile/Training APIs, architecture documents/tests, backend
  conformance, integration tests, Gradle files, dependencies, and every other module retain their
  existing contracts. Compile/Tensor API prose and glossary need only the mandatory independent
  documentation review/finalization; this implementation context did not revise them.

## Implementation notes

Implementation-context work is complete and remains uncommitted for the mandatory independent
documentation-focused handoff. The five exact signatures are now in the closed coverage inventory;
Pool3d dispatches to the existing pooling owner and all three volumetric window signatures dispatch
to the existing layout owner. Complete-forward rejection now retains only fixed recurrence and
Conv3d.

`AutogradPreflight` independently revalidates each selected forward occurrence, checks the
three-factor structural kernel product, and proves the public 3D unfold/fold counterpart restores
the exact input descriptor before derivative allocation. `PoolingGradientRules` implements the
rank-specific average and maximum formulas without a backward kind, saved index, hidden output,
direct graph construction, or host numerical divisor. `LayoutGradientRules` implements the two
UNFOLD3D adjoints and one FOLD3D adjoint with exact captured Shape/window references and ignores
typed padding as configuration.

The mandatory independent documentation-focused review finalized the changed source Javadocs,
Compile/Tensor API explanations, glossary impact, synchronized planning status/evidence, and this
completion summary without changing executable Java behavior. It found no concrete executable
defect and therefore did not repeat the successful Java suites.

Documentation-context evidence on 2026-08-30:

- The isolated documentation review independently inspected all four production sources, all five
  tests, relevant Model/Compiler contracts, public Tensor and Compile surfaces, and the retained
  implementation evidence. No executable defect, architecture conflict, or stale-evidence risk
  was found.
- `./gradlew :modules:compiler:javadoc :modules:model:javadoc` completed successfully. Generated
  Compiler Javadocs contain the finalized Pool3d/3D-window contracts, including the corrected
  two- and three-dimensional pooling owner description; Model Javadocs remain unchanged and
  current.
- `/tmp/validate_synaptik_markdown.py` validates the seven changed Markdown files without the
  repository's long-standing repeated-example headings. Its repository-wide run reports only
  pre-existing duplicate-heading diagnostics; `docs/api/tensor-api.md` changed no heading, and
  its task edits introduce no link-target, fence, or final-newline error. A separate fragment
  check validated every local Markdown anchor across all eight changed documentation/planning
  files.
- Generated-Javadoc/source inspection confirms both UNFOLD3D adjoints, direct-positive-zero
  FOLD3D adjoint, logical fixed-divisor Average Pool3d formula, explicit eligible-first-winner
  Maximum Pool3d formula, and fixed-mask higher-order policy are described consistently with the
  implementation.
- `javap -public` reports exactly 213 public `Tensor` declarations. Retained JUnit XML still
  reports the final Compiler 35 suites / 241 tests and focused Model 5 suites / 54 tests, with
  zero failures, errors, or skips. The authoritative reflection-derived test evidence remains
  40/115/137 forward, 38/111/133 supported, and exactly recurrent RNN/GRU/LSTM plus Conv3d
  deferred.
- The final audit contains exactly 17 authorized paths: four production sources, five tests, and
  eight documentation/planning files. The Git index is empty, `git diff --check` passes, Compiler
  0006C and 0007 remain Draft without detailed specifications, and no CPU 0008G1 detailed task
  file exists. CPU 0008G1 remains the next Draft frontier.
- Reasoned no-change conclusions: public Tensor, Compile, and Training Java APIs remain exact;
  Model source/tests, architecture contracts/tests, backend conformance, integration tests,
  Gradle/build/dependencies, CPU planning files, and unrelated modules require no change. The
  existing glossary terms cover the behavior, so only their current status and explanation were
  synchronized.

## Completion summary

- Completed changes: closed the five Pool3d/3D-window first-order signatures through public Tensor
  algebra, finalized their preflight, formula, exceptional-value, higher-order, and inventory
  documentation, and synchronized current/planned status.
- Files changed or created: exactly four Compiler production sources, five Compiler tests, and the
  eight authorized documentation/planning paths listed under Package and documentation impact.
- Validation: reused the successful focused Compiler 34-test, focused Model 54-test, and final
  Compiler 241-test evidence; regenerated Compiler/Model Javadocs; inspected generated output and
  public surface; validated Markdown links, anchors, fences, and final newlines; confirmed exact
  inventory, scope, empty index, whitespace, task ordering, and absence of a CPU 0008G1 task file.
- Unresolved issues: none.
- Required follow-up: none for this task. Compiler 0006C, Compiler 0007, CPU 0008G1, and CPU 0008H
  remain separately owned Draft work in the recorded order.

Status: Complete
