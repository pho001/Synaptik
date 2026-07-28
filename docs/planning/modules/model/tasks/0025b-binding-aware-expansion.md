# Task 0025B: Binding-Aware Expansion

## Status

Complete

## Goal

Broaden the existing `EXPAND` model contract so public expansion can retain an exact target
`Shape` when right-aligned compatibility depends on unresolved Dimensions.

```text
input Shape + exact target Shape
  -> prove locally when possible
     -> equal pair or static source singleton: accept
     -> incompatible fully static pair: reject
     -> pair containing an unresolved Dimension: retain for later proof
        -> required binding proof: source extent == 1 or source extent == target extent
```

This task changes only Model-owned expression construction and semantic documentation. It adds no
public API, operation kind, attributes type, Shape or constraint type, binding implementation, or
compiler behavior. Compiler adoption is a separate prerequisite step inside Draft
[Compiler 0005B](../../compiler/master-plan.md#task-list).

## Scope

- Preserve exactly the two current public methods:

  ```java
  public Tensor expand(long... requestedShape)
  public Tensor expand(Shape targetShape)
  ```

- Preserve `ShapeTransformKind.EXPAND` and `TargetShapeAttrs(targetShape)` as the sole expansion
  semantic spelling.
- Broaden only aligned compatibility that current immutable Shape metadata cannot decide.
- Preserve target rank greater than or equal to input rank and right alignment.
- Continue to accept structurally equal aligned Dimensions and a statically known source
  singleton.
- Continue to reject the first fully static incompatible aligned pair before Tensor identity
  allocation with the existing exception type and message.
- Accept an aligned pair when at least one Dimension is unresolved and neither existing local
  success rule proves compatibility.
- Retain the exact target `Shape` reference in the descriptor and `TargetShapeAttrs`.
- Define the retained concrete obligation as:

  ```text
  source extent == 1 OR source extent == target extent
  ```

- Keep new leading target axes as implicit-singleton expansion without an aligned deferred pair.
- Preserve all current fully resolved static zero-stride view geometry.
- Keep layout unresolved whenever compatibility or geometry is unresolved.
- Preserve exact data type, `requiresGrad`, producer/provenance, output index, label/storage
  absence, freshness, validation order, unaffected messages, and identifier side effects.
- Add focused tests for the complete static, symbolic, expression, zero, leading-axis, layout, and
  ID matrix.
- Finalize affected Javadocs, Tensor/Compile API text, glossary text, and planning records through
  the required separate documentation-focused pass.

## Out of scope

- a public `broadcastTo`, `broadcast_to`, `expandTo`, or second semantic spelling
- another Tensor method, overload, static facade, factory entry, operation kind, enum constant,
  attributes record, result carrier, alias, or deprecation bridge
- changes to `Shape`, `Dimension`, `StaticDimension`, `DynamicDimension`,
  `ExpressionDimension`, `DimensionExpression`, `DimensionExpressions`, `ShapeBroadcast`,
  `LayoutDescriptor`, `LayoutKind`, `TensorDescriptor`, `TensorFactory`, `TensorProducer`, or
  `TensorProvenance`
- a public or Model-owned binding, constraint, predicate, proof, assignment, substitution, or
  unification type
- assigning values to unresolved Dimensions, replacing the exact target Dimension, inferring a
  result extent, or manufacturing layout/runtime facts
- changing reshape, expand-dimensions, squeeze, tile, ordinary broadcasting, or sum-to-Shape
  behavior
- compiler inference, `DeferredGraphConstraint`, graph-predicate proof, autograd preflight,
  gradient formulas, graph capture, canonicalization, optimization, publication, or compile
  artifacts
- implementing or specifying Compiler 0005B; that row remains Draft without a task file
- prepare, planning, runtime, backend, engine, training, execution, materialization, storage
  aliasing, value repetition, or kernels
- architecture contracts or explanatory architecture pages, ADRs, architecture tests,
  backend-conformance tests, integration tests, dependencies, Gradle, or Java version changes
- capability-baseline changes: the existing selected expand capability and Model ownership remain
  unchanged

## Exact behavior contract

### Existing semantic and public surface

The implementation keeps:

```text
ShapeTransformKind.EXPAND
  + TargetShapeAttrs(exactTargetShape)
  + ordered input [source]
  + one output at index zero
```

`ShapeTransformKind` retains its exact enum constants, order, and one-input/one-output
`TargetShapeAttrs` signature. `TargetShapeAttrs` retains exactly one non-null `Shape targetShape`
component and performs no input-aware validation. `Tensor` retains exactly two public `expand`
methods. There is no `broadcastTo` alias or second operation representation.

The package-private `TensorExpandExpressions` helper remains final, field-free, non-instantiable,
and limited to its existing six methods:

```java
static Tensor apply(Tensor input, long[] requestedShape)
static Tensor apply(Tensor input, Shape targetShape)
private static void validateExpansion(Shape inputShape, Shape targetShape)
private static Optional<LayoutDescriptor> resolveViewLayout(
        TensorDescriptor inputDescriptor, Shape targetShape)
private static long[] deriveExpandedStrides(
        Shape inputShape, LayoutDescriptor inputLayout, Shape targetShape)
private static Tensor create(
        Tensor input,
        TensorDescriptor inputDescriptor,
        Shape targetShape,
        Optional<LayoutDescriptor> resultLayout)
```

Do not add a field, nested type, helper method, overload, or separate compatibility object.

### Validation order and diagnostics

`apply(Tensor, long[])` preserves this order:

1. null-check `input` with message `input`;
2. null-check `requestedShape` with message `requestedShape`;
3. read the exact input descriptor once and its Shape once;
4. construct the target with `Shape.of(requestedShape)`;
5. validate target rank and aligned compatibility;
6. resolve layout only when all required geometry is available;
7. construct the result once.

`Shape.of` continues to copy the raw request semantics, treat an empty array as scalar Shape,
accept zero, and reject the first negative literal with:

```text
Static dimension size must be non-negative: <value>
```

`apply(Tensor, Shape)` preserves this order:

1. null-check `input` with message `input`;
2. null-check `targetShape` with message `targetShape`;
3. read the exact input descriptor once and its Shape once;
4. validate target rank and aligned compatibility;
5. resolve layout only when all required geometry is available;
6. construct the result once.

Target rank below input rank still fails with:

```text
expand target rank <targetRank> must be at least input rank <inputRank>
```

After computing `rankOffset = targetRank - inputRank`, inspect aligned input axes in increasing
input-axis order. The first pair whose Dimensions are both static, unequal, and whose source is
not static one still fails with:

```text
cannot expand input shape <inputShape> to target shape <targetShape> at target axis <targetAxis>
```

The task changes no null, raw-size, rank, fully static incompatibility, arithmetic-overflow, or
identifier-exhaustion message. Every local failure remains before the sole factory delegation and
consumes no Tensor ID.

### Aligned compatibility matrix

An unresolved Dimension is any `Dimension` whose `isStatic()` result is false, including named
dynamic Dimensions, symbolic expression Dimensions, and constrained unknown Dimensions.

| Source Dimension | Target Dimension | Model construction | Retained binding rule |
|---|---|---|---|
| Structurally equal | Structurally equal | Accept | Already proved equal |
| Static `1` | Any static or unresolved target | Accept | Source singleton already proved |
| Static non-`1` | Unequal static | Reject immediately | Contradiction is fully known |
| Unresolved | Unequal static | Accept deferred | Source must bind to `1` or target |
| Static non-`1` | Unresolved | Accept deferred | Target must bind equal to source |
| Unresolved | Structurally unequal unresolved | Accept deferred | Source must bind to `1` or equal target |

Structural equality remains the complete Model-local equality proof. Equal named symbols, equal
symbolic formulas, and the same constrained-unknown identity are therefore accepted without a
new obligation. Distinct names, structurally unequal formulas, different unknown identities, and
static/unresolved mixtures are accepted but not equated.

The binding rule is directional:

```text
source == 1 OR source == target
```

It is not symmetric ordinary broadcasting. A target singleton does not itself prove that an
unresolved source can shrink; it defers the requirement that the source eventually bind to one.
A statically known source `2` still cannot expand to static target `1`.

### Shape examples

The examples describe construction metadata and later proof, not execution.

| Input Shape | Target Shape | Construction result | Later requirement |
|---|---|---|---|
| `[2, 3]` | `[2, 3]` | Accept | Exact equality |
| `[1, 3]` | `[4, 3]` | Accept | Source singleton |
| `[1, 3]` | `[0, 3]` | Accept | Source singleton; result axis is empty |
| `[2, 3]` | `[4, 3]` | Reject at target axis `0` | Fully static contradiction |
| `[2, 3]` | `[1, 3]` | Reject at target axis `0` | Fully static shrink |
| `[0, 3]` | `[1, 3]` | Reject at target axis `0` | Empty source is not singleton |
| `[N, 3]` | `[N, 3]` | Accept | Structural equality |
| `[N, 3]` | `[4, 3]` | Accept deferred | `N == 1 OR N == 4` |
| `[4, 3]` | `[N, 3]` | Accept deferred | `N == 4` |
| `[N, 3]` | `[M, 3]` | Accept deferred | `N == 1 OR N == M` |
| `[N + 1, 3]` | `[M, 3]` | Accept deferred | `N + 1 == 1 OR N + 1 == M` |
| `[N, 3]` | `[B, N, 3]` | Accept | `B` is a new leading axis; aligned `N` is equal |

For the final row, no source/target pair exists for leading axis `B`; it is implicit-singleton
expansion and needs no deferred aligned pair. Its numeric geometry remains unresolved because
`B` is unresolved.

Scalar input continues to expand to any target Shape. A non-scalar input cannot use a lower-rank
target. Same-Shape, repeated, and nested calls remain fresh explicit occurrences.

### Layout policy

Preserve the existing resolved layout exactly when the current facts are sufficient:

- target Shape is fully static;
- input layout is resolved, which already requires compatible fully static input geometry; and
- the aligned compatibility pass has no unresolved pair.

In that case:

- every new leading target axis receives stride zero;
- a changed statically known source-singleton axis receives stride zero;
- every unchanged aligned axis preserves the exact source stride, including an existing zero or
  non-canonical stride;
- the exact input storage offset is retained;
- the result is marked as a logical view; and
- current `LayoutDescriptor` kind/span/overflow behavior is unchanged.

Return unresolved layout when any required fact is unavailable, including:

- an unresolved target Dimension;
- an unresolved source Dimension;
- an accepted deferred aligned compatibility pair; or
- absent input layout.

Do not choose a stride from a later alternative, bind a Dimension, replace an unresolved extent,
assume a materialization, or manufacture a runtime/layout fact. Logical view metadata continues
to attach no host storage and promises neither physical aliasing nor zero-copy execution.

### Descriptor, ownership, metadata, and ID effects

Every success preserves the current construction:

1. one `TensorDescriptor` with exact input `DataType`, exact target Shape reference, selected
   resolved/unresolved layout, and exact input `requiresGrad`;
2. one `TargetShapeAttrs` retaining that same target Shape reference;
3. one `Operation(ShapeTransformKind.EXPAND, attrs)`;
4. one producer with exact ordered input `[input]` and one output descriptor;
5. canonical output Tensor at provenance index zero;
6. one fresh Tensor ID, absent label, and absent host storage.

The input identity, descriptor, Shape, layout, label, provenance, storage association, storage
liveness, and values remain unchanged. Early validation and layout-arithmetic failures consume no
ID. Successful construction consumes exactly one fresh ID. Existing terminal identifier
exhaustion remains the final factory failure, with no rollback or ID reuse.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Model master plan](../master-plan.md)
- [Model capabilities](../capabilities.md)
- [Shape and Dimension model](0002-shape-and-dimension-model.md)
- [Reshape and expand semantics](0017c-reshape-and-expand-semantics.md)
- [Expand Tensor expressions](0017d1-expand-tensor-expressions.md)
- [Symbolic extent expressions](0018m-symbolic-extent-expressions.md)
- [Binding-aware sum-to-Shape](0023a-binding-aware-sum-to-shape.md)
- [Canonical TensorProducer outputs](0025-canonical-tensor-producer-outputs.md)
- [Portable floating semantics](0025a-portable-floating-comparison-extrema-and-clamp-semantics.md)
- [Compiler master plan](../../compiler/master-plan.md)
- [Captured-graph inference](../../compiler/tasks/0002-captured-graph-inference-and-validation.md)
- [Exact-composition gradient rules](../../compiler/tasks/0004a-exact-composition-gradient-rule-extensions.md)
- [Compiler 0005A](../../compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Model owns public Tensor expression semantics, immutable descriptors, operation attributes, and
  pre-capture producer/provenance construction.
- Tensor remains public model state and is not graph IR.
- `Operation` remains backend-neutral and exposes no binding, backend support, route, or
  executable state.
- Model may reject a contradiction provable from immutable Shape values and retain an unresolved
  obligation, but it must not solve or bind that obligation.
- Compiler owns graph-wide inference, validation, deferred graph predicates, autograd preflight,
  and gradient construction.
- The existing compiler predicate vocabulary already demonstrates the later shape of the proof:

  ```text
  AnyOf(
      DimensionEqual(source, StaticDimension(1)),
      DimensionEqual(source, target))
  ```

  This is downstream evidence only. This task neither creates that predicate nor changes
  Compiler.
- Planning and prepare own materialization/lowering handoffs; concrete backends own route
  selection and implementation; runtime executes prepared work.
- No module dependency, package boundary, architecture contract, or lifecycle ownership changes.

## Package impact

No package is added, moved, or renamed.

Existing packages used:

- `io.github.pho001.synaptik.model.operation.layout`
- `io.github.pho001.synaptik.model.tensor`

Type placement:

- `...operation.layout.ShapeTransformKind` continues to own the sole `EXPAND` semantic identity.
- `...operation.layout.TargetShapeAttrs` continues to own the exact immutable target Shape.
- `...tensor.TensorExpandExpressions` continues to own package-private input-aware validation,
  layout derivation, and expression construction.
- `...tensor.Tensor` remains the public fluent facade.

No public surface or package map changes.

## Affected files

Production and Javadoc (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/ShapeTransformKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/layout/TargetShapeAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorExpandExpressions.java`

Tests (1):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorExpandExpressionTest.java`

Documentation and planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless repository evidence contradicts this specification:

- Shape/Dimension/DimensionExpressions and LayoutDescriptor foundations
- TensorDescriptor, TensorFactory, TensorProducer, TensorProvenance, and identifier allocation
- reshape, expand-dimensions/squeeze, broadcasting, and sum-to-Shape contracts
- current Compiler `LayoutInference`, `ReductionNormalizationInference`,
  `DeferredGraphConstraint`, `AutogradPreflight`, and gradient rules
- model capabilities, Training/Runtime APIs, architecture/ADRs/tests, conformance/integration
  tests, Gradle/Java 26 configuration, dependencies, and all other modules

## Maximum scope

At most the exact 12 paths listed above may change: four existing Model production/Javadoc paths,
one focused Model test, and seven documentation/planning paths.

Stop before changing a thirteenth path, adding a public/helper type or method, changing the package
map, editing compiler Java/tests, changing capabilities, architecture, dependencies, Gradle, or
creating any Compiler 0005B–0005E/0006 task file. If implementation evidence requires another
path, update this task through a planning decision before proceeding.

## Javadoc and documentation requirements

- Update all four affected production Javadocs to describe local proof, deferred aligned
  compatibility, exact target retention, and unresolved layout without claiming binding or
  execution.
- Preserve meaningful `@param`, `@return`, and `@throws` coverage for both public methods and
  helper methods.
- Keep the current validation order, exact unaffected messages, no-ID failures, metadata,
  producer/provenance, storage/label absence, freshness, and identifier exhaustion visible.
- Update Tensor API examples and failure summaries from “unprovable dynamic pairs reject” to the
  selected source-one-or-equal deferred rule.
- Update Compile API current-status text precisely:
  - structural capture can retain the broadened Model occurrence;
  - current compiler `LayoutInference` does not yet adopt binding-aware EXPAND;
  - Compiler 0005B owns the later deferred constraint and gradient/preflight integration; and
  - no concrete binding, lowering, backend, or execution support is claimed.
- Update the glossary's existing expand/broadcasting/Dimension/deferred-constraint wording only
  where needed. Do not add a second term or synonym for expansion.
- Keep completed task history intact. On completion, synchronize Model 0025B as Complete. Compiler
  0005B–0005E/0006 remain Draft without detailed specifications.
- Record reasoned no-change conclusions for capabilities, related Javadocs, Training/Runtime APIs,
  architecture/ADRs/tests, conformance/integration, Gradle/Java 26, dependencies, and other
  modules.

## Acceptance criteria

- Exactly two public `expand` overloads remain; public Tensor method count remains 200.
- `ShapeTransformKind` constants/order/signature, `TargetShapeAttrs` shape, and the exact helper
  surface remain unchanged.
- No `broadcastTo`, second semantic spelling, new kind, attrs, Shape/constraint type, alias,
  package, or facade exists.
- Exact null/raw/rank/static-incompatibility validation order, exception types/messages, and no-ID
  behavior remain unchanged.
- Equal pairs and static source-singleton pairs remain accepted.
- Every aligned pair containing at least one unresolved Dimension is accepted unless structural
  equality or static source one has already proved it.
- Every unequal fully static pair whose source is not one still fails immediately at the first
  aligned target axis.
- New leading target axes remain valid without an aligned deferred pair.
- The exact target Shape reference is retained in descriptor and attributes for raw-normalized and
  exact-Shape requests.
- Fully resolved static dense, offset, strided, broadcast, scalar, zero-extent, identity, leading,
  and singleton cases retain their exact current stride, offset, kind, view, span, and overflow
  behavior.
- Every unresolved compatibility or geometry case has unresolved layout; no dimension, stride,
  materialization, or runtime fact is manufactured.
- Every success retains exact type and eligibility, one exact EXPAND producer, `[input]`,
  provenance index zero, freshness, absent label/storage, and exactly one new ID without mutating
  input state.
- Focused tests cover named, expression, constrained-unknown, static/unresolved,
  unresolved/static, unresolved/unresolved, equal unresolved, leading unresolved, static
  contradiction, zero, layout, validation order, message, and ID cases.
- No compiler Java/test behavior changes; Compiler 0005B remains Draft and is recorded as the
  adoption owner.
- Exact 12-path scope, package placement, documentation pass, validation evidence, and planning
  status synchronization all pass before task completion.

## Tests / validation

Focused implementation validation:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.tensor.TensorExpandExpressionTest \
  --tests io.github.pho001.synaptik.model.operation.layout.ShapeTransformSemanticsTest
```

After executable Java stabilizes, run exactly one final Model suite:

```bash
./gradlew :modules:model:test
```

The implementation pass records exact test/suite counts and hands that evidence to the
documentation-focused pass. The documentation pass does not rerun successful Java tests unless it
changes executable Java behavior or records a concrete reason.

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate:

- every changed Markdown relative link and heading anchor;
- balanced fences, final newlines, and trailing whitespace;
- exact two-method public expand surface and unchanged 200-method public Tensor count;
- exact six-method field-free helper and unchanged operation/attrs surface;
- exact 12 touched paths and only Model Java/test plus documentation/planning changes;
- Model 0025B Complete synchronization in its task, Model master plan, and roadmap;
- Compiler 0005B–0005E/0006 Draft synchronization;
- no Compiler 0005B or later task file;
- no capability, compiler Java/test, Gradle, dependency, architecture, ADR, architecture-test,
  conformance, integration, runtime, prepare, backend, engine, or training edit.

Repository-wide validation is deferred to the first-order gradient capability checkpoint and CI.
This focused Model task changes no dependency, build, architecture boundary, or executable module
outside `modules/model`.

## Required separate documentation handoff

After implementation and final Model test evidence, hand the actual diff to a separate
documentation-focused agent or thread with clean context in the same overall change.

The handoff must include:

- this task specification and exact goal;
- the four affected production contracts and focused test;
- the exact implementation diff;
- exact focused/final Model test commands and results;
- the source-one-or-source-equal binding rule and complete behavior matrix;
- validation order/messages, layout policy, metadata/provenance, and ID effects;
- the current Compiler non-adoption boundary and Compiler 0005B follow-up;
- expected Tensor/Compile API, glossary, task/master/roadmap edits;
- the unchanged capability conclusion; and
- all documentation, scope, status, and whitespace checks.

The documentation agent must independently inspect source/tests and the downstream compiler
evidence, finalize affected Javadocs and explanatory/planning text, check glossary impact, and
record files reviewed, changes made, reused evidence, commands/results, limitations, and
unresolved issues. Task 0025B cannot become Complete before that pass finishes.

## Dependencies

- Model 0002: static, named dynamic, scalar, zero, structural equality, and immutable Shape rules.
- Model 0017C: the sole EXPAND identity and exact `TargetShapeAttrs` semantic representation.
- Model 0017D1: the two public expand methods, helper, current validation/messages, zero-stride
  layout, metadata, provenance, and ID contract being broadened.
- Model 0018M: expression Dimensions and unresolved Shape vocabulary without binding.
- Model 0023A: precedent for accepting only statically provable compatibility failures while
  retaining an exact target Shape for later compiler/binding proof.
- Model 0025 and 0025A: completed current Model/Compiler-enabling frontier and preserved producer
  and forward-semantics history.
- Compiler 0001–0005A: read-only downstream evidence for capture, typed deferred predicates,
  autograd, and the current first-order frontier.

All implementation prerequisites are Complete.

## Follow-up tasks

- Compiler 0005B remains Draft and must adopt binding-aware EXPAND during its existing reduction,
  scan, softmax, statistics, and normalization gradient frontier. It owns:
  - graph inference that proves, rejects, or retains each aligned
    `source == 1 OR source == target` predicate;
  - deterministic deferred-constraint subject/order;
  - preflight and generated-gradient consistency for binding-dependent EXPAND and
    sum-to-Shape inversion; and
  - focused compiler validation.
- Compiler 0005C–0005E and 0006 remain Draft in their existing order without detailed task files.
- Concrete binding, prepare/backend lowering, materialization, execution, and conformance remain
  with their later owners.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None.

If implementation requires Model-owned constraints or binding, a new public method/kind/attrs
type, compiler Java changes, a package/dependency change, or architecture-contract edits, stop and
report the conflict rather than expanding this task.

## Implementation prompt

Use this prompt in a separate clean-context task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md, the Model and Compiler master plans,
and docs/planning/modules/model/tasks/0025b-binding-aware-expansion.md in full. Read the completed
Shape/dynamic/expand/sum-to-Shape and current compiler tasks linked by 0025B. Inspect every exact
affected and review-only source/test named by the task.

Implement Model task 0025B exactly within its 12 authorized paths. Broaden only existing EXPAND:
accept aligned compatibility involving an unresolved Dimension, retain the exact target Shape,
and keep layout unresolved until compatibility and geometry are known. Preserve fully static
rejection/messages, resolved zero-stride geometry, public/helper/semantic surfaces, metadata,
provenance, freshness, validation order, and ID effects. Add no broadcastTo spelling, public
method, kind, attrs, Shape/constraint type, binding, compiler behavior, or later task spec. Stop
on an architecture, completed-contract, package, affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final Model suite after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must inspect final
source/tests and current compiler evidence, finalize affected Javadocs, Tensor/Compile APIs,
glossary and planning status, record the reasoned capabilities no-change conclusion, and run
Javadoc/Markdown/scope/status/whitespace validation without repeating successful Java tests
unless executable behavior changes or a concrete reason is recorded.

Do not mark 0025B Complete until implementation, tests, the independent documentation pass, every
acceptance criterion, and final evidence succeed. Keep Compiler 0005B–0005E and 0006 Draft
without detailed task specifications.
```

## Local decisions

- Broadened the existing directional EXPAND contract rather than adding `broadcastTo` or another
  semantic spelling.
- Reused exact source and target Dimensions as the later proof inputs; no Model constraint object
  is needed.
- Selected immediate failure only for a fully static contradiction. Any aligned pair involving an
  unresolved Dimension is representable without choosing a bound result because the caller's
  exact target Shape is already retained.
- Kept leading target axes outside the aligned-pair obligation because they expand from implicit
  source singletons.
- Kept layout conservative: current static resolved geometry remains exact, while every
  unresolved compatibility or geometry case stays unresolved.
- Assigned compiler adoption to existing Draft Compiler 0005B because that task already owns
  binding-dependent sum-to-Shape and the gradient family that consumes expansion.
- Left `docs/planning/modules/model/capabilities.md` unchanged because it already selects expand
  and assigns graph-wide/binding proof outside Model; this task changes the completeness of an
  existing semantic contract, not the selected capability inventory.

## Known limitations

- At task completion, Model construction will represent binding-aware EXPAND, but current Compiler
  `LayoutInference` will not yet accept and retain its unresolved aligned obligation.
- Current autograd/preflight support does not yet close binding-dependent EXPAND and inverse
  sum-to-Shape cases. Compiler 0005B owns that adoption.
- No current concrete dimension binding, execution, materialization, or backend implementation is
  introduced.

## Validation evidence

- The implementation context
  `/root/implement_model_0025b_expand` ran the focused command exactly as specified. It passed
  22 tests: 14 `TensorExpandExpressionTest` tests plus 8
  `ShapeTransformSemanticsTest` tests, with zero failures, errors, or skipped tests.
- After executable Java stabilized, that context ran `./gradlew :modules:model:test` exactly once.
  The final Model report passed 1,019 tests across 127 suites with zero failures, errors, or
  skipped tests.
- The separate documentation context
  `/root/implement_model_0025b_expand/docs_0025b` reused those successful Java results and did not
  repeat either test command. It ran `./gradlew :modules:model:javadoc`; compilation and Javadoc
  completed with `BUILD SUCCESSFUL`.
- The Tensor API example was compiled with Java 26 against final Model classes and executed. Its
  exact output matched the documented Shape, `[0, 0, 1]` strides, offset, layout kind, span, view,
  EXPAND provenance, input identity, exact dynamic target identity, unresolved layout, and absent
  host storage.
- `javap` confirmed exactly 200 public Tensor methods and exactly two public `expand` overloads.
  It also confirmed that `TensorExpandExpressions` remains final and field-free with exactly six
  declared methods, `ShapeTransformKind` remains ordered `RESHAPE`, `EXPAND` with the same
  signature method, and `TargetShapeAttrs` remains the one-component record surface.
- Generated Javadoc inspection confirmed the right-aligned source-one-or-source-equal obligation,
  fully static rejection including zero-to-one, exact-target ownership, no Model binding or
  deferred constraint, and binding-dependent unresolved-layout wording. The package-private
  helper source was inspected directly because the standard public Javadoc task does not publish
  package-private types.
- `python3 /tmp/validate_synaptik_markdown.py` passed with 12 Markdown files and 695
  repository-local links checked. Heading anchors, relative links, fences, final newlines, and
  Markdown structure passed.
- `git diff --check` passed. The union of tracked and untracked changed paths is exactly the 12
  authorized paths: four Model production/Javadoc files, the focused Model test, and seven
  documentation/planning files.
- Source and test review confirmed the complete aligned matrix: structural equality, static
  source singleton, named Dimensions, expression Dimensions, constrained unknowns,
  static/unresolved, unresolved/static, unresolved/unresolved, unresolved leading axes, fully
  static contradictions, and ordinary zero behavior. It also confirmed exact target identity,
  resolved stride/offset/view behavior, unresolved binding-dependent layout, metadata,
  provenance index zero, freshness, input non-mutation, failure messages/order, no-ID failures,
  one-ID successes, and terminal identifier exhaustion.
- Compiler source review confirmed current `LayoutInference` still rejects a non-equal,
  non-static-source-singleton unresolved EXPAND pair. Model creates no
  `DeferredGraphConstraint`; Draft Compiler 0005B owns the later
  `AnyOf(DimensionEqual(source, 1), DimensionEqual(source, target))` adoption. Existing
  `SUM_TO_SHAPE` behavior is unchanged.
- Planning review confirmed Model 0025B Complete in this task, the Model master plan, and the
  roadmap. Compiler 0005B–0005E and 0006 remain Draft, and no detailed task file exists for any of
  those rows.

Reasoned no-change conclusions:

- `docs/planning/modules/model/capabilities.md` remains unchanged because EXPAND was already a
  selected Model capability and its ownership boundary was already correct; this task improves
  representability within that capability rather than selecting a new one.
- The Training API remains unchanged because this task adds no training request, differentiable
  role, saved value, optimizer, or session behavior. No binding-dependent gradient claim is made.
- Related reshape, expand-dimensions, squeeze, tile, ordinary broadcasting, and sum-to-Shape
  contracts remain unchanged because this task alters only the existing one-input directional
  EXPAND validation rule.
- Runtime, Prepare, Backend, Engine, and tracing documentation remains unchanged because no
  prepared state, binding implementation, materialization, alias guarantee, lowering, execution,
  or trace event changed.
- `ARCHITECTURE.md`, focused architecture pages, ADRs, and architecture tests remain unchanged
  because Model and Compiler ownership, module boundaries, and dependencies did not change.
- Backend-conformance and integration tests remain unchanged because this task constructs Model
  metadata only and adds no executable backend or end-to-end behavior.
- Gradle, Java 26 configuration, dependency declarations, and all other modules remain unchanged
  because no build or cross-module contract changed.

## Implementation notes

- Changed only `TensorExpandExpressions.validateExpansion` executable logic: it now rejects an
  unequal aligned pair only when both dimensions are static and the source extent is not one.
  Validation order, diagnostics, allocation point, and every other helper statement remain
  unchanged.
- Expanded the focused test to cover all unresolved Dimension pairings, leading axes, static and
  zero contradictions, exact target and layout state, metadata/provenance, input state, validation
  ordering, messages, and Tensor-ID effects.
- Finalized all four affected production Javadocs, the Tensor API executable example and
  summaries, the Compile API non-adoption boundary, glossary distinctions, and planning status.
- Preserved exactly two public expand overloads, the 200-method Tensor surface, the six-method
  field-free helper, enum order/signature, record surface, exact target ownership, one-output
  provenance, resolved layout behavior, and all out-of-scope layers.
- Added no new public or semantic spelling, type, predicate, constraint, binding, compiler
  behavior, gradient rule, materialization, lowering, execution, dependency, or build change.

## Completion summary

Completed binding-aware Model expansion within the exact 12 authorized paths. Any aligned pair
containing an unresolved Dimension is now representable with the exact target Shape and unresolved
layout, while structural equality, static source-singleton acceptance, right alignment, leading
axes, and fully static rejection remain intact. Model retains the later
source-one-or-source-equal requirement without creating a constraint or binding dimensions.

Files changed: the exact four Model production/Javadoc files, one focused Model test, and seven
documentation/planning files listed under [Affected files](#affected-files).

Validation: the focused 22-test command and final 1,019-test/127-suite Model run passed in the
implementation context; the independent documentation context passed Model Javadoc, the compiled
and executed API example, generated-Javadoc inspection, Markdown with 695 local links, bytecode
surface checks, exact 12-path scope, planning-status checks, and `git diff --check` without
repeating successful Java tests.

Documentation review: all four affected Javadocs, Tensor API, Compile API, glossary, this task,
both relevant master plans, and roadmap required updates. Capabilities, Training/Runtime APIs,
related operation contracts, architecture/ADRs/tests, conformance/integration, Gradle/Java 26,
dependencies, and other modules required no change for the reasons recorded above.

Unresolved issues: None within Model task 0025B.

Required follow-up: Draft Compiler 0005B must adopt the occurrence-owned
source-one-or-source-equal predicate and align compiler inference, preflight, and gradient
construction. Compiler 0005B–0005E and 0006 remain Draft without detailed task specifications.

Status: Complete
