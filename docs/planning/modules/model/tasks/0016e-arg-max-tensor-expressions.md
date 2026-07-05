# Task 0016E: Arg-Max Tensor Expressions

## Status

Complete

## Goal

Expose the implemented `ARG_MAX` semantics through public, backend-independent Tensor expressions
that select logical indices along exactly one input axis. Public convenience overloads use
`FIRST_INDEX`, while the complete overload retains an explicit caller-supplied tie policy. Every
successful call returns a fresh storage-free, non-differentiable `INT64` Tensor with a locally
derived Shape, exact `ArgMaxAttrs`, and one-input provenance.

This task constructs index-producing expression metadata only. It does not compare values, select
an actual index, define NaN or empty-domain behavior, create gradient rules, capture a graph, or
report backend support.

## Scope

- Add exactly three public instance methods to `Tensor`: axis, axis-with-`keepDimensions`, and
  axis-with-`keepDimensions`-and-`tiePolicy` overloads for `argMax`.
- Add one package-private final `TensorArgMaxExpressions` helper with one construction entry and
  private input-validation, result-shape, and common-construction methods.
- Accept all current numeric input types: `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, and `INT64`.
- Reject `BOOL` without numeric truthiness, conversion, or cast insertion.
- Require a non-null explicit `ArgMaxTiePolicy` at the helper boundary. Convenience overloads
  supply `ArgMaxTiePolicy.FIRST_INDEX`; they do not make `ArgMaxAttrs` itself default a policy.
- Normalize each positive or negative caller axis exactly once through `Shape.normalizeAxis`.
- Remove the normalized axis when `keepDimensions` is false. When true, replace only that
  dimension with `StaticDimension(1)`. Retain every unaffected Dimension reference.
- Produce exact `INT64` results with `requiresGrad=false`, unresolved layout, no label, and no host
  storage.
- Construct exact `Operation(AggregateReductionKind.ARG_MAX,
  new ArgMaxAttrs(normalizedAxis, keepDimensions, tiePolicy))` semantics.
- Record exact one-input provenance and delegate identity allocation exactly once to
  `TensorFactory.createDerived`.
- Update the exact Tensor reflection test and add one focused arg-max expression test.
- Finalize Javadocs, Tensor API, Compile API, glossary, task evidence, master plan, and roadmap
  through the required independent documentation pass.

## Out of scope

- full/all-axes arg-max, zero-argument `argMax()`, a negative all-axis sentinel, or retained
  dimensions for a full reduction
- output data-type selection, `INT32` index output, native index width, platform-dependent index
  type, or caller-supplied result descriptor
- multiple axes, an axis collection, named axes, flattened arg-max, top-k, arg-min, sort, or
  another index-selection family
- value or storage access, comparison, actual index selection, allocation, copying,
  materialization, aliasing, mutation, or output storage
- NaN, signed-zero, infinity, equality, comparison-order, or empty-axis policy
- an implicit tie-policy default inside `ArgMaxAttrs`, nullable policy, policy inference, or
  backend-specific policy substitution
- gradient eligibility propagation, gradient values/rules, straight-through estimators, autograd,
  optimizer, or training execution
- changes to `AggregateReductionKind`, `ArgMaxAttrs`, `ArgMaxTiePolicy`, ordinary reduction helper,
  numeric/boolean reduction behavior, Operation foundations, Shape/Dimension, descriptor,
  provenance, factory, or another existing contract
- aliases, public helpers/builders, static Tensor forms, labels, or overloads beyond the exact
  three
- graph capture, compiler inference/optimization, planning, prepare, runtime, backend, tracing,
  ONNX, dependencies, Gradle, architecture, another module, or a task-0016F specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016B](0016b-sum-mean-and-product-tensor-expressions.md)
- [Task 0016C](0016c-min-and-max-tensor-reduction-expressions.md)
- [Task 0016D](0016d-boolean-all-and-any-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes exactly three fluent axis-only overloads:
`argMax(axis)`, `argMax(axis, keepDimensions)`, and
`argMax(axis, keepDimensions, tiePolicy)`. The first two select `FIRST_INDEX`; the complete form
supports explicit `FIRST_INDEX` and `LAST_INDEX`. Negative axes are accepted, BOOL is rejected,
numeric floating and integral inputs are supported by the CPU route, and the public result uses
`INT64` logical indices. Legacy tests exercise both policies, retained and removed axes, and all
three floating types.

The new model retains that public capability but requires the complete helper boundary and
`ArgMaxAttrs` to receive a non-null explicit policy. Legacy nullable-policy fallback, operation
traits, result-kind metadata, storage handling, value comparison, runtime graph state, lowering,
and kernels are not copied. Later backend and conformance work owns actual selection, NaN/equality
behavior, empty-axis handling, and route support.

## Architecture constraints

- `Tensor` remains public mutable API state, not IR or an executable value.
- `TensorArgMaxExpressions` performs deterministic local input validation and Shape derivation
  only; it never reads values/storage, traverses provenance, captures a graph, compares maxima, or
  queries backends.
- `ARG_MAX` always uses `ArgMaxAttrs`; it never uses `NoOperationAttrs` or `AxisReductionAttrs`.
- No full form or negative all-axis sentinel is introduced.
- The caller axis is normalized against the exact input Shape before constructing attributes.
  Stored axes are always non-negative.
- Shape derivation creates no symbolic constraint, broadcast plan, stride, layout,
  materialization, or compiler state.
- Unaffected Dimension references are retained. A retained selected axis becomes a new
  `StaticDimension(1)`; rank-one removal produces canonical `Shape.scalar()`.
- Dynamic dimensions and static zero extents remain structurally valid. No maximum index for an
  empty selected axis is defined.
- All five numeric input types are accepted without promotion or conversion. BOOL is rejected.
- Every result has fixed `DataType.INT64`, `requiresGrad=false`, unresolved layout, no label or
  storage, fresh identity, and exact one-input provenance.
- Convenience defaults belong only to the public overload mapping. `ArgMaxAttrs` continues to
  require an explicit policy and remains unchanged.
- The ordinary `TensorReductionExpressions` helper remains unchanged because `ARG_MAX` has no
  full form, uses different attributes, fixes a different result type, and never propagates
  gradient eligibility.
- Package direction is `model.tensor -> model.operation.reduction`, operation, datatype, and
  shape. The reduction package must not import Tensor.
- Stop if implementation requires a full form, changed semantic attributes, storage access,
  comparison/empty-domain policy, gradient rule, graph behavior, dependency, or architecture
  decision.

## Package impact

Existing packages used:

- `model.tensor` owns the public surface, local validation, Shape/result-descriptor construction,
  provenance, and derived factory seam.
- `model.operation` supplies `Operation`.
- `model.operation.reduction` supplies `ARG_MAX`, `ArgMaxAttrs`, and `ArgMaxTiePolicy`.
- `model.datatype` supplies numeric-category checks and fixed `INT64` result identity.
- `model.shape` supplies Shape, Dimension, StaticDimension, scalar creation, and axis normalization.

No package is added.

Type placement:

- `io.github.pho001.synaptik.model.tensor.Tensor` — exact public fluent arg-max overloads.
- `io.github.pho001.synaptik.model.tensor.TensorArgMaxExpressions` — package-private local
  validation, Shape derivation, typed descriptor/operation/provenance construction, and factory
  delegation.
- `TensorArgMaxExpressionTest` — same-package focused helper and public-behavior test.

## Required contract

### Public Tensor surface

Add exactly:

```java
public Tensor argMax(int axis)
public Tensor argMax(int axis, boolean keepDimensions)
public Tensor argMax(
        int axis,
        boolean keepDimensions,
        ArgMaxTiePolicy tiePolicy)
```

Map and delegate exactly once:

| Tensor method | Exact helper arguments |
|---|---|
| `argMax(axis)` | `this, axis, false, ArgMaxTiePolicy.FIRST_INDEX` |
| `argMax(axis, keepDimensions)` | `this, axis, keepDimensions, ArgMaxTiePolicy.FIRST_INDEX` |
| `argMax(axis, keepDimensions, tiePolicy)` | `this, axis, keepDimensions, tiePolicy` |

Convenience methods call the helper directly, not another public overload. All methods are public
instance, non-static, non-synchronized, and return the helper's exact result.

The number of declared public Tensor methods becomes exactly 70. The public method-name set gains
exactly `argMax`.

### Package-private helper shape

Create exactly one package-private final non-record class with zero fields/nested types, one
private constructor, one package-private static entry, and three private static methods:

```java
final class TensorArgMaxExpressions {
    private TensorArgMaxExpressions() {
    }

    static Tensor apply(
            Tensor input,
            int axis,
            boolean keepDimensions,
            ArgMaxTiePolicy tiePolicy)

    private static void validateNumericInput(Tensor input)

    private static Shape reduceShape(
            Shape inputShape,
            int normalizedAxis,
            boolean keepDimensions)

    private static Tensor create(
            Tensor input,
            Shape shape,
            ArgMaxAttrs attrs)
}
```

Add no overload, additional method, field, nested type, public/protected member, cache, registry,
strategy, service, allocator, shared-helper modification, or test hook.

The private Shape method deliberately keeps this task isolated from the completed ordinary-
reduction helper. It must reproduce the same observable Shape contract but must not call, widen,
rename, or otherwise modify `TensorReductionExpressions`. A future extraction would require a
separate focused refactor after another concrete consumer justifies changing both completed
helpers.

### Validation and construction order

`apply` performs exactly:

1. `Objects.requireNonNull(input, "input")`.
2. `Objects.requireNonNull(tiePolicy, "tiePolicy")`.
3. Call `validateNumericInput(input)` exactly once.
4. Read exact `input.descriptor().shape()`.
5. Call `inputShape.normalizeAxis(axis)` exactly once.
6. Call private `reduceShape(inputShape, normalizedAxis, keepDimensions)` exactly once.
7. Construct one `ArgMaxAttrs(normalizedAxis, keepDimensions, tiePolicy)`.
8. Call private `create(input, shape, attrs)` exactly once and return its exact result.

Null policy therefore fails before input type or axis inspection. Input type fails before axis
validation. Pre-factory failures consume no Tensor identity.

`validateNumericInput` accepts exactly data types for which `isFloating()` or `isIntegral()` is
true. Otherwise it throws `IllegalArgumentException` with exact message
`input must have a numeric data type, but was <dataType>`. It performs no promotion, conversion,
storage check, or backend query.

### Result Shape derivation

For removal, allocate one `Dimension[]` of rank minus one, copy every nonselected Dimension in
order by exact reference, and call `Shape.ofDimensions` once. Rank one becomes canonical scalar.

For retention, copy dimensions to one same-rank array, replace exactly the selected entry with
`new StaticDimension(1)`, preserve every other reference, and call `Shape.ofDimensions` once.

Do not require static Shape, inspect element count or selected extent, mutate the dimension list,
or preserve input layout. Scalar input has no valid axis and fails through existing Shape behavior.

### Common construction

`create` receives validated values and constructs in exact order:

1. one `TensorDescriptor(DataType.INT64, shape, Optional.empty(), false)`;
2. one `Operation(AggregateReductionKind.ARG_MAX, attrs)` retaining the exact attrs reference;
3. one `TensorProvenance(operation, List.of(input))` retaining the exact input reference;
4. one `TensorFactory.createDerived(descriptor, Optional.empty(), provenance)` call.

It performs no further semantic validation. Every successful call is fresh, including repeated
calls with equal arguments. The input Tensor, descriptor, Shape, dimensions, label, provenance,
storage association, and contents remain unchanged. Output label and storage are absent.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorArgMaxExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorArgMaxExpressionTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless inconsistent: Training API, capabilities,
`TensorReductionExpressions`, existing reduction tests, related foundational and reduction
Javadocs/tests, focused architecture, ADRs/tests, conformance/integration tests, and Gradle
configuration.

## Maximum scope

At most two production files, two tests, and six documentation/planning files: ten paths total.
Tensor and TensorTest may change only for the exact three overloads, imports/Javadocs, reflection
expectations, signatures, and non-synchronization assertions. Stop beyond this scope or if the
ordinary helper, another semantic type, full form, output-type option, empty-axis policy, storage
access, graph/compiler behavior, or gradient rule is needed. Do not create task 0016F.

## Javadoc requirements

- Update Tensor type Javadoc only as needed for index-producing aggregate construction.
- Document every public method with axis-only scope, convenience/default versus explicit policy,
  numeric eligibility, normalization, result Shape, fixed INT64/false-gradient result, unresolved
  layout, freshness, storage absence, provenance, deferred comparison/NaN/empty-domain/gradient/
  execution behavior, all parameters, return, and failures.
- Explain rank-one removal to scalar and retained extent one.
- Document helper type, constructor, entry, and private methods with validation/construction order,
  exact policy/input/Shape ownership, index result facts, identity effects, and failures.
- Explain acceptance of zero/dynamic extents without claiming a valid maximum index for an empty
  selected axis.
- Explain that public convenience defaults do not weaken the explicit `ArgMaxAttrs` contract.
- Review related Javadocs and record reasoned no-change conclusions or stop.

## Acceptance criteria

- Exactly three public overloads and the exact four-method helper are added; Tensor has exactly 70
  declared public methods and its name set gains only `argMax`.
- Every public method is one exact direct helper delegation with correct keep-dimension/policy
  arguments; convenience forms use `FIRST_INDEX`.
- Exact null-policy/type/axis validation order and messages hold.
- All three floating and two integral types succeed; BOOL fails without conversion.
- Both tie policies are retained by exact enum reference in exact `ArgMaxAttrs`.
- Axis results normalize once, remove/retain correctly, produce canonical scalar from rank one,
  and preserve unaffected static/dynamic Dimension references.
- Results are exact INT64 with false eligibility, empty layout/label/storage, fresh identity,
  exact attributes, and one-input provenance.
- Inputs remain unchanged; no comparison, actual index, NaN/equality/empty-axis policy, gradient
  rule, graph/backend/build, or architecture behavior is added.
- Existing ordinary aggregate contracts, helper, and tests remain unchanged.
- Focused/aggregate tests, Javadoc, root tests, reflection/javap/import/bytecode/scope and docs
  validation pass.
- A separate clean-context docs agent finalizes permitted Javadocs/APIs/glossary/planning and
  records no-change conclusions.
- 0016E becomes Complete only after both passes; 0016F remains Draft without a specification.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorArgMaxExpressionTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test covers exact public/helper shape and bytecode delegation; all five numeric input
types; BOOL rejection; default and explicit tie policies; exact attributes/provenance; positive/
negative and invalid axes; removal, retention, rank-one scalar, static/dynamic/zero Dimension
behavior and reference retention; fixed INT64/false eligibility; freshness/nesting; unchanged
input descriptor/Shape/layout/label/provenance/storage/contents; null/type/axis precedence; ID
effects; and identifier exhaustion.

Manually inspect `javap -p -c -s`, reflection, imports, and source for exact overload descriptors,
four-method helper shape, one direct delegation, one null-policy/type validation path, one axis
normalization, structural Shape derivation, exact attrs/policy reference, fixed descriptor, one
createDerived call, and absence of value/storage access, comparison, full form, sentinel,
multi-axis, element-count, output-type option, cast, gradient, ordinary-helper change, cross-layer
type, registry/service, dependency, or build change. Validate generated Javadoc, APIs/glossary,
examples, links/anchors/fences/whitespace, exact ten paths, synchronized statuses, and absence of
task 0016F.

## Dependencies

- 0001 supplies floating/integral/BOOL DataTypes and non-differentiable INT64 metadata.
- 0002 supplies Dimensions, scalar Shape, Shape creation, and axis normalization.
- 0006 supplies Operation; 0007 supplies TensorDescriptor.
- 0011–0013 supply Tensor, centralized identity allocation, provenance, and createDerived.
- 0016A supplies `ARG_MAX`, `ArgMaxAttrs`, and explicit tie-policy semantics.
- 0016B–0016D establish adjacent aggregate public and Shape conventions that must remain unchanged.

## Follow-up tasks

- 0016F remains Draft for masked sum/mean; detailed planning decides composition versus a
  dedicated semantic form.
- 0016G–0016J remain Draft for cumulative sum and softmax families.
- Compiler tasks own capture/canonicalization; backend/conformance tasks own value comparison,
  NaN/equality and empty-axis behavior, lowering, storage, kernels, and route support.

Do not create a detailed follow-up specification here.

## Architecture impact

Expected impact: None. This task composes model-owned Tensor, operation, descriptor, Shape, and
provenance contracts without executable or cross-layer state. Stop if architecture change is
required.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0002/0006/0007/0011/0012/0013/0016A/0016B/0016C/
0016D/0016E, Tensor API, Compile API, Training API, glossary, current DataType/Dimension/
StaticDimension/Shape/TensorDescriptor/Tensor/TensorFactory/TensorProvenance/Operation/
AggregateReductionKind/ArgMaxAttrs/ArgMaxTiePolicy contracts and tests, ordinary reduction helper/
tests, and Java 26 Gradle configuration.

Implement task 0016E exactly. Modify only Tensor.java and add TensorArgMaxExpressions.java for
production. Update only TensorTest and add TensorArgMaxExpressionTest. Add exactly argMax(axis),
argMax(axis,keepDimensions), and argMax(axis,keepDimensions,tiePolicy); each delegates once to the
new helper. Convenience forms explicitly use FIRST_INDEX.

The helper has exactly apply, validateNumericInput, reduceShape, and create. Null-check input then
tiePolicy, accept exactly floating/integral inputs, normalize the axis once, derive remove/retain
Shape locally, create exact ArgMaxAttrs and a fixed unresolved INT64/non-differentiable descriptor,
record one-input provenance, and call createDerived once. Preserve exact policy/input references
and fresh identity.

Do not add a full form, output-type option, modify the ordinary reduction helper, inspect values/
storage, define NaN/equality/empty-axis behavior, add other operations, define gradients, capture
graphs, change contracts, or add cross-layer behavior. Stop beyond ten paths or on architecture
uncertainty.

Run all task validation, then hand the actual diff/evidence to a separate clean-context docs agent
in the same change. It must inspect source/tests/generated Javadoc, finalize permitted Javadocs/
Tensor API/Compile API/glossary/planning, record related-contract/capability/Training API/
architecture no-change conclusions, and rerun validation.

Update task 0016E, model master plan, and roadmap only for planning status/evidence. Do not mark
0016E Complete until both passes succeed. Leave 0016F Draft without a specification. Do not commit
or push.
```

## Local decisions

- Match the selected legacy public overload set: axis-only convenience, axis/keep convenience,
  and one complete explicit-policy form. Do not add a full or axis/policy shorthand overload.
- Convenience forms explicitly select `FIRST_INDEX`; the semantic attribute record continues to
  reject null and never supplies a default itself.
- Accept floating and integral inputs because both represent ordered numeric values in the
  selected baseline. Reject BOOL rather than interpreting truth as numeric order.
- Fix result type to INT64 for stable model-level logical indices; do not expose backend/native
  width or legacy kernel-internal INT32 support.
- Use a dedicated arg-max helper because it differs from ordinary reductions in full-form support,
  attributes, result type, and gradient eligibility.
- Keep Shape derivation private in that helper instead of widening or changing the completed
  ordinary-helper contract. A future shared extraction requires a separate focused refactor when
  another concrete consumer justifies it.
- Zero and dynamic extents are structurally accepted; numerical index validity is deferred.

## Known limitations

- No value comparison or selected index, only semantics and provenance.
- Exactly one axis and fixed INT64 output only.
- Empty-axis, NaN, equality, and backend route behavior are unimplemented.
- No full arg-max, capture, canonicalization, ONNX/backend support, or execution.

## Validation evidence

Planning reviewed the architecture contract and focused architecture documentation;
documentation/planning rules; roadmap; model capabilities/master plan; completed tasks 0001, 0002,
0006, 0007, 0011, 0012, 0013, and 0016A through 0016D; current Tensor, datatype, operation,
reduction attributes/policy, descriptor, Shape, provenance, factory, ordinary-helper, and focused
test contracts; Tensor/Compile/Training APIs; glossary; and Java 26 Gradle configuration.

The legacy branch was read directly. It confirms the three axis-only overloads, default
`FIRST_INDEX`, explicit `FIRST_INDEX`/`LAST_INDEX`, negative axes, retained/removed result Shapes,
BOOL rejection, numeric CPU input routes, fixed public INT64 results, and execution tests across
the three floating types. Nullable-policy fallback, operation traits, comparison, storage,
runtime graph state, lowering, and kernels are excluded or reassigned.

Planning selected three overloads, one dedicated four-method helper, one reflection-test update,
and one new focused test. Existing public/foundational contracts suffice; no existing helper,
package, dependency, foundation, or architecture change is required.

Planning validation:

- `git diff --check` passed. A targeted trailing-whitespace scan found no matches in the three
  changed planning paths, including this untracked task file.
- The canonical section scan found every required task section. The specification contains exact
  public/helper shapes, default/explicit policy mapping, numeric/result validation and order,
  package placement, ten-path limit, acceptance criteria, validation, clean-context implementation/
  documentation handoff, decisions, limitations, evidence, and completion placeholders.
- The local Markdown target checker resolved 164 links across this task, the model master plan,
  and roadmap with zero missing files. Fence validation found balanced Markdown fences in all
  three paths; none of the changed links uses a heading anchor.
- Status inspection found task 0016E `Ready` in this specification, its linked master-plan row,
  roadmap frontier, and roadmap row. Task 0016F remains `Draft`, and no task-0016F specification
  exists.
- Package/scope review found no new package and exactly three planning paths changed. No Java,
  test, API, glossary, Gradle, architecture, AGENTS, or other-module path changed.

Implementation and independent documentation validation:

- Implementation context `/root/implement_model_0010` changed exactly the four authorized Java/
  test paths: `Tensor.java`, the new package-private `TensorArgMaxExpressions`, `TensorTest`, and
  the new same-package `TensorArgMaxExpressionTest`. It added the exact three overloads and direct
  mappings, the exact four-method/state-free helper, the Tensor reflection expectations, and a
  focused seven-test suite after the independent documentation review identified and returned an
  initial focused-test coverage gap. No other implementation path, package, dependency, build
  file, or cross-layer behavior changed.
- Clean documentation context `/root/review_model_0016c_docs` independently reread the architecture
  contract; focused current-architecture, overview, lifecycle, module-boundary, and dependency
  explanations; documentation workflow and General, API/Javadoc, Planning, and Example profiles;
  planning guide and roadmap; model capabilities/master plan; tasks 0001, 0002, 0006, 0007, 0011,
  0012, 0013, and 0016A through 0016E; Tensor, Compile, and Training API references; glossary;
  actual final source/tests and complete diff; related datatype, Shape, descriptor, factory,
  provenance, operation, reduction, policy, and ordinary-helper contracts/tests; generated
  Javadoc and XML reports; and Java 26 Gradle configuration. It inspected behavior and artifacts
  directly rather than relying on the implementation handoff.
- The documentation pass found the initial four-test arg-max suite insufficient for the task's
  explicit acceptance contract. It paused completion and returned only
  `TensorArgMaxExpressionTest` to the implementation context. The corrected seven-test suite adds
  exact method modifiers/helper signatures, no-ID-consumption validation precedence, scalar-axis
  failure, zero/dynamic/static Shape behavior, complete input metadata/storage/content
  immutability, and identifier exhaustion while retaining all type, policy, descriptor,
  provenance, freshness, and nesting coverage.
- The documentation pass finalized all three public `Tensor.argMax` Javadocs and every
  `TensorArgMaxExpressions` type/member contract. They now state axis-only scope, positive/
  negative normalization, removed/retained Shape and exact Dimension-reference behavior,
  FIRST_INDEX convenience versus explicit non-null policy, floating/integral eligibility, BOOL
  rejection, fixed INT64 false-gradient result, unresolved layout, freshness, storage absence,
  one-input provenance, validation precedence and identity effects, and deferred comparison, NaN,
  signed-zero, infinity, equality, empty-axis, gradient, compiler, backend, and execution behavior.
  The Tensor type Javadoc now includes index-producing aggregate construction.
- `docs/api/tensor-api.md` now documents the three-method arg-max surface in its purpose, mental
  model, current expression summary, semantic-family status, planned boundary, failure summary,
  and a focused method/policy table. It explains exact numeric eligibility, fixed result facts,
  local Shape derivation, exact attributes/provenance, freshness, validation order, and deferred
  numerical/executable behavior. `docs/api/compile-api.md` includes arg-max as current expression
  input while keeping compiler capture, reduction inference/canonicalization, artifacts, backend
  ownership, and execution planned. `docs/glossary.md` synchronizes aggregate reduction, arg-max
  tie policy, normalized axis, Tensor, operation-family, and Tensor-versus-graph status. No new
  reusable project term was needed. No existing example code changed, so Example-format compile/
  run validation was not applicable.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorArgMaxExpressionTest --rerun-tasks` — `BUILD
  SUCCESSFUL`; the XML report contains 7 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorTest --rerun-tasks` — `BUILD SUCCESSFUL`; the XML
  report contains 14 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --rerun-tasks` — `BUILD SUCCESSFUL`; 52 XML suites contain 396
  tests with zero failures, errors, or skips.
- `./gradlew :modules:model:javadoc --rerun-tasks` — `BUILD SUCCESSFUL`. Generated public Javadoc
  contains all three overloads with rendered policy, input, result, Shape, provenance, failure,
  identity-effect, and deferred-behavior contracts. The package-private helper is intentionally
  absent from public generated pages and was reviewed completely in source.
- `./gradlew test --rerun-tasks` — `BUILD SUCCESSFUL`; the repository lifecycle completed 36
  actionable tasks with no failing task.
- The first post-documentation Gradle command group stopped before task execution because the
  restricted sandbox could not open the user Gradle-distribution lock file. The approved rerun of
  every required command passed with the results above; this environmental failure did not skip
  validation.
- `javap -p -c -s` confirmed the exact three public descriptors, one direct helper delegation per
  overload, false and `FIRST_INDEX` in the axis-only form, exact caller retention flag plus
  `FIRST_INDEX` in the two-argument form, and exact caller policy in the complete form. It also
  confirmed the helper's private zero-argument constructor, exact package-private `apply` plus
  three private methods, input/policy/type/axis order, one normalization, structural removal/
  retention, exact `ArgMaxAttrs`, fixed INT64/false descriptor, one-input provenance, and one
  `createDerived` call. Verbose bytecode inspection found no synthetic helper member.
- Reflection tests confirmed 70 declared public Tensor methods, exact new method-name set and
  overload modifiers/signatures, helper final/package-private shape, zero fields/nested types, and
  four declared methods. Focused behavior covers all five numeric types, BOOL rejection, both
  policies, convenience mappings, positive/negative/invalid/scalar axes, removal/retention,
  static/dynamic/zero extents and Dimension identity, exact descriptor/attributes/provenance,
  freshness/nesting, input immutability, validation precedence/messages, no pre-factory ID
  consumption, and identifier exhaustion.
- Production import/source/bytecode inspection found only the permitted tensor, datatype,
  operation, reduction, shape, and JDK contracts. It found no value/storage access, comparison,
  element-count/static-shape demand, full form, negative all-axis sentinel, multiple-axis API,
  output-type option, cast, gradient rule, graph/compiler/planning/runtime/prepare/backend type,
  registry, service, dependency, or build behavior. A zero-diff check confirmed
  `TensorReductionExpressions`, `TensorNumericReductionTest`, and `TensorBooleanReductionTest`
  remain unchanged.
- The local Markdown target-and-GitHub-heading checker resolved all 250 links in the six changed
  documentation/planning paths with zero errors. Markdown fences are balanced; terminology and
  current/planned status scans found no stale arg-max claim; targeted whitespace inspection and
  `git diff --check` passed.
- Final scope is exactly the authorized ten paths: two production files, two tests, Tensor API,
  Compile API, glossary, this task, model master plan, and roadmap. Task 0016E is synchronized as
  Complete. Task 0016F remains Draft, and no task-0016F specification exists. No commit or push
  occurred.
- `docs/api/training-api.md` remains accurate unchanged because INT64 arg-max results are fixed
  non-differentiable and this task adds no gradient value/rule, autograd, optimizer, session, or
  executable training behavior. The capability baseline already inventories axis-only arg-max,
  first/last tie policy, and layered model/public versus executable support, so it required no
  status edit.
- `ARCHITECTURE.md`, focused architecture explanations, ADRs, architecture tests, backend-
  conformance tests, and integration tests remain accurate unchanged because the work stays inside
  model-owned Tensor expression metadata and changes no module boundary, dependency, lifecycle,
  backend behavior, or end-to-end execution. Java 26 Gradle configuration remains accurate
  unchanged because no build, dependency, preview, incubator, or toolchain behavior changed.
- `DataType`, `Dimension`, `StaticDimension`, `Shape`, `TensorDescriptor`, `TensorFactory`,
  `TensorProvenance`, `Operation`, `AggregateReductionKind`, `ArgMaxAttrs`, and `ArgMaxTiePolicy`
  contracts and Javadocs remain accurate unchanged. The new helper composes their existing numeric
  category, immutable Shape, fixed non-differentiable INT64 descriptor, central-ID, typed-operation,
  explicit-policy, and provenance contracts without changing them. The ordinary reduction helper
  and numeric/boolean reduction tests/Javadocs remain accurate and byte-for-byte unchanged.

## Implementation notes

- Added exactly three fluent axis-only `argMax` overloads as direct delegations to one dedicated
  package-private helper. The convenience forms explicitly supply `FIRST_INDEX`; the complete form
  retains the caller's exact policy.
- Added exact numeric validation, one axis normalization, structural one-axis removal/retention,
  fixed unresolved INT64 false-gradient descriptors, exact `ARG_MAX` attributes, and one-input
  provenance without value comparison or cross-layer behavior.
- Added a seven-test focused suite after independent review identified the initial coverage gap;
  it now covers exact surface, types, policies, Shapes, validation/ID effects, freshness, input
  immutability, and exhaustion.
- Finalized affected Tensor/helper Javadocs, Tensor API, Compile API, glossary, task evidence,
  model master plan, and roadmap without adding executable index selection.

## Completion summary

- Completed changes: Implemented and documented numeric single-axis arg-max Tensor expression
  construction with explicit tie semantics, fixed INT64 non-differentiable results, exact reduced
  shapes, and one-input provenance.
- Files changed or created: Exactly two production Java files, two tests, Tensor API, Compile API,
  glossary, this task, model master plan, and roadmap.
- Tests and validation: Focused suites 7/7 and 14/14; all 396 model tests across 52 suites; model
  Javadoc; root tests; bytecode/reflection/import/source/ordinary-helper checks; 250 Markdown link/
  anchor checks; fence/terminology/whitespace checks; exact scope/status checks; and
  `git diff --check` passed. The restricted Gradle-cache lock denial and approved passing rerun are
  recorded above.
- Documentation-agent review: Clean context `/root/review_model_0016c_docs` completed the
  independent pass using General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor API, Compile API, and glossary now describe current arg-max
  construction and its explicit model/compiler/execution boundaries. Training API, capabilities,
  architecture/ADRs/tests, conformance/integration tests, and build configuration remain accurate
  unchanged for the recorded reasons.
- Javadoc review: The three Tensor overloads, helper type/members, and Tensor type summary were
  finalized. Related foundational, descriptor, factory, provenance, operation, reduction,
  attribute, policy, and ordinary-reduction contracts remain accurate unchanged.
- Glossary impact: Synchronized existing aggregate-reduction, arg-max tie-policy, normalized-axis,
  Tensor, and operation-family entries; no new reusable term was necessary.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0016E. Task 0016F remains Draft without a detailed
  specification.

Status: Complete
