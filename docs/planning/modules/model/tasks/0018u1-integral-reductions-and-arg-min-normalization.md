# Task 0018U1: Integral Reductions and Arg-Min Normalization

## Status

Complete

## Goal

Complete the selected signed-integral reduction baseline and make minimum-index and
maximum-index reduction one coherent public model family.

Extend the existing full, one-axis, and retained-axis `SUM`, `PROD`, `MIN`, and `MAX` Tensor
expressions to exact `INT32` and `INT64` input. Preserve the input data type, use fixed-width
modular accumulation for sum and product, use signed ordering for value extrema, and define every
integral empty-domain result.

Add `argMin` with the same three overload forms as `argMax`. Atomically generalize the existing
arg-max-only attributes, tie-policy type, and package-private construction helper into shared
arg-extrema contracts. Remove the old arg-max-specific semantic type names without aliases.
Both arg families accept floating or integral input, produce fixed `INT64` indices, use explicit
first/last tie selection, and share one complete comparison and empty-axis contract.

This task constructs and documents backend-independent expression metadata. It does not read or
reduce values, execute an operation, add gradient rules, capture a graph, lower an operation, or
add backend support.

## Why this is one atomic task

Integral ordinary reductions can reuse the existing aggregate kinds and public overloads, but
`argMin` cannot be added coherently while the shared axis/retention/tie parameters are named only
for arg-max. Leaving `ArgMaxAttrs` or `ArgMaxTiePolicy` on an `ARG_MIN` occurrence would publish a
misleading permanent model contract. Adding parallel min-only values would duplicate identical
semantics and make later sort/top-K policy inconsistent.

The old policy, attributes, and helper names are therefore replaced in the same change that adds
`ARG_MIN` and `argMin`. This is a source-incompatible model API migration with no transitional
alias. The bounded 22-path scope is above the normal task guardrail only because a partial rename
would leave production source, signatures, tests, Javadocs, and public documentation unable to
compile or describe one valid state.

## Mental model

### Integral value reduction

```java
Tensor totals = int32Input.sum(1, true);
```

Conceptually records:

```text
result data type = INT32
result shape = input shape with axis 1 replaced by extent one
requiresGrad = false
operation = AggregateReductionKind.SUM
attrs = AxisReductionAttrs(1, true)
inputs = [int32Input]
arithmetic meaning = sum modulo 2^32
```

No widening or cast producer is inserted. `INT64` input uses the corresponding modulo-`2^64`
domain and produces `INT64`.

### Shared arg-extrema family

```java
Tensor firstMinimum = input.argMin(-1);
Tensor lastMaximum = input.argMax(-1, false, ArgExtremaTiePolicy.LAST_INDEX);
```

Both occurrences normalize the final axis, remove it, and produce `INT64` with false gradient
eligibility. Their only semantic difference is the exact aggregate kind:

```text
firstMinimum = ARG_MIN + ArgExtremaAttrs(axis, false, FIRST_INDEX)
lastMaximum  = ARG_MAX + ArgExtremaAttrs(axis, false, LAST_INDEX)
```

## Selected integral reduction contract

### Accepted families and result types

The existing public overloads below accept `INT32` and `INT64` in addition to their current
floating domain:

| Family | Full | Axis removed | Axis retained | Integral result type |
|---|---|---|---|---|
| Sum | `sum()` | `sum(axis)` | `sum(axis, keepDimensions)` | Exact input type |
| Product | `prod()` | `prod(axis)` | `prod(axis, keepDimensions)` | Exact input type |
| Minimum | `min()` | `min(axis)` | `min(axis, keepDimensions)` | Exact input type |
| Maximum | `max()` | `max(axis)` | `max(axis, keepDimensions)` | Exact input type |

`mean()` and both axis mean forms remain floating-only because an integral mean needs a result
type, division, rounding, and invalid-divisor contract that this task deliberately does not
invent. Masked sum and masked mean remain floating-only; this task changes only the existing
ordinary full/axis/retained-axis forms. BOOL remains eligible only for `ALL` and `ANY`.

An integral unary reduction never promotes: `INT32` input produces `INT32`, and `INT64` input
produces `INT64`. `DataTypePromotion` is unchanged because a reduction has one input data type and
does not need pairwise promotion.

### Accumulation, overflow, and ordering

Integral `SUM` and `PROD` accumulate in the exact result type:

- `INT32` sum and product are reduced modulo `2^32` and reinterpreted as signed INT32;
- `INT64` sum and product are reduced modulo `2^64` and reinterpreted as signed INT64;
- no widening, saturation, overflow exception, or hidden higher-precision accumulator is part of
  the semantics; and
- regrouping or parallel reassociation is permitted because addition and multiplication modulo a
  fixed power of two have the same final result under reassociation. A backend still must include
  every selected value exactly once.

Integral `MIN` and `MAX` compare exact signed values in the input/result type. They have no NaN,
infinity, signed-zero, tolerance, promotion, or overflow policy.

### Empty domains

Integral reductions use the algebraic identity representable by their bounded result type:

| Kind | Empty INT32 result | Empty INT64 result |
|---|---:|---:|
| `SUM` | `0` | `0L` |
| `PROD` | `1` | `1L` |
| `MIN` | `Integer.MAX_VALUE` | `Long.MAX_VALUE` |
| `MAX` | `Integer.MIN_VALUE` | `Long.MIN_VALUE` |

For a full form, the domain is empty when the input has zero logical elements. For an axis form,
the rule applies independently to each output position whose selected reduction axis has extent
zero. `keepDimensions` changes only Shape and never the identity. If another retained output axis
is itself zero, the result contains no positions on which to materialize the per-slice identity.
The same rule applies when a dynamic dimension is later bound to zero.

Model construction continues to accept static zero and dynamic extents for these ordinary
reductions. It records the selected meaning without inspecting an element count or creating an
eager constant.

All existing floating ordinary reduction behavior remains unchanged: the same floating types,
result metadata, validation, and structural acceptance continue to work, and this task does not
select a new floating accumulation, overflow, reassociation, NaN, signed-zero, or empty-domain
policy. Existing BOOL aggregate and masked-reduction behavior also remains unchanged.

## Shared arg-extrema contract

### Semantic vocabulary and atomic migration

Append `ARG_MIN` to `AggregateReductionKind` after the existing `ARG_MAX` constant. Preserve the
order and identity of every existing constant. `ARG_MIN` accepts the same one-input/one-output
structural signature as `ARG_MAX`.

Replace, with no deprecated or transitional alias:

```text
ArgMaxTiePolicy          -> ArgExtremaTiePolicy
ArgMaxAttrs              -> ArgExtremaAttrs
TensorArgMaxExpressions  -> TensorArgExtremaExpressions
```

Delete the three old production source files. No old public type, wrapper, forwarding overload,
duplicate attributes record, or compatibility bridge remains.

Create exactly:

```java
public enum ArgExtremaTiePolicy {
    FIRST_INDEX,
    LAST_INDEX
}

public record ArgExtremaAttrs(
        int axis,
        boolean keepDimensions,
        ArgExtremaTiePolicy tiePolicy) implements OperationAttrs
```

`ArgExtremaAttrs` preserves the old component order and validation behavior: reject a negative
axis first with `axis must be non-negative: <axis>`, then reject a null policy with
`NullPointerException("tiePolicy")`. It retains exact values and uses record-generated equality,
hashing, and diagnostic text.

`AggregateReductionKind.ARG_MIN` and `ARG_MAX` both pair only with `ArgExtremaAttrs`. Neither has
a full or `NoOperationAttrs` form. No other aggregate kind accepts these attributes.

### Public Tensor surface

Keep the existing three `argMax` overload descriptors, changing only the policy parameter type:

```java
public Tensor argMax(int axis)
public Tensor argMax(int axis, boolean keepDimensions)
public Tensor argMax(
        int axis,
        boolean keepDimensions,
        ArgExtremaTiePolicy tiePolicy)
```

Add the exact matching surface:

```java
public Tensor argMin(int axis)
public Tensor argMin(int axis, boolean keepDimensions)
public Tensor argMin(
        int axis,
        boolean keepDimensions,
        ArgExtremaTiePolicy tiePolicy)
```

The one-argument form removes the selected axis and supplies `FIRST_INDEX`. The two-argument form
uses the caller's retention flag and supplies `FIRST_INDEX`. The complete form retains the exact
non-null policy. Convenience methods delegate directly to the shared helper rather than another
public overload.

The declared public Tensor method count becomes exactly 130, and the public method-name set gains
only `argMin`. No axis/policy shorthand, full form, output-type choice, alias, static form, or
builder is added.

### Input, comparison, and tie semantics

Both arg families accept exactly `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, and `INT64`; BOOL is
rejected without conversion or truth ordering. There is no input promotion.

For integral input, values use ordinary signed order. For floating input, both families use this
shared total selection policy:

- a NaN is selected in preference to every non-NaN value for both `ARG_MIN` and `ARG_MAX`;
- when multiple NaNs are present, they are equal candidates and `tiePolicy` selects the first or
  last logical index;
- negative zero orders below positive zero, so `ARG_MIN` selects negative zero and `ARG_MAX`
  selects positive zero when both signs occur;
- negative infinity orders below every finite value and positive infinity orders above every
  finite value; and
- otherwise ordinary equal numeric values are ties resolved by the explicit policy.

This policy aligns arg-extrema signed-zero selection with the existing pairwise floating extrema
contract while giving NaN-producing input a deterministic index. It fixes semantic results, not
an algorithm, traversal strategy, instruction sequence, or backend route.

`FIRST_INDEX` means the smallest logical coordinate along the normalized selected axis;
`LAST_INDEX` means the largest. A logical coordinate is independent of physical offset, stride,
layout, traversal direction, or storage.

### Empty selected axes

No valid arg-extrema index exists for an empty selected axis. After axis normalization, the shared
model helper rejects a statically known selected extent of zero with exact message:

```text
arg-extrema reduction axis must be non-empty, but axis <normalizedAxis> has static extent 0
```

The failure is `IllegalArgumentException` and occurs before attributes, producer, or Tensor
identity construction. A zero on a non-selected axis remains structurally valid and may make the
result itself empty.

A dynamic or expression selected extent is accepted because model construction cannot bind its
runtime size. The semantic occurrence is valid only when that extent is later proven or bound
positive. Compiler validation and later prepared/execution bindings must reject zero before
index selection; their callable APIs and exception types remain outside this model task. No `-1`,
zero, or arbitrary sentinel result represents an empty arg-extrema domain.

This intentionally replaces arg-max's previously deferred zero-selected-axis behavior while
preserving all non-empty floating and integral arg-max construction.

### Result metadata

Both arg families:

- normalize one positive or negative caller axis exactly once;
- remove it when `keepDimensions` is false, or replace it with a new `StaticDimension(1)` when
  true;
- retain every unaffected immutable Dimension reference;
- produce exact `DataType.INT64` with `requiresGrad=false`;
- leave layout unresolved and attach no label or host storage;
- record one exact `ArgExtremaAttrs` and the selected `ARG_MIN` or `ARG_MAX` kind;
- retain exactly the receiver as the ordered sole producer input;
- use provenance output index zero; and
- create a fresh Tensor and producer for every valid call.

## Scope

- Broaden ordinary `SUM`, `PROD`, `MIN`, and `MAX` full/axis/retained forms to INT32 and INT64.
- Preserve exact integral input/result type and fixed false gradient eligibility.
- Document modular sum/product, signed min/max, and complete bounded empty-domain identities.
- Keep all mean and masked reductions floating-only and preserve current floating/BOOL behavior.
- Add `AggregateReductionKind.ARG_MIN` with the exact shared one-input/one-output signature.
- Replace `ArgMaxTiePolicy`/`ArgMaxAttrs` with shared `ArgExtremaTiePolicy`/`ArgExtremaAttrs`, with
  no alias.
- Replace the arg-max-only helper with one shared package-private arg-extrema helper.
- Add the exact three `argMin` overloads and migrate the three `argMax` policy signatures.
- Define shared numeric eligibility, floating/integral ordering, ties, empty-axis validation,
  result metadata, Shape, provenance, freshness, and failure/identity behavior.
- Update focused semantic, ordinary-reduction, arg-extrema, and Tensor surface tests.
- Finalize affected Javadocs, Tensor API, Compile API, glossary, capability baseline, task, master
  plan, and roadmap through the mandatory independent documentation pass.

## Out of scope

- integral `MEAN`, masked integral SUM/MEAN, multiple axes, an axis collection, initial values,
  caller-selected accumulation/output type, or full-reduction `keepDimensions`
- widening INT32 reductions to INT64, arbitrary precision, saturation, overflow exceptions,
  checked arithmetic, compensated summation, or execution-time reduction algorithms
- floating accumulation/empty-domain policy, boolean empty identities, changes to floating extrema
  semantics, or masked reduction policy
- full/all-axis arg-min or arg-max, flattened arg extrema, output index selection, INT32 indices,
  native index width, top-K, sort, argsort, or pooling
- nullable/default policy inside attributes, stable-sort metadata, traversal order, a NaN-ignore
  variant, or a sentinel index for an empty axis
- value/storage access, eager aggregation, eager index selection, constant folding,
  simplification, canonicalization, common-subexpression elimination, or result interning
- gradient values/rules, extrema tie-gradient distribution, autograd, backward construction,
  optimizer, or training execution
- compiler capture or implementation, dynamic-shape binding implementation, graph-wide inference,
  planning, prepare, backend support, lowering, kernels, runtime, execution, tracing, or engine
- changes to DataType, DataTypePromotion, Shape, Dimension, TensorDescriptor, TensorFactory,
  producer/provenance structure, storage, graph records, dependencies, Gradle, Java version,
  architecture rules/docs/tests, conformance/integration tests, or another module
- a detailed task 0018V or later specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model ownership of public
  Tensor, operation semantics, descriptors, and immutable provenance
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md)
- [Task 0016B](0016b-sum-mean-and-product-tensor-expressions.md)
- [Task 0016C](0016c-min-and-max-tensor-reduction-expressions.md)
- [Task 0016D](0016d-boolean-all-and-any-tensor-expressions.md)
- [Task 0016E](0016e-arg-max-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018U](0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns data-type eligibility, backend-independent numeric meaning, public Tensor
  construction, descriptor derivation, immutable operation values, and provenance.
- `OperationKind` and attributes identify stable requested semantics. They contain no Tensor,
  Shape, descriptor, storage, compiler, backend support, execution route, or runtime state.
- Ordinary reduction kinds retain their current structural signatures. Integral eligibility and
  result facts are operand-aware Tensor-construction rules, not new signature fields.
- Shared arg-extrema attributes carry only a normalized axis, retention flag, and tie policy.
  Comparison results, NaN classification, input type, and empty-axis proofs are not duplicated in
  attributes.
- Public Tensor remains mutable API state rather than IR. Every valid call creates one fresh
  storage-free result through the central factory with one immutable producer and output index
  zero.
- Compiler owns graph capture, operand revalidation, canonicalization, autograd, and backward
  construction. Backend prepare owns lowering and route selection. Model records semantic
  requirements without implementing either layer.
- Runtime hot paths do not consume Operation values.
- No registry, service locator, reflection discovery, string dispatch, dependency, or architecture
  rule is added.
- If implementation needs another operation kind, attributes component, public overload, data
  type, result type, module, or twenty-third affected path, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.reduction` — owns aggregate kind, ordinary axis
  attributes, and the generalized arg-extrema semantic values.
- `io.github.pho001.synaptik.model.tensor` — owns public overloads, local operand/Shape validation,
  result descriptors, and producer/provenance construction.
- `io.github.pho001.synaptik.model.datatype` and `model.shape` remain consumed foundations.

Packages added, moved, or removed:

- None.

Type placement:

- `AggregateReductionKind` remains the aggregate-family semantic enum and appends `ARG_MIN`.
- `ArgExtremaTiePolicy` replaces the arg-max-only policy in the reduction package because its
  first/last logical-index meaning is identical for both extrema directions.
- `ArgExtremaAttrs` replaces the arg-max-only record in the same package because both operations
  have exactly the same normalized occurrence parameters.
- `TensorArgExtremaExpressions` replaces the arg-max-only helper in `model.tensor` because input,
  Shape, descriptor, validation, and provenance construction are shared except for exact kind.
- Existing focused tests remain in their matching production packages. The current
  `TensorArgMaxExpressionTest` path is expanded to cover both families so the atomic migration
  does not add a second mostly duplicate test file.

## Required implementation contracts

### Ordinary reduction helper

Keep `TensorReductionExpressions` field-free and preserve its exact six-method surface. Do not add
an accumulator value, type helper, policy object, or result-type option.

Both entries retain this order:

1. null-check `input` with message `input`;
2. null-check `kind` with message `kind`;
3. validate the existing seven ordinary kinds;
4. apply kind-aware input validation; and
5. for an axis form only, normalize the caller axis once, derive Shape, and construct attributes.

Kind-aware input validation is:

- `SUM`, `PROD`, `MIN`, and `MAX` require floating or integral input; BOOL fails with exact message
  `input must have a numeric data type for <KIND>, but was BOOL`;
- `MEAN` remains floating-only and retains exact message
  `input must be a floating data type, but was <dataType>`;
- `ALL` and `ANY` remain BOOL-only and retain exact message
  `input must have BOOL data type for <KIND>, but was <dataType>`.

All failures before factory delegation consume no Tensor identity. Common construction continues
to preserve exact input type, Shape rules, input `requiresGrad`, unresolved layout, no label or
storage, exact one-input producer, and output index zero. Integral inputs necessarily have false
gradient eligibility.

### Aggregate kind and shared semantic values

`AggregateReductionKind` preserves all existing signatures and adds one stable shared signature
list for `ARG_MIN` and `ARG_MAX`, each accepting exact `ArgExtremaAttrs` with one input and one
output. `ARG_MAX` no longer accepts the deleted `ArgMaxAttrs` class. No ordinary or masked
signature changes.

The two new shared public types have only the exact enum constants/record components and complete
Javadocs required above. Add no factory, alias, default constructor, parsing, serialization,
comparison method, policy method, field, nested type, or compatibility adapter.

### Shared arg-extrema helper

Replace the old helper with one package-private final, field-free, non-record class with no nested
types, one private constructor, one package-private entry, and four private methods:

```java
static Tensor apply(
        Tensor input,
        AggregateReductionKind kind,
        int axis,
        boolean keepDimensions,
        ArgExtremaTiePolicy tiePolicy)

private static void validateKind(AggregateReductionKind kind)
private static void validateNumericInput(Tensor input)
private static Shape reduceShape(
        Shape inputShape, int normalizedAxis, boolean keepDimensions)
private static Tensor create(
        Tensor input,
        AggregateReductionKind kind,
        Shape shape,
        ArgExtremaAttrs attrs)
```

`apply` validates and constructs in this exact order:

1. null-check `input` with message `input`;
2. null-check `kind` with message `kind`;
3. null-check `tiePolicy` with message `tiePolicy`;
4. accept only `ARG_MIN` or `ARG_MAX`, otherwise throw exact message
   `kind must be ARG_MIN or ARG_MAX, but was <kind>`;
5. accept only floating or integral input, otherwise throw exact message
   `input must have a numeric data type, but was <dataType>`;
6. read the exact input Shape and normalize `axis` exactly once;
7. inspect only the selected Dimension's `staticSize`; reject exact zero with the specified
   empty-axis message;
8. derive removal/retention Shape while preserving unaffected references;
9. construct one exact `ArgExtremaAttrs`; and
10. construct one fixed INT64/false descriptor, exact operation, one-input producer, output index
    zero, and fresh Tensor through one central factory delegation.

Invalid kind precedes input type and axis. A null policy precedes kind support, input type, and
axis. BOOL precedes axis. Invalid axis precedes static-zero inspection. Every local failure
consumes no Tensor identity. Identifier exhaustion occurs only after all local immutable model
values are valid and retains exact existing message `tensor identifier space exhausted`.

### Public Tensor mapping

Map methods directly:

| Method | Exact kind | Retention | Policy |
|---|---|---|---|
| `argMin(axis)` | `ARG_MIN` | `false` | `FIRST_INDEX` |
| `argMin(axis, keepDimensions)` | `ARG_MIN` | caller | `FIRST_INDEX` |
| complete `argMin` | `ARG_MIN` | caller | caller |
| `argMax(axis)` | `ARG_MAX` | `false` | `FIRST_INDEX` |
| `argMax(axis, keepDimensions)` | `ARG_MAX` | caller | `FIRST_INDEX` |
| complete `argMax` | `ARG_MAX` | caller | caller |

Every method is public, instance, non-static, non-synchronized, and returns the helper's exact
result. Update Tensor type and method Javadocs for the complete shared contract and migration.

## Affected files

Expected production Java (nine paths):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/AggregateReductionKind.java`
- remove `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/ArgMaxAttrs.java`
- add `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/ArgExtremaAttrs.java`
- remove `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/ArgMaxTiePolicy.java`
- add `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/ArgExtremaTiePolicy.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorReductionExpressions.java`
- remove `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorArgMaxExpressions.java`
- add `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorArgExtremaExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected tests (six paths):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/reduction/ReductionSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorNumericReductionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorArgMaxExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`

Expected documentation and planning (seven paths):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless an inconsistency requires stopping:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataType.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataTypePromotion.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/AxisReductionAttrs.java`
- DataType, DataTypePromotion, AxisReductionAttrs, BOOL/masked reduction, and cumulative-sum tests
- `docs/api/training-api.md`
- architecture contract and focused architecture documentation/tests
- backend conformance, integration tests, Gradle, dependencies, and other modules

## Maximum scope

This task may create, modify, or remove exactly the 22 paths listed above: nine production paths,
six test paths, and seven documentation/planning paths. Renames count as their old removal and new
addition paths.

The atomic arg-extrema public migration is the documented reason for exceeding the usual
12–18-file guardrail. Do not retain an old type to reduce the path count. If implementation needs
a twenty-third path, another helper/member, a second arg-min test file, another kind, a result-type
option, or a change to a review-only file, stop and report the required expansion before editing.

## Javadoc requirements

- Finalize every changed production type/member and all affected Tensor methods with meaningful
  semantics, parameters, result ownership/nullability, Shape, descriptor, provenance, validation
  order, failure/ID effects, and deferred execution boundaries.
- Document the four integral reduction families, exact result/accumulation type, modular overflow,
  signed ordering, reassociation allowance, and full/per-slice empty identities.
- Keep mean, masked reductions, floating reductions, and BOOL aggregates explicitly unchanged.
- Document the shared policy/attributes family, atomic old-name removal, first/last logical index,
  floating NaN/signed-zero/infinity order, integral signed order, and invalid empty selected axis.
- Document every `ArgExtremaAttrs` constructor component and accessor with complete `@param`,
  `@return`, and exact `@throws` entries.
- Document helper validation/construction order and no-ID effects without claiming value access or
  execution.
- Review unchanged DataType, promotion, AxisReductionAttrs, masked/BOOL/scan, descriptor, factory,
  producer, and provenance Javadocs and record reasoned conclusions.

## Acceptance criteria

- Ordinary full/axis/retained `SUM`, `PROD`, `MIN`, and `MAX` accept INT32 and INT64, preserve the
  exact input/result type, produce false-gradient unresolved descriptors, and retain all current
  Shape/provenance/freshness behavior.
- Integral sum/product specify exact fixed-width modular accumulation with permitted
  reassociation; min/max specify signed order; no widening, saturation, overflow exception, or
  hidden promotion is introduced.
- Every full and per-axis integral empty domain has the exact zero/one/bounded-extrema identity,
  including dynamic extents later bound to zero.
- Mean and masked reductions remain floating-only. Floating ordinary reductions, BOOL aggregates,
  and masked behavior remain unchanged.
- `ARG_MIN` is appended after `ARG_MAX` and has only the exact shared one-input/one-output
  `ArgExtremaAttrs` signature.
- `ArgExtremaTiePolicy` and `ArgExtremaAttrs` exactly replace the old public types. Old source,
  bytecode/API names, imports, aliases, overloads, and adapters are absent.
- One shared helper owns exact ARG_MIN/ARG_MAX construction with the required five-method surface,
  validation order/messages, static-zero rejection, Shape behavior, fixed INT64 result, exact
  attributes/kind/provenance, freshness, and identity effects.
- All six public arg-extrema methods map directly to the exact kind, retention flag, and policy.
  Tensor has exactly 130 declared public methods and gains only the `argMin` method name.
- Both arg families accept all five numeric inputs, reject BOOL, use the exact floating/integral
  comparison and tie policy, reject a statically empty selected axis, and accept unselected zero
  axes plus unbound selected extents.
- Every successful expression is fresh, unlabeled, storage-free, one-output, and has exact ordered
  inputs and provenance index zero; every specified local failure consumes no Tensor identity.
- Focused tests cover exact surfaces/signatures, old-name absence, all kinds/types/forms, empty
  Shape acceptance or rejection as applicable, Shape/reference behavior, both tie policies,
  validation precedence/messages, freshness, input immutability, no-ID failures, and identifier
  exhaustion. Javadocs and API documentation record the modular, identity, signed-order, NaN,
  signed-zero, and infinity semantics that metadata-only model tests cannot evaluate.
- Tensor API, Compile API, glossary, capability baseline, task, master plan, and roadmap are
  synchronized. Training API and all architecture/build/cross-layer areas remain unchanged with
  reasoned conclusions.
- One final `:modules:model:test` run passes after executable Java stabilizes.
- A separate clean-context documentation-focused pass finalizes all permitted Javadocs and
  documentation in the same overall change, reusing the successful Java evidence unless it
  changes executable Java or records a concrete rerun reason.
- Model Javadoc, one runnable Java 26 metadata example, generated-page/source inspection,
  old-vocabulary scans, links/anchors/fences/final newlines, exact 22-path scope, status/order
  synchronization, terminology, whitespace, and `git diff --check` pass.
- Task 0018U remains Complete. Task 0018U1 becomes Complete only after both passes. Task 0018V and
  every later task remain Draft, and no detailed 0018V specification is created.

## Tests / validation

Focused tests while executable Java stabilizes:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.operation.reduction.ReductionSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorArgMaxExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

Final Java validation after executable behavior stabilizes:

```bash
./gradlew :modules:model:test
```

The implementation context records exact test counts and hands this evidence to the documentation
context. Do not repeat the successful final model suite unless executable Java changes afterward
or a concrete risk is recorded.

Documentation-focused pass:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass must also:

- compile and run one Java 26 metadata example covering one integral empty-capable reduction
  request, one `argMin`, one explicit-policy `argMax`, exact shared attributes, normalized Shape,
  fixed result types, ordered provenance, output index zero, and storage absence without claiming
  evaluated values;
- validate every local Markdown link and heading anchor in changed documents;
- inspect generated public Javadoc and complete package-private source Javadocs;
- use reflection/`javap`/imports/source scans only for the concrete risks of the 130-method surface,
  exact helper/record/kind signatures, direct delegation, old-name absence, validation order,
  one factory call, and no cross-layer/value/storage implementation;
- verify exact 22-path scope, no Java/Gradle/architecture/other-module spill, 0018U Complete,
  0018U1 synchronized, 0018V Draft with no detailed spec, and coherent dependencies/order; and
- review terminology, current-versus-planned claims, examples, fences, final newlines, trailing
  whitespace, authority boundaries, and the complete final diff.

Repository-wide validation is deferred to the recorded capability-reset checkpoint after task
0018V and CI. This task changes only `modules/model` and its documentation, without dependencies,
architecture boundaries, shared build configuration, or executable backend behavior.

## Dependencies

- [Task 0001](0001-data-type-model.md) — exact signed integral types and differentiability.
- [Task 0016A](0016a-reduction-semantic-kinds-and-attributes.md),
  [0016B](0016b-sum-mean-and-product-tensor-expressions.md),
  [0016C](0016c-min-and-max-tensor-reduction-expressions.md),
  [0016D](0016d-boolean-all-and-any-tensor-expressions.md), and
  [0016E](0016e-arg-max-tensor-expressions.md) — current aggregate kinds, axis attributes, public
  ordinary reduction shapes, and arg-max surface/result conventions.
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md) — exact family-owned
  kind/attributes and input/output structural validation.
- [Task 0018U](0018u-integral-elementwise-arithmetic-and-comparisons.md) — selected fixed-width
  modular arithmetic and signed integral ordering baseline.

All dependencies are Complete.

## Follow-up tasks

- **0018V Multi-axis and statistical reductions** remains Draft. It may build on this task's
  stable ordinary integral and arg-extrema contracts but owns ordered multi-axis reduction,
  log-sum-exp, variance, standard deviation, and L1/L2 norms.
- Task 0019C later owns sort, argsort, and true multi-output top-K with its own stability and
  ordering contract.
- Compiler, backend, conformance, runtime, and integration tasks own graph revalidation,
  gradients, dynamic-bound enforcement, lowering, modular execution, numeric conformance,
  kernels, and end-to-end behavior.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None.

This task extends and normalizes backend-independent model semantics within the existing
operation-reduction and Tensor ownership boundaries. It changes no module dependency, lifecycle,
graph representation, backend contract, runtime path, or architecture rule.

If implementation requires an architecture change, stop and report the conflicting rule and
decision needed before editing `ARCHITECTURE.md`, focused architecture documentation, ADRs, or
architecture tests.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, current architecture index, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0001/0016A–0016E/0018K/0018U/0018U1, Tensor/Compile/
Training APIs, glossary, and every affected or review-only source/test named by task 0018U1 in
full.

Implement task 0018U1 exactly. Add selected INT32/INT64 ordinary SUM/PROD/MIN/MAX with exact
modular/signed/empty-domain semantics. Add ARG_MIN and the exact three argMin overloads. Atomically
replace ArgMaxTiePolicy, ArgMaxAttrs, and TensorArgMaxExpressions with the shared arg-extrema
types/helper, without aliases. Preserve floating, BOOL, masked, Shape, descriptor, producer,
provenance, and architecture boundaries except where the task explicitly normalizes arg-max
ordering and empty-axis behavior. Stay within the exact 22 paths. Stop on scope or architecture
conflict. Do not commit or push.

Run focused tests as needed and one final model test after executable Java stabilizes. Then hand
the actual diff and exact evidence to a separate clean-context documentation-focused agent in the
same overall change. That agent must inspect final source/tests, finalize every permitted Javadoc,
Tensor/Compile API, glossary, capabilities/task/master/roadmap, run model Javadoc and the specified
documentation/example/scope checks, and reuse successful Java evidence unless executable behavior
changes or it records a concrete rerun reason.

Mark 0018U1 Complete only after both passes succeed. Keep 0018U Complete. Leave 0018V and all later
tasks Draft and do not create their detailed specifications.
```

## Local decisions

- The originally planned 20-path scope was expanded with explicit user approval to 21 paths after
  the required old-type deletion exposed `OperationSignatureTest` as a stale signature consumer,
  then to 22 paths after the final model suite exposed `TensorBinaryArithmeticTest`'s independent
  127-method assertion. Only the exact stale signature matrix and method-count assertions changed
  in those added paths.
- Shared arg-extrema construction inspects the selected Dimension's `staticSize()` after one axis
  normalization. A present zero fails locally; an absent size remains structurally accepted for
  later compiler or binding validation.
- Ordinary reductions retain the existing six-method helper surface. Kind-aware validation treats
  SUM/PROD/MIN/MAX as numeric, MEAN as floating-only, and ALL/ANY as BOOL-only without introducing
  promotion or a result-type option.

## Known limitations

- This model change records integral arithmetic, extrema ordering, tie, and empty-domain meaning;
  it does not evaluate values, implement compiler revalidation, bind dynamic dimensions, define
  gradients, lower operations, add backend kernels, or execute reductions.
- A dynamic or expression selected arg-extrema extent remains conditionally valid until a later
  owner proves or binds it positive. No current model callable performs that later validation.
- Floating ordinary reduction numerical and empty-domain policies, integral mean, masked integral
  reductions, and multi-axis/statistical reductions remain outside this task.

## Validation evidence

Planning context: `/root/plan_0018u1`.

Planning selected exact-type modular integral sum/product, signed integral min/max, bounded-domain
identities, one shared arg-extrema policy/attributes/helper vocabulary, deterministic floating
arg-extrema ordering, fixed INT64 indices, and invalid empty selected axes. Current architecture
and completed model contracts are sufficient; no architecture decision is required.

Planning reviewed the architecture contract and current architecture index; documentation and
planning rules with the General, Planning, API/Javadoc, and Example profiles; roadmap, model
capabilities and master plan; completed tasks 0001, 0016A through 0016E, 0018K, and 0018U; current
DataType/promotion, Shape, aggregate kind/attributes, ordinary reduction, arg-max, Tensor,
descriptor, factory, producer, and provenance source/tests; Tensor, Compile, and Training API
boundaries; glossary; and the current Java/test surface named by this task.

Planning validation:

- A local Markdown target checker resolved 349 links across this task, model master plan,
  capability baseline, and roadmap with zero missing files. None of the four changed planning
  documents contains a local heading-anchor link, so there was no changed anchor target to
  resolve.
- Fence checks found 28 balanced fences in this task, two in the master plan, and none in the
  capability baseline or roadmap. All four files end in a newline and have no trailing whitespace.
- The required-section scan found Status, goal/scope/exclusions, architecture/package contracts,
  exact affected files and maximum scope, Javadocs, acceptance, validation, dependencies,
  follow-ups, architecture impact, full implementation prompt, and all implementation-time
  placeholders.
- Status/order checks found 0018U Complete, 0018U1 Ready and linked in this task/master/roadmap,
  and 0018V Draft with no detailed task file. Dependencies are complete and table order is
  unchanged.
- Exact-scope inspection found only four authorized planning paths changed: this new task, model
  capabilities, model master plan, and roadmap. No Java, test, Gradle, architecture/focused
  architecture, architecture-test, other-module, conformance, or integration path changed.
- `git diff --check` passed; explicit untracked-file whitespace and final-newline checks also
  passed.

Implementation context: `/root/task_0018u1_implementation`.

Implementation evidence:

- The focused Gradle selection covering `OperationSignatureTest`, `ReductionSemanticsTest`,
  `TensorNumericReductionTest`, `TensorArgMaxExpressionTest`, and `TensorTest` passed 48 tests with
  zero failures before the separately authorized count-only `TensorBinaryArithmeticTest` update.
- The final `./gradlew :modules:model:test` after all executable Java and test changes passed 735
  tests across 90 suites with zero skipped, failures, or errors. No executable Java changed after
  this run, so the documentation and final audit reused it without rerunning Java tests.
- The final focused-class totals present in that suite were 5 operation-signature, 12 reduction
  semantics, 10 numeric-reduction, 6 arg-extrema, 12 binary-arithmetic, and 15 Tensor tests.
- Automated tests cover the exact kind/record/helper/public surfaces, old-class absence, all
  numeric types and forms, Shape/reference rules, static-empty selected-axis failure, unselected
  zero and dynamic acceptance, validation precedence and messages, provenance/output index,
  freshness, input immutability, no-ID local failures, and identifier exhaustion.

Independent documentation context:
`/root/task_0018u1_implementation/task_0018u1_docs`.

Documentation and final-audit evidence:

- The clean-context documentation pass inspected the final production and test diff and finalized
  every permitted production Javadoc, Tensor API, Compile API, glossary, capabilities, task,
  master plan, and roadmap without changing executable behavior.
- `./gradlew :modules:model:javadoc` passed. Generated pages contain `ArgExtremaAttrs`,
  `ArgExtremaTiePolicy`, all six Tensor arg-extrema overloads, exact failures, and integral
  reduction contracts; no generated old arg-max-specific type/helper page remains.
- A compiled Java 26 metadata example created an INT32 empty-capable retained-axis sum request,
  `argMin(0)`, and explicit-LAST `argMax(-1)`. It printed exact normalized
  `AxisReductionAttrs`/`ArgExtremaAttrs`, Shapes `[2, 1, 3]`, `[0, 3]`, and `[2, 0]`, INT32/INT64
  result types, exact ordered input references, output index zero, and absent result storage.
- `javap` reported exactly 130 declared public Tensor methods; the shared helper has one
  package-private entry and four private methods; the record, policy, and appended `ARG_MIN`
  surfaces are exact. Source inspection found six direct Tensor delegations, one central factory
  call, and no value/storage/cross-layer implementation.
- Production source and compiled-bytecode scans found no `ArgMaxAttrs`, `ArgMaxTiePolicy`, or
  `TensorArgMaxExpressions`. Those names remain only where migration/history documentation needs
  to identify the removed API.
- The targeted Markdown checker resolved 509 local links, including 141 heading anchors, across
  the seven changed documentation/planning files with zero errors. Fences are balanced; all seven
  files have final newlines and no trailing whitespace.
- Exact scope inspection found the authorized 22 paths: nine production paths, six test paths,
  and seven documentation/planning paths. There is no Java/Gradle/architecture/other-module,
  conformance, or integration spill and no detailed 0018V task specification.
- `git diff --check` passed after the final combined change.
- Status and dependency inspection found task 0018U Complete, task 0018U1 Complete in this task,
  model master plan, capabilities, and roadmap, and task 0018V plus every later task Draft.
- Training API remains unchanged because this task adds no gradient or training callable.
  DataType, DataTypePromotion, AxisReductionAttrs, masked/BOOL/scan contracts, descriptor,
  factory, producer, and provenance remain unchanged because the existing contracts already
  express the required categories, Shape metadata, central construction, and immutable ordered
  provenance. Architecture documents/tests, backend conformance, integration tests, Gradle,
  dependencies, and other modules remain unchanged because ownership and boundaries did not move.

## Implementation notes

- `AggregateReductionKind.ARG_MIN` is appended after `ARG_MAX`; both share the one-input,
  one-output `ArgExtremaAttrs` signature.
- `ArgExtremaTiePolicy`, `ArgExtremaAttrs`, and `TensorArgExtremaExpressions` replace the three
  arg-max-only types without aliases. Tensor exposes the matching three `argMin` and three migrated
  `argMax` overloads.
- `TensorReductionExpressions` now accepts integral input only for SUM/PROD/MIN/MAX and preserves
  its existing construction and helper shape. Tests and documentation carry the modular,
  signed-order, and bounded empty-identity contracts that metadata construction does not execute.

## Completion summary

- Completed changes: added exact-type integral ordinary reductions, appended ARG_MIN, normalized
  arg-min/arg-max into one shared semantic and construction family, and removed every old
  arg-max-specific type/helper without an alias.
- Files changed or created: exactly the authorized nine production paths, six test paths, and
  seven documentation/planning paths.
- Tests and validation: focused 48 tests passed; final model suite passed 735 tests across 90
  suites with zero skipped/failures/errors; model Javadoc, Java 26 example, generated-page/source,
  javap/reflection, old-name, Markdown, scope/status, whitespace, and diff checks passed.
- Documentation-agent review: the separate clean-context pass finalized all affected Javadocs,
  explanatory API documentation, glossary impact, planning evidence, and no-change conclusions.
- Documentation impact: Tensor API, Compile API, glossary, capabilities, task, master plan, and
  roadmap now describe the implemented integral and shared arg-extrema contracts.
- Javadoc review: all affected public and package-private contracts document semantics,
  parameters, results, failures, identity effects, ownership, and deferred execution boundaries.
- Glossary impact: aggregate reduction and arg-extrema terminology now uses the shared vocabulary
  and records the selected modular, signed-order, tie, and empty-domain meanings.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
