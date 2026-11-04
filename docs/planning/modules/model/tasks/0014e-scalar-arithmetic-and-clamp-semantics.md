# Task 0014E: [Scalar](../../../../glossary.md#term-scalar) Arithmetic and Clamp Semantics

**Editorial reading key (later context, not a status update).**

A scalar attribute is one fixed parameter applied across positions, not an additional Tensor input; clamp constrains values to an inclusive range. The original attributes retain binary64 literals. Their ability to represent NaN or infinity does not itself specify execution at those values. Here Model defines a typed meaning for later construction code; this semantic-family task neither receives stored operands nor evaluates them. Its completion is not evidence of Compiler capture, derivatives or backend execution.

Focused current reading: [Current scalar/clamp reference](../../../../reference/tensor/elementwise.md#scalar-arithmetic-and-clamp-expressions). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define the [backend](../../../../glossary.md#backend)-independent typed semantic vocabulary and [immutable](../../../../glossary.md#term-immutable) attributes for [scalar](../../../../glossary.md#term-scalar)
multiplication, scalar power, inclusive range clamp, lower-bound clamp, and upper-bound clamp.
The contracts must preserve exact scalar [parameters](../../../../glossary.md#term-parameter) as [model](../../../../glossary.md#term-model) values without retaining a [Tensor](../../../../glossary.md#tensor),
inferring a result, choosing a data-type conversion, executing mathematics, or reporting backend
support.

This task creates the complete [operation](../../../../glossary.md#operation)-family foundation consumed by task 0014F. It does not add
public Tensor methods or expression behavior.

## [Scope](../../../../glossary.md#term-scope)

- Add one public `ScalarElementwiseKind` enum implementing [`OperationKind`](../../../../glossary.md#operationkind).
- Define exactly `MUL`, `POW`, `CLAMP`, `CLAMP_MIN`, and `CLAMP_MAX` in that order.
- Add one public `ScalarValueAttrs` record implementing [`OperationAttrs`](../../../../glossary.md#operationattrs) with exactly one
  `double value` component.
- Add one public `ClampRangeAttrs` record implementing `OperationAttrs` with exactly
  `double minValue` and `double maxValue` components in that order.
- Preserve every supplied primitive value without conversion, normalization, cached alternate
  precision, or defaulting.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [precision glossary](../../../../glossary.md#term-precision))</span>
- Reject a clamp range only when `minValue > maxValue`, using the exact [validation](../../../../glossary.md#term-validation) contract below.
- Define the valid kind-to-attributes pairings through typed family documentation without adding
  runtime compatibility validation to <span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>[`Operation`](../../../../glossary.md#operation).
- Add one focused same-package test covering the complete family vocabulary, attribute contracts,
  standard record value semantics, composition, and exclusions.
- Finalize affected Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
  through the required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public `Tensor.mul(double)`, `pow(double)`, `clamp(double, double)`, `clampMin(double)`, or
  `clampMax(double)` methods
- Tensor result descriptors, labels, identity allocation, storage, provenance, expression chaining,
  or <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>`TensorFactory.createDerived`
- input data-type eligibility, [scalar](../../../../glossary.md#term-scalar) conversion or quantization to [`BFLOAT16`](../../../../glossary.md#term-bfloat16)/[`FLOAT32`](../../../../glossary.md#term-float32), result
  type, shape preservation, layout state, or gradient-eligibility propagation<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- eager numeric execution, storage reads, constant folding, algebraic rewriting, nested-clamp
  collapsing, identity elimination, or special exponent rewrites<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [constant folding glossary](../../../../glossary.md#term-constant-folding); [exponent glossary](../../../../glossary.md#term-exponent))</span>
- finite-only validation, NaN rejection, infinity canonicalization, signed-zero normalization,
  domain rules, overflow/underflow, rounding, precision, accuracy, or other numerical edge behavior<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [validation glossary](../../../../glossary.md#term-validation); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [normalization glossary](../../../../glossary.md#term-normalization); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [rounding glossary](../../../../glossary.md#term-rounding); [precision glossary](../../../../glossary.md#term-precision))</span>
- gradient rules, clamp boundary derivative conventions, scalar-power differentiation, backward
  graph generation, autograd, optimizer, or training behavior<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [backward graph glossary](../../../../glossary.md#backward-graph); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- an attributes hierarchy, [sealed](../../../../glossary.md#term-sealed) family interface, generic scalar wrapper, boxed `Number`,
  `BigDecimal`, string-keyed map, optional bound, collection, union, or variant container
- a factory, builder, registry, parser, kind-to-attributes validator, reflective discovery, service,
  visitor, or string dispatch<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- kind fields for arity, category, cost, fusion, result type, differentiability, backend support,
  lowering, route, or kernel metadata<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- changes to [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), existing kind
  families, Tensor, graph records, or existing tests<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- compiler, planning, prepare, runtime, backend, engine, tracing, dependency, Gradle, architecture,
  or another-module changes<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- a detailed task-0014F specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md)
- [Task 0014C](0014c-unary-elementwise-semantic-kinds.md)
- [Task 0014D](0014d-unary-elementwise-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Capability origin

The read-only `legacy/pre-rewrite` branch exposes public [Tensor](../../../../glossary.md#tensor) capabilities `mul(double)`,
`pow(double)`, `clamp(double, double)`, `clampMin(double)`, and `clampMax(double)`. Its [operation](../../../../glossary.md#operation)
[descriptors](../../../../glossary.md#term-descriptor) retain [scalar](../../../../glossary.md#term-scalar) multiplier, [exponent](../../../../glossary.md#term-exponent), minimum, or maximum values. The range form is
public but is implemented by eagerly composing upper-bound and lower-bound operations instead of
having its own [immutable](../../../../glossary.md#term-immutable) semantic descriptor.

Legacy descriptors duplicate every [parameter](../../../../glossary.md#term-parameter) as both `double` and cached `float`, select the
descriptor constructor from the current Tensor [data type](../../../../glossary.md#data-type--datatype), expose operation cost/[fusion](../../../../glossary.md#term-fusion)/result
[metadata](../../../../glossary.md#term-metadata), and create diagnostic expression strings. Legacy builders also inspect Tensor
[provenance](../../../../glossary.md#provenance) and scalar values to replace multiply by zero/one/minus-one, special powers, unbounded
clamps, and nested clamps; they attach [gradient](../../../../glossary.md#term-gradient) callbacks and allocate result Tensors.

The new [model](../../../../glossary.md#term-model) retains the five public mathematical capabilities and their scalar parameters but
does not copy that coupling. Each parameter is stored once as the exact supplied Java `double`.
Task 0014F will own public-expression eligibility and descriptor/provenance construction. [Compiler](../../../../glossary.md#term-compiler)
optimization will own [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), compiler [autograd](../../../../glossary.md#autograd) will own gradient expansion, and
[backend](../../../../glossary.md#backend)/[conformance](../../../../glossary.md#term-conformance) work will own input-type conversion and numerical execution.

`CLAMP` is represented as a first-class semantic kind rather than prescribing the legacy [eager](../../../../glossary.md#term-eager-evaluation)
two-operation expansion. This records the public request without requiring a compiler to recover
it from [mutable](../../../../glossary.md#term-mutable) construction history. A later compiler or backend may still lower it to equivalent
primitive operations under its own contracts.

## Architecture constraints

- [Operation](../../../../glossary.md#operation) kinds and attributes are [immutable](../../../../glossary.md#term-immutable) [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model) semantics owned by
  `modules/model`.
- `ScalarElementwiseKind` identifies which computation is requested. It stores no parameter,
  Tensor, graph occurrence, result fact, executable behavior, or backend information.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `ScalarValueAttrs` is the complete attribute value for `MUL`, `POW`, `CLAMP_MIN`, and
  `CLAMP_MAX`. The kind determines whether `value` means multiplier, [exponent](../../../../glossary.md#term-exponent), minimum, or maximum.
- `ClampRangeAttrs` is the complete attribute value for `CLAMP`; component order is lower bound
  then upper bound.
- Family documentation defines valid pairings. Generic `Operation` continues to validate only
  non-null components and must not be changed to introduce family discovery or compatibility
  checks.
- A [scalar](../../../../glossary.md#term-scalar) [parameter](../../../../glossary.md#term-parameter) is stored as one Java [binary64](../../../../glossary.md#floating-point-precision-and-rounding) value. Attributes do not retain input
  [`DataType`](../../../../glossary.md#data-type--datatype), [cache](../../../../glossary.md#term-cache) a float conversion, or decide how a future floating [Tensor](../../../../glossary.md#tensor) consumes that value.
- Accessors return the exact stored primitive value. No finite check, NaN canonicalization,
  infinity replacement, signed-zero normalization, or precision conversion occurs.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [NaN glossary](../../../../glossary.md#term-nan); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [normalization glossary](../../../../glossary.md#term-normalization); [precision glossary](../../../../glossary.md#term-precision))</span>
- `ClampRangeAttrs` rejects only a strictly inverted range under Java primitive `>` comparison.
  Equal values, either ordering of signed zeros, ordered infinities, and NaN endpoints remain
  representable. Numerical special-value behavior remains outside this semantic descriptor.<span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- All five kinds have one logical Tensor input, but arity is family context rather than enum
  metadata. Scalar attributes are semantic parameters, not additional Tensor inputs.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Stable enum names are diagnostics, not serialization tokens, registry keys, backend dispatch
  keys, or reflective plugin identifiers.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [tokens glossary](../../../../glossary.md#term-token); [registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch); [plugin glossary](../../../../glossary.md#term-plugin))</span>
- Package direction is `model.operation.elementwise.scalar -> model.operation`. It must not depend
  on Tensor, datatype, graph, compiler, planning, runtime, prepare, backend, storage, or training
  packages.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Stop if implementation requires a Tensor or DataType reference, another attributes concept, a
  compatibility validator, numerical policy, dependency, or architecture change.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation` — supplies the generic [`OperationKind`](../../../../glossary.md#operationkind),
  [`OperationAttrs`](../../../../glossary.md#operationattrs), and [`Operation`](../../../../glossary.md#operation) contracts.

Package added:

- `io.github.pho001.synaptik.model.operation.elementwise.scalar` — owns typed parameterized
  one-input scalar arithmetic and clamp semantics.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind` — public
  typed semantic identities shared by future public expressions, compiler, and backends.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backends glossary](../../../../glossary.md#backend))</span>
- `io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs` — public
  immutable one-value attributes for the four one-parameter kinds.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- `io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs` — public immutable
  ordered lower/upper attributes for the range kind.
- `ScalarElementwiseSemanticsTest` — same-package focused test for all three cohesive contracts.

## Required contract

### Semantic kind vocabulary

Create exactly:

```java
public enum ScalarElementwiseKind implements OperationKind {
    MUL,
    POW,
    CLAMP,
    CLAMP_MIN,
    CLAMP_MAX
}
```

The enum declares no project field, method, constructor, nested type, [constant](../../../../glossary.md#term-constant) body, [alias](../../../../glossary.md#aliasing-and-arena),
[metadata](../../../../glossary.md#term-metadata), or override. Standard enum machinery and the inherited `Enum.name()` implementation are
the entire [runtime](../../../../glossary.md#term-runtime) surface.

Meanings are exact:

| Kind | [Elementwise](../../../../glossary.md#term-elementwise) meaning | Required attributes |
|---|---|---|
| `MUL` | multiply each input value by one [scalar](../../../../glossary.md#term-scalar) multiplier | `ScalarValueAttrs` |
| `POW` | raise each input value to one scalar [exponent](../../../../glossary.md#term-exponent) | `ScalarValueAttrs` |
| `CLAMP` | constrain each input value to an inclusive lower/upper range | `ClampRangeAttrs` |
| `CLAMP_MIN` | constrain each input value to be no lower than one minimum | `ScalarValueAttrs` |
| `CLAMP_MAX` | constrain each input value to be no greater than one maximum | `ScalarValueAttrs` |

This table is a typed family contract, not runtime matching logic. Constructing [`Operation`](../../../../glossary.md#operation) with
an incompatible attributes implementation remains possible because task 0006 deliberately keeps
the generic record independent of concrete families. Consumers that understand this family use
the typed kind and documented pairing; they do not use [classpath](../../../../glossary.md#term-classpath) scanning or a [registry](../../../../glossary.md#term-registry).

### One-value attributes

Create exactly:

```java
public record ScalarValueAttrs(double value) implements OperationAttrs {
}
```

The record has no explicit [validation](../../../../glossary.md#term-validation) because every Java `double` bit pattern is representable as
a [scalar](../../../../glossary.md#term-scalar) semantic [parameter](../../../../glossary.md#term-parameter). Construction stores the supplied primitive through ordinary record
assignment. The accessor returns the same primitive value without conversion. This includes
[finite](../../../../glossary.md#term-finite) values, positive and [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero), positive and [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero), and [NaN](../../../../glossary.md#term-nan) payloads.

Record-generated equality and hashing use Java's standard record semantics for a `double`
component: signed zeros remain distinct and NaN values compare under the generated [floating-point](../../../../glossary.md#term-floating-point)
component semantics. The accessor still exposes the stored primitive bits. Generated diagnostic
text is not a [serialization](../../../../glossary.md#term-serialization) or parsing contract.

### Clamp-range attributes

Create exactly:

```java
public record ClampRangeAttrs(double minValue, double maxValue) implements OperationAttrs {
    public ClampRangeAttrs {
        if (minValue > maxValue) {
            throw new IllegalArgumentException(
                    "minValue must be less than or equal to maxValue");
        }
    }
}
```

[Validation](../../../../glossary.md#term-validation) uses the exact primitive comparison shown. It rejects only when the supplied lower
bound is strictly greater than the supplied upper bound. The failure message is exact. It does not
use `Double.compare`, reject [NaN](../../../../glossary.md#term-nan), require finiteness, or normalize either value. Therefore equal
bounds, either [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) ordering, ordered [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and one or two NaN endpoints are accepted
and retained unchanged.

The record stores lower then upper bound and uses generated [structural equality](../../../../glossary.md#term-structural-equality), hashing, and
diagnostic text over both components. Diagnostic text is not a [serialization](../../../../glossary.md#term-serialization) or [backend](../../../../glossary.md#backend) format.

### [Operation](../../../../glossary.md#operation) composition

Valid explicit compositions include:

```java
Operation scale = new Operation(
        ScalarElementwiseKind.MUL,
        new ScalarValueAttrs(0.5));

Operation power = new Operation(
        ScalarElementwiseKind.POW,
        new ScalarValueAttrs(2.0));

Operation range = new Operation(
        ScalarElementwiseKind.CLAMP,
        new ClampRangeAttrs(0.0, 1.0));
```

`CLAMP_MIN` and `CLAMP_MAX` use `ScalarValueAttrs`. None of these values stores an input [Tensor](../../../../glossary.md#tensor),
result [descriptor](../../../../glossary.md#term-descriptor), [graph](../../../../glossary.md#graph) [identity](../../../../glossary.md#term-identity-meanings), label, storage, [gradient](../../../../glossary.md#term-gradient) state, or execution route.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarElementwiseKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarValueAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ClampRangeAttrs.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarElementwiseSemanticsTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a task-related inconsistency requires stopping:

- `docs/api/compile-api.md`
- `docs/api/training-api.md`
- `docs/planning/modules/model/capabilities.md`
- Existing [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), [`Operation`](../../../../glossary.md#operation), [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs),
  `BinaryArithmeticKind`, and `UnaryElementwiseKind` Javadocs and tests.
- Tensor and Tensor expression contracts, focused architecture documents, ADRs, architecture
  tests, backend-conformance tests, and integration tests.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most three production files, one test file, and five documentation/[planning](../../../../glossary.md#term-planning) files: nine paths
total.

No existing Java source or test may change. If implementation requires another production/test
concept, a [Tensor](../../../../glossary.md#tensor)/[DataType](../../../../glossary.md#data-type--datatype) reference, a factory or compatibility validator, another documentation
file, or more than nine paths, stop and propose a follow-up or architecture decision. Do not create
task 0014F.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Document `ScalarElementwiseKind` as backend-independent, parameterized, one-input semantic
  vocabulary and distinguish scalar attributes from Tensor inputs.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [scalar glossary](../../../../glossary.md#term-scalar); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Document every enum constant with its mathematical operand roles, required attributes type, and
  explicit deferral of eligibility, type conversion, shape/result inference, numerical edge
  behavior, gradients, execution, and backend support.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [shape glossary](../../../../glossary.md#shape); [inference glossary](../../../../glossary.md#term-inference); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Document the valid kind/attributes table and state that generic [`Operation`](../../../../glossary.md#operation) does not enforce it.
- Document both public records with complete component `@param` tags, immutable ownership, exact
  primitive retention, supported special values, record equality/hashing, and diagnostic-only
  text.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Add explicit accessor Javadocs with non-null not applicable to primitives, exact returned-value
  meaning, no conversion, and role determined by kind where applicable.
- Document the `ClampRangeAttrs` canonical constructor with validation order, accepted edge values,
  exact failure, and both <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [validation glossary](../../../../glossary.md#term-validation))</span>`@param` tags plus `@throws`.
- Review related foundational Javadocs and record why they remain accurate, or stop on an
  out-of-scope inconsistency.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Exactly one public enum and two public records are added in the planned [scalar](../../../../glossary.md#term-scalar) package.
- The enum implements [`OperationKind`](../../../../glossary.md#operationkind) and declares exactly five constants in the specified order
  and spelling.
- The enum adds no project fields, methods, nested types, [metadata](../../../../glossary.md#term-metadata), [aliases](../../../../glossary.md#aliasing-and-arena), or [constant](../../../../glossary.md#term-constant) bodies.
- `ScalarValueAttrs` implements [`OperationAttrs`](../../../../glossary.md#operationattrs) and has exactly one `double value` record
  component with no additional instance state or validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `ClampRangeAttrs` implements `OperationAttrs` and has exactly `double minValue` then
  `double maxValue`, with no additional instance state.
- Every primitive scalar/bound is retained unchanged and returned unchanged, including raw
  signed-zero, infinity, and NaN cases.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan))</span>
- Clamp range rejects only strict `minValue > maxValue` with the exact exception type and message;
  all documented edge cases remain accepted.
- Equal component values produce equal records and equal hash codes; changed components produce
  unequal values under generated record semantics.<span data-glossary-navigation="true"> ([hash codes glossary](../../../../glossary.md#hash-codes-and-content-digests))</span>
- Every kind constructs a valid [`Operation`](../../../../glossary.md#operation) with the exact documented immutable attributes
  reference. No operation factory or compatibility validator is added.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Scalar kinds remain typed-distinct from equally named binary kinds, and inherited names remain
  diagnostic rather than dispatch/serialization keys.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [dispatch glossary](../../../../glossary.md#term-dispatch); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- No Tensor, DataType, descriptor, provenance, graph, storage, gradient, compiler, planning,
  runtime, backend, dependency, or architecture behavior is added.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [DataType glossary](../../../../glossary.md#data-type--datatype); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused and aggregate model tests, model Javadoc, root tests, reflection/javap/import/scope
  checks, documentation links/formatting, and status synchronization pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- A separate clean-context documentation-focused agent finalizes all new Javadocs, Tensor API,
  glossary, task evidence, master plan, and roadmap in the same change and records reasoned
  no-change conclusions for related APIs, capabilities, architecture, and existing contracts.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Task 0014E becomes Complete only after both passes. Task 0014F remains Draft without a detailed
  specification.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after the documentation-focused review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseSemanticsTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

The focused test must cover:

- exact package, public/final enum and record shapes, interfaces, record components, constructor
  parameters, fields, methods, and absence of nested types;<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>
- exact enum [constant](../../../../glossary.md#term-constant) vocabulary/order and inherited stable names;
- typed distinction from `BinaryArithmeticKind.MUL` and `BinaryArithmeticKind.POW`;
- every documented kind-to-attributes composition with exact reference retention through
  [`Operation`](../../../../glossary.md#operation);
- `ScalarValueAttrs` [finite](../../../../glossary.md#term-finite), infinite, [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), and multiple [NaN](../../../../glossary.md#term-nan)-bit-pattern retention;
- `ClampRangeAttrs` ordinary ordered, equal, signed-zero, infinite, and NaN bounds;
- every strictly inverted representative range and exact failure message;
- generated equality, hashing, and diagnostic text without treating text as [serialization](../../../../glossary.md#term-serialization); and
- absence of Tensor/DataType/backend/runtime state, factories, registries, validation maps, or
  added dependencies.

Manually inspect `javap -p -c -s` and reflection for exact enum/record shape, component order,
constructor validation, and absence of additional project [API](../../../../glossary.md#term-application-programming-interface-api)/state. Scan production imports and
[Gradle](../../../../glossary.md#term-gradle) dependencies for forbidden layers. Confirm no [Tensor](../../../../glossary.md#tensor), [DataType](../../../../glossary.md#data-type--datatype), storage, [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler),
[planning](../../../../glossary.md#term-planning), [runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend), [gradient](../../../../glossary.md#term-gradient), cost, [fusion](../../../../glossary.md#term-fusion), route, [registry](../../../../glossary.md#term-registry), map, reflection, or service
type appears. Validate generated [Javadoc](../../../../glossary.md#term-javadoc), Tensor API current-versus-planned wording, glossary
terminology, local links/anchors/fences/whitespace, exact nine-path [scope](../../../../glossary.md#term-scope), synchronized statuses,
and absence of a task-0014F specification.

## Dependencies

- Task 0005 supplies [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), and [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs) foundations.
- Task 0006 supplies [immutable](../../../../glossary.md#term-immutable) generic [`Operation`](../../../../glossary.md#operation) composition and exact reference retention.
- Completed kind families 0014A and 0014C establish package and typed-enum conventions reused by
  this family without creating a dependency between family packages.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Follow-up tasks

- Task 0014F remains Draft for floating input eligibility, effective scalar handling, public Tensor
  methods, result descriptor derivation, exact provenance, and absence of eager canonicalization.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [eager glossary](../../../../glossary.md#term-eager-evaluation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- [Compiler](../../../../glossary.md#term-compiler) optimization will later own multiply/power/clamp rewrites and range simplification.
- Compiler [autograd](../../../../glossary.md#autograd) will later own [scalar](../../../../glossary.md#term-scalar)-power and clamp [gradient](../../../../glossary.md#term-gradient) expansion and boundary policy.
- Backend and conformance tasks will later define scalar conversion, numerical special-value
  behavior, and executable support.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None. The architecture already assigns [backend](../../../../glossary.md#backend)-independent [operation](../../../../glossary.md#operation) kinds and
[immutable](../../../../glossary.md#term-immutable) typed attributes to `modules/model`. The new package refines that existing [ownership](../../../../glossary.md#term-ownership-meanings) and
adds no [dependency](../../../../glossary.md#term-dependency) or [lifecycle](../../../../glossary.md#lifecycle) rule.

If implementation requires [Tensor](../../../../glossary.md#tensor)/public-expression behavior, [DataType](../../../../glossary.md#data-type--datatype) coupling, [graph](../../../../glossary.md#graph)/[compiler](../../../../glossary.md#term-compiler)
logic, backend support, execution [metadata](../../../../glossary.md#term-metadata), service discovery, a [module](../../../../glossary.md#term-module) dependency, or an
architecture change, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks 0005/0006/0014A/0014C/0014D/0014E, Tensor API, Compile API,
Training API, glossary, current OperationKind/OperationAttrs/NoOperationAttrs/Operation/
BinaryArithmeticKind/UnaryElementwiseKind contracts and tests, and Java 26 Gradle configuration.

Implement task 0014E exactly. Add only ScalarElementwiseKind.java, ScalarValueAttrs.java,
ClampRangeAttrs.java, and ScalarElementwiseSemanticsTest.java under
io.github.pho001.synaptik.model.operation.elementwise.scalar for Java code/tests.

The public enum must contain exactly MUL, POW, CLAMP, CLAMP_MIN, CLAMP_MAX in that order and add no
project fields/methods/nested types/metadata. ScalarValueAttrs is exactly a public record of one
double value with no validation. ClampRangeAttrs is exactly a public record of double minValue and
double maxValue and rejects only primitive minValue > maxValue with exact message
"minValue must be less than or equal to maxValue". Preserve every supplied primitive without
conversion or normalization. Valid composition pairs MUL/POW/CLAMP_MIN/CLAMP_MAX with
ScalarValueAttrs and CLAMP with ClampRangeAttrs; document them but add no compatibility validator.

Do not add Tensor methods, expression construction, DataType, scalar conversion, result inference,
canonicalization, numerical edge policy, gradients, graph/compiler/planning/runtime/backend
behavior, factories, registries, dependencies, build/architecture changes, existing Java edits, or
later specs. Stop beyond nine paths or on architecture uncertainty.

Run every specified focused/aggregate test, Javadoc, javap/reflection/import/manual,
documentation/link/whitespace/scope/status check. Then hand the actual diff and evidence to a
separate clean-context documentation agent in the same change. It must inspect source/tests/
generated Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning, record related-
contract/capability/Compile API/Training API/architecture no-change conclusions, and rerun
validation.

Update task 0014E, model master plan, and roadmap only for planning status/evidence. Do not mark
0014E Complete until both passes succeed. Leave 0014F Draft without a specification. Do not commit
or push.
```

## Local decisions

- One typed family contains scalar arithmetic and all three clamp requests because they share
  one-input elementwise semantics with scalar parameters while remaining distinct from
  parameterless unary kinds.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [elementwise glossary](../../../../glossary.md#term-elementwise); [parameters glossary](../../../../glossary.md#term-parameter); [unary glossary](../../../../glossary.md#term-unary))</span>
- `CLAMP` is first-class model semantics. The public capability is not forced to expose the
  legacy builder's eager two-operation expansion as semantic identity.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [eager glossary](../../../../glossary.md#term-eager-evaluation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- One `ScalarValueAttrs` contract is sufficient for multiplier, exponent, lower bound, and upper
  bound because the typed kind gives the stored scalar its role. This avoids four structurally
  identical records without introducing an unbounded generic attributes container.<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent))</span>
- A separate `ClampRangeAttrs` is necessary because a two-bound inclusive range has a real
  ordering invariant and cannot be represented safely by an optional second scalar.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant))</span>
- Attributes store one exact [binary64](../../../../glossary.md#floating-point-precision-and-rounding) value per public `double` parameter and no alternate
  precision. Input-dependent conversion is not knowable without a Tensor and remains outside this
  semantic-only task.<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Strict inversion is rejected, while all other IEEE-754 values remain representable. Numerical
  edge behavior is not silently decided by the model record.<span data-glossary-navigation="true"> ([IEEE glossary](../../../../glossary.md#term-ieee))</span>
- Generic [`Operation`](../../../../glossary.md#operation) remains family-agnostic. Typed composition is documented and tested without
  a registry or compatibility service.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>

## Known limitations

- No public [Tensor](../../../../glossary.md#tensor) method consumes these semantics yet; task 0014F owns expression construction.
- The attributes do not choose how binary64 scalar parameters are converted for FLOAT32 or
  BFLOAT16 inputs.<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [scalar glossary](../../../../glossary.md#term-scalar); [parameters glossary](../../../../glossary.md#term-parameter); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- NaN and infinity parameters are representable, but their numerical execution behavior is not
  defined here.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- No canonicalization is performed for zero/one/minus-one multipliers, special exponents,
  unbounded ranges, or nested clamps.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- No [gradient](../../../../glossary.md#term-gradient) rule or clamp-boundary differentiation policy exists.
- Generic [`Operation`](../../../../glossary.md#operation) does not enforce kind-to-attributes compatibility by design.
- No [backend capability](../../../../glossary.md#backend-capability) or executable implementation is implied by semantic representation.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed the architecture/documentation rules; planning guide, [model](../../../../glossary.md#term-model) capability baseline,
master plan, and roadmap; completed [operation](../../../../glossary.md#operation) foundation/model tasks 0005 and 0006; completed
[binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/[unary](../../../../glossary.md#term-unary) family tasks 0014A–0014D; current [OperationKind](../../../../glossary.md#operationkind), [OperationAttrs](../../../../glossary.md#operationattrs), [NoOperationAttrs](../../../../glossary.md#nooperationattrs),
Operation, BinaryArithmeticKind, and UnaryElementwiseKind source/tests; and the read-only legacy
public [scalar](../../../../glossary.md#term-scalar)/clamp [Tensor](../../../../glossary.md#tensor) surface, operation [descriptors](../../../../glossary.md#term-descriptor), builders, [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) tests,
execution/[lowering](../../../../glossary.md#lowering) coverage, and [dtype](../../../../glossary.md#data-type--datatype) behavior.

Planning confirmed that the family can be represented through one typed enum and two [immutable](../../../../glossary.md#term-immutable)
records without a Tensor/[DataType](../../../../glossary.md#data-type--datatype) [dependency](../../../../glossary.md#term-dependency), existing contract change, [Gradle](../../../../glossary.md#term-gradle) change, or
architecture change. The new scalar package is cohesive and preserves acyclic direction toward
the foundational operation package.

Planning validation:

- `git diff --check` passed, and the three changed planning files contain no trailing whitespace.
- The [canonical](../../../../glossary.md#term-canonical-form) section scan found every required task-specification section.
- The relative Markdown-[target](../../../../glossary.md#term-target) scan resolved every local `.md` link in this task, the model master
  plan, and the roadmap.
- Status inspection found task 0014E [`Ready`](../../../../glossary.md#term-ready) exactly once in this specification, its model-master
  row, and its roadmap row.
- Scope inspection found exactly this new task plus the model master plan and roadmap changed; no
  Java, test, Gradle, AGENTS, architecture, API, glossary, or other module file changed during
  planning.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>
- No task-0014F specification exists; 0014F remains only a Draft queue entry.

Implementation and documentation validation:

- Implementation context `/root/implement_model_0014d` added exactly the three planned production
  types and one focused same-package test. Independent documentation context
  `/root/implement_model_0014b` then performed the mandatory fresh reread of architecture,
  documentation/planning rules, current APIs, glossary, completed task chain, actual source/tests,
  generated reports, bytecode, and the complete diff. It applied General plus API/Javadoc style to
  the Java and Tensor API review, Planning style to planning files, and Example format to the new
  Tensor API example.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseSemanticsTest`
  passed; its XML report contains 9 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` passed; 39 XML suites contain 296 tests with zero failures,
  errors, or skips. `./gradlew test` also passed for the repository with no failing task. An initial
  attempt to run those two commands concurrently collided only in Gradle's shared model test-result
  files; both required commands passed on the clean sequential rerun recorded here.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `./gradlew :modules:model:javadoc` passed. Generated Javadoc contains all three public types,
  every enum constant, record components, explicit accessors, canonical-constructor validation,
  exact primitive retention, special-value behavior, record equality/hashing, family pairings,
  and diagnostic-only text.
- `javap -p -c -s` confirms exactly one five-[constant](../../../../glossary.md#term-constant) enum and records with exactly one `double`
  component or ordered `double minValue, double maxValue` components, no additional project API or
  instance state, and the exact constructor descriptors. Clamp bytecode performs one primitive
  comparison before field assignment and throws `IllegalArgumentException` with the exact message
  only on the strict-inversion branch.
- Reflection tests confirm public/final enum and record shapes, exact components, fields, methods,
  constructor parameters, absence of nested types, inherited names, and typed distinction from
  binary <span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>`MUL` and `POW`. Composition tests cover all five kinds and exact attribute-reference
  retention through `Operation` without family compatibility validation.
- Raw-bit tests cover finite values, infinities, both signed zeros, and multiple positive and
  negative NaN payloads. They confirm unchanged accessor bits, accepted ordered/equal/signed-zero/
  infinity/NaN clamp bounds, rejection only of representative strict inversions, distinct signed
  zeros under generated record equality, and equal NaN components despite different retained raw
  payloads.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- Production imports point only from the scalar package to the foundational `OperationKind` or
  `OperationAttrs` contracts. Source, reflection, and diff scans found no Tensor, DataType, graph,
  storage, gradient, compiler, planning, prepare, runtime, backend, cost, fusion, route, registry,
  map, reflection, service, dependency, or build addition.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [fusion glossary](../../../../glossary.md#term-fusion); [registry glossary](../../../../glossary.md#term-registry))</span>
- The complete Tensor API example compiled with Java 26 against the model classes and produced the
  documented `MUL`/`ScalarValueAttrs[-0.0]` and `CLAMP`/`ClampRangeAttrs[0.0, 1.0]` descriptors.
  It does not construct a Tensor expression or imply execution support.
- The corrected local Markdown validator resolved every local file target and heading anchor in
  the five changed documentation/planning files. Fence counts are balanced, targeted
  trailing-whitespace scans found no matches, and `git diff --check` passed.
- Final scope contains exactly the planned nine paths: three production files, one focused test,
  Tensor API, glossary, this task, model master plan, and roadmap. No Compile API, Training API,
  capabilities, architecture/ADR/test, Gradle/build, existing Java/test, another-module,
  backend-conformance, integration-test, or task-0014F specification path changed.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Task 0014E is synchronized as Complete in this specification, the model master plan, and the
  roadmap. Task 0014F remains the next Draft frontier without a detailed specification.
- `OperationKind`, `OperationAttrs`, `NoOperationAttrs`, `Operation`, `BinaryArithmeticKind`, and
  `UnaryElementwiseKind` Javadocs remain accurate because the new types implement their existing
  open typed-kind, immutable-attributes, generic-composition, and family-separation contracts
  without changing them. Tensor and existing expression Javadocs remain accurate because this task
  adds no public Tensor method, descriptor behavior, provenance construction, or execution.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- The Compile API requires no edit because scalar semantic values do not add public expressions,
  graph capture, compiler inference, optimization, artifacts, or execution. The Training API
  requires no edit because no gradient eligibility, rule, autograd, optimizer, or training behavior
  changed. <span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [artifacts glossary](../../../../glossary.md#term-artifact); [autograd glossary](../../../../glossary.md#autograd); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>`capabilities.md` requires no edit because it already selects the scalar/clamp public
  capabilities and distinguishes semantic/public-expression support from later compiler/backend
  execution.
- `ARCHITECTURE.md`, focused architecture pages, ADRs, architecture tests, backend-conformance and
  integration tests, and build configuration require no edit because the change remains inside the
  existing model-owned operation-semantics boundary and adds no dependency, lifecycle, backend,
  numerical-execution, Java-toolchain, preview/incubator, or end-to-end rule.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>

## Implementation notes

- Added exactly `ScalarElementwiseKind`, `ScalarValueAttrs`, `ClampRangeAttrs`, and their focused
  same-package test without changing any existing Java source or test.
- The enum is a semantic identity vocabulary only. The records retain exact Java binary64 values;
  clamp validation is exactly the primitive strict-inversion rule, and generic <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [validation glossary](../../../../glossary.md#term-validation))</span>[`Operation`](../../../../glossary.md#operation) remains
  family-agnostic.
- The independent documentation pass made only Javadoc refinements in the two new record files,
  added the current scalar semantic model, pairings, edge cases, equality distinction, limits, and
  compiled example to the Tensor API, and updated existing glossary terms. It introduced no new
  reusable glossary term because the concrete types instantiate the existing operation-kind and
  operation-attributes concepts.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scalar glossary](../../../../glossary.md#term-scalar); [model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- No Java declaration, executable behavior, or test changed during the documentation pass.

## Completion summary

- Completed changes: Implemented and documented the five typed scalar arithmetic/clamp semantics,
  exact one-value attributes, and validated inclusive clamp-range attributes.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Files changed or created: Exactly three production files, one test, Tensor API, glossary, this
  task, model master plan, and roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Focused scalar semantics 9/9, all 296 model tests across 39 suites, model
  Javadoc, root tests, bytecode/reflection/import/absence checks, compiled documentation example,
  local links/anchors, fences, terminology, whitespace, exact scope/status, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`
  passed.
- Documentation-agent review: Clean documentation context `/root/implement_model_0014b` completed
  the independent pass using General, API/Javadoc, Planning, and Example-format guidance.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Tensor API and glossary now describe the current scalar semantic family
  and attributes while keeping public scalar Tensor expressions planned. Compile API, Training API,
  and capabilities remain accurate unchanged.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Javadoc review: All three new production contracts are final; the two record equality paragraphs
  now explicitly distinguish retained NaN payload bits from generated record equality. Existing
  operation foundation, binary/unary family, Tensor, and expression contracts remain accurate
  unchanged.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [operation glossary](../../../../glossary.md#operation); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [unary glossary](../../../../glossary.md#term-unary))</span>
- Glossary impact: Existing implementation-status, [`OperationAttrs`](../../../../glossary.md#operationattrs), [`OperationKind`](../../../../glossary.md#operationkind), and common
  distinction entries now include scalar kinds and attributes; no new reusable domain term was
  needed.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0014E. Task 0014F remains Draft without a specification.

Status: Complete
