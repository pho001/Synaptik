# Task 0018V: Multi-axis and Statistical Reductions

## Status

Complete

## Goal

Close the capability-reset reduction frontier with one coherent backend-independent model
contract for reductions over an ordered set of distinct axes and for first-class log-sum-exp,
variance, standard deviation, L1 norm, and L2 norm expressions.

This task extends the seven existing ordinary aggregate families to multi-axis forms, appends five
first-class meanings to `AggregateReductionKind`, adds immutable normalized multi-axis attributes,
and exposes exact public `Tensor` construction. It fixes portable result, empty-domain,
special-value, correction, and accumulation-type semantics without reading values, selecting an
execution algorithm, constructing gradients, capturing a graph, or claiming backend support.

## Why this remains one task

All selected operations share one caller-axis normalization contract, one structural reduced-Shape
rule, one result/provenance construction boundary, and one floating numerical policy. Variance and
standard deviation additionally share correction. Splitting semantic kinds, attributes, and
Tensor construction would temporarily leave operation signatures and public construction
inconsistent.

The implementation changes exactly 17 paths: five production, five test, and seven
documentation/planning paths. This is within the planning guide's 12–18-file upper guardrail and
is a justified atomic extension of one model family, not a cross-module exception. Task 0019
remains Draft and receives no detailed specification.

## Mental model and examples

For input Shape `[2, 3, 4]`:

```java
Tensor totals = input.sum(new int[] {2, 0}, false);
Tensor retained = input.mean(new int[] {-1, 0}, true);
```

Both normalize caller order to `[2, 0]`. `totals` has Shape `[3]`; `retained` has Shape
`[1, 3, 1]`. Shape derivation uses membership, not sorting. Reversing the requested order selects
the same Cartesian domain and Shape but remains a distinct attribute value and operation.

For floating input `[1, 2, 3]` reduced over its only axis:

```text
logSumExp                         ~= 3.407605964
variance(correction = 0)         = 2 / 3
variance(correction = 1)         = 1
standardDeviation(correction=0)  = sqrt(2 / 3)
l1Norm                            = 6
l2Norm                            = sqrt(14)
```

These are mathematical targets. They require numerical conformance to the selected meaning but do
not prescribe max subtraction, pass count, compensation, traversal, vectorization, or a kernel.

## Scope

- Append exactly `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM`, in
  that order, after `ARG_MIN` in `AggregateReductionKind`.
- Add `MultiAxisReductionAttrs(List<Integer> axes, boolean keepDimensions)` for ordinary
  multi-axis reductions, log-sum-exp, and L1/L2 norm.
- Add `StatisticalReductionAttrs(List<Integer> axes, boolean keepDimensions, long correction)`
  for variance and standard deviation.
- Add a multi-axis operation signature to all seven ordinary kinds while preserving every current
  full, single-axis, masked, and arg-extrema signature.
- Add the exact 26-method public `Tensor` surface below.
- Normalize caller axes in caller order, reject duplicates after normalization, and retain an
  immutable normalized order in attributes.
- Define an empty axis list as reduction over no axes, never full reduction or an error.
- Derive exact removal/retention Shapes while preserving unaffected `Dimension` references.
- Preserve exact input DataType and `requiresGrad`, leave layout unresolved, and attach no label or
  storage.
- Record a fresh one-output producer with exact one-input provenance and output index zero.
- Fix all numerical policies below, including the previously deferred floating and BOOL ordinary
  empty/special-value policies, without changing construction metadata.
- Finalize Javadocs, Tensor/Compile APIs, glossary, capabilities, planning state, and evidence in a
  mandatory separate clean-context documentation pass.
- Run the capability-reset checkpoint after executable behavior and documentation stabilize.

## Exact public Tensor surface

Add exactly:

```java
public Tensor sum(int... axes)
public Tensor sum(int[] axes, boolean keepDimensions)
public Tensor mean(int... axes)
public Tensor mean(int[] axes, boolean keepDimensions)
public Tensor prod(int... axes)
public Tensor prod(int[] axes, boolean keepDimensions)
public Tensor min(int... axes)
public Tensor min(int[] axes, boolean keepDimensions)
public Tensor max(int... axes)
public Tensor max(int[] axes, boolean keepDimensions)
public Tensor all(int... axes)
public Tensor all(int[] axes, boolean keepDimensions)
public Tensor any(int... axes)
public Tensor any(int[] axes, boolean keepDimensions)

public Tensor logSumExp(int... axes)
public Tensor logSumExp(int[] axes, boolean keepDimensions)
public Tensor variance(int... axes)
public Tensor variance(int[] axes, boolean keepDimensions)
public Tensor variance(int[] axes, boolean keepDimensions, long correction)
public Tensor standardDeviation(int... axes)
public Tensor standardDeviation(int[] axes, boolean keepDimensions)
public Tensor standardDeviation(int[] axes, boolean keepDimensions, long correction)
public Tensor l1Norm(int... axes)
public Tensor l1Norm(int[] axes, boolean keepDimensions)
public Tensor l2Norm(int... axes)
public Tensor l2Norm(int[] axes, boolean keepDimensions)
```

Varargs forms use `keepDimensions=false`. Variance and standard-deviation forms that omit
correction use population correction zero. The caller array is cloned and never retained.
Existing fixed zero-argument ordinary methods still win overload resolution, so `input.sum()` is
the unchanged full reduction while `input.sum(new int[0])` explicitly reduces no axes. A
zero-argument call to a new varargs family selects an empty axis list.

Add no `std`, generic `norm`, arbitrary-p overload, collection overload, options object, full
sentinel, static form, output-type override, accuracy mode, or backend hint. The declared public
Tensor method count becomes exactly 156.

## Semantic and attribute contracts

### Ordered normalized axes

Create exactly:

```java
public record MultiAxisReductionAttrs(
        List<Integer> axes,
        boolean keepDimensions) implements OperationAttrs
```

Its compact constructor validates `axes`, then each element in index order, with exact failures:

```text
NullPointerException("axes")
NullPointerException("axes[<index>]")
IllegalArgumentException("axes[<index>] must be non-negative: <axis>")
IllegalArgumentException("axes contains duplicate axis <axis> at index <index>")
```

It finishes with `List.copyOf(axes)`, and its explicit accessor returns that immutable snapshot.
An empty list is valid. The record stores no Shape/rank and cannot validate an upper bound.

Create exactly:

```java
public record StatisticalReductionAttrs(
        List<Integer> axes,
        boolean keepDimensions,
        long correction) implements OperationAttrs
```

It applies identical axis validation and snapshotting first, then rejects negative correction with
exact message `correction must be non-negative: <correction>`. Correction is `long` because
domain cardinality is not bounded by `int`. Fractional, negative, inferred, per-axis, or
data-type-dependent correction is not selected. Zero means population variance; one expresses the
usual sample estimator when the domain has at least two elements.

Public construction normalizes each raw axis exactly once with `Shape.normalizeAxis`, in caller
order, and rejects a duplicate immediately. An earlier duplicate therefore precedes a later
invalid raw axis. The duplicate message uses the normalized value and raw request index. Axis
order is semantic metadata for equality, diagnostics, transformation, and interchange, but does
not prescribe physical traversal or a sequential floating fold.

### Kind/signature pairings

Preserve every current constant and signature. Extend exact pairings:

- `SUM`/`MEAN`: current full, single-axis, and masked variants plus one-input/one-output
  `MultiAxisReductionAttrs`;
- `PROD`/`MIN`/`MAX`/`ALL`/`ANY`: current full and single-axis variants plus
  `MultiAxisReductionAttrs`;
- `ARG_MIN`/`ARG_MAX`: unchanged `ArgExtremaAttrs` only;
- `LOG_SUM_EXP`/`L1_NORM`/`L2_NORM`: exactly one one-input/one-output
  `MultiAxisReductionAttrs` variant; and
- `VARIANCE`/`STANDARD_DEVIATION`: exactly one one-input/one-output
  `StatisticalReductionAttrs` variant.

Do not add another kind family, registry, traits/result flag, compatibility alias, or hidden public
decomposition. Log-sum-exp is not stored as exp/sum/log, standard deviation is not stored as
variance/sqrt, and norms are not stored as unary/power/sum subgraphs.

### Shape and empty-axis meaning

When `keepDimensions=false`, remove every selected axis and retain unselected Dimensions by exact
reference in input order. When true, retain rank, replace each selected position with a new
`StaticDimension(1)`, and retain every unselected reference. Construct Shape once. Removing every
axis or reducing a scalar over an empty list produces canonical `Shape.scalar()`.

An empty list selects a one-value domain at each input position. It creates a fresh occurrence and
is never simplified to the input:

- ordinary sum/mean/product/minimum/maximum/all/any and log-sum-exp return the point value;
- L1/L2 norm return absolute value;
- population variance/standard deviation return positive zero; and
- correction at least one is invalid because `N=1`.

`keepDimensions` has no Shape effect for an empty list but remains part of attribute identity.
Existing full forms remain distinct reductions over every input axis.

### Data types and result metadata

Multi-axis ordinary eligibility exactly matches completed full/single-axis behavior:

| Kind | Accepted exact input/result type |
|---|---|
| `SUM`, `PROD`, `MIN`, `MAX` | BFLOAT16, FLOAT32, FLOAT64, INT32, or INT64 |
| `MEAN` | BFLOAT16, FLOAT32, or FLOAT64 |
| `ALL`, `ANY` | BOOL |

All five new kinds are floating-only and preserve exact input type. No integral/BOOL acceptance,
cast, promotion, or widening occurs.

Every result has the derived Shape, unresolved layout, no label/storage, and exact input
`requiresGrad`. Floating eligibility is not a gradient rule; integral/BOOL input is already false.
Each valid call uses one `TensorFactory.createDerived` delegation with exact input `[input]`, one
output descriptor, one fresh producer/Tensor, and provenance output index zero. Inputs are not
mutated.

## Numerical policy

### Common floating policy

Result DataType is the accumulator contract. The portable target is exact real arithmetic plus the
special rules below, rounded once to the result format using round-to-nearest, ties-to-even. An
implementation may use equal-or-wider intermediates, reassociate, parallelize, compensate, or use
another stable algorithm only when its observable result satisfies future conformance tolerance
and every exact special rule. It must not use narrower accumulation, hidden promotion in the
model, saturation, or promise bitwise equality. NaN payload/sign and signaling preservation are
unspecified.

Finite overflow produces signed infinity. Finite underflow rounds in the result format, including
signed zero where the mathematical result has a sign. Axis-list order cannot change the abstract
target. Existing 0018U1 integral semantics are inherited unchanged: exact-type modular sum/product
with reassociation, signed extrema, bounded empty identities, and each selected value included
once.

### Ordinary floating and BOOL reductions

- `SUM`: NaN propagates; opposite infinities produce NaN; otherwise infinity is preserved; empty
  result is positive zero. An exact-zero non-empty result is negative zero only when every selected
  value is negative zero; cancellation or any positive zero produces positive zero.
- `MEAN`: exact sum divided by count; NaN propagates; opposite infinities or empty domain produce
  NaN; a sole infinity sign is preserved; zero sign follows the sum rule because the divisor is
  positive.
- `PROD`: NaN propagates; zero times infinity produces NaN; otherwise zero/infinity sign follows
  multiplication parity; empty result is positive one.
- `MIN`/`MAX`: any NaN produces NaN; infinities order normally; minimum selects negative zero and
  maximum positive zero; empty minimum is positive infinity and empty maximum negative infinity.
- `ALL`: empty is true. `ANY`: empty is false.

These policies apply equally to existing full/single-axis and new multi-axis meanings. They add no
value execution or metadata change to completed public methods. Equal finite extrema produce the
same value, so no tie-selection or traversal-order policy is observable for value-returning
`MIN`/`MAX`; the existing explicit logical-index policy for `ARG_MIN`/`ARG_MAX` is unchanged.

### Log-sum-exp

For selected values `x_i`, `LOG_SUM_EXP` means `log(sum_i(exp(x_i)))` as a stable mathematical
target without selecting an algorithm. Empty domain is negative infinity. NaN produces NaN.
Positive infinity produces positive infinity unless NaN exists. An all-negative-infinity domain
produces negative infinity; other negative infinities contribute zero. A singleton finite domain,
including an empty-axis point domain, returns that value and preserves signed zero.

### Variance and standard deviation

For domain count `N`, correction `c`, and exact mean `mu`:

```text
variance = sum_i((x_i - mu)^2) / (N - c)
standardDeviation = sqrt(variance)
```

The denominator must be positive: `N > c`. Reject a statically known selected-domain count at
most correction. Zero extents yield `N=0`. Accept dynamic/expression selected extents; later
compiler/binding validation must prove `N > c` before execution. No NaN fallback, denominator
clamp, or sentinel represents an invalid domain.

NaN or any infinity produces NaN because deviation from the exact mean is undefined. A valid
constant finite domain, including signed zeros, produces positive zero. Standard deviation is the
non-negative principal root and never negative zero.

Compute static `N` from selected static Dimensions only with an early zero check and checked
multiplication. If a non-zero product overflows `long`, it exceeds every non-negative `long`
correction and is valid. Empty axes have `N=1`.

### L1 and L2 norms

`L1_NORM = sum_i(abs(x_i))`; `L2_NORM = sqrt(sum_i(x_i*x_i))` under the common abstract policy.
Empty domains produce positive zero; point domains produce absolute value; NaN produces NaN; any
infinity produces positive infinity unless NaN exists; finite results are non-negative and zero is
positive zero.

## Validation, failure, and construction order

Add one final package-private `TensorMultiAxisReductionExpressions` with no fields, nested types,
state, registry, or test hook; one private constructor; and package-private entries for ordinary,
uncorrected advanced, and corrected statistical construction. Private methods may own kind/type
validation, normalization, Shape derivation, selected-domain count, and common construction.

The package-private entry surface is exactly:

```java
static Tensor applyOrdinary(
        Tensor input, AggregateReductionKind kind, int[] axes, boolean keepDimensions)
static Tensor applyAdvanced(
        Tensor input, AggregateReductionKind kind, int[] axes, boolean keepDimensions)
static Tensor applyStatistical(
        Tensor input, AggregateReductionKind kind, int[] axes,
        boolean keepDimensions, long correction)
```

`applyOrdinary` accepts exactly `SUM`, `MEAN`, `PROD`, `MIN`, `MAX`, `ALL`, or `ANY` and otherwise
uses `kind must be SUM, MEAN, PROD, MIN, MAX, ALL, or ANY, but was <kind>`.
`applyAdvanced` accepts exactly `LOG_SUM_EXP`, `L1_NORM`, or `L2_NORM` and otherwise uses
`kind must be LOG_SUM_EXP, L1_NORM, or L2_NORM, but was <kind>`. `applyStatistical` accepts exactly
`VARIANCE` or `STANDARD_DEVIATION` and otherwise uses
`kind must be VARIANCE or STANDARD_DEVIATION, but was <kind>`.

Every entry follows this observable order:

1. null-check `input` (`input`);
2. null-check `kind` (`kind`);
3. null-check caller `axes` (`axes`);
4. validate the entry-specific permitted kind;
5. validate input DataType, reusing exact current ordinary messages and using
   `input must have a floating data type, but was <dataType>` for new kinds;
6. for statistics, reject negative correction with the exact correction message;
7. clone caller axes;
8. normalize each raw axis once in order and reject a normalized duplicate immediately;
9. for variance/standard deviation, reject a statically invalid denominator with exact message
   `reduction domain count <count> must be greater than correction <correction>`;
10. derive Shape, create exact attributes, descriptor, Operation, producer, and Tensor in order
    through one `createDerived` call.

Invalid-kind messages list the permitted constants for that entry. `Shape.normalizeAxis` supplies
invalid-axis exception type/message. Every failure through static denominator validation consumes
no Tensor identity and occurs before attributes/descriptor/producer allocation. Identifier
exhaustion occurs only at final factory delegation after local immutable metadata and propagates
the existing `IllegalStateException`.

Every public method delegates exactly once with exact kind, array, retention, and correction; it
does not call another public overload.

## Out of scope

- arbitrary/zero/infinity p-norm, complex magnitude, weighted statistics, median, quantiles,
  covariance, covariance matrices, or multi-output statistics
- cumulative product or another scan; cumulative variance; multi-axis arg extrema or masked
  reductions; changes to cumulative sum, softmax, masking, or arg-extrema
- normalization layers, attention, loss, pooling, convolution, sorting, or top-K
- accumulator/output override, integral mean/statistics/norms, BOOL numeric interpretation,
  implicit cast, unsigned/complex/quantized types, or new DataType
- algorithm/pass-count/tree/traversal/vector/kernel/tolerance selection or eager evaluation
- gradients/backward kinds/autograd, graph capture, compiler validation/canonicalization, planning,
  lowering, prepare, runtime, execution, ONNX, tracing, storage, or materialization
- simplification/decomposition/CSE/producer sharing for empty-axis or other calls
- changes to Shape, Dimension, DataType, TensorFactory, producer/provenance, operation foundations,
  existing helpers, Gradle, dependencies, architecture, another module, or detailed task 0019

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
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
- [Task 0016F](0016f-masked-reduction-semantics-and-axis-mapping.md)
- [Task 0016F1](0016f1-masked-sum-and-mean-tensor-expressions.md)
- [Task 0016G](0016g-cumulative-sum-semantic-kind-and-attributes.md)
- [Task 0016H](0016h-cumulative-sum-tensor-expressions.md)
- [Task 0016I](0016i-softmax-semantic-kinds-and-attributes.md)
- [Task 0016J](0016j-softmax-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018M](0018m-symbolic-extent-expressions.md)
- [Task 0018T1](0018t1-unary-numeric-gaps-and-floating-diagnostics.md)
- [Task 0018U](0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Task 0018U1](0018u1-integral-reductions-and-arg-min-normalization.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Work remains inside model-owned Tensor, Shape-derived metadata, operation semantics, and
  immutable provenance; Tensor remains public API state, not IR or executable state.
- Kinds contain mathematics/signatures only, never backend, algorithm, gradient, cost, or route.
- Attributes are immutable normalized values with no Tensor, Shape, rank, descriptor, storage,
  graph, or execution state.
- Package direction remains `model.tensor -> model.operation.reduction`, datatype, and shape; the
  reduction package must not import Tensor.
- Compiler later owns graph-wide revalidation, dynamic-domain proof, canonicalization, and
  gradients; backend prepare owns lowering and algorithm choice.
- No dependency, module boundary, lifecycle, architecture contract, or focused architecture change.
  Stop and report if implementation requires one.

## Package impact

Existing packages only:

- `io.github.pho001.synaptik.model.operation.reduction` owns aggregate meanings and immutable
  axes/correction.
- `io.github.pho001.synaptik.model.tensor` owns public methods, local validation/Shape/result
  construction, provenance, and factory delegation.
- `io.github.pho001.synaptik.model.shape` supplies immutable Dimensions/Shapes and normalization.
- `io.github.pho001.synaptik.model.datatype` supplies category eligibility.

No package is added or moved. Type placement:

- `AggregateReductionKind` — extended vocabulary/signatures.
- `MultiAxisReductionAttrs` — shared ordinary/log-sum-exp/norm axes and retention.
- `StatisticalReductionAttrs` — variance/standard-deviation axes, retention, correction.
- `Tensor` — exact 26-method public surface.
- `TensorMultiAxisReductionExpressions` — package-private construction boundary.
- `ReductionSemanticsTest` — semantic/signature/record coverage.
- `TensorMultiAxisReductionTest` — focused public/helper behavior.
- `TensorTest` — exact public surface/modifiers only.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/AggregateReductionKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/MultiAxisReductionAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/reduction/StatisticalReductionAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorMultiAxisReductionExpressions.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/reduction/ReductionSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMultiAxisReductionTest.java`

Documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inconsistent: Training API; DataType/Dimension/Shape, Operation,
producer/provenance/factory, full/single/masked/arg-extrema/scan/softmax Javadocs/tests;
architecture/ADRs/tests; conformance/integration; Gradle and other modules.

## Maximum scope

At most five production, five test, and seven documentation/planning files: exactly 17 paths.
`Tensor`/`TensorTest` may change only for the exact 26 methods, imports/Javadocs, method count/set,
descriptors, and modifiers. `OperationSignatureTest` changes only for the exact new 0018V
signatures. `TensorBinaryArithmeticTest` changes only for the public Tensor count from 130 to 156.
Every other existing focused helper/test remains unchanged.

This atomic scope is justified because enum signatures, two attributes, public construction,
focused tests, and documentation must agree in one compilable state. The two-test expansion was
explicitly authorized after the first final model run exposed stale global signature and public
method-count assertions. Stop for another Java/test type, documentation file, 18th path,
cross-module work, or architecture change.

## Javadoc requirements

- Document all five new kinds, formulas, pairings, empty/special policies, types, and boundaries.
- Update ordinary kinds for multi-axis pairing and completed floating/BOOL numerical policy while
  preserving 0018U1 integral semantics.
- Document both records' normalization, order, empty-list meaning, snapshot ownership, validation,
  equality, and diagnostic text.
- Document every new Tensor method's axes, correction/default, Shape/type/metadata/provenance,
  freshness, numerical target, failures/order/ID effects, and deferred behavior.
- Document helper/member validation, selected count, construction order, allocation, and effects.
- Review related Javadocs and record reasoned no-change conclusions.

## Acceptance criteria

- Exact five kinds append in order; existing constants/signatures remain stable.
- Exact signature pairings/cardinalities work and wrong attribute classes fail through Operation.
- Both records expose exact components/API, immutable snapshots, messages/order, and no extra API.
- Exactly 26 Tensor methods are added; public count is 156; no unrelated descriptor/name changes.
- Existing full/single/masked/arg-extrema/scan/softmax behavior remains unchanged.
- Axes normalize once in order; duplicates fail; normalized order is retained and mutation-safe.
- Empty/some/all axes and scalar/static-zero/dynamic/expression Shapes derive exact results and
  preserve required Dimension identity.
- Ordinary eligibility and 0018U1 integral behavior remain; new families are floating-only.
- Correction defaults/explicit values, static checks, dynamic acceptance, failures/order match.
- Results have exact type/eligibility, unresolved layout, no label/storage, fresh exact producer
  and one-input provenance index zero; inputs remain unchanged.
- Numerical policies are consistent across Javadocs, APIs, glossary, and capabilities without
  selecting an algorithm or executing values.
- No decomposition, eager data/storage, gradients, compiler/backend/runtime, registry, dependency,
  Gradle, architecture, or other-module change.
- Focused tests, one final model suite, checkpoint root tests, Javadoc/docs validation pass.
- Separate clean-context docs pass finalizes authorized documentation and no-change conclusions.
- 0018V becomes Complete only after both passes; 0018U1 stays Complete; 0019+ stay Draft without
  detailed specs.

## Tests / validation

Required focused command:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.reduction.ReductionSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorMultiAxisReductionTest --tests io.github.pho001.synaptik.model.tensor.TensorTest --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest --tests io.github.pho001.synaptik.model.tensor.TensorBooleanReductionTest --tests io.github.pho001.synaptik.model.tensor.TensorArgMaxExpressionTest --tests io.github.pho001.synaptik.model.tensor.TensorMaskedReductionTest --tests io.github.pho001.synaptik.model.tensor.TensorSoftmaxExpressionTest
```

After executable Java stabilizes, run exactly one final model suite:

```bash
./gradlew :modules:model:test
```

Tests cover exact API/helper/enum/record shapes, signatures, all kinds/types/retention/defaults,
validation/messages/ID effects, caller mutation, axis cases, Shape/Dimension identity, result
metadata/provenance/freshness, contract examples/special policies without storage evaluation, and
unchanged prior behavior.

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Capability-reset checkpoint after the combined diff:

```bash
./gradlew test
```

Also validate local links/anchors, balanced fences, terminology, generated Javadoc, final newlines,
trailing whitespace, exact 17 paths, package placement, method count, synchronized status,
completed history, 0018U1 Complete, final 0018V Complete, and no task-0019 spec. Use manual
reflection/bytecode checks only for a concrete risk not covered by stable automated tests.

## Dependencies

- 0001: DataTypes; 0002/0018M: Dimensions, Shapes, normalization, symbolic extents.
- 0006–0007, 0011–0013, 0018L: Operation, descriptor, Tensor, IDs, producer/provenance/factory.
- 0016A–0016J including 0016F1: reduction, masked, scan, softmax conventions.
- 0018K: family-owned exact signatures.
- 0018T1: floating special-value vocabulary.
- 0018U/0018U1: integral arithmetic/reductions and shared arg-extrema policies to preserve.

## Follow-up tasks

- 0019 remains Draft for linear algebra/attention after this checkpoint.
- 0020–0022 remain Draft consumers of this reduction foundation.
- Compiler later owns capture/revalidation/canonicalization/autograd; backend/conformance later
  owns algorithms, tolerances, lowering, storage, and execution.
- Arbitrary p-norm, cumulative product, multi-output statistics, and normalization remain deferred.

Do not create another detailed task specification during 0018V.

## Architecture impact

Expected impact: None. If implementation requires `ARCHITECTURE.md`, focused architecture,
dependency, boundary, or other-module changes, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in Synaptik without commit or push. Read AGENTS.md, ARCHITECTURE.md, current architecture,
documentation/planning rules, roadmap, model capabilities/master plan, completed tasks 0016A–J
including 0016F1, tasks 0018K/0018M/0018T1/0018U/0018U1, and task 0018V.

Implement docs/planning/modules/model/tasks/0018v-multi-axis-and-statistical-reductions.md exactly
inside its 17 authorized paths. Preserve current contracts. Stop on architecture uncertainty,
scope overflow, or need for another type/module.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and exact Java evidence to a separate clean-context documentation agent in the same change.
That agent finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and docs checks,
reusing successful Java evidence. Run the root checkpoint after the combined diff. Synchronize
status only after all criteria pass; keep 0019+ Draft without specs.
```

## Documentation-agent handoff

Give the separate clean-context documentation agent this task, complete diff, exact focused/final
model evidence/counts and whether Java changed afterward, all affected contracts/policies, the
seven authorized docs, and required Javadoc/Markdown/scope/status/checkpoint validation.

It independently reads AGENTS, architecture, documentation rules and General/API-Javadoc/
Planning/Example profiles, this task, actual source/tests/generated Javadoc, Tensor/Compile/
Training APIs, glossary, capabilities/master/roadmap, and directly related reduction/softmax
contracts. It finalizes affected Javadocs/docs and glossary terminology for multi-axis reduction,
correction, log-sum-exp, and norms.

It does not repeat successful Java tests unless executable Java changes or evidence is stale for a
recorded concrete reason. It records reasoned no-change conclusions for Training API,
DataType/Dimension/Shape, operation/provenance/factory foundations, completed reduction variants,
architecture/ADRs/tests, conformance/integration, Gradle, and other modules.

## Local decisions

- Caller axes are copied, normalized once against the input rank in encounter order, and retained
  as an immutable list. Duplicate detection follows normalization, so equivalent positive and
  negative spellings of one axis conflict.
- An empty axes list is a point-domain reduction: result Shape equals input Shape and no ordinary
  reduction identity is substituted. Reducing every axis produces a scalar unless retained
  dimensions were requested.
- Ordinary multi-axis signatures use `MultiAxisReductionAttrs`; variance and standard deviation
  use `StatisticalReductionAttrs` so non-negative integral correction remains explicit semantic
  data. Log-sum-exp and norms use the ordinary multi-axis attribute shape.
- Static reduced element count must exceed correction before a statistical expression is
  constructed. A dynamic count is accepted for later compiler/runtime validation because the
  model cannot prove the boundary.
- All five new families accept floating input only. Their result preserves the exact input type,
  while ordinary reductions retain the eligibility and exact result-type contracts completed by
  0018U1.
- Multi-axis construction is owned by the package-private, field-free
  `TensorMultiAxisReductionExpressions`; the public `Tensor` surface remains the only caller
  entry point.
- User authorization expanded the original stale-test repair to include both
  `OperationSignatureTest` and `TensorBinaryArithmeticTest`, preserving the exact 17-path scope.

## Known limitations

- This task constructs storage-free model expressions only. It does not evaluate values, select a
  numerical algorithm, define tolerances, lower operations, claim backend support, or attach
  gradients.
- Dynamic statistical domains can be represented even when a later binding makes
  `reducedElementCount <= correction`; the compiler/runtime layer must reject that bound case.
- Arbitrary p-norm, cumulative product, multi-output statistics, and normalization remain
  deferred. Task 0019 and all later tasks remain Draft and have no new detailed specification.
- No public list-taking overload is added; callers use the exact array and varargs surface defined
  here, while semantic attributes retain immutable lists internally.

## Validation evidence

- The implementation context passed the required eight-suite focused command: 78 tests across
  eight suites, with zero failures. Its first final model run exposed only two stale assertions in
  the two user-authorized existing tests; after those assertions were aligned, the single final
  model result passed 744 tests across 91 suites with zero failures, errors, or skips.
- The separate clean-context documentation pass `/root/task_0018u1_docs_recovery` reused that
  stable Java evidence and made no executable Java changes. `./gradlew :modules:model:javadoc`
  succeeded, and generated pages for `Tensor`, `AggregateReductionKind`, and both new public
  attribute records were inspected. The package-private helper correctly has no public page.
- A Java 26 compile-and-run metadata example produced `Shape[3]`, `Shape[1, 3, 1]`, normalized
  axes `[2, 0]`, correction `1`, output index `0`, and confirmed caller-array mutation isolation
  and absent storage.
- Reflection verified 156 public `Tensor` methods, the exact 26 new descriptors, five appended
  kinds, both record shapes, and the final/package-private/field-free helper with three package
  entries. `javap -p -s` confirmed both record descriptors and the helper boundary; source/import
  review found no compiler, backend, runtime, reflection, registry, service-loader, or storage
  coupling in the new semantic types.
- Markdown validation passed 516 local links including 142 anchors with zero errors. Balanced
  fences, final newlines, trailing whitespace, terminology, generated-Javadoc content, exact
  17-path scope, synchronized Complete status, 0018U1 Complete history, and absence of a task-0019
  specification were checked.
- The required capability-reset checkpoint `./gradlew test` succeeded with 36 actionable tasks,
  two executed and 34 up-to-date. `git diff --check` passed after the combined implementation and
  documentation diff.

## Implementation notes

- Appended `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM` without
  reordering prior aggregate kinds, and added each exact family-owned signature.
- Added immutable `MultiAxisReductionAttrs` and `StatisticalReductionAttrs`, plus focused shared
  construction for normalization, Shape derivation, static correction checks, and fresh
  one-output provenance.
- Added the exact 26 public `Tensor` entries: fourteen ordinary multi-axis overloads and twelve
  log-sum-exp/statistical/norm overloads. Focused tests cover signatures, eligibility, Shape and
  Dimension identity, validation order/messages, mutation safety, metadata, provenance, and
  freshness.
- The documentation pass finalized all five affected production Javadocs, Tensor and Compile API
  references, glossary terminology, capability baseline, task evidence, master plan, and roadmap.
- Training API, DataType/Dimension/Shape foundations, operation/provenance/factory foundations,
  completed single-axis/masked/scan/softmax variants, architecture and ADRs, architecture tests,
  backend conformance, integration tests, Gradle configuration, and other modules required no
  changes because this task adds model-only expression metadata within existing boundaries.

## Completion summary

Completed ordered multi-axis forms for all seven ordinary reduction families and first-class
floating log-sum-exp, corrected variance/standard deviation, and L1/L2 norm model expressions.
The exact 17-path change contains five production files, five test files, and seven
documentation/planning files. Focused, final model, root checkpoint, Javadoc, Java 26 example,
reflection/bytecode/source, generated-page, Markdown, scope/status, formatting, whitespace, and
diff validation passed. No unresolved issue or required follow-up remains inside task 0018V;
deferred compiler, runtime, backend, gradient, and later capability work stays in its owning Draft
tasks.

Status: Complete
