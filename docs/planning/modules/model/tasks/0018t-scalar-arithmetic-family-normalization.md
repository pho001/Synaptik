# Task 0018T: Scalar Arithmetic Family Normalization

## Status

Complete

## Goal

Complete one uniform floating-point elementwise arithmetic vocabulary across Tensor-to-Tensor and
Tensor-to-scalar expressions before the public model API stabilizes.

The final arithmetic families are parallel:

```text
Tensor-to-Tensor                    Tensor-to-scalar
----------------                   ----------------
ADD  + NoOperationAttrs            ADD  + ScalarValueAttrs
SUB  + NoOperationAttrs            SUB  + ScalarValueAttrs
MUL  + NoOperationAttrs            MUL  + ScalarValueAttrs
DIV  + NoOperationAttrs            DIV  + ScalarValueAttrs
MIN  + NoOperationAttrs            MIN  + ScalarValueAttrs
MAX  + NoOperationAttrs            MAX  + ScalarValueAttrs
POW  + NoOperationAttrs            POW  + ScalarValueAttrs
```

Keep `CLAMP` as the one distinct two-bound scalar operation. Remove first-class `CLAMP_MIN` and
`CLAMP_MAX`: public `clampMin` is a convenience for scalar `MAX`, and public `clampMax` is a
convenience for scalar `MIN`.

Atomically rename only the public elementwise Tensor-to-Tensor extrema methods from `min(Tensor)`
and `max(Tensor)` to `minimum(Tensor)` and `maximum(Tensor)`. Use the same `minimum` and `maximum`
names for scalar overloads. Full and axis reductions remain `min(...)` and `max(...)`, so Java
call sites cannot confuse an integral scalar literal with an axis number.

This task constructs metadata only. It does not execute arithmetic, inspect values, implement
gradients, or select a backend route.

## Mental model

One method name identifies one elementwise relationship regardless of whether the right operand
is another Tensor or an exact typed scalar:

```java
Tensor tensorSum = left.add(right);
Tensor scalarSum = left.add(ScalarValue.float32(2.0f));

Tensor tensorMinimum = left.minimum(right);
Tensor scalarMinimum = left.minimum(ScalarValue.float32(0.0f));

Tensor reducedMinimum = left.min();
```

The first pair creates different typed operation families because their structural inputs differ:

```text
tensorSum
  operation = BinaryArithmeticKind.ADD + NoOperationAttrs
  inputs    = [left, right]

scalarSum
  operation = ScalarElementwiseKind.ADD + ScalarValueAttrs(FLOAT32 2.0)
  inputs    = [left]
```

The scalar parameter is immutable operation metadata rather than an eager rank-zero Tensor. This
avoids allocating a second Tensor identity and storage merely to represent a constant parameter,
while keeping the compiler-visible operation explicit.

`minimum` and `maximum` distinguish pairwise element selection from aggregate reduction:

```text
input.minimum(other)  -> one value for every broadcast output position
input.minimum(value)  -> one value for every input position
input.min()            -> one aggregate value over the full input
input.min(axis)        -> one aggregate value per remaining coordinate
```

## Selected API

### Tensor-to-Tensor arithmetic

The final public methods are exactly:

```java
public Tensor add(Tensor right)
public Tensor sub(Tensor right)
public Tensor mul(Tensor right)
public Tensor div(Tensor right)
public Tensor minimum(Tensor right)
public Tensor maximum(Tensor right)
public Tensor pow(Tensor right)
```

Remove `min(Tensor)` and `max(Tensor)` atomically. Add no alias, deprecated bridge, forwarding
method, or alternate spelling. The existing reduction overloads named `min` and `max` remain
unchanged.

### Tensor-to-scalar arithmetic

Every arithmetic kind has one exact typed overload and one retained exact-FLOAT64 convenience:

```java
public Tensor add(ScalarValue value)
public Tensor add(double value)
public Tensor sub(ScalarValue value)
public Tensor sub(double value)
public Tensor mul(ScalarValue value)
public Tensor mul(double value)
public Tensor div(ScalarValue value)
public Tensor div(double value)
public Tensor minimum(ScalarValue value)
public Tensor minimum(double value)
public Tensor maximum(ScalarValue value)
public Tensor maximum(double value)
public Tensor pow(ScalarValue value)
public Tensor pow(double value)
```

The `double` overloads construct `ScalarValue.float64(value)` and delegate to their typed
overloads. They never infer the receiver type or narrow the parameter. Therefore a FLOAT32 or
BFLOAT16 receiver must use an exact matching `ScalarValue`.

### Clamp conveniences

Retain these existing methods:

```java
public Tensor clamp(ScalarValue minValue, ScalarValue maxValue)
public Tensor clamp(double minValue, double maxValue)
public Tensor clampMin(ScalarValue minValue)
public Tensor clampMin(double minValue)
public Tensor clampMax(ScalarValue maxValue)
public Tensor clampMax(double maxValue)
```

`clamp` remains one first-class `ScalarElementwiseKind.CLAMP` operation with
`ClampRangeAttrs`. The one-bound methods are conveniences:

```java
clampMin(value) -> maximum(value) -> ScalarElementwiseKind.MAX
clampMax(value) -> minimum(value) -> ScalarElementwiseKind.MIN
```

Each valid convenience call still creates exactly one fresh producer. It must not create an
intermediate expression, eager scalar Tensor, or hidden composition.

## Final scalar semantic vocabulary

`ScalarElementwiseKind` must contain exactly these constants in this declaration order:

```java
ADD,
SUB,
MUL,
DIV,
MIN,
MAX,
POW,
CLAMP
```

The first seven kinds accept exactly `ScalarValueAttrs`, one logical Tensor input, and one output.
`CLAMP` accepts exactly `ClampRangeAttrs`, one logical Tensor input, and one output. The family
continues to own these variants through `OperationKind.signatures()`; do not add a registry,
switch in `Operation`, reflective discovery, or string dispatch.

`ScalarValueAttrs` documents its one value as the addend, subtrahend, multiplier, denominator,
minimum candidate, maximum candidate, or exponent selected by the owning kind. It retains the
exact supplied `ScalarValue` reference and does no receiver-aware validation.

## Numerical semantics selected here

This task records operation meaning but adds no evaluator. Future compiler, prepare, backend, and
conformance work must preserve these model-level choices.

### Arithmetic and power

- `ADD`, `SUB`, `MUL`, and `DIV` mean the ordinary ordered IEEE-754 floating operation in the
  result data type. NaN, infinity, signed-zero, overflow, and underflow classifications follow
  that operation. No NaN payload, exact machine instruction, intermediate precision, or bitwise
  reproducibility is promised by this model-only task.
- `POW` retains its existing ordered base/exponent mathematical request. This task does not choose
  an approximation algorithm or extend its numerical contract beyond the current portable
  meaning.
- Scalar operations preserve the input data type. Tensor-to-Tensor operations preserve the
  current floating promotion hierarchy and broadcast behavior.

### Minimum, maximum, and clamp

The same extrema meaning applies to Binary and Scalar `MIN`/`MAX`:

- if either compared value is NaN, the result is NaN; a payload or sign is not promised;
- negative infinity orders below every finite value and positive infinity above every finite
  value;
- `minimum(-0.0, +0.0)` is negative zero;
- `maximum(-0.0, +0.0)` is positive zero; and
- otherwise ordinary numerical order selects the lesser or greater value.

`CLAMP(x, lower, upper)` means `minimum(maximum(x, lower), upper)` under those rules, but remains
one first-class semantic operation rather than two stored producers. A NaN endpoint is still a
structurally valid exact `ScalarValue`; the selected extrema policy makes the eventual result NaN
where that endpoint participates. `clampMin` and `clampMax` are public conveniences over `MAX` and
`MIN`, respectively.

This task selects no gradient or subgradient convention at equality, a clamp boundary, NaN, or
infinity. Compiler-owned autograd planning must make that decision later.

## Scope

- Expand `ScalarElementwiseKind` atomically to the exact final eight-kind vocabulary.
- Remove `CLAMP_MIN` and `CLAMP_MAX` without compatibility aliases.
- Update `ScalarValueAttrs` Javadoc for all seven scalar roles.
- Preserve `ClampRangeAttrs` declaration, validation, equality, and behavior unchanged.
- Keep one exact family-owned signature for scalar-value variants and one for range-clamp.
- Add exact typed and exact-FLOAT64 scalar overloads for ADD, SUB, DIV, MIN, and MAX.
- Preserve the existing scalar MUL and POW overloads with the same construction behavior.
- Make `clampMin` and `clampMax` delegate to scalar MAX and MIN with one producer.
- Rename public Tensor-to-Tensor `min(Tensor)` and `max(Tensor)` to `minimum(Tensor)` and
  `maximum(Tensor)` without changing `BinaryArithmeticKind.MIN` or `.MAX`.
- Preserve full and axis reduction `min`/`max` methods unchanged.
- Preserve all existing Binary arithmetic validation, promotion, broadcasting, result metadata,
  producer/provenance, freshness, and failure ordering.
- Preserve all existing Scalar validation, exact typed-value matching, result metadata,
  producer/provenance, freshness, and failure ordering.
- Document the selected extrema policy in affected semantic Javadocs and current API guides.
- Update exact enum, operation-signature, helper-surface, public-surface, provenance, delegation,
  failure-order, absence, and freshness tests.
- Update the Tensor API, Compile API current-surface summary, glossary, capability baseline, task,
  model master plan, and roadmap in the required independent documentation pass.

## Out of scope

- `rsqrt`, `log1p`, `expm1`, `isFinite`, `isNaN`, or `isInf`; Draft task 0018T1 owns them
- integral or BOOL arithmetic, comparison, reduction, clamp, or public scalar eligibility; task
  0018U owns the selected signed-integral expansion
- `argMin`, multiple-axis reductions, `logSumExp`, variance, standard deviation, or norms
- general scalar conversion, numeric promotion between a Tensor and `ScalarValue`, inferred scalar
  type, boxed `Number`, generic value payloads, or caller-selected rounding
- eager rank-zero Tensor creation for a scalar parameter
- float-specific, BFLOAT16-specific, primitive integral, or BOOL public scalar overloads
- changes to `ScalarValue`, `ClampRangeAttrs`, `DataType`, `DataTypePromotion`, Shape,
  `TensorDescriptor`, `TensorProducer`, `TensorProvenance`, or `TensorFactory`
- numerical evaluation, host-storage reads or writes, constant folding, algebraic simplification,
  canonicalization, common-subexpression elimination, or result interning
- gradient formulas, subgradient policy, backward graph construction, optimizer, or training
- compiler capture, graph validation, prepare lowering, backend support, kernel selection, fusion,
  runtime behavior, execution, tracing, or engine behavior
- dependencies, Gradle, Java version, preview/incubator features, architecture changes, focused
  architecture documentation, architecture tests, another module, or a detailed 0018T1-or-later
  task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially `modules/model` ownership of
  public Tensor and backend-independent operation semantics
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md)
- [Task 0014B](0014b-binary-arithmetic-tensor-expressions.md)
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md)
- [Task 0014F](0014f-scalar-arithmetic-and-clamp-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0018P](0018p-elementwise-semantic-cleanup.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns both arithmetic kind families, exact scalar attributes, public Tensor
  expression validation, descriptor derivation, and producer/provenance construction.
- `Operation` continues to validate exact family-owned attributes and occurrence cardinality. It
  must not inspect input descriptors or scalar data types.
- The Tensor scalar-construction boundary owns exact receiver/value data-type matching because it
  has the input descriptor.
- Tensor-to-Tensor and Tensor-to-scalar kinds remain distinct typed semantic values even when their
  enum constant names match.
- `Tensor` remains mutable public API state rather than graph IR. Every valid expression creates a
  fresh storage-free result with one immutable producer occurrence and provenance output index
  zero.
- The public facade may delegate to package-private cohesive expression boundaries; it must not
  absorb inference, execution, or backend decisions.
- Compiler owns gradient rules and backward graph construction. Backend prepare owns lowering,
  specialization, fusion, and kernel selection.
- Runtime hot paths never consume these operations directly.
- No module/package dependency, registry, service locator, reflective discovery, or architecture
  rule is added.
- If implementation requires an attribute variant beyond the two existing records, a new public
  production type, cross-type scalar conversion, or another module, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.elementwise.binary` — retains the seven
  Tensor-to-Tensor kinds and receives only terminology/numerical-policy Javadoc updates.
- `io.github.pho001.synaptik.model.operation.elementwise.scalar` — owns the normalized eight-kind
  Tensor-to-scalar family and its two existing attribute records.
- `io.github.pho001.synaptik.model.tensor` — owns the public fluent methods and the two existing
  cohesive construction helpers.
- `io.github.pho001.synaptik.model.operation` — owns unchanged signature and operation validation
  consumed by both families.

Packages added, moved, or removed:

- None.

Type placement:

- `ScalarElementwiseKind` and `ScalarValueAttrs` remain in the scalar semantic package because the
  scalar parameter is operation metadata rather than a second Tensor input.
- `TensorScalarExpressions` remains the one package-private field-free scalar construction
  boundary.
- `TensorBinaryExpressions` remains the one package-private field-free binary construction
  boundary and requires no executable change.
- `Tensor` remains the public fluent facade; the added overloads must remain thin delegations.

## Required implementation contracts

### Scalar kind and signature migration

After implementation:

- `ScalarElementwiseKind.values()` is exactly
  `[ADD, SUB, MUL, DIV, MIN, MAX, POW, CLAMP]`;
- `valueOf("CLAMP_MIN")` and `valueOf("CLAMP_MAX")` fail with ordinary enum lookup failure;
- ADD, SUB, MUL, DIV, MIN, MAX, and POW each return the exact immutable
  `ScalarValueAttrs`/one-input/one-output signature list;
- CLAMP returns the exact immutable `ClampRangeAttrs`/one-input/one-output signature list;
- incompatible kind/attributes combinations fail through the existing `Operation` contract; and
- no new field, nested type, registry, string mapping, or mutable signature state appears.

### Scalar helper contract

Keep `TensorScalarExpressions` field-free, package-private, final, and non-instantiable. Its method
surface remains exactly:

```java
static Tensor applyScalar(Tensor input, ScalarElementwiseKind kind, ScalarValue value)
static Tensor applyClamp(Tensor input, ScalarValue minValue, ScalarValue maxValue)
private static Tensor create(Tensor input, DataType dataType, Operation operation)
```

`applyScalar` uses this exact validation and construction order:

1. null-check `input`, `kind`, then `value`, with the parameter name as the message;
2. reject only `CLAMP`, with exact message `CLAMP requires ClampRangeAttrs`;
3. require a floating input, with existing exact message
   `input must be a floating data type, but was <type>`;
4. require exact scalar/input type equality, with existing exact message
   `scalar data type <scalar> must match input data type <input>`;
5. construct one `ScalarValueAttrs` retaining the exact value reference;
6. construct one `Operation` retaining the exact kind and attributes;
7. construct the current unresolved descriptor with exact input type, exact Shape reference, and
   unchanged `requiresGrad`;
8. call `TensorFactory.createDerived` exactly once with no label and ordered `[input]`.

`applyClamp` preserves its current validation, construction order, failure messages, exact bound
references, and first-class CLAMP producer unchanged.

### Public delegation

- Every Tensor-to-Tensor method delegates exactly once to `TensorBinaryExpressions.apply` with the
  exact kind. Only the two public extrema names change.
- Every typed scalar arithmetic method delegates exactly once to
  `TensorScalarExpressions.applyScalar` with the exact kind.
- Every scalar `double` overload constructs exactly one `ScalarValue.float64(value)` and delegates
  to its corresponding typed overload.
- Typed `clampMin` delegates exactly once to typed `maximum`; typed `clampMax` delegates exactly
  once to typed `minimum`.
- Double `clampMin` delegates through the selected FLOAT64 maximum path; double `clampMax`
  delegates through the selected FLOAT64 minimum path.
- `clamp` continues to delegate to `applyClamp`, not to two extrema calls.
- No successful method returns the receiver, reuses an existing producer, creates storage, or
  simplifies identity-like values.

### Public naming and absence

After implementation:

- reflection finds `minimum(Tensor)`, `maximum(Tensor)`, `minimum(ScalarValue)`,
  `maximum(ScalarValue)`, `minimum(double)`, and `maximum(double)`;
- reflection does not find `min(Tensor)`, `max(Tensor)`, `min(ScalarValue)`, `max(ScalarValue)`,
  `min(double)`, or `max(double)`;
- reflection still finds reduction `min()`, `min(int)`, `min(int, boolean)`, `max()`, `max(int)`,
  and `max(int, boolean)` unchanged;
- the public method inventory and count match the exact new surface; and
- no compatibility alias, deprecated bridge, or alternate scalar factory entry remains.

## Affected files

Expected production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/binary/BinaryArithmeticKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarElementwiseKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarValueAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScalarExpressions.java`

Expected tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarElementwiseSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScalarElementwiseTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorNumericReductionTest.java`

Expected documentation and planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/tasks/0018t-scalar-arithmetic-family-normalization.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless an out-of-scope discrepancy requires stopping:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ClampRangeAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorBinaryExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/ScalarValue.java`
- `docs/api/training-api.md`
- focused architecture documentation
- architecture tests

## Maximum scope

This task may create or modify at most the exact 18 affected paths listed above: five production
Java files, six test files, and seven documentation/planning files including this task.

The original 17-path maximum was explicitly expanded by the user only for
`TensorNumericReductionTest.java` after the first focused compilation exposed its two stale
pairwise `min(Tensor)`/`max(Tensor)` calls. No other production, test, or documentation path is
authorized. If implementation needs a nineteenth path, a new type, helper-surface change, or any
review-only file, stop and report why before editing it.

## Acceptance criteria

- The scalar enum has exactly the selected eight constants and exact signature variants.
- `CLAMP_MIN` and `CLAMP_MAX` are absent from live production Java and current API/glossary
  documentation; focused tests may name them only to prove enum lookup failure.
- All seven arithmetic relationships have both Tensor-to-Tensor and exact typed scalar public
  forms with matching semantic-kind names.
- All seven scalar relationships also have exact-FLOAT64 `double` conveniences.
- Public pairwise extrema are named only `minimum` and `maximum`; reductions remain only `min` and
  `max`.
- `clampMin` produces exactly one scalar MAX producer and `clampMax` exactly one scalar MIN
  producer; `clamp` remains exactly one CLAMP producer.
- Exact scalar references, data-type equality, validation order, descriptor metadata, one-input
  producer/provenance, output index zero, freshness, and no-ID-on-preconstruction-failure are
  covered by focused tests.
- Binary promotion, broadcasting, ordered two-input provenance, freshness, and error behavior are
  unchanged apart from public extrema method names.
- The selected NaN, infinity, and signed-zero extrema policy is documented consistently without
  claiming model-level evaluation or backend implementation.
- Existing reduction, comparison, unary, typed scalar, clamp-range, Tensor producer/provenance,
  storage, factory, random, indexing, shape, and graph tests remain green.
- Public and helper reflection tests prove the exact final surface and removed-name absence.
- Every affected Java declaration and public method has complete accurate Javadoc, including all
  parameters, returns, failures, metadata behavior, and numerical-semantic boundaries.
- Tensor API, Compile API, glossary, capability baseline, task, master plan, and roadmap describe
  the current final surface without rewriting completed task history.
- Training API, architecture docs/tests, `ARCHITECTURE.md`, Gradle, other modules, and dependency
  direction are unchanged, with reasoned no-change conclusions recorded.
- One final `:modules:model:test` run passes after executable Java stabilizes.
- A separate clean-context documentation-focused pass finalizes Javadocs and documentation in the
  same overall change without repeating successful Java tests unless executable behavior changes
  or it records a concrete reason.
- Model Javadoc, Markdown links/anchors/fences/final-newlines, removed-vocabulary scans, exact-scope
  review, status synchronization, and `git diff --check` pass.
- Task 0018T is Complete only after both passes and all evidence succeed. Task 0018T1 remains Draft
  without a detailed specification.

## Tests / validation

Implementation-focused tests while stabilizing:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScalarElementwiseTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorNumericReductionTest
```

Final Java checkpoint after executable code stabilizes:

```bash
./gradlew :modules:model:test
```

The implementation pass records the exact result and hands it to the documentation pass. Do not
repeat that successful suite unless executable Java changes afterward or a concrete risk is
recorded.

Documentation-focused pass after final Javadoc/documentation edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass must also:

- validate local Markdown links and anchors in changed documents;
- validate Java examples affected by the naming/API change against the compiled model classes;
- scan live production Java and current API/glossary docs for removed `CLAMP_MIN`, `CLAMP_MAX`,
  `min(Tensor)`, and `max(Tensor)` vocabulary while allowing historical completed task records;
- verify the exact 18-path cap and no Java/Gradle/module/architecture/dependency changes outside
  scope;
- verify task 0018T/master/roadmap status agreement and that 0018T1 has no detailed spec; and
- inspect the final diff for stale numerical claims, duplicate terminology, trailing whitespace,
  malformed fences, missing final newlines, and accidental authority changes.

Repository-wide validation is deferred to the capability-reset checkpoint after task 0018V and
CI. This single-module task does not change dependencies, architecture, or build configuration.

## Dependencies

- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md) — current seven binary arithmetic kinds.
- [Task 0014B](0014b-binary-arithmetic-tensor-expressions.md) — current promotion, broadcasting,
  and public binary construction.
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md) — provisional scalar kinds and
  attribute roles.
- [Task 0014F](0014f-scalar-arithmetic-and-clamp-tensor-expressions.md) — current scalar helper and
  public construction.
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md) — family-owned signature
  validation.
- [Task 0018N](0018n-typed-scalar-value-contract.md) — exact typed scalar parameters and
  receiver-aware matching.
- [Task 0018P](0018p-elementwise-semantic-cleanup.md) — final reciprocal vocabulary and removal of
  backend-route-like fast kinds.

All dependencies are Complete.

## Follow-up tasks

- **0018T1 Unary numeric gaps and floating diagnostics** — add `RSQRT`, `LOG1P`, `EXPM1`,
  `IS_FINITE`, `IS_NAN`, and `IS_INF` semantics and public construction in a separate cohesive
  task.
- **0018U Integral arithmetic and comparison domains** — deliberately broaden selected arithmetic,
  comparisons, arg-min, and reductions only after overflow and accumulation policies are fixed.
- **0018V Multi-axis and statistical reductions** — add ordered axes and statistical/norm
  semantics after the numeric foundations are stable.
- Later compiler and backend tasks own graph capture validation, gradients, execution, numerical
  conformance, lowering, and kernels.

## Architecture impact

Expected impact: None.

This is a backend-independent model vocabulary and public-surface normalization inside the
existing `modules/model` ownership boundary. It changes no module dependency, lifecycle,
architecture rule, or backend contract.

If implementation requires an architecture change, stop and report it before editing
`ARCHITECTURE.md` or focused architecture documentation.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, roadmap, model capabilities/master
plan, tasks 0014A/0014B/0014E/0014F/0018K/0018N/0018P/0018T, Tensor API, Compile API, Training
API, glossary, and every affected or review-only source/test named by task 0018T in full.

Implement task 0018T exactly. Atomically normalize the complete seven-operation floating
Tensor-to-Tensor/Tensor-to-scalar arithmetic family, rename pairwise public min/max to
minimum/maximum while preserving reduction names, remove CLAMP_MIN/CLAMP_MAX, retain first-class
CLAMP, and make one-bound clamp methods conveniences over scalar MAX/MIN. Preserve exact typed
scalar matching, validation order, descriptor and producer/provenance behavior, and every
architecture boundary. Stay within the exact 18 paths, stop on scope or architecture conflict,
and do not commit or push.

Run focused tests as needed and one final model test after executable Java stabilizes. Then hand
the actual diff and exact evidence to a separate clean-context documentation-focused agent in the
same overall change. That agent must inspect final source/tests, finalize permitted Javadocs,
Tensor/Compile API, glossary, capability/task/master/roadmap documentation, run model Javadoc and
documentation/scope checks, and must not repeat successful Java tests unless executable behavior
changes or it records a concrete reason.

Mark 0018T Complete only after both passes succeed. Leave 0018T1 and every later task Draft
without a detailed specification.
```

## Local decisions

- The scalar family mirrors all seven existing binary arithmetic kind names and retains one
  `ScalarValueAttrs` variant for those one-parameter requests.
- Pairwise public extrema use `minimum`/`maximum` so integral scalar literals cannot be confused
  with reduction axes; aggregate `min`/`max` remain unchanged.
- `CLAMP` remains one first-class range request. One-bound clamp methods delegate directly to
  scalar MAX/MIN, so each successful convenience call creates one producer and no intermediate.
- The explicitly authorized eighteenth path corrects only stale pairwise method calls in
  `TensorNumericReductionTest`; no further scope expansion is permitted.

## Known limitations

- This task constructs model metadata only. It does not calculate values, define gradients,
  capture a graph, lower operations, select kernels, or make a backend executable.
- Scalar arithmetic remains floating-only with exact receiver/value data-type equality. Retained
  `double` overloads mean exact FLOAT64 and do not infer or narrow a scalar type.
- No NaN payload, exact instruction, intermediate precision, bitwise reproducibility, or extrema
  equality/subgradient convention is promised.

## Validation evidence

- Implementation context `/root/task_0018t_implementation` ran the focused command containing
  `OperationSignatureTest`, `ScalarElementwiseSemanticsTest`, `TensorBinaryArithmeticTest`,
  `TensorScalarElementwiseTest`, `TensorTest`, and `TensorNumericReductionTest`; it passed after an
  initial test-compilation failure exposed the two stale pairwise calls in the latter test. The
  user explicitly expanded the task from 17 to 18 paths only for that correction.
- The same implementation context ran `./gradlew :modules:model:test` after executable Java
  stabilized. It passed with 715 tests across 88 suites and zero failures, errors, or skips.
  Executable Java did not change afterward; documentation context
  `/root/task_0018t_implementation/task_0018t_docs` reused this evidence and did not rerun Java
  tests.
- The clean documentation context applied the General, API/Javadoc, Planning, and Example
  profiles to the actual final diff, five production contracts, six affected tests, Tensor and
  Compile API references, glossary, capability baseline, task, master plan, and roadmap.
- Its first `./gradlew :modules:model:javadoc` run found two stale reduction-Javadoc links to the
  removed pairwise signatures. After correcting them to `minimum(Tensor)` and `maximum(Tensor)`,
  the final run passed with `BUILD SUCCESSFUL`; two actionable tasks executed.
- A Java 26 `ScalarArithmeticExample` compiled and ran against the final model classes. Its six
  checks confirmed binary MIN, scalar MAX, `clampMin` as MAX, `clampMax` as MIN, first-class CLAMP,
  and aggregate `min()` retaining scalar result Shape.
- The targeted Markdown checker resolved 492 local links, including 138 heading anchors, across
  the seven changed documentation/planning files with zero errors. Generated Javadoc contains the
  new pairwise methods and scalar MIN/MAX/CLAMP links and contains none of the removed vocabulary.
- Live production Java plus current Tensor API, Compile API, and glossary scans contain no
  `CLAMP_MIN`, `CLAMP_MAX`, `min(Tensor)`, or `max(Tensor)` contract. Historical completed task
  records retain their original vocabulary as implementation history.
- Final scope inspection found exactly 18 authorized paths: five production Java files, six tests,
  and seven documentation/planning files. No Gradle, dependency, architecture, other-module,
  backend-conformance, or integration path changed. Task 0018T is Complete in the task, master
  plan, and roadmap; 0018T1 and all later rows remain Draft, and no 0018T1 task specification
  exists.
- Formatting checks found balanced fences, final newlines, and no trailing whitespace in changed
  documentation. `git diff --check` passed on the final combined change.
- Training API remains accurate unchanged because this task adds no gradient object, autograd,
  optimizer, or training behavior. `ClampRangeAttrs`, `ScalarValue`, and
  `TensorBinaryExpressions` remain accurate unchanged because their validation, representation,
  and construction behavior did not change. Architecture documents/tests, backend conformance,
  integration tests, Gradle/dependencies, and other modules require no change because ownership,
  module boundaries, build structure, and executable behavior remain unchanged.

## Implementation notes

- Expanded `ScalarElementwiseKind` to exact `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, `POW`, and
  `CLAMP`, with exact family-owned signature variants and no one-bound clamp kinds.
- Added exact typed and exact-FLOAT64 scalar overloads for the five previously missing arithmetic
  relationships, preserved MUL/POW, and renamed only pairwise Tensor extrema to
  `minimum`/`maximum`.
- Retained first-class range CLAMP and made one-bound conveniences delegate to scalar MAX/MIN.
  Validation order, exact scalar references, descriptors, producer/provenance output zero,
  freshness, and preconstruction failure effects remain unchanged and covered by focused tests.
- Finalized all affected Javadocs plus Tensor/Compile APIs, glossary, capability baseline, and
  planning records for the numerical semantics and model-only lifecycle boundary.

## Completion summary

- Completed changes: normalized the full seven-operation binary/scalar arithmetic family,
  pairwise-extrema naming, first-class range clamp, and one-bound clamp conveniences.
- Files changed or created: exactly five production Java files, six tests, and seven
  documentation/planning files in the authorized 18-path scope.
- Tests and validation: reused the passing focused set and final 715-test/88-suite model run;
  final model Javadoc, compiled example, 492-link/138-anchor check, generated-page and removed-
  vocabulary scans, exact scope/status/no-later-spec checks, formatting, and `git diff --check`
  passed.
- Documentation-agent review: clean context
  `/root/task_0018t_implementation/task_0018t_docs` independently finalized the change using the
  General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Tensor and Compile APIs, glossary, capability baseline, task, master plan,
  and roadmap describe the final current surface without claiming compiler, gradient, backend, or
  execution support.
- Javadoc review: all five affected production files were reviewed and finalized; two stale
  reduction links found by generated Javadoc were corrected. Adjacent unchanged contracts remain
  accurate for the reasons recorded above.
- Glossary impact: existing arithmetic, operation-kind, provenance, Tensor, reduction, and status
  distinctions now use the final vocabulary; no new reusable domain term was required.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0018T. Task 0018T1 and later work remain Draft.

Status: Complete
