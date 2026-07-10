# Task 0018T1: Unary Numeric Gaps and Floating Diagnostics

## Status

Ready

## Goal

Complete the selected floating unary foundation before integral-domain and statistical-reduction
work begins.

Add three shape-preserving floating transforms:

```text
RSQRT   -> reciprocal square root
LOG1P   -> natural logarithm of one plus the input
EXPM1   -> natural exponential of the input minus one
```

Add three shape-preserving floating classifications with BOOL results:

```text
IS_FINITE -> finite value classification
IS_NAN    -> not-a-number classification
IS_INF    -> positive-or-negative infinity classification
```

The two groups deliberately use separate semantic families and construction helpers. The numeric
transforms retain the floating input type and its gradient-eligibility request. Classifications
return fixed BOOL descriptors with `requiresGrad=false`. Do not hide that result-contract
difference behind a result-kind flag, a generic registry, or a switch in one catch-all helper.

This task constructs semantic metadata only. It does not evaluate values, inspect storage, define
gradient formulas, capture a graph, select an implementation route, or execute kernels.

## Mental model

### Floating transform

```java
Tensor adjusted = input.log1p();
```

Conceptually records:

```text
TensorProducer
  operation = UnaryElementwiseKind.LOG1P + NoOperationAttrs.INSTANCE
  inputs = [input]
  outputDescriptors = [same floating type, exact input Shape, unresolved layout,
                       unchanged requiresGrad]

adjusted.provenance = (that producer, output index 0)
```

### Floating classification

```java
Tensor invalid = input.isNaN();
```

Conceptually records:

```text
TensorProducer
  operation = FloatingClassificationKind.IS_NAN + NoOperationAttrs.INSTANCE
  inputs = [input]
  outputDescriptors = [BOOL, exact input Shape, unresolved layout, requiresGrad=false]

invalid.provenance = (that producer, output index 0)
```

The classification is a real operation occurrence rather than an already known Java boolean. A
public Tensor expression normally describes values that a future prepared execution will produce;
model construction does not read those values from optional host storage.

## Why two families

`RSQRT`, `LOG1P`, and `EXPM1` have the same structural and result-metadata contract as the existing
floating unary operations. They therefore extend `UnaryElementwiseKind` and reuse
`TensorUnaryExpressions`.

`IS_FINITE`, `IS_NAN`, and `IS_INF` differ in two stable ways:

- their output data type is always BOOL rather than the input floating type; and
- their result is never gradient-eligible.

Those are model semantics, not backend details. A separate `FloatingClassificationKind` and
`TensorFloatingClassifications` boundary make the difference explicit without adding attributes,
an interface hierarchy, a broad operation registry, or result metadata to `OperationKind`.

The package name uses “classification” rather than “diagnostic” because these are graph-visible
value-producing operations. They are not trace events, logging, validation warnings, or debugging
services.

## Selected public API

Add exactly these zero-argument public methods to `Tensor`:

```java
public Tensor rsqrt()
public Tensor log1p()
public Tensor expm1()
public Tensor isFinite()
public Tensor isNaN()
public Tensor isInf()
```

All six accept only `BFLOAT16`, `FLOAT32`, or `FLOAT64` input. `INT32`, `INT64`, and BOOL are
rejected rather than creating trivially constant classifications. Task 0018U owns integral-domain
selection and may reconsider a concrete integral classification use case later; this task adds no
implicit cast or constant-folding rule.

No aliases such as `inverseSqrt`, `logOnePlus`, `expMinusOne`, `isInfinite`, or `isNotFinite` are
added. No overload accepts a tolerance, output type, algorithm, accuracy mode, or backend hint.

## Final semantic vocabularies

### `UnaryElementwiseKind`

After implementation, the existing enum contains exactly these sixteen constants in declaration
order:

```java
ABS,
NEG,
RECIPROCAL,
LOG,
LOG1P,
EXP,
EXPM1,
ERF,
SQRT,
RSQRT,
FLOOR,
CEIL,
SIGN,
RELU,
SIGMOID,
TANH
```

All sixteen variants accept exactly `NoOperationAttrs`, one logical Tensor input, and one output.
The three additions are placed beside their mathematical relatives; enum ordinal remains an
implementation detail and is not serialization or dispatch state.

### `FloatingClassificationKind`

Create one public enum containing exactly:

```java
IS_FINITE,
IS_NAN,
IS_INF
```

Every variant accepts exactly `NoOperationAttrs`, one logical Tensor input, and one output. The
enum adds no fields, per-constant class bodies, result-type flags, gradient flags, aliases,
symbols, categories, nested types, or other project methods beyond the family-owned immutable
signature implementation required by `OperationKind`.

## Numerical semantics selected here

The model records the portable mathematical target and exact special-value classification. It
does not select an algorithm, promise a particular machine instruction, or require bitwise-equal
results across backends. A backend claiming these capabilities must later pass the applicable
conformance tolerance while preserving the exact special-value behavior below.

### `RSQRT`

`rsqrt(x)` means `1 / sqrt(x)` as one first-class semantic request, not two stored operations.

- positive finite input produces its positive reciprocal square root;
- positive zero produces positive infinity;
- negative zero produces negative infinity, following reciprocal of the sign-preserving principal
  square root;
- positive infinity produces positive zero;
- a negative finite value or negative infinity produces NaN; and
- NaN produces NaN.

No NaN sign or payload, exact rounding, ULP bound, intermediate precision, or bitwise result is
promised by this model task.

### `LOG1P`

`log1p(x)` means the natural logarithm of `1 + x` as one first-class request. It exists separately
from `add(1).log()` so a future compiler/backend can preserve useful accuracy near zero without an
eager scalar Tensor or forced decomposition.

- input greater than `-1` uses the ordinary real-valued function;
- input exactly `-1` produces negative infinity;
- input below `-1`, including negative infinity, produces NaN;
- positive infinity produces positive infinity;
- positive and negative zero are preserved; and
- NaN produces NaN.

No NaN payload, exact rounding, ULP bound, or implementation algorithm is promised here.

### `EXPM1`

`expm1(x)` means `exp(x) - 1` as one first-class request. It exists separately from
`exp().sub(1)` so a future compiler/backend can preserve useful accuracy near zero.

- finite input uses the ordinary real-valued function rounded to the result format by the eventual
  execution implementation;
- positive infinity produces positive infinity;
- negative infinity produces exactly negative one;
- positive and negative zero are preserved; and
- NaN produces NaN.

Finite overflow produces positive infinity in the result format; sufficiently negative finite
input may round to negative one. No exact overflow threshold, NaN payload, bitwise result, ULP
bound, or algorithm is promised by this model task.

### Floating classifications

Classification uses the logical value represented by the floating input:

- `IS_FINITE` is true for every finite normal, subnormal, positive-zero, or negative-zero value and
  false for both infinities and every NaN;
- `IS_NAN` is true only for NaN, independent of sign, quiet/signaling encoding, or payload, and
  false for finite values and infinities; and
- `IS_INF` is true only for positive or negative infinity and false for finite values and NaN.

For every represented floating value, exactly one of finite, NaN, or infinity classification is
true. The output carrier representation remains the existing BOOL contract; this task does not
read or populate it.

## Gradient and accuracy boundary

`requiresGrad` remains model eligibility metadata, not a derivative guarantee:

- `rsqrt`, `log1p`, and `expm1` preserve the exact input flag, including at values where an
  eventual result or derivative is non-finite;
- `isFinite`, `isNaN`, and `isInf` always set it to false; and
- compiler-owned autograd later decides formulas and discontinuity/domain behavior for the numeric
  transforms and treats classification results as non-differentiable.

The operation names denote the portable mathematical functions. This model task deliberately does
not promise correct rounding or a fixed ULP bound. Backend conformance must choose and publish
per-data-type tolerances before execution support is claimed; it may not replace these kinds with
the already rejected public “fast” variants or change their special-value meaning.

## Scope

- Extend `UnaryElementwiseKind` with exactly `LOG1P`, `EXPM1`, and `RSQRT` in the selected order.
- Preserve every existing unary kind, signature, typed identity, and portable meaning.
- Add public `FloatingClassificationKind` with exactly three parameterless variants.
- Give both families exact `NoOperationAttrs`, one-input, one-output signatures.
- Add `Tensor.rsqrt`, `Tensor.log1p`, and `Tensor.expm1` through the unchanged
  `TensorUnaryExpressions.apply` construction path.
- Add field-free package-private `TensorFloatingClassifications` with one shared `apply` method for
  `isFinite`, `isNaN`, and `isInf`.
- Add the six exact zero-argument methods to `Tensor` without aliases or overloads.
- Accept exactly the three current floating data types for both groups.
- Preserve exact input data type, Shape reference, unresolved layout, and `requiresGrad` for the
  numeric transforms.
- Produce exact BOOL, the input's exact Shape reference, unresolved layout, and
  `requiresGrad=false` for classifications.
- Preserve one exact input reference, one identity-distinct producer, output index zero, no label,
  no storage, and fresh identity for every valid call.
- Preserve validation-before-ID-allocation and exact null/type failure order.
- Add exact enum, signature, helper/public-surface, metadata, provenance, freshness, classification,
  invalid-type, domain-noninspection, and no-alias tests.
- Update the model package map with the cohesive classification package.
- Update Tensor API, Compile API, glossary, capability baseline, task, model master plan, and
  roadmap through the mandatory independent documentation-focused pass.

## Out of scope

- numerical evaluation, host-storage reads or writes, output allocation, eager classification, or
  constant folding
- `square`, `cbrt`, trigonometric functions, `log2`, `log10`, `exp2`, `isNormal`, `isSubnormal`,
  `isZero`, `isPositive`, `isNegative`, `isNotFinite`, or separate positive/negative-infinity
  operations
- integral or BOOL classification, implicit cast insertion, data-type promotion, caller-selected
  result type, or another public overload
- changes to existing `reciprocal`, `log`, `exp`, `sqrt`, or any other completed unary behavior
- decomposition into existing operations, eager scalar constants, canonicalization, algebraic
  simplification, common-subexpression elimination, fusion, or result interning
- attributes, factories, registries, service locators, reflective discovery, generic result-kind
  metadata, category interfaces, or operation-class hierarchies
- exact rounding, fixed ULP/relative-error promises, backend-specific accuracy flags, algorithm
  choice, approximation variants, or restored `FAST_EXP`/`FAST_TANH`
- gradient formulas, subgradient conventions, backward graph construction, optimizer, or training
  behavior
- compiler capture, graph-wide descriptor validation, planning, prepare, backend support, lowering,
  kernel selection, runtime, execution, tracing, or engine behavior
- task 0018U integral arithmetic/comparison/reduction work, task 0018V statistical reductions, or
  any later operation family
- changes to `Operation`, `OperationKind`, `OperationSignature`, `NoOperationAttrs`, DataType,
  Shape, layout, descriptor, Tensor producer/provenance, storage, factory, randoms, or graph model
- dependencies, Gradle, Java version, preview/incubator features, architecture rules, focused
  architecture documents/tests, another module, or a detailed 0018U-or-later task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model ownership of public
  Tensor and backend-independent operation semantics
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0014C](0014c-unary-elementwise-semantic-kinds.md)
- [Task 0014D](0014d-unary-elementwise-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018P](0018p-elementwise-semantic-cleanup.md)
- [Task 0018T](0018t-scalar-arithmetic-family-normalization.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns both semantic families, public Tensor methods, local descriptor derivation,
  and producer/provenance construction.
- `OperationKind` remains a compact family-owned structural contract. It does not receive a result
  data type, differentiability flag, operation category, backend support, or executable behavior.
- Numeric transforms and floating classifications remain distinct typed kind families even though
  both are parameterless and consume one floating input.
- `Operation` continues to validate only exact attributes type and occurrence cardinality. It has
  no operand descriptor and therefore does not validate floating eligibility or result metadata.
- Public Tensor construction owns local input-type validation and result-descriptor derivation.
- `Tensor` remains mutable public API state rather than graph IR. Each result has one immutable
  identity-bearing producer occurrence and provenance output index zero.
- Compiler owns graph capture, descriptor revalidation, gradient rules, and backward graph
  construction. Backend prepare owns lowering, numerical implementation, specialization, fusion,
  and kernel selection.
- Runtime hot paths do not consume these `Operation` values directly.
- Classification terminology must not imply or import `modules/trace`; no trace event or diagnostic
  service is added.
- Package direction remains `model.tensor -> model.operation.elementwise.* -> model.operation`.
  No dependency, registry, service locator, reflection, or classpath discovery is added.
- If implementation needs attributes, a common result-kind abstraction, a change to the existing
  unary helper surface, integral classification, another production type, another module, or a
  nineteenth path, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.elementwise.unary` — owns parameterless unary numeric
  transforms and activations whose public construction retains the floating input type.
- `io.github.pho001.synaptik.model.tensor` — owns public methods and the two cohesive package-private
  construction boundaries.
- `io.github.pho001.synaptik.model.operation` — owns unchanged operation/signature foundations.

Package added:

```text
io.github.pho001.synaptik.model.operation.elementwise.classification
  Parameterless floating value-classification semantics with fixed BOOL results.
```

Type placement:

- `UnaryElementwiseKind` remains in the unary package because the three new numerical functions
  share its exact input/result construction contract.
- `FloatingClassificationKind` belongs in the new classification package because it names
  graph-visible floating value classifications, not tracing diagnostics.
- `TensorFloatingClassifications` remains package-private beside `Tensor` because it owns local
  input validation, descriptor derivation, and derived construction for the public facade.
- Focused tests mirror production packages; Tensor construction tests remain in `model.tensor` for
  package-private helper access.

## Required implementation contracts

### Unary-family extension

- `UnaryElementwiseKind.values()` equals the exact sixteen-value sequence in this specification.
- Every value returns the existing immutable `NoOperationAttrs`/one-input/one-output signature.
- `TensorUnaryExpressions` remains final, package-private, field-free, non-instantiable, and has
  exactly its existing one package-private `apply(Tensor, UnaryElementwiseKind)` method.
- Its null/type validation, descriptor construction, operation construction, factory delegation,
  exception messages, and side-effect order remain unchanged.
- The three new Tensor methods delegate exactly once with the exact new kind.

### Classification-family contract

Create `FloatingClassificationKind` as one public enum with the exact three constants and one
private static final immutable signature list. Its only project method is:

```java
@Override
public List<OperationSignature> signatures()
```

The returned list contains exactly
`OperationSignature.fixed(NoOperationAttrs.class, 1, 1)`. Standard compiler-generated enum state
does not count as project state.

Create `TensorFloatingClassifications` as one package-private final non-record class with:

- no fields, nested types, interfaces, or instance state;
- exactly one private zero-argument constructor; and
- exactly one package-private static method:

```java
static Tensor apply(Tensor input, FloatingClassificationKind kind)
```

`apply` performs this exact order:

1. `Objects.requireNonNull(input, "input")`;
2. `Objects.requireNonNull(kind, "kind")`;
3. read `input.descriptor().dataType()`;
4. require `dataType.isFloating()`, otherwise throw `IllegalArgumentException` with exact message
   `input must be a floating data type, but was <type>`;
5. construct one `TensorDescriptor` with `DataType.BOOL`, the exact input Shape reference,
   `Optional.empty()`, and `false`;
6. construct one `Operation(kind, NoOperationAttrs.INSTANCE)`;
7. call `TensorFactory.createDerived` exactly once with the descriptor, no label, operation, and
   ordered `List.of(input)`.

A pre-factory failure allocates no Tensor identity. A valid call returns the factory's exact fresh,
unlabeled, storage-free result and does not mutate or retain input storage as output storage.

### Public facade and exact surface

- `rsqrt`, `log1p`, and `expm1` each delegate exactly once to `TensorUnaryExpressions.apply`.
- `isFinite`, `isNaN`, and `isInf` each delegate exactly once to
  `TensorFloatingClassifications.apply`.
- All six methods are public, non-static, non-synchronized, zero-argument, and return `Tensor`.
- The final declared public Tensor method count is exactly 127.
- Reflection finds all six new names and no alternate spelling listed in Out of scope.
- The existing 121-method surface, including `reciprocal`, ordinary `log`/`exp`/`sqrt`, pairwise
  `minimum`/`maximum`, reduction `min`/`max`, scalar operations, and every non-numeric method,
  remains unchanged.

## Affected files

Expected production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/classification/FloatingClassificationKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorUnaryExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFloatingClassifications.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKindTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/classification/FloatingClassificationKindTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorUnaryElementwiseTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFloatingClassificationTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`

Expected documentation and planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/tasks/0018t1-unary-numeric-gaps-and-floating-diagnostics.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless an out-of-scope discrepancy requires stopping:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/Operation.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/OperationSignature.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/NoOperationAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataType.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProvenance.java`
- `docs/api/training-api.md`
- focused architecture documentation and architecture tests

## Maximum scope

This task may create or modify exactly the 18 affected paths listed above: five production Java
files, six test files, and seven documentation/planning files including this specification.

If implementation needs a nineteenth path, an attributes type, another helper method, integral
classification, a shared result-kind abstraction, or a review-only file, stop and report the
reason before editing it. Do not trade away a required focused test or documentation path merely
to stay under the cap.

## Acceptance criteria

- The unary enum contains exactly sixteen selected kinds with three additions in the specified
  positions and unchanged family signature.
- The classification enum contains exactly three selected kinds and the exact parameterless
  one-input/one-output signature.
- All six public Tensor methods exist with the exact names and no aliases or overloads.
- Numeric transforms accept every current floating type, retain the exact input type/Shape/
  `requiresGrad`, and leave layout unresolved.
- Classifications accept every current floating type, return BOOL with the exact input Shape,
  false gradient eligibility, and unresolved layout.
- INT32, INT64, and BOOL fail both construction paths before result identity allocation with the
  exact existing floating-input message.
- Every successful call is fresh, unlabeled, storage-free, one-output, and records exact operation,
  canonical no-attributes, exact sole input, and provenance output index zero.
- Scalar, zero-sized, static, dynamic, resolved-layout, storage-backed, and already-derived inputs
  preserve the specified metadata and ownership boundaries.
- Tests prove no value or domain inspection by constructing metadata from representative negative,
  zero, infinity, and NaN host-backed leaves without evaluating them.
- Javadocs and current API documentation explain the selected special-value semantics, result-type
  distinction, accuracy boundary, and newcomer-readable examples without claiming execution.
- Operation signature coverage includes both families and rejects incompatible attributes through
  the existing foundation.
- The exact public Tensor method count is 127 and helper/enum reflection shapes match the plan.
- Tensor API, Compile API, glossary, capability baseline, task, master plan, and roadmap are
  synchronized without rewriting completed task history.
- Training API, architecture contract/docs/tests, dependencies, Gradle, other modules, backend
  conformance, and integration tests remain unchanged with reasoned no-change conclusions.
- One final `:modules:model:test` run passes after executable Java stabilizes.
- A separate clean-context documentation-focused pass finalizes all affected Javadocs and
  explanatory/planning documentation in the same overall change and reuses successful Java-test
  evidence unless executable Java changes afterward or a concrete risk is recorded.
- Model Javadoc, a runnable Java 26 metadata example, Markdown links/anchors/fences/final-newlines,
  exact 18-path scope, status synchronization, terminology, and `git diff --check` pass.
- Task 0018T1 is Complete only after both passes succeed. Task 0018U and every later task remain
  Draft without a detailed specification.

## Tests / validation

Focused tests while executable Java stabilizes:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKindTest \
  --tests io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKindTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorUnaryElementwiseTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorFloatingClassificationTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

Final Java checkpoint after executable code stabilizes:

```bash
./gradlew :modules:model:test
```

The implementation pass records exact results and hands them to the documentation pass. Do not
repeat the successful final model suite unless executable Java changes afterward or a concrete
risk is recorded.

Documentation-focused pass:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass must also:

- compile and run one Java 26 example that observes one numeric transform and one classification
  result descriptor, exact operation kinds, one-input producers, and output index zero without
  claiming calculated values;
- validate local Markdown links and anchors in changed documents;
- inspect generated Javadoc for all affected/new public contracts and Tensor methods;
- verify exact enum values, six new public methods, final public count 127, helper shape, and
  absence of aliases or restored fast variants;
- verify the exact 18-path inventory and no dependency/Gradle/architecture/other-module changes;
- verify synchronized 0018T1 Complete status and no detailed 0018U-or-later specification; and
- review terminology, examples, fences, final newlines, trailing whitespace, authority boundaries,
  and the final diff.

Repository-wide validation is deferred to the capability-reset checkpoint after task 0018V and
CI. This task changes one module without changing dependencies, architecture, or shared build
configuration.

## Dependencies

- [Task 0014C](0014c-unary-elementwise-semantic-kinds.md) — current parameterless unary family.
- [Task 0014D](0014d-unary-elementwise-tensor-expressions.md) — current floating-preserving unary
  construction path.
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md) — family-owned exact
  signature validation.
- [Task 0018P](0018p-elementwise-semantic-cleanup.md) — final portable unary vocabulary and no-fast-
  variant boundary.
- [Task 0018T](0018t-scalar-arithmetic-family-normalization.md) — completed floating arithmetic
  family and numerical-policy conventions.

All dependencies are Complete.

## Follow-up tasks

- **0018U Integral arithmetic and comparison domains** — add only selected signed-integral
  arithmetic, comparisons, arg-min, and reductions after overflow and accumulation policies are
  explicit.
- **0018V Multi-axis and statistical reductions** — add ordered axes, log-sum-exp, variance,
  standard deviation, and norms after floating/integral foundations are stable.
- Later compiler and backend tasks own gradients, numerical conformance, lowering, execution, and
  kernels.

## Architecture impact

Expected impact: None.

This adds backend-independent model semantics, public metadata construction, and one cohesive
operation subpackage within existing module ownership. It changes no module dependency, lifecycle,
architecture rule, backend contract, runtime path, or graph representation.

If implementation requires an architecture change, stop and report it before editing
`ARCHITECTURE.md`, focused architecture documentation, ADRs, or architecture tests.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, roadmap, model capabilities/master
plan, tasks 0014C/0014D/0018K/0018P/0018T/0018T1, Tensor API, Compile API, Training API,
glossary, and every affected or review-only source/test named by task 0018T1 in full.

Implement task 0018T1 exactly. Extend the existing floating-preserving unary family with RSQRT,
LOG1P, and EXPM1, and add the separate parameterless FloatingClassificationKind family plus
public isFinite/isNaN/isInf construction with fixed non-differentiable BOOL results. Preserve
validation order, descriptor and producer/provenance behavior, existing unary contracts, and every
architecture boundary. Stay within the exact 18 paths, stop on scope or architecture conflict,
and do not commit or push.

Run focused tests as needed and one final model test after executable Java stabilizes. Then hand
the actual diff and exact evidence to a separate clean-context documentation-focused agent in the
same overall change. That agent must inspect final source/tests, finalize permitted Javadocs,
Tensor/Compile API, glossary, capability/task/master/roadmap documentation, run model Javadoc and
documentation/scope checks, and must not repeat successful Java tests unless executable behavior
changes or it records a concrete reason.

Mark 0018T1 Complete only after both passes succeed. Leave 0018U and every later task Draft
without a detailed specification.
```

## Local decisions

Empty until implemented.

## Known limitations

Empty until implemented.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
