# Task 0018U: Integral Elementwise Arithmetic and Comparisons

## Status

Complete

## Goal

Make current INT32 and INT64 Tensors useful as backend-independent elementwise numeric expressions
without broadening into division, reductions, or execution.

Extend the existing public operations so the selected integral domain supports:

```text
Tensor-to-Tensor arithmetic: ADD, SUB, MUL, MIN, MAX
Tensor-to-scalar arithmetic: ADD, SUB, MUL, MIN, MAX
Tensor-to-Tensor comparison: GREATER_THAN, GREATER_OR_EQUAL,
                             LESS_THAN, LESS_OR_EQUAL, EQUAL, NOT_EQUAL
```

Preserve every existing floating behavior. Reject BOOL and mixed floating/integral pairs. Keep
integral `DIV`, `POW`, range `CLAMP`, reductions, and `argMin` outside this task.

This task constructs typed result descriptors and producer/provenance metadata only. It does not
read values, execute arithmetic, detect overflow at construction, generate gradients, capture a
graph, lower operations, or add kernels.

## Why this is a separate task

Today an integral Tensor can carry model inputs, labels, counters, or indices, but the ordinary
arithmetic and comparison helpers reject it before creating an expression:

```java
Tensor shifted = tokenIds.add(offset);       // currently rejected
Tensor equal = predictions.equalTo(labels);  // currently rejected
```

After this task the same public vocabulary works for selected floating and signed-integral domains:

```java
Tensor shifted = tokenIds.add(ScalarValue.int32(1));
Tensor equal = predictions.equalTo(labels);
```

The later task 0018U1 owns integral reductions and `argMin`. Separating it avoids combining local
elementwise promotion and broadcasting with accumulation, empty-domain, tie-policy, and public
arg-reduction type normalization in one oversized implementation session.

## Mental model

### Mixed-width integral Tensor operands

```java
Tensor result = int32Left.add(int64Right);
```

Conceptually records:

```text
promotion = INT64
shape = broadcast(int32Left.shape, int64Right.shape)

TensorProducer
  operation = BinaryArithmeticKind.ADD + NoOperationAttrs.INSTANCE
  inputs = [int32Left, int64Right]
  outputDescriptors = [INT64, broadcast Shape, unresolved layout, requiresGrad=false]
```

The model does not insert a visible CAST producer. Promotion is a semantic operand/result-domain
rule for this operation occurrence, just like the existing floating promotion hierarchy.

### Exact typed scalar

```java
Tensor result = int32Input.mul(ScalarValue.int32(4));
```

Conceptually records one input and one exact attribute:

```text
operation = ScalarElementwiseKind.MUL
attrs = ScalarValueAttrs(ScalarValue.int32(4))
inputs = [int32Input]
result type = INT32
```

Scalar parameters do not promote. `ScalarValue.int64(4)` is rejected for an INT32 receiver because
the exact typed attribute is not another Tensor operand and no implicit scalar conversion contract
exists.

### Integral comparison

```java
Tensor mask = int32Left.lessThan(int64Right);
```

The comparison domain promotes to INT64, but the result is:

```text
data type = BOOL
shape = broadcast(left.shape, right.shape)
requiresGrad = false
operation = BinaryComparisonKind.LESS_THAN
inputs = [left, right]
```

## Selected promotion contract

Extend the existing stateless public `DataTypePromotion` utility with exactly one new method:

```java
public static DataType promoteNumeric(DataType left, DataType right)
```

The method accepts only two operands from the same numeric category.

### Floating pairs

Preserve the existing hierarchy exactly:

```text
BFLOAT16 < FLOAT32 < FLOAT64
```

Every floating pair returns the same result as `promoteFloating(left, right)`.

### Integral pairs

Use this exact symmetric hierarchy:

```text
INT32 < INT64
```

The complete matrix is:

| Left | Right | Result |
|---|---|---|
| INT32 | INT32 | INT32 |
| INT32 | INT64 | INT64 |
| INT64 | INT32 | INT64 |
| INT64 | INT64 | INT64 |

INT32 values conceptually sign-extend into the INT64 operation domain. No explicit CAST producer is
stored by public expression construction.

### Rejected pairs

- Floating plus integral is rejected in either order. Callers must make conversion intent visible
  through `Tensor.cast(...)`.
- BOOL with any operand is rejected. BOOL remains logical rather than numeric.
- No integral value promotes to floating merely because floating has a wider bit width.

`promoteFloating` remains public with its exact existing behavior, messages, and tests. Do not
replace or weaken it.

## Integral arithmetic domain

### Supported kinds

Integral Tensor operands support exactly:

```text
ADD
SUB
MUL
MIN
MAX
```

The same five kinds are supported for exact typed scalar attributes.

### Deliberately unsupported kinds

Integral operands reject:

```text
DIV
POW
```

`DIV` requires explicit decisions for truncation versus floor rounding, division by zero, and
`MIN_VALUE / -1`. `POW` requires negative-exponent, overflow, and result-domain rules. Neither is
needed for the selected minimal integral baseline, so this task does not invent those policies.

First-class two-bound `CLAMP` remains floating-only. The existing one-bound `clampMin` and
`clampMax` conveniences delegate to scalar MAX and MIN, so they inherit exact typed integral
eligibility without adding an integral CLAMP operation:

```java
int32Tensor.clampMin(ScalarValue.int32(0)); // scalar MAX, valid
int32Tensor.clampMax(ScalarValue.int32(9)); // scalar MIN, valid
int32Tensor.clamp(int32Zero, int32Nine);    // first-class CLAMP, still rejected
```

## Overflow and ordering policy

The model selects ordinary fixed-width two's-complement modular arithmetic for integral ADD, SUB,
and MUL:

```text
INT32_MAX + 1 -> INT32_MIN
INT32_MIN - 1 -> INT32_MAX
INT64_MAX * 2 -> wrap modulo 2^64 and reinterpret as signed INT64
```

This means:

- no construction-time or execution-time overflow exception is part of the operation semantics;
- no saturation or widening beyond the promoted result type occurs;
- INT32 arithmetic wraps modulo `2^32` and INT64 modulo `2^64`; and
- later backends must reproduce the corresponding low-width two's-complement result.

Integral MIN/MAX and all six comparisons use ordinary signed numerical ordering after promotion.
There is no NaN, infinity, signed-zero, tolerance, or approximate-equality policy in the integral
domain. EQUAL and NOT_EQUAL compare exact promoted signed values.

This task records those semantics in model Javadocs and API documentation but does not calculate a
result.

## Result metadata

### Tensor-to-Tensor arithmetic

- Result data type is `promoteNumeric(leftType, rightType)`.
- Floating pairs retain the current floating behavior.
- Integral pairs use the matrix above.
- Result Shape is the existing right-aligned `ShapeBroadcast` result.
- Layout remains unresolved.
- `requiresGrad` remains the logical OR for floating pairs and is necessarily false for integral
  pairs because integral descriptors cannot request gradients.
- Ordered provenance remains `[left, right]`.

### Tensor-to-scalar arithmetic

- Receiver must be floating or integral.
- The exact `ScalarValue.dataType()` must equal the receiver data type.
- Floating receivers retain all seven existing scalar arithmetic kinds.
- Integral receivers accept only ADD, SUB, MUL, MIN, and MAX.
- Result retains the exact receiver DataType and Shape reference, unresolved layout, and exact
  existing `requiresGrad` value, which is false for integral input.
- Provenance contains exactly the receiver; the scalar remains exact operation attributes.

### Comparisons

- Both operands must form a valid same-category numeric promotion pair.
- Result data type is always BOOL.
- Result Shape is the existing right-aligned broadcast result.
- Layout is unresolved and `requiresGrad=false`.
- Ordered provenance remains `[left, right]`.

## Scope

- Add exact `DataTypePromotion.promoteNumeric` with floating, integral, BOOL, mixed-category, null,
  symmetry, and idempotence contracts.
- Preserve `promoteFloating` unchanged.
- Make `TensorBinaryExpressions` use numeric promotion and accept the five selected integral kinds.
- Keep floating support for all seven binary arithmetic kinds unchanged.
- Reject integral DIV and POW before Shape broadcast and result identity allocation.
- Make `TensorComparisonExpressions` validate through numeric promotion and accept all integral
  pairs for all six existing comparison kinds.
- Preserve comparison BOOL result metadata and all floating behavior.
- Make `TensorScalarExpressions.applyScalar` accept floating or integral input, with exact
  scalar/input type equality and kind-aware integral restrictions.
- Preserve first-class `applyClamp` as floating-only.
- Let public scalar `clampMin`/`clampMax` inherit integral MAX/MIN convenience behavior.
- Update every affected `Tensor` Javadoc and its type-level summary without changing public method
  signatures or method count.
- Add focused promotion, binary arithmetic, scalar arithmetic, comparison, validation-order,
  overflow-semantic metadata, broadcasting, provenance, and no-ID-on-failure tests.
- Update Tensor API, Compile API, glossary, capability baseline, this task, master plan, and roadmap
  through the mandatory independent documentation-focused pass.

## Out of scope

- integral DIV, POW, remainder, modulo, floor division, truncating division, shifts, bitwise
  operations, absolute value, negation, range CLAMP, or another arithmetic kind
- integral `double` convenience conversion; existing double overloads remain exact FLOAT64 and
  therefore mismatch integral receivers
- generic `Number`, boxed values, caller-selected conversion, scalar promotion, eager scalar Tensor
  creation, or visible cast insertion
- floating/integral mixed-category promotion, automatic integral-to-floating conversion, BOOL
  arithmetic/comparison, unsigned types, smaller integer widths, big integers, quantization, or
  saturation
- numeric or statistical reductions, `mean`, accumulation types, empty-domain behavior, `argMin`,
  or changes to `argMax`; task 0018U1 owns selected integral reductions and arg-min normalization
- changing operation-kind enums, signatures, attributes, public method names, or Tensor method
  count
- numerical evaluation, storage reads/writes, overflow checks, constant folding, simplification,
  canonicalization, common-subexpression elimination, or result interning
- gradient formulas, autograd, backward graph construction, optimizer, or training behavior
- compiler capture, graph-wide validation, planning, prepare, backend support, lowering, kernel
  selection, runtime, execution, tracing, or engine behavior
- changes to DataType constants/categories, ScalarValue representation, Shape/Broadcast,
  TensorDescriptor, producer/provenance, factory, storage, or graph model
- dependencies, Gradle, Java version, preview/incubator features, architecture rules, focused
  architecture documents/tests, another module, or a detailed 0018U1-or-later task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model ownership of public
  Tensor, DataType, and backend-independent operation semantics
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md)
- [Task 0014B](0014b-binary-arithmetic-tensor-expressions.md)
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md)
- [Task 0014F](0014f-scalar-arithmetic-and-clamp-tensor-expressions.md)
- [Task 0015A](0015a-binary-comparison-semantic-kinds.md)
- [Task 0015B](0015b-binary-comparison-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0018T](0018t-scalar-arithmetic-family-normalization.md)
- [Task 0018T1](0018t1-unary-numeric-gaps-and-floating-diagnostics.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns data-type promotion, public expression validation, backend-independent
  arithmetic/comparison semantics, descriptor derivation, and producer/provenance construction.
- Promotion is a semantic data-type rule, not an inserted Tensor, graph node, compiler pass, or
  backend route.
- `Operation`, kind enums, and signatures remain operand-independent. They receive no domain flag,
  overflow mode, promotion table, backend capability, or result descriptor.
- Public/package-private Tensor construction owns operand-aware domain validation because it has
  descriptors and ordered Tensor inputs.
- Scalar attributes remain exact typed model values rather than Tensor inputs. No implicit scalar
  conversion is added.
- `Tensor` remains mutable public API state rather than IR. Every valid call creates one fresh
  storage-free result with one immutable producer and provenance output index zero.
- Compiler owns graph capture, operand revalidation, canonicalization, gradients, and backward graph
  construction. Backend prepare owns lowering, route selection, specialization, and fusion.
- Runtime hot paths do not consume Operation values.
- No registry, service locator, reflection, classpath discovery, module/package dependency, or
  architecture rule is added.
- If implementation needs a new operation kind, public method, promotion object, cast producer,
  attributes type, reduction change, another module, or a seventeenth path, stop and report the
  conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype` — owns public numeric promotion rules.
- `io.github.pho001.synaptik.model.tensor` — owns public methods and three existing cohesive
  package-private expression-construction boundaries.
- Existing operation-family packages remain review-only because their semantic identities and
  structural signatures do not change.

Packages added, moved, or removed:

- None.

Type placement:

- `DataTypePromotion` remains the single stateless promotion contract rather than creating a second
  integral-specific utility or a promotion-result object.
- `TensorBinaryExpressions`, `TensorScalarExpressions`, and `TensorComparisonExpressions` retain
  their existing packages and method surfaces because only their accepted data-type domains change.
- `Tensor` remains the public fluent facade; no new method or overload is required.

## Required implementation contracts

### `DataTypePromotion.promoteNumeric`

The utility remains public, final, field-free, non-instantiable, and gains exactly one public static
method. No other public or package-private method is added.

Validation occurs in this exact order:

1. null-check `left` with message `left`;
2. null-check `right` with message `right`;
3. require left to be floating or integral; otherwise throw `IllegalArgumentException` with exact
   message `left must be a numeric data type, but was BOOL`;
4. require right to be floating or integral; otherwise throw the analogous `right` message;
5. require both operands to share the same category; otherwise throw `IllegalArgumentException`
   with exact message
   `numeric data types must share a category, but were <left> and <right>`;
6. return the existing widest floating result or selected widest integral result.

The method is symmetric and idempotent for every valid pair. `promoteFloating` retains its current
implementation contract, validation messages, and accepted matrix.

### Binary arithmetic helper

Keep `TensorBinaryExpressions` field-free with its exact existing constructor and one `apply`
method. Its order becomes:

1. null-check left, right, then kind exactly as today;
2. call `DataTypePromotion.promoteNumeric` exactly once;
3. if the promoted type is integral, accept ADD, SUB, MUL, MIN, or MAX and reject DIV or POW with
   exact message `<KIND> does not support integral data types`;
4. broadcast Shapes exactly once;
5. construct the current result descriptor, exact operation, and ordered provenance;
6. delegate exactly once to `TensorFactory.createDerived`.

An unsupported integral kind fails before Shape broadcast and result-ID allocation. Floating
validation, promotion, broadcasting, result metadata, and errors remain unchanged except that BOOL
and cross-category failures now come from the new numeric contract.

### Scalar arithmetic helper

Keep `TensorScalarExpressions` at exactly its existing three-method surface. `applyClamp` and
private `create` retain their current behavior.

`applyScalar` order becomes:

1. null-check input, kind, then value exactly as today;
2. reject CLAMP with exact existing message `CLAMP requires ClampRangeAttrs`;
3. read the input data type and require floating or integral, otherwise throw exact message
   `input must be a numeric data type, but was BOOL`;
4. for integral input accept ADD, SUB, MUL, MIN, or MAX; reject DIV or POW with exact message
   `<KIND> does not support integral data types`;
5. require exact scalar/input type equality with the existing message
   `scalar data type <scalar> must match input data type <input>`;
6. construct exact attributes, operation, descriptor, and one-input producer as today.

Integral `clampMin` and `clampMax` work only because the existing public conveniences delegate to
MAX and MIN. `applyClamp` continues to reject integral input with its existing floating-input
message before range construction.

### Comparison helper

Keep `TensorComparisonExpressions` field-free with its exact existing one-method surface.

After null checks it calls `promoteNumeric` exactly once to validate a same-category numeric pair,
ignores the promoted type because the result is BOOL, then preserves the existing broadcast,
descriptor, operation, ordered provenance, and factory order.

### Public Tensor contract

No signature or delegation statement changes. Complete Javadocs must explain:

- ADD/SUB/MUL/MIN/MAX accept floating or integral Tensor pairs;
- DIV/POW remain floating-only;
- all six comparisons accept floating or integral pairs;
- scalar ADD/SUB/MUL/MIN/MAX and one-bound clamp conveniences accept exact matching floating or
  integral `ScalarValue`;
- scalar DIV/POW and first-class CLAMP remain floating-only;
- mixed numeric categories require explicit cast; and
- integral results are non-differentiable, unresolved-layout metadata with modular arithmetic or
  signed-order meaning as applicable.

The public Tensor method count remains exactly 127.

## Affected files

Expected production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataTypePromotion.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorBinaryExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorComparisonExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScalarExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/datatype/DataTypePromotionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryComparisonTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScalarElementwiseTest.java`

Expected documentation and planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/tasks/0018u-integral-elementwise-arithmetic-and-comparisons.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review-only unless an out-of-scope discrepancy requires stopping:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataType.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/DataTypeCategory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/datatype/ScalarValue.java`
- operation kind/attributes/signature contracts and their tests
- reduction and arg-max contracts/helpers/tests
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `docs/api/training-api.md`
- focused architecture documentation and architecture tests

## Maximum scope

This task may create or modify exactly the 16 affected paths listed above: five production Java
files, four test files, and seven documentation/planning files including this specification.

If implementation needs a seventeenth path, a new production/test type, another helper method, an
operation-kind edit, a reduction change, or a review-only file, stop and report why before editing
it. Do not omit a required focused test or documentation path merely to stay under the cap.

## Acceptance criteria

- `promoteNumeric` implements the exact floating and integral matrices, null/type/category errors,
  symmetry, and idempotence while `promoteFloating` remains unchanged.
- Binary ADD/SUB/MUL/MIN/MAX accept every INT32/INT64 ordered pair and derive the exact promoted
  result type and broadcast Shape.
- Binary integral DIV/POW fail with exact kind-specific messages before Shape broadcast and ID
  allocation; their floating behavior remains green.
- Scalar ADD/SUB/MUL/MIN/MAX accept exact matching INT32/INT64 `ScalarValue`; scalar DIV/POW reject
  integral receivers before scalar mismatch, and BOOL is rejected as non-numeric.
- Integral one-bound clamp conveniences create scalar MAX/MIN producers; first-class range CLAMP
  remains floating-only.
- All six comparisons accept every INT32/INT64 pair, produce BOOL/false-gradient unresolved
  descriptors, and preserve broadcasting and ordered provenance.
- Floating behavior for all affected operations remains unchanged, and mixed floating/integral or
  BOOL pairs fail before Shape work and ID allocation.
- Tests and Javadocs record modular two's-complement overflow semantics and signed integral
  extrema/comparison ordering without evaluating values in model.
- Every successful expression is fresh, unlabeled, storage-free, one-output, and has exact kind,
  attributes, ordered inputs, and provenance output index zero.
- Public Tensor method count and signatures remain unchanged; no alias, overload, visible cast, or
  new operation kind appears.
- Tensor API, Compile API, glossary, capability baseline, task, master plan, and roadmap are
  synchronized without rewriting completed task history.
- Training API, architecture contract/docs/tests, Gradle, dependencies, other modules, backend
  conformance, and integration tests remain unchanged with reasoned conclusions.
- One final `:modules:model:test` run passes after executable Java stabilizes.
- A separate clean-context documentation-focused pass finalizes Javadocs and documentation in the
  same overall change without repeating successful Java tests unless executable behavior changes
  or a concrete risk is recorded.
- Model Javadoc, a runnable Java 26 integral-metadata example, Markdown links/anchors/fences/final
  newlines, exact 16-path scope, status synchronization, terminology, and `git diff --check` pass.
- Task 0018U is Complete only after both passes succeed. Task 0018U1 and every later task remain
  Draft without a detailed specification.

## Tests / validation

Focused tests while executable Java stabilizes:

```bash
./gradlew :modules:model:test --tests \
  io.github.pho001.synaptik.model.datatype.DataTypePromotionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryComparisonTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScalarElementwiseTest
```

Final Java checkpoint after executable code stabilizes:

```bash
./gradlew :modules:model:test
```

The implementation pass records the exact evidence and hands it to the documentation pass. Do not
repeat a successful final model suite unless executable Java changes afterward or a concrete risk
is recorded.

Documentation-focused pass:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass must also:

- compile and run one Java 26 metadata example covering INT32/INT64 promotion, exact scalar
  attributes, BOOL comparison result, ordered provenance, and output index zero without claiming
  evaluated values;
- validate local Markdown links and anchors in changed documents;
- inspect generated Javadoc for `DataTypePromotion`, all affected helpers, and Tensor methods;
- verify the exact promotion matrices, helper method surfaces, unchanged 127-method public Tensor
  surface, unsupported-kind failures, and absence of aliases/new kinds;
- verify exact 16-path scope and no dependency/Gradle/architecture/other-module changes;
- verify synchronized 0018U Complete status and no detailed 0018U1-or-later specification; and
- review terminology, examples, fences, final newlines, trailing whitespace, authority boundaries,
  and the final diff.

Repository-wide validation is deferred to the capability-reset checkpoint after task 0018V and
CI. This task changes one module without changing dependencies, architecture, or shared build
configuration.

## Dependencies

- [Task 0001](0001-data-type-model.md) — current DataType categories and floating promotion.
- [Task 0014A](0014a-binary-arithmetic-semantic-kinds.md) and
  [0014B](0014b-binary-arithmetic-tensor-expressions.md) — binary kinds and construction.
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md) and
  [0014F](0014f-scalar-arithmetic-and-clamp-tensor-expressions.md) — scalar kinds, exact attributes,
  and construction.
- [Task 0015A](0015a-binary-comparison-semantic-kinds.md) and
  [0015B](0015b-binary-comparison-tensor-expressions.md) — comparison kinds and construction.
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md) — exact structural family
  validation.
- [Task 0018N](0018n-typed-scalar-value-contract.md) — exact INT32/INT64 scalar representation.
- [Task 0018T](0018t-scalar-arithmetic-family-normalization.md) — final parallel arithmetic
  vocabulary and extrema naming.
- [Task 0018T1](0018t1-unary-numeric-gaps-and-floating-diagnostics.md) — completed floating unary
  foundation preceding this frontier.

All dependencies are Complete.

## Follow-up tasks

- **0018U1 Integral reductions and arg-min normalization** — add selected integral SUM/PROD/MIN/
  MAX domains plus `argMin` and normalize the shared arg-extrema tie-policy contract.
- **0018V Multi-axis and statistical reductions** — add ordered axes and floating statistical/norm
  semantics after integral reduction policy is stable.
- Later compiler and backend tasks own graph revalidation, gradients, lowering, modular execution,
  numerical conformance, and kernels.

## Architecture impact

Expected impact: None.

This extends backend-independent model promotion and public expression-construction domains within
existing module/package ownership. It changes no operation structure, module dependency, lifecycle,
architecture rule, backend contract, runtime path, or graph representation.

If implementation requires an architecture change, stop and report it before editing
`ARCHITECTURE.md`, focused architecture documentation, ADRs, or architecture tests.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, roadmap, model capabilities/master
plan, tasks 0001/0014A/0014B/0014E/0014F/0015A/0015B/0018K/0018N/0018T/0018T1/0018U,
Tensor API, Compile API, Training API, glossary, and every affected or review-only source/test
named by task 0018U in full.

Implement task 0018U exactly. Add same-category numeric promotion, selected INT32/INT64
ADD/SUB/MUL/MIN/MAX Tensor and exact-scalar domains, and all six integral comparisons. Preserve all
floating behavior, reject mixed categories/BOOL and integral DIV/POW/range CLAMP, retain exact
validation/descriptor/producer/provenance order, and add no public method or operation kind. Stay
within the exact 16 paths, stop on scope or architecture conflict, and do not commit or push.

Run focused tests as needed and one final model test after executable Java stabilizes. Then hand
the actual diff and exact evidence to a separate clean-context documentation-focused agent in the
same overall change. That agent must inspect final source/tests, finalize permitted Javadocs,
Tensor/Compile API, glossary, capability/task/master/roadmap documentation, run model Javadoc and
documentation/scope checks, and must not repeat successful Java tests unless executable behavior
changes or it records a concrete reason.

Mark 0018U Complete only after both passes succeed. Leave 0018U1 and every later task Draft
without a detailed specification.
```

## Local decisions

- `DataTypePromotion.promoteNumeric` delegates valid floating pairs to the unchanged
  `promoteFloating` contract and selects INT64 for either mixed-width integral ordering. No cast
  producer or promotion object is stored.
- Binary and scalar helpers reject integral DIV/POW after numeric-domain validation but before
  broadcasting or result construction. Exact scalar type equality remains after kind eligibility,
  preserving the specified validation order.
- Existing `clampMin` and `clampMax` delegation supplies integral one-bound behavior through scalar
  MAX/MIN. First-class two-bound CLAMP remains on its unchanged floating-only helper path.
- No method, operation kind, attributes type, package, dependency, or architecture boundary was
  added.

## Known limitations

- Integral DIV, POW, range CLAMP, remainder/modulo, reductions, and arg-min remain unsupported.
  Task 0018U1 owns the selected integral reduction and arg-min policy frontier.
- Model construction records fixed-width modular arithmetic and signed ordering semantics without
  reading or evaluating values. Compiler capture/revalidation, gradients, lowering, backend
  support, kernels, and execution remain unimplemented or separately owned.
- Result layout remains unresolved and every successful expression remains a fresh, unlabeled,
  storage-free producer result with provenance output index zero.

## Validation evidence

- Implementation context `/root/task_0018u_implementation` ran the focused command from this task
  after final executable edits. It passed with `DataTypePromotionTest` 6 tests,
  `TensorBinaryArithmeticTest` 12, `TensorBinaryComparisonTest` 10, and
  `TensorScalarElementwiseTest` 8; all 36 tests had zero failures, errors, or skips.
- The same implementation context then ran `./gradlew :modules:model:test`. It passed with 90 XML
  suites and 734 tests, zero failures, errors, or skips. The documentation context changed only
  Javadocs and documentation afterward and therefore did not repeat this successful Java suite.
- Documentation context `/root/task_0018u_implementation/task_0018u_docs` applied the General,
  API/Javadoc, Planning, and Example profiles. It independently reviewed the final production and
  test diff, finalized all permitted Javadocs and seven documentation/planning paths, and reviewed
  the glossary impact.
- The documentation context ran `./gradlew :modules:model:javadoc` after final Javadoc edits and
  `git diff --check`; both passed. Generated pages for public `DataTypePromotion` and the affected
  `Tensor` methods were inspected. The default public Javadoc output omits the three package-private
  helpers, so their complete source Javadocs were inspected directly instead.
- A Java 26 metadata example compiled and ran against the final model classes. It verified
  INT32/INT64 promotion to INT64, exact retained INT32 scalar attributes, a BOOL comparison result,
  ordered provenance, output index zero, unresolved layout, and absent storage without evaluating
  arithmetic or comparison values.
- Targeted structural checks confirmed both promotion matrices, exact helper method surfaces,
  exactly 127 public declared `Tensor` methods, unchanged seven binary kinds/eight scalar kinds/six
  comparison kinds, integral DIV/POW validation messages and pre-broadcast failure order, no
  aliases or new kinds, valid changed-document links/anchors/fences/final newlines, and the exact
  16-path scope.
- Training API required no change because task 0018U adds no training, gradient, optimizer, or
  publication contract. Architecture and focused architecture documents/tests required no change
  because ownership and dependency rules are unchanged. Operation, reduction, and scan contracts
  required no change because no kind, attributes, signature, or reduction domain changed. Gradle,
  dependencies, other modules, backend conformance, and integration tests required no change
  because this is model-only metadata construction with no backend or end-to-end execution.
- Repository-wide validation remains deferred to the recorded capability-reset checkpoint after
  task 0018V and CI, as planned for this single-module change.

## Implementation notes

- `promoteNumeric` performs exact left/right null, numeric, and same-category validation, preserves
  floating promotion, and implements the complete INT32/INT64 matrix symmetrically and
  idempotently.
- Binary ADD/SUB/MUL/MIN/MAX and all six comparisons now accept every ordered INT32/INT64 pair.
  Binary and scalar integral DIV/POW retain exact failure messages and allocate no result identity.
- Exact scalar INT32/INT64 ADD/SUB/MUL/MIN/MAX and delegated one-bound clamp conveniences preserve
  exact attributes and one-input provenance; range CLAMP remains floating-only.
- Focused tests cover promotion laws and failures, selected integral domains, broadcasting,
  modular semantic requests, signed comparison domains, exact scalar attributes, provenance,
  freshness, storage absence, validation messages/order, and no-ID-on-failure behavior.

## Completion summary

- Completed changes: added same-category numeric promotion and selected signed-integral binary,
  comparison, and exact-scalar expression domains while preserving floating behavior and all
  producer/provenance invariants.
- Files changed or created: exactly the five production Java, four test, and seven
  documentation/planning paths listed under Affected files.
- Tests and validation: focused 36-test command and final 734-test model suite passed in the
  implementation context; model Javadoc, Java 26 metadata example, structural/API checks,
  Markdown validation, exact-scope validation, and `git diff --check` passed in the documentation
  context.
- Documentation-agent review: clean context
  `/root/task_0018u_implementation/task_0018u_docs` completed the independent targeted pass.
- Documentation impact: Tensor API, Compile API, capability baseline, task, master plan, and
  roadmap now distinguish current integral elementwise metadata from planned compiler/backend and
  reduction work.
- Javadoc review: finalized `DataTypePromotion`, the three helpers, the `Tensor` type summary, and
  every affected public Tensor method without changing signatures or method count.
- Glossary impact: documented numeric promotion, modular integral arithmetic, signed ordering,
  exact scalar matching, and current-vs-planned boundaries.
- Unresolved issues: None within task scope.
- Follow-up required: None. Task 0018U1 remains the next Draft frontier without a detailed spec.

Status: Complete
