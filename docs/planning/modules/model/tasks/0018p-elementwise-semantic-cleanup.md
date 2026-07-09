# Task 0018P: Elementwise Semantic Cleanup

## Status

Complete

## Goal

Finalize the parameterless unary elementwise vocabulary before the public API stabilizes:

```text
one portable mathematical request -> one semantic kind -> one public Tensor method
backend prepare                  -> implementation-route selection
```

Atomically rename the reciprocal request from `INV`/`inv()` to
`RECIPROCAL`/`reciprocal()`, and remove `FAST_EXP`/`fastExp()` and
`FAST_TANH`/`fastTanh()` completely. No old and new spelling may coexist in the live model API.

Retain `EXP`/`exp()` and `TANH`/`tanh()` as portable mathematical requests. They identify the
natural exponential and hyperbolic tangent, respectively, without selecting an algorithm,
promising a bitwise result or approximation bound, or naming a backend route. Concrete backend
prepare later owns implementation-route selection within the eventual numerical contract; this
task adds no prepare or backend behavior.

The cleanup must preserve the completed unary construction contract and the completed typed
scalar/clamp contract. It changes vocabulary, public surface, tests, Javadocs, and explanatory
documentation only; it does not calculate values or add the numeric gaps assigned to task 0018T.

## Mental model

The final unary family contains one kind for each public request:

```text
Tensor.reciprocal() -> UnaryElementwiseKind.RECIPROCAL
Tensor.exp()        -> UnaryElementwiseKind.EXP
Tensor.tanh()       -> UnaryElementwiseKind.TANH
```

Every successful call records the chosen kind with `NoOperationAttrs.INSTANCE`, one exact input,
one exact output descriptor, and provenance output index zero. The model records meaning. It does
not record whether a backend later uses a library routine, hardware instruction, polynomial,
table, vectorized kernel, or another implementation route.

The scalar family remains a separate parameterized family:

```text
ScalarElementwiseKind + ScalarValueAttrs/ClampRangeAttrs
```

This task neither adds scalar add/subtract/divide/minimum/maximum nor moves reciprocal into the
scalar family. Those additions and the separate `rsqrt`, `log1p`, `expm1`, and floating diagnostic
requests remain task 0018T work.

## Current problems

- `INV` and `inv()` abbreviate a mathematical reciprocal and conflict with the selected explicit
  public name.
- `FAST_EXP` and `FAST_TANH` expose implementation intent without a portable accuracy or
  special-value contract.
- Existing Javadocs call `EXP` and `TANH` “strict” requests only to distinguish them from the fast
  variants. After those variants are removed, that wording can be misread as a bitwise or
  algorithm guarantee.
- The current unary API reference uses `fastExp()` as its complete example and therefore teaches
  a request that is leaving the selected baseline.
- Reflection inventories, exact enum-order tests, operation-composition tests, and public method
  counts encode the provisional names and must change in the same migration.
- The completed scalar family is adjacent to this cleanup, but adding its missing operations here
  would silently broaden the task and overlap task 0018T.

## Before / after API

| Current live contract | Final contract | Disposition |
|---|---|---|
| `UnaryElementwiseKind.INV` | `UnaryElementwiseKind.RECIPROCAL` | Atomic rename; no alias. |
| `Tensor.inv()` | `Tensor.reciprocal()` | Atomic rename; no deprecated bridge. |
| `UnaryElementwiseKind.FAST_EXP` | none | Delete completely. |
| `Tensor.fastExp()` | none | Delete completely. |
| `UnaryElementwiseKind.FAST_TANH` | none | Delete completely. |
| `Tensor.fastTanh()` | none | Delete completely. |
| `UnaryElementwiseKind.EXP` / `Tensor.exp()` | unchanged names | Retain as the portable natural-exponential request. |
| `UnaryElementwiseKind.TANH` / `Tensor.tanh()` | unchanged names | Retain as the portable hyperbolic-tangent request. |
| five `ScalarElementwiseKind` values and existing public scalar methods | unchanged | Preserve exactly; task 0018T owns missing scalar operations. |

Completed tasks 0014C and 0014D remain historical records of the provisional surface. They are
not rewritten to pretend they originally implemented the final vocabulary.

## Scope

- Replace `UnaryElementwiseKind.INV` with `RECIPROCAL` in the same declaration position.
- Remove `FAST_EXP` and `FAST_TANH` from `UnaryElementwiseKind` without replacement.
- Make the final enum contain exactly thirteen constants in the order specified below.
- Replace public `Tensor.inv()` with `Tensor.reciprocal()` and remove `fastExp()` and `fastTanh()`.
- Make the final public unary method inventory contain exactly thirteen zero-argument methods in
  one-to-one correspondence with the final enum.
- Preserve the existing parameterless one-input, one-output operation signature.
- Preserve floating-only unary eligibility, exact result metadata, construction order,
  one-input producer/provenance, freshness, and absence of value inspection.
- Remove “strict versus fast” terminology from current Javadocs and API documentation.
- Document `EXP` and `TANH` as portable mathematical semantic requests with no algorithm,
  bitwise-result, approximation-bound, or backend-route promise.
- Update exact enum, method, reflection, public-API, absence, operation-signature, provenance, and
  chaining tests.
- Replace the Tensor API's current complete `fastExp()` example with a complete
  `reciprocal()`/`RECIPROCAL` metadata example.
- Update the Compile API's current unary method count and terminology without adding compiler
  behavior.
- Update glossary, capability baseline, task evidence, model master plan, and roadmap through the
  required independent documentation-focused pass.
- Review and explicitly preserve the completed `ScalarElementwiseKind`, `ScalarValueAttrs`, and
  `ClampRangeAttrs` contracts from task 0018N.

## Out of scope

- aliases, deprecated methods, forwarding bridges, duplicate enum constants, compatibility
  adapters, parsers, migration shims, or serialized-name translation for removed spellings
- `rsqrt`, `log1p`, `expm1`, `isFinite`, `isNaN`, or `isInf`; task 0018T owns them
- scalar add, scalar subtract, scalar divide, scalar minimum, or scalar maximum; task 0018T owns
  their semantics and public methods
- changes to `ScalarElementwiseKind`, `ScalarValueAttrs`, `ClampRangeAttrs`, `ScalarValue`,
  `TensorScalarExpressions`, or their public Tensor overloads
- integral or BOOL unary eligibility, implicit conversion, cast insertion, result-type changes, or
  additional overloads
- numerical evaluation, domain checks, zero handling, overflow, underflow, NaN, infinity,
  signed-zero policy, rounding, accuracy, determinism, or special-value promises
- gradient formulas, subgradient conventions, autograd traversal, backward graph construction,
  optimizer, or training behavior
- compiler capture, canonicalization, common-subexpression elimination, constant folding, or
  graph-wide validation
- planning ownership, capability-provider implementation, prepare lowering, backend routing,
  kernels, fusion, runtime, execution, tracing, or engine behavior
- changes to `OperationSignature`, `OperationKind`, `Operation`, `TensorProducer`,
  `TensorProvenance`, `TensorFactory`, descriptors, Shapes, layouts, storage, or identifiers
- dependencies, Gradle, Java version, preview/incubator configuration, architecture rules,
  focused architecture documents, architecture tests, another module, or a detailed 0018Q-or-
  later task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially model ownership of public
  Tensor and operation semantics, backend-prepare ownership of lowering/kernel selection, and the
  prohibition on backend support in `Operation`
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially the existing capability decisions
  for `fastExp`, `fastTanh`, and `inv`
- [Model master plan](../master-plan.md)
- [Task 0014C](0014c-unary-elementwise-semantic-kinds.md)
- [Task 0014D](0014d-unary-elementwise-tensor-expressions.md)
- [Task 0014E](0014e-scalar-arithmetic-and-clamp-semantics.md)
- [Task 0014F](0014f-scalar-arithmetic-and-clamp-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0018O](0018o-indexing-taxonomy-and-unstack-normalization.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns the final backend-independent unary kind names and public Tensor expression
  construction.
- `Operation` records semantic meaning, never backend support or an implementation route.
- `EXP` and `TANH` remain mathematical meanings. Backend prepare may later choose a concrete route
  only within the numerical contract established by an owning execution/conformance task.
- No model field, enum flag, attribute, tag, cost, accuracy class, “fast” bit, backend hint, or
  route selector may replace the removed fast kinds.
- `Tensor` remains mutable public API state rather than graph intermediate representation.
- Unary construction remains metadata-only and must not inspect host storage, values, provenance
  ancestors, or backend capabilities.
- The parameterless signature remains exact `NoOperationAttrs`, one input, and one output. Kind
  vocabulary cleanup must not change signature mechanics.
- Every result remains one output of one identity-distinct `TensorProducer`; its
  `TensorProvenance.outputIndex()` is zero.
- Package direction remains `model.tensor -> model.operation.elementwise.unary ->
  model.operation`. No package or module dependency changes.
- Completed scalar semantics remain exact typed attributes. They are not folded into the unary
  family and do not expand in this task.
- Stop if implementation reveals another semantic ambiguity, requires numerical policy, or needs
  a production concept or path outside the bounded scope. Do not silently broaden the cleanup.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.elementwise.unary` — owns the final parameterless
  unary semantic enum.
- `io.github.pho001.synaptik.model.tensor` — owns the public fluent unary methods, package-private
  construction helper, producer/provenance attachment, and exact public API inventory tests.
- `io.github.pho001.synaptik.model.operation.elementwise.scalar` — review-only adjacent family;
  its vocabulary and attributes remain unchanged.
- `io.github.pho001.synaptik.model.operation` — review-only signature and parameterless-attributes
  foundation.

Packages added or moved:

- None.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind` remains the
  public unary semantic family; only its constants and Javadocs change.
- `io.github.pho001.synaptik.model.tensor.Tensor` remains the public fluent surface; only the
  specified unary declarations, type-level wording, and Javadocs change.
- `io.github.pho001.synaptik.model.tensor.TensorUnaryExpressions` remains the package-private,
  field-free common construction boundary; executable behavior remains unchanged.

No package-map or visibility change is permitted.

## Required contracts

### Atomic migration and compatibility policy

The migration is deliberately breaking and atomic. After implementation:

- `INV`, `FAST_EXP`, and `FAST_TANH` do not exist as enum constants;
- `inv`, `fastExp`, and `fastTanh` do not exist as declared Tensor methods;
- `UnaryElementwiseKind.valueOf("INV")`, `valueOf("FAST_EXP")`, and
  `valueOf("FAST_TANH")` fail with standard enum lookup failure;
- reflection cannot find the three removed Tensor methods;
- no deprecated bridge, forwarding alias, package-private duplicate, string mapping, or alternate
  semantic kind retains the old names; and
- completed historical task documents may retain old names as history, while live Java and
  current API/glossary documentation must not.

### Final unary semantic vocabulary

`UnaryElementwiseKind` declares exactly:

```java
ABS,
NEG,
RECIPROCAL,
LOG,
EXP,
ERF,
SQRT,
FLOOR,
CEIL,
SIGN,
RELU,
SIGMOID,
TANH
```

The declaration order is part of the exact tested public surface. The meanings are:

| Kind | Portable elementwise meaning |
|---|---|
| `ABS` | Absolute magnitude. |
| `NEG` | Additive inverse. |
| `RECIPROCAL` | Multiplicative reciprocal. |
| `LOG` | Natural logarithm. |
| `EXP` | Natural exponential. |
| `ERF` | Gaussian error function. |
| `SQRT` | Principal square root. |
| `FLOOR` | Greatest integer-valued result not greater than the input. |
| `CEIL` | Least integer-valued result not less than the input. |
| `SIGN` | Negative, zero, or positive classification represented numerically. |
| `RELU` | Rectified linear unit. |
| `SIGMOID` | Logistic sigmoid. |
| `TANH` | Hyperbolic tangent. |

The enum retains one stable immutable signature list containing exactly
`OperationSignature.fixed(NoOperationAttrs.class, 1, 1)`. It adds no field or method other than
the existing family-owned signature implementation and compiler-generated enum machinery.

`EXP` and `TANH` do not mean “strict algorithm,” correctly rounded result, bitwise reproducibility,
or a mandatory library/kernel route. They are portable mathematical requests whose later
executable implementations must satisfy numerical policy defined outside this task. Removing the
fast variants does not strengthen their current numerical promise.

### Final public unary Tensor surface

`Tensor` declares exactly these zero-argument unary elementwise methods:

```java
public Tensor abs()
public Tensor neg()
public Tensor reciprocal()
public Tensor log()
public Tensor exp()
public Tensor erf()
public Tensor sqrt()
public Tensor floor()
public Tensor ceil()
public Tensor sign()
public Tensor relu()
public Tensor sigmoid()
public Tensor tanh()
```

The exact mapping is one-to-one and order-aligned with the enum:

| Tensor method | Kind |
|---|---|
| `abs()` | `ABS` |
| `neg()` | `NEG` |
| `reciprocal()` | `RECIPROCAL` |
| `log()` | `LOG` |
| `exp()` | `EXP` |
| `erf()` | `ERF` |
| `sqrt()` | `SQRT` |
| `floor()` | `FLOOR` |
| `ceil()` | `CEIL` |
| `sign()` | `SIGN` |
| `relu()` | `RELU` |
| `sigmoid()` | `SIGMOID` |
| `tanh()` | `TANH` |

Each method delegates exactly once to
`TensorUnaryExpressions.apply(this, <matching kind>)` and returns that exact result. It performs
no duplicate validation, result construction, value inspection, canonicalization, or exception
translation.

The complete declared public `Tensor` method count becomes 110: the current 112 declarations,
minus `fastExp()` and `fastTanh()`, with `inv()` replaced one-for-one by `reciprocal()`.

### Preserved unary validation and result construction

`TensorUnaryExpressions` remains a final package-private non-record class with no fields or nested
types, one private zero-argument constructor, and exactly one package-private static method:

```java
static Tensor apply(Tensor input, UnaryElementwiseKind kind)
```

`apply` retains this exact order and behavior:

1. null-check `input`, then `kind`, with messages `input` and `kind`;
2. read the exact input `DataType` and reject non-floating input with
   `IllegalArgumentException("input must be a floating data type, but was " + dataType)`;
3. create one new `TensorDescriptor` with the exact input data type, exact input `Shape`
   reference, `Optional.empty()` layout, and exact input `requiresGrad` value;
4. create one `Operation(kind, NoOperationAttrs.INSTANCE)`;
5. create one identity-distinct one-output producer through the existing factory path with exact
   ordered inputs `List.of(input)`; and
6. return the exact fresh Tensor at output index zero.

Validation failures precede Tensor identity allocation. Identity exhaustion is allowed to fail
only through the existing factory path after local immutable model values have been constructed.
Do not add an identifier inspection API.

Every successful method therefore preserves:

- exact `BFLOAT16`, `FLOAT32`, or `FLOAT64` input eligibility;
- exact input data type, exact immutable Shape reference, and exact `requiresGrad` value;
- unresolved result layout for scalar, zero-sized, static, dynamic, and resolved-layout inputs;
- a fresh Tensor identity, empty label, and absent host storage;
- exact matching kind with `NoOperationAttrs.INSTANCE`;
- one identity-distinct `TensorProducer` with exactly the receiver as its sole ordered input and
  exactly one output descriptor;
- `TensorProvenance.outputIndex() == 0` and the exact result descriptor reference; and
- input descriptor, provenance, label, storage association, and storage contents unchanged.

Repeated calls and chains remain fresh. `reciprocal().reciprocal()` is not replaced by the
original input, and no other unary chain is simplified. `reciprocal()` does not inspect zero;
`log()` and `sqrt()` do not inspect mathematical domains; no unary method reads values.

### Retained scalar elementwise contract

No scalar production or test file changes. The final adjacent scalar vocabulary remains exactly:

```java
MUL,
POW,
CLAMP,
CLAMP_MIN,
CLAMP_MAX
```

The pairings remain:

- `ScalarValue` remains one final immutable `DataType` plus exact primitive-bit payload with the
  existing named FLOAT64, FLOAT32, converted/raw BFLOAT16, INT32, INT64, and BOOL factories,
  strict type-specific inspectors, exact type-and-bit equality/hashing, and no general conversion
  API;
- `MUL`, `POW`, `CLAMP_MIN`, and `CLAMP_MAX` accept exactly `ScalarValueAttrs`, one input, and one
  output;
- `CLAMP` accepts exactly `ClampRangeAttrs`, one input, and one output;
- `ScalarValueAttrs` retains one non-null exact `ScalarValue` reference without conversion;
- `ClampRangeAttrs` null-checks `minValue` then `maxValue`, requires the same data type, rejects
  BOOL, and rejects only strict inversion in the represented primitive type;
- `ClampRangeAttrs` continues to accept equal bounds, either signed-zero ordering, ordered
  infinities, and floating NaN endpoints while retaining exact typed bits; and
- public typed scalar/clamp overloads, exact FLOAT64 `double` adapters, floating receiver/value
  equality, validation order, metadata, freshness, and provenance remain unchanged.

This task does not add missing scalar operations and does not change `ScalarValue`. Focused scalar
tests run as regression coverage and should require no source edit.

### Boundary with task 0018T

Task 0018P removes ambiguous or non-portable surface and finalizes the existing unary vocabulary.
Task 0018T later adds capabilities:

- scalar add, subtract, divide, minimum, and maximum using the exact typed scalar foundation;
- reciprocal-adjacent `rsqrt`, `log1p`, and `expm1`; and
- floating diagnostics `isFinite`, `isNaN`, and `isInf`.

Task 0018T may consume `RECIPROCAL`; it must not reintroduce `INV`, fast variants, or aliases. No
detailed task 0018T specification is created here.

### Test migration

Update only these executable tests:

- `UnaryElementwiseKindTest` — assert the exact thirteen constants/order/names, canonical
  no-attributes composition, `RECIPROCAL` identity, absence of removed enum names through
  `valueOf`, and the unchanged signature enum shape. Remove strict-versus-fast assertions.
- `TensorUnaryElementwiseTest` — use the exact thirteen method-to-kind mappings; replace every
  inverse chain/domain assertion with reciprocal terminology; remove fast-variant construction;
  prove fresh reciprocal chains, no zero inspection, exact producer/output-index-zero behavior,
  and absence of `inv`, `fastExp`, and `fastTanh` through reflection.
- `TensorTest` — update the exact declared public method count to 110, replace `inv` with
  `reciprocal` in the name inventory, remove `fastExp` and `fastTanh`, and assert the exact
  thirteen zero-argument unary declarations.

`OperationSignatureTest` remains unchanged because it already applies the same exact one-input,
one-output no-attributes signature to `UnaryElementwiseKind.values()`. `TensorCastExpressionTest`
remains unchanged because it uses retained `ABS` only. Scalar semantic/expression tests remain
unchanged and supply regression evidence.

### Documentation migration

- `UnaryElementwiseKind`, `Tensor`, and `TensorUnaryExpressions` Javadocs use “thirteen,”
  `RECIPROCAL`, and `reciprocal()`, remove the fast variants and strict-versus-fast wording, and
  preserve all current ownership/failure/result boundaries.
- Tensor API lists exactly thirteen kinds and methods, uses `RECIPROCAL`/`reciprocal`, removes all
  current `INV`/`inv`/fast entries, and explains `EXP`/`TANH` portability without algorithm,
  bitwise-result, bound, or route promises.
- Replace the complete `fastExp()` example with a runnable metadata-only reciprocal example using
  `input.reciprocal()` and expected `kind=RECIPROCAL`. Its interpretation must state that it proves
  metadata/provenance only, not division, zero/special-value behavior, numerical accuracy,
  gradients, capture, backend support, or execution.
- Compile API changes the current unary method count from fifteen to thirteen and aligns the
  retained semantic wording. It must not claim compiler capture, inference, optimization,
  lowering, or execution.
- Glossary changes the exact unary inventory and current expression count, removes fast-request
  terminology, and uses reciprocal terminology in provenance/status descriptions.
- Capability baseline records task 0018P as the ready cleanup owner during planning and as the
  completed owner only after implementation evidence is final. It keeps task 0018T's additions
  separate.
- Model master plan and roadmap link this specification and synchronize status/evidence without
  rewriting completed 0014C/0014D history.
- Training API receives a reasoned no-change conclusion because no gradient, autograd, optimizer,
  or training behavior changes.

## Affected files

Production:

1. `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKind.java`
2. `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorUnaryExpressions.java`
3. `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests:

4. `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/unary/UnaryElementwiseKindTest.java`
5. `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorUnaryElementwiseTest.java`
6. `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`

Documentation and planning:

7. `docs/api/tensor-api.md`
8. `docs/api/compile-api.md`
9. `docs/glossary.md`
10. `docs/planning/modules/model/capabilities.md`
11. this task specification
12. `docs/planning/modules/model/master-plan.md`
13. `docs/planning/roadmap.md`

Required review without modification unless a contradiction requires stopping:

- `ScalarElementwiseKind`, `ScalarValueAttrs`, `ClampRangeAttrs`, `ScalarValue`,
  `TensorScalarExpressions`, `ScalarElementwiseSemanticsTest`, and
  `TensorScalarElementwiseTest`
- `OperationSignature`, `OperationKind`, `Operation`, `OperationSignatureTest`,
  `TensorProducer`, `TensorProvenance`, and `TensorFactory`
- `TensorCastExpressionTest`, whose retained `ABS` provenance setup is unaffected
- Training API, `ARCHITECTURE.md`, focused architecture docs, ADRs/tests, backend-conformance,
  integration tests, Gradle configuration, dependencies, and other modules

## Maximum scope

This task may modify exactly the thirteen listed paths and may not create or delete another path.

This is a bounded one-path exception to the planning guide's approximate 3–12-file preference.
The thirteenth path is not a second capability: the three production surfaces, their three exact
inventory/behavior tests, and seven required API/glossary/capability/task/master/roadmap records
must migrate together so old and new names cannot coexist. Splitting documentation or public-
inventory updates would leave a contradictory stable API; splitting semantic and public names
would make the build uncompilable. The task remains within one module and one cohesive cleanup.

Do not use this allowance for scalar changes, operation-signature changes, unrelated formatting,
another test, another document, a later detailed specification, or a compatibility bridge. If a
fourteenth path or another semantic ambiguity is discovered, stop and report the exact need before
editing it.

## Javadoc requirements

- `UnaryElementwiseKind` type Javadoc must define the final thirteen-kind family, exact shared
  signature, semantic-only role, and diagnostic-name boundary.
- Every retained constant must have meaningful mathematical Javadoc and defer type/result,
  special-value, numerical, gradient, execution, and backend rules as applicable.
- `RECIPROCAL` must say multiplicative reciprocal and must not use “inverse” as the API name or
  promise zero handling.
- `EXP` and `TANH` must state portable mathematical meaning without “strict” contrast and without
  promising an algorithm, correctly rounded/bitwise result, approximation bound, or route.
- Every retained public Tensor method must document floating eligibility, exact metadata
  preservation, unresolved layout, fresh identity, no label/storage, exact kind/no-attributes,
  one-input producer/provenance output zero, deferred value/numerical/gradient/backend behavior,
  `@return`, and expected failures.
- `Tensor.reciprocal()` must document zero and special-value policy as deferred.
- `TensorUnaryExpressions` must say thirteen and retain its exact validation, ownership, ID-side-
  effect, and failure documentation.
- Review the scalar family Javadocs and record a reasoned no-change conclusion rather than editing
  them.
- A separate clean-context documentation-focused agent must inspect final source, tests,
  generated Javadoc, Tensor/Compile APIs, glossary, capability baseline, and the complete diff
  before completion.

## Acceptance criteria

- `UnaryElementwiseKind` contains exactly the thirteen specified constants in exact order.
- `Tensor` contains exactly the thirteen specified public zero-argument unary elementwise methods
  and maps each once to the matching kind.
- `INV`, `FAST_EXP`, `FAST_TANH`, `inv`, `fastExp`, and `fastTanh` are absent from live production
  and test Java, including reflection and `valueOf` checks.
- No alias, deprecated bridge, string translation, alternate kind, or route flag preserves a
  removed spelling.
- `EXP` and `TANH` remain representable portable mathematical requests with no strengthened
  numerical or backend promise.
- Every final unary kind retains exact `NoOperationAttrs`, one-input, one-output signature
  behavior.
- Null/type validation order and exact messages remain unchanged.
- Every successful unary call accepts only floating input, preserves exact Shape/type/
  `requiresGrad`, leaves layout unresolved, is fresh/unlabeled/storage-free, and records one exact
  input in an identity-distinct one-output producer with provenance output index zero.
- Reciprocal and other unary chains are not canonicalized; values, storage, and mathematical
  domains are not inspected.
- `ScalarElementwiseKind` remains exactly `MUL`, `POW`, `CLAMP`, `CLAMP_MIN`, `CLAMP_MAX`; exact
  `ScalarValueAttrs`/`ClampRangeAttrs` contracts and scalar public methods remain unchanged.
- No task-0018T scalar or unary gap is implemented here.
- Tensor's exact public declared-method count and name inventory are updated to the final surface.
- Tensor API's former fast-exp example is replaced with the specified reciprocal metadata
  example, and current API/glossary text contains no removed vocabulary.
- Compile API reports thirteen unary methods without claiming compiler behavior.
- Focused tests, final model tests, model Javadoc, documentation checks, exact scope checks, and
  `git diff --check` pass.
- The independent documentation pass records reasoned no-change conclusions for scalar contracts,
  operation/signature/provenance foundations, Training API, architecture/ADRs/tests,
  conformance/integration, Gradle/dependencies, and other modules.
- Task, model master-plan row, capability baseline, and roadmap status are synchronized only after
  evidence is final.
- 0018Q and later tasks remain Draft without detailed specifications.

## Tests / validation

During implementation, run the focused contract set:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKindTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorUnaryElementwiseTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScalarElementwiseTest
```

After executable Java stabilizes, record one final module run:

```bash
./gradlew :modules:model:test
```

The implementation pass also runs exact live-Java absence checks:

```bash
rg -n '\bINV\b|FAST_EXP|FAST_TANH|\binv\(|fastExp\(|fastTanh\(' \
  modules/model/src/main/java modules/model/src/test/java
```

The result must be empty. Automated reflection and enum tests must prove the same public absence,
the exact thirteen-item inventories, method count 110, signature, producer, descriptor, and
output-index contracts. Do not use repeated manual bytecode checks for facts covered by stable
tests.

The separate documentation-focused pass reuses the successful model-test evidence unless it
changes executable Java. After final Javadoc edits it runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It also:

- compiles and runs the replacement reciprocal example against the final Java 26 model classes;
- confirms exact enum and Tensor surfaces with the automated evidence and a targeted generated-
  Javadoc inspection;
- checks that current Tensor API, Compile API, and glossary contain none of the removed names;
- checks local Markdown file links and GitHub-style anchors in all seven changed documentation
  and planning files;
- checks balanced fences, final newlines, trailing whitespace, terminology, and authority
  boundaries;
- verifies exactly the thirteen authorized paths;
- verifies 0018P is `Complete` only after both passes, 0018Q is the next Draft row, and no 0018Q-
  or-later detailed task exists; and
- confirms no architecture, Gradle, dependency, other-module, commit, or push change.

Repository-wide `./gradlew test` is deferred to the recorded public-surface cleanup checkpoint
after task 0018S. This task changes one module and no dependency, build, or architecture boundary.

## Dependencies

- Tasks 0014C and 0014D — completed provisional unary kinds and public expression construction.
- Tasks 0014E and 0014F — completed scalar vocabulary and expression boundary retained unchanged.
- Task 0018K — completed exact kind/attributes signatures and occurrence cardinality.
- Task 0018N — completed exact typed scalar values and attributes retained unchanged.
- Task 0018O — completed the immediately preceding public-taxonomy cleanup.

All dependencies are Complete. No unresolved architecture or package dependency blocks this task.

## Follow-up tasks

- 0018Q — masked reduction redesign; independent of this unary cleanup.
- 0018T — add scalar add/subtract/divide/minimum/maximum plus `rsqrt`, `log1p`, `expm1`, and
  floating diagnostic semantics on the cleaned vocabulary.
- 0019A — add selected modern activation conveniences after 0018P and 0018T.
- Prepare/backend/conformance tasks — select implementation routes and establish executable
  numerical/special-value contracts for retained mathematical requests.

Do not create a detailed follow-up specification in this task.

## Architecture impact

Expected impact: None.

The architecture already assigns backend-independent operation semantics and public Tensor
construction to `modules/model`, while concrete backend prepare owns lowering and kernel route
selection. Removing route-oriented semantic names and adopting explicit reciprocal terminology
clarifies that boundary without changing it.

If implementation requires a module/dependency/lifecycle rule change, numerical contract, or
backend behavior, stop and report the exact conflict before editing architecture files.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md, model capabilities/master plan, tasks
0014C/0014D/0014E/0014F/0018K/0018N/0018O/0018P, and every affected/review-only source and test
listed by task 0018P in full.

Implement task 0018P exactly. Perform the atomic INV/inv to RECIPROCAL/reciprocal migration and
complete FAST_EXP/FAST_TANH/fastExp/fastTanh removal. Preserve unary construction, scalar
contracts, signatures, producer/provenance, metadata, and failure order. Do not add aliases,
numeric policy, task-0018T capabilities, compiler/prepare/backend behavior, dependencies, build,
architecture changes, or paths outside the exact thirteen-path scope. Stop on any new semantic
ambiguity or scope/architecture conflict. Do not commit or push.

Run the focused command, final model suite, and live-Java absence check after executable code
stabilizes. Then hand the actual diff and exact test evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently finalize
the affected Javadocs, replace the Tensor API fast-exp example with the reciprocal example,
finalize Tensor/Compile APIs, glossary, capabilities, task/master/roadmap evidence, and run model
Javadoc plus all specified documentation/scope checks without repeating successful Java tests
unless executable behavior changes or a concrete risk is recorded.

Mark 0018P Complete only after both passes succeed. Leave 0018Q and every later task Draft without
a detailed specification.
```

## Local decisions

- The final kind and method name is `RECIPROCAL`/`reciprocal`, not `INV`, `INVERSE`, or a scalar
  division convenience, because it names the elementwise mathematical request directly.
- No compatibility aliases are retained. The API is still pre-stabilization, and coexistence
  would preserve duplicate semantic identities and enlarge every downstream inventory.
- `EXP` and `TANH` remain one portable semantic request each. Backend prepare selects the route;
  model does not expose a speed/accuracy route preference without a portable contract.
- The current reciprocal declaration position is retained, so the final enum and public method
  ordering changes only by the rename and removal of the two trailing fast variants.
- The scalar family remains unchanged. Its missing operations and reciprocal-adjacent additions
  are cohesive task 0018T work, not cleanup fallout.
- The complete API example becomes reciprocal-focused because it demonstrates the renamed public
  contract and metadata boundary without implying that `EXP` now has a stronger numerical
  guarantee.

## Known limitations

- The final unary expressions still have no computed values or host storage.
- Numerical domain, special-value, accuracy, determinism, and gradient policies remain undefined
  until owning compiler/backend/conformance tasks specify them.
- No backend is made capable of executing the retained requests by this cleanup.
- Existing completed task documents retain the provisional history and therefore still contain
  the removed names outside current API/glossary and live Java.
- Task 0018T remains Draft and its additions are not available after this task alone.

## Validation evidence

Planning context `/root/plan_0018p` read the architecture contract and current architecture index;
documentation rules and General, API/Javadoc, Planning, and Example profiles; planning guide and
roadmap; model capabilities and master plan; completed tasks 0014C, 0014D, 0014E, 0014F, 0018K,
0018N, and 0018O; relevant Tensor/Compile API and glossary sections; Java 26 Gradle configuration;
and the current unary/scalar/signature/producer/provenance production and focused test surfaces.

Planning found no architecture conflict or additional semantic ambiguity. The current helper,
signature, descriptor, producer, and provenance contracts already support the final vocabulary
without executable redesign. The exact thirteen-path scope is cohesive and the scalar family can
remain source- and behavior-unchanged.

Planning-only validation is recorded before handoff:

- a canonical-section scan found every required task-specification section exactly once and no
  missing or duplicate section; the task contains no unresolved design placeholder;
- the final enum/method inventories, failure order, before/after mapping, scalar no-change
  contract, 0018T boundary, affected paths, and maximum scope are explicit;
- status scans found 0018P Ready exactly once in this task, the master-plan row, and the roadmap
  row; 0018O remains Complete, and 0018Q and every later row remain Draft;
- a task-file inventory found no 0018Q-or-later detailed specification;
- a local Markdown target-and-GitHub-anchor check resolved all 309 relative links across this
  task, model master plan, capability baseline, and roadmap;
- fence checks found 24 balanced backtick fences in this task and two in the master plan, with no
  unbalanced tilde fence; all four changed files have final newlines and no trailing whitespace;
- final planning-only scope contains exactly four paths: this new task plus the model master plan,
  capability baseline, and roadmap; no Java, test, Gradle, architecture, API, glossary, other-
  module, completed-task, or later-task file changed; and
- `git diff --check` passed with no output; the untracked task file separately passed the
  trailing-whitespace and final-newline checks that a tracked diff does not inspect.

No Gradle test or Javadoc task was run because this planning-only change modifies no Java,
executable behavior, public Javadoc, or current API documentation.

Implementation context `/root/task_0018p_implementation` completed the executable migration and
reported this final Java evidence, which the documentation pass reused because no executable Java
changed afterward:

- the exact focused Gradle command from this task passed with `BUILD SUCCESSFUL`; XML reports
  contain 50 tests across `UnaryElementwiseKindTest` (6), `TensorUnaryElementwiseTest` (8),
  `TensorTest` (15), `OperationSignatureTest` (5), `ScalarElementwiseSemanticsTest` (10), and
  `TensorScalarElementwiseTest` (6), with zero failures, errors, or skips;
- `./gradlew :modules:model:test` passed with `BUILD SUCCESSFUL in 1s`; XML reports contain 725
  tests across 88 suites with zero failures, errors, or skips;
- the exact live-Java search for `INV`, both fast constants, and the three removed method
  spellings returned no matches; and
- `git diff --check` passed with no output after implementation.

Independent documentation context `/root/task_0018p_implementation/task_0018p_docs` read the
architecture contract and current architecture index; documentation rules and General,
API/Javadoc, Planning, and Example profiles; planning guide and roadmap; model capabilities,
master plan, and task 0018P; the final affected source/tests and actual combined diff; Tensor,
Compile, and Training APIs; glossary; and the adjacent scalar, signature, operation, producer,
provenance, and factory contracts. It found no architecture, executable-contract, or scope defect
and finalized the three affected Javadocs plus all seven authorized documentation/planning paths.

Final documentation evidence:

- `./gradlew :modules:model:javadoc` passed after the final Javadocs with `BUILD SUCCESSFUL in
  1s`; two actionable tasks executed and the configuration cache was reused. Targeted generated
  pages show the thirteen-kind family, `RECIPROCAL`, the portable `EXP`/`TANH` boundaries, and the
  thirteen-method Tensor summary, with no removed public vocabulary.
- `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-reciprocal-doc-example /tmp/UnaryExpressionExample.java && java -cp
  modules/model/build/classes/java/main:/tmp/synaptik-reciprocal-doc-example
  UnaryExpressionExample` compiled and ran against the final Java 26 model classes. It printed
  `FLOAT32`, exact Shape retention, unresolved layout, true gradient eligibility, absent label and
  storage, `kind=RECIPROCAL`, canonical no-attributes, exact input, `outputIndex=0`, one output,
  and fresh identity exactly as documented.
- Current Tensor API, Compile API, and glossary searches contain none of `INV`, `FAST_EXP`,
  `FAST_TANH`, `inv(`, `fastExp`, or `fastTanh`. The matching live production/test Java search is
  also empty.
- The targeted local Markdown checker resolved 441 local links, including 113 heading anchors,
  across the seven changed documentation/planning files with zero failures. Formatting checks
  found balanced fences, final newlines, and no trailing whitespace in all seven files.
- Preliminary local checker runs exposed only checker defects: an unescaped Ruby heading marker,
  collapsed double-hyphen GitHub slugs, and a scope-list expression that retained only the
  untracked-file command output. The corrected checker reruns produced the successful Markdown and
  exact-scope results recorded here; they did not expose repository defects.
- The final inventory contains exactly the thirteen authorized paths: three production Java
  files, three tests, and seven documentation/planning files. Task 0018P, its master-plan row, and
  its roadmap row are Complete; 0018Q and every later task remain Draft, and no detailed
  0018Q-or-later task file exists.
- Final `git diff --check` passed with no output. No architecture contract or focused architecture
  page, ADR, architecture test, Training API, backend-conformance/integration test, Gradle or
  dependency file, other module, commit, or push changed.

Repository-wide `./gradlew test` remains deferred to the recorded public-surface cleanup
checkpoint after task 0018S, as specified. No validation was skipped beyond that planned
checkpoint deferral.

## Implementation notes

- The migration is deliberately atomic and retains no alias or deprecated bridge. Historical
  completed task documents remain unchanged records of the provisional names.
- `EXP` and `TANH` remain portable mathematical requests. Their Javadocs and API reference select
  no algorithm, bitwise result, approximation bound, or backend route.
- Scalar-family Javadocs and behavior remain accurate unchanged: the exact five-kind vocabulary,
  typed scalar/range attributes, validation order, metadata, and provenance do not depend on the
  renamed unary kind.
- `OperationSignature`, `OperationKind`, `Operation`, `TensorProducer`, `TensorProvenance`, and
  `TensorFactory` remain accurate unchanged because the task preserves the same no-attributes,
  one-input, one-output construction mechanics.
- Training API remains accurate unchanged because no gradient, autograd, optimizer, or training
  behavior changed. Architecture/ADRs/tests remain unchanged because ownership, dependencies, and
  lifecycle boundaries did not change. Backend-conformance/integration tests remain unchanged
  because no lowering or execution behavior changed. Java 26 Gradle configuration, dependencies,
  and other modules remain unchanged because the cleanup is confined to model vocabulary and
  metadata construction.

## Completion summary

- Completed changes: atomically renamed unary reciprocal vocabulary to
  `RECIPROCAL`/`reciprocal`, removed both fast variants without aliases, retained thirteen exact
  unary kinds/methods, and preserved unary construction plus typed scalar contracts.
- Files changed or created: exactly the thirteen authorized production, test, API, glossary,
  capability, task, master-plan, and roadmap paths.
- Tests and validation: reused the passing 50-test focused set and 725-test/88-suite final model
  run; final model Javadoc, runnable reciprocal example, generated-page, removed-vocabulary,
  Markdown link/anchor, fence/newline/whitespace, exact-scope/status, and `git diff --check`
  validation passed.
- Documentation-agent review: clean-context review
  `/root/task_0018p_implementation/task_0018p_docs` completed.
- Documentation impact: Tensor and Compile APIs, glossary, capability baseline, task, master plan,
  and roadmap now describe the final current vocabulary and ownership boundaries.
- Javadoc review: affected unary kind, Tensor, and helper Javadocs are final; adjacent scalar and
  foundational Javadocs remain accurate unchanged for the reasons recorded above.
- Glossary impact: current inventories and Tensor status use thirteen unary requests and
  reciprocal terminology; removed route-oriented vocabulary is absent.
- Unresolved issues: None.
- Follow-up required: None. Task 0018Q remains the next Draft frontier; task 0018T retains the
  missing scalar/unary numeric additions.

Status: Complete
