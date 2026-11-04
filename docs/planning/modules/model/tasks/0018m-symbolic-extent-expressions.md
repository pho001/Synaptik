# Task 0018M: Symbolic [Extent](../../../../glossary.md#term-extent) Expressions

## Status

Complete

## Goal

Extend the [model](../../../../glossary.md#term-model)-owned [dimension](../../../../glossary.md#dimension) value system so a [`Shape`](../../../../glossary.md#shape) can honestly retain a small,
[backend](../../../../glossary.md#backend)-independent symbolic [extent](../../../../glossary.md#term-extent) formula instead of replacing every derived dynamic size with
an unrelated name or rejecting it. The task provides [canonical](../../../../glossary.md#term-canonical-form) checked addition, signed [constant](../../../../glossary.md#term-constant)
offset, multiplication by a non-negative constant, floor/ceiling division by a positive constant,
and [identity](../../../../glossary.md#term-identity-meanings)-based constrained unknown extents.

This is a shape-value foundation task. It does not bind [runtime](../../../../glossary.md#term-runtime) sizes or change any [Tensor](../../../../glossary.md#tensor)
[operation](../../../../glossary.md#operation) to produce the new expressions.

## [Mental model](../../../../glossary.md#term-mental-model)

```text
Dimension
  StaticDimension(32)                 known value
  DynamicDimension("N")              named input variable
  ExpressionDimension(N + 2)         exact derived formula
  ExpressionDimension(unknown...)    distinct value with known bounds
```

For example, the [model](../../../../glossary.md#term-model) can represent `N + 2` during [Tensor](../../../../glossary.md#tensor) expression construction. A later
[compiler](../../../../glossary.md#term-compiler)/[lifecycle](../../../../glossary.md#lifecycle) contract may bind `N = 32` and obtain `34`; that [binding](../../../../glossary.md#term-binding) is not part of this
task.

## [Scope](../../../../glossary.md#term-scope)

- Extend the [sealed](../../../../glossary.md#term-sealed) [`Dimension`](../../../../glossary.md#dimension) hierarchy with one public `ExpressionDimension` variant.
- Add one public sealed `DimensionExpression` inspection contract with exactly four expression
  forms:
  - [canonical](../../../../glossary.md#term-canonical-form) [linear](../../../../glossary.md#term-linear-meanings) combination;
  - floor division;
  - ceiling division; and
  - [identity](../../../../glossary.md#term-identity-meanings)-based constrained unknown.
- Add the public stateless `DimensionExpressions` construction boundary.
- Canonicalize static arithmetic, neutral operations, nested linear combinations, duplicate
  terms, and operand-order-independent sums.
- Represent a signed [constant](../../../../glossary.md#term-constant) offset without permitting a known negative dimension.
- Preserve checked `long` arithmetic and explicit non-negative-dimension semantics.
- Generalize `Dimension` and [`Shape`](../../../../glossary.md#shape) inspection/diagnostics for expression dimensions.
- Confirm that structurally equal independently constructed expressions participate in existing
  conservative broadcasting.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- Add focused tests and update the [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) reference and glossary in the implementation change.

## Out of [scope](../../../../glossary.md#term-scope)

- changing [`Tensor`](../../../../glossary.md#tensor), `TensorDescriptor`, [operation](../../../../glossary.md#operation) kinds, attributes, or expression helpers
- adopting symbolic extents in pad, tile, concat, unfold, fold, convolution, pooling, slice, or
  another Tensor operation<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [slice glossary](../../../../glossary.md#slice))</span>
- [runtime](../../../../glossary.md#term-runtime) size [binding](../../../../glossary.md#term-binding), a binding map, expression [evaluation](../../../../glossary.md#term-evaluation) against concrete inputs, or caching
- [graph](../../../../glossary.md#graph)-wide equality solving, symbolic simplification across graph values, or [compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference)
- [prepare](../../../../glossary.md#prepare) [specialization](../../../../glossary.md#term-specialization), runtime allocation, memory [planning](../../../../glossary.md#term-planning), [backend](../../../../glossary.md#backend) [lowering](../../../../glossary.md#lowering), or [kernels](../../../../glossary.md#kernel)
- arbitrary subtraction between two dimensions, negative coefficients, modulo, min/max, products
  of two symbolic dimensions, general rational arithmetic, or a computer-algebra system<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension))</span>
- mutable dimensions, negative dimension sentinels, reflection registries, service locators, or
  new dependencies<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>
- [Gradle](../../../../glossary.md#term-gradle), `ARCHITECTURE.md`, focused architecture documentation, or another [module](../../../../glossary.md#term-module)

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model ownership of the
  shape model and compiler ownership of graph-wide shape inference<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [inference glossary](../../../../glossary.md#term-inference))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Model capability baseline](../capabilities.md), especially dynamic [shape](../../../../glossary.md#shape) expressions
- [Model master plan](../master-plan.md)
- [Task 0002](0002-shape-and-dimension-model.md), the completed foundational shape contract

## Legacy evidence

The read-only `legacy/pre-rewrite` branch contains only positive concrete `int[]` [shapes](../../../../glossary.md#shape).
`TensorShape` rejects dynamic [dimensions](../../../../glossary.md#dimension), and the [ONNX](../../../../glossary.md#term-onnx) importer rejects symbolic input extents.
Legacy [convolution](../../../../glossary.md#term-channels-first-convolution-layers), [pooling](../../../../glossary.md#term-pooling), and window rules nevertheless confirm the required arithmetic forms:
checked [constant](../../../../glossary.md#term-constant) offset, multiplication, floor division, and ceiling division. They provide
formula/test evidence only; their static arrays, package structure, [eager](../../../../glossary.md#term-eager-evaluation) [validation](../../../../glossary.md#term-validation), and [runtime](../../../../glossary.md#term-runtime)
coupling are not copied.

## Architecture constraints

- All implementation remains in `modules/model` and uses only the [JDK](../../../../glossary.md#term-java-development-kit-jdk) and local [model](../../../../glossary.md#term-model) types.
- Symbolic extents are immutable backend-independent model values, not runtime variables or
  prepared state.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Local expression construction may validate and canonicalize model values. Graph traversal,
  cross-value solving, binding concrete input sizes, and compiler passes remain outside model.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- The task must not add physical layout, storage, device, kernel, backend, prepare, runtime, or
  execution facts to a dimension.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [device glossary](../../../../glossary.md#term-device); [kernel glossary](../../../../glossary.md#kernel); [prepare glossary](../../../../glossary.md#prepare))</span>
- [`Shape`](../../../../glossary.md#shape) remains an [immutable](../../../../glossary.md#term-immutable) ordered list of [`Dimension`](../../../../glossary.md#dimension) values. A dimension expression does
  not turn `Shape` or [`Tensor`](../../../../glossary.md#tensor) into [graph](../../../../glossary.md#graph) [IR](../../../../glossary.md#term-intermediate-representation-ir).
- If implementation requires a dependency or architecture change, stop and report it instead of
  modifying <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>`ARCHITECTURE.md`.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.model.shape`

No new package is added.

Type placement:

- `io.github.pho001.synaptik.model.shape.DimensionExpression` — typed read-only expression forms
  belong beside dimensions and shapes.
- `io.github.pho001.synaptik.model.shape.ExpressionDimension` — one derived or constrained extent
  remains a <span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>[`Dimension`](../../../../glossary.md#dimension) value.
- `io.github.pho001.synaptik.model.shape.DimensionExpressions` — domain-specific validated
  construction and canonicalization boundary; it is not a generic utility.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>

Tests remain in the mirrored [shape](../../../../glossary.md#shape) package so they can validate package-private constructors and
ensure callers cannot bypass the public [canonical](../../../../glossary.md#term-canonical-form) factory.

## Required contracts

### [Dimension](../../../../glossary.md#dimension) categories

`Dimension` remains [sealed](../../../../glossary.md#term-sealed) and permits exactly:

```text
StaticDimension
DynamicDimension
ExpressionDimension
```

Its existing methods retain these meanings:

- `isStatic()` is true only for `StaticDimension`.
- `isDynamic()` is true for every non-static dimension, including named, exact-expression, and
  constrained-unknown dimensions.
- `staticSize()` is present only for `StaticDimension`.
- `dynamicSymbol()` is present only for the named `DynamicDimension` leaf. It is empty for an
  expression or unknown because neither has one caller-defined symbol.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [symbol glossary](../../../../glossary.md#term-symbol))</span>

No new category query is added. Callers that need typed expression inspection use
`instanceof ExpressionDimension` and `expression()`.

### ExpressionDimension

`ExpressionDimension` is a public final non-record value with exactly one field:

```java
private final DimensionExpression expression;
```

It has one package-private constructor so every instance is created by `DimensionExpressions`.
It exposes exactly `expression()` in addition to ordinary `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and `toString`.

Equality and hashing delegate to the contained expression. Exact expressions therefore have
structural value semantics. An unknown expression deliberately uses object [identity](../../../../glossary.md#term-reference-identity), so two
separate unknown-construction calls remain distinct even when their bounds are equal.

### DimensionExpression forms

`DimensionExpression` is a public [sealed](../../../../glossary.md#term-sealed) interface permitting exactly four public nested final
classes. Their constructors are package-private; their state is read-only through public
accessors.

1. `DimensionExpression.LinearCombination`

   ```text
   coefficients: Map<Dimension, Long>
   offset: long
   value = sum(coefficient[dimension] * dimension) + offset
   ```

   The coefficient map is immutable, non-empty, has no null keys/values, and every coefficient is
   positive. Map order is not semantic. The signed offset permits exact formulas such as <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`N - 3`;
   every later concrete binding must still produce a non-negative value.

2. `DimensionExpression.FloorDivision`

   ```text
   dividend: Dimension
   divisor: positive long
   ```

3. `DimensionExpression.CeilingDivision`

   ```text
   dividend: Dimension
   divisor: positive long
   ```

4. `DimensionExpression.Unknown`

   ```text
   minimum: non-negative long
   maximum: Optional<Dimension>
   ```

   Bounds are inclusive. Empty maximum means no known upper bound. The exact optional maximum
   dimension reference is retained. Unknown deliberately keeps ordinary object identity and does
   not override <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension))</span>`equals` or [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests); it is not a named input [symbol](../../../../glossary.md#term-symbol) and receives no global ID.

Expression objects contain no [binding](../../../../glossary.md#term-binding), [runtime](../../../../glossary.md#term-runtime) value, [graph](../../../../glossary.md#graph) [identity](../../../../glossary.md#term-identity-meanings), [operation](../../../../glossary.md#operation), storage, or
[backend](../../../../glossary.md#backend) [metadata](../../../../glossary.md#term-metadata).

### DimensionExpressions [API](../../../../glossary.md#term-application-programming-interface-api)

`DimensionExpressions` is one public final field-free static construction class with one private
constructor and exactly these public methods:

```java
public static Dimension add(Dimension left, Dimension right)
public static Dimension addConstant(Dimension input, long offset)
public static Dimension multiply(Dimension input, long factor)
public static Dimension floorDivide(Dimension input, long divisor)
public static Dimension ceilingDivide(Dimension input, long divisor)
public static Dimension unknown(long minimum, Optional<Dimension> maximum)
```

No public raw-[node](../../../../glossary.md#node) factory, evaluator, [binding](../../../../glossary.md#term-binding) API, overload, varargs method, builder, [registry](../../../../glossary.md#term-registry),
or [singleton](../../../../glossary.md#term-singleton) is added.

[Argument](../../../../glossary.md#term-argument) [validation](../../../../glossary.md#term-validation) is deterministic:

- reference [parameters](../../../../glossary.md#term-parameter) are null-checked in declaration order with their parameter names;
- negative factor fails with `factor must be non-negative: <factor>`;
- non-positive floor/ceiling divisor fails with `divisor must be positive: <divisor>`;
- negative unknown minimum fails with `minimum must be non-negative: <minimum>`;
- a static maximum smaller than minimum fails with
  `maximum static size must be greater than or equal to minimum: minimum=<minimum>, maximum=<size>`;
- checked `long` [overflow](../../../../glossary.md#term-overflow) propagates as `ArithmeticException`; and
- a fully static negative result fails with
  `dimension expression must not produce a negative static size: <value>`.

### [Canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form)

Construction returns the simplest truthful [`Dimension`](../../../../glossary.md#dimension):

- all-static operations return `StaticDimension`;
- `x + 0`, `x * 1`, `floorDivide(x, 1)`, and `ceilingDivide(x, 1)` return the exact `x` reference;
- `x * 0` returns static zero;
- nested [linear](../../../../glossary.md#term-linear-meanings) combinations are flattened;
- static terms are folded into the signed offset with checked arithmetic;
- equal terms are combined with checked positive coefficients, so `N + N` equals `N * 2`;
- coefficient maps make `N + M` equal to `M + N` independently of construction order;
- a linear combination with one coefficient-one term and offset zero unwraps to that exact term;
- a linear combination with no symbolic terms becomes a static dimension and is rejected if its
  checked value is negative; and
- floor and ceiling division of a static value use non-negative integer arithmetic without an
  overflow-prone <span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>`value + divisor - 1` formula.

Do not add speculative rewriting across division [nodes](../../../../glossary.md#node). In particular, no rule may reassociate a
division with addition or multiplication unless this task explicitly requires it.

### [Shape](../../../../glossary.md#shape) and [broadcasting](../../../../glossary.md#broadcasting) integration

`Shape` must treat `ExpressionDimension` as non-static:

- `isFullyStatic()` is false;
- `knownElementCount()` is empty;
- `toLongArray()` rejects the shape through its existing dynamic-shape failure;
- equality and hashing remain ordered structural [dimension](../../../../glossary.md#dimension) equality; and
- diagnostic text renders named symbols, exact expressions, and unknown bounds readably without
  becoming a serialization format.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>

`ShapeBroadcast` keeps its conservative rules. Equal independently built [canonical](../../../../glossary.md#term-canonical-form) expressions
broadcast as equal dimensions, and a static [singleton](../../../../glossary.md#term-singleton) broadcasts to an expression dimension.
Unequal expressions and an expression paired with a non-singleton static size remain unprovable
and are rejected.

## Affected files

Expected production and test files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/Dimension.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DynamicDimension.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DimensionExpression.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/ExpressionDimension.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/DimensionExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/Shape.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/shape/ShapeBroadcast.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/shape/DimensionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/shape/DimensionExpressionsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/shape/ShapeTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/shape/ShapeBroadcastTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task file
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

`StaticDimension.java` may receive a [Javadoc](../../../../glossary.md#term-javadoc)-only clarification if the documentation pass finds
that its current description incorrectly implies it is the only non-symbolic [leaf](../../../../glossary.md#term-leaf). No declaration
or behavior change is authorized.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 18 paths: the 17 expected paths above plus the optional
[Javadoc](../../../../glossary.md#term-javadoc)-only `StaticDimension.java` clarification.

If implementation needs a [Tensor](../../../../glossary.md#tensor) expression helper, [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime) [binding](../../../../glossary.md#term-binding) contract, another
public type, another [module](../../../../glossary.md#term-module), or more paths, stop and propose a follow-up task.

## Acceptance criteria

- The public type and method surface matches the required contracts exactly.
- Every expression instance can be obtained only through `DimensionExpressions`; raw expression
  constructors and the `ExpressionDimension` constructor are inaccessible outside the package.
- [Linear](../../../../glossary.md#term-linear-meanings)-combination equality is independent of addition order and combines repeated terms.
- Static folding, neutral identities, checked coefficient/offset arithmetic, signed offset, floor
  division, ceiling division, and their documented failure conditions are tested.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- Separately generated unknowns with equal bounds remain unequal, while repeated use of the same
  unknown reference remains equal.
- Existing static and named dynamic behavior remains compatible except for the intentional
  expansion of `isDynamic()` to expression [dimensions](../../../../glossary.md#dimension).
- [`Shape`](../../../../glossary.md#shape) and broadcasting tests cover exact expressions, unknowns, static singletons, unequal
  expressions, known counts, static extraction, equality, and diagnostics.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- No binding/evaluation API, Tensor-operation adoption, dependency, architecture, or build change
  is present.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [evaluation glossary](../../../../glossary.md#term-evaluation); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [operation glossary](../../../../glossary.md#operation); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Complete Javadoc documents every affected public type, constructor where accessible, method,
  parameter, result, equality rule, checked failure, and deferred lifecycle boundary.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [parameter glossary](../../../../glossary.md#term-parameter); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A separate clean-context documentation pass finalizes Javadoc, Tensor API, glossary, planning
  status/evidence, links, examples, and terminology in the same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During development, [run](../../../../glossary.md#term-run) focused [shape](../../../../glossary.md#shape) tests as needed. After executable Java stabilizes, record
one final [module](../../../../glossary.md#term-module) run:

```bash
./gradlew :modules:model:test
```

The documentation-focused pass reuses that successful test evidence unless it changes executable
Java, then runs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It must also validate changed Markdown links, anchors, fences, trailing whitespace, final
newlines, conceptual examples, synchronized task/master/roadmap status, exact package placement,
the 18-path maximum, and absence of a detailed task-0018M1 specification.

Repository-wide tests are deferred to the foundation-contract [checkpoint](../../../../glossary.md#term-checkpoint) after task 0018N and to
[CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one module, no [dependency](../../../../glossary.md#term-dependency), and no architecture boundary.

## Dependencies

- [Task 0002](0002-shape-and-dimension-model.md) — completed [dimension](../../../../glossary.md#dimension), [shape](../../../../glossary.md#shape), and broadcast base.
- Completed tasks 0017C–0017N — capability evidence for derived shape requirements; their Tensor
  helpers are not modified here.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>

## Follow-up tasks

- Task 0018M1, Dynamic extent adoption in pad, tile, and concat — Draft in the master plan only;
  no detailed specification is created by this task.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- Task 0018R owns later [slice](../../../../glossary.md#slice)/window public-contract cleanup and may adopt symbolic window formulas.
- Compiler/prepare/runtime shape binding requires later layer-owned tasks and is not implied by
  model completion.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [shape glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding); [model glossary](../../../../glossary.md#term-model))</span>
- Task 0018N remains the next existing foundation task after the 0018M1 adoption row.

## Architecture impact

Expected impact: None.

The architecture already assigns [shape](../../../../glossary.md#shape) values to [model](../../../../glossary.md#term-model) and [graph](../../../../glossary.md#graph)-wide shape [inference](../../../../glossary.md#term-inference) to
[compiler](../../../../glossary.md#term-compiler). This task implements only the model-owned value representation. If implementation
requires [runtime](../../../../glossary.md#term-runtime) [binding](../../../../glossary.md#term-binding) or changes [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings), stop and report the architecture issue.

## Implementation prompt

Use this prompt in a separate clean-context agentic task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, model capabilities/master plan,
roadmap, tasks 0002 and 0018M, current shape production/tests, Tensor API, glossary, and Java 26
Gradle configuration.

Implement task 0018M exactly. Stay inside modules/model shape production/tests and the explicitly
allowed documentation/planning files. Add the exact expression hierarchy and canonical factory,
preserve existing static/named-symbol behavior, and do not adopt expressions in Tensor operations
or add binding/compiler/prepare/runtime/backend behavior. Stop beyond 18 paths or on architecture
uncertainty.

Run the final model test once after executable code stabilizes. Then hand the actual diff and test
evidence to a separate clean-context documentation agent in the same change. That agent must
inspect final source/tests, finalize affected Javadocs, Tensor API, glossary, task/master/roadmap
status and documentation validation, and must not repeat successful Java tests unless executable
behavior changes or it records a concrete reason.

Do not mark 0018M Complete until both passes succeed. Leave 0018M1 Draft without a detailed
specification. Do not commit or push.
```

## Local decisions

- `ExpressionDimension` is the only added [`Dimension`](../../../../glossary.md#dimension) leaf and delegates equality, hashing, and
  diagnostics to one retained expression. This gives exact formulas structural equality while an<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
  `Unknown` retains ordinary object [identity](../../../../glossary.md#term-reference-identity).
- [Canonical](../../../../glossary.md#term-canonical-form) [linear](../../../../glossary.md#term-linear-meanings) arithmetic uses a `Map<Dimension, Long>` with positive coefficients and one
  signed offset. Public construction flattens only linear combinations; division nodes remain
  structural and are not reassociated.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node))</span>
- Neutral identities are reference-preserving. In particular, either operand position for adding
  static zero returns the exact opposing reference.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- [`Shape`](../../../../glossary.md#shape) continues to ask only whether a Dimension is static. Expression dimensions therefore
  reuse all existing dynamic-count and static-extraction behavior without a new category query.
- Diagnostic text is readable and deterministic enough for tests and error messages but remains
  explicitly outside any serialization contract.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>

## Known limitations

- The [model](../../../../glossary.md#term-model) stores formulas but cannot bind or evaluate them against concrete [runtime](../../../../glossary.md#term-runtime) inputs.
- Signed offsets may imply a binding-time non-negative-result constraint; this task records the
  formula but does not add a solver.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Unknown constraints are limited to a non-negative static minimum and an optional inclusive
  dimension upper bound.
- Products of two symbolic [dimensions](../../../../glossary.md#dimension) and general algebraic equivalence are deliberately absent.
- Existing Tensor operations continue using their current conservative dynamic-shape behavior
  until focused follow-up tasks adopt this foundation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [shape glossary](../../../../glossary.md#shape))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation and test evidence on 2026-07-09:

- Implementation context `/root/task_0018m_implementation` ran
  `./gradlew :modules:model:test --tests 'io.github.pho001.synaptik.model.shape.*'` during
  development. The first run failed 1 of 40 tests because adding static zero reconstructed a
  linear combination instead of preserving the opposing reference. After the identity fix, the
  repeated focused run passed all 40 tests.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [linear glossary](../../../../glossary.md#term-linear-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- The same implementation context ran final `./gradlew :modules:model:test` after executable Java
  stabilized: `BUILD SUCCESSFUL in 1s`; the generated report contains 765 tests with 0 failures,
  0 errors, and 0 skipped tests. Shape coverage is <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>`DimensionExpressionsTest` 10,
  `DimensionTest` 7, `ShapeTest` 11, and `ShapeBroadcastTest` 12.
- Clean documentation-focused context
  `/root/task_0018m_implementation/task_0018m_documentation` independently reviewed the required
  architecture contract, documentation rules and General/API-Javadoc/Planning/Example profiles,
  planning guide, capability baseline, model master plan and roadmap, tasks 0002 and 0018M, final
  shape production/tests and actual diff, Tensor API, glossary, and Java 26 build configuration.
  It changed only Javadoc and Markdown after the successful final model test, so it reused the
  implementation evidence and did not repeat a Java test suite.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor))</span>
- The documentation context ran `./gradlew :modules:model:javadoc` after final [Javadoc](../../../../glossary.md#term-javadoc) edits:
  `BUILD SUCCESSFUL in 1s`; 2 actionable tasks executed and the configuration [cache](../../../../glossary.md#term-cache) was reused.
- The documented `SymbolicExtentExample` compiled with
  `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-symbolic-extent-doc-example /tmp/SymbolicExtentExample.java` and ran with
  `java -cp modules/model/build/classes/java/main:/tmp/synaptik-symbolic-extent-doc-example
  SymbolicExtentExample`. It printed [structural equality](../../../../glossary.md#term-structural-equality) `true`, offset `2`, the documented
  formula/unknown Shape diagnostic, and separate-unknown equality `false` exactly as documented.
- `javap -classpath modules/model/build/classes/java/main -p ...` confirmed package placement,
  the sealed <span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed))</span>[`Dimension`](../../../../glossary.md#dimension) and `DimensionExpression` surfaces, the four final nested expression
  forms with package-private constructors, the one-field final `ExpressionDimension`, and the
  field-free `DimensionExpressions` with one private constructor and exactly six public methods.
- The targeted local Markdown validator checked Tensor API, glossary, this task, capabilities,
  model master plan, and roadmap. It resolved 438 local links, including 139 heading anchors, with
  zero errors and confirmed balanced fences and final newlines in all six files.
- Final trailing-whitespace, terminology, current-versus-planned, source/import, exact scope,
  status, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` inspections passed. The change contains exactly 17 authorized
  paths, remains in `io.github.pho001.synaptik.model.shape` plus the six permitted documentation
  files, and creates no detailed 0018M1 task specification.
- Root `build.gradle.kts` continues to set both Java toolchain language version and compiler
  release to 26. No Gradle file, dependency, or build option changed.<span data-glossary-navigation="true"> ([toolchain glossary](../../../../glossary.md#term-toolchain); [compiler glossary](../../../../glossary.md#term-compiler); [release glossary](../../../../glossary.md#term-release); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Tensor API and glossary now document symbolic extent construction, equality, diagnostics,
  conservative broadcasting, checked failures, and the absence of binding/evaluation. The
  capability baseline now records the implemented foundation. Compile API remains accurate
  unchanged because no Tensor operation, graph capture, inference, or compiler contract changed.
  Training API remains accurate unchanged because no gradient or autograd behavior changed.
  Related Tensor, operation, layout, graph, storage, and lifecycle contracts remain accurate
  unchanged because expression dimensions are model-owned Shape values only.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [broadcasting glossary](../../../../glossary.md#broadcasting); [binding glossary](../../../../glossary.md#term-binding); [evaluation glossary](../../../../glossary.md#term-evaluation); [Compile glossary](../../../../glossary.md#compile); [operation glossary](../../../../glossary.md#operation); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [layout glossary](../../../../glossary.md#layout); [graph glossary](../../../../glossary.md#graph); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `ARCHITECTURE.md`, focused architecture documents, ADRs, and architecture tests remain accurate
  unchanged because no module ownership, dependency direction, or architecture rule changed.
  Backend-conformance and integration tests remain unchanged because no backend behavior or
  end-to-end execution was added. Prepare, runtime, compiler, backend, training, and every other
  module remain untouched.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Implementation notes

- Added the sealed expression inspection hierarchy, canonical public construction boundary, and
  expression-backed Dimension leaf in the existing shape package.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [canonical glossary](../../../../glossary.md#term-canonical-form); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- Extended Dimension inspection, Shape diagnostics/count behavior, and conservative broadcasting
  to exact expressions and same-reference unknowns.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- Added focused structural, canonicalization, checked-failure, identity, Shape, broadcasting, and
  public-surface tests.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finalized affected public and package-contract Javadocs, including [`Dimension`](../../../../glossary.md#dimension),
  `DynamicDimension`, `ExpressionDimension`, every expression form/accessor, construction methods,
  [`Shape`](../../../../glossary.md#shape), and `ShapeBroadcast`. `StaticDimension` was reviewed and remains accurate unchanged: it
  describes only the known non-negative leaf and does not imply that it is the sole non-symbolic
  form.
- Updated [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), glossary, capability baseline, [model](../../../../glossary.md#term-model) master plan, roadmap, and this task.

## Completion summary

- Completed changes: Implemented canonical symbolic extent values, expression inspection,
  constrained unknowns, dynamic Shape integration, and conservative expression broadcasting.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [extent glossary](../../../../glossary.md#term-extent); [Shape glossary](../../../../glossary.md#shape); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- Files changed or created: Seventeen authorized production, test, API, glossary, capability, and
  planning paths; no optional <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>`StaticDimension.java` change was needed.
- Tests and validation: Reused the final 765-test model result after the documented 40-test
  development cycle; model Javadoc, runnable example, public-surface inspection, 438-link/
  139-anchor Markdown validation, fence/newline/whitespace checks, scope/status checks, and final
  diff checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: Clean context
  `/root/task_0018m_implementation/task_0018m_documentation` completed the independent targeted
  review and changed no executable Java behavior.
- Documentation impact: Tensor API, glossary, capability baseline, model master plan, roadmap,
  and this task now distinguish current symbolic extent values from deferred Tensor adoption and
  lifecycle binding/evaluation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [lifecycle glossary](../../../../glossary.md#lifecycle); [binding glossary](../../../../glossary.md#term-binding); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: All affected contracts are complete; `StaticDimension` remains accurate without
  modification.
- Glossary impact: Expanded Dimension and broadcasting definitions and added the reusable
  symbolic extent expression term.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [symbolic extent expression glossary](../../../../glossary.md#symbolic-extent-expression))</span>
- Unresolved issues: None.
- Follow-up required: None. Task 0018M1 remains Draft without a detailed specification.

Status: Complete
