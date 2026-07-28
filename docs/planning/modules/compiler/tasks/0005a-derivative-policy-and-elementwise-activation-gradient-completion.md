# Task 0005A: Derivative Policy and Elementwise/Activation Gradient Completion

## Status

Complete

## Goal

Complete first-order automatic differentiation for the current elementwise and activation
inventory inside the established compiler-owned, pre-capture Tensor-expression pipeline.

This task preserves every implemented Compiler 0004–0004B rule and adds the missing floating
binary/scalar arithmetic, extrema, clamp, unary, and activation rules. It also closes the
differentiability classification for conditional selection, casts, comparisons, Boolean logic,
and floating classifications:

```text
selected forward occurrence + accumulated output cotangent
  -> exact local Tensor-expression contribution for each differentiable input
  -> reverse broadcasting and floating-type normalization
  -> deterministic request-local accumulation
  -> one immutable combined forward/backward capture
```

Model 0025A fixes forward represented-value comparison, extrema, and clamp meaning. This task
selects compiler-owned first-order derivative conventions; it does not change those forward
contracts or add model-owned derivative behavior.

## Scope

### Exact operation and role inventory

The implementation must derive and lock this inventory from the production enums and signatures,
not from this table alone. It covers exactly 48 current kind constants:

| Family | Exact kinds | Differentiable input roles in 0005A |
|---|---|---|
| `BinaryArithmeticKind` | `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, `POW` | Floating left and right Tensor roles. Existing ADD/SUB/MUL/DIV rules remain; MIN/MAX/POW are added. |
| `ScalarElementwiseKind` | `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, `POW`, `CLAMP` | The floating Tensor receiver only. Scalar attributes and clamp bounds are immutable non-differentiable configuration. Existing ADD/SUB/MUL/DIV rules remain; MIN/MAX/POW/CLAMP are added. |
| `WhereSelectionKind` | `WHERE` | Floating true and false branches. The BOOL condition is non-differentiable. Preserve the existing rule. |
| `CastKind` | `CAST` | Floating source only when the target is also floating. Preserve the existing same-type identity and cross-floating cotangent cast-back policy. All roles involving a BOOL or integral source or target are non-differentiable. |
| `UnaryElementwiseKind` | `ABS`, `NEG`, `RECIPROCAL`, `LOG`, `LOG1P`, `EXP`, `EXPM1`, `ERF`, `SQRT`, `RSQRT`, `FLOOR`, `CEIL`, `SIGN`, `RELU`, `SIGMOID`, `TANH`, `GELU`, `GELU_TANH_APPROXIMATION`, `SILU` | The one floating input. Preserve NEG/EXP/EXPM1/ERF/FLOOR/CEIL/SIGN/SIGMOID/TANH and add the other ten rules. |
| `BinaryComparisonKind` | `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, `NOT_EQUAL` | None. BOOL outputs and both numeric input roles are non-differentiable through the comparison occurrence. |
| `BooleanLogicalKind` | `AND`, `OR`, `NOT` | None. BOOL inputs and outputs are non-differentiable. |
| `FloatingClassificationKind` | `IS_FINITE`, `IS_NAN`, `IS_INF` | None. BOOL outputs and the classified floating input role are non-differentiable through the classification occurrence. |

Every assigned occurrence has exactly one canonical output at slot zero. Binary/scalar arithmetic,
WHERE, floating-to-floating CAST, and unary occurrences have one floating output on a selected
route. Comparisons, Boolean logic, floating classifications, and casts to BOOL or integral types
have one non-differentiable output. No 0005A kind has a hidden or auxiliary output slot.

Comparison and classification expressions may appear as BOOL conditions inside generated
piecewise formulas. That use does not make their numeric inputs differentiable through those BOOL
operations.

### Notation and cotangent normalization

For one output `y`, incoming cotangent `g`, and selected input `x`:

- `Z_x`, `O_x`, `N_x`, and `H_x` are exact typed positive-zero, positive-one, negative-one, and
  one-half splats expanded to `x.shape`.
- `S_x(v)` is an exact typed scalar-value splat for immutable `ScalarValue v`, expanded to
  `x.shape`.
- `B_x(t)` restores `x.shape` by applying ordinary `sumToShape(x.shape)` when `t.shape` differs,
  then restores `x.type` by applying ordinary floating `cast(x.type)` when its type differs.
- A direct local result whose Shape and type already equal `x` requires no redundant
  normalization expression.

Every binary and broadcast branch contribution must end in `B_x`. Every returned input
contribution must have exactly the selected input Tensor's Shape and floating data type before
ordinary deterministic accumulation. Normalization uses the existing public Tensor algebra and
the current Compiler 0002 inference/validation boundary; it is not a gradient-only shape, cast,
or arithmetic system.

### Exact local formulas

The formulas below are structural Tensor-expression contracts. `where` means public
`Tensor.where`, comparisons and classifications mean the current public represented-value
operations, and scalar coefficients mean exact typed `ScalarValue` operation attributes unless a
splat is explicitly named.

#### Binary arithmetic

Preserve the existing rules:

| Kind | Left contribution | Right contribution |
|---|---|---|
| `ADD` | `B_left(g)` | `B_right(g)` |
| `SUB` | `B_left(g)` | `B_right(-g)` |
| `MUL` | `B_left(g * right)` | `B_right(g * left)` |
| `DIV` | `B_left(g / right)` | `B_right(-(g * left) / (right * right))` |

Add:

| Kind | Left contribution | Right contribution |
|---|---|---|
| `MIN` | `B_left(where(left < right, g, where(left == right, g * 0.5, Z_y)))` | `B_right(where(right < left, g, where(right == left, g * 0.5, Z_y)))` |
| `MAX` | `B_left(where(left > right, g, where(left == right, g * 0.5, Z_y)))` | `B_right(where(right > left, g, where(right == left, g * 0.5, Z_y)))` |
| `POW` | `B_left(g * right * pow(left, right - O_right))` | `B_right(g * y * log(left))` |

The Tensor/Tensor `POW` formula subtracts an ordinary exact typed one from the exponent Tensor.
It does not replace `pow(left, right - 1)` with `y / left`, because that changes the selected
zero-base boundary behavior.

#### Scalar arithmetic and first-class clamp

Preserve scalar `ADD -> g`, `SUB -> g`, `MUL -> g * value`, and `DIV -> g / value`.

For scalar MIN/MAX, let `b = S_x(attrs.value)`:

| Kind | Receiver contribution |
|---|---|
| `MIN` | `where(x < b, g, where(x == b, g * 0.5, Z_x))` |
| `MAX` | `where(x > b, g, where(x == b, g * 0.5, Z_x))` |
| `POW` | `g * exponent * pow(x, exponentMinusOne)` |

`exponentMinusOne` is one exact same-typed `ScalarValue` produced at compile time by subtracting
positive one in the represented type:

- BFLOAT16 expands the exact bits to binary32, subtracts binary32 one, and converts back with the
  existing round-to-nearest-ties-to-even `BFloat16Bits.fromFloat`;
- FLOAT32 performs one binary32 subtraction; and
- FLOAT64 performs one binary64 subtraction.

No host transcendental function is used, and the original exponent remains the exact multiplier
attribute.

For first-class `CLAMP`, preserve its ordered forward meaning instead of inventing an unrelated
endpoint rule:

```text
lowered = x.maximum(minValue)
upperRouted =
    where(lowered < S_x(maxValue), g,
          where(lowered == S_x(maxValue), g * 0.5, Z_x))
inputContribution =
    where(x > S_x(minValue), upperRouted,
          where(x == S_x(minValue), upperRouted * 0.5, Z_x))
```

This is the chain rule for the selected scalar-extrema tie convention applied to the exact ordered
`MIN(MAX(x, minValue), maxValue)` value contract. It is expressed with ordinary public Tensor
operations; it does not add stored forward intermediates or decompose the original forward
occurrence.

#### Unary arithmetic and activations

Preserve:

| Kind | Input contribution |
|---|---|
| `NEG` | `-g` |
| `EXP` | `g * y` |
| `EXPM1` | `g * (y + 1)` |
| `ERF` | `g * exp(-(x * x)) * (2 / sqrt(pi))` with the existing fixed typed coefficient |
| `FLOOR`, `CEIL`, `SIGN` | `Z_x` |
| `SIGMOID` | `g * y * (1 - y)` |
| `TANH` | `g * (1 - y * y)` |

Add:

| Kind | Input contribution |
|---|---|
| `ABS` | `where(x > Z_x, g, where(x < Z_x, -g, Z_x))` |
| `RECIPROCAL` | `-g / (x * x)` |
| `LOG` | `g / x` |
| `LOG1P` | `g / (x + 1)` |
| `SQRT` | `g / (2 * y)` |
| `RSQRT` | `-0.5 * g * y * y * y` |
| `RELU` | `where(x > Z_x, g, Z_x)` |

For exact GELU, define:

```text
phiCdf = 0.5 * (1 + erf(x * invSqrt2))
phiPdfTerm = x * exp(-0.5 * x * x) * invSqrt2Pi
regular = g * (phiCdf + phiPdfTerm)
result = where(isInf(x), where(x > Z_x, g, Z_x), regular)
```

For fixed tanh-approximation GELU, define:

```text
u = sqrt2OverPi * (x + 0.044715 * x * x * x)
t = tanh(u)
uPrime = sqrt2OverPi * (1 + 0.134145 * x * x)
regular = g * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * uPrime)
result = where(isInf(x), where(x > Z_x, g, Z_x), regular)
```

For SiLU, define:

```text
s = sigmoid(x)
regular = g * s * (1 + x * (1 - s))
result = where(isInf(x), where(x > Z_x, g, Z_x), regular)
```

The infinity selections give exact continuous-extension contributions: `g` at positive infinity
and exact positive zero at negative infinity. They avoid an otherwise spurious `infinity * zero`
NaN. Finite and NaN inputs use the regular formulas.

### Fixed typed coefficient representations

Compilation must not calculate transcendental coefficients from host `Math` functions. Use these
fixed correctly rounded represented values:

| Coefficient | BFLOAT16 bits | FLOAT32 bits | FLOAT64 bits |
|---|---:|---:|---:|
| `0.5` | `0x3F00` | `0x3F000000` | `0x3FE0000000000000` |
| `-0.5` | `0xBF00` | `0xBF000000` | `0xBFE0000000000000` |
| `2` | `0x4000` | `0x40000000` | `0x4000000000000000` |
| `invSqrt2` | `0x3F35` | `0x3F3504F3` | `0x3FE6A09E667F3BCD` |
| `invSqrt2Pi` | `0x3ECC` | `0x3ECC422A` | `0x3FD9884533D43651` |
| `sqrt2OverPi` | `0x3F4C` | `0x3F4C422A` | `0x3FE9884533D43651` |
| `0.044715` | `0x3D37` | `0x3D372713` | `0x3FA6E4E26D4801F7` |
| `0.134145` | `0x3E09` | `0x3E095D4F` | `0x3FC12BA9D1F60179` |

Positive zero, positive one, negative one, and the existing ERF coefficient retain their current
exact typed representations. Scalar coefficients used only by arithmetic remain operation
metadata. Values needed as Tensor comparison operands are explicit logical splat leaves.

### Tie, endpoint, discontinuity, domain, NaN, and infinity policy

The complete 0005A policy is:

- Tensor/Tensor MIN/MAX split an exact represented-numeric tie equally between the two Tensor
  roles. Equal finite values, equal same-sign infinities, and either ordering of opposite signed
  zeros are ties because current numeric equality treats the zero signs as equal.
- Scalar MIN/MAX assign one half to the Tensor receiver at an exact numeric tie. The scalar
  attribute remains non-differentiable configuration.
- First-class CLAMP follows the ordered composition above. With distinct ordered finite bounds,
  the receiver contribution is `g / 2` at either endpoint, `g` strictly inside, and exact zero
  outside. When both extrema stages tie, including equal same-representation bounds and the
  applicable signed-zero bound cases, the contribution is `g / 4`.
- Any NaN operand makes ordered and equality comparisons false. Piecewise MIN/MAX, scalar
  extrema, CLAMP, ABS, and RELU therefore return an exact positive-zero contribution at unordered
  NaN positions, independent of the incoming cotangent. No NaN payload is selected.
- ABS returns `g` for positive finite values and positive infinity, `-g` for negative finite
  values and negative infinity, and exact positive zero for either signed zero and NaN.
- RELU returns `g` only for values strictly greater than positive zero. It returns exact positive
  zero for negative values, both signed zeros, negative infinity, and NaN; positive infinity
  returns `g`.
- FLOOR, CEIL, and SIGN preserve Compiler 0004B's direct exact-positive-zero contribution for
  every accepted represented value, including integers, discontinuities, both signed zeros,
  infinities, and NaN.
- GELU, tanh-approximation GELU, and SiLU use their regular formulas for finite and NaN inputs,
  exact positive-zero contribution at negative infinity, and `g` at positive infinity.
- ADD, SUB, MUL, DIV, scalar ADD/SUB/MUL/DIV, floating CAST, NEG, RECIPROCAL, LOG, LOG1P, EXP,
  EXPM1, ERF, SQRT, RSQRT, SIGMOID, TANH, and POW use the exact formulas above without
  compiler-inserted finite/domain masks or continuous-extension repair. Zero denominators,
  negative logarithm domains, zero/negative power bases, overflow products, signed zeros,
  infinities, and NaNs flow through the ordinary shared Tensor operations in formula order.
  This raw-formula rule is the selected derivative convention even where the real derivative is
  undefined or a different limiting value exists.
- WHERE routes the incoming cotangent with selection, not multiplication by a zero mask, so an
  unselected branch receives exact positive zero even if its value or the incoming cotangent is
  exceptional.
- Floating CAST returns the incoming cotangent unchanged for equal source/target types and uses
  one ordinary cast to the source floating type otherwise. No gradient-specific rounding or NaN
  payload policy is added.
- Comparison, Boolean logical, and floating-classification occurrences stop differentiation.
  Their BOOL outputs never receive cotangents and their numeric inputs receive no contribution
  through those occurrences.

No backend value evaluator exists in this task. Focused tests lock formula structure, constants,
role routing, generated provenance, and fail-closed selection; future backend conformance owns
numeric execution evidence.

### Request-local constants

Extend the existing request-local derivative-constant owner into one exact typed-splat cache:

- cache by `ScalarValue` exact data type and bits;
- create at most one provenance-free, storage-free, unlabeled, non-gradient scalar Tensor leaf for
  each exact value used by one expansion;
- register every created leaf explicitly as one `CompileTimeConstantGraph.Splat`;
- return Shape-specific values only through ordinary public `expand`;
- preserve deterministic first-use binding order;
- make existing zero/one helpers delegate to the same cache; and
- include only bindings reachable from returned gradient roots in combined-capture ingress.

This cache may hold zero, one, negative one, one half, or an exact scalar MIN/MAX/CLAMP bound. It
must not infer constants from Tensor storage, labels, descriptors, provenance absence, factory
history, or operation attributes without explicit registration. It is ephemeral compiler
bookkeeping, not a public constant API or a second graph.

### Preflight, validation, and failure order

Preserve the existing top-level order:

1. `GraphCompiler` validates compile arguments and mode/request agreement.
2. `AutogradPreflight` validates the scalar floating gradient-eligible objective and exact forward
   output membership.
3. It inventories all forward producers and validates explicit ingress membership.
4. It inventories objective ancestry and validates ordered targets, identity uniqueness,
   membership, floating type, gradient eligibility, and selected reachability.
5. It visits selected producer occurrences in deterministic producer postorder and validates the
   complete selected route before seed, derivative constant, bound splat, coefficient, or formula
   Tensor construction.

For each selected 0005A occurrence, preserve this order:

1. exact one-output and selected canonical output position;
2. floating output type and selected-output gradient eligibility;
3. selected-input gradient eligibility in input-position order;
4. exact typed kind/attributes pairing;
5. exact signature arity and role differentiability in input-position order;
6. floating input promotion or same-type requirements;
7. exact output Shape/type consistency and availability of `sumToShape`/cast normalization; and
8. the fixed local policy/formula row selected by kind and attributes.

Do not inspect Tensor values or reject NaN, infinity, signed-zero, zero-denominator, power-domain,
or clamp-bound cases accepted by the forward model. The formulas above cover those represented
values. A selected comparison condition, logical role, classification role, non-floating cast
role, unknown operation, wrong attributes variant, unsupported output slot, or deferred family
must fail closed before derivative Tensor allocation.

A known preflight failure consumes no seed, derivative constant, bound splat, coefficient, formula
Tensor, or derivative `TensorId`. After successful preflight, public Tensor construction, exact
scalar derivation, capture, inference, validation, optimization, publication, or planning may
fail after consuming opaque Tensor IDs. IDs are never rolled back or reused.

### Accumulation, producers, provenance, and combined graph

- Preserve exact Tensor-object and `TensorProducer` identity for request-local ancestry, selected
  occurrence, contribution, and accumulation maps.
- Append contributions in reverse producer-postorder and input-position order. Repeated operand
  positions remain repeated contributions.
- Normalize every contribution to its input before accumulation, then use left-associated
  ordinary `Tensor.add` in deterministic append order.
- Scalar attributes, bounds, and generated logical-splat leaves are not differentiation targets
  or original forward producers.
- Generated formula producers and exact-splat expansion producers are BACKWARD work. Exact
  original producer identities remain FORWARD work.
- Preserve canonical exact producer output wrappers. Do not reconstruct a Tensor from a captured
  `ValueId` or manufacture a sibling output.
- Capture ordered forward outputs and gradient roots together exactly once. Assign each `NodeId`
  and `ValueId` once, retain per-node `GraphPhase`, and keep target roles independent when two
  targets share one gradient value.
- Run the existing shared inference, validation, canonicalization, exact arithmetic rewriting,
  constant folding, whole-graph liveness, phase-local common-subexpression elimination, and final
  validation pipeline. Add no gradient-specific rewrite, fold, simplification, or pass.
- Return the existing immutable `GraphCompilation` and `CompileArtifacts` boundaries unchanged.

## Out of scope

- reductions, products, reduction extrema, scans, softmax/log-softmax, statistics, norms,
  layer/RMS/batch normalization, or any Compiler 0005B work
- layout, reshape/expand/permute, slice/update/crop, composition, windows, Gather/scatter,
  ordering/top-K, dropout, RNG state, or any Compiler 0005C work
- attention, convolution, pooling, losses, structured ML formulas, or any Compiler 0005D work
- the source-backed all-operation/output/role closure audit, transitive formula-operation closure,
  capability checkpoint, or any Compiler 0005E work
- public objectives, targets, seeds, disconnected-result behavior, create-graph, derivative order,
  higher-order execution, order-aware phases, or Compiler 0006
- a new model operation kind, backward-only kind, Tensor method, attributes type, comparison
  semantics, extrema/clamp forward policy, cast conversion contract, or Tensor gradient state
- a public compiler gradient registry, facade, derivative-policy object, selectable subgradient
  mode, generic algebra builder, direct graph-node formula construction, placeholder Tensor,
  runtime tape, physical saved value, or backend-owned global autograd
- a gradient-specific arithmetic, cast, numerical, validation, constant-folding, simplification,
  optimization, or exceptional-value pipeline
- executable Tensor evaluation, backend lowering, kernel selection, preparation, runtime,
  storage, physical buffers, schedules, optimizer updates, publication delivery, or training APIs
- architecture, ADR, module-boundary, dependency, Gradle, config, planning, trace, model,
  runtime, prepare, engine, extension, backend, architecture-test, backend-conformance, or
  integration-test changes
- a detailed Compiler 0005B, 0005C, 0005D, 0005E, or 0006 specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Compiler 0004A](0004a-exact-composition-gradient-rule-extensions.md)
- [Compiler 0004B](0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md)
- [Compiler 0005](0005-publication-planning-orchestration-and-compile-artifacts.md)
- [Model adjoint expressibility audit](../../model/adjoint-expressibility-audit.md)
- [Model 0025A](../../model/tasks/0025a-portable-floating-comparison-extrema-and-clamp-semantics.md)
- [Compile API](../../../../api/compile-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Training API](../../../../api/training-api.md)
- [Autograd strategy](../../../../design/notes/autograd-strategy.md)
- [Autograd user guide](../../../../user-guide/autograd.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Compiler owns fail-closed preflight, reverse traversal, local derivative policy, formula
  construction, contribution accumulation, and combined graph construction.
- Model continues to own immutable forward Tensor and operation semantics. Model 0025A's forward
  comparison/extrema/clamp contract is consumed unchanged.
- Every formula uses existing public Tensor operations. No derivative rule moves into model and no
  second low-level algebra or direct generated graph-node representation is introduced.
- Tensor retains immutable identity, descriptor, and provenance plus its existing borrowed storage
  association; it gains no gradient field, backward method, tape, or derivative lifecycle.
- Exact Tensor identity maps and generated scalar caches are request-local ephemeral compiler
  bookkeeping and do not escape capture.
- The combined graph remains one immutable compile-time graph with per-node phase. Compiler output
  contains no physical, prepared, backend-specific, runtime, or executable state.
- Compiler dependencies remain unchanged and continue to exclude runtime, prepare, engine, and
  concrete backends.
- `ARCHITECTURE.md` remains unchanged. If implementation requires a model API, changed forward
  semantic, dependency, public gradient surface, or lifecycle change, stop and report the
  conflict.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — remains the single cohesive package-private compiler
  front-end/autograd boundary for preflight, formula construction, request-local constants,
  reverse accumulation, and focused same-package tests.

Packages added or changed:

- None.

Type placement:

- `io.github.pho001.synaptik.compiler.AutogradPreflight` — extends its closed selected-occurrence
  matrix and exact attrs/role/normalization validation.
- `io.github.pho001.synaptik.compiler.ElementwiseGradientRules` — owns all 0005A local formulas,
  fixed typed coefficients, scalar-bound comparison construction, and ordinary contribution
  normalization.
- `io.github.pho001.synaptik.compiler.FirstOrderAutograd` — retains reverse accumulation and
  generalizes only its nested request-local derivative-constant owner to exact typed splats.

No public type, package, facade, registry, or policy object is added. Tests remain in the mirrored
compiler package.

## Affected files

Expected compiler production:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseGradientRules.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`

Expected compiler tests:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`

Expected documentation and planning:

- `docs/api/compile-api.md`
- `docs/design/notes/autograd-strategy.md`
- `docs/user-guide/autograd.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a conflict requires stopping: Tensor/Training/Public APIs,
model capabilities/audits, model source/tests/Javadocs, compiler inference and
optimization source/tests, architecture documents/ADRs/tests, Gradle, config, planning, trace,
runtime, prepare, engine, extensions, backends, backend conformance, and integration tests.

## Maximum scope

This task may create or modify at most the exact fifteen paths listed under
[Affected files](#affected-files):

- 3 compiler production files;
- 4 compiler test files; and
- 8 documentation/planning files.

No other path is authorized. If a formula, exact policy, test, or documentation contract requires
a fifteenth path, a model or public API change, or another module, stop and report the concrete
blocker rather than silently narrowing an assigned operation or expanding scope.

## Acceptance criteria

- Source-backed inventory tests lock all 7 binary, 8 scalar, 19 unary, 1 WHERE, 1 CAST, 6
  comparison, 3 Boolean logical, and 3 floating-classification kinds with no unclassified 0005A
  role.
- Existing ADD/SUB/MUL/DIV, scalar ADD/SUB/MUL/DIV, WHERE branch, floating CAST,
  NEG/EXP/EXPM1/ERF/FLOOR/CEIL/SIGN/SIGMOID/TANH formulas and their Shape/type behavior remain
  unchanged.
- MIN/MAX, POW, scalar MIN/MAX/POW/CLAMP, ABS/RECIPROCAL/LOG/LOG1P/SQRT/RSQRT/RELU/GELU/
  GELU_TANH_APPROXIMATION/SILU implement exactly the formulas and policies above.
- Focused formula-structure tests cover both roles of broadcast and mixed-floating binary
  operations, exact normalization order, repeated operands, scalar exact-bit attributes, every
  new unary kind, and first-class CLAMP without decomposing the forward occurrence.
- Tests lock equal finite, opposite-signed-zero, equal-infinity, unordered-NaN, clamp endpoint,
  equal-bound, signed-zero-bound, discontinuity, raw-domain, and activation-infinity formula
  structure without adding a model evaluator.
- Every input contribution has the selected input's exact Shape and floating type before
  accumulation. Mixed-floating POW and extrema cover both selected roles independently.
- The exact typed coefficient bit table and represented-type scalar exponent-minus-one conversion
  are tested for BFLOAT16, FLOAT32, and FLOAT64.
- Request-local exact splats are identity-cached by exact typed bits, explicitly registered,
  deterministically ordered, expanded through public Tensor operations, pruned when unreachable,
  and never inferred from storage or metadata.
- Comparison, logical, classification, WHERE-condition, scalar-attribute, clamp-bound, and
  non-floating cast roles remain non-differentiable and fail closed when selected as a route.
- Unsupported/deferred operations, wrong attrs/signatures, unsupported output roles, and new
  unknown kinds fail before derivative Tensor allocation, with deterministic occurrence context.
- Repeated operand positions remain repeated deterministic contributions. Multiple target roles
  may share one captured gradient value without identity nodes.
- One combined capture preserves exact original producer identity as FORWARD, generated producers
  as BACKWARD, canonical wrappers, unique graph-local IDs, immutable result roles, and current
  publication/artifact behavior.
- Generated formulas pass existing compiler inference/validation and only the shared exact
  optimization pipeline. No gradient-specific simplifier, constant folder, or exceptional-value
  pass is added.
- No model, public Tensor, request/result, registry/facade, dependency, build, architecture,
  backend, prepare, runtime, training, or higher-order contract changes.
- Compile API, autograd strategy, autograd user guide, glossary, task, compiler and model master
  plans, and roadmap consistently describe the implemented policy and current-versus-planned
  boundary after implementation.
- Tensor API needs no change because model forward semantics and public Tensor declarations remain
  unchanged; Training and Public APIs need no change because no public request, publication, or
  training workflow changes. The documentation-focused pass must record these no-change reasons.
- Exactly the fifteen authorized paths change. Compiler 0005A becomes `Complete` only after all
  implementation, module, Javadoc, documentation, scope, and independent documentation-review
  evidence is final. Compiler 0005B–0005E and 0006 remain Draft without detailed specifications.
- A separate documentation-focused agent pass has independently finalized affected Javadocs,
  explanatory documentation, glossary impact, planning status/evidence, terminology, links, and
  rendered output in the same overall change.

## Tests / validation

During implementation, run focused suites as needed:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GradientRulesTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.FirstOrderAutogradTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GraphCompilerTest
```

After executable Java stabilizes, run one final compiler module suite:

```bash
./gradlew :modules:compiler:test
```

Record JUnit XML suite/test counts and zero skipped/failure/error outcomes. The implementation
context hands that exact evidence and the final executable diff to the documentation-focused
agent. That pass must not repeat successful Java tests unless executable behavior changes after
the evidence or it records a concrete stale-evidence risk.

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass must additionally verify generated package-private Javadocs, complete
`@param`/`@return`/expected-`@throws` coverage, local links and anchors, balanced fences, final
newlines, terminology, exact fifteen-path scope, source-backed 48-kind inventory, exact
coefficient bits, no model/test/Gradle/dependency/architecture change, synchronized task/master/
roadmap status, exactly one Ready-or-current detailed compiler task, and absence of detailed
0005B–0005E/0006 specifications.

Repository-wide validation and the compiler first-order capability checkpoint are deferred to
Compiler 0005E or continuous integration. This task changes one module's package-private autograd
behavior and no dependency, shared build, architecture boundary, or public API.

## Dependencies

- Model 0025A portable floating comparison, extrema, and clamp semantics — Complete.
- Compiler 0005 publication, planning orchestration, and compile artifacts — Complete.
- Compiler 0004–0004B pre-capture autograd, exact-composition extensions, and shared-algebra
  normalization/local rules — Complete.
- Model 0025 canonical producer outputs and the completed adjoint-expressibility audit — Complete.
- Existing public Tensor binary/scalar/unary/comparison/selection/classification/cast operations,
  typed `ScalarValue`, explicit logical-splat ingress, compiler inference/validation, and exact
  combined optimization — implemented and sufficient.

No Config, Trace, Runtime, Prepare, Engine, Training, backend, or new model task is a dependency.

## Follow-up tasks

- Compiler 0005B remains Draft and follows 0005A. It owns reductions, scans, softmax,
  statistics, norms, and normalization policies/formulas.
- Compiler 0005C remains Draft and follows 0005B. It owns remaining layout/window/indexing/
  scatter/ordering/stochastic roles.
- Compiler 0005D remains Draft and follows 0005B/0005C. It owns structured attention,
  convolution, pooling, and loss roles.
- Compiler 0005E remains Draft and follows 0005A–0005D. It owns the complete source-backed
  first-order role/output closure audit, transitive formula-operation closure, and capability
  checkpoint.
- Compiler 0006 remains Draft and follows 0005E plus the stable public compile/artifact boundary.
  It owns explicit functional requests, seeds, disconnected results, derivative order, and
  higher-order differentiation.

Do not create any of those detailed specifications during this task.

## Architecture impact

Expected impact: None.

This task fills the existing compiler-owned local differentiation extension point through the
current public Tensor algebra. It changes no authority, module ownership, dependency direction,
public lifecycle, forward semantic, or artifact shape.

If implementation requires a new model operation/API, changed forward meaning, public gradient
surface, direct graph-node formula construction, dependency, runtime tape, physical saved value,
or architecture change, stop and report the exact conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the compiler master plan, and
docs/planning/modules/compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md.
Read every directly referenced architecture, completed compiler/model task and audit, final
compiler/model source and tests, Compile/Tensor/Training APIs, autograd strategy/user guide, and
glossary needed to verify the exact current operation inventory and shared Tensor algebra.

Implement Compiler 0005A exactly within its fifteen authorized paths. Preserve the existing
request, fail-closed preflight, identity accumulation, one combined capture, phase, validation,
optimization, publication, and artifact contracts. Do not implement 0005B–0005E or 0006, change
model forward semantics or APIs, add a second algebra/registry/facade/policy mode, add
gradient-specific optimization, or touch Gradle, architecture, another module, backend, prepare,
runtime, or training behavior. Stop on any scope, formula-expressibility, inventory, or
architecture conflict rather than inventing a policy or silently omitting an assigned role.

After executable Java stabilizes, run the focused tests as needed and one final compiler module
suite. Hand the actual diff and exact test evidence to a separate documentation-focused agent or
thread with clean context. That targeted pass must follow
docs/developer-guide/documentation-rules.md, independently finalize affected Javadocs, Compile
API, autograd strategy/user guide, glossary impact, planning status/evidence, terminology, links,
and rendered Javadoc in the same overall change, and must not repeat successful Java tests unless
executable behavior changes or a concrete stale-evidence risk is recorded.

Run the final Javadoc/documentation/scope/status checks. Update this task with local decisions,
exact evidence, implementation notes, completion summary, and final status. Do not mark it
Complete before every acceptance criterion and the documentation-focused pass finish.
```

## Local decisions

- The task-specified derivative policies were implemented without additional policy choices:
  symmetric Tensor extrema ties, one-half scalar ties, ordered-composition CLAMP, direct
  ABS/RELU discontinuities, raw analytic domains, and explicit modern-activation infinity
  selection.
- `ElementwiseGradientRules` remains the sole local formula owner. The implementation adds no
  registry, facade, selectable policy object, second algebra, backward-only operation, or
  gradient-specific optimization.
- Fixed arithmetic coefficients remain exact scalar operation metadata. Values needed as Tensor
  comparison operands use the existing request-local derivative-constant owner generalized to
  exact `ScalarValue` data-type/bit identity.
- Scalar POW derives exponent-minus-one once in the represented type: BFLOAT16 through
  binary32 subtraction and the existing round-to-nearest-ties-to-even conversion, FLOAT32 through
  one binary32 subtraction, and FLOAT64 through one binary64 subtraction.
- Generated splat bindings remain deterministic by first use and are pruned by exact
  gradient-expression reachability before the caller ingress and derivative ingress are combined.

## Known limitations

- This task completes only its exact 48-kind elementwise/activation classification. Complete
  first-order current-model coverage waits for Compiler 0005B–0005E.
- The internal request remains one implicit-unit-seed scalar-objective first-order request.
  Public requests, seeds, disconnected gradients, create-graph, and derivative order wait for
  Compiler 0006.
- Formula-structure tests cannot prove backend numerical execution. Future backend implementations
  require conformance coverage for the selected tie, endpoint, NaN, infinity, singularity, and
  coefficient behavior.
- Raw analytic formulas intentionally retain ordinary shared-operation exceptional behavior at
  mathematically undefined points instead of promising finite gradients or continuous
  extensions, except for the explicit modern-activation infinity selections.

## Validation evidence

Implementation evidence reused by the documentation pass:

- Implementation context `/root/implement_compiler_0005a` ran the focused combined
  `AutogradPreflightTest`, `GradientRulesTest`, `FirstOrderAutogradTest`, and
  `GraphCompilerTest` command successfully during development.
- After the final test-only represented-boundary and source-inventory strengthening, that context
  ran `./gradlew :modules:compiler:test`; Gradle reported `BUILD SUCCESSFUL`, and the JUnit XML
  audit found 22 suites and 159 tests with 0 skipped, 0 failures, and 0 errors.
- Production executable Java did not change after that module evidence. Clean documentation
  context `/root/implement_compiler_0005a/docs_0005a` therefore reused it and did not duplicate
  the successful Java suite.

Documentation and final-gate evidence:

- The clean documentation context independently reviewed the authoritative architecture and ADR,
  General/API-Javadoc/User Guide/Planning/Example profiles, planning guide, completed Compiler
  0004–0005 and Model 0025A contracts, model audits, final three-production/four-test diff,
  source enums/signatures, public APIs, autograd documentation, glossary, and planning state.
- `./gradlew :modules:compiler:javadoc` passed with `BUILD SUCCESSFUL in 1s`, 7 actionable tasks
  (`1 executed`, `6 up-to-date`), and no warning. Generated package-private pages for
  `AutogradPreflight`, `ElementwiseGradientRules`, `FirstOrderAutograd`, and its nested request,
  plan, role, expansion, and derivative-constant contracts were inspected. Parameter, result,
  expected-failure, identity, ownership, ordering, and lifecycle descriptions are present.
- `python3 /tmp/validate_synaptik_markdown.py` reported `validated 12 Markdown files and 688 local
  links`, including file targets, heading anchors, balanced fences, final newlines, and trailing
  whitespace.
- `git diff --check` passed with no output.
- The sorted union of modified and untracked paths contains exactly the authorized fifteen paths:
  three production Java files, four compiler tests, and eight documentation/planning files.
- Manual source-backed inventory review confirmed exactly 7 binary, 8 scalar, 19 unary, 1 WHERE,
  1 CAST, 6 comparison, 3 Boolean logical, and 3 floating-classification kinds, totaling 48.
  The production enum-array test locks every constant and order rather than only counts.
- Manual formula and policy review confirmed symmetric extrema ties, ordered-composition CLAMP,
  represented-type scalar exponent-minus-one, direct ABS/RELU and FLOOR/CEIL/SIGN conventions,
  raw analytic domains, activation infinity selection, mixed-floating normalization, exact
  request-local splat identity/order/registration/pruning, deterministic accumulation, original
  versus generated producer phase, canonical wrappers, one combined capture, and shared
  inference/validation/optimization.
- The fixed coefficient table in source, focused tests, task, and Compile API matches exactly for
  BFLOAT16, FLOAT32, and FLOAT64, including the unchanged ERF coefficient. No host
  transcendental coefficient derivation was introduced.
- Task, compiler master plan, model master plan, and roadmap are synchronized to 0005A
  `Complete`. Compiler 0005B–0005E and 0006 remain concise `Draft` rows, and their detailed task
  files are absent.
- Tensor API needs no change because 0005A changes no public Tensor declaration, forward model
  semantic, descriptor, provenance, or storage contract. Training and Public APIs need no change
  because no public gradient request, publication shape, optimizer, session, prepare, run, or
  execution workflow changes.
- Model capabilities, adjoint/closure audits, model source/tests/Javadocs, and Model 0025A need no
  change because the compiler consumes their completed operation inventory and represented-value
  forward semantics unchanged.
- Compiler inference/validation and optimization documentation/source/tests need no change because
  generated formulas use the ordinary existing Tensor algebra and the unchanged shared pipeline;
  no gradient-specific inference, arithmetic, fold, rewrite, simplifier, or pass was added.
- `ARCHITECTURE.md`, focused architecture pages, ADR 0009, and architecture tests need no change
  because compiler ownership, dependency direction, one-capture lifecycle, phase model, and
  artifact boundaries are unchanged. Gradle needs no change because no module, dependency, build,
  or toolchain contract changed.
- Config, Planning, Trace, Runtime, Prepare, Engine, extensions, concrete backends, backend
  conformance, and integration tests need no change because the task adds no cross-module,
  backend, runtime, training, physical-value, end-to-end, or executable behavior.
- Repository-wide tests remain deferred to Compiler 0005E or continuous integration under the
  task's validation tier.

## Implementation notes

- Extended `AutogradPreflight` to admit the exact 0005A floating arithmetic/unary signatures and
  to reject comparison, Boolean logical, and floating-classification routes explicitly as
  non-differentiable before derivative Tensor allocation.
- Extended `ElementwiseGradientRules` with the exact extrema, POW, CLAMP, unary, GELU,
  tanh-approximation GELU, and SiLU formulas plus the fixed coefficient table.
- Generalized `FirstOrderAutograd.DerivativeConstants` from zero/one-only helpers to one
  request-local exact typed-splat cache while preserving identity, order, explicit registration,
  reachability pruning, reverse accumulation, and one combined capture.
- Added focused tests for the source-backed 48-kind inventory, formula structure, mixed-floating
  role normalization, fixed bits, represented exponent subtraction, exceptional-value policy
  structure, exact splat identity/order/pruning, fail-closed behavior, and shared-pipeline
  capture/optimization.
- The independent documentation context finalized the three affected implementation Javadocs,
  Compile API, autograd strategy, user guide, glossary, task, compiler/model master plans, and
  roadmap without changing executable Java behavior.

## Completion summary

- Completed changes: the exact 48-kind Compiler 0005A elementwise/activation classification,
  formulas, derivative policies, exact coefficients, request-local typed splats, and focused
  regression coverage.
- Files changed or created: exactly three compiler production files, four compiler test files,
  and eight documentation/planning files.
- Tests and validation: the compiler module passed 22 suites/159 tests with no skips, failures,
  or errors; compiler Javadoc, Markdown links/anchors/fences, exact scope, coefficient,
  inventory, status, final-newline, and whitespace checks passed.
- Documentation-agent review: clean context
  `/root/implement_compiler_0005a/docs_0005a` independently finalized affected Javadocs,
  explanatory documentation, glossary impact, planning state, and no-change conclusions.
- Documentation impact: Compile API, strategy, and user guide now distinguish the implemented
  0005A policy from the Draft 0005B–0005E and 0006 frontiers.
- Javadoc review: all three affected package-private production owners were reviewed and
  finalized without executable changes.
- Glossary impact: synchronized GELU, pre-capture autograd, and logical-splat entries; no new
  reusable term was required.
- Unresolved issues: None.
- Follow-up required: None. Compiler 0005B–0005E and 0006 remain their ordered Draft work.

Status: Complete
