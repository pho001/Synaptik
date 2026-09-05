# Task 0025L: Cross-type CAST conversion semantics

## Status

Complete

## Goal

Define one backend-independent, raw-bit-testable observable meaning for every ordered `CAST`
pair among the six current data types: `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT64`, `INT32`, and
`BOOL`. Preserve the existing exact-bit same-type operation and floating-only differentiation
boundary, and add one Model-owned scalar reference conversion so Model tests and later backend
conformance work share an executable oracle without placing that oracle in backend hot paths.

This task closes the numerical-policy gap left deliberately open by Model 0015G–0015H. Public
`Tensor.cast(DataType)` already constructs all 36 ordered pairs; this task gives those existing
expressions exact meaning. It adds no Tensor overload, operation kind, attributes field, data
type, backend support, or execution route.

## Scope

- Admit all 36 ordered source/target pairs: six same-type pairs and all 30 cross-type pairs.
- Preserve every same-type raw represented bit, including signed zero and every NaN sign,
  quiet/signaling state, and payload.
- Define exact floating-to-floating rounding, overflow, underflow, infinity, signed-zero, and NaN
  behavior, including direct `FLOAT64 -> BFLOAT16` conversion without a `FLOAT32` intermediate.
- Define exact signed integral widening/narrowing and all floating/integral conversions.
- Define exact `BOOL` conversion and numeric truthiness.
- Preserve the existing descriptor and differentiation contract: only floating-to-floating
  `CAST` can retain `requiresGrad`; Compiler reverses it with an ordinary `CAST` to the source
  floating type. A role with integral or `BOOL` source or target has no cotangent.
- Add one public final stateless `CastValueConversions` reference utility in the existing cast
  semantic package. Its single conversion entry accepts one exact `ScalarValue` and a target
  `DataType`, and returns one exact target-typed `ScalarValue` according to this specification.
  This is the concrete shared oracle needed by Model tests and the immediately following CPU
  conformance work. It is semantic reference code, not a Tensor evaluator or backend route.
- Update affected Javadocs and the Tensor API reference with the exact observable policy.
- Keep `BFloat16Bits.fromFloat(float)` and `BFloat16Bits.toFloat(short)` behavior unchanged,
  including canonical `FLOAT32 -> BFLOAT16` NaN and exact raw `BFLOAT16 -> FLOAT32` expansion.

### Scalar reference contract

The exact new public surface is:

```java
public final class CastValueConversions {
    public static ScalarValue convert(ScalarValue source, DataType targetDataType);
}
```

The class has one private zero-argument constructor, no mutable state, and no other public or
protected member declared by the project. `convert` checks `source` and then `targetDataType` with
`Objects.requireNonNull`, using exact messages `source` and `targetDataType`. A same-type call
returns the exact `source` reference, preserving every represented bit without a primitive
round-trip. A cross-type call returns a non-null `ScalarValue` of exactly `targetDataType` and
applies the corresponding matrix cell below. Inexact, overflow, underflow, NaN, infinity,
saturation, and modulo outcomes are valid results rather than exceptions.

This utility is the executable scalar reference only. `Tensor.cast` still constructs a fresh
expression for a same-type request and does not call `convert` or inspect storage.

### Exhaustive ordered-pair matrix

The row is the source and the column is the target. Every cell is admitted. `BITS` means exact raw
bit preservation; `RNE` means direct round-to-nearest, ties-to-even; `WIDEN` means exact finite
widening plus the NaN mapping below; `TRUNC-SAT` means truncate toward zero and saturate; `SIGNEXT`
means signed extension; `LOW32` means retain the low 32 two's-complement bits; `01` means Boolean
false/true becomes numeric zero/one; and `TRUTH` means false only for integer zero or either
floating signed zero.

| Source \\ Target | `FLOAT64` | `FLOAT32` | `BFLOAT16` | `INT64` | `INT32` | `BOOL` |
|---|---:|---:|---:|---:|---:|---:|
| `FLOAT64` | `BITS` | `RNE` | direct `RNE` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `FLOAT32` | `WIDEN` | `BITS` | `RNE` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `BFLOAT16` | `WIDEN` | exact `WIDEN` | `BITS` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `INT64` | `RNE` | `RNE` | direct `RNE` | `BITS` | `LOW32` | `TRUTH` |
| `INT32` | `RNE` | `RNE` | direct `RNE` | `SIGNEXT` | `BITS` | `TRUTH` |
| `BOOL` | `01` | `01` | `01` | `01` | `01` | `BITS` |

### Floating formats and finite rounding

`FLOAT64` is IEEE-754 binary64 (`p = 53`, exponent range `-1022..1023`), `FLOAT32` is binary32
(`p = 24`, exponent range `-126..127`), and `BFLOAT16` has binary32's eight-bit exponent with
precision `p = 8` and exponent range `-126..127`.

For every finite floating-to-floating or integer-to-floating conversion, choose the target finite
number nearest to the exact mathematical source value. An exact midpoint chooses the target whose
least-significant significand bit is zero. Preserve source sign when the rounded result is zero.
Rounding is directly to the requested target format; it is never defined as a chain through an
intermediate format.

If the rounded magnitude exceeds the target's finite range, the result is signed infinity. Tiny
values round gradually through target subnormals and then to signed zero. The boundary cases are:

| Target | Largest finite bits | Positive infinity bits | Smallest positive subnormal | Zero/subnormal midpoint |
|---|---:|---:|---:|---:|
| `FLOAT64` | `0x7FEFFFFFFFFFFFFF` | `0x7FF0000000000000` | `2^-1074` / `0x0000000000000001` | `2^-1075` (conceptual exact midpoint) |
| `FLOAT32` | `0x7F7FFFFF` | `0x7F800000` | `2^-149` / `0x00000001` | `2^-150` |
| `BFLOAT16` | `0x7F7F` | `0x7F80` | `2^-133` / `0x0001` | `2^-134` |

At a zero/subnormal midpoint, ties-to-even selects signed zero. Immediately above it selects the
signed smallest subnormal. For `FLOAT32` and `BFLOAT16`, the positive finite/infinity RNE overflow
midpoints are respectively `(2 - 2^-24) * 2^127` and `(2 - 2^-8) * 2^127`; the midpoint and larger
magnitudes produce infinity, while smaller magnitudes round to a finite result. Negative behavior
is sign-symmetric.

`FLOAT64 -> BFLOAT16` must operate from the exact binary64 represented value. These required
vectors distinguish direct conversion from forbidden double rounding through `FLOAT32`:

| Exact binary64 input | Direct BFLOAT16 | Intermediate FLOAT32 | BFLOAT16 after the forbidden intermediate |
|---:|---:|---:|---:|
| `0x3FF0100000400000` (`1 + 2^-8 + 2^-30`) | `0x3F81` | `0x3F808000` | `0x3F80` |
| `0x3FF02FFFFFC00000` (`1 + 3 * 2^-8 - 2^-30`) | `0x3F81` | `0x3F818000` | `0x3F82` |

Tests must cover both vectors, their negative counterparts, exact halfway values on both even/odd
target neighbors, carry into a new exponent, finite/infinity boundaries, and zero/subnormal
boundaries.

### Edge-case audit tables

The following tables are normative partitions of the same 36-pair matrix. They make each edge
class independently auditable; they do not add another conversion mode.

#### Zero

| Source zero | Floating target | Integral target | `BOOL` target |
|---|---|---|---|
| floating `+0` | exact `+0` | `0` | `false` |
| floating `-0` | exact `-0` | `0` | `false` |
| integer `0` | `+0` | exact zero under `BITS`, `SIGNEXT`, or `LOW32` | `false` |
| `BOOL false` | `+0` | `0` | `false` |

#### Subnormal and underflow

| Conversion class | Required behavior |
|---|---|
| same floating type | preserve the exact subnormal raw bits |
| lossless floating widening | preserve the exact finite value; a source subnormal may become a normal target value |
| lossy floating narrowing | direct target RNE, including gradual target subnormals; exact half of the least target subnormal rounds to signed zero and the next representable source value above it rounds to the signed least target subnormal |
| floating to integral | every source subnormal truncates to integer zero |
| floating to `BOOL` | every positive or negative non-zero subnormal maps to `true` |

#### Finite overflow

| Conversion class | Positive result | Negative result |
|---|---|---|
| lossy floating narrowing at or beyond the target RNE overflow midpoint | `+infinity` | `-infinity` |
| floating to integral above/below the target interval | target maximum | target minimum |
| current integer to floating | direct finite RNE result; current `INT32`/`INT64` magnitudes do not overflow any current floating exponent range | sign-symmetric finite RNE result |

#### Infinity

| Source | Floating target | Integral target | `BOOL` target |
|---|---|---|---|
| `+infinity` | `+infinity` | target maximum | `true` |
| `-infinity` | `-infinity` | target minimum | `true` |

### Floating NaN policy

Same-type casts preserve the complete raw pattern. Cross-floating narrowing is lossy and maps
every source NaN, regardless of sign, signaling/quiet bit, or payload, to the positive canonical
quiet target pattern:

| Target | Canonical NaN bits |
|---|---:|
| `FLOAT64` | `0x7FF8000000000000` |
| `FLOAT32` | `0x7FC00000` |
| `BFLOAT16` | `0x7FC0` |

No current lossy conversion targets `FLOAT64`, because all current sources are no wider than
binary64. Its canonical pattern is nevertheless fixed so the family contract names all three
current floating targets; none of the current 36 pairs emits it through lossy conversion.

The only lossless cross-floating directions are `BFLOAT16 -> FLOAT32`, `BFLOAT16 -> FLOAT64`, and
`FLOAT32 -> FLOAT64`. For a NaN in those directions, preserve the sign bit and exponent of all
ones, then place the complete source fraction in the most-significant bits of the target fraction
and fill newly available low fraction bits with zero:

```text
BFLOAT16 -> FLOAT32: targetFraction = sourceFraction << 16
BFLOAT16 -> FLOAT64: targetFraction = sourceFraction << 45
FLOAT32  -> FLOAT64: targetFraction = sourceFraction << 29
```

This preserves all source payload bits and the source quiet/signaling bit deterministically. The
first mapping is exactly the existing `BFloat16Bits.toFloat(bits)` raw expansion. Representative
required mappings include `BF16 0x7F81 -> F32 0x7F810000`, `BF16 0xFFC1 -> F64
0xFFF8200000000000`, and `F32 0x7FA12345 -> F64 0x7FF42468A0000000`. Tests must independently
recalculate these patterns rather than relying on Java's unspecified NaN payload propagation.

The NaN edge cases are therefore:

| Source/target class | Required result |
|---|---|
| same floating type | exact source raw bits |
| lossy cross-floating narrowing | exact positive canonical quiet target NaN from the table above |
| lossless cross-floating widening | exact sign/exponent/left-aligned-fraction mapping above |
| floating NaN to integral | zero |
| floating NaN to `BOOL` | `true` |

### Integral policy

- `INT32 -> INT64` sign-extends the exact two's-complement value.
- `INT64 -> INT32` retains bits 31 through 0 and interprets them as a signed two's-complement
  `INT32`, equivalently reduction modulo `2^32`. Examples: `0x0000000100000001L -> 1`,
  `0x00000000FFFFFFFFL -> -1`, `Long.MIN_VALUE -> 0`, and `Long.MAX_VALUE -> -1`.
- Same-type integral casts preserve all bits.
- Integer-to-floating uses the direct target RNE rule. Inexact results are valid. Required boundary
  examples include `Integer.MAX_VALUE -> FLOAT32 0x4F000000` (`2^31`),
  `Long.MAX_VALUE -> FLOAT64 0x43E0000000000000` (`2^63`), and `Long.MIN_VALUE -> FLOAT64
  0xC3E0000000000000` (exact `-2^63`). Integer-to-BFLOAT16 is direct, not FLOAT32-mediated. For
  example, `INT32 1077936129` (`2^30 + 2^22 + 1`) converts directly to BFLOAT16 `0x4E81`;
  conversion through FLOAT32 first produces midpoint `0x4E808000` and then forbidden BFLOAT16
  `0x4E80`. `INT64 2155872257` similarly converts directly to `0x4F01`, not the double-rounded
  `0x4F00` obtained through FLOAT32 midpoint `0x4F008000`.
- Floating-to-integer first interprets the source's represented value. NaN becomes zero. Positive
  and negative infinity become the target maximum and minimum. For finite `x`, compute the exact
  mathematical truncation toward zero, then clamp that integer to the inclusive target interval:
  `[-2^63, 2^63 - 1]` or `[-2^31, 2^31 - 1]`.

Floating-to-integer boundary tests must include positive/negative fractional values, both signed
zeros, smallest/largest source subnormals, exact integral values, and these target edges:

| Target | Required source/result examples |
|---|---|
| `INT32` | binary64 `2147483646.9 -> 2147483646`; `2147483647.9 -> 2147483647`; `-2147483648.9 -> -2147483648`; binary32 bits `0x4EFFFFFF -> 2147483520`; `0x4F000000 -> 2147483647`; `0xCF000000 -> -2147483648` |
| `INT64` | binary64 bits `0x43DFFFFFFFFFFFFF -> 9223372036854774784`; `0x43E0000000000000 -> Long.MAX_VALUE`; `0xC3E0000000000000 -> Long.MIN_VALUE` |

The integral-width edge cases are:

| Source | Target | Required result |
|---|---|---|
| `Integer.MIN_VALUE` / `Integer.MAX_VALUE` | `INT64` | exact sign-extended values |
| `Long.MIN_VALUE` | `INT32` | `0` |
| `Long.MAX_VALUE` | `INT32` | `-1` |
| any same-type integral value | same integral type | exact source bits |
| current integral limit | any floating type | one direct target RNE, with no intermediate floating format |

### Boolean policy

| Source | Target/result |
|---|---|
| `BOOL false` | numeric positive zero, integral zero, or `BOOL false` |
| `BOOL true` | numeric positive one, integral one, or `BOOL true` |
| numeric `+0` or floating `-0` | `BOOL false` |
| any non-zero finite numeric value or subnormal | `BOOL true` |
| either infinity or any NaN raw pattern | `BOOL true` |

`ScalarValue` accepts only the canonical Boolean domain, so this task does not define truthiness
for arbitrary storage bytes.

### Public expression and differentiation behavior

`Tensor.cast` continues to create one fresh explicit storage-free expression for every valid call,
including same-type casts. Shape identity, unresolved result layout, provenance, allocation order,
and null failures remain unchanged. This task changes the documented observable value meaning, not
construction mechanics.

The exact existing differentiation matrix remains:

| Source category | Target category | Result retains requested `requiresGrad` | Input cotangent |
|---|---|---:|---|
| floating | floating | yes | incoming cotangent unchanged for same type; otherwise one ordinary `CAST` to the source type |
| floating | integral or `BOOL` | no | none |
| integral or `BOOL` | any | no | none |

The cast-back uses the same forward numerical contract in this task; there is no gradient-specific
rounding, NaN, saturation, straight-through estimator, or cotangent for a non-floating role.

### Current and planned behavior

Current after implementation: `Tensor.cast` constructs all 36 expressions, Model owns their exact
value meaning through `CastValueConversions`, and Compiler retains its existing floating-only
reverse boundary. CPU remains fail-closed for cross-type CAST until CPU 0008K implements and
truthfully advertises execution.

CPU 0008K is now unblocked by this completed semantic prerequisite but remains Draft; it owns
scalar/parallel-scalar execution, not this task. Model 0026 remains the separate future Draft owner
of `FLOAT16` and mixed-precision semantics.

## Out of scope

- Java implementation by this planning/documentation pass
- CPU or another backend capability, lowering, loops, code generation, vectorization, execution,
  storage conversion, hot-path dispatch, or performance evidence
- constant folding, redundant-cast elimination, cast-chain rewriting, promotion changes, or
  implicit casts
- a new `Tensor` method/overload, operation kind, `CastAttrs` field, configurable conversion mode,
  rounding mode, saturation mode, error flag, or exception-on-loss policy
- changing Compiler inference, derivative inventory, or gradient formulas
- adding `FLOAT16`; quantized, unsigned, complex, string, or arbitrary-precision types
- mixed-precision accumulation/intermediate policies for non-cast operations; Model 0026 retains
  all true binary16 and mixed-precision scope
- treating non-canonical backend BOOL storage bytes as Model Boolean values

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants,
  `modules/model`, Compiler-owned automatic differentiation, `modules/compiler`, and concrete
  backend ownership
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Model master plan](../master-plan.md)
- [CPU master plan](../../../backends/cpu/master-plan.md)
- [Model 0001 data type model](0001-data-type-model.md)
- [Model 0003A data type package migration](0003a-data-type-package-migration.md)
- [Model 0015G cast kind and attributes](0015g-cast-semantic-kind-and-attributes.md)
- [Model 0015H cast Tensor expression](0015h-cast-tensor-expression.md)
- [Model 0018N typed scalar value contract](0018n-typed-scalar-value-contract.md)
- [Model 0018U integral arithmetic and comparisons](0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Model 0025K 3D window transforms](0025k-public-ncdhw-unfold3d-and-fold3d-window-transforms.md)
- [Compiler 0005A derivative policy](../../../modules/compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
- [Compiler 0005E gradient coverage audit](../../../modules/compiler/tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md)
- [CPU 0008J BFLOAT16 scalar pointwise closure](../../../backends/cpu/tasks/0008j-bfloat16-scalar-pointwise-closure.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md), with General,
  Planning, Example, and API/Javadoc profiles

## Architecture constraints

- Model owns the conversion meaning and reference scalar semantics but no backend capability,
  physical carrier selection, loop, kernel, or execution route.
- `Operation` remains backend-neutral and exposes no support query.
- `Tensor` remains immutable identity/descriptor/provenance model state, not IR, storage execution,
  or gradient lifecycle state.
- Compiler continues to own derivative rules and constructs them only through public Tensor
  expressions. This task changes no Compiler source.
- CPU 0008K may consume the completed contract but must implement specialized conversion directly;
  generated or direct backend hot loops must not call `CastValueConversions`, allocate
  `ScalarValue` per element, box, reflect, or dispatch through maps/strings.
- The scalar reference utility may be used by Model tests, backend-conformance tests, and cold
  verification. It is not a fallback executable and does not advertise backend support.
- No dependency direction, module boundary, build structure, or architecture rule changes.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype` — owns `DataType`, exact typed `ScalarValue`, and the
  existing BFLOAT16 bit contract.
- `io.github.pho001.synaptik.model.operation.elementwise.cast` — owns CAST meaning and attributes.
- `io.github.pho001.synaptik.model.tensor` — owns public expression construction and Javadoc.

Packages added or changed:

- No package is added.
- The existing cast package gains the conversion reference because it owns the complete CAST
  semantic family rather than general promotion or implicit data-type conversion.

Type placement:

- `io.github.pho001.synaptik.model.operation.elementwise.cast.CastValueConversions` — public final
  stateless scalar reference for the exact operation semantics; public visibility is justified by
  the concrete cross-module conformance consumer, while its Javadoc excludes hot-path use and
  backend-support meaning.
- Existing `CastKind`, `CastAttrs`, `TensorCastExpressions`, and `Tensor.cast` remain in place and
  gain exact numerical-contract Javadoc only.

## Affected files

Expected implementation and documentation files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastValueConversions.java` (new)
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorCastExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastValueConversionsTest.java` (new)
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/elementwise/cast/CastSemanticsTest.java`, only if needed to lock the new public type/family relationship
- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/tasks/0025l-cross-type-cast-conversion-semantics.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Reviewed but normally unchanged:

- `BFloat16Bits.java` and `BFloat16BitsTest.java`; their existing `FLOAT32 <-> BFLOAT16` behavior is
  preserved and the widening/narrowing oracle must agree with it.
- `DataType.java`, `DataTypePromotion.java`, and their tests; explicit CAST does not alter type
  vocabulary or promotion.
- `ScalarValue.java` and `ScalarValueTest.java`; the existing typed raw-value carrier is sufficient.
- `TensorCastExpressionTest.java`; it already exhaustively locks 36-pair construction and the
  gradient-eligibility matrix.
- Compiler cast source/tests; the existing floating cast-back and non-differentiable boundary are
  preserved.

## Maximum scope

This task may create or modify at most 14 files total, including this task and synchronized
planning files. At most two new Java files are allowed: the reference type and its focused test.
No Gradle, architecture, Compiler, CPU/backend, conformance, or integration-test file may change.

If a different public method/type, another package, more files, or backend/compiler work is needed,
stop and propose a follow-up task.

## Acceptance criteria

- All 36 ordered pairs are admitted and tested or covered by exhaustive matrix iteration.
- Same-type conversion returns a target `ScalarValue` with identical type and raw represented bits.
- Every cross-floating finite, zero, infinity, overflow, underflow, and NaN rule above is covered at
  raw-bit level, including direct binary64-to-BFLOAT16 double-rounding counterexamples.
- Lossy NaNs produce exactly the three canonical patterns; widening NaNs implement exactly the
  three left-aligned mappings and preserve sign, quiet/signaling state, and payload.
- Integral sign extension, low-32-bit narrowing, integer-to-floating RNE, floating truncation/
  saturation, NaN-to-zero, and infinity saturation match the exact boundaries and examples above.
- Boolean conversions match positive 0/1 and numeric truthiness for signed zero, subnormal,
  finite non-zero, infinity, and every selected NaN sign/payload.
- `CastValueConversions` is null-safe with deterministic parameter-order failures, accepts every
  `ScalarValue`/target pair, has no mutable state, storage access, Tensor dependency, backend
  dependency, registry, generic map, reflection, or per-call policy object.
- Its exact declared API is one private zero-argument constructor and one public static
  `convert(ScalarValue, DataType)` method; same-type conversion returns the exact source reference,
  and null checks use the specified order/messages without conversion work.
- Existing `BFloat16Bits.fromFloat` canonicalization and `toFloat` exact expansion remain passing
  and unchanged in meaning.
- `Tensor.cast` public shape, fresh occurrence, descriptor/provenance, all-pair construction, and
  exact differentiation matrix remain unchanged.
- Javadocs define value behavior without claiming evaluation at Tensor construction or universal
  backend support. Every public parameter/result/failure is documented.
- The Tensor API explains all conversion categories, canonical/mapped NaNs, direct RNE, integer
  boundaries, Boolean truthiness, and differentiation, with at least the direct-BFLOAT16,
  modulo-narrowing, saturation, signed-zero, and NaN examples above.
- The Compile API remains aligned with the already implemented floating-only reverse rule and now
  points to the Model-owned forward CAST value contract. The glossary updates the existing Cast
  expression entry from numerical conversion being deferred to Model-defined semantics whose
  executable support remains backend-specific.
- Existing Compiler tests prove floating-to-floating reverse casts the cotangent to the source
  type, same-type floating reverse may reuse the incoming cotangent, and any cast involving an
  integral or `BOOL` role has no cotangent. Compiler production source remains unchanged.
- The separate documentation-focused pass finalizes affected Javadocs, API text, examples, links,
  and glossary impact in the same overall change.
- Model 0025L becomes `Complete` only after implementation, Model tests, Javadoc, documentation
  validation, and completion evidence pass. CPU 0008K remains `Draft` and becomes dependent on
  completed Model 0025L; CPU 0008K–0008P retain their exact relative order and scope. Model 0026
  remains separate future `FLOAT16`/mixed-precision work without a detailed specification.

## Tests / validation

Implementation pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.elementwise.cast.CastValueConversionsTest --tests io.github.pho001.synaptik.model.operation.elementwise.cast.CastSemanticsTest --tests io.github.pho001.synaptik.model.datatype.BFloat16BitsTest --tests io.github.pho001.synaptik.model.tensor.TensorCastExpressionTest
./gradlew :modules:model:test
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GradientRulesTest --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest --tests io.github.pho001.synaptik.compiler.FirstOrderGradientCoverageTest
```

The focused tests must use independent expected raw constants and exact arithmetic/reasoning for
boundary generation; they must not compute expected results by calling the production oracle or
by chaining Java conversions in cases where direct target rounding is required.

Documentation pass:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also perform targeted checks for:

- task/master/roadmap status and `0025K -> 0025L -> 0026` ordering;
- CPU `0008J -> 0008K -> 0008L -> 0008M -> 0008N -> 0008O -> 0008P` preservation and CPU 0008K's
  exact dependency on completed Model 0025L;
- all local Markdown targets and heading anchors changed by this task;
- balanced Markdown fences and final newlines;
- exact changed-path membership, the 14-path ceiling, and absence of Java changes outside Model;
- exact 36 matrix cells, three canonical NaNs, three widening mappings, and required edge examples;
- no change to architecture/build/dependency rules, `DataTypePromotion`, Compiler behavior, CPU
  execution, or Model 0026 scope.

Repository-wide Java validation is deferred to CPU 0009/CI because this task changes one Model
semantic family and no dependency, architecture, build, Compiler, or backend source. The focused
Compiler command is the concrete preservation check for the already implemented reverse boundary;
it does not authorize a Compiler edit. No JMH,
performance, generated-Class-File, backend-conformance, integration, or CPU test is required here.

## Dependencies

- Project-owner policy in this task — approved and complete; it resolves all conversion-policy
  questions that previously blocked Model meaning and CPU execution.
- Model 0001 and 0003A — complete; supply the six data types and BFLOAT16 contracts/package.
- Model 0015G and 0015H — complete; supply `CAST`, attributes, all-pair expression construction,
  and current gradient eligibility.
- Model 0018N — complete; supplies exact `ScalarValue` representation for all six types.
- Model 0018U — complete; supplies signed integral/two's-complement conventions.
- Model 0025K — complete; immediately precedes this authorized interleave.
- Compiler 0005A/0005E — complete; supply and audit the preserved cast differentiation boundary.
- CPU 0008J — complete; supplies exact current BFLOAT16 scalar operation boundaries before CPU
  cast execution.

## Follow-up tasks

- CPU 0008K remains the next Draft CPU implementation frontier. Only after this task is Complete
  may its detailed specification be created; it owns scalar/parallel-scalar backend execution and
  must implement direct unboxed conversion logic rather than call the Model oracle per element.
- CPU 0008L–0008P retain their existing order and scope after 0008K.
- Model 0026 remains the separate future owner of true IEEE-754 binary16 `FLOAT16` and operation-
  family mixed-precision contracts. It must not reinterpret this task as adding binary16.
- Backend-conformance CAST vectors may consume `CastValueConversions` in test/cold code when the
  first executable backend task establishes that suite; this task does not create that suite.

## Architecture impact

Expected impact: None.

This task fills in Model-owned operation semantics within existing packages and dependency rules.
It adds no architecture owner, lifecycle phase, module dependency, execution contract, or backend
support. If implementation requires any architecture or dependency change, stop and report it.

Required no-change conclusions:

- `ARCHITECTURE.md`, focused architecture pages, ADRs, and architecture tests remain unchanged
  because ownership and dependency direction do not change.
- `DataType`, `DataTypePromotion`, and Model capabilities remain unchanged because all six types
  and all-pair CAST construction already exist; this task defines values rather than adding a type,
  promotion, expression, or executable capability.
- `BFloat16Bits` remains unchanged because its exact BFLOAT16-to-FLOAT32 expansion and canonical
  FLOAT32-to-BFLOAT16 narrowing are incorporated, not replaced; direct FLOAT64-to-BFLOAT16 belongs
  to the new CAST reference.
- Compiler production and Training API remain unchanged because the current floating-only
  cotangent policy is preserved rather than redesigned.
- CPU/backend source, backend-conformance tests, and integration tests remain unchanged because no
  backend truthfully executes this new contract until CPU 0008K or a later backend task.
- Gradle and Java toolchain configuration remain unchanged because the task adds no dependency,
  source set, module, or language-level requirement.

The implementation and documentation passes must record these conclusions again against the final
diff. If the final implementation invalidates one, stop instead of silently widening the task.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are the clean-context implementation agent for Synaptik Model task 0025L.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/model/tasks/0025l-cross-type-cast-conversion-semantics.md in full. Read the
directly referenced current source/tests and implement exactly that task. Do not broaden scope or
implement CPU/backend execution. Stop and report any architecture, semantic, path-budget, or
direct-rounding conflict instead of inventing policy. Do not commit, push, stage, reset, or alter
unrelated work.

After implementation and the specified Model tests pass, hand the exact diff and test evidence to
a separate clean-context documentation-focused agent. That agent must follow
docs/developer-guide/documentation-rules.md, finalize the affected Javadocs, Tensor API, examples,
links, glossary impact, planning synchronization, and documentation validation without rerunning
successful Java tests unless executable Java changes afterward or a concrete risk requires it.

Record local decisions, limitations, all validation evidence, implementation notes, completion
summary, and the required task status in the task. Do not mark Complete until every acceptance
criterion and the documentation pass are complete.
```

## Local decisions

- The owner-approved policy fixes positive canonical quiet NaNs for lossy narrowing and
  left-aligned fraction mapping for lossless widening; no implementation discretion remains over
  NaN sign/payload behavior.
- A single public `CastValueConversions` scalar reference is justified by the current 30-pair
  semantic test matrix and immediate CPU conformance dependency. A test-only duplicate would not
  provide a shared oracle, while backend use in an element loop would violate the performance and
  ownership boundary.
- The reference consumes and returns `ScalarValue`, reusing the complete current typed raw-value
  carrier rather than introducing a union, carrier map, or overload per source/target pair.
- Conversion is explicit CAST meaning and does not alter `DataTypePromotion` or authorize implicit
  cross-category arithmetic.

## Known limitations

- No backend executes the newly defined cross-type semantics until an implementation task such as
  CPU 0008K truthfully advertises and lowers them.
- The scalar oracle is not a bulk conversion API, Tensor evaluator, storage adapter, or performance
  implementation.
- `FLOAT16` and non-cast mixed-precision semantics remain deferred exclusively to Model 0026.

## Validation evidence

- Implementation context ran the focused Model command from this task after executable code and
  tests stabilized: 32 tests passed. During development, one mistyped FLOAT32 finite-overflow test
  input constant was corrected to the intended boundary value; this was test-fixture correction,
  not a product failure.
- Implementation context ran `./gradlew :modules:model:test`: 1,114 tests passed.
- Implementation context ran the focused Compiler command from this task: 42 tests passed,
  preserving floating-to-floating cast-back, same-type cotangent reuse, and the no-cotangent
  boundary for every cast involving an integral or BOOL role.
- Implementation context ran `git diff --check`: passed.
- One attempted non-Gradle shell-wrapper invocation failed because Java was absent from that
  wrapper's `PATH`. The Gradle validation commands above completed successfully in the repository
  environment, so this environmental diagnostic is not a product or test failure.
- Clean-context documentation finalization agent for task 0025L independently reviewed the final
  implementation diff, `CastValueConversions`, all affected CAST Javadocs, the focused Model test,
  Tensor/Compile APIs, glossary, Compiler CAST-gradient contracts, `BFloat16Bits`, `ScalarValue`,
  planning status/dependencies, and required no-change areas. No executable Java behavior or test
  was changed, so the successful implementation test evidence remained current and was not rerun.
- Documentation-agent validation: `./gradlew :modules:model:javadoc` passed; targeted local
  Markdown target/anchor, balanced-fence, trailing-whitespace/final-newline, exact matrix/example,
  status/order/dependency, 13-path membership/ceiling, and Java-scope checks passed; final
  `git diff --check` passed; final `git status --short` showed exactly the 13 task paths.
- The first ad hoc Ruby link-check invocation used `Array.filter_map`, which is unavailable in the
  repository shell's Ruby version. The compatible `map.compact` form then checked the same targets
  and anchors successfully; this was checker portability, not a documentation or product failure.
- Package placement remains exactly as planned: the new public scalar reference and its test are
  in `io.github.pho001.synaptik.model.operation.elementwise.cast`; no package was added.
- Repository-wide Java validation remains deferred to CPU 0009/CI as specified. No JMH,
  generated-Class-File, backend-conformance, integration, or CPU execution validation applies to
  this Model semantic task.

## Implementation notes

- Added the public final stateless `CastValueConversions` scalar oracle with the exact one-method
  API and deterministic source-then-target null checks. Same-type calls return the exact source;
  all cross-type calls return an exact target-typed `ScalarValue`.
- Implemented all 36 ordered conversions, including direct binary64/integer-to-BFLOAT16 rounding,
  exact same-type bits, canonical lossy NaNs, left-aligned widening NaNs, signed zero, gradual
  underflow, signed infinity, floating-to-integral truncation/saturation, integral sign extension
  and modulo narrowing, and canonical BOOL conversion/truthiness.
- Finalized the affected CAST Javadocs and Tensor/Compile/glossary documentation while preserving
  the boundary that Tensor construction records but does not evaluate values and backend support
  remains separately advertised.
- No-change conclusions against the final diff: architecture contracts/pages, ADRs, and
  architecture tests remain unchanged because ownership and dependencies did not change;
  `DataType`, `DataTypePromotion`, and Model capabilities remain unchanged because no type,
  promotion, expression family, or backend capability was added; `BFloat16Bits` and `ScalarValue`
  remain unchanged because their exact carriers already support the policy; Compiler production
  and Training API remain unchanged because the existing floating-only reverse rule was only
  documented; backend source, conformance tests, and integration tests remain unchanged because
  execution is deferred to CPU 0008K; Gradle remains unchanged because no build or toolchain
  requirement changed; and no other module required modification.

## Completion summary

- Completed changes: implemented the exact backend-independent 36-pair CAST value contract and
  scalar oracle; finalized affected Javadocs, API/glossary documentation, planning status, CPU
  dependency wording, validation evidence, and no-change conclusions.
- Files changed or created: `CastValueConversions.java`, `CastKind.java`, `CastAttrs.java`,
  `TensorCastExpressions.java`, `Tensor.java`, `CastValueConversionsTest.java`, this task,
  `docs/api/tensor-api.md`, `docs/api/compile-api.md`, `docs/glossary.md`, the Model master plan,
  the CPU master plan, and the roadmap (13 paths total).
- Tests and validation: reused the implementation agent's passing 32-test focused Model,
  1,114-test full Model, 42-test focused Compiler, and `git diff --check` evidence; the independent
  documentation pass then passed Model Javadoc and all requested documentation, status, order,
  dependency, scope, path-count, whitespace, and final-diff checks without changing executable
  Java or tests.
- Documentation-agent review: complete in the mandatory separate clean context.
- Documentation impact: Tensor and Compile API references now state the exact conversion and
  differentiation boundaries without claiming Tensor evaluation or backend execution.
- Javadoc review: all new and affected CAST contracts meaningfully document purpose, semantics,
  inputs, results, failures, mutation/identity behavior, and ownership boundaries.
- Glossary impact: the existing Cast expression entry now names the Model-owned value policy and
  keeps backend support and execution separate; no invented term was added.
- Unresolved issues: None.
- Follow-up required: CPU 0008K remains the next Draft CPU implementation frontier; Model 0026
  remains separate future Draft FLOAT16/mixed-precision work.

Status: Complete
