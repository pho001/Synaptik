# Task 0025A: Portable Floating Comparison, [Extrema](../../../../glossary.md#saturation-pole-and-extremum), and Clamp Semantics

**Editorial reading key (current context, not a status update).**

This record documents a forward-contract clarification, not execution or derivative completion; its
compiler-planning statements describe that historical scope. A caller supplies a receiver Tensor, a
right Tensor for binary methods, or exact same-typed `ScalarValue` attributes for scalar methods.
Model construction records ordered input references and operation descriptions, not calculated
elements. Provenance means the producing operation and inputs; a leaf lacks a producing expression,
not necessarily storage. Storage-free results own no result buffers. Model semantic responsibility
is not storage ownership; execution providers realize supported routes without redefining meaning.

For current comparison rules, follow [binary
comparisons](../../../../reference/tensor/arithmetic.md#binary-comparison-expressions): numeric
equality equates opposite signed zeros but never NaN, unlike raw-bit metadata equality. For clamp
ordering and separately described current compiler derivatives, follow [scalar arithmetic and
clamps](../../../../reference/tensor/elementwise.md#scalar-arithmetic-and-clamp-expressions). Clamp
applies maximum with the lower bound before minimum with the upper bound, yet records one producer.
Thus bounds `[+0, -0]` yield negative zero for every non-NaN input; any NaN input or bound yields
NaN without a payload promise.

## Status

Complete

## Goal

Close the smallest [model](../../../../glossary.md#term-model)-owned forward-semantics prerequisite needed before [Compiler](../../../../glossary.md#term-compiler) task 0005A
can plan floating-comparison-dependent [gradient](../../../../glossary.md#term-gradient) formulas.

Make the current comparison, pairwise minimum/maximum, [scalar](../../../../glossary.md#term-scalar) minimum/maximum, and first-class
range-clamp contracts state one fixed portable meaning for [NaN](../../../../glossary.md#term-nan), [signed zero](../../../../glossary.md#term-signed-zero), equality, and exact
typed scalar values. The task changes [Javadoc](../../../../glossary.md#term-javadoc) and explanatory documentation only. It adds no
evaluator, [operation](../../../../glossary.md#operation), attribute, [Tensor](../../../../glossary.md#tensor) method, test-only numerical implementation, compiler
formula, [derivative](../../../../glossary.md#term-derivative) convention, [backend](../../../../glossary.md#backend) behavior, or execution.

The semantic relationship is:

```text
represented floating values
  -> fixed comparison and extrema meaning in modules/model
  -> later backend implementations reproduce that meaning

fixed forward meaning
  -> later Compiler 0005A selects derivative behavior separately
```

## [Scope](../../../../glossary.md#term-scope)

- Finalize `BinaryComparisonKind` Javadoc so all six floating relations have explicit NaN and
  signed-zero behavior.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Finalize [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) and [scalar](../../../../glossary.md#term-scalar) `MIN`/`MAX` Javadoc so both families state the same complete
  NaN-propagating, signed-zero-selecting portable contract.
- Finalize `CLAMP` and `ClampRangeAttrs` Javadoc so first-class range clamp is exactly the ordered
  composition `MIN(MAX(input, minValue), maxValue)` under those [extrema](../../../../glossary.md#saturation-pole-and-extremum) rules.
- Preserve the current `ClampRangeAttrs` validation that accepts equal bounds, both opposite-
  signed-zero bound orders, and one or two NaN endpoints while rejecting only a strictly inverted
  represented numeric range after its existing type/BOOL checks.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Finalize the corresponding public [`Tensor`](../../../../glossary.md#tensor) method Javadocs for binary comparisons, pairwise
  binary/scalar extrema, range clamp, and one-bound clamp conveniences.
- Clarify the current Tensor API, the Compile API's model-semantics prerequisite status, and the
  glossary atomically.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>
- Record this task as the sole detailed [`Ready`](../../../../glossary.md#term-ready) task in the [model](../../../../glossary.md#term-model) master plan and roadmap.
- Keep [Compiler](../../../../glossary.md#term-compiler) 0005A `Draft` without a task file, make it depend on completed Model 0025A before
  promotion, and preserve Compiler 0005B–0005E and 0006 as `Draft` without task files.
- During implementation, synchronize this task, the model master plan, compiler master plan, and
  roadmap only after all documentation and validation acceptance criteria pass.

## Fixed forward semantic contract

### Floating comparisons

For [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), and [BFLOAT16](../../../../glossary.md#term-bfloat16), comparisons use the ordinary numeric value represented by
each operand under the existing same-category promotion contract:

- `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, and `LESS_OR_EQUAL` are ordered numeric
  comparisons. If either operand is NaN, the result is false.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan))</span>
- Negative zero and positive zero compare numerically equal. Neither is strictly before or after
  the other; both inclusive relations are true in either operand order.<span data-glossary-navigation="true"> ([Negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- `EQUAL` is exact numeric equality, not bit equality or approximate/tolerance equality. NaN is
  unequal to every value, including itself, and negative zero equals positive zero.<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- `NOT_EQUAL` is the exact logical complement of `EQUAL`. It is therefore true when either operand
  is NaN and false for either ordering of opposite-signed zero.<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- [Finite](../../../../glossary.md#term-finite) values and [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) otherwise use ordinary numeric order.

The existing integral contract is unchanged: [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) operands use current signed exact
comparison after current promotion, with exact equality and no floating special-value policy.

### Floating pairwise minimum and maximum

[Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) and [scalar](../../../../glossary.md#term-scalar) floating `MIN`/`MAX` use one typed rule:

- if either candidate is NaN, the result is NaN, without a payload, sign, source-operand, or
  bitwise-result promise;<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- for opposite-signed zeros, `MIN` produces [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) and `MAX` produces positive zero,
  independent of operand order;<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) and unequal non-[NaN](../../../../glossary.md#term-nan) values use ordinary numeric order; and
- equal nonzero candidates produce that numeric value without promising which operand's
  representation or identity is selected.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>

Scalar operations compare the input's represented value with the exact represented value in the
same-typed [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue). `ScalarValue` continues to preserve exact raw floating bits for [metadata](../../../../glossary.md#term-metadata)
equality and hashing. That storage/equality contract does not turn [elementwise](../../../../glossary.md#term-elementwise) numeric equality
into bit equality.

The existing integral `MIN`/`MAX` contract remains ordinary signed order after current promotion.

### First-class range clamp

For each logical input value:

```text
CLAMP(input, minValue, maxValue)
  = MIN(MAX(input, minValue), maxValue)
```

The order is semantic: apply `MAX` with `minValue` first, then `MIN` with `maxValue`. The [operation](../../../../glossary.md#operation)
remains one first-class `CLAMP` [occurrence](../../../../glossary.md#term-occurrence) with no stored intermediate [producers](../../../../glossary.md#term-producer).

Consequences that documentation must state explicitly:

| Input or bounds | Result |
|---|---|
| input, `minValue`, or `maxValue` is [NaN](../../../../glossary.md#term-nan) | NaN, with no [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) promise |
| equal non-NaN, same-representation bounds | that bound for every non-NaN input |
| bounds `[-0, +0]` | negative inputs and [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) produce `-0`; positive inputs and [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) produce `+0` |
| bounds `[+0, -0]` | every non-NaN input produces `-0` |

These outcomes follow from the ordered [extrema](../../../../glossary.md#saturation-pole-and-extremum) definition; they do not add [validation](../../../../glossary.md#term-validation). Existing
construction continues to accept either [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) bound order and NaN bounds because primitive
`>` is false for those cases. `clampMin` remains exactly [scalar](../../../../glossary.md#term-scalar) `MAX`, and `clampMax` remains
exactly scalar `MIN`.

### Data-type and representation boundary

- [FLOAT64](../../../../glossary.md#term-float64) and [FLOAT32](../../../../glossary.md#term-float32) use their represented [IEEE](../../../../glossary.md#term-ieee)-754 [binary64](../../../../glossary.md#floating-point-precision-and-rounding) and [binary32](../../../../glossary.md#floating-point-precision-and-rounding) values.
- BFLOAT16 uses the value represented by its current exact 16-bit storage; widening for the
  existing comparison domain does not introduce a new scalar conversion or rounding contract.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [widening glossary](../../../../glossary.md#term-widening); [scalar glossary](../../../../glossary.md#term-scalar); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) raw bits remain immutable operation metadata, including signed-zero and NaN
  payload bits.<span data-glossary-navigation="true"> ([raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [immutable glossary](../../../../glossary.md#term-immutable); [operation glossary](../../../../glossary.md#operation); [metadata glossary](../../../../glossary.md#term-metadata); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- This task adds no FLOAT16 data type and promises no NaN payload preservation, operand-source
  selection, instruction identity, intermediate precision, backend algorithm, or bitwise result
  beyond existing contracts.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [data type glossary](../../../../glossary.md#data-type--datatype); [identity glossary](../../../../glossary.md#term-identity-meanings); [precision glossary](../../../../glossary.md#term-precision); [backend glossary](../../../../glossary.md#backend))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- any Java declaration, executable statement, validation branch, failure message, operation kind,
  operation signature, attribute component, Tensor method, data type, promotion rule, helper
  method, or test behavior<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [operation glossary](../../../../glossary.md#operation); [operation signature glossary](../../../../glossary.md#operation-signature); [Tensor glossary](../../../../glossary.md#tensor); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- a `model.numerics` package, numerical-policy object, registry, service, selectable mode,
  generic evaluator, backend utility, runtime utility, or test-only fake evaluator<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- FLOAT16, implicit scalar conversion, a new scalar carrier, payload preservation, exact
  instruction selection, bitwise reproducibility, or backend tolerance<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [scalar glossary](../../../../glossary.md#term-scalar); [carrier glossary](../../../../glossary.md#term-storage-carrier); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- changes to reduction `MIN`/`MAX`, arg extrema, sorting/top-K, pooling, scatter reduction, or
  other operation families<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- constant folding, algebraic rewriting, canonicalization, graph capture, inference, validation,
  planning, lowering, execution, or backend conformance implementation<span data-glossary-navigation="true"> ([constant folding glossary](../../../../glossary.md#term-constant-folding); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [planning glossary](../../../../glossary.md#term-planning); [lowering glossary](../../../../glossary.md#lowering); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- gradient formulas or any derivative choice, including MIN/MAX ties, clamp endpoints, ABS/RELU
  zero, discontinuities, infinities, NaNs, singularities, or invalid-domain behavior<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [derivative glossary](../../../../glossary.md#term-derivative); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- changing `ARCHITECTURE.md`, an ADR, architecture documentation, module boundaries, dependencies,
  Gradle, Java 26 configuration, architecture tests, backend-conformance tests, integration tests,
  or another module's source/tests<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- creating a [Compiler](../../../../glossary.md#term-compiler) 0005A task file or promoting Compiler 0005A–0005E or 0006
- unrelated documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially model-owned operation
  semantics, compiler-owned gradient rules, and backend-owned execution<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capabilities](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Model capability and contract closure audit](../model-capability-contract-closure-audit.md)
- [Adjoint expressibility audit](../adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([Adjoint glossary](../../../../glossary.md#term-adjoint))</span>, especially E8–E17
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0018T](0018t-scalar-arithmetic-family-normalization.md)
- [Task 0018U](0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Task 0025](0025-canonical-tensor-producer-outputs.md)
- [Compiler master plan](../../compiler/master-plan.md)
- [Completed Compiler task 0005](../../compiler/tasks/0005-publication-planning-orchestration-and-compile-artifacts.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns backend-independent forward operation semantics. Recording the fixed
  comparison/extrema/clamp meaning in its existing semantic kinds and public API documentation is
  inside the current architecture.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The model layer does not evaluate Tensor values. No executable helper may be added merely to
  make these rules unit-testable.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Compiler owns derivative policy and gradient-expression construction. This task must not select
  or imply any derivative at a tie, endpoint, discontinuity, singularity, infinity, or NaN.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [derivative glossary](../../../../glossary.md#term-derivative); [gradient glossary](../../../../glossary.md#term-gradient); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan))</span>
- Forward and generated gradient expressions will later share this same fixed forward numerical
  contract. Compiler 0005A may choose local derivative conventions but may not add a separate
  forward comparison/extrema algebra.<span data-glossary-navigation="true"> ([local derivative glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Concrete backends later own evaluation and must be checked through backend-conformance tests
  when execution exists. This task does not implement or test a backend.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) remains the exact typed-bit metadata carrier. Its Java equality is intentionally
  distinct from numeric elementwise equality.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [carrier glossary](../../../../glossary.md#term-storage-carrier); [elementwise glossary](../../../../glossary.md#term-elementwise))</span>
- `ClampRangeAttrs` retains its current primitive represented-value validation and exact bound
  references. The task clarifies consequences; it does not change accepted or rejected ranges.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- No [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), package, architecture rule, [operation](../../../../glossary.md#operation) [shape](../../../../glossary.md#shape), or public surface changes.
- If accurate implementation requires executable semantics, a new type or member, a changed
  validation rule, or a derivative decision, stop and report the exact conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.operation.elementwise.comparison` — owns the six comparison
  meanings whose floating edge behavior becomes explicit.
- `io.github.pho001.synaptik.model.operation.elementwise.binary` — owns pairwise [Tensor](../../../../glossary.md#tensor)/Tensor
  `MIN` and `MAX`.
- `io.github.pho001.synaptik.model.operation.elementwise.scalar` — owns [scalar](../../../../glossary.md#term-scalar) `MIN`, scalar
  `MAX`, first-class `CLAMP`, and its exact typed attributes.
- `io.github.pho001.synaptik.model.tensor` — owns public Tensor method contracts and metadata-only
  expression construction.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- `io.github.pho001.synaptik.model.datatype` — owns unchanged [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) represented bits and
  current promotion contracts; review only.<span data-glossary-navigation="true"> ([represented bits glossary](../../../../glossary.md#term-represented-bits))</span>

Packages added, moved, or removed:

- None.

Type placement:

- No type is created or moved.
- Numerical meaning remains in the existing operation-family Javadocs and public Tensor contract.
  A general numerics package or policy object would create an unneeded abstraction and is
  forbidden by this task.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

## Affected files

Expected production Java, [Javadoc](../../../../glossary.md#term-javadoc)-only:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/comparison/BinaryComparisonKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/binary/BinaryArithmeticKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ScalarElementwiseKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/scalar/ClampRangeAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Expected documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification:

- [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), [`DataType`](../../../../glossary.md#data-type--datatype), `DataTypePromotion`, `ScalarValueAttrs`,
  `TensorBinaryExpressions`, `TensorScalarExpressions`, and `TensorComparisonExpressions` — their
  declarations, storage, promotion, validation, and metadata-construction behavior stay exact.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `BinaryComparisonKindTest`, `BinaryArithmeticKindTest`, `ScalarElementwiseSemanticsTest`,
  `ScalarValueTest`, `TensorBinaryComparisonTest`, `TensorBinaryArithmeticTest`, and
  `TensorScalarElementwiseTest` — existing tests lock the immutable declarations, attributes,
  type domains, construction, provenance, and accepted clamp bounds. They do not evaluate Tensor
  values and must not be expanded with a fake evaluator.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor))</span>
- model capabilities and both completed audits — they already record that forward numerical
  policy must be fixed while derivative policy remains compiler-owned; this task makes the
  directly consumed Javadocs and APIs agree with that closure-level statement.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [derivative glossary](../../../../glossary.md#term-derivative); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Training and Runtime APIs — no training request, optimizer, prepared state, runtime, or
  execution contract changes.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `ARCHITECTURE.md`, focused architecture documents, ADRs, and architecture tests — ownership and
  dependency rules do not change.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Gradle](../../../../glossary.md#term-gradle) files — the repository already uses Java 26 and current [module](../../../../glossary.md#term-module) dependencies remain exact.
- backend-conformance and integration tests — no backend or end-to-end execution exists in this
  task; future executable implementations must receive conformance coverage.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact twelve paths listed under expected production Java and expected documentation
and [planning](../../../../glossary.md#term-planning) may change: five [Javadoc](../../../../glossary.md#term-javadoc)-only Java paths and seven documentation/planning paths.

No test, [Gradle](../../../../glossary.md#term-gradle), architecture, capability/audit, other Java, other [module](../../../../glossary.md#term-module), [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), or
integration path may change. If another path is required for accuracy, stop and request a
replanning decision rather than expanding scope.

## Acceptance criteria

- Every floating comparison relation states the exact NaN and signed-zero result defined in this
  task; no tolerance or bit-equality ambiguity remains.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- `NOT_EQUAL` is documented as the exact logical complement of `EQUAL`.
- [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) and [scalar](../../../../glossary.md#term-scalar) floating `MIN`/`MAX` use one complete rule for NaN, infinities, opposite-
  signed zero, and otherwise equal or ordered values.<span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- Scalar Javadocs distinguish exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) bit retention from represented-value numerical
  comparison.
- First-class CLAMP is exactly ordered `MIN(MAX(input, minValue), maxValue)` and documents NaN
  input/bounds, equal bounds, both opposite-signed-zero endpoint orders, and the absence of stored
  intermediate producers.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span>
- `ClampRangeAttrs` retains and accurately documents its current acceptance of equal, signed-zero,
  infinity, and NaN bounds and its unchanged strict-inversion rejection.<span data-glossary-navigation="true"> ([infinity glossary](../../../../glossary.md#term-infinity))</span>
- FLOAT64, FLOAT32, and BFLOAT16 represented-value behavior is explicit; FLOAT16 and NaN payload
  preservation are not promised.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Integral comparison and [extrema](../../../../glossary.md#saturation-pole-and-extremum) semantics remain unchanged and exact.
- All Java changes are Javadoc-only. Declarations, imports, annotations, enum order, signatures,
  record components, executable statements, validation, messages, and bytecode-visible API remain
  unchanged.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [signatures glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation); [bytecode glossary](../../../../glossary.md#term-bytecode); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- No executable comparison/extrema/clamp helper or test evaluator is added. Existing focused tests
  remain unchanged and pass as regression evidence for the unmodified declarations and
  construction contracts.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- Tensor API, Compile API, glossary, task, model master plan, compiler master plan, and roadmap
  use consistent current/planned wording and do not imply execution or gradient completion.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- The clean documentation-focused implementation context records reasoned no-change conclusions
  for tests, capabilities/audits, Training/Runtime APIs, architecture/ADRs/tests, Gradle, other
  modules, backend conformance, and integration.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- After implementation, [Model](../../../../glossary.md#term-model) 0025A is `Complete`; [Compiler](../../../../glossary.md#term-compiler) 0005A remains `Draft` without a task
  file and depends on Model 0025A plus Compiler 0005; Compiler 0005B–0005E and 0006 remain `Draft`
  without task files.
- Exactly the twelve authorized paths change. Markdown links/anchors, fences, final newlines,
  trailing whitespace, status/dependency consistency, and <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>`git diff --check` pass.

## Tests / [validation](../../../../glossary.md#term-validation)

This task changes semantic contracts in Javadocs and explanatory documentation but no executable
Java behavior. A single clean documentation-focused implementation context is the required
documentation pass: it is separate from this [planning](../../../../glossary.md#term-planning) context, reads the final source/tests
directly, and finalizes the Javadocs and documentation itself. Do not create a redundant second
documentation context solely to repeat that work.

[Run](../../../../glossary.md#term-run) one final [model](../../../../glossary.md#term-model) [compile](../../../../glossary.md#compile)/test/[Javadoc](../../../../glossary.md#term-javadoc) command after all Javadocs are stable:

```bash
./gradlew :modules:model:compileJava :modules:model:test :modules:model:javadoc
```

The [module](../../../../glossary.md#term-module) test is [regression](../../../../glossary.md#term-regression) evidence for unchanged declaration, attribute, validation, public
surface, [descriptor](../../../../glossary.md#term-descriptor), and [provenance](../../../../glossary.md#provenance) behavior. It is not evidence that [Tensor](../../../../glossary.md#tensor) values were
evaluated. No focused semantic execution test is appropriate because the model has no evaluator
and no [immutable](../../../../glossary.md#term-immutable) declaration member changes; adding one would test a fake implementation rather
than the model contract.

Then run:

```bash
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

If the temporary Markdown validator is absent, create an equivalent validator outside the
repository. It must check repository-local [targets](../../../../glossary.md#term-target) and heading anchors, balanced fences, final
newlines, and trailing whitespace.

Required manual/source checks:

- inspect every Java diff hunk and confirm that only Javadoc comments changed;
- inspect generated Javadoc for all five affected production types and the relevant Tensor
  methods;
- verify the exact twelve-path ceiling and absence of test/[Gradle](../../../../glossary.md#term-gradle)/architecture changes;
- verify Model 0025A status and [Compiler](../../../../glossary.md#term-compiler) 0005A [dependency](../../../../glossary.md#term-dependency)/status in all three planning indexes;
- verify that no Compiler 0005A task file exists and that 0005B–0005E/0006 remain Draft without
  task files;
- verify no `model.numerics`, policy, registry, selectable mode, operation, attribute, Tensor
  method, data type, FLOAT16, evaluator, or derivative convention was added; and<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [operation glossary](../../../../glossary.md#operation); [data type glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- verify terminology, examples/tables, links, anchors, fences, final newlines, trailing
  whitespace, and `git diff --check`.

Repository-wide and architecture validation are deferred to [CI](../../../../glossary.md#term-continuous-integration-ci) because this is one module's
Javadoc plus explanatory/planning documentation and changes no executable behavior, dependency,
build configuration, or architecture boundary.

## Dependencies

- [Model](../../../../glossary.md#term-model) 0018N [typed scalar value](../../../../glossary.md#typed-scalar-value--scalarvalue) contract — Complete.
- Model 0018T scalar arithmetic family normalization and selected extrema/clamp meaning —
  Complete.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [normalization glossary](../../../../glossary.md#term-normalization); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Model 0018U integral arithmetic/comparison expansion — Complete and unchanged by this task.
- Model 0025 [canonical](../../../../glossary.md#term-canonical-form) [producer](../../../../glossary.md#term-producer) outputs — Complete and the immediate model [planning](../../../../glossary.md#term-planning) predecessor.
- Compiler 0005 publication/planning/artifacts — Complete; Compiler 0005A is the downstream
  consumer whose planning remains blocked until this task is implemented.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [publication glossary](../../../../glossary.md#term-publication); [artifacts glossary](../../../../glossary.md#term-artifact))</span>

## Follow-up tasks

- Compiler 0005A — after Model 0025A is Complete, separately plan and implement compiler-owned
  elementwise/activation gradient completion. It must choose MIN/MAX tie routing or sharing,
  clamp endpoint behavior, ABS/RELU zero conventions, and singular/exceptional formula behavior.
  This task chooses none of those derivative policies.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model); [elementwise glossary](../../../../glossary.md#term-elementwise); [activation glossary](../../../../glossary.md#term-activation); [gradient glossary](../../../../glossary.md#term-gradient); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Future concrete backends — when comparison/extrema/clamp execution is implemented, add or update
  backend-conformance tests for NaN, infinities, both zero signs, equal candidates, scalar exact
  values, NaN bounds, equal bounds, and both signed-zero clamp endpoint orders.<span data-glossary-navigation="true"> ([backends glossary](../../../../glossary.md#backend); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [conformance glossary](../../../../glossary.md#term-conformance); [NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [scalar glossary](../../../../glossary.md#term-scalar); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>

Do not create either follow-up specification during this task.

## Architecture impact

Expected impact: None.

The architecture already assigns forward [operation](../../../../glossary.md#operation) semantics to [model](../../../../glossary.md#term-model), [compiler](../../../../glossary.md#term-compiler)-owned [gradient](../../../../glossary.md#term-gradient)
rules to compiler, and concrete [evaluation](../../../../glossary.md#term-evaluation) to [backends](../../../../glossary.md#backend). This task makes existing documentation
complete without changing [ownership](../../../../glossary.md#term-ownership-meanings) or dependencies.

If implementation requires an architecture change, executable model evaluation, another [module](../../../../glossary.md#term-module),
or a [derivative](../../../../glossary.md#term-derivative) decision, stop and report the exact conflict.

## Implementation prompt

Use this prompt in one separate clean documentation-focused task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, the focused model/compiler/backend-conformance boundary docs,
documentation rules plus the General, API/Javadoc, Planning, and Example profiles, planning guide
and roadmap, model capabilities/master plan, closure and adjoint audits E8-E17, completed Model
0018N/0018T/0018U/0025, compiler master plan and completed Compiler 0005, this task, all five
affected production files, the review-only helpers/tests, Tensor/Compile/Training/Runtime APIs,
glossary, and Java 26/module Gradle boundaries.

Implement Model 0025A exactly within its twelve authorized paths. Change Java only in Javadoc.
Finalize the fixed portable floating comparison, pairwise/scalar MIN/MAX, and ordered first-class
CLAMP semantics without adding an evaluator, policy object, operation, attribute, Tensor method,
data type, test change, derivative convention, compiler behavior, backend behavior, or execution.
Preserve current ClampRangeAttrs validation and exact ScalarValue bit storage. Stop on any scope,
behavior, or architecture conflict.

This clean documentation-focused context itself is the required documentation pass; do not create
a redundant second pass. Run the specified model compile/test/Javadoc and documentation/scope/
status checks, inspect generated Javadoc, record exact evidence and no-change conclusions, then
mark Model 0025A Complete only if every acceptance criterion passes. Keep Compiler 0005A Draft
without a task file, dependent on completed Model 0025A and Compiler 0005; keep 0005B-0005E and
0006 Draft without task files.
```

## Local decisions

- Ordinary represented numeric comparison is the sole portable floating relation. No total-order,
  raw-bit, approximate, tolerance, or selectable-policy alternative was added.<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- `NOT_EQUAL` is the exact logical complement of numeric `EQUAL`. This supplies the NaN and
  opposite-signed-zero outcomes without a second independent policy.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Binary and scalar MIN/MAX now use the same complete NaN, infinity, signed-zero, unequal-value,
  and equal-nonzero contract. No payload or source-operand identity becomes observable.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [infinity glossary](../../../../glossary.md#term-infinity); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [Scalar](../../../../glossary.md#term-scalar) numerical comparison uses the exact same-typed [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) represented value.
  `ScalarValue` raw-bit equality and hashing remain distinct [immutable](../../../../glossary.md#term-immutable) [metadata](../../../../glossary.md#term-metadata) behavior.
- First-class CLAMP remains one [operation](../../../../glossary.md#operation) with ordered
  `MIN(MAX(input, minValue), maxValue)` value meaning and no stored intermediate producers.
  Existing primitive bound validation remains unchanged.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- The model intentionally has no evaluator. The existing API/metadata tests, source-only checks,
  model compilation, generated Javadoc, and documentation validation are the correct evidence
  boundary.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- This clean documentation-focused implementation context performed the required final Javadoc,
  API, glossary, and planning pass. No second documentation context or duplicate test run was
  needed.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [run glossary](../../../../glossary.md#term-run))</span>

## Known limitations

- The completed contract specifies forward values, not an algorithm, tolerance, NaN payload,
  operand identity, instruction sequence, bitwise cross-backend result, or backend availability.<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [identity glossary](../../../../glossary.md#term-identity-meanings); [backend glossary](../../../../glossary.md#backend))</span>
- Reduction extrema, arg extrema, sort/top-K, pooling, scatter extrema, and other comparison-using
  families remain outside this focused task.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- Compiler 0005A remains Draft without a task specification and must still choose derivative tie,
  endpoint, discontinuity, singularity, and exceptional-value policies.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Numerical execution cannot be verified until a concrete backend exists. That future
  implementation must receive backend-conformance coverage.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- `./gradlew :modules:model:compileJava :modules:model:test :modules:model:javadoc` passed with
  `BUILD SUCCESSFUL`. The model report contains 127 suites and 1,018 tests with zero failures,
  errors, or skipped tests.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed after the semantic edits with 12 Markdown
  files and 681 repository-local links checked. The final planning/status edit was validated again
  before completion.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Generated Javadoc exists for all five affected types. Manual inspection found the four ordered
  NaN/signed-zero outcomes, exact equality and complement wording, both extrema zero selections,
  exact ordered CLAMP composition, and the signed-zero clamp endpoint consequences in the rendered
  HTML.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- A mechanical before/after comparison removed Javadoc blocks from each affected Java source and
  found every remaining byte identical. Imports, annotations, declarations, enum order,
  signatures, record components, executable statements, messages, and bytecode-visible API
  therefore remain unchanged.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [bytecode glossary](../../../../glossary.md#term-bytecode); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `git diff --check` passed. The union of tracked and untracked changed paths is exactly the five
  authorized Javadoc-only Java files and seven authorized documentation/planning files.
- Manual semantic review confirmed all six floating comparison truth rules, ordinary infinity
  order, NaN propagation, MIN negative-zero selection, MAX positive-zero selection, equal-nonzero
  wording, same-typed scalar represented-value comparison, all four required CLAMP cases, and the
  unchanged signed-integral contracts.<span data-glossary-navigation="true"> ([infinity glossary](../../../../glossary.md#term-infinity); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Planning review confirmed Model 0025A Complete in its task, model master plan, and roadmap.
  Compiler 0005A still depends on completed Model 0025A plus Compiler 0005 and remains Draft;
  Compiler 0005B–0005E and 0006 also remain Draft. No task specification exists for any of those
  compiler rows.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>

Reasoned no-change conclusions:

- The seven named model tests remain unchanged. They already lock declarations, exact scalar
  metadata, range validation, domains, descriptors, provenance, and the public surface; adding a
  fake value evaluator would test behavior the model does not implement.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Model capabilities and both completed audits remain unchanged. They already assign fixed forward
  semantics to model and derivative choices to compiler.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative))</span>
- Training and Runtime APIs remain unchanged because no training request, optimizer, prepared
  state, runtime, or execution contract changed. Compile API alone needed its prerequisite-status
  correction.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [optimizer glossary](../../../../glossary.md#term-optimizer); [Compile glossary](../../../../glossary.md#compile))</span>
- `ARCHITECTURE.md`, focused architecture documentation, ADRs, architecture tests, module
  boundaries, dependencies, Gradle, and Java 26 configuration remain unchanged because ownership,
  structure, and executable behavior did not change.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- No other module source or tests changed. Backend-conformance and integration tests remain
  unchanged because this task adds no backend or end-to-end execution.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- Updated only [Javadoc](../../../../glossary.md#term-javadoc) in `BinaryComparisonKind`, `BinaryArithmeticKind`,
  `ScalarElementwiseKind`, `ClampRangeAttrs`, and [`Tensor`](../../../../glossary.md#tensor).
- Finalized current represented-value behavior in the Tensor API, updated the Compile API from an
  incomplete-model prerequisite to deferred compiler derivative policy, and synchronized glossary
  terminology.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Synchronized this task, the model and compiler master plans, and the roadmap. Compiler
  0005A–0005E and 0006 were not promoted and no later task specification was created.
- Added no declaration, evaluator, numerical-policy object, operation, attribute, Tensor method,
  data type, FLOAT16 support, test, derivative convention, compiler implementation, backend
  behavior, dependency, or build change.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [data type glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

Completed the portable represented-value contracts for all floating comparisons, [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) and [scalar](../../../../glossary.md#term-scalar)
MIN/MAX, and ordered first-class CLAMP. Finalized the five affected Javadocs, [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), [Compile](../../../../glossary.md#compile)
API, glossary, and three [planning](../../../../glossary.md#term-planning) indexes without changing executable Java or any public surface.

Files changed: the exact five [Javadoc](../../../../glossary.md#term-javadoc)-only Java files and seven documentation/planning files listed
under [Affected files](#affected-files).

[Validation](../../../../glossary.md#term-validation): the final [model](../../../../glossary.md#term-model) compile/test/Javadoc command passed; 127 suites and 1,018 tests passed;
generated Javadoc was inspected; Markdown, local links/anchors, fences, final newlines, trailing
whitespace, exact [scope](../../../../glossary.md#term-scope), Java-comment isolation, status/dependencies, absence of later [compiler](../../../../glossary.md#term-compiler)
specifications, semantic wording, and `git diff --check` passed.

Documentation review: Tensor API, Compile API, glossary, and planning required updates. Model
capabilities/audits, Training/[Runtime](../../../../glossary.md#term-runtime) APIs, architecture/ADRs/tests, [Gradle](../../../../glossary.md#term-gradle), other [modules](../../../../glossary.md#term-module),
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration required no change for the reasons recorded above.

Unresolved issues: None within task scope.

Required follow-up: Compiler 0005A may be separately planned to choose [derivative](../../../../glossary.md#term-derivative) policy while
reusing this fixed forward contract. Future backends must receive conformance coverage when they
implement these operations.

Status: Complete
