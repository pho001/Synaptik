# Task 0005G: [Extrema](../../../../glossary.md#saturation-pole-and-extremum), Clamp, [Tensor](../../../../glossary.md#tensor) Power, and Logical Coverage

## Status

Complete

## Goal

Close the remaining exact algebraic and [boolean](../../../../glossary.md#data-type--datatype) [pointwise](../../../../glossary.md#term-pointwise) rows that can be implemented from the
current [Model](../../../../glossary.md#term-model) contracts without inventing a conversion or transcendental-accuracy policy:

```text
binary MIN / MAX / POW
scalar MIN / MAX / CLAMP
boolean AND / OR / NOT
  -> existing canonical CPU pointwise IR
  -> existing scalar or parallel-scalar generated execution
  -> one compatible generated artifact and one partition executable
```

Add same-typed [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `POW`; same-typed FLOAT32/FLOAT64/[INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) binary and
[scalar](../../../../glossary.md#term-scalar) `MIN`/`MAX`; same-typed FLOAT32/FLOAT64 first-class range `CLAMP`; and [canonical](../../../../glossary.md#term-canonical-form)-[BOOL](../../../../glossary.md#data-type--datatype)
`AND`/`OR`/`NOT`. Preserve the exact Model [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence), attributes, input ordering,
[descriptor](../../../../glossary.md#term-descriptor), [graph](../../../../glossary.md#graph) [identity](../../../../glossary.md#term-identity-meanings), [publication](../../../../glossary.md#term-publication) obligations, and one-through-eight pointwise-[fusion](../../../../glossary.md#term-fusion)
contract. This is [backend](../../../../glossary.md#backend) realization, not graph rewriting or semantic [decomposition](../../../../glossary.md#term-decomposition).

The task deliberately does not absorb the nineteen-value [unary](../../../../glossary.md#term-unary) family. [CPU](../../../../glossary.md#term-cpu) 0005H remains the
next Draft frontier for the remaining unary, transcendental, and [activation](../../../../glossary.md#term-activation) rows because their
algorithm, special-value, and [conformance](../../../../glossary.md#term-conformance)-[tolerance](../../../../glossary.md#term-tolerance) [matrix](../../../../glossary.md#term-matrix-meanings) is a separate cohesive decision. The
task also keeps cross-type `CAST` [fail-closed](../../../../glossary.md#term-fail-closed): the current Model contract permits construction but
does not yet define numerical conversion, so a CPU backend must not invent it.

## [Scope](../../../../glossary.md#term-scope)

### Exact supported [matrix](../../../../glossary.md#term-matrix-meanings)

- Extend `BinaryArithmeticKind.MIN` and `MAX` only when:
  - the [operation](../../../../glossary.md#operation) has exactly two inputs and one output;
  - attributes are exactly `NoOperationAttrs.INSTANCE`;
  - both inputs and the output have one exact common type from [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [INT32](../../../../glossary.md#data-type--datatype), or [INT64](../../../../glossary.md#data-type--datatype);
  - the output [Shape](../../../../glossary.md#shape) is the exact ordinary right-aligned broadcast of the two input Shapes; and
  - every [descriptor](../../../../glossary.md#term-descriptor) is fully static with a resolved [layout](../../../../glossary.md#layout) under the existing [CPU](../../../../glossary.md#term-cpu) boundary.
- Extend `ScalarElementwiseKind.MIN` and `MAX` only when:
  - the operation has exactly one input and one output;
  - attributes are exactly `ScalarValueAttrs`;
  - input, output, and scalar value have one exact common type from FLOAT64, FLOAT32, INT32, or
    INT64;<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
  - output Shape equals input Shape; and
  - every descriptor is fully static with a resolved layout.
- Extend first-class `ScalarElementwiseKind.CLAMP` only when:
  - the operation has exactly one input and one output;
  - attributes are exactly `ClampRangeAttrs`;
  - input, output, lower bound, and upper bound have one exact common type, FLOAT32 or FLOAT64;
  - output Shape equals input Shape; and
  - every descriptor is fully static with a resolved layout.
- Extend `BinaryArithmeticKind.POW` only when:
  - the operation has exactly two inputs and one output;
  - attributes are exactly `NoOperationAttrs.INSTANCE`;
  - both inputs and the output have one exact common type, FLOAT32 or FLOAT64;
  - the output Shape is the exact ordinary right-aligned broadcast of the base and exponent
    Shapes; and<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent))</span>
  - every descriptor is fully static with a resolved layout.
- Extend `BooleanLogicalKind.AND` and `OR` only when:
  - the operation has exactly two inputs and one output;
  - attributes are exactly `NoOperationAttrs.INSTANCE`;
  - both inputs and output have exact [BOOL](../../../../glossary.md#data-type--datatype) type;
  - the output Shape is the exact ordinary right-aligned broadcast of the two input Shapes; and
  - every descriptor is fully static with a resolved layout.
- Extend `BooleanLogicalKind.NOT` only when:
  - the operation has exactly one input and one output;
  - attributes are exactly `NoOperationAttrs.INSTANCE`;
  - input and output have exact BOOL type and equal Shapes; and
  - every descriptor is fully static with a resolved layout.
- Capability reporting returns `false` for BFLOAT16, every mixed-type row, integral or BOOL power,
  integral range clamp, numeric truthiness, malformed attributes/arity/result, unresolved layout,
  and dynamic Shape. Complete-partition lowering rejects an unsupported occurrence before
  declarations, specialization, artifact lookup, or executable construction.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [arity glossary](../../../../glossary.md#term-arity); [partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [occurrence glossary](../../../../glossary.md#term-occurrence); [specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup))</span>

### Exact [extrema](../../../../glossary.md#saturation-pole-and-extremum) and clamp semantics

- [FLOAT64](../../../../glossary.md#term-float64) and [FLOAT32](../../../../glossary.md#term-float32) [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/[scalar](../../../../glossary.md#term-scalar) `MIN` and `MAX` implement the completed [Model](../../../../glossary.md#term-model) 0025A contract:
  - if either candidate is NaN, the result is NaN without a payload, sign, selected-source, or
    bitwise-result promise;<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
  - opposite signed zeros produce [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) for `MIN` and [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) for `MAX`, independent
    of operand order;
  - [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) and unequal non-[NaN](../../../../glossary.md#term-nan) values use ordinary represented numeric order; and
  - equal nonzero candidates produce that numeric value without identifying one operand.
- [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype) `MIN`/`MAX` use exact signed order without conversion, widening, saturation, or
  overflow behavior.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [saturation glossary](../../../../glossary.md#term-saturation); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- `CLAMP(input, lower, upper)` executes exactly the ordered represented-value meaning
  `MIN(MAX(input, lower), upper)` while remaining one `CLAMP` opcode and one Model occurrence.
  The generated calculation evaluates the lower comparison first and the upper comparison second.
  It must reproduce the completed Model contract for NaN bounds, NaN input, infinities, equal
  bounds, <span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`[-0,+0]`, and `[+0,-0]`.
- Scalar and clamp attributes retain exact raw FLOAT64/FLOAT32/INT32/INT64 bits in CPU IR. The
  backend does not recreate, normalize, widen, stringify, or reinterpret a <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [backend glossary](../../../../glossary.md#backend))</span>[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue).
- Scalar generated and reference execution use the applicable primitive/`Math.min`/`Math.max`
  behavior or an equivalent implementation proved against those fixed represented-value results.
  They may not use a simple comparison/select sequence that loses NaN propagation or signed-zero
  selection.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>

### [Tensor](../../../../glossary.md#tensor) power semantics

- [FLOAT64](../../../../glossary.md#term-float64) [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `POW` evaluates `StrictMath.pow(base, exponent)` for every element after existing
  right-aligned access/broadcast resolution.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- [FLOAT32](../../../../glossary.md#term-float32) binary `POW` exactly widens the represented binary32 base and exponent to binary64,
  evaluates <span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`StrictMath.pow`, and narrows once to FLOAT32. This is a CPU-private realization of a
  same-typed FLOAT32 request, not Model promotion or cross-type conversion.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model))</span>
- Binary `POW` preserves the same exact/default numerical boundary selected for direct [scalar](../../../../glossary.md#term-scalar)
  `POW` by CPU 0005F: no relaxed approximation, reassociation, exponent classification, multiply
  chain, reciprocal replacement, square-root replacement, finite-only assumption, flush-to-zero,
  contraction, or changed exceptional-value rule.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [finite glossary](../../../../glossary.md#term-finite); [flush-to-zero glossary](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- [Runtime](../../../../glossary.md#term-runtime) [exponent](../../../../glossary.md#term-exponent) values are never inspected during [prepare](../../../../glossary.md#prepare). `POW` therefore always remains one
  direct binary instruction. CPU 0005F's `CpuScalarPowerAnalysis` continues to apply only to exact
  scalar attributes and is not generalized to Tensor inputs.
- Generated/reference tests compare exact special cases where the contract fixes them and use the
  direct `StrictMath.pow` contract for ordinary finite accuracy. They classify NaN without
  inventing a payload promise.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>

### [Canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) semantics

- `AND` and `OR` consume ordered canonical BOOL bytes and produce canonical BOOL bytes according
  to conjunction and disjunction. `NOT` produces the exact complement.
- The existing cold binding validation remains the sole canonical-BOOL boundary check. Internal
  comparison/classification/logical results are already canonical and stay canonical through
  fusion. The generated hot loop performs no numeric-truthiness conversion or repeated validity
  check.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) logical operations use the existing right-aligned broadcast access plans. [Unary](../../../../glossary.md#term-unary) `NOT`
  preserves Shape. None short-circuits, reorders operands, collapses repeated expressions, or
  changes graph topology.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [graph glossary](../../../../glossary.md#graph); [topology glossary](../../../../glossary.md#term-topology))</span>

### Common [IR](../../../../glossary.md#term-intermediate-representation-ir) and [lowering](../../../../glossary.md#lowering)

- Add exactly these nine family-oriented values to `CpuPointwiseOpcode`, growing the closed
  vocabulary from twenty-two to thirty-one:

```text
MIN
MAX
POW
SCALAR_MIN
SCALAR_MAX
SCALAR_CLAMP
LOGICAL_AND
LOGICAL_OR
LOGICAL_NOT
```

- Keep the existing opcode order grouped by binary arithmetic, scalar arithmetic/range, unary,
  classification, comparison, logical, selection, and cast families. Add only the minimum family
  metadata needed to validate BOOL logical inputs/results and the two-bound clamp immediate.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [unary glossary](../../../../glossary.md#term-unary); [metadata glossary](../../../../glossary.md#term-metadata); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Extend `CpuKernelIr.Instruction` with one optional [immutable](../../../../glossary.md#term-immutable) `ClampImmediate` fact containing
  exact lower and upper `ScalarImmediate` values. Exactly `SCALAR_CLAMP` requires it and forbids
  the ordinary one-value scalar immediate and power realization. Existing scalar opcodes retain
  exactly one `ScalarImmediate`; exactly `SCALAR_POW` additionally retains one
  `PowerRealization`; parameterless operations retain neither.
- `ClampImmediate` validates two non-null values with the same exact FLOAT32 or FLOAT64 type. IR
  validation also proves that this type equals input/output type. It does not repeat<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
  `ClampRangeAttrs` ordering [validation](../../../../glossary.md#term-validation) or expose another public [scalar](../../../../glossary.md#term-scalar)/range abstraction.
- Extend common lowering with direct kind-to-opcode mapping and exact attribute extraction. One
  first-class CLAMP node lowers to one first-class <span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>`SCALAR_CLAMP` instruction; it is not expanded
  into two graph nodes or two IR instructions.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Preserve topology-local values, exact stored node/instruction order, one output per occurrence,
  virtual intermediates, one final materialized store, and the existing one-through-eight graph-
  occurrence fusion limit. Each new Model occurrence counts as exactly one instruction.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [occurrence glossary](../../../../glossary.md#term-occurrence); [fusion glossary](../../../../glossary.md#term-fusion); [Model glossary](../../../../glossary.md#term-model))</span>
- Keep same-type `CAST` exactly as implemented. Every cross-type cast remains unsupported because
  the current Model contract deliberately supplies no numerical conversion policy.

### Generated and reference execution

- Extend the existing family-grouped scalar emitter and scalar reference kernel. Do not add one
  class, emitter, lowerer, executable, route, or planner per operation.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [operation glossary](../../../../glossary.md#operation))</span>
- [Scalar](../../../../glossary.md#term-scalar) and parallel-scalar execute every admitted row for both [heap](../../../../glossary.md#term-heap) and [`MemorySegment`](../../../../glossary.md#term-memorysegment)
  carriers, mixed boundary patterns, all five access regimes, arbitrary start/end ranges,
  offsets, positive/zero strides, and right-aligned broadcasting where applicable.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- All nine new opcodes are vector-ineligible in this bounded task. Binary Tensor power has no
  exact Vector API implementation selected; BOOL byte-vector infrastructure does not exist; and
  extrema/clamp need a separate proof that a vector realization preserves NaN and signed-zero
  semantics. Supported work therefore selects scalar or parallel-scalar rather than failing.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [Tensor glossary](../../../../glossary.md#tensor); [Vector API glossary](../../../../glossary.md#term-vector-api); [BOOL glossary](../../../../glossary.md#data-type--datatype); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- This bounded choice does not remove the existing FLOAT64 vector/parallel-vector route for
  already supported opcodes and proved scalar-power special realizations. A future performance
  task may add a vector row only with exact edge-conformance evidence; support is not implied here.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The scalar reference remains an already-lowered cold conformance oracle. It does not inspect
  Model operations in Runtime or become a second compiler.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [Model glossary](../../../../glossary.md#term-model); [Runtime glossary](../../../../glossary.md#term-runtime); [compiler glossary](../../../../glossary.md#term-compiler))</span>

### [Fusion](../../../../glossary.md#term-fusion), access, [materialization](../../../../glossary.md#materialization), and [lifecycle](../../../../glossary.md#lifecycle) preservation

- Preserve CPU 0005A's connected straight-line one-through-eight pointwise partition and one
  generated class/executable. Numeric and logical chains may fuse when their ordinary typed value
  edges and the existing topology rules permit it; the task adds no mixed-family exception.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [pointwise glossary](../../../../glossary.md#term-pointwise); [partition glossary](../../../../glossary.md#partition); [topology glossary](../../../../glossary.md#term-topology))</span>
- Preserve CPU 0005B's five access regimes, checked spans, complete write-injectivity decision,
  heap/segment/mixed carriers, arbitrary ranges, and broadcast odometers.<span data-glossary-navigation="true"> ([injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [heap glossary](../../../../glossary.md#term-heap))</span>
- Preserve CPU 0005D's direct plus at most three eligible one-input materialization candidates,
  at most one realized FLOAT64 contiguous copy, four-candidate ceiling, one-artifact budget, zero
  fixed-shape variants, and zero unrolled variants. New non-vector opcodes do not create another
  materialization or specialization family.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [contiguous glossary](../../../../glossary.md#term-contiguous); [artifact glossary](../../../../glossary.md#term-artifact); [shape glossary](../../../../glossary.md#shape); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [vector glossary](../../../../glossary.md#term-vector); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Preserve declarations-before-binding, strong prepared ownership, concurrent-run isolation,
  worker/chunk behavior, no hot-path operation dispatch, and direct Runtime invocation of one
  prepared executable.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [run glossary](../../../../glossary.md#term-run); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch); [Runtime glossary](../../../../glossary.md#term-runtime); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>

### [Specialization](../../../../glossary.md#term-specialization), [schema](../../../../glossary.md#term-schema), and optional [persistence](../../../../glossary.md#term-persistence)

- Canonical IR structural identity includes each new opcode, exact ordinary scalar immediate, and
  exact two-bound clamp type/raw bits. Changing either clamp bound changes the lowering
  fingerprint. Runtime slots, addresses, concrete carriers, and extents remain instance facts.<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings); [opcode glossary](../../../../glossary.md#term-opcode); [scalar glossary](../../../../glossary.md#term-scalar); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [lowering glossary](../../../../glossary.md#lowering); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Bump `CpuGeneratorSchema.CURRENT_VERSION` from 6 to 7 because the opcode vocabulary,
  instruction shape, scalar emission, and generated bytecode change. Current-schema hits remain
  eligible; every older schema entry is an incompatible miss, with no migration reader.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Existing specialization compatibility continues to carry the lowering fingerprint, numerical
  mode, route/strategy, carrier/access/materialization facts, and scalar-power realizations.
  Clamp bounds require no duplicate specialization field because exact immutable constants already
  participate in the canonical IR fingerprint, just as ordinary scalar immediates do.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [carrier glossary](../../../../glossary.md#term-storage-carrier); [materialization glossary](../../../../glossary.md#materialization); [immutable glossary](../../../../glossary.md#term-immutable); [constants glossary](../../../../glossary.md#term-constant))</span>
- Preserve process-local compatible interning, complete structural verification, optional bounded
  current-schema persistence, assignment-before-artifact order, and the CPU 0005D<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [CPU glossary](../../../../glossary.md#term-cpu))</span>
  `KEEP_DISABLED` default. Do not rerun the timing benchmark or enable persistent class storage.

## Out of [scope](../../../../glossary.md#term-scope)

- any [unary](../../../../glossary.md#term-unary)/transcendental/[activation](../../../../glossary.md#term-activation) row beyond the already implemented `NEG` and exact [FLOAT64](../../../../glossary.md#term-float64)
  [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu); [CPU](../../../../glossary.md#term-cpu) 0005H owns their later bounded completion
- cross-type CAST, BFLOAT16 arithmetic, FLOAT16, mixed-type CPU arithmetic, CPU-side promotion,
  scalar conversion, integral DIV/POW/range CLAMP, numeric truthiness, bitwise operations, or
  another data type<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [scalar glossary](../../../../glossary.md#term-scalar); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- Vector API implementation for any new opcode, byte/BOOL vectors, per-lane scalarization hidden
  inside a vector loop, a vector approximation, or a new vector policy<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api); [opcode glossary](../../../../glossary.md#term-opcode); [BOOL glossary](../../../../glossary.md#data-type--datatype); [vectors glossary](../../../../glossary.md#term-vector); [lane glossary](../../../../glossary.md#term-lane))</span>
- graph canonicalization, algebraic rewriting, constant folding, CLAMP decomposition, power
  exponent discovery, operation reordering, a new producer/value, or publication remapping<span data-glossary-navigation="true"> ([graph canonicalization glossary](../../../../glossary.md#graph-canonicalization); [constant folding glossary](../../../../glossary.md#term-constant-folding); [decomposition glossary](../../../../glossary.md#term-decomposition); [exponent glossary](../../../../glossary.md#term-exponent); [operation glossary](../../../../glossary.md#operation); [producer glossary](../../../../glossary.md#term-producer); [publication glossary](../../../../glossary.md#term-publication))</span>
- fast math, relaxed numerical permission, configurable NaN/zero behavior, a broad numerical
  policy, accuracy mode, registry, service locator, or string dispatch<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- a per-operation lowerer/emitter/kernel/executable class, a second backend compiler, a public CPU
  orchestration facade, another route, or another backend identity<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [backend glossary](../../../../glossary.md#backend); [compiler glossary](../../../../glossary.md#term-compiler); [facade glossary](../../../../glossary.md#term-facade); [backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>
- reductions, scans, layout/indexing/random, linear algebra, convolution, normalization, loss,
  gradient-specific behavior, compiler adoption, or later CPU tasks<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [linear glossary](../../../../glossary.md#term-linear-meanings); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [normalization glossary](../../../../glossary.md#term-normalization); [loss glossary](../../../../glossary.md#loss); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Tensor storage inspection, eager evaluation, Runtime operation interpretation, physical-layout
  changes, dependency/Gradle/build changes, architecture changes, or another module's source/tests<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [eager evaluation glossary](../../../../glossary.md#term-eager-evaluation); [Runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Backend-owned lowering ADR](../../../../design/decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([Backend-owned lowering glossary](../../../../glossary.md#backend-owned-lowering); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Backend-guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Task 0005A](0005a-atomic-partition-kernel-architecture-reset.md)
- [Task 0005B](0005b-universal-access-plans-and-right-aligned-broadcasting.md)
- [Task 0005C](0005c-vector-and-parallel-portable-strategies.md)
- [Task 0005D](0005d-materialization-specialization-and-persistence-evidence-gate.md)
- [Task 0005E](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
- [Task 0005F](0005f-floating-division-and-exact-scalar-power-realization.md)
- [Model task 0018T](../../../modules/model/tasks/0018t-scalar-arithmetic-family-normalization.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model task 0018U](../../../modules/model/tasks/0018u-integral-elementwise-arithmetic-and-comparisons.md)
- [Model task 0025A](../../../modules/model/tasks/0025a-portable-floating-comparison-extrema-and-clamp-semantics.md)

## Architecture constraints

- Planning owns backend assignment and supplies CPU-owned partitions. CPU prepare may perform
  backend-private lowering, exact numerical eligibility, specialization, artifact lookup, and
  executable construction; those decisions do not escape as another shared IR.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Generated code and [Runtime](../../../../glossary.md#term-runtime) never consume [`Operation`](../../../../glossary.md#operation), `CompiledNode`, Model attributes, or
  string dispatch. Common CPU lowering translates them once into typed CPU-private IR.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Model owns the fixed forward semantics. This task realizes the current exact extrema/clamp,
  power, and logical meanings and may not strengthen or replace them.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Compiler owns graph rewrites and gradient construction. CPU keeps every semantic occurrence and
  executes forward and compiler-generated occurrences through the same policy.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Config owns future relaxed numerical permission. This task remains exact/default and reads no
  Config policy in the generated hot path.<span data-glossary-navigation="true"> ([hot path glossary](../../../../glossary.md#term-hot-path))</span>
- Shared Prepare continues to see one opaque CPU plan plus exact declarations; Runtime binds
  resources and invokes one prepared executable without route, operation, Shape, or layout
  decisions.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- If implementation requires a Model conversion rule, another public/shared type, another module,
  changed architecture boundary, or an operation-specific class hierarchy, stop and report the
  conflict.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — truthful [occurrence](../../../../glossary.md#term-occurrence)-local capability reporting.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — closed family opcodes, exact typed scalar
  and clamp immediates, topology-local values, and structural identity.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [topology glossary](../../../../glossary.md#term-topology); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — one common Model-occurrence-to-IR
  translation and access/boundary derivation.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — family-grouped scalar bytecode
  emission and existing generated-class validation.<span data-glossary-navigation="true"> ([bytecode emission glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — cold already-lowered conformance
  realization.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — generator [schema](../../../../glossary.md#term-schema) compatibility.
- Existing prepare, route, executable, memory, and cache packages are reviewed and changed only
  if the new validated IR facts must flow through an existing contract.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [cache glossary](../../../../glossary.md#term-cache))</span>

Packages added or changed:

- No package is added and no responsibility moves.

Type placement:

- `CpuKernelIr.ClampImmediate` — nested immutable two-bound fact because it belongs to exactly one
  canonical IR instruction and does not justify a public or package-level abstraction.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- The nine new `CpuPointwiseOpcode` values remain in the existing family vocabulary. No new
  operation-specific production type is planned.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcode.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFusedGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcodeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0005g-extrema-clamp-tensor-power-and-logical-coverage.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized by default.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 29 paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production/package | 15 | The exact 15 listed production/package paths |
| CPU tests | 9 | The exact nine listed test paths |
| Explanatory documentation | 2 | CPU [backend](../../../../glossary.md#backend) guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **29** | **15 + 9 + 2 + 3** |

The larger-than-usual path ceiling is justified by one [atomic](../../../../glossary.md#resource-transaction-and-rollback) closed-[opcode](../../../../glossary.md#term-opcode) evolution: capability,
[IR](../../../../glossary.md#term-intermediate-representation-ir) [validation](../../../../glossary.md#term-validation), [lowering](../../../../glossary.md#lowering), generated emission, reference behavior, [schema](../../../../glossary.md#term-schema) compatibility, and their
focused tests must agree before any new row may be advertised. The path map is exact; unused
authorized paths may remain unchanged, but no substitute or additional path is implied. If work
needs another path, new production type, [module](../../../../glossary.md#term-module), [Gradle](../../../../glossary.md#term-gradle)/build change, or package structure, stop
and revise planning rather than [widening](../../../../glossary.md#term-widening) scope during implementation.

## Acceptance criteria

- Capability reporting admits exactly the selected same-typed extrema, clamp, Tensor-power, and
  canonical-BOOL logical matrix. Every adjacent excluded row is unadvertised and fails before
  declarations/artifact work.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [Tensor glossary](../../../../glossary.md#tensor); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Binary/scalar floating MIN/MAX and first-class CLAMP reproduce the fixed Model 0025A NaN,
  infinity, signed-zero, equality, and ordered-bound semantics. Integral extrema use exact signed
  order. Exact raw scalar/range bits remain in IR identity.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [Model glossary](../../../../glossary.md#term-model); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- FLOAT64/FLOAT32 binary POW uses the direct CPU 0005F power contract, including one final FLOAT32
  narrowing, with no exponent inspection or alternate realization.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [narrowing glossary](../../../../glossary.md#term-narrowing); [exponent glossary](../../../../glossary.md#term-exponent))</span>
- AND/OR/NOT consume and produce canonical BOOL, use ordinary broadcasting/Shape preservation,
  and add no numeric truthiness or hot-loop validation.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation))</span>
- The family-oriented vocabulary becomes exactly thirty-one opcodes. `SCALAR_CLAMP` owns exactly
  one validated two-bound IR fact; every other immediate/realization relationship remains exact.
- One Model CLAMP occurrence remains one IR instruction. No graph or IR decomposition, new
  producer/value, reordering, constant folding, or canonicalization is introduced.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [graph glossary](../../../../glossary.md#graph); [decomposition glossary](../../../../glossary.md#term-decomposition); [producer glossary](../../../../glossary.md#term-producer); [constant folding glossary](../../../../glossary.md#term-constant-folding); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Scalar and parallel-scalar execute every admitted row over all existing access regimes and
  heap/segment/mixed carriers. All new opcodes are deterministically vector-ineligible while
  existing vector rows remain unchanged.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [vector glossary](../../../../glossary.md#term-vector))</span>
- One-through-eight fusion, virtual values, one final store, one-copy/four-candidate/one-artifact
  budgets, zero fixed-shape/unroll variants, cold binding, concurrent isolation, and direct
  Runtime invocation remain enforced.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [binding glossary](../../../../glossary.md#term-binding); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Structural identity includes opcodes and exact clamp bounds. Schema 7 rejects every stale schema
  without migration. Persistence stays bounded and disabled by default.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [Persistence glossary](../../../../glossary.md#term-persistence))</span>
- Cross-type CAST, BFLOAT16 execution, all unselected unary rows, and every malformed or
  underspecified row remain fail-closed.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [unary glossary](../../../../glossary.md#term-unary); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- No Model, Compiler, Config, Planning, shared Prepare, Runtime, Backend Contract, Trace, Engine,
  OpenBLAS provider, other backend, Gradle/build, architecture, conformance, or integration path
  changes.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- A separate clean documentation-focused pass finalizes affected Javadocs, CPU guide, glossary,
  task evidence, master plan, and roadmap; records reasoned no-change conclusions; and reuses the
  successful Java evidence unless executable behavior changes.
- [CPU](../../../../glossary.md#term-cpu) 0005A–0005F remain `Complete`; CPU 0005G changes from [`Ready`](../../../../glossary.md#term-ready) to `Complete` only after
  implementation, validation, documentation review, and final status synchronization pass. CPU
  0005H and CPU 0006–0017 remain <span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>`Draft` without detailed specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused implementation tests while executable Java is changing:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcodeTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPointwisePartitionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuFusedGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest
```

After executable Java stabilizes, run exactly one final affected-[module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :backends:cpu:test
```

The ordinary suite must continue to skip the opt-in [CPU](../../../../glossary.md#term-cpu) 0005D [persistence](../../../../glossary.md#term-persistence) timing method. Do not
rerun that benchmark because the task changes neither its verdict nor the default.

The separate documentation-focused pass runs after final Javadocs and Markdown:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also records checks for local Markdown links/anchors, balanced fences, final newlines, trailing
whitespace, exact authorized paths and the 29-path ceiling, [opcode](../../../../glossary.md#term-opcode)/[IR](../../../../glossary.md#term-intermediate-representation-ir)/immediate inventories,
capability/type/attribute [matrices](../../../../glossary.md#term-matrix-meanings), special-value tables, [schema](../../../../glossary.md#term-schema) compatibility, unchanged [fusion](../../../../glossary.md#term-fusion)/
access/[materialization](../../../../glossary.md#materialization)/strategy budgets, [fail-closed](../../../../glossary.md#term-fail-closed) CAST/[unary](../../../../glossary.md#term-unary)/[BFLOAT16](../../../../glossary.md#term-bfloat16) rows, and synchronized
status. It confirms that Java/tests outside `backends/cpu`, [Gradle](../../../../glossary.md#term-gradle), `ARCHITECTURE.md`, architecture
docs/ADRs/tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), and integration tests did not change.

Repository-wide validation remains deferred to CPU 0009, the portable generated-coverage closure
[checkpoint](../../../../glossary.md#term-checkpoint), or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). This task changes one private [concrete backend](../../../../glossary.md#concrete-backend) without a
[dependency](../../../../glossary.md#term-dependency), shared contract, public [API](../../../../glossary.md#term-application-programming-interface-api), or architecture boundary.

The documentation pass reuses successful implementation tests and does not repeat them unless it
changes executable Java or records a concrete stale-evidence risk.

## Dependencies

- Complete CPU 0005A–0005F partition lowering, access/carrier, strategy, materialization,
  five-type pointwise, floating division, and direct-power foundations.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [carrier glossary](../../../../glossary.md#term-storage-carrier); [materialization glossary](../../../../glossary.md#materialization); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Complete [Model](../../../../glossary.md#term-model) 0018T/0018U arithmetic and exact typed [scalar](../../../../glossary.md#term-scalar) contracts.
- Complete Model 0025A floating comparison/[extrema](../../../../glossary.md#saturation-pole-and-extremum)/clamp forward semantics.
- Current Model `BooleanLogicalKind`, [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype), `ShapeBroadcast`, and unresolved/exact
  descriptor contracts.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Current Java 26 primitive arithmetic, `Math.min`/`Math.max`, `StrictMath.pow`, Class-File API,
  and existing incubating Vector API toolchain.<span data-glossary-navigation="true"> ([Class-File API glossary](../../../../glossary.md#term-class-file-api); [Vector API glossary](../../../../glossary.md#term-vector-api); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>
- Current staged [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime) cold-[binding](../../../../glossary.md#term-binding)/execution, and [backend](../../../../glossary.md#backend)-contract [ownership](../../../../glossary.md#term-ownership-meanings).

## Follow-up tasks

- CPU 0005H remains Draft without a detailed specification. It will close the remaining selected
  FLOAT32/FLOAT64 unary, transcendental, and activation rows through one explicit algorithm,
  special-value, accuracy, and vector-eligibility matrix.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [unary glossary](../../../../glossary.md#term-unary); [activation glossary](../../../../glossary.md#term-activation); [vector glossary](../../../../glossary.md#term-vector); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Cross-type CAST remains fail-closed until a future Model task defines exact numerical conversion
  semantics. CPU 0005H planning must re-audit that prerequisite; it must not silently infer a
  conversion from Java casts.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [Model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning))</span>
- CPU 0006–0008 remain Draft for later portable operation families. CPU 0009 remains the portable
  generated-coverage checkpoint and depends explicitly on CPU 0005A–0005H.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Config 0006 and CPU 0017 remain Draft for future explicitly relaxed candidates. They do not
  alter the exact/default behavior selected here.

## Architecture impact

Expected impact: None.

The architecture already assigns [backend](../../../../glossary.md#backend)-private [lowering](../../../../glossary.md#lowering), exact numerical realization,
[specialization](../../../../glossary.md#term-specialization), and executable construction to [CPU](../../../../glossary.md#term-cpu) [prepare](../../../../glossary.md#prepare). No [owner](../../../../glossary.md#term-owner), [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle),
[backend identity](../../../../glossary.md#backend-identity--backendid), public surface, or shared [IR](../../../../glossary.md#term-intermediate-representation-ir) changes. Stop if implementation proves otherwise.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, completed
CPU tasks 0005A–0005F, and
docs/planning/backends/cpu/tasks/0005g-extrema-clamp-tensor-power-and-logical-coverage.md in full.
Read every affected and directly relevant Model/Compiler/Config/Prepare/Runtime contract named by
the task before editing.

Implement CPU 0005G exactly inside its 29 authorized paths. Add only the selected same-typed
binary/scalar extrema, first-class range clamp, direct Tensor/Tensor power, and canonical-BOOL
logical rows through the existing family-oriented CPU pipeline. Preserve exact Model semantics,
one-occurrence/one-instruction CLAMP, fusion/access/carrier/materialization/artifact/lifecycle
contracts, and fail-closed boundaries. Keep every new opcode scalar or parallel-scalar in this
task. Do not add unary completion, cross-type CAST, BFLOAT16/FLOAT16 execution, vector
realizations for new rows, graph rewrites, relaxed math, another abstraction/route, public/shared
API, dependency/build/architecture changes, later task specs, commits, or pushes. Stop on a
semantic proof, architecture, dependency, affected-file, or maximum-scope conflict.

After executable Java stabilizes and the focused plus exactly one final CPU suite pass, hand the
actual diff and exact Java evidence to a separate clean documentation-focused agent/thread. That
pass must follow docs/developer-guide/documentation-rules.md; independently finalize affected
Javadocs, CPU guide, glossary, planning evidence, links, scope/status checks, and reasoned
no-change conclusions; and not repeat successful Java tests unless executable behavior changes or
a concrete stale-evidence risk is recorded.

Update this task's decisions, limitations, evidence, implementation notes, completion summary,
and synchronized final status. Do not mark Complete until every criterion and the documentation
pass succeed. Leave CPU 0005H and every later CPU task Draft without a detailed specification.
```

## Local decisions

- Split the former broad remaining-pointwise row into this exact algebraic/logical task and Draft
  CPU 0005H unary closure. This keeps each implementation session reviewable while preserving
  dependency order and the CPU 0009 closure gate.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [unary glossary](../../../../glossary.md#term-unary); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Direct [Tensor](../../../../glossary.md#tensor)/Tensor `POW` reuses [CPU](../../../../glossary.md#term-cpu) 0005F's exact/default `StrictMath.pow` policy but never
  runs scalar-exponent analysis because its exponent is a runtime Tensor value.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [exponent glossary](../../../../glossary.md#term-exponent); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- `CLAMP` remains one semantic and one IR instruction. A nested two-bound IR fact is the smallest
  representation that retains exact attributes without graph/IR decomposition or a broad
  parameter hierarchy.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [graph glossary](../../../../glossary.md#graph); [decomposition glossary](../../../../glossary.md#term-decomposition); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Every new row uses scalar or parallel-scalar compute. Correct coverage precedes another
  vector-conformance decision; existing vector coverage is preserved.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Cross-type CAST remains unsupported because accepting it would invent Model numerical semantics.
  Same-type CAST remains unchanged.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- No new production class is justified. Existing family switches and the nested clamp fact are
  sufficient.

## Known limitations

- New execution remains limited to fully static, resolved-layout occurrences inside the current
  one-through-eight straight-line pointwise topology.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [pointwise glossary](../../../../glossary.md#term-pointwise); [topology glossary](../../../../glossary.md#term-topology))</span>
- BFLOAT16 remains storage-only for this family; FLOAT16 is absent; mixed types and cross-type
  casts remain unsupported.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16))</span>
- New opcodes do not use Vector API compute. Existing optional parallel-scalar orchestration is
  available, and existing vector opcodes remain unchanged.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector))</span>
- Unary/transcendental/activation completion remains CPU 0005H. Reductions and broader operation
  families remain later tasks.<span data-glossary-navigation="true"> ([Unary glossary](../../../../glossary.md#term-unary); [activation glossary](../../../../glossary.md#term-activation); [CPU glossary](../../../../glossary.md#term-cpu); [operation glossary](../../../../glossary.md#operation))</span>
- Optional generated-class persistence remains disabled by default. Native/vendor routes, tuning,
  backend conformance, and Engine integration remain deferred.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) evidence recorded before promotion to [`Ready`](../../../../glossary.md#term-ready):

- Read the architecture contract, current architecture index, runtime/prepare/backend boundary,
  planning guide, documentation rules and planning profile, roadmap, CPU master plan, completed
  CPU 0005A–0005F tasks, current CPU production/test inventory, and the directly relevant Model
  arithmetic, scalar, logical, cast, and numerical-semantics contracts.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- Compared the complete live Model pointwise vocabulary with the current closed twenty-two-opcode
  CPU matrix. The selected nine rows have sufficient semantics; remaining unary rows form a
  separate numerical/conformance decision; cross-type CAST explicitly lacks a Model numerical
  conversion contract.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [opcode glossary](../../../../glossary.md#term-opcode); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [unary glossary](../../../../glossary.md#term-unary); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Audited the current capability, IR, lowering, scalar/vector emission, reference, prepare,
  schema, specialization, fusion, and focused-test seams. The selected work fits the existing
  package map with one nested clamp fact and no new production type.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [vector glossary](../../../../glossary.md#term-vector); [schema glossary](../../../../glossary.md#term-schema); [specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Verified every local Markdown file [target](../../../../glossary.md#term-target) in the specification after correcting the [ADR](../../../../glossary.md#term-architecture-decision-record-adr) link.
- `git diff --check` passed for the planning diff.

Clean implementation context `/root/cpu_0005g_impl` supplied the stabilized executable evidence:

- The exact focused nine-suite command passed 9 suites and 41 tests with zero skips, failures, or
  errors.
- Exactly one final `./gradlew :backends:cpu:test` passed 25 suites and 106 tests with zero
  failures or errors and one existing opt-in CPU 0005D persistence-timing skip.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
- `git diff --check` passed, and the implementation context reported no path outside the exact
  authorized map.
- No executable Java or test changed after those successful commands. Clean documentation context
  `/root/cpu_0005g_docs` therefore reused the evidence and did not repeat Java tests, as required
  by the planning and documentation rules.

Clean documentation context `/root/cpu_0005g_docs` applied the General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend) Guide,
Planning, Developer Guide, and Example profiles. It independently reviewed the affected [CPU](../../../../glossary.md#term-cpu)
production and tests; the directly relevant [Model](../../../../glossary.md#term-model) arithmetic, [scalar](../../../../glossary.md#term-scalar), logical, typed-value,
broadcast, and floating-semantics contracts; and the Planning, [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Compiler](../../../../glossary.md#term-compiler), Config,
and backend [lifecycle](../../../../glossary.md#lifecycle) boundaries. It finalized the affected Javadocs and package summaries, CPU
guide, glossary, this task, CPU master plan, and roadmap. Its final validation established:

- `./gradlew :backends:cpu:javadoc` passed after the final Javadoc edits: 11 actionable tasks,
  2 executed and 9 up-to-date. Its four non-error warnings were the incubating Vector module and
  the two pre-existing documented default-constructor findings in <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>`CpuClassFileKernelGenerator`
  and `CpuPartitionLowering`.
- `git diff --check` passed after all documentation and status edits.
- Targeted local Markdown target/anchor, balanced-fence, final-newline, and trailing-whitespace
  checks passed for the five authorized Markdown files.
- The final change uses 22 authorized paths: 11 of 15 CPU production/package paths, 6 of 9 CPU
  test paths, both explanatory documentation paths, and all 3 planning/status paths. No path lies
  outside the exact map or exceeds the 29-path ceiling.
- Inventory checks confirmed exactly 31 opcodes; one optional `ClampImmediate` owned only by
  `SCALAR_CLAMP`; exact scalar/power/clamp immediate relationships; one CLAMP occurrence and one
  IR instruction; and schema 7 with no migration reader.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Capability/type/attribute and special-value matrices agree with the implementation and tests:
  selected same-typed extrema, floating direct Tensor power/range clamp, and canonical BOOL logic
  are admitted; cross-type CAST, unselected unary rows, BFLOAT16 execution, mixed types, numeric
  truthiness, and malformed rows remain fail-closed.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [Tensor glossary](../../../../glossary.md#tensor); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Existing five access regimes, heap/segment/mixed carriers, one-through-eight fusion, one final
  store, direct plus at most three materialization candidates, one-copy/one-artifact budgets, zero
  fixed-shape/unroll variants, and scalar/parallel-scalar selection for all new opcodes remain
  unchanged. Existing vector rows remain unchanged.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [materialization glossary](../../../../glossary.md#materialization); [artifact glossary](../../../../glossary.md#term-artifact); [shape glossary](../../../../glossary.md#shape))</span>
- CPU 0005A–0005G, the CPU master plan, and roadmap are synchronized as `Complete`; CPU 0005H and
  CPU 0006–0017 remain `Draft`, and no later CPU detailed specification exists.
- Java/tests outside `backends/cpu`, [Gradle](../../../../glossary.md#term-gradle)/build configuration, `ARCHITECTURE.md`, focused
  architecture explanations, ADRs, architecture tests, Backend Contract, Model, Compiler, Config,
  shared Prepare, Runtime, Trace, Engine, OpenBLAS provider, other backends, backend conformance,
  and integration paths did not change. No update is needed because CPU 0005G adds only
  backend-private realization inside existing ownership, dependencies, and lifecycle contracts.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Implementation notes

The implementation extended the existing capability, [opcode](../../../../glossary.md#term-opcode), [IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering), [scalar](../../../../glossary.md#term-scalar)-emission, and
scalar-reference family switches without adding a production type or route. The closed vocabulary
now has 31 opcodes. `CpuKernelIr.Instruction` owns one optional nested `ClampImmediate`; constructor
and type [validation](../../../../glossary.md#term-validation) require it exactly for `SCALAR_CLAMP`, while ordinary scalar immediates and
scalar-power realizations retain their previous exact relationships. The structural key includes
both raw clamp bounds, and generator [schema](../../../../glossary.md#term-schema) 7 invalidates every earlier envelope without migration.

Generated and reference [extrema](../../../../glossary.md#saturation-pole-and-extremum) use the corresponding Java primitive/`Math.min`/`Math.max`
semantics. First-class clamp evaluates lower `MAX` then upper `MIN` without decomposing the [graph](../../../../glossary.md#graph)
or IR. [Tensor](../../../../glossary.md#tensor) power directly applies `StrictMath.pow`, with one final [FLOAT32](../../../../glossary.md#term-float32) [narrowing](../../../../glossary.md#term-narrowing). Logical
emission consumes [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) locals with integer AND/OR or complement and stores canonical
bytes; the existing cold boundary validation rejects a noncanonical external BOOL before hot
execution. Every new opcode is [vector](../../../../glossary.md#term-vector)-ineligible, so cold strategy selection uses scalar or
parallel-scalar while preserving every previously eligible vector row.

No documentation outside the [CPU](../../../../glossary.md#term-cpu) guide, glossary, and [planning](../../../../glossary.md#term-planning) records required change. No
public [API](../../../../glossary.md#term-application-programming-interface-api), shared contract, [ownership](../../../../glossary.md#term-ownership-meanings) rule, [dependency](../../../../glossary.md#term-dependency), build setting, or architecture decision
changed; the affected internal and package Javadocs now describe the current 31-opcode/schema-7
contract and the exact scalar-only boundary for the new rows.

## Completion summary

- Completed changes: implemented and documented exact same-typed binary/scalar extrema,
  first-class floating CLAMP, direct FLOAT32/FLOAT64 Tensor power, and canonical-BOOL logic through
  the existing family-oriented CPU pipeline.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [Tensor glossary](../../../../glossary.md#tensor); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Files changed or created: 11 authorized CPU production/package paths, 6 authorized CPU tests,
  the CPU backend guide, glossary, this task, CPU master plan, and roadmap; 22 total paths under
  the exact 29-path ceiling.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Tests and validation: reused the implementation context's focused 9-suite/41-test pass and sole
  final 25-suite/106-test pass with one existing opt-in skip; final CPU Javadoc, Markdown, scope,
  inventory, semantics, schema, status, and whitespace gates passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [schema glossary](../../../../glossary.md#term-schema))</span>
- Documentation-agent review: `/root/cpu_0005g_docs` independently finalized Javadocs, package
  summaries, the CPU guide, glossary, and synchronized planning evidence without changing
  executable Java or repeating successful Java tests.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: the current 31-opcode semantic matrix, signed-zero clamp example,
  scalar/parallel-scalar boundary, direct Tensor-power behavior, canonical BOOL boundary, and
  schema-7 compatibility are explicit.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Javadoc review: every changed production contract and directly affected package summary is
  accurate; unaffected CPU/public and shared-module Javadocs require no change because their
  ownership, lifecycle, and callable contracts did not change.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Glossary impact: current CPU route, specialization, IR, and power entries now describe CPU
  0005G and schema 7; no new reusable public term was introduced.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
