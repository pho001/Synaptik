# Task 0005F: Floating Division and Exact [Scalar-Power Realization](../../../../glossary.md#scalar-power-realization)

## Reader context

The exponent in scalar POW was an exact typed attribute supplied when the expression was
described; it was not a Tensor of exponents read at every position. `CpuScalarPowerAnalysis`
classified those attribute bits during lowering. The generated body then used the chosen
direct-power, identity, square, reciprocal, or positive-one realization without a hot policy
lookup. Tensor division still read its ordered numerator and denominator values from the bound
carriers.

This did not authorize rewriting the graph or replacing arbitrary powers with multiplication
chains: intermediate floating-point rounding can change the result. The special cases and proofs
below belong to this historical numerical contract. The current scalar-expression reference
explains the public typed-attribute boundary separately from the CPU realization.

Current reading alongside this historical record:

- [Typed scalar arithmetic and clamp construction](../../../../reference/tensor/elementwise.md#scalar-arithmetic-and-clamp-expressions)
  — identifies exact scalar attributes and the public expression boundary.
- [CPU typed intermediate representation](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalir)
  — distinguishes code-shaping structure from concrete geometry and live values.

## Status

Complete

## Goal

Add exact/default portable [CPU](../../../../glossary.md#term-cpu) execution for same-typed [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [tensor](../../../../glossary.md#tensor) division, [scalar](../../../../glossary.md#term-scalar)
division, and scalar `POW` without rewriting the compiled or [Model](../../../../glossary.md#term-model) [graph](../../../../glossary.md#graph). [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) and scalar
division lower as ordinary family operations with exact result type and existing right-aligned
broadcast or [Shape](../../../../glossary.md#shape)-preserving access. CPU [lowering](../../../../glossary.md#lowering) keeps each semantic
`ScalarElementwiseKind.POW` [occurrence](../../../../glossary.md#term-occurrence) and its exact same-typed `ScalarValueAttrs` [exponent](../../../../glossary.md#term-exponent). One
CPU-private, route-independent analysis classifies that exponent once before [code generation](../../../../glossary.md#term-code-generation) and
selects one of five realization plans:

```text
supported DIV | SCALAR_DIV | semantic SCALAR_POW + exact typed exponent
  -> primitive division | direct power | positive one | identity | square | reciprocal
  -> existing scalar/vector and single/parallel portable machinery when eligible
  -> one compatible generated artifact and one partition executable
```

The exact selected [matrix](../../../../glossary.md#term-matrix-meanings) is FLOAT32 and FLOAT64. Division uses the ordinary primitive
left-divided-by-right or input-divided-by-scalar [operation](../../../../glossary.md#operation). Every admitted exponent has a direct
portable power fallback; exact typed positive or [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero), positive one, positive two, and
negative one may use the narrower realizations below only because each has a complete
default-[conformance](../../../../glossary.md#term-conformance) proof. Other integral exponents do not use multiply chains or exponentiation
by squaring: their intermediate [rounding](../../../../glossary.md#term-rounding), [overflow](../../../../glossary.md#term-overflow), and [underflow](../../../../glossary.md#term-underflow) can differ from one semantic
power [evaluation](../../../../glossary.md#term-evaluation), and the current contracts supply no universal proof that permits that change.

This is [backend](../../../../glossary.md#backend) realization selection, not [graph optimization](../../../../glossary.md#forward-graph-optimization). The compiled [node](../../../../glossary.md#node), operation kind,
attributes, [graph phase](../../../../glossary.md#graph-phase), value [identity](../../../../glossary.md#term-identity-meanings), [publication](../../../../glossary.md#term-publication) obligations, and [compiler](../../../../glossary.md#term-compiler)-generated [gradient](../../../../glossary.md#term-gradient)
expressions remain unchanged.

## [Scope](../../../../glossary.md#term-scope)

### Supported division and power [matrix](../../../../glossary.md#term-matrix-meanings)

- Extend the existing [pointwise](../../../../glossary.md#term-pointwise) matrix with `BinaryArithmeticKind.DIV` only when:
  - the [operation](../../../../glossary.md#operation) has exactly two inputs and one output;
  - attributes are exactly `NoOperationAttrs.INSTANCE`;
  - both inputs and the output have the same exact [data type](../../../../glossary.md#data-type--datatype), [FLOAT32](../../../../glossary.md#term-float32) or [FLOAT64](../../../../glossary.md#term-float64);
  - the output [Shape](../../../../glossary.md#shape) is the ordinary right-aligned broadcast of the two input Shapes; and
  - every Shape is fully static and every [layout](../../../../glossary.md#layout) is resolved, as required by [CPU](../../../../glossary.md#term-cpu) 0005E.
- Extend the matrix with `ScalarElementwiseKind.DIV` only when:
  - the operation has exactly one input and one output;
  - attributes are exactly `ScalarValueAttrs`;
  - input, output, and [scalar](../../../../glossary.md#term-scalar) denominator have the same exact data type, FLOAT32 or FLOAT64;
  - output Shape equals input Shape; and
  - every Shape is fully static and every layout is resolved.
- Retain the scalar denominator's exact typed primitive bits in canonical CPU IR. Do not widen a
  FLOAT32 semantic attribute, create a graph input, inspect Tensor storage, or infer a value from
  Tensor factory history.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [graph glossary](../../../../glossary.md#graph); [Tensor glossary](../../../../glossary.md#tensor); [Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>
- Extend the existing pointwise matrix with `ScalarElementwiseKind.POW` only when:
  - the operation has exactly one input and one output;
  - attributes are exactly `ScalarValueAttrs`;
  - input, output, and scalar [exponent](../../../../glossary.md#term-exponent) have the same exact data type;
  - that data type is FLOAT32 or FLOAT64;
  - output Shape equals input Shape; and
  - every Shape is fully static and every layout is resolved, as required by CPU 0005E.
- Retain the exponent's exact typed primitive bits in canonical CPU IR. Do not convert a FLOAT32
  exponent into a FLOAT64 semantic attribute, create another graph input, inspect Tensor storage,
  or infer a constant from Tensor factory history.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Classify the represented exponent exactly once during common CPU [lowering](../../../../glossary.md#lowering):

  | Exact typed exponent | Selected realization | Required computation |
  |---|---|---|
  | positive or [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) | `POSITIVE_ONE` | store exact positive typed `1.0` without reading the base |
  | positive one | [`IDENTITY`](../../../../glossary.md#term-identity-meanings) | forward the represented base value |
  | positive two | `SQUARE` | one [correctly rounded](../../../../glossary.md#nan-infinity-and-signed-zero) primitive multiply in the result type |
  | negative one | `RECIPROCAL` | one correctly rounded primitive division of positive typed one by the base |
  | every other [finite](../../../../glossary.md#term-finite) value, [infinity](../../../../glossary.md#term-infinity), or [NaN](../../../../glossary.md#term-nan) | `DIRECT` | the direct exact/default power realization |

- Exact classification uses typed represented values, not decimal text, widening-dependent
  equality, integer casts, string dispatch, or a tolerance. Distinct NaN payloads remain exact
  metadata and all select <span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [dispatch glossary](../../../../glossary.md#term-dispatch); [tolerance glossary](../../../../glossary.md#term-tolerance); [metadata glossary](../../../../glossary.md#term-metadata))</span>`DIRECT`; the [Model](../../../../glossary.md#term-model) makes no NaN-[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) result promise.
- FLOAT64 `DIRECT` delegates to `StrictMath.pow(base, exponent)`. FLOAT32 `DIRECT` exactly widens
  the two represented binary32 operands to binary64, invokes <span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`StrictMath.pow`, and narrows the
  result once to FLOAT32. This internal computation does not create cross-type scalar semantics:
  the Model input, attribute, IR value, and output all remain exact FLOAT32.
- BFLOAT16 remains fail-closed even though Model permits floating division and scalar power and CPU
  storage can represent two-byte values. CPU 0005E established no BFLOAT16 arithmetic route.
  FLOAT16 remains absent pending Model 0026. Integral and BOOL division or scalar <span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`POW` are outside
  the Model expression contract, and mismatched floating scalar types remain invalid.
- Capability reporting returns `false` for every excluded row. Complete-partition lowering rejects
  an unsupported or malformed occurrence before resource declarations, specialization, artifact
  lookup, or executable construction. No excluded row falls back through a different type or
  route.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [occurrence glossary](../../../../glossary.md#term-occurrence); [specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup))</span>

### Exact/default numerical contract and proofs

- [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) DIV evaluates the ordered primitive [operation](../../../../glossary.md#operation) `left / right` in the exact FLOAT32 or
  FLOAT64 result type. Scalar DIV evaluates <span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>`input / denominator` in that same exact type. Both
  preserve ordinary Java/IEEE-754 default behavior for NaN, infinities, signed zero,
  divide-by-zero, overflow, underflow, and subnormal inputs and results. Division by floating zero
  does not add an integer-style exception. The task promises no NaN payload or bitwise result
  beyond results fixed by the primitive operation contract.<span data-glossary-navigation="true"> ([IEEE glossary](../../../../glossary.md#term-ieee); [NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [subnormal glossary](../../../../glossary.md#term-subnormal); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Division permits no reassociation, reciprocal replacement, estimate, finite-only assumption,
  flush-to-zero, contraction, denominator normalization, or relaxed exception rule. In particular,
  public/scalar DIV is not implemented as multiplication by a reciprocal.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [finite glossary](../../../../glossary.md#term-finite); [flush-to-zero glossary](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow); [contraction glossary](../../../../glossary.md#contraction-dimension); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Exact/default means the ordinary scalar-power contract with no relaxed permission,
  reassociation, approximation, finite-only assumption, reciprocal estimate, flush-to-zero,
  contraction, or altered exceptional-value rule. The direct oracle follows the current Java 26<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>
  `StrictMath.pow` special cases, finite accuracy, and semi-monotonicity contract; FLOAT32 applies
  the one specified final narrowing.<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing))</span>
- `POSITIVE_ONE` is legal for both exponent signs of zero because the direct contract returns
  positive typed one for every base, including NaN, positive or negative zero, and either
  infinity. This is a CPU realization plan attached to semantic <span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [CPU glossary](../../../../glossary.md#term-cpu))</span>`POW`, not Compiler 0007's future
  graph-level <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph))</span>`POW(0)` replacement. It creates no logical splat, graph value, node bypass, or
  publication remapping.<span data-glossary-navigation="true"> ([logical splat glossary](../../../../glossary.md#logical-splat); [graph value glossary](../../../../glossary.md#graph-value); [node glossary](../../../../glossary.md#node); [publication glossary](../../../../glossary.md#term-publication))</span>
- [`IDENTITY`](../../../../glossary.md#term-identity-meanings) is legal only for exact positive one. The direct contract returns the base, so the
  plan preserves finite values, subnormals, infinities, and both zero signs. NaN classification is
  preserved; no NaN-payload promise is added. Compiler 0003A may already remove some guarded
  internal occurrences, but graph-output, gradient-eligible, optimization-disabled, or otherwise
  retained occurrences remain valid CPU inputs.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- `SQUARE` is legal only for exact positive two. One primitive multiply is the correctly rounded
  operation for the exact mathematical square in FLOAT32 or FLOAT64. It produces positive zero
  from either zero sign, positive infinity from either infinity, NaN from NaN, and the expected
  nonnegative finite result. Overflow, subnormal production, and underflow occur through that one
  typed multiply; there is no intermediate reassociation or extra rounding.<span data-glossary-navigation="true"> ([correctly rounded glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- `RECIPROCAL` is legal only for exact negative one. One primitive typed `+1.0 / base` implements
  the exact mathematical reciprocal with one rounding. It preserves the required signs for zero
  and infinity, propagates NaN classification, and exposes overflow, subnormal production, and
  underflow through the ordinary typed division. This realization remains semantically
  `SCALAR_POW`; it may share low-level division emission but must not become binary or [scalar](../../../../glossary.md#term-scalar) DIV.
- The conformance proof for each special plan must be documented in the analysis Javadoc and
  locked by generated/reference and direct-oracle edge tests. Tests may require exact bits where
  the contract fixes them, including positive-one results and signed-zero/infinity results. They
  must compare NaN by classification and finite direct-power results under the specified direct
  accuracy contract rather than inventing a payload or universal bitwise promise.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- `POW(0.5)` remains `DIRECT` and is never replaced by `SQRT`. Negative bases, signed zero,
  infinities, NaNs, domain behavior, and rounding make the intuitive identity insufficient.
- Positive or negative integral exponents other than `0`, `1`, `2`, and `-1` remain `DIRECT`.
  Repeated multiplication, multiply chains, reciprocal-plus-chain forms, and exponentiation by
  squaring introduce one or more rounded intermediate results and may change finite rounding,
  overflow, underflow, and subnormal transitions. Randomized agreement is not a proof. Reject
  these plans until a later contract establishes a universal error and exceptional-value proof.
- The same matrix applies to forward and compiler-generated gradient occurrences. CPU adds no
  derivative-specific realization or numerical permission.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [derivative glossary](../../../../glossary.md#term-derivative))</span>

### Common analysis, [IR](../../../../glossary.md#term-intermediate-representation-ir), and [code generation](../../../../glossary.md#term-code-generation)

- Add one `CpuScalarPowerAnalysis` in `internal.lowering`. It is stateless and classifies one
  already-validated exact typed exponent into the closed realization enum. It does not inspect a
  graph, choose a route, access Config, cache decisions, or evaluate Tensor storage.<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent); [graph glossary](../../../../glossary.md#graph); [cache glossary](../../../../glossary.md#term-cache); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Add family-oriented `DIV`, `SCALAR_DIV`, and `SCALAR_POW` values to
  `CpuPointwiseOpcode`, growing the closed vocabulary from nineteen to exactly twenty-two opcodes,
  rather than adding per-operation or per-exponent planner, instruction, emitter, executable, or
  route classes. Binary and scalar DIV use the existing arithmetic-family validation with no
  power-realization fact.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [validation glossary](../../../../glossary.md#term-validation))</span>
- Extend `CpuKernelIr.Instruction` with one optional/nested [scalar-power realization](../../../../glossary.md#scalar-power-realization) fact. Exactly
  `SCALAR_POW` requires both the exact scalar immediate and one realization; other scalar opcodes
  require an immediate and forbid a power realization; parameterless opcodes forbid both. IR
  validation must recheck the opcode/type/immediate/realization relationship fail-closed.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Keep the semantic [scalar](../../../../glossary.md#term-scalar) immediate even for `POSITIVE_ONE`, [`IDENTITY`](../../../../glossary.md#term-identity-meanings), `SQUARE`, and
  `RECIPROCAL`. The plan supplements the semantic [operation](../../../../glossary.md#operation); it does not erase or replace it.
- Extend the existing scalar emitter and scalar reference kernel through their family-grouped
  dispatch for both DIV forms and every selected power realization. Code generation consumes the
  selected realization directly and performs no exponent comparison or policy lookup in generated
  code.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [dispatch glossary](../../../../glossary.md#term-dispatch); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Scalar and parallel-scalar realize binary DIV, scalar DIV, and scalar POW for both FLOAT32 and
  FLOAT64. The existing FLOAT64 Vector emitter may realize DIV and the four special power plans
  only when the opcode and every existing access-run, carrier, topology, and strategy eligibility
  requirement passes. <span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [Vector glossary](../../../../glossary.md#term-vector); [run glossary](../../../../glossary.md#term-run); [carrier glossary](../../../../glossary.md#term-storage-carrier); [topology glossary](../../../../glossary.md#term-topology))</span>`IDENTITY`, `SQUARE`, and `RECIPROCAL` use direct vector identity,
  multiply, or divide operations; `POSITIVE_ONE` produces a typed one vector without loading the
  base. `DIRECT` power is vector-ineligible and deterministically selects scalar or
  parallel-scalar compute. FLOAT32 remains scalar or parallel-scalar under CPU 0005E's existing
  FLOAT64-only vector boundary.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Code generation must not call `Math.pow` or `StrictMath.pow` per vector lane, scalarize a
  vector body secretly, use a reciprocal approximation, or grow one code path per exponent.<span data-glossary-navigation="true"> ([vector lane glossary](../../../../glossary.md#simd))</span>
- The scalar reference remains a CPU-private conformance oracle over already-selected IR facts;
  it must not interpret Model operations in Runtime or become a second lowering compiler.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [oracle glossary](../../../../glossary.md#term-test-oracle); [Model glossary](../../../../glossary.md#term-model); [Runtime glossary](../../../../glossary.md#term-runtime); [lowering glossary](../../../../glossary.md#lowering); [compiler glossary](../../../../glossary.md#term-compiler))</span>

### [Fusion](../../../../glossary.md#term-fusion), access, carriers, [materialization](../../../../glossary.md#materialization), and execution

- Preserve [CPU](../../../../glossary.md#term-cpu) 0005E's one-to-eight connected straight-line [pointwise](../../../../glossary.md#term-pointwise) [partition](../../../../glossary.md#partition). `DIV`,
  `SCALAR_DIV`, and semantic `SCALAR_POW` each count as exactly one ordinary pointwise instruction;
  DIV participates normally in legal connected fusion. The selected power realization never
  changes its instruction count. The task does not expand the fusion budget, create multiple
  units, or consume an additional graph node.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node))</span>
- Preserve stored node and instruction order. A special realization changes only the implementation
  of that one semantic instruction; it does not reassociate surrounding operations or fuse a
  multiply chain across node boundaries.
- Preserve all five `CpuAccessPlan` regimes, right-aligned broadcasting, arbitrary start/end
  ranges, offsets, positive and zero strides, injective writes, overlap checks, and heap/<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective))</span>
  [`MemorySegment`](../../../../glossary.md#term-memorysegment)/mixed carriers. Binary DIV uses ordinary broadcast access; scalar DIV and scalar
  POW preserve Shape and add no scalar boundary.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape))</span>
- Preserve CPU 0005D's direct plus at most three eligible one-input materialization candidates,
  at most one selected FLOAT64 input copy, four-complete-candidate ceiling, one-artifact budget,
  zero fixed-shape variants, and zero unrolled variants. <span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span>`POSITIVE_ONE` may avoid reading the base
  in generated computation, but it does not alter the already-derived graph boundary,
  declaration, alias, or materialization accounting in this task.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Preserve scalar, vector, parallel-scalar, and parallel-vector strategy names and worker/chunk
  behavior. A partition containing <span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>`DIRECT` power or any FLOAT32 operation in this task uses
  scalar compute with the same optional parallel orchestration; supported work must not fail
  merely because vector compute is unavailable. FLOAT64 DIV or special power uses vector or
  parallel-vector only through the current eligibility route.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Preserve the existing exact [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [heap](../../../../glossary.md#term-heap) carriers (`float[]`, `double[]`), native-order
  segment access, representative mixed patterns, cold binding, one partition executable,
  concurrent-run isolation, and direct Runtime invocation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [run glossary](../../../../glossary.md#term-run); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

### [Specialization](../../../../glossary.md#term-specialization), fingerprints, manifests, and [artifact](../../../../glossary.md#term-artifact) compatibility

- [Canonical](../../../../glossary.md#term-canonical-form) [IR](../../../../glossary.md#term-intermediate-representation-ir) structural [identity](../../../../glossary.md#term-identity-meanings) must include `DIV` or `SCALAR_DIV` where present, the exact
  scalar-DIV denominator type and raw bits, and, for <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>`SCALAR_POW`, exact exponent type/raw bits and
  selected power realization. Changing any of those facts changes the lowering fingerprint even
  when two instructions would currently emit similar bytecode.<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent); [lowering glossary](../../../../glossary.md#lowering); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- The selected `PowerRealization` must also be an explicit validated code-shaping fact in the
  generated specialization/compatibility bytes and optional cold lowering manifest. Do not rely
  only on an opaque digest to prove compatibility.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest); [opaque glossary](../../../../glossary.md#term-opaque); [digest glossary](../../../../glossary.md#term-digest))</span>
- Bump the generator schema because the opcode vocabulary, IR instruction shape, scalar/vector
  emission, and compatibility facts change. Current-schema stored hits remain eligible; every old
  schema entry is an incompatible miss. Add no migration reader.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [opcode glossary](../../../../glossary.md#term-opcode); [shape glossary](../../../../glossary.md#shape); [vector glossary](../../../../glossary.md#term-vector))</span>
- Preserve process-local compatible interning, complete structural verification, optional bounded
  current-schema persistence, assignment-before-artifact order, and strong prepared-executable
  ownership. CPU 0005D's <span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [CPU glossary](../../../../glossary.md#term-cpu))</span>`KEEP_DISABLED` default remains unchanged; this task adds no benchmark or
  persistence-policy claim.
- Exact/default remains the only `CpuKernelSpecialization.NumericalMode`. Do not add a broad
  numerical-policy framework, registry, service locator, Config projection, generic parameter
  map, or another compiler. Config 0006 and CPU 0017 remain the only planned relaxed-permission
  path.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [parameter glossary](../../../../glossary.md#term-parameter); [compiler glossary](../../../../glossary.md#term-compiler))</span>

### Tests and documentation

- Add table-driven capability/lowering tests for both DIV forms, every admitted type and power
  plan, and representative adjacent rejection: BFLOAT16, INT32/INT64, BOOL, mismatched scalar
  type, wrong attributes, unresolved layouts, dynamic Shapes, and unsupported topology.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [layouts glossary](../../../../glossary.md#layout); [Shapes glossary](../../../../glossary.md#shape); [topology glossary](../../../../glossary.md#term-topology))</span>
- Add generated/reference differential division tests for ordinary and broadcast binary inputs and
  exact scalar denominators across NaN, infinities, both zero signs, divide-by-zero, overflow,
  underflow, subnormal transitions, finite extrema, and representative ordinary values.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [subnormal glossary](../../../../glossary.md#term-subnormal); [finite glossary](../../../../glossary.md#term-finite); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Add generated/reference/direct-[oracle](../../../../glossary.md#term-test-oracle) differential tests for [FLOAT32](../../../../glossary.md#term-float32) and [FLOAT64](../../../../glossary.md#term-float64) over [exponent](../../../../glossary.md#term-exponent)
  `+0`, `-0`, `+1`, `+2`, `-1`, representative `+3`/`-2`, fractional values including `0.5`,
  subnormal/large finite exponents, positive/negative infinity, and multiple NaN bit patterns.<span data-glossary-navigation="true"> ([negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- For each exponent plan, cover bases `+0`, `-0`, smallest/largest subnormals, minimum normals,
  representative positive and negative finite values around one, finite extrema, both infinities,
  and multiple NaNs. Include values that force overflow, underflow, and subnormal results.
- Cover scalar/parallel-scalar for both types and scalar/vector/parallel-scalar/parallel-vector for
  eligible FLOAT64 DIV and special power plans. Prove <span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>`DIRECT` power and FLOAT32 select scalar
  compute without losing parallel orchestration.
- Reuse all access/carrier/materialization/fusion/artifact/executable suites. Add representative
  scalar, zero-element, all five access regimes, arbitrary ranges, all-heap, all-segment, and mixed
  carrier cases without constructing an exponential type/exponent/carrier Cartesian product.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [materialization glossary](../../../../glossary.md#materialization); [fusion glossary](../../../../glossary.md#term-fusion); [artifact glossary](../../../../glossary.md#term-artifact); [element glossary](../../../../glossary.md#term-element); [heap glossary](../../../../glossary.md#term-heap); [Cartesian product glossary](../../../../glossary.md#term-cartesian-product))</span>
- Lock one-through-eight fusion accounting, one artifact, no stale schema or realization-plan
  cache hit, exact manifest/fingerprint differences, and fail-before-artifact behavior.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [cache hit glossary](../../../../glossary.md#cache-and-memoization); [manifest glossary](../../../../glossary.md#term-manifest); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- A separate clean documentation-focused agent must finalize affected CPU Javadocs/package
  summaries, the CPU backend guide, glossary, this task, CPU master plan, and roadmap. It must
  review Tensor, Compile, Config, Runtime, and Prepare API documentation and record reasoned
  no-change conclusions where their current contracts remain accurate.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [Tensor](../../../../glossary.md#tensor)/Tensor `BinaryArithmeticKind.POW`, Tensor exponents, compiler constant discovery, or
  inference of uniform Tensor values from host storage or factory history.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [inference glossary](../../../../glossary.md#term-inference); [host storage glossary](../../../../glossary.md#host-storage))</span>
- [Compiler](../../../../glossary.md#term-compiler) [graph](../../../../glossary.md#graph) rewrites, [constant folding](../../../../glossary.md#term-constant-folding), a graph-level `POW(0)` logical splat, changes to the
  completed guarded <span data-glossary-navigation="true"> ([logical splat glossary](../../../../glossary.md#logical-splat))</span>`POW(+1)` bypass, phase/[publication](../../../../glossary.md#term-publication) remapping, or any new Compiler pass.
- Multiply chains or exponentiation by squaring for exponents beyond positive two; reciprocal
  chains for exponents below negative one; `POW(0.5) -> SQRT`; or another unproved algebraic
  identity.<span data-glossary-navigation="true"> ([algebraic identity glossary](../../../../glossary.md#term-algebraic-identity))</span><span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- BFLOAT16/FLOAT16 power, integral power, mixed-type scalar values, cross-type promotion, a new
  public Tensor method, a new Model operation kind, or changes to Model scalar semantics.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [scalar glossary](../../../../glossary.md#term-scalar); [Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>
- Binary/scalar MIN and MAX, scalar CLAMP, remaining unary/transcendental/activation and
  boolean-logical families, cross-type CAST, and other pointwise coverage assigned to Draft CPU
  0005G.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [unary glossary](../../../../glossary.md#term-unary); [activation glossary](../../../../glossary.md#term-activation); [boolean glossary](../../../../glossary.md#data-type--datatype); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Relaxed/fast math, Config 0006, CPU 0017, finite-only assumptions, approximate reciprocal,
  vector <span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [vector glossary](../../../../glossary.md#term-vector))</span>`DIRECT` power, another [Vector species](../../../../glossary.md#term-vector-species), [gather](../../../../glossary.md#gather), masked tails, fixed [shapes](../../../../glossary.md#shape), or [unrolling](../../../../glossary.md#term-unrolling).
- Native/vendor/OpenBLAS routes, tuning or tuning-cache work, persistence-policy enablement,
  benchmarks, Engine composition, backend-conformance integration, or end-to-end execution claims.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Changes to Model, Compiler, Config, Planning, shared Prepare, Runtime, Backend Contract, Trace,
  Engine, OpenBLAS provider, another backend, Gradle, dependencies, <span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [Gradle glossary](../../../../glossary.md#term-gradle))</span>`ARCHITECTURE.md`, focused
  architecture documents, ADRs, architecture tests, `testing/backend-conformance`, or
  `testing/integration-tests`.
- Creating a detailed [CPU](../../../../glossary.md#term-cpu) 0005G or later task specification, committing, or pushing.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Compiler,
  Prepare, Runtime, concrete backend ownership, CPU routes, and dependency rules.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [concrete backend glossary](../../../../glossary.md#concrete-backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>.
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md).
- [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.
- [CPU backend master plan](../master-plan.md).
- [Completed CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md).
- [Completed CPU 0005B](0005b-universal-access-plans-and-right-aligned-broadcasting.md).
- [Completed CPU 0005C](0005c-vector-and-parallel-portable-strategies.md).
- [Completed CPU 0005D](0005d-materialization-specialization-and-persistence-evidence-gate.md).
- [Completed CPU 0005E](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md).
- [Compiler 0003A exact arithmetic rewriting](../../../modules/compiler/tasks/0003a-exact-arithmetic-rewriting.md).
- [Compiler 0003B compile-time constants](../../../modules/compiler/tasks/0003b-compile-time-constants-and-constant-folding.md)<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [constants glossary](../../../../glossary.md#term-constant))</span>.
- [Config master plan](../../../modules/config/master-plan.md).
- [Tensor API scalar arithmetic](../../../../api/tensor-api.md#scalar-arithmetic-and-clamp-expressions)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scalar glossary](../../../../glossary.md#term-scalar))</span>.
- [Compile API forward optimization](../../../../api/compile-api.md#current-package-private-canonicalization-and-whole-graph-optimization).
- [Glossary: Scalar-power realization](../../../../glossary.md#scalar-power-realization).

## Architecture constraints

- [Model](../../../../glossary.md#term-model) [`Operation`](../../../../glossary.md#operation) remains the semantic owner. CPU maps supported binary DIV, scalar DIV, and
  scalar <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar))</span>`POW` occurrences into private typed IR, adding a realization plan only for scalar power,
  without changing the graph or publishing backend semantics through Model.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [graph glossary](../../../../glossary.md#graph))</span>
- [Planning](../../../../glossary.md#term-planning) selects only `BackendId("cpu")`. CPU analysis owns exact/default division eligibility,
  exponent classification, pointwise lowering, bounded fusion, access/materialization planning,
  strategy and route selection, specialization, and exact declarations.<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent); [pointwise glossary](../../../../glossary.md#term-pointwise); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Shared Prepare receives an opaque selected CPU plan and exact resource requirements. It must not
  inspect an exponent, power plan, opcode, route, or numerical policy.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [opcode glossary](../../../../glossary.md#term-opcode))</span>
- Finalization occurs after slot assignment and cannot change the exponent, realization, IR,
  strategy, carrier pattern, materialization, or declared resources.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Runtime receives one immutable prepared executable and cold-bound invocation. No operation,
  compiled node, exponent classification, IR, policy, route, or artifact lookup reaches its hot
  path.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [node glossary](../../../../glossary.md#node); [artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- The implementation changes no module boundary, dependency direction, lifecycle stage, backend
  identity, or supported public CPU type. If implementation needs a new Model numerical decision,
  shared contract, another module, build edge, or architecture change, stop and report it.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [lifecycle glossary](../../../../glossary.md#lifecycle); [backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>

The architecture audit conclusion is `None`: [backend](../../../../glossary.md#backend)-private realization selection for a retained
semantic operation is already owned by [concrete backend](../../../../glossary.md#concrete-backend) [prepare](../../../../glossary.md#prepare).

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — truthful [occurrence](../../../../glossary.md#term-occurrence)-local capability reporting.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — common route-independent pointwise
  division lowering and scalar-power exponent classification.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [lowering glossary](../../../../glossary.md#lowering); [exponent glossary](../../../../glossary.md#term-exponent))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — division and power opcodes, exact scalar
  immediates, selected power realization, structural validation, and fingerprint input.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — scalar and eligible FLOAT64
  vector realization from already-selected facts.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [vector glossary](../../../../glossary.md#term-vector))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — [scalar](../../../../glossary.md#term-scalar) [conformance](../../../../glossary.md#term-conformance) [oracle](../../../../glossary.md#term-test-oracle).
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — strategy selection and cold [manifest](../../../../glossary.md#term-manifest).
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema, specialization, compatibility,
  verification, and optional persistence.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [specialization glossary](../../../../glossary.md#term-specialization); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Existing memory, portable-route, and executable packages are reviewed and changed only where the
  exact plan must flow through an existing contract.

Packages added or changed:

- No package is added and no responsibility moves.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScalarPowerAnalysis` — one new
  stateless classifier colocated with common route-independent lowering because it consumes exact
  semantic attributes before route/code generation.<span data-glossary-navigation="true"> ([code generation glossary](../../../../glossary.md#term-code-generation))</span>
- `CpuKernelIr.PowerRealization` — nested closed enum because it is one validated code-shaping fact
  of a scalar-power IR instruction, not a registry, policy framework, or public configuration.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [registry glossary](../../../../glossary.md#term-registry))</span>
- `CpuPointwiseOpcode.DIV`, `SCALAR_DIV`, and `SCALAR_POW` — the existing family opcode vocabulary
  owns the additional semantic instructions; no operation-specific class hierarchy is introduced.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [operation glossary](../../../../glossary.md#operation))</span>

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcode.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuScalarPowerAnalysis.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelPersistenceEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFusedGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcodeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuScalarPowerAnalysisTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0005f-floating-division-and-exact-scalar-power-realization.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized by default.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 40 paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production/package | 21 | The exact 21 listed production/package paths |
| CPU tests | 14 | The exact 14 listed test paths |
| Explanatory documentation | 2 | CPU [backend](../../../../glossary.md#backend) guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **40** | **21 + 14 + 2 + 3** |

The path map is exact; unused authorized paths may remain unchanged, but no substitute or
additional path is implied. It does not authorize another abstraction, package, [operation](../../../../glossary.md#operation) family,
executor, route, or unrelated cleanup. If implementation needs another path, new production type,
[module](../../../../glossary.md#term-module), [Gradle](../../../../glossary.md#term-gradle)/build change, or different package structure, stop and revise planning or report
the prerequisite.

## Acceptance criteria

- Capability reporting admits exactly same-typed FLOAT32/FLOAT64 binary DIV with ordinary
  right-aligned broadcast, scalar DIV with an exact matching <span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>`ScalarValueAttrs`, and [scalar](../../../../glossary.md#term-scalar) `POW`
  with an exact matching `ScalarValueAttrs`; Shapes are fully static and layouts resolved. Every
  adjacent excluded row remains unadvertised and fails before declarations/artifact work.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Every admitted exact [exponent](../../../../glossary.md#term-exponent) deterministically selects exactly one of `DIRECT`, `POSITIVE_ONE`,
  [`IDENTITY`](../../../../glossary.md#term-identity-meanings), `SQUARE`, or `RECIPROCAL` once during common lowering. The exact matrix and proofs in
  Scope are implemented without tolerance classification or hot lookup.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [Scope glossary](../../../../glossary.md#term-scope); [tolerance glossary](../../../../glossary.md#term-tolerance); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Semantic `POW`, exact scalar bits, graph phase/value/publication identity, and compiled graph
  topology remain intact. CPU adds no graph rewrite, logical constant, Tensor exponent discovery,
  or new public API.<span data-glossary-navigation="true"> ([graph phase glossary](../../../../glossary.md#graph-phase); [publication glossary](../../../../glossary.md#term-publication); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [topology glossary](../../../../glossary.md#term-topology); [graph glossary](../../../../glossary.md#graph); [constant glossary](../../../../glossary.md#term-constant); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Generated and scalar-reference division results match primitive FLOAT32/FLOAT64 behavior for
  NaN, infinities, signed zero, divide-by-zero, overflow, underflow, subnormal transitions, and
  ordinary finite values. Power results satisfy the exact edge, sign, classification, overflow,
  underflow, subnormal, and finite-accuracy rules. Exact-result assertions and non-bitwise
  comparisons follow the explicit contracts rather than convenience.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [subnormal glossary](../../../../glossary.md#term-subnormal); [finite glossary](../../../../glossary.md#term-finite))</span>
- Every exponent outside `±0`, `+1`, `+2`, and `-1` uses `DIRECT`, including `0.5`, `3`, `-2`,
  infinities, and NaNs. Production contains no multi-multiply chain or exponentiation-by-squaring
  realization.
- `DIV`, `SCALAR_DIV`, and `SCALAR_POW` integrate into the family-oriented pipeline, whose closed
  vocabulary becomes exactly twenty-two opcodes, without a per-exponent/per-operation class
  explosion, registry, service locator, generic policy framework, or second compiler. Reciprocal<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [compiler glossary](../../../../glossary.md#term-compiler))</span>
  `SCALAR_POW(-1)` remains semantically distinct from both DIV opcodes even if emission shares a
  primitive division helper.
- Scalar and parallel-scalar realize all three opcodes for FLOAT32/FLOAT64. Eligible FLOAT64 DIV
  and special power plans may use vector or parallel-vector only when every existing opcode,
  access, carrier, topology, and strategy check passes. FLOAT64 <span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [opcode glossary](../../../../glossary.md#term-opcode); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`DIRECT` power and all FLOAT32
  work use scalar or parallel-scalar compute without losing supported execution.
- DIV participates normally in the existing one-to-eight connected pointwise fusion limit and
  preserves the existing instruction order, virtual-intermediate, and one-final-store rules.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- One-to-eight fusion, virtual intermediates, one final store, all five access regimes,
  heap/segment/mixed carriers, broadcasting/access plans, arbitrary ranges, one-copy
  materialization, four candidates, one artifact, zero fixed-shape/unroll variants, and fail-closed
  alias/topology rules remain enforced.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [broadcasting glossary](../../../../glossary.md#broadcasting); [materialization glossary](../../../../glossary.md#materialization); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Opcode, exact scalar bits/type, realization, numerical mode, strategy, access form, carrier
  pattern, materialization, and schema produce compatible structural identity. A stale schema or a
  mismatched realization can never reuse an artifact; instance facts remain excluded.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Optional [persistence](../../../../glossary.md#term-persistence) remains bounded and disabled by default under `KEEP_DISABLED`; ordinary
  validation does not run the timing evidence benchmark.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run))</span>
- No Model, Compiler, Config, Planning, shared Prepare, Runtime, Backend Contract, Trace, Engine,
  OpenBLAS provider, other backend, build, architecture, conformance, or integration path changes.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- A separate clean documentation-focused pass finalizes affected Javadocs, CPU guide, glossary,
  task evidence, master plan, and roadmap; records reasoned no-change conclusions for Tensor,
  Compile, Config, Runtime, Prepare, architecture, tests outside CPU, and Gradle; and reuses final
  Java evidence unless executable code changes afterward.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- [CPU](../../../../glossary.md#term-cpu) 0005A–0005E remain `Complete`; CPU 0005F remains [`Ready`](../../../../glossary.md#term-ready) until implementation, validation,
  documentation review, and status synchronization all pass. CPU 0005G and CPU 0006–0017 remain<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
  `Draft` without detailed task specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused implementation tests:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcodeTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScalarPowerAnalysisTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPointwisePartitionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest
```

After executable Java stabilizes, run exactly one final affected-[module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :backends:cpu:test
```

The ordinary suite must continue to skip the opt-in [CPU](../../../../glossary.md#term-cpu) 0005D [persistence](../../../../glossary.md#term-persistence) timing method. Do not
rerun that evidence benchmark: persistence remains disabled and this task makes no enablement or
performance claim.

The separate documentation-focused pass, after final Javadocs and Markdown, runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates with recorded commands and results:

- local Markdown targets and heading anchors in the CPU guide, glossary, this task, CPU master
  plan, and roadmap;<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- balanced fences, final newlines, trailing whitespace, exact authorized-path membership, and the
  40-path ceiling;
- exact FLOAT32/FLOAT64 exponent/realization matrix, direct fallback, IEEE edge tables,
  finite-rounding boundaries, and rejection of unproved chains;<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [exponent glossary](../../../../glossary.md#term-exponent); [IEEE glossary](../../../../glossary.md#term-ieee); [finite glossary](../../../../glossary.md#term-finite); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- exact binary/scalar DIV and scalar-POW capability/lowering/opcode matrices, primitive division
  edge behavior, and reciprocal semantic separation;<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [lowering glossary](../../../../glossary.md#lowering); [opcode glossary](../../../../glossary.md#term-opcode))</span>
- opcode/IR/analysis/emitter/reference/package inventories and absence of per-exponent classes,
  registries, service locators, broad numerical frameworks, or another compiler;<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- [schema](../../../../glossary.md#term-schema)/[fingerprint](../../../../glossary.md#term-fingerprint)/[specialization](../../../../glossary.md#term-specialization)/[manifest](../../../../glossary.md#term-manifest) realization compatibility and stale-[cache](../../../../glossary.md#term-cache) rejection;
- unchanged one-to-eight [fusion](../../../../glossary.md#term-fusion), access/[carrier](../../../../glossary.md#term-storage-carrier)/strategy/[materialization](../../../../glossary.md#materialization)/[artifact](../../../../glossary.md#term-artifact) budgets;
- CPU 0005A–0005E `Complete`, CPU 0005F [`Ready`](../../../../glossary.md#term-ready) until completion, CPU 0005G and CPU 0006–0017
  `Draft`, and no later detailed CPU specification; and
- unchanged Java/test paths outside `backends/cpu`, [Gradle](../../../../glossary.md#term-gradle), `ARCHITECTURE.md`, focused architecture
  docs, ADRs, architecture tests, backend conformance, and integration tests.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

Repository-wide Java validation is deferred to CPU 0009, the portable generated-coverage closure
[checkpoint](../../../../glossary.md#term-checkpoint), or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). This task changes one [concrete backend](../../../../glossary.md#concrete-backend) privately and no
[dependency](../../../../glossary.md#term-dependency), shared contract, public end-to-end [API](../../../../glossary.md#term-application-programming-interface-api), or architecture boundary. The CPU-internal
generated/reference/direct-[oracle](../../../../glossary.md#term-test-oracle) [matrix](../../../../glossary.md#term-matrix-meanings) is the proportionate task-level proof.

The documentation pass reuses the successful implementation tests and does not repeat them unless
it changes executable Java or records a concrete stale-evidence risk.

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0005A [partition](../../../../glossary.md#partition)-[kernel](../../../../glossary.md#kernel) reset and common route-independent [lowering](../../../../glossary.md#lowering)/[IR](../../../../glossary.md#term-intermediate-representation-ir) architecture.
- Complete CPU 0005B static access, [broadcasting](../../../../glossary.md#broadcasting), [layouts](../../../../glossary.md#layout), spans, and [heap](../../../../glossary.md#term-heap)/segment [carrier](../../../../glossary.md#term-storage-carrier) [model](../../../../glossary.md#term-model).
- Complete CPU 0005C [scalar](../../../../glossary.md#term-scalar)/[vector](../../../../glossary.md#term-vector) and single/parallel strategy implementation.
- Complete CPU 0005D [materialization](../../../../glossary.md#materialization)/[specialization](../../../../glossary.md#term-specialization) budgets and optional-[persistence](../../../../glossary.md#term-persistence) evidence gate.
- Complete CPU 0005E five-type [pointwise](../../../../glossary.md#term-pointwise) family/[opcode](../../../../glossary.md#term-opcode) pipeline and derived boundaries.
- Current Model floating-only [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/scalar DIV and scalar `POW`, exact typed [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue),
  right-aligned broadcasting, Shape, and layout contracts.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Current [Compiler](../../../../glossary.md#term-compiler) 0003A exact guarded positive-one bypass and 0003B explicit-[constant](../../../../glossary.md#term-constant) boundary.
- Current Config exact/default boundary and Draft Config 0006 relaxed-permission separation.
- Current staged [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime) cold-[binding](../../../../glossary.md#term-binding)/execution, and [backend](../../../../glossary.md#backend)-contract [ownership](../../../../glossary.md#term-ownership-meanings) contracts.
- Java 26 `StrictMath.pow`, primitive floating division/arithmetic, Class-File API, and Vector API
  contracts already used by the CPU module.<span data-glossary-navigation="true"> ([Class-File API glossary](../../../../glossary.md#term-class-file-api); [Vector API glossary](../../../../glossary.md#term-vector-api); [module glossary](../../../../glossary.md#term-module))</span>

## Follow-up tasks

- CPU 0005G remains Draft, without a detailed specification, for all remaining selected portable
  pointwise-family coverage. CPU 0006–0008 remain Draft for later portable operation families;
  CPU 0009 remains the portable coverage checkpoint and depends on 0005G closure.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [pointwise glossary](../../../../glossary.md#term-pointwise); [operation glossary](../../../../glossary.md#operation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Compiler 0007 remains Draft for any future graph-level identities. This CPU task neither creates
  nor requires its <span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph))</span>`POW(0)` [constant](../../../../glossary.md#term-constant)/[publication](../../../../glossary.md#term-publication) proof.
- Config 0006 remains Draft as the future backend-neutral relaxed numerical permission. CPU 0017
  remains Draft as its consumer; neither blocks exact/default CPU 0005F.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- A later bounded task may reconsider additional integral-exponent realizations only if it first
  supplies a universal finite-rounding and exceptional-value proof. This task does not preapprove
  multiply chains or exponentiation by squaring.<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent); [finite glossary](../../../../glossary.md#term-finite); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Remaining pointwise rows excluded by CPU 0005E/0005F belong to CPU 0005G and require separately
  bounded planning plus any missing Model contracts before CPU 0009 can close selected portable
  coverage.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Model glossary](../../../../glossary.md#term-model))</span>

## Architecture impact

Expected impact: None.

The architecture already assigns [backend](../../../../glossary.md#backend)-private [lowering](../../../../glossary.md#lowering), numerical eligibility, route/strategy
selection, generated [specialization](../../../../glossary.md#term-specialization), exact declarations, and executable realization to concrete
[CPU](../../../../glossary.md#term-cpu) [prepare](../../../../glossary.md#prepare). The task changes no [owner](../../../../glossary.md#term-owner), [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), [backend identity](../../../../glossary.md#backend-identity--backendid), or public [API](../../../../glossary.md#term-application-programming-interface-api). If
implementation requires such a change, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0005f-floating-division-and-exact-scalar-power-realization.md.
Read the directly referenced completed CPU tasks and current Model/Compiler/Config/Prepare/Runtime
contracts needed to verify the exact division, exponent, and conformance matrices.

Implement CPU 0005F exactly as specified within its authorized paths and 40-path ceiling. Add
same-typed FLOAT32/FLOAT64 binary and scalar DIV through the existing family pipeline. Retain
semantic POW in the compiled graph; classify exact typed scalar exponents once in common CPU
analysis; preserve the direct FLOAT32/FLOAT64 power fallback; and implement only the four proved
special plans. Keep reciprocal POW semantically distinct from DIV. Do not add integer/BOOL
division, multi-multiply/exponentiation-by-squaring realizations, relaxed math, a broad policy
framework, public API, shared/build/architecture changes, later task specs, commits, or pushes.
Stop on a semantic proof, architecture, dependency, or scope conflict.

After executable Java stabilizes and the focused plus one final CPU suite pass, hand the exact diff
and recorded test evidence to a separate clean documentation-focused agent/thread. That pass must
follow docs/developer-guide/documentation-rules.md; independently finalize affected Javadocs, CPU
guide, glossary, planning evidence, links, scope/status checks, and reasoned no-change conclusions;
and not repeat successful Java tests unless executable behavior changes or a concrete risk
requires it.

Update this task's local decisions, known limitations, validation evidence, implementation notes,
completion summary, and synchronized final status. Do not mark Complete until every acceptance
criterion and the documentation pass succeed.
```

## Local decisions

- Binary and scalar DIV use exact primitive FLOAT32/FLOAT64 division in operand order. They are
  ordinary fused pointwise opcodes, not power realizations or reciprocal-multiply rewrites.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [FLOAT64 glossary](../../../../glossary.md#term-float64); [fused glossary](../../../../glossary.md#specialization-and-fusion); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- FLOAT64 DIV may use the existing vector route when all current eligibility checks pass;
  FLOAT32 DIV remains scalar or parallel-scalar. This adds no broader vector promise.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>
- `DIRECT` is mandatory for every admitted exponent outside the four proved special cases. This
  makes scalar <span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent))</span>`POW` truthful coverage rather than a capability that accepts only optimization-
  friendly exponents.
- Positive and [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) share `POSITIVE_ONE` because both have the same specified result,
  while their exact raw exponent bits remain in IR and structural identity.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Positive two and negative one are admitted because each requires exactly one correctly rounded
  primitive operation and has a complete edge proof. Positive three, negative two, and other
  integral exponents remain direct because multi-step rounding is observably different in general.<span data-glossary-navigation="true"> ([correctly rounded glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [operation glossary](../../../../glossary.md#operation); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- [FLOAT32](../../../../glossary.md#term-float32) direct power uses exact operand [widening](../../../../glossary.md#term-widening), `StrictMath.pow`, and one final narrowing. This
  is one CPU-private algorithm for an exact FLOAT32 semantic request, not cross-type Model
  promotion.<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model))</span>
- Only FLOAT64 special plans may use the current vector emitter. Direct vector power and FLOAT32
  vector power remain scalar compute because availability of a vector operator is not an
  exact/default conformance proof.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- `SCALAR_POW(-1)` may reuse the emitter's primitive division mechanism, but its opcode,
  immediate, realization, structural identity, manifest, and semantic tests remain scalar power.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [manifest glossary](../../../../glossary.md#term-manifest))</span>
- One new stateless analysis type plus a nested IR realization enum is the complete abstraction
  increment. No registry, policy service, per-exponent class, or alternative IR is justified.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>

## Known limitations

- Executable division and scalar power are limited to fully static, resolved-layout, exact
  same-typed FLOAT32 or FLOAT64 occurrences inside CPU 0005E's bounded straight-line pointwise
  topology.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [layout glossary](../../../../glossary.md#layout); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [CPU glossary](../../../../glossary.md#term-cpu); [pointwise glossary](../../../../glossary.md#term-pointwise); [topology glossary](../../../../glossary.md#term-topology))</span>
- BFLOAT16 remains storage-only for this family; FLOAT16 is absent; integral, BOOL, mixed-type,
  Tensor/Tensor power, and other Tensor-exponent power remain unsupported. Integral and BOOL
  division remain unsupported.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [Tensor glossary](../../../../glossary.md#tensor); [exponent glossary](../../../../glossary.md#term-exponent))</span>
- Only `±0`, `+1`, `+2`, and `-1` receive special realizations. Every other exponent uses direct
  power even when a faster multiply chain is mathematically attractive.
- Direct power and all FLOAT32 work do not use Vector API compute. Existing parallel-scalar
  orchestration remains available. FLOAT64 DIV and special power plans vectorize only through the
  existing compatible route.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api))</span>
- The task makes no NaN-payload promise and no bitwise-equality promise beyond results explicitly
  fixed by the ordinary contract.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Optional persistence remains disabled by default. Native/vendor routes, tuning, backend
  conformance, and Engine integration remain deferred.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context completed executable validation before this independent documentation
pass. The clean documentation-focused context `/root` changed only Javadocs/package summaries and
the five authorized Markdown records, so it reused the implementation evidence and did not rerun
Java tests:

- `./gradlew :backends:cpu:compileTestJava` passed.
- The focused generated-kernel suite passed after correction of one FLOAT32 constant-width
  bytecode issue found during implementation.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [constant glossary](../../../../glossary.md#term-constant); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- The final focused [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/generated-[kernel](../../../../glossary.md#kernel) [run](../../../../glossary.md#term-run) passed 16 tests.
- Exactly one final `./gradlew :backends:cpu:test` passed. The preserved JUnit XML contains 25
  suites and 102 tests, with one skipped opt-in CPU 0005D persistence timing test, zero failures,
  and zero errors.<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- `git diff --check` passed at implementation [handoff](../../../../glossary.md#term-handoff).

The documentation pass independently reviewed the final production and test diff, generated
Javadocs, [CPU](../../../../glossary.md#term-cpu) 0005A–0005E contracts, [Tensor](../../../../glossary.md#tensor) and [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) documentation, current Config package,
shared [Prepare](../../../../glossary.md#prepare) analysis contracts, [Runtime](../../../../glossary.md#term-runtime) prepared-execution contracts, the CPU guide, glossary,
and [planning](../../../../glossary.md#term-planning) records. Its final evidence is:

- `./gradlew :backends:cpu:javadoc` passed with `BUILD SUCCESSFUL`; 11 actionable tasks reported
  two executed and nine up-to-date. The output contained the two expected incubating-Vector
  warnings and two unchanged implicit-constructor warnings, with no Javadoc error.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- A repository-local Markdown audit of the CPU guide, glossary, this task, CPU master plan, and
  roadmap passed local target, heading-anchor, balanced-fence, final-newline, and trailing-
  whitespace checks.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- The exact-scope audit found 18 authorized CPU production/package paths, eight authorized CPU
  test paths, two explanatory-documentation paths, and three planning/status paths: 31 total,
  with no unauthorized path and below the 40-path ceiling.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Static source/test audits confirmed exactly 22 opcodes; same-typed FLOAT32/FLOAT64 binary and
  scalar DIV; all five exact-bit power classifications; FLOAT64-only vector eligibility under the
  existing gates; scalar or parallel-scalar fallback for direct power and all FLOAT32 work;
  schema 6; explicit realization compatibility; and no exponent comparison in generated code.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [schema glossary](../../../../glossary.md#term-schema); [exponent glossary](../../../../glossary.md#term-exponent))</span>
- Fusion inspection confirmed that the generic connected one-through-eight lowering admits binary
  DIV, scalar DIV, and scalar POW as ordinary single instructions, preserves stored order and
  virtual intermediates, and emits one final store. The existing one-through-eight test plus the
  new DIV/power lowering and generated-opcode tests cover the participating contracts without a
  second operation-specific fusion path.<span data-glossary-navigation="true"> ([Fusion glossary](../../../../glossary.md#term-fusion); [opcode glossary](../../../../glossary.md#term-opcode); [operation glossary](../../../../glossary.md#operation))</span>
- Status checks confirmed CPU 0005A–0005F `Complete`; CPU 0005G and CPU 0006–0017 `Draft`; and no
  detailed CPU 0005G or later task specification.
- Final `git diff --check` passed.

No architecture, [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module)-boundary, public API, or shared [lifecycle](../../../../glossary.md#lifecycle) contract changed.
Tensor and Compile API pages remain accurate because the task realizes existing semantics only;
Config remains unchanged because exact/default needs no new permission; shared Prepare still sees
one [opaque](../../../../glossary.md#term-opaque) CPU plan plus exact declarations; Runtime still invokes one already-[prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable);
and no public Tensor result, [gradient](../../../../glossary.md#term-gradient) rule, [compiler](../../../../glossary.md#term-compiler) transformation, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance),
integration, or [Engine](../../../../glossary.md#term-engine) behavior is claimed. [Gradle](../../../../glossary.md#term-gradle), `ARCHITECTURE.md`, focused architecture pages,
ADRs, architecture tests, other modules/backends, and tests outside CPU required no change.

## Implementation notes

[CPU](../../../../glossary.md#term-cpu) capability and whole-[partition](../../../../glossary.md#partition) [lowering](../../../../glossary.md#lowering) now admit exact same-typed [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) DIV,
[scalar](../../../../glossary.md#term-scalar) DIV, and scalar POW. Binary DIV retains two ordered [tensor](../../../../glossary.md#tensor) inputs and ordinary right-aligned
[broadcasting](../../../../glossary.md#broadcasting); scalar DIV retains one exact denominator immediate; scalar POW retains one exact
[exponent](../../../../glossary.md#term-exponent) immediate plus its selected realization. The route-independent [IR](../../../../glossary.md#term-intermediate-representation-ir) therefore distinguishes
Tensor/Tensor division, Tensor/scalar division, and reciprocal scalar power even when generated
emission shares primitive division.

`CpuScalarPowerAnalysis` classifies raw FLOAT32/FLOAT64 bits once during lowering. Both zero signs
select `POSITIVE_ONE`, positive one selects [`IDENTITY`](../../../../glossary.md#term-identity-meanings), positive two selects `SQUARE`, negative one
selects `RECIPROCAL`, and every other [finite](../../../../glossary.md#term-finite) value, [infinity](../../../../glossary.md#term-infinity), or [NaN](../../../../glossary.md#term-nan) selects `DIRECT`. FLOAT32
direct power widens exact represented operands, calls `StrictMath.pow`, and narrows once; FLOAT64
calls it directly. No multiply chain, reciprocal chain, square-root replacement, [tolerance](../../../../glossary.md#term-tolerance), or
[runtime](../../../../glossary.md#term-runtime) policy [lookup](../../../../glossary.md#term-lookup) was added.

Scalar and reference emission implement primitive Java/[IEEE](../../../../glossary.md#term-ieee)-754 division and every selected power
plan. Eligible FLOAT64 DIV and special power remain available to the existing [vector](../../../../glossary.md#term-vector) and parallel-
vector strategies; direct power and all FLOAT32 work use scalar compute with existing optional
parallel orchestration. The generic one-through-eight [fusion](../../../../glossary.md#term-fusion), five access regimes, derived typed
carriers, one-copy [materialization](../../../../glossary.md#materialization) ceiling, four candidates, one [artifact](../../../../glossary.md#term-artifact), zero fixed-[shape](../../../../glossary.md#shape) or
[unrolled](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) variants, cold [binding](../../../../glossary.md#term-binding), concurrent-[run](../../../../glossary.md#term-run) isolation, and one final store remain unchanged.

[Canonical](../../../../glossary.md#term-canonical-form) IR identity includes [opcode](../../../../glossary.md#term-opcode), exact scalar type/bits, and realization. [Specialization](../../../../glossary.md#term-specialization) and
the cold [manifest](../../../../glossary.md#term-manifest) expose the ordered [scalar-power realization](../../../../glossary.md#scalar-power-realization) list, and generator [schema](../../../../glossary.md#term-schema) 6 makes
older artifacts incompatible. Optional [persistence](../../../../glossary.md#term-persistence) remains bounded and disabled by default under
the CPU 0005D `KEEP_DISABLED` verdict.

## Completion summary

- Completed changes: implemented and documented exact/default FLOAT32/FLOAT64 binary DIV, scalar
  DIV, and scalar POW with direct fallback plus the four proved exact-bit special realizations.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Files changed or created: 18 CPU production/package paths, eight CPU test paths, the CPU guide,
  glossary, this task, CPU master plan, and roadmap; 31 total within the exact authorized map and
  below the 40-path ceiling.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Tests and validation: reused the successful compile-test, focused generated-kernel, final
  16-test focused, and sole final 25-suite/102-test CPU evidence with one opt-in timing skip and no
  failures or errors; final CPU Javadoc, Markdown, exact-scope, semantic, status, and whitespace
  checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [compile glossary](../../../../glossary.md#compile); [kernel glossary](../../../../glossary.md#kernel); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean documentation-focused context `/root` independently finalized
  affected Javadocs/package summaries, the CPU guide, glossary, task evidence, CPU master plan, and
  roadmap without changing executable Java or tests.
- Documentation impact: the current guide now distinguishes Tensor/Tensor DIV, Tensor/scalar DIV,
  and reciprocal scalar POW; records exact Java/IEEE behavior, realization plans, fusion,
  strategy, compatibility, schema, and lifecycle boundaries; and makes no public or end-to-end
  claim.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [IEEE glossary](../../../../glossary.md#term-ieee); [fusion glossary](../../../../glossary.md#term-fusion); [schema glossary](../../../../glossary.md#term-schema); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Javadoc review: affected provider, schema, specialization, IR, lowering, emitter, reference,
  prepare, route-plan, and package contracts describe the exact semantic, numerical,
  compatibility, and cold/hot boundaries.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [prepare glossary](../../../../glossary.md#prepare))</span>
- Glossary impact: CPU implementation status, kernel IR, and scalar-power realization now reflect
  the twenty-two-opcode schema-6 implementation and reciprocal-power separation.<span data-glossary-navigation="true"> ([scalar-power realization glossary](../../../../glossary.md#scalar-power-realization); [opcode glossary](../../../../glossary.md#term-opcode))</span>
- Architecture/build/[dependency](../../../../glossary.md#term-dependency) impact: None.
- Unresolved issues: None.
- Follow-up required: None. CPU 0005G and CPU 0006–0017 remain Draft without detailed task
  specifications.

Status: Complete
