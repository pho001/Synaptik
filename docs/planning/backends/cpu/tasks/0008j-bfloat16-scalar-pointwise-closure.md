# [CPU](../../../../glossary.md#term-cpu) Task 0008J: [BFLOAT16](../../../../glossary.md#term-bfloat16) [Scalar](../../../../glossary.md#term-scalar) [Pointwise](../../../../glossary.md#term-pointwise) Closure

## Reader context

This task records `BFLOAT16` pointwise closure through generated scalar and caller-parallel scalar
kernels. Model defines numerical meaning; ScalarValueAttrs and ClampRangeAttrs carry exact typed
bits into `CpuPartitionLowering`. `CpuScalarEmitter` realizes numerical operations by decoding raw
`BFLOAT16` locals, evaluating the established `FLOAT32`-domain algorithm, and encoding each logical
result before consumption. `WHERE` instead preserves selected raw bits; predicates produce canonical
`BOOL` bytes.

Here, semantic raw-bit tests establish results, while Class-File inspection and optimal clean-Java
comparison establish structural algorithm, loop, and overhead parity. Typed arrays and
`MemorySegment` are execution carriers, not descriptor metadata. Preparation selects execution
before Runtime invocation; fusion adds no intermediate buffer or workspace.

Current-topic navigation (not a replacement for the historical record):

- [current raw low-format storage and working values](../../../../reference/tensor/data-types.md#sixteen-bit-floating-representations) — Distinguishes represented BFLOAT16 bits from Java FLOAT32 conversion and computation.
- [current generated-class boundary](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit) — Explains emitters, typed entries, hidden classes, and the limits of Class-File inspection.

## Status

Complete

## Goal

Close the existing generated [scalar](../../../../glossary.md#term-scalar) and caller-parallel scalar [pointwise](../../../../glossary.md#term-pointwise) route for the complete
current [Model](../../../../glossary.md#term-model)-defined [BFLOAT16](../../../../glossary.md#term-bfloat16) pointwise subset. Preserve exact typed attributes, operand order,
[broadcasting](../../../../glossary.md#broadcasting), current [carrier](../../../../glossary.md#term-storage-carrier) and access-plan support, bounded pointwise-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [fusion](../../../../glossary.md#term-fusion), [publication](../../../../glossary.md#term-publication),
and failure behavior while making every BFLOAT16-producing logical [operation](../../../../glossary.md#operation) decode represented
[raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) to [FLOAT32](../../../../glossary.md#term-float32), evaluate with the established [CPU](../../../../glossary.md#term-cpu) FLOAT32-domain semantic algorithm, and
[round-to-nearest](../../../../glossary.md#floating-point-precision-and-rounding)-[ties-to-even](../../../../glossary.md#term-ties-to-even) exactly once into BFLOAT16 at that operation boundary.

Directly generated Class-Files must match an optimal clean-Java implementation of the same
specialized case in semantic algorithm, hot-loop and [dataflow](../../../../glossary.md#term-dataflow) shape, and avoidable-overhead
profile. This task adds no [Vector API](../../../../glossary.md#term-vector-api) BFLOAT16 route and no generic fallback.

## [Scope](../../../../glossary.md#term-scope)

### Source-backed [operation](../../../../glossary.md#operation) inventory

The live [Model](../../../../glossary.md#term-model) vocabulary contains exactly nineteen `UnaryElementwiseKind` [constants](../../../../glossary.md#term-constant), so the old
draft count is accurate. This task admits exactly the following forty-four existing
`CpuPointwiseOpcode` forms when every numeric input, [BFLOAT16](../../../../glossary.md#term-bfloat16)-producing output, [scalar](../../../../glossary.md#term-scalar) immediate,
and clamp bound has the exact type required below:

- [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic (7): `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, `POW`;
- scalar arithmetic/range (8): `SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, `SCALAR_DIV`,
  `SCALAR_MIN`, `SCALAR_MAX`, `SCALAR_POW`, `SCALAR_CLAMP`;
- [unary](../../../../glossary.md#term-unary) (19): `ABS`, `NEG`, `RECIPROCAL`, [`LOG`](../../../../glossary.md#term-log), `LOG1P`, [`EXP`](../../../../glossary.md#term-expx), `EXPM1`, [`ERF`](../../../../glossary.md#term-erf), `SQRT`,
  `RSQRT`, `FLOOR`, `CEIL`, `SIGN`, `RELU`, [`SIGMOID`](../../../../glossary.md#term-sigmoid), [`TANH`](../../../../glossary.md#term-tanh), `GELU_EXACT`,
  `GELU_TANH_APPROXIMATION`, `SILU`;
- comparisons (6): `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`,
  `NOT_EQUAL` over two BFLOAT16 operands with [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) output;
- classifications (3): `IS_FINITE`, `IS_NAN`, `IS_INF` over BFLOAT16 with canonical BOOL output;
  and
- selection (1): `WHERE` with canonical BOOL condition, two BFLOAT16 branches, and BFLOAT16
  output in exact condition/true-branch/false-branch order.

The inventory excludes all three BOOL logical opcodes because they are not BFLOAT16 operations,
and excludes `CAST` because conversion semantics belong to [CPU](../../../../glossary.md#term-cpu) 0008K. It also excludes any future
Model kind or CPU [opcode](../../../../glossary.md#term-opcode) until separately planned. Binary and scalar `POW` are admitted because
the current Model contracts permit BFLOAT16 floating operands/results; base is the left/[Tensor](../../../../glossary.md#tensor)
input and [exponent](../../../../glossary.md#term-exponent) is the right/scalar operand.

### Exact [BFLOAT16](../../../../glossary.md#term-bfloat16) [operation](../../../../glossary.md#operation) boundary

- A materialized or virtual BFLOAT16 value is represented by its raw 16-bit pattern in a Java
  `short` carrier or an integer local containing those bits. A numerical consumer expands it by
  the <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`BFloat16Bits.toFloat` rule: unsigned [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) become the high sixteen bits of [binary32](../../../../glossary.md#floating-point-precision-and-rounding).
- Each BFLOAT16-producing arithmetic, scalar, clamp, unary, or power instruction evaluates the
  same FLOAT32-domain algorithm currently selected for the corresponding FLOAT32 opcode,
  including that algorithm's required binary64 JDK calls and its existing final FLOAT32
  narrowing. It then applies exactly one <span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [FLOAT32 glossary](../../../../glossary.md#term-float32); [opcode glossary](../../../../glossary.md#term-opcode); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [narrowing glossary](../../../../glossary.md#term-narrowing))</span>`BFloat16Bits.fromFloat`-equivalent conversion:
  round-to-nearest with ties to even for finite values, preserve signed zero and infinities, and
  canonicalize every produced NaN to <span data-glossary-navigation="true"> ([round-to-nearest glossary](../../../../glossary.md#floating-point-precision-and-rounding); [finite glossary](../../../../glossary.md#term-finite); [signed zero glossary](../../../../glossary.md#term-signed-zero); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`0x7FC0`.
- `MIN` and `MAX` preserve the [Model](../../../../glossary.md#term-model) directional [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) and [NaN](../../../../glossary.md#term-nan)-class rules; `CLAMP` remains
  the one logical operation `MIN(MAX(input, lower), upper)` and performs one BFLOAT16 encode only
  after that complete operation, not after its internal extrema steps.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Comparisons first decode both represented BFLOAT16 values and apply the existing represented-
  numeric FLOAT32 comparison contract: ordered comparisons are false for NaN, `NOT_EQUAL` is the
  complement of numeric equality, and opposite signed zeros compare equal. Classifications decode
  once and classify the resulting binary32 value. Both families store canonical BOOL zero/one.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- `WHERE` is selection, not arithmetic conversion: it copies the exact raw bits of the selected
  BFLOAT16 branch. It therefore preserves selected signed zero, infinity, and every selected raw
  NaN pattern rather than canonicalizing an unchanged selected value.<span data-glossary-navigation="true"> ([infinity glossary](../../../../glossary.md#term-infinity))</span>
- Exact BFLOAT16 [scalar](../../../../glossary.md#term-scalar) [metadata](../../../../glossary.md#term-metadata) enters through `ScalarValue.bfloat16Bits` and
  `ScalarValueAttrs`; lowering preserves the low sixteen raw bits without FLOAT32 scalar
  substitution. Scalar <span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>`SUB` is input minus scalar, scalar `DIV` is input divided by scalar, and
  scalar `POW` is input base raised to scalar [exponent](../../../../glossary.md#term-exponent). `ClampRangeAttrs` preserves exact ordered
  BFLOAT16 lower/upper raw bits.
- Extend scalar-power realization only where its BFLOAT16 result obeys the same boundary. In
  particular, an identity realization may not raw-forward a NaN around the required operation
  encode; square and reciprocal operate in FLOAT32 then encode once. Any realization not proved
  equivalent must use the direct FLOAT32-domain power path.<span data-glossary-navigation="true"> ([scalar-power realization glossary](../../../../glossary.md#scalar-power-realization); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

### Carriers, [layouts](../../../../glossary.md#layout), ranges, and [fusion](../../../../glossary.md#term-fusion)

- Support `short[]`, native-order writable/readable [`MemorySegment`](../../../../glossary.md#term-memorysegment), and legal mixed array/segment
  boundary assignments through the existing typed generated entry shape. BOOL outputs retain<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
  `byte[]`/segment carriers. Do not add `Object` bridges or [carrier](../../../../glossary.md#term-storage-carrier) conversion [buffers](../../../../glossary.md#term-buffer).
- Retain all currently supported [pointwise](../../../../glossary.md#term-pointwise) access regimes: `DENSE_LINEAR`; read-only
  `SCALAR_ALL_ZERO`; `LAST_AXIS_BIAS`; `BLOCK_OUTER`; and `GENERAL_ODOMETER`. Preserve right-
  aligned broadcasting, arbitrary legal non-negative offsets/strides, referenced-span checks,
  injective outputs, and existing binding/overlap rejection. A broadcast or otherwise
  non-injective output remains unsupported.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective); [binding glossary](../../../../glossary.md#term-binding))</span>
- Reuse the same generated scalar invocation for single-thread scalar and caller-parallel scalar
  execution. Every half-open range owns disjoint logical output coordinates; empty and partial
  ranges, worker validation/joining, failure-before-publication, and zero-workspace behavior stay
  unchanged.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [half-open glossary](../../../../glossary.md#term-half-open-interval); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [validation glossary](../../../../glossary.md#term-validation); [publication glossary](../../../../glossary.md#term-publication))</span>
- Bounded pointwise fusion is legal only when every virtual BFLOAT16 value remains raw represented
  bits. A BFLOAT16-producing instruction must encode into its raw local before any consumer, and a
  numerical consumer must decode that local. This preserves one rounding boundary per logical DAG
  node without materializing a buffer and prevents final-store-only narrowing.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [represented bits glossary](../../../../glossary.md#term-represented-bits); [rounding glossary](../../../../glossary.md#term-rounding); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [node glossary](../../../../glossary.md#node); [narrowing glossary](../../../../glossary.md#term-narrowing))</span>
- If any generator, fixed guarded loop, multi-store form, or fused topology cannot prove that raw-
  local invariant, reject that contraction and retain CPU 0008B's deterministic materialized split.
  If even the bounded split topology cannot be represented, capability/preparation fails closed
  before declaration or artifact lookup. Never silently keep a FLOAT32 virtual intermediate,
  disable a required encode/decode, or route to a generic/reference fallback.<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion); [topology glossary](../../../../glossary.md#term-topology); [invariant glossary](../../../../glossary.md#term-invariant); [CPU glossary](../../../../glossary.md#term-cpu); [artifact glossary](../../../../glossary.md#term-artifact); [lookup glossary](../../../../glossary.md#term-lookup); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>

### [Schema](../../../../glossary.md#term-schema), [artifacts](../../../../glossary.md#term-artifact), evidence, and documentation

- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly once from 58 to 59 after emitted bytes
  change. Schema 58 and older envelopes remain incompatible safe misses; add no migration,
  compatibility alias, or dual-schema reader.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Existing IR structural identity already includes value data types, opcodes, typed immediate
  bits, power realization, clamp bounds, stores, and access regimes; specialization identity
  includes exact carriers and execution strategy. Extend validation/encoding for BFLOAT16 without
  adding Shape, extent, address, slot, range, or concrete segment identity to the cache key.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings); [specialization glossary](../../../../glossary.md#term-specialization); [validation glossary](../../../../glossary.md#term-validation); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [Shape glossary](../../../../glossary.md#shape); [extent glossary](../../../../glossary.md#term-extent); [cache key glossary](../../../../glossary.md#term-cache-key))</span>
- Prove unchanged FLOAT64/FLOAT32/INT32/INT64/BOOL projections produce byte-identical generated
  Class-Files. BFLOAT16 creates new pointwise projections only; it must not rewrite older family
  algorithms or specialize by fixed Shape.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Finalize affected Javadocs, package summaries, CPU backend guide, glossary impact, task evidence,
  master-plan status, and roadmap status in a distinct clean documentation-focused context after
  executable work stabilizes.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Java Vector API or BFLOAT16 SIMD, vector tails selected from a BFLOAT16 vector body, native
  execution, OpenBLAS, or a new peer route.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [SIMD glossary](../../../../glossary.md#simd); [vector glossary](../../../../glossary.md#term-vector); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- `CAST`, mixed numeric promotion, implicit FLOAT32 scalar metadata, cross-type WHERE branches,
  BOOL logical closure, or a new Model/public API/Compiler/Prepare/Runtime contract.<span data-glossary-navigation="true"> ([numeric promotion glossary](../../../../glossary.md#numeric-promotion); [FLOAT32 glossary](../../../../glossary.md#term-float32); [scalar glossary](../../../../glossary.md#term-scalar); [metadata glossary](../../../../glossary.md#term-metadata); [BOOL glossary](../../../../glossary.md#data-type--datatype); [Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- New materialization policy, representation candidate, profitability rule, fusion grammar,
  autotuning, fixed-Shape/fixed-trip specialization, packing, workspace, or generic fallback.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [fusion glossary](../../../../glossary.md#term-fusion); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [Shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Fusion through numerical-order, stateful, random, scan, reduction, normalization, convolution,
  attention, loss, movement, indexing, scatter, fold, or ordering units.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [loss glossary](../../../../glossary.md#loss))</span>
- Changing Model numerical meaning, approximation formula, gradient behavior, publication order,
  alias policy, module dependencies, build structure, architecture rules, conformance, or
  integration ownership.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [alias glossary](../../../../glossary.md#aliasing-and-arena); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [CPU](../../../../glossary.md#term-cpu) 0008K–0008P or CPU 0009 implementation or detailed [planning](../../../../glossary.md#term-planning).

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`general documentation profile`](../../../../developer-guide/documentation/general-style.md)
- [`planning documentation profile`](../../../../developer-guide/documentation/planning-style.md)
- [`CPU master plan`](../master-plan.md)
- [CPU 0005E pointwise coverage](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- [CPU 0005F scalar POW](0005f-floating-division-and-exact-scalar-power-realization.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [CPU 0005G pointwise closure](0005g-extrema-clamp-tensor-power-and-logical-coverage.md)
- [CPU 0005H unary closure](0005h-portable-unary-transcendental-and-activation-closure.md)<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>
- [CPU 0005J vector/mask closure](0005j-bounded-pointwise-coverage-and-parity-hardening.md)<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- [CPU 0007A0 generated parity](0007a0-generated-hot-path-parity-correction.md)
- [CPU 0007A1A scalar-body self-containment](0007a1a-generated-scalar-body-self-containment.md)
- [CPU 0007A1L general-loop parity](0007a1l-pointwise-general-loop-residual-parity.md)
- [CPU 0008B bounded DAG/fusion](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- [CPU 0008E1 shared DAG adoption](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md)
- [CPU 0008I loss closure](0008i-portable-loss-family-execution.md)<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss))</span>

## Architecture constraints

- Model remains the semantic authority. CPU capability and cold analysis admit only exact current
  occurrences; CPU lowering/fusion/strategy/resource selection precede shared assignment;
  finalization realizes the selected artifact; Runtime invokes immutable prepared primitives.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [artifact glossary](../../../../glossary.md#term-artifact); [Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Preserve one atomic CPU partition executable, exact resource declarations, validation before
  write, deterministic topological unit order, and caller-parallel completion/join before the next
  unit or publication.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [partition glossary](../../../../glossary.md#partition); [validation glossary](../../../../glossary.md#term-validation); [topological glossary](../../../../glossary.md#term-topological-order); [publication glossary](../../../../glossary.md#term-publication))</span>
- Generated code remains typed, direct, allocation-free in normal invocation, and free of graph,
  operation, layout, route, cache, resource, and worker interpretation.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [operation glossary](../../../../glossary.md#operation); [layout glossary](../../../../glossary.md#layout); [cache glossary](../../../../glossary.md#term-cache))</span>
- The optimal clean-Java specialized implementation is the design and review oracle. Any generated
  deviation in semantic algorithm, per-node BFLOAT16 round-trip, hot-loop/dataflow shape, or
  avoidable overhead needs an explicit technical reason and supporting evidence; otherwise stop.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [node glossary](../../../../glossary.md#node); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [round-trip glossary](../../../../glossary.md#schema-serialization-and-checkpoint); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>
- If a proposed BFLOAT16 operation lacks a complete current Model contract, exclude it and report
  the prerequisite. Do not infer semantics from a draft planning row.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.backend.cpu` — exact [occurrence](../../../../glossary.md#term-occurrence)-local capability;
- `...internal.ir` — [BFLOAT16](../../../../glossary.md#term-bfloat16) [validation](../../../../glossary.md#term-validation) for the existing [opcode](../../../../glossary.md#term-opcode)/value/immediate vocabulary;
- `...internal.lowering` — exact raw [scalar](../../../../glossary.md#term-scalar)/clamp [lowering](../../../../glossary.md#lowering) and [fusion](../../../../glossary.md#term-fusion) legality;
- `...internal.codegen.emit` — raw BFLOAT16 locals, conversion, scalar formulas, carriers, and
  generated Class-File assembly;
- `...internal.prepare` and `...internal.executable` — unchanged scalar/parallel-scalar selection,
  binding, and range execution, edited only if BFLOAT16 validation is currently closed;<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- `...internal.cache` — [schema](../../../../glossary.md#term-schema) 59 and current-only compatibility; and
- `...internal.reference` — test/evidence [oracle](../../../../glossary.md#term-test-oracle) only, never a production generated [dependency](../../../../glossary.md#term-dependency).

No package or public type is added. At most one focused test-only BFLOAT16 performance [owner](../../../../glossary.md#term-owner) may be
added; production BFLOAT16 behavior belongs in the existing [pointwise](../../../../glossary.md#term-pointwise) owners.

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcode.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionDagDecomposer.java` only if the raw-local proof needs an explicit [fusion](../../../../glossary.md#term-fusion) gate
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- directly affected package-info files in those existing packages.

Expected focused tests are the matching capability, [opcode](../../../../glossary.md#term-opcode)/[IR](../../../../glossary.md#term-intermediate-representation-ir), [pointwise](../../../../glossary.md#term-pointwise) [lowering](../../../../glossary.md#lowering)/[decomposition](../../../../glossary.md#term-decomposition),
pointwise/[fused](../../../../glossary.md#specialization-and-fusion)/[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) generated, preparer/prepared-executable, [specialization](../../../../glossary.md#term-specialization)/[artifact](../../../../glossary.md#term-artifact)-store, and
reference-[oracle](../../../../glossary.md#term-test-oracle) [owners](../../../../glossary.md#term-owner), plus at most one new
`CpuBFloat16PointwisePerformanceTest.java`. Documentation paths are the [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) guide, glossary
only if terminology changes, this task, CPU master plan, and roadmap.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 38 repository paths may change: 16 production/[Javadoc](../../../../glossary.md#term-javadoc) paths, 16 test/evidence paths, and
6 documentation/[planning](../../../../glossary.md#term-planning) paths. No new production type or package is authorized. Evidence
Class-Files, [decompilation](../../../../glossary.md#term-decompilation), benchmark sources/classes/[logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), manifests, and environment records live
under one fresh `/private/tmp/synaptik-cpu-0008j-*` directory and do not count. If correct support
requires a [Model](../../../../glossary.md#term-model)/shared-[module](../../../../glossary.md#term-module) change, a second production abstraction, more than 38 paths, or an
out-of-scope route/policy, stop and replan.

## Acceptance criteria

1. Capability, IR, and lowering accept exactly the forty-four listed BFLOAT16 forms and reject
   mixed types, wrong result types, non-BFLOAT16 scalar/clamp attributes, malformed arity/order,
   BOOL logic, CAST, non-injective outputs, unsupported layouts, and unproved spans.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [scalar glossary](../../../../glossary.md#term-scalar); [arity glossary](../../../../glossary.md#term-arity); [BOOL glossary](../../../../glossary.md#data-type--datatype); [injective glossary](../../../../glossary.md#term-injective); [layouts glossary](../../../../glossary.md#layout))</span>
2. Raw-bit semantic tests cover every admitted opcode, scalar-power realization, arrays, segments,
   representative mixed carriers/access regimes, scalar and parallel-scalar ranges, and fused and
   split topologies. Cases include both zeros, infinities, extrema, subnormals, halfway ties with
   even/odd retained least-significant bits, overflow/underflow, and quiet/signaling/signed/payload
   NaNs according to the exact production contract.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [scalar-power realization glossary](../../../../glossary.md#scalar-power-realization); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [subnormals glossary](../../../../glossary.md#term-subnormal); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
3. Every BFLOAT16-producing DAG node encodes once before a consumer; generated/decompiled fused
   evidence exposes the raw-local encode/decode boundary and differs from a deliberately
   final-store-only control on a rounding-sensitive chain. <span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [node glossary](../../../../glossary.md#node); [rounding glossary](../../../../glossary.md#term-rounding))</span>`WHERE` preserves selected raw bits;
   comparisons/classifications produce canonical BOOL.<span data-glossary-navigation="true"> ([raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
4. Existing array/segment/mixed carrier binding, all five access regimes where legal, injective
   output, overlap rejection, arbitrary half-open ranges, caller-parallel ownership/join,
   zero-resource behavior, and atomic publication remain intact.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [half-open glossary](../../../../glossary.md#term-half-open-interval); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [atomic publication glossary](../../../../glossary.md#term-atomic-publication))</span>
5. Generated classes are final and field-free with one typed static entry and no hidden Synaptik
   helper/fallback/reference call, allocation, boxing, reflection, method handle,<span data-glossary-navigation="true"> ([method handle glossary](../../../../glossary.md#term-method-handle))</span>
   `invokedynamic`, collection/string dispatch, monitor, or operation/graph/runtime dispatch.
   Permitted JDK numerical calls are enumerated and match the FLOAT32-domain oracle.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime); [FLOAT32 glossary](../../../../glossary.md#term-float32); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
6. Schema advances exactly 58 to 59. Schema-58/older envelopes are safe misses, malformed entries
   fail safely, BFLOAT16 typed immediates and access/carrier shapes cannot alias, and every unchanged
   pre-0008J pointwise/family projection retains byte-identical generated Class-File bytes.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [shapes glossary](../../../../glossary.md#shape); [alias glossary](../../../../glossary.md#aliasing-and-arena); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
7. A fresh five-fork generated-versus-direct matrix covers five representative, genuinely distinct
   emitted/dataflow forms:<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>
   - [contiguous](../../../../glossary.md#term-contiguous) `short[]` [unary](../../../../glossary.md#term-unary) `GELU_EXACT` ([binary64](../../../../glossary.md#floating-point-precision-and-rounding) internal [activation](../../../../glossary.md#term-activation) branch);
   - dense all-segment [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `POW` (two-input [JDK](../../../../glossary.md#term-java-development-kit-jdk)-power path);
   - mixed-[carrier](../../../../glossary.md#term-storage-carrier) [BFLOAT16](../../../../glossary.md#term-bfloat16) `SCALAR_CLAMP` with general-affine injective output (two exact raw
     immediates and general addressing);
   - broadcast/mixed BFLOAT16 comparison or classification to canonical BOOL (different result
     type and predicate branch); and
   - a bounded [fused](../../../../glossary.md#specialization-and-fusion) `ADD -> SIGMOID -> WHERE` or equivalently rounding-sensitive chain spanning
     raw BFLOAT16 and virtual BOOL values (per-node round-trip and fusion branch).
   This matrix is representative rather than a carrier Cartesian product because array/segment
   access, access regime, predicate output, JDK numerical branch, typed immediates, and fused raw-
   local dataflow are the code-shaping dimensions; swapping every boundary independently does not
   create a new semantic algorithm. Add a row only if implementation inspection proves another
   generated code shape. Every row in every fork and the median of fork medians must be
   generated/direct <span data-glossary-navigation="true"> ([round-trip glossary](../../../../glossary.md#schema-serialization-and-checkpoint); [fusion glossary](../../../../glossary.md#term-fusion); [Cartesian product glossary](../../../../glossary.md#term-cartesian-product); [dimensions glossary](../../../../glossary.md#dimension); [median glossary](../../../../glossary.md#term-median))</span>`<= 1.15x` under the repository's fixed-heap, warmup, randomized-order,
   adaptive-at-least-25-ms, no-retry/no-discard protocol.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [warmup glossary](../../../../glossary.md#term-warmup))</span>
8. The direct oracle uses ordinary clean Java with the same typed carriers, cold geometry/ranges,
   operation order, FLOAT32/binary64 work, raw-local per-node encodes, stores, and loop/address
   form. Source plus complete `javap -c -v -p` comparison proves neither side hides dispatch,
   allocation, helper indirection, fixed fixture extents, or avoidable per-element work.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [element glossary](../../../../glossary.md#term-element))</span>
9. A distinct clean documentation pass finalizes affected Javadocs/package summaries, CPU guide,
   glossary decision, evidence, and planning state. CPU 0008K–0008P retain their exact existing
   scope and order.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning); [scope glossary](../../../../glossary.md#term-scope))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Tier 1: [run](../../../../glossary.md#term-run) focused capability, [opcode](../../../../glossary.md#term-opcode)/[IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering)/[decomposition](../../../../glossary.md#term-decomposition), reference, generated [pointwise](../../../../glossary.md#term-pointwise),
[fused](../../../../glossary.md#specialization-and-fusion)/[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag), [binding](../../../../glossary.md#term-binding)/preparation/executable, and [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) tests while developing. The [matrix](../../../../glossary.md#term-matrix-meanings) must
prove all forty-four forms semantically, exact raw attributes, five access regimes where legal,
array/segment/mixed carriers without exhaustive redundant permutations, range [ownership](../../../../glossary.md#term-ownership-meanings), [alias](../../../../glossary.md#aliasing-and-arena)
failure before writes, and fused per-[node](../../../../glossary.md#node) [rounding](../../../../glossary.md#term-rounding).

Tier 2: after executable work stabilizes, run once:

```text
./gradlew :backends:cpu:test --rerun-tasks
```

Generate the representative [artifacts](../../../../glossary.md#term-artifact) and retain their Class-Files, checksums, complete
[decompilation](../../../../glossary.md#term-decompilation), [descriptor](../../../../glossary.md#term-descriptor)/member-reference/forbidden-opcode reports, [oracle](../../../../glossary.md#term-test-oracle) source/classes, raw
five-[fork](../../../../glossary.md#term-fork-meanings) [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), summary, commands, [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)/OS/[CPU](../../../../glossary.md#term-cpu) [metadata](../../../../glossary.md#term-metadata), and verified [manifest](../../../../glossary.md#term-manifest) under the one fresh
evidence root. No failed fork may be discarded or averaged away.

Tier 3: the separate documentation-focused context reuses stable Java/performance evidence unless
it changes executable behavior, then runs:

```text
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also checks Markdown links/anchors/fences/headings/final newlines, schema/status [synchronization](../../../../glossary.md#term-synchronization),
exact path bounds, unchanged CPU 0008K–0008P rows/order, no 0008K task file, no staged change, and
preservation of unrelated dirty work. Repository-wide, architecture, [conformance](../../../../glossary.md#term-conformance), and integration
tests remain deferred to CPU 0009/[CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes no shared boundary; discovering such
a change is a stop condition.

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0008I and current [schema](../../../../glossary.md#term-schema)-58 generated infrastructure.
- Complete CPU 0005E–0005J pointwise operation, scalar-power, closure, unary, and vector-boundary
  contracts.<span data-glossary-navigation="true"> ([pointwise operation glossary](../../../../glossary.md#elementwise-unary-binary-and-interval-vocabulary); [scalar glossary](../../../../glossary.md#term-scalar); [unary glossary](../../../../glossary.md#term-unary); [vector glossary](../../../../glossary.md#term-vector))</span>
- Complete CPU 0007A0/0007A1A/0007A1L generated-loop, self-containment, and direct-[oracle](../../../../glossary.md#term-test-oracle) rules.
- Complete CPU 0008B and 0008E1 bounded [fusion](../../../../glossary.md#term-fusion)/shared-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) infrastructure.
- Current Model typed-scalar, promotion, binary/scalar arithmetic, comparison/classification,
  WHERE, unary/activation, floating-extrema/clamp, and <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [activation glossary](../../../../glossary.md#term-activation); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>`BFloat16Bits` contracts.

## Follow-up tasks

- CPU 0008K remains the next Draft task after this task completes and owns only Model-defined
  cross-type CAST execution. Do not create its detailed specification here.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [CPU](../../../../glossary.md#term-cpu) 0008L–0008P retain their current user-owned [scope](../../../../glossary.md#term-scope) and order unchanged.
- CPU 0009 remains the generated-coverage checkpoint and also owns the missing corrected CPU 0008I
  loss performance evidence.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [loss glossary](../../../../glossary.md#loss))</span>

## Architecture impact

Expected impact: None.

This task extends a [CPU](../../../../glossary.md#term-cpu)-private generated representation already authorized by the architecture.
It changes no semantic authority, [module](../../../../glossary.md#term-module) edge, public/shared contract, resource [lifecycle](../../../../glossary.md#lifecycle), or
[Runtime](../../../../glossary.md#term-runtime) responsibility. If implementation needs such a change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0008J. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, or modifying unrelated work.
Do not use GSD. Preserve exactly the test-only CpuLossPerformanceTest correction and the completed
CPU 0008I documentation.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, CPU master plan,
task 0008J, its referenced completed CPU tasks, the relevant Model contracts, and every affected
pointwise production/test owner in full. Implement exactly the forty-four-form BFLOAT16 scalar and
caller-parallel scalar closure. Keep BFLOAT16 locals as raw bits; decode for numerical work and
encode once at every BFLOAT16-producing logical node before any fused consumer. Preserve WHERE raw
selection and canonical BOOL predicates. Reject fusion to the legal split topology whenever that
invariant is not proved. Do not add SIMD, CAST, mixed promotion, public API, materialization,
autotuning, fixed-Shape specialization, native routing, or fallback.

Run the focused/full CPU, semantic raw-bit, binding/range/publication, schema/cache, complete
Class-File/decompilation/forbidden-reference, and five-fork representative direct-Java performance
gates. Retain reproducible evidence under one fresh /private/tmp directory. Then hand the stable
diff and evidence to a distinct clean documentation-focused context following documentation-
rules.md. Do not mark Complete until every acceptance criterion and that final pass succeed. Stop
on architecture, Model-contract, oracle-equivalence, fusion-rounding, or path-budget uncertainty.
```

## Local decisions

- The live Model and CPU vocabularies confirm nineteen unary kinds and forty-four BFLOAT16-relevant
  opcode forms; the three BOOL logical forms and same-type <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu); [unary glossary](../../../../glossary.md#term-unary); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [opcode glossary](../../../../glossary.md#term-opcode); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`CAST` are outside this closure.
- Per-node rounding is resolved by typed operation results, not by fusion policy. Raw BFLOAT16
  virtual locals make legal fusion possible without materialization; final-store-only narrowing is
  semantically invalid. Split retention is the fail-closed alternative.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [rounding glossary](../../../../glossary.md#term-rounding); [operation glossary](../../../../glossary.md#operation); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization); [narrowing glossary](../../../../glossary.md#term-narrowing); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- `WHERE` copies a selected represented value and therefore preserves raw branch bits. Produced
  numerical NaNs pass through the required encode and become canonical <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>`0x7FC0`.
- The five-row performance matrix covers each distinct algorithm/dataflow/access branch without
  demanding a redundant full carrier Cartesian product.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [dataflow glossary](../../../../glossary.md#term-dataflow); [carrier glossary](../../../../glossary.md#term-storage-carrier); [Cartesian product glossary](../../../../glossary.md#term-cartesian-product))</span>

## Known limitations

- No [BFLOAT16](../../../../glossary.md#term-bfloat16) [SIMD](../../../../glossary.md#simd) or mixed-type [pointwise](../../../../glossary.md#term-pointwise) execution is claimed.
- Performance conclusions apply only to the retained representative code shapes, environment,
  JVM, and protocol; CPU 0009 remains responsible for whole-backend evidence classification.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation closes exactly 44 BFLOAT16 pointwise forms: 7 binary, 8 scalar/range, 19
  unary, 6 comparison, 3 classification, and <span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [pointwise glossary](../../../../glossary.md#term-pointwise); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [unary glossary](../../../../glossary.md#term-unary))</span>`WHERE`. Generated scalar and caller-parallel scalar
  execution passed across arrays, native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment), mixed carriers, and all five legal
  current access regimes. No BFLOAT16 SIMD, CAST, mixed promotion, native route, new
  materialization/autotuning policy, fixed-Shape specialization, or generic fallback was added.<span data-glossary-navigation="true"> ([SIMD glossary](../../../../glossary.md#simd); [materialization glossary](../../../../glossary.md#materialization); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [Shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- The authoritative CPU rerun completed successfully with 132 suites, 692 tests, 14 expected
  opt-in skips, and zero failures or errors. Focused/evidence results contain 14 suites, 178 tests,
  3 expected opt-in skips, and zero failures or errors. Independent review corrected BFLOAT16
  scalar-POW analysis and made the historical CPU 0008E external-evidence test opt-in through<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
  `-Dsynaptik.cpu.0008e.retainedEvidence=true` without weakening any assertion when enabled.
- Accepted retained evidence is `/private/tmp/synaptik-cpu-0008j-K4FprD`. Its 85-artifact manifest
  verifies completely. Five exact generated descriptors cover GELU over arrays, POW over segments,
  mixed general CLAMP, broadcast comparison to BOOL, and fused WHERE. Every generated and direct
  scan passed: each generated class is final and field-free with exactly one typed static entry;
  there are no forbidden references, allocation, boxing, reflection, dynamic dispatch, or generic<span data-glossary-navigation="true"> ([artifact manifest glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [descriptors glossary](../../../../glossary.md#term-descriptor); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [BOOL glossary](../../../../glossary.md#data-type--datatype); [fused glossary](../../../../glossary.md#specialization-and-fusion); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
  `Object` carriers. Complete direct timed source, class, and `javap` output are retained.
- Every row in every accepted [fork](../../../../glossary.md#term-fork-meanings) passed the `<= 1.15x` generated/direct gate. Median-of-five
  ratios are <span data-glossary-navigation="true"> ([Median glossary](../../../../glossary.md#term-median))</span>`0.9334689173615348` for `GELU_ARRAY`, `1.093502884572582` for `POW_SEGMENT`,
  `0.9634384935191979` for `CLAMP_MIXED_GENERAL`, `0.9369720889236289` for
  `COMPARISON_BROADCAST`, and `0.9312897148034954` for `FUSED_WHERE`. The maximum individual
  accepted ratio is POW at `1.1023740371679942`.
- The first attempt at `/private/tmp/synaptik-cpu-0008j-cfHknH` remains retained and rejected:
  POW measured `2.170433141804421` because its direct oracle used mismatched address/code shape.
  The accepted oracle instead uses matching cold geometry-derived addressing. Neither protocol
  retried nor discarded samples, and the failed attempt was not relabelled as accepted evidence.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Generated code was compared with an optimal clean Java implementation for equivalent semantic
  algorithm, hot-loop/dataflow shape, and avoidable overhead. This is not byte-for-byte Class-File
  identity, guaranteed identical JIT assembly, a whole-backend performance-parity result, or a
  hardware-universal claim. CPU 0009 still owns CPU 0008I's missing corrected full loss evidence
  and the later complete evidence inventory.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [identity glossary](../../../../glossary.md#term-identity-meanings); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [backend glossary](../../../../glossary.md#backend); [loss glossary](../../../../glossary.md#loss))</span>
- The independent documentation context reviewed every changed production Javadoc and found the
  final behavior, invariants, parameters, results, failures, and formatting accurate. The earlier
  passing <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [parameters glossary](../../../../glossary.md#term-parameter))</span>`:backends:cpu:javadoc` result remains applicable because this pass changed no Java
  comment. The CPU guide now documents current BFLOAT16 support, exact exclusions, schema-59
  isolation, and fused per-node rounding. Existing glossary entries were corrected; no new
  reusable BFLOAT16 operation-boundary term was introduced.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [node glossary](../../../../glossary.md#node); [rounding glossary](../../../../glossary.md#term-rounding); [operation glossary](../../../../glossary.md#operation))</span>
- Documentation validation passed local links and anchors, heading/status/frontier/schema/path
  assertions, balanced fences, terminal newlines, whitespace, exact task scope, protected CPU
  0008I and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`CpuLossPerformanceTest` preservation, unchanged CPU 0008K–0008P order/scope, absence
  of a detailed 0008K task, empty staging, `git diff --check`, and `git diff --cached --check`.

## Implementation notes

- BFLOAT16 values remain raw 16-bit virtual locals. Numerical consumers decode at use, and every
  arithmetic-producing logical DAG node performs exactly one inline ties-to-even/canonical-NaN
  encode before any fused consumer. <span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [node glossary](../../../../glossary.md#node); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [canonical glossary](../../../../glossary.md#term-canonical-form); [NaN glossary](../../../../glossary.md#term-nan); [fused glossary](../../../../glossary.md#specialization-and-fusion))</span>`WHERE` copies selected raw bits; comparisons and
  classifications emit canonical BOOL. Any unproved contraction retains the deterministic legal
  split or fails closed.<span data-glossary-navigation="true"> ([raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Schema 59 applies only to BFLOAT16 pointwise projections. Older projections and generated bytes
  remain unchanged and older envelopes are safe misses.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Architecture/current-plan/ADR/architecture-test conclusion: no ownership, module boundary,
  dependency direction, public/shared contract, or architecture rule changed, so no architecture
  document, ADR, or architecture-test edit is required.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- API/lifecycle/test conclusion: Model, Compiler, Runtime, shared Prepare, Tensor, Compile, and
  Training API contracts remain accurate because this is a CPU-private realization of existing
  Model semantics. No conformance/integration, Gradle/build, module-dependency, or unrelated
  documentation change is required. Cross-type CAST remains gated by future Model conversion
  semantics in Draft CPU 0008K.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

Completed the exact 44-form [BFLOAT16](../../../../glossary.md#term-bfloat16) [scalar](../../../../glossary.md#term-scalar) [pointwise](../../../../glossary.md#term-pointwise) closure at [schema](../../../../glossary.md#term-schema) 59, including raw-local
per-[node](../../../../glossary.md#node) [rounding](../../../../glossary.md#term-rounding), raw-preserving WHERE, [canonical](../../../../glossary.md#term-canonical-form) predicates, all current access/[carrier](../../../../glossary.md#term-storage-carrier) forms,
safe [fusion](../../../../glossary.md#term-fusion) splitting, semantic/Class-File evidence, and representative five-[fork](../../../../glossary.md#term-fork-meanings) generated/direct
performance evidence. Finalized affected Javadocs, the [CPU](../../../../glossary.md#term-cpu) guide, established glossary entries,
this task, CPU master plan, and roadmap without changing executable Java or tests in the
documentation pass. CPU 0008I's waived exception remains unchanged; CPU 0008K is the next Draft
frontier with its future [Model](../../../../glossary.md#term-model) CAST-semantics prerequisite, and CPU 0008K–0008P retain their exact
[scope](../../../../glossary.md#term-scope) and order.

Unresolved issues: None for CPU 0008J.

Required follow-up: None for CPU 0008J. CPU 0009 retains the separately recorded missing CPU 0008I
[loss](../../../../glossary.md#loss) evidence and whole-[backend](../../../../glossary.md#backend) evidence inventory responsibility.

Status: Complete
