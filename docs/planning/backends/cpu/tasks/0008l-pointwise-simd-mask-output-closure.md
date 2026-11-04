# Task 0008L: [Pointwise](../../../../glossary.md#term-pointwise) [SIMD](../../../../glossary.md#simd) [Mask](../../../../glossary.md#term-boolean-mask) Output Closure

## Reader context

This task records dense `FLOAT32`/`FLOAT64` predicate publication and `WHERE`-condition reload
across the `VectorMask`/canonical-`BOOL` boundary. `CpuCarrierEmitter` uses packed lane bits and a
bounded fixed-species `ByteVector` bridge; private predicates remain live masks. Existing lowering
alone determines whether a graph-required mask gets storage, and same-unit fan-out reuses that mask
after its ordered store.

Here, `CpuPartitionPreparer` owns vector eligibility and fallback, while executable preflight
validates actual `BOOL` bytes before workers or writes. Scalar reference results provide the
semantic oracle; retained Class-Files and clean-Java Vector API comparison provide structural
evidence. Invocation-local bridge masks are neither persistent model state nor workspace. These
historical SIMD measurements do not authorize current production defaults.

The opening arrows describe mask representation changes, not new Model operations: a private
predicate can stay in a vector mask, while a graph-visible result needs canonical Boolean bytes.

Current-topic navigation (not a replacement for the historical record):

- [current generated-class boundary](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit) — Explains emitters, typed entries, hidden classes, and the limits of Class-File inspection.
- [current ordinary-production route policy](../../../../design/notes/cpu-kernel-strategy.md#current-ordinary-production-decision) — Prevents historical vector or parallel evidence from being read as current production-default authorization.

## Status

Complete

## Goal

Close the remaining dense [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [pointwise](../../../../glossary.md#term-pointwise) single-instruction, multiple-data ([SIMD](../../../../glossary.md#simd))
[Vector API](../../../../glossary.md#term-vector-api) boundary for [Boolean](../../../../glossary.md#data-type--datatype) predicate values. Generated [vector](../../../../glossary.md#term-vector) and parallel-vector [kernels](../../../../glossary.md#kernel)
will publish comparison, classification, and floating-[mask](../../../../glossary.md#term-boolean-mask) logical results as [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype)
bytes and reload a dense canonical BOOL condition as the matching floating `VectorMask` for
`WHERE`.

```text
floating vectors -> predicate/logical VectorMask -> canonical 0/1 BOOL boundary
canonical 0/1 BOOL boundary -> bounded ByteVector -> floating VectorMask -> WHERE
same-unit private predicate -> WHERE
  remains a VectorMask and performs no BOOL store/reload
```

“Portable” means generated [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) Class-Files using the Java 26 Vector API. It is not a native,
vendor, or hardware-specific route.

## [Scope](../../../../glossary.md#term-scope)

### Exact semantic and [topology](../../../../glossary.md#term-topology) inventory

The current 48-[opcode](../../../../glossary.md#term-opcode) [pointwise](../../../../glossary.md#term-pointwise) inventory contains exactly 13 [mask](../../../../glossary.md#term-boolean-mask)-related opcodes, or 26 typed
opcode variants across [FLOAT32](../../../../glossary.md#term-float32) and [FLOAT64](../../../../glossary.md#term-float64):

| Role | Exact opcodes | Typed variants | 0008L change |
|---|---|---:|---|
| Classification [producer](../../../../glossary.md#term-producer) | `IS_FINITE`, `IS_NAN`, `IS_INF` | 3 x 2 = 6 | Permit dense [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) output as well as a virtual mask. |
| Comparison producer | `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, `NOT_EQUAL` | 6 x 2 = 12 | Permit dense canonical BOOL output as well as a virtual mask. |
| Mask algebra | `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT` | 3 x 2 = 6 | Permit a final floating-[lane](../../../../glossary.md#term-lane) mask result to be published; preserve BOOL-only `ByteVector` execution. |
| Mask consumer | `WHERE` | 1 x 2 = 2 | Permit a dense canonical BOOL input to be reloaded as a floating-lane mask. |

The 26 typed variants are semantic [operation](../../../../glossary.md#operation) variants. [Publication](../../../../glossary.md#term-publication), reload, and [fan-out](../../../../glossary.md#term-fan-out) are
generated topology roles, not additional [Model](../../../../glossary.md#term-model) operations. Apply this closed role grammar to any
otherwise eligible existing one-through-eight-instruction pointwise [directed acyclic graph](../../../../glossary.md#term-directed-acyclic-graph-dag) ([DAG](../../../../glossary.md#intermediate-representation-dag-and-ssa)):

- FLOAT32/FLOAT64 predicate results may be `VIRTUAL` or dense `OUTPUT` BOOL values.
- Floating-mask logical instructions require mask-represented operands; their result may be
  `VIRTUAL` or a dense `OUTPUT` BOOL value.
- A dense `INPUT` BOOL value may become a floating mask only as the condition of `WHERE`.
- Existing [scalar](../../../../glossary.md#term-scalar)/all-zero BOOL input to `WHERE` remains admitted.
- A private mask used only inside one computation unit remains virtual.
- A mask already required as a [graph](../../../../glossary.md#graph) output or unit boundary may also feed later same-unit
  `WHERE`. One live `VectorMask` feeds both its ordered BOOL store and `WHERE`; it is not
  reloaded, and this task does not create the materialization.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>

External dense BOOL inputs to mixed floating logical operations remain scalar. Integral
comparisons, [BFLOAT16](../../../../glossary.md#term-bfloat16), cross-type CAST, and mixed floating topologies outside this grammar retain
their existing scalar fallback or [fail-closed behavior](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

### [Carrier](../../../../glossary.md#term-storage-carrier), access, strategy, and range support

| [Dimension](../../../../glossary.md#dimension) | [Vector](../../../../glossary.md#term-vector) support | Fallback or boundary |
|---|---|---|
| Floating type | [FLOAT32](../../../../glossary.md#term-float32) preferred species; [FLOAT64](../../../../glossary.md#term-float64) preferred species | No [BFLOAT16](../../../../glossary.md#term-bfloat16) or integral predicate [SIMD](../../../../glossary.md#simd). |
| Carrier | Exact primitive array or [`MemorySegment`](../../../../glossary.md#term-memorysegment) per boundary; every ordered mixed pattern. | No generic carrier [argument](../../../../glossary.md#term-argument) or hot carrier switch. |
| [Layout](../../../../glossary.md#layout) | `DENSE_LINEAR`, including zero and positive cold base offsets. | New [mask](../../../../glossary.md#term-boolean-mask) boundaries in `LAST_AXIS_BIAS`, `BLOCK_OUTER`, or `GENERAL_ODOMETER` use [scalar](../../../../glossary.md#term-scalar)/parallel-scalar. No [gather](../../../../glossary.md#gather)/scatter. |
| Strategy | `VECTOR` and `PARALLEL_VECTOR`, sharing one vector [artifact](../../../../glossary.md#term-artifact). | Existing deterministic `SCALAR` and `PARALLEL_SCALAR` fallback. |
| Range | Any valid `[start,end)`; unmasked full numeric vectors plus existing scalar remainder. | Fewer than one numeric [vector lane](../../../../glossary.md#simd) executes scalar. |
| [Shape](../../../../glossary.md#shape) | Any fully static [rank](../../../../glossary.md#term-rank) with the admitted dense boundary. | Zero [extent](../../../../glossary.md#term-extent) is zero work; rank zero is one scalar [element](../../../../glossary.md#term-element). |

Parallel-vector preserves current configured/available parallelism, minimum-elements policy,
quotient/remainder chunking, disjoint worker ranges, joining, failure behavior, and [run](../../../../glossary.md#term-run) isolation.

### [Canonical](../../../../glossary.md#term-canonical-form) byte [publication](../../../../glossary.md#term-publication) and reload

Materialized [BOOL](../../../../glossary.md#data-type--datatype) remains one byte per [element](../../../../glossary.md#term-element): false is exact byte `0`, true is exact byte `1`,
and no other external byte is valid.

The Java 26 [Vector API](../../../../glossary.md#term-vector-api) has no valid direct floating-[mask](../../../../glossary.md#term-boolean-mask)-to-byte-[vector](../../../../glossary.md#term-vector) conversion.
`VectorMask.cast` requires equal [lane](../../../../glossary.md#term-lane) counts, while a byte vector of the numeric vector's shape
has four times the lanes of a `FloatVector` and eight times those of a `DoubleVector`.
Reinterpreting a mask vector would produce four or eight bytes per logical lane. The platform
preferred byte species is also unsuitable because a scalable-vector implementation may expose
more than 64 byte lanes, beyond `VectorMask.toLong()`'s contract.

For every full numeric chunk:

1. Cold analysis obtains the numeric preferred species and proves
   `1 < numericLanes <= 64`. It then selects the smallest fixed Java 26 byte species among
   `ByteVector.SPECIES_64`, `SPECIES_128`, `SPECIES_256`, and `SPECIES_512` whose lane count
   covers `numericLanes`. This yields `numericLanes <= helperByteLanes <= 64`; otherwise vector
   eligibility fails to scalar fallback. Numeric species bit size remains the only species fact in
   existing specialization because helper species is deterministically derived from it.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
2. Build and invocation-locally reuse
   `VectorMask.fromLong(byteSpecies, lowBits(numericLanes))`. This bounds byte access to the first
   `numericLanes`; it is not a loop-tail mask.
3. Publication packs the numeric mask with `toLong()`, creates the corresponding byte mask with
   `fromLong`, blends exact byte one into a zero `ByteVector`, and performs bounded
   `intoArray` or `intoMemorySegment`.
4. Reload performs bounded `ByteVector.fromArray` or `fromMemorySegment`, compares loaded bytes
   with exact byte one, packs with `toLong()`, and calls `VectorMask.fromLong(numericSpecies,
   bits)`.

Lane zero maps to the least-significant bit. BOOL segment offsets are byte offsets. Numeric segment
access retains element-width scaling and native [byte order](../../../../glossary.md#text-encoding-and-file-boundaries). Full chunks allocate no array/object,
call no `toArray`, execute no [scalar](../../../../glossary.md#term-scalar) per-lane conversion, and access no byte beyond the numeric
lane count. The existing scalar body owns the remainder. Complete canonical-input [validation](../../../../glossary.md#term-validation)
remains a cold [preflight](../../../../glossary.md#term-preflight) before worker submission or output mutation.

### [Fan-out](../../../../glossary.md#term-fan-out), multi-output, [alias](../../../../glossary.md#aliasing-and-arena), and zero work

- Existing lowering alone decides graph publication and unit boundaries. This task adds no
  automatic materialization or profitability policy.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization))</span>
- A [mask](../../../../glossary.md#term-boolean-mask) materialized between units is reloaded by the later unit. No [fusion](../../../../glossary.md#term-fusion) crosses that boundary.
- Existing ordered multi-store publishes a graph-required same-unit mask once and the floating
  result in store order. A virtual mask receives no slot.
- Read-only inputs may alias. Every output span remains disjoint from every input and other output
  span; no in-place BOOL form is added.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Zero elements perform no carrier access, conversion, generated call, or worker submission.
  Positive offset-dense forms retain exact cold offsets and surrounding sentinels.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>

### [Schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), capability, and preparation

- Advance the generated compatibility envelope from 60 to 61.
- Select class-identity schema 61 only for a VECTOR pointwise specialization with one FLOAT32 or
  FLOAT64 lane type and at least one dense non-scalar BOOL <span data-glossary-navigation="true"> ([VECTOR glossary](../../../../glossary.md#term-vector); [pointwise glossary](../../../../glossary.md#term-pointwise); [specialization glossary](../../../../glossary.md#term-specialization); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [lane glossary](../../../../glossary.md#term-lane); [scalar glossary](../../../../glossary.md#term-scalar))</span>`INPUT` or materialized [BOOL](../../../../glossary.md#data-type--datatype) `OUTPUT`.
- Preserve schema 52 and byte-identical classes for unchanged ordinary pointwise forms, including
  virtual-mask-only <span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask))</span>`WHERE`; preserve schema 59 for BFLOAT16 and schema 60 for cross-type CAST.
  Scalar predicate forms retain their existing schema.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Existing IR already records opcode order, types, roles, access plans, stores, and structural key.
  Existing specialization records ordered carriers, strategy, numeric species bit size, numerical
  mode, and power facts. Add no second mask IR or conversion-policy type.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [opcode glossary](../../../../glossary.md#term-opcode))</span>
- Code-shaping/cache facts are schema 61, complete canonical IR, exact ordered carrier pattern,
  VECTOR compute, and numeric preferred-species bit size.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [canonical glossary](../../../../glossary.md#term-canonical-form); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Extents, rank, base offsets, carrier instances, addresses, slots, graph/value/run identities,
  caller bounds, parallel ranges, chunking, workers, and reusable byte-width mask remain cold or
  invocation-local and outside class identity.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [graph glossary](../../../../glossary.md#graph); [run glossary](../../../../glossary.md#term-run))</span>
- `CpuCapabilityProvider` remains semantic and gains no strategy [API](../../../../glossary.md#term-application-programming-interface-api). `CpuPartitionPreparer`
  owns exact eligibility and scalar fallback. Shared Prepare and Runtime remain unchanged.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Add no workspace, materialization candidate, requirement kind, route, tuning input, persistence
  policy, or runtime decision.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [persistence glossary](../../../../glossary.md#term-persistence))</span>

### Generated-code discipline

For each specialized [shape](../../../../glossary.md#shape), a well-written optimal clean Java 26 [Vector API](../../../../glossary.md#term-vector-api) implementation is the
design/review [oracle](../../../../glossary.md#term-test-oracle). Generated [bytecode](../../../../glossary.md#term-bytecode) preserves its semantic algorithm, full-[vector](../../../../glossary.md#term-vector)/[scalar](../../../../glossary.md#term-scalar)-tail
loop, [mask](../../../../glossary.md#term-boolean-mask)-bit/byte-vector [dataflow](../../../../glossary.md#term-dataflow), direct [carrier](../../../../glossary.md#term-storage-carrier) accesses, store order, and avoidable-overhead
profile.

The generated [hot path](../../../../glossary.md#term-hot-path) contains no Synaptik mask helper call, [dispatch](../../../../glossary.md#term-dispatch), allocation, boxing,
reflection, map/string/[opcode](../../../../glossary.md#term-opcode) [lookup](../../../../glossary.md#term-lookup), per-[element](../../../../glossary.md#term-element) virtual call, generic carrier branch, temporary
array, or `VectorMask.toArray`. Existing unrelated chunk-level `CpuVectorMath` calls remain
unchanged and are absent from mask-only evidence forms.

## Out of [scope](../../../../glossary.md#term-scope)

- General CAST [SIMD](../../../../glossary.md#simd) or changes to [CPU](../../../../glossary.md#term-cpu) 0008K.
- [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16) SIMD, mixed promotion, integral predicate SIMD, or non-[canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) storage.
- Non-[contiguous](../../../../glossary.md#term-contiguous) [gather](../../../../glossary.md#gather)/scatter; vectorized `LAST_AXIS_BIAS`, `BLOCK_OUTER`, or general stride;
  negative strides, dynamic layouts, or non-injective outputs.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [layouts glossary](../../../../glossary.md#layout); [injective glossary](../../../../glossary.md#term-injective))</span>
- Native/vendor/OpenBLAS routes, relaxed numerics, automatic materialization policy, autotuning,
  tuning-cache mutation, benchmark-selected behavior, or persistence policy.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [materialization glossary](../../../../glossary.md#materialization); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- General [partition](../../../../glossary.md#partition)-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag), [fusion](../../../../glossary.md#term-fusion), [fan-out](../../../../glossary.md#term-fan-out), [publication](../../../../glossary.md#term-publication), [alias](../../../../glossary.md#aliasing-and-arena), [Runtime](../../../../glossary.md#term-runtime), or shared [Prepare](../../../../glossary.md#prepare) redesign.
- FLOOR, CEIL, [SIGMOID](../../../../glossary.md#term-sigmoid), SILU, [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu)-[TANH](../../../../glossary.md#term-tanh) approximation, exact GELU changes, or general POW SIMD.
- [Vector](../../../../glossary.md#term-vector) [MSE](../../../../glossary.md#mean-squared-error--mse) `NONE`; CPU 0008M is the next [`Ready`](../../../../glossary.md#term-ready) frontier without a detailed specification.
- Changes to Model, Compiler, Planning, Config, Backend Contract, Trace, Engine, Training, other
  backends, architecture contracts, ADRs, module dependencies, or public APIs.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially backend/Prepare/Runtime
  ownership and generated-code discipline.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.
- [CPU backend master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>.
- [CPU 0005C vector and parallel strategies](0005c-vector-and-parallel-portable-strategies.md)<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>.
- [CPU 0005J bounded pointwise vector coverage](0005j-bounded-pointwise-coverage-and-parity-hardening.md)<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise))</span>.
- [CPU 0007A0 generated hot-path parity](0007a0-generated-hot-path-parity-correction.md).
- [CPU 0007A1A scalar-body self-containment](0007a1a-generated-scalar-body-self-containment.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>.
- [CPU 0007A1L pointwise general-loop parity](0007a1l-pointwise-general-loop-residual-parity.md).
- [CPU 0007A1O pointwise ledger reconciliation](0007a1o-pointwise-ledger-evidence-reconciliation.md).
- [CPU 0008B partition-DAG decomposition](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>.
- [CPU 0008E1 shared partition-DAG adoption](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md).
- [CPU 0008J BFLOAT16 scalar pointwise closure](0008j-bfloat16-scalar-pointwise-closure.md)<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>.
- [CPU 0008K cross-type CAST execution](0008k-cross-type-cast-execution.md).
- [CPU backend guide](../../../../backend-guide/cpu-backend.md).
- [Glossary](../../../../glossary.md).

## Architecture constraints

- Planning selects CPU ownership only. CPU analysis owns eligibility, species, specialization,
  direct carrier signature, and scalar fallback.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [specialization glossary](../../../../glossary.md#term-specialization); [carrier glossary](../../../../glossary.md#term-storage-carrier); [signature glossary](../../../../glossary.md#term-signature); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Analysis selects the route and declares unchanged resources before assignment; finalization only
  realizes it. Runtime executes the immutable prepared recipe.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Generated code receives direct typed arguments and sees no Model/Compiler graph object, storage
  discovery, route choice, graph inspection, or backend lookup.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Prepared recipes stay immutable/reusable; each run has isolated mutable state. Existing
  caller-owned workers remain the only parallel resource.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Exact/default predicate, logical, and `WHERE` semantics do not change.
- Any public/shared contract, dependency, lifecycle, architecture, or materialization-policy need
  is a stop condition.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [materialization glossary](../../../../glossary.md#materialization))</span>

## Package impact

Existing packages used:

- `.internal.prepare` — eligibility and [schema](../../../../glossary.md#term-schema) choice.
- `.internal.codegen.emit` — direct conversion, [vector](../../../../glossary.md#term-vector) locals, generation, and evidence.
- `.internal.cache` — schema-61 [identity](../../../../glossary.md#term-identity-meanings) and compatibility.
- `.internal.executable` — unchanged canonicality, spans, overlap, ranges, workers, and failures.
- `.internal.ir` and `.internal.lowering` — unchanged roles, access, stores, [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag), and [publication](../../../../glossary.md#term-publication).
- `.internal.reference` — unchanged independent [scalar](../../../../glossary.md#term-scalar) [oracle](../../../../glossary.md#term-test-oracle).

Packages added or changed:

- No package or responsibility is added or moved.

Type placement:

- No production type is expected; extend existing [carrier](../../../../glossary.md#term-storage-carrier)/generator/preparer [owners](../../../../glossary.md#term-owner).
- Add at most two package-private test types under `.internal.codegen.emit`: one
  semantic/structural evidence owner and one performance owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths (5):

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`

Expected test/evidence/build paths (17):

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuBatchNormTrainingEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv2dEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv3dEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPartitionDagGeneratedEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseLedgerEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseMaskEvidenceTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseMaskPerformanceTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/operation-family-form-ledger-v3.tsv` (new [immutable](../../../../glossary.md#term-immutable) successor)
- `backends/cpu/build.gradle.kts` — guarded evidence properties only.

Expected documentation/[planning](../../../../glossary.md#term-planning) paths (5):

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0008l-pointwise-simd-mask-output-closure.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Reviewed and expected unchanged: the public, [cache](../../../../glossary.md#term-cache), emitter, and preparation package summaries;
`CpuCapabilityProvider`, `CpuKernelIr`, `CpuPointwiseOpcode`, `CpuVectorInstructionEmitter`,
`CpuLoopEmitter`, `CpuScalarEmitter`, [pointwise](../../../../glossary.md#term-pointwise) [lowering](../../../../glossary.md#lowering) production, `CpuPreparedExecutable`,
[scalar](../../../../glossary.md#term-scalar) reference semantics, shared [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/Training APIs, architecture
tests, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration tests, and root Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 27 paths: five production/[Javadoc](../../../../glossary.md#term-javadoc), 17 test/evidence/build, and five
documentation/[planning](../../../../glossary.md#term-planning). At most two test types and no production type or package may be added.

This exceeds the normal range because one generated boundary must update eligibility, direct
emission, compatibility, semantic/structural/performance evidence, [immutable](../../../../glossary.md#term-immutable) ledger [provenance](../../../../glossary.md#provenance),
Javadocs, and [backend](../../../../glossary.md#backend) documentation together, while five prior-family evidence/finalization
[owners](../../../../glossary.md#term-owner) must advance their hard-coded current-envelope assertion from 60 to 61. Those five test
edits authorize assertion maintenance only, not changed prior-family generated bytes or behavior.
Stop and replan if another production [owner](../../../../glossary.md#term-owner), test type, shared/public path, access regime, route,
or architecture edit is required. A listed review path need not change merely because it is
allowed.

## Acceptance criteria

1. Inventory remains 48 opcodes and exactly 13 mask opcodes/26 F32-or-F64 typed variants;
   semantic opcodes are distinguished from publication, reload, and fan-out roles.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [publication glossary](../../../../glossary.md#term-publication))</span>
2. Preparation admits exactly the dense role grammar for VECTOR/PARALLEL_VECTOR, preserves virtual
   masks, and retains existing fallback/fail-closed behavior elsewhere.<span data-glossary-navigation="true"> ([VECTOR glossary](../../../../glossary.md#term-vector); [fail-closed behavior glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
3. Publication/reload use the bounded fixed-byte-species bridge, preserve lane order, access
   exactly numeric lane count, allocate nothing, and retain scalar tails.<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane); [scalar glossary](../../../../glossary.md#term-scalar))</span>
4. Both types, arrays/segments/every mixed pattern, zero/positive offsets, both vector strategies,
   arbitrary ranges, zero/rank-zero/short ranges, aliases/overlaps, canonical rejection,
   multi-output/fan-out, later-unit reload, and concurrency are covered.<span data-glossary-navigation="true"> ([rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [canonical glossary](../../../../glossary.md#term-canonical-form); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
5. [Graph](../../../../glossary.md#graph)-required same-unit [fan-out](../../../../glossary.md#term-fan-out) stores one live [mask](../../../../glossary.md#term-boolean-mask) once and reuses it for `WHERE`; no
   unnecessary materialization or reload is introduced.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
6. Schema advances 60 to 61 only for changed vector bytes. Unchanged schema-52, schema-59, and
   schema-60 classes regenerate byte-identically and incompatible envelopes miss.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema))</span>
7. The exact 72 retained Class-File dossiers below pass complete decompilation, determinism,
   descriptor/member, tail/store-order, and forbidden-overhead checks. Lightweight normalization
   covers omitted ordered carrier combinations without retaining a Cartesian corpus.<span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation); [descriptor glossary](../../../../glossary.md#term-descriptor); [normalization glossary](../../../../glossary.md#term-normalization); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
8. Every [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)-pair ratio, [fork](../../../../glossary.md#term-fork-meanings) [median](../../../../glossary.md#term-median), and aggregate in the exact 12-row performance [matrix](../../../../glossary.md#term-matrix-meanings) is
   `<= 1.15x` its same-algorithm optimal clean Java [oracle](../../../../glossary.md#term-test-oracle), without retry/discard.
9. Invalid BOOL bytes, carriers, spans, inaccessible segments, or overlap fail before worker
   submission and output mutation.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
10. Immutable v2 ledger remains unchanged; tested v3 records 0008L evidence and historical
    provenance.<span data-glossary-navigation="true"> ([Immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance))</span>
11. No public/shared API, workspace, materialization policy, route, tuning, dependency,
    architecture, conformance, integration, or unrelated SIMD behavior changes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance); [SIMD glossary](../../../../glossary.md#simd))</span>
12. A separate clean documentation pass finalizes Javadocs, package summaries, CPU guide, glossary
    impact, task evidence, master plan, and roadmap before Complete. CPU 0008M then becomes the sole
    [`Ready`](../../../../glossary.md#term-ready) [CPU](../../../../glossary.md#term-cpu) frontier with no detailed specification.

## Tests / [validation](../../../../glossary.md#term-validation)

### Semantic execution [matrix](../../../../glossary.md#term-matrix-meanings)

Use 28 explicit all-array, zero-offset, [VECTOR](../../../../glossary.md#term-vector), exact-multiple baseline forms:

```text
direct predicate publication  = (3 classification + 6 comparison) x 2 types = 18
logical publication witnesses = (NOT(IS_FINITE),
                                 AND(GREATER_THAN, IS_FINITE),
                                 OR(LESS_OR_EQUAL, IS_NAN)) x 2 types         =  6
external dense BOOL WHERE     = 1 topology x 2 types                         =  2
required fan-out              = publish GREATER_THAN + WHERE x 2 types       =  2
                                                                              ----
                                                                                28
```

These are evidence forms, not 28 operations. Add 16 boundary-role probes:

```text
2 types x 2 roles (publication, reload) x
  4 paired scenarios (array/zero/vector/exact,
                      segment/offset/vector/tail,
                      numeric-array+BOOL-segment/dense/parallel/exact,
                      numeric-segment+BOOL-array/offset/parallel/tail) = 16

28 baselines + 16 boundary-role probes = 44 normal execution cases
```

Focused tests separately cover all three non-dense fallbacks, zero [extent](../../../../glossary.md#term-extent), [rank](../../../../glossary.md#term-rank) zero, short and
arbitrary non-zero ranges, [canonical](../../../../glossary.md#term-canonical-form) rejection for array/segment, read-only input [alias](../../../../glossary.md#aliasing-and-arena),
input/output and output/output overlap rejection, failure-before-[publication](../../../../glossary.md#term-publication), later-unit reload,
sentinels, concurrent calls, and unchanged [BOOL](../../../../glossary.md#data-type--datatype)-only/virtual-[mask](../../../../glossary.md#term-boolean-mask) controls for both types.

### Class-File [matrix](../../../../glossary.md#term-matrix-meanings)

Retain 72 complete Class-File dossiers. The 28 baselines establish each changed
operator/[topology](../../../../glossary.md#term-topology)/type. [Carrier](../../../../glossary.md#term-storage-carrier) emission has five unique ordered boundary [shapes](../../../../glossary.md#shape):

```text
[F, BOOL]          classification or logical-NOT publication       B=2
[F, F, BOOL]       comparison publication                          B=3
[F, F, F, BOOL]    binary logical publication                      B=4
[BOOL, F, F, F]    external WHERE reload                           B=4
[F, F, BOOL, F]    publication+WHERE fan-out                       B=4
```

For each shape and type, retain one class for each single boundary changed from array to segment,
plus one all-segment class. All-array is already among the baselines:

```text
per-type carrier probes = (2+1) + (3+1) + (4+1) + (4+1) + (4+1) = 22
carrier probes          = 22 x 2 types = 44
retained dossiers       = 28 baselines + 44 probes = 72
```

An automated [normalization](../../../../glossary.md#term-normalization) check may enumerate all ordered carrier patterns but retains only these
72 dossiers. It proves each omitted pattern consists solely of already-proved boundary-local
[descriptor](../../../../glossary.md#term-descriptor)/access substitutions and per-type segment-[layout](../../../../glossary.md#layout) initialization. Any additional
[normalized shape](../../../../glossary.md#normalized-shape) expands the retained matrix or stops the task. Offset, [extent](../../../../glossary.md#term-extent), tail, and parallel
orchestration are cold and do not create classes.

### Performance [matrix](../../../../glossary.md#term-matrix-meanings)

Measure six materially distinct dataflows for both types: classification [publication](../../../../glossary.md#term-publication), comparison
publication, [unary](../../../../glossary.md#term-unary) logical publication, [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) logical publication, external [BOOL](../../../../glossary.md#data-type--datatype) `WHERE`
reload, and publication-plus-`WHERE` [fan-out](../../../../glossary.md#term-fan-out).

| [Dataflow](../../../../glossary.md#term-dataflow) | [FLOAT32](../../../../glossary.md#term-float32) row | [FLOAT64](../../../../glossary.md#term-float64) row |
|---|---|---|
| Classification publication | all array, zero offset, [VECTOR](../../../../glossary.md#term-vector), exact multiple, `IS_NAN` | all segment, positive offset, PARALLEL_VECTOR, tail, `IS_FINITE` |
| Comparison publication | all segment, positive offset, VECTOR, tail, `GREATER_THAN` | all array, zero offset, VECTOR, exact multiple, `EQUAL` |
| Unary logical publication | numeric array/BOOL segment, positive offset, PARALLEL_VECTOR, exact multiple, `NOT(IS_FINITE)` | numeric segment/BOOL array, zero offset, VECTOR, tail, `NOT(IS_NAN)` |
| Binary logical publication | all segment, zero offset, PARALLEL_VECTOR, tail, `AND(GREATER_THAN, IS_FINITE)` | all array, positive offset, VECTOR, exact multiple, `OR(LESS_OR_EQUAL, IS_NAN)` |
| External BOOL `WHERE` reload | BOOL array/numeric segment, positive offset, VECTOR, tail | BOOL segment/numeric array, zero offset, PARALLEL_VECTOR, exact multiple |
| Publication-plus-`WHERE` fan-out | numeric inputs array/outputs segment, zero offset, PARALLEL_VECTOR, exact multiple | numeric inputs segment/outputs array, positive offset, VECTOR, tail |

```text
6 dataflows x 2 types = 12 rows
12 rows x 5 fresh forks x 9 retained randomized sample pairs = 540 sample pairs
12 rows x 5 forks = 60 fork medians
12 median-of-fork-medians aggregates
```

These rows cover array, segment, both mixed directions, zero/positive offset,
VECTOR/PARALLEL_VECTOR, and exact/tail ranges for each type. Every [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)-pair ratio, [fork](../../../../glossary.md#term-fork-meanings) [median](../../../../glossary.md#term-median),
and aggregate is `<= 1.15x`. Use five fresh isolated fixed-[heap](../../../../glossary.md#term-heap) forks, at
least five [warmup](../../../../glossary.md#term-warmup) [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), adaptive minimum-25-ms batches, nine randomized AB/BA rounds, exact
pre/post verification, checksums, and no retry/discard. Timing never selects production behavior.

### Commands

Implementation:

```bash
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBatchNormTrainingEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGeneratorTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConv2dEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConv3dEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPartitionDagGeneratedEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseLedgerEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseMaskEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPointwisePartitionLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest
./gradlew :backends:cpu:test --rerun-tasks
```

Retained evidence:

```bash
CPU_0008L_EVIDENCE_ROOT="$(mktemp -d '/private/tmp/synaptik-cpu-0008l-XXXXXXXX')"
printf '%s\n' "$CPU_0008L_EVIDENCE_ROOT" > "$CPU_0008L_EVIDENCE_ROOT/evidence-root.txt"
touch "$CPU_0008L_EVIDENCE_ROOT/RUN-STRUCTURAL-EVIDENCE"
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseMaskEvidenceTest -Dsynaptik.cpu.pointwiseMask.structuralEvidenceRoot="$CPU_0008L_EVIDENCE_ROOT" --rerun-tasks
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseMaskPerformanceTest -Dsynaptik.cpu.pointwiseMask.performance=true -Dsynaptik.cpu.pointwiseMask.performanceEvidenceRoot="$CPU_0008L_EVIDENCE_ROOT" --rerun-tasks
```

Both evidence [owners](../../../../glossary.md#term-owner) use the same fresh root. Record its exact path/checksums and fail on any
missing or duplicate row, class, [decompilation](../../../../glossary.md#term-decompilation), [fork](../../../../glossary.md#term-fork-meanings), [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), [manifest](../../../../glossary.md#term-manifest) entry, or [checksum](../../../../glossary.md#term-checksum).

Documentation:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

Also validate local links/anchors, fences, final newlines, arithmetic, [schema](../../../../glossary.md#term-schema) `60 -> 61`, exact
27-path ceiling/type placement, synchronized status/order, 0008L Complete, 0008M as the sole [Ready](../../../../glossary.md#term-ready)
[CPU](../../../../glossary.md#term-cpu) frontier without a task file, empty staging, unrelated-work preservation, and no-change
conclusions. Repository-wide Java [validation](../../../../glossary.md#term-validation) is deferred to CPU 0009/[CI](../../../../glossary.md#term-continuous-integration-ci). Shared architecture,
[conformance](../../../../glossary.md#term-conformance), and integration suites are not [run](../../../../glossary.md#term-run) unless a shared boundary changes, which requires
stopping and replanning.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0008K — Complete; [schema](../../../../glossary.md#term-schema)-60 CAST and current [pointwise](../../../../glossary.md#term-pointwise) machinery.
- CPU 0005C/0005J — Complete; preferred-species loops, carriers, worker ranges, virtual masks,
  BOOL-only vectors, and tails.<span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask))</span>
- CPU 0007A0/0007A1A/0007A1L/0007A1O — Complete; generated/direct parity, self-containment,
  fallback, and immutable ledger discipline.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- CPU 0008B/0008E1 — Complete; bounded multi-store [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) and shared DAG adoption.
- CPU 0008J — Complete; [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) pointwise [publication](../../../../glossary.md#term-publication)/[preflight](../../../../glossary.md#term-preflight) and [BFLOAT16](../../../../glossary.md#term-bfloat16) exclusions.
- Java 26.0.1 [Vector](../../../../glossary.md#term-vector)/Class-File APIs through the existing build.

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0008M becomes the sole [`Ready`](../../../../glossary.md#term-ready) frontier and owns only [vector](../../../../glossary.md#term-vector) [MSE](../../../../glossary.md#mean-squared-error--mse) `NONE`; do not create its
  specification in this completion pass.
- CPU 0008N–0008P retain Draft order/[scope](../../../../glossary.md#term-scope).
- CPU 0009 owns generated-coverage reconciliation, including CPU 0008I's gap and 0008L evidence.
- General non-[contiguous](../../../../glossary.md#term-contiguous) [mask](../../../../glossary.md#term-boolean-mask) vectorization and excluded [unary](../../../../glossary.md#term-unary) [SIMD](../../../../glossary.md#simd) gain no task here.

## Architecture impact

Expected impact: None.

Required no-change conclusions:

- Architecture contract/pages/ADRs/tests stay unchanged: ownership, lifecycle, dependencies, and
  generated-code authority do not change.<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Model](../../../../glossary.md#term-model) operations, [`DataType`](../../../../glossary.md#data-type--datatype), `LayoutDescriptor`, Tensor/Compiler/Training APIs, and
  capability-query contracts stay unchanged because semantics/support already exist.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Shared Prepare/Runtime stay unchanged; current resources, slots, finalization, ranges, workers,
  publication, and run isolation suffice.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication); [run glossary](../../../../glossary.md#term-run))</span>
- `CpuKernelIr`, `CpuPointwiseOpcode`, `CpuAccessPlan`, [pointwise](../../../../glossary.md#term-pointwise) [lowering](../../../../glossary.md#lowering) production,
  `CpuLoopEmitter`, `CpuScalarEmitter`, `CpuPreparedExecutable`, and scalar reference
  semantics stay behaviorally unchanged.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Backend-conformance/integration tests stay unchanged because no shared/public behavior owner
  changes; CPU-private evidence covers this route.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Dependencies, plugins, source sets, root Java toolchain, and ordinary tests stay unchanged; CPU
  Gradle forwards guarded evidence properties only.<span data-glossary-navigation="true"> ([toolchain glossary](../../../../glossary.md#term-toolchain); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Other backends, native routes, materialization/tuning policy, and 0008M onward stay behaviorally
  unchanged; only the required 0008M planning status advances to <span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [planning glossary](../../../../glossary.md#term-planning))</span>[`Ready`](../../../../glossary.md#term-ready).
- Prior-family Conv2d, Conv3d, partition-DAG, batch-normalization-training, and finalization
  behavior and generated bytes stay unchanged; their five test owners update only the hard-coded
  current compatibility-envelope assertion required by the repository-wide 60-to-61 advance.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization))</span>

Re-evaluate these conclusions against the final diff and stop if one becomes false.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean-context implementation agent for Synaptik CPU task 0008L.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, task 0008L,
its completed prerequisites, and every affected current CPU source/test owner in full. Implement
the Ready specification within its path/new-type ceilings.

Preserve virtual masks, graph-required publication only, scalar fallback/tails, caller ranges,
canonical BOOL preflight, bounded DAG/store rules, and schema isolation. Implement only dense
FLOAT32/FLOAT64 mask publication/reload. Stop on architecture, shared-contract, semantics, Vector
API, clean-Java parity, evidence, or scope conflict. Do not commit, push, stage, reset, or modify
unrelated work.

After Java/Class-File/performance gates stabilize, hand the exact diff/evidence to a distinct clean
documentation-focused agent under the documentation rules. Do not mark Complete before all
criteria and that pass succeed.
```

## Local decisions

- Same-unit materialized-and-consumed fan-out belongs only when existing graph publication/unit
  boundaries require its store. Reuse the live mask; never materialize a private mask for 0008L.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Packed low [lane](../../../../glossary.md#term-lane) bits plus a bounded fixed-species `ByteVector` is the only planned full-chunk
  bridge. Equal-lane cast, reinterpretation, arrays, and scalar full-chunk loops are rejected.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- The 28 forms are the explicit semantic evidence basis, not the semantic inventory. The inventory
  is 13 opcodes/26 typed variants. Pairwise role probes and normalized structural proof replace
  an all-Cartesian semantic and decompilation corpus.<span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation))</span>
- [Schema](../../../../glossary.md#term-schema) 61 follows changed generated bytes, not the mere presence of a predicate in semantic [IR](../../../../glossary.md#term-intermediate-representation-ir).

## Known limitations

- Only dense F32/F64 mask boundaries vectorize. External BOOL inputs to mixed floating logical
  operations and non-dense boundaries stay scalar.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [BOOL glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- The bridge requires the numeric preferred species to have at most 64 lanes and derives one
  covering fixed byte species with at most 64 lanes. A target failing that proof falls back and
  never calls <span data-glossary-navigation="true"> ([lanes glossary](../../../../glossary.md#term-lane); [target glossary](../../../../glossary.md#term-target))</span>`toLong` on a wider [mask](../../../../glossary.md#term-boolean-mask).
- No scalar speedup is claimed; performance compares generated code with its same-algorithm clean
  Java oracle on the recorded host/JVM.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [host glossary](../../../../glossary.md#term-host); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Pre-implementation [planning](../../../../glossary.md#term-planning) and review evidence:

- Implementation-discovered [scope](../../../../glossary.md#term-scope) correction: after `CpuGeneratorSchema.CURRENT_VERSION`
  advanced from 60 to 61, a focused run executed 22 tests with 10 failures solely in<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
  `CpuPartitionFinalizerTest`, `CpuConv2dEvidenceTest`, `CpuConv3dEvidenceTest`,
  `CpuPartitionDagGeneratedEvidenceTest`, and `CpuBatchNormTrainingEvidenceTest`; each owner
  hard-codes the repository's current envelope/schema value. These five existing test paths are
  therefore required assertion-maintenance owners. They replace five unused production
  allowances (the four package summaries and <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [schema glossary](../../../../glossary.md#term-schema))</span>`CpuVectorInstructionEmitter`), which remain
  review-only and expected unchanged. The exact ceiling remains 27 paths: five
  production/Javadoc + 17 test/evidence/build + five documentation/planning, with at most two new
  test types and no new production type.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

- Replacement planning/review context read in full: `AGENTS.md`, `ARCHITECTURE.md`, current
  architecture index, documentation rules, General/Planning profiles, Planning Guide, CPU master
  plan, this draft, completed 0005C, 0005J, 0007A0, 0007A1A, 0007A1L, 0007A1O, 0008B, 0008E1,
  0008J, and 0008K, plus directly relevant current CPU production/test/package/ledger owners.
  The roadmap CPU status/order sections were read in full.
- Local Java 26.0.1 source confirms least-significant-bit [lane](../../../../glossary.md#term-lane) order, `toLong()`'s at-most-64
  precondition, `fromLong`, equal-length `cast`, and masked `ByteVector` array/segment APIs.
  Current CPU source confirms 48/13/26 inventory, virtual/BOOL-only behavior, dense boundary gap,
  canonical preflight, DAG/store fan-out representation, prior schema-60 ownership, and the
  partial schema-61 isolation change.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [canonical glossary](../../../../glossary.md#term-canonical-form); [preflight glossary](../../../../glossary.md#term-preflight); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [fan-out glossary](../../../../glossary.md#term-fan-out); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Scope-correction validation recounted the three `Affected files` groups as 5 + 17 + 5 = 27,
  found all 630 local Markdown links resolving across this task, the CPU master plan, and the
  roadmap, and confirmed balanced fences and final newlines in all three files. Status/order scans
  found the planned frontier statuses internally consistent at that checkpoint. Final
  whitespace/index/path checks are recorded below and were rerun after every planning edit.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Final planning validation passed for exactly this task, the CPU master plan, and the roadmap:
  all 630 local Markdown links resolve; all three files have balanced fences and final newlines;
  the task has 31 unique heading anchors and no fragment links; all 12 inventory/evidence/scope
  arithmetic assertions passed, and the then-current frontier statuses were internally
  consistent. `git diff --check` and `git diff --cached --check` passed, the cached path list was
  empty, and combined tracked/untracked path enumeration returned exactly the three authorized
  planning paths. No Java test was required or run for this planning-only change.
- Clean documentation-focused context `01a072a9-a208-7931-9feb-90176a70e268` independently
  reviewed the final implementation diff under the General, API/Javadoc, Planning, and backend-
  guide profiles. It read the five changed production owners and their Javadocs, the semantic and
  structural evidence owner, the performance evidence owner, guarded Gradle property forwarding,
  ledger v3, directly affected CPU guide/glossary/planning text, package summaries, and the public
  and shared boundary conclusions. It finalized only the five authorized documentation/planning
  paths and changed no Java, build, test, architecture, conformance, or integration source.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Documentation review confirmed the 48-opcode/13-mask-opcode/26-typed-variant distinction; dense
  F32/F64 publication and reload; virtual-versus-materialized boundary; fixed covering byte
  species; least-significant-bit lane mapping; exact numeric-lane byte access; scalar tails;
  shared VECTOR/PARALLEL_VECTOR artifact; non-dense, mixed-logical, BFLOAT16, integral, and wider-
  than-64-lane fallback boundaries; graph-required store-once fan-out; later-unit reload; and
  schema-61 isolation from schema 52, 59, and 60.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [publication glossary](../../../../glossary.md#term-publication); [scalar glossary](../../../../glossary.md#term-scalar); [VECTOR glossary](../../../../glossary.md#term-vector); [artifact glossary](../../../../glossary.md#term-artifact); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [graph glossary](../../../../glossary.md#graph))</span>
- Javadocs on all five changed production files remain accurate after review. The changed members
  document schema 61, dense floating-mask eligibility, fixed-byte-species setup, direct array or
  segment publication/reload, numeric-lane bounds, and scalar-tail ownership. Package summaries
  remain unchanged because package responsibility, visibility, and lifecycle did not change.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Glossary impact is real but narrow: the existing [`CPU portable route`](../../../../glossary.md#cpu-portable-route) entry now distinguishes
  virtual floating masks from canonical dense BOOL boundaries and records schema-61 isolation.
  No new reusable project term was introduced, so no separate glossary heading was added.
- Public [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/Training APIs, `CpuCapabilityProvider`, shared Prepare/Runtime contracts,
  materialization and tuning policy, dependencies, architecture pages/ADRs/tests, backend-
  conformance tests, and integration tests remain unchanged because this is a CPU-private
  generated-route eligibility and emission closure using existing roles, slots, ranges, workers,
  publication, and run isolation.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [materialization glossary](../../../../glossary.md#materialization))</span>
- Documentation context ran `./gradlew :backends:cpu:javadoc`: it passed with 11 actionable tasks
  (one executed, ten up-to-date) and 53 pre-existing missing-`@param` warnings in unchanged
  `CpuPartitionLowering.LoweredPartition` and `CpuPartitionPreparationPlan.ExecutionUnitPlan`
  constructors, plus the two expected incubating-Vector-module warnings. No warning names a
  0008L-changed source member.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Targeted Markdown validation passed for the five authorized documentation/planning files: 743
  repository-local file links and 96 heading fragments resolve, fences are balanced, every file
  has a final newline, and none has trailing whitespace. Independent arithmetic checks passed for
  26 typed mask variants, 44 semantic/boundary cases, 72 retained dossiers, 540 sample pairs, 60
  fork medians, and 12 aggregates. `git diff --check` and `git diff --cached --check` passed; the
  index is empty. The worktree contains 23 task paths within the 27-path ceiling: five production,
  13 test/evidence/build, and exactly five authorized documentation/planning paths. This bullet
  records only the pre-final-evidence checkpoint, not the current frontier: status advancement was
  deliberately deferred there until final retained evidence became available, and no CPU 0008M
  task file existed. The final status is recorded below.
- Final implementation validation passed. The exact focused command listed under
  [Commands](#commands) completed in 10 seconds. The final uncached
  `./gradlew :backends:cpu:test --rerun-tasks` completed in 26 seconds and reported 715 tests,
  zero failures, zero errors, and 18 expected opt-in skips. No executable Java changed after
  these final runs. Schema-52, schema-59, and schema-60 preservation and selective schema-61
  targeting all passed.
- Final retained structural and performance validation used the shared evidence root
  `/private/tmp/synaptik-cpu-0008l-1lMuloJp`. Structural retention contains 72 Class-Files, 72
  `javap` reports, and 72 unique records in the 73-line [manifest](../../../../glossary.md#term-manifest). The manifest [SHA-256](../../../../glossary.md#term-sha-256) is
  `b7e89b8a9c07c730df5bd382640eec549260cd293e7be264163bebf343b22a57`. Complete inspection found
  no Synaptik helper, allocation, temporary array, boxing, reflection, dispatch,<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
  `CpuVectorMath`, or `VectorMask.toArray` in the changed mask artifacts. The fan-out artifact has
  exactly one publication, one blend, and one `toLong`, then reuses the live [mask](../../../../glossary.md#term-boolean-mask) for `WHERE`.
- The retained performance command completed in 3 minutes 33 seconds. Its exact
  `12 x 5 x 9 = 540` sample pairs, 60 fork medians, and 12 median-of-fork-medians aggregates all
  passed the <span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>`<= 1.15x` gate. [Sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)-pair ratios ranged from `0.1715377840834143x` through
  `0.8171099525620247x`; [fork](../../../../glossary.md#term-fork-meanings) medians ranged from `0.18067302308736136x` through
  `0.7176738221779834x`; and aggregates ranged from `0.24111596838095475x` through
  `0.7030305670814165x`. Individual timed sides ranged from 48,323,667 through 386,857,375 ns, so
  every retained sample exceeded 25 ms. All pre-run, warmup, sample, and post-run equality checks
  passed, with retry and discard both false.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup))</span>
- The performance manifest contains 80 entries and has SHA-256
  `01c84d6f6925d01a71d30879b51f74206b644d7fc9a077372e1db885aea7a55e`. The aggregate SHA-256 is
  `0ecfbb53d921f8bc79064d7cc9764c2b11a76d1abd352f0197de99b9c63b9d23`, and the retained source
  snapshot SHA-256 is <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>`7b0635e8ac4e84e5e7b43abe0528c875d4e485a069ff3f4fa9f02333522ac082`.
- Ledger v2 remains byte-identical with SHA-256
  `41774384007bdb17b011cb8fbaae3b6928baa6ac8c04027a82540977f046031a`; ledger v3 has SHA-256
  `f1b47d1a10472b8c9c2b92bf3bc768e78c80bf9b5f286e6a4025d2098f0c7da7` and records CPU 0008L,
  schema 61, the new evidence owners, and historical provenance.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Final scope is 23 of 27 authorized paths: five production/Javadoc, 13 test/evidence/build, and
  five documentation/planning. Exactly two test types and no production type were added. Final
  CPU Javadoc, local links/fragments, fences, final newlines, whitespace, arithmetic, staging,
  and diff checks passed. CPU 0008L is `Complete`; [CPU](../../../../glossary.md#term-cpu) 0008M is the sole [`Ready`](../../../../glossary.md#term-ready) frontier and has
  no detailed task file.

## Implementation notes

- The implementation extends only the five planned production owners. CPU analysis admits dense
  FLOAT32/FLOAT64 mask inputs and outputs for vector compute, proves the preferred numeric lane
  count is from two through 64, and selects schema 61 only when such a boundary changes generated
  vector bytes.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [vector glossary](../../../../glossary.md#term-vector); [lane glossary](../../../../glossary.md#term-lane); [schema glossary](../../../../glossary.md#term-schema))</span>
- Generated [publication](../../../../glossary.md#term-publication) and reload derive the smallest covering fixed `ByteVector` species,
  preserve least-significant-bit lane order, mask byte access to the numeric lane count, retain
  direct array or native-order segment access, and leave the existing scalar body responsible for
  short ranges and tails.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Virtual [masks](../../../../glossary.md#term-boolean-mask) remain `VectorMask` locals. A graph-required materialized mask that also feeds
  same-unit <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>`WHERE` is stored once and reused live; only a later unit reloads a dense boundary.
  Existing lowering continues to decide publication and materialization.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization))</span>
- Build changes only forward the guarded structural/performance evidence properties. Ledger v3
  names CPU 0008L and schema 61 while retaining the immutable v2 predecessor checksum and prior
  provenance.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [checksum glossary](../../../../glossary.md#term-checksum); [provenance glossary](../../../../glossary.md#provenance))</span>
- Final retained evidence is recorded above and closes every acceptance criterion.

## Completion summary

- Completed changes: Added dense FLOAT32/FLOAT64 vector-mask publication and reload through a
  bounded fixed-<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [vector glossary](../../../../glossary.md#term-vector); [mask glossary](../../../../glossary.md#term-boolean-mask); [publication glossary](../../../../glossary.md#term-publication))</span>`ByteVector` bridge for arrays, segments, and ordered mixed carriers while
  preserving virtual masks, graph-required store-once fan-out, later-unit reload, scalar tails,
  and exact fallback boundaries.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [fan-out glossary](../../../../glossary.md#term-fan-out); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Files changed or created: Five production/Javadoc paths, 13 test/evidence/build paths, and the
  five authorized documentation/planning paths; exactly two test types and no production type
  were added.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: The exact focused command passed in 10 seconds; the final full CPU rerun
  passed in 26 seconds with 715 tests, zero failures/errors, and 18 expected opt-in skips. The
  retained 72-Class-File structural gate and complete 540-sample-pair performance gate passed at<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
  `/private/tmp/synaptik-cpu-0008l-1lMuloJp`, and [CPU](../../../../glossary.md#term-cpu) [Javadoc](../../../../glossary.md#term-javadoc) and documentation [integrity](../../../../glossary.md#term-integrity) passed.
- Documentation-agent review: Clean context `01a072a9-a208-7931-9feb-90176a70e268` independently
  finalized the CPU guide, glossary impact, task evidence, CPU master plan, and roadmap after
  reviewing the five production Javadocs and executable evidence owners.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>
- Documentation impact: The CPU guide now explains the fixed-byte-species bridge, virtual and
  materialized boundaries, lane mapping, carrier/range/tail behavior, fallback limits, fan-out,
  selective schema identity, and generated-versus-optimal-clean-Java evidence boundary.<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane); [carrier glossary](../../../../glossary.md#term-storage-carrier); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Javadoc review: All five changed production Javadocs are accurate; CPU Javadoc passed. Package
  summaries remain unchanged because package ownership and responsibility did not change.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: The existing CPU portable-route entry now distinguishes virtual masks from
  dense canonical BOOL boundaries and records selective schema-61 identity; no new reusable term
  required a separate heading.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Unresolved issues: None.
- Follow-up required: CPU 0008M is the sole [`Ready`](../../../../glossary.md#term-ready) CPU frontier without a detailed task file.

Status: Complete
