# Task 0005I: [FLOAT32](../../../../glossary.md#term-float32) [vector](../../../../glossary.md#term-vector) parity and vector-emission boundary

## Status

Complete

## Goal

Close the intentional [FLOAT64](../../../../glossary.md#term-float64)-only Java [Vector API](../../../../glossary.md#term-vector-api) boundary before the portable [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) adds
more [operation](../../../../glossary.md#operation) families. Add truthful preferred-species `FloatVector` execution for the exact
current [pointwise](../../../../glossary.md#term-pointwise) [opcode](../../../../glossary.md#term-opcode) subset already eligible for `DoubleVector`, preserve [scalar](../../../../glossary.md#term-scalar) fallback for
every other opcode or access pattern, and move operation-level [vector](../../../../glossary.md#term-vector) [bytecode emission](../../../../glossary.md#compiler-engine-and-execution-stage-names) out of the
top-level class generator into one cohesive CPU-private family emitter.

The completed state is:

```text
typed pointwise CpuKernelIr
  -> CPU prepare selects scalar or the preferred species for the one homogeneous floating type
  -> CpuClassFileKernelGenerator owns class, loop, carrier, and store orchestration
  -> CpuVectorInstructionEmitter emits one typed FloatVector or DoubleVector instruction
  -> CpuVectorMath supplies only pure multi-instruction vector formulas
```

This task also makes the existing error-function ([`ERF`](../../../../glossary.md#term-erf)) and exact/default [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) approximation
auditable. It records the Cephes [provenance](../../../../glossary.md#provenance) of the selected piecewise [rational approximation](../../../../glossary.md#range-reduction-and-approximation),
uses an explicit [binary32](../../../../glossary.md#floating-point-precision-and-rounding) coefficient derivation for `FloatVector`, and proves that derivation
against the completed scalar semantics and an independent mathematical [oracle](../../../../glossary.md#term-test-oracle). It does not add a
public numerical framework or another CPU operation vocabulary.

## [Scope](../../../../glossary.md#term-scope)

### Exact [vector](../../../../glossary.md#term-vector) [opcode](../../../../glossary.md#term-opcode) [matrix](../../../../glossary.md#term-matrix-meanings)

Preserve the existing forty-eight `CpuPointwiseOpcode` values and the existing
`vectorEligible()` set. Generalize that set from “implemented only by the [FLOAT64](../../../../glossary.md#term-float64) emitter” to
“implemented by both current floating vector types” for exactly these twenty-one opcodes:

| Family | Exact opcodes |
|---|---|
| [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic | `ADD`, `SUB`, `MUL`, `DIV` |
| Exact [scalar](../../../../glossary.md#term-scalar) arithmetic | `SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, `SCALAR_DIV`, `SCALAR_POW` |
| [Unary](../../../../glossary.md#term-unary) | `NEG`, `ABS`, `RECIPROCAL`, [`LOG`](../../../../glossary.md#term-log), `LOG1P`, [`EXP`](../../../../glossary.md#term-expx), `EXPM1`, [`ERF`](../../../../glossary.md#term-erf), `SQRT`, `RSQRT`, [`TANH`](../../../../glossary.md#term-tanh), `GELU_EXACT` |

`SCALAR_POW` remains vector-realizable only for the already-proved `POSITIVE_ONE`, [`IDENTITY`](../../../../glossary.md#term-identity-meanings),
`SQUARE`, and `RECIPROCAL` realizations. `DIRECT` remains scalar. Every other opcode remains
vector-ineligible and selects scalar or parallel-scalar compute without failing capability.

Vector [specialization](../../../../glossary.md#term-specialization) requires all [IR](../../../../glossary.md#term-intermediate-representation-ir) values, including virtual values and the output, to have
one exact [homogeneous](../../../../glossary.md#term-homogeneous) type: either all [`FLOAT32`](../../../../glossary.md#term-float32) or all `FLOAT64`. Mixed types, [canonical](../../../../glossary.md#term-canonical-form)-[BOOL](../../../../glossary.md#data-type--datatype)
outputs, and `WHERE` conditions remain scalar even if individual floating instructions would be
vectorizable.

### Type-aware [prepare](../../../../glossary.md#prepare) selection

- Replace the hard-coded `DoubleVector` species selection with one type-aware choice derived from
  the validated homogeneous IR data type.<span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- [`FLOAT32`](../../../../glossary.md#term-float32) uses `FloatVector.SPECIES_PREFERRED`; [`FLOAT64`](../../../../glossary.md#term-float64) preserves
  `DoubleVector.SPECIES_PREFERRED`.
- Minimum element count, contiguous-run eligibility, loop step, scalar tail, parallel ranges,
  recorded species bit size, manifest, specialization identity, and generated validation use the
  selected type's exact lane count.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [contiguous glossary](../../../../glossary.md#term-contiguous); [run glossary](../../../../glossary.md#term-run); [scalar tail glossary](../../../../glossary.md#term-scalar-tail); [manifest glossary](../../../../glossary.md#term-manifest); [specialization glossary](../../../../glossary.md#term-specialization); [identity glossary](../../../../glossary.md#term-identity-meanings); [validation glossary](../../../../glossary.md#term-validation); [lane glossary](../../../../glossary.md#term-lane))</span>
- `vectorSpeciesBitSize` remains the existing specialization fact. Boundary data types already
  distinguish otherwise equal FLOAT32 and FLOAT64 specializations; add no duplicate lane-type
  field.
- Preserve all five access regimes. Vector execution remains limited to complete direct
  contiguous runs and scalar/all-zero reads under the completed access rules. There is no gather
  or masked tail.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [scalar glossary](../../../../glossary.md#term-scalar); [gather glossary](../../../../glossary.md#gather))</span>
- Preserve CPU 0005D materialization exactly as its current FLOAT64-only optional candidate.
  FLOAT32 direct vector support does not silently broaden copy/materialization selection.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [materialization glossary](../../../../glossary.md#materialization))</span>

### [Vector](../../../../glossary.md#term-vector) emission [ownership](../../../../glossary.md#term-ownership-meanings)

- Add package-private final `CpuVectorInstructionEmitter` in `internal.codegen.emit`. It is bound
  to one `CodeBuilder` and emits one already-validated vector IR instruction into preallocated
  topology-local vector locals for either <span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [topology glossary](../../../../glossary.md#term-topology))</span>[`FLOAT32`](../../../../glossary.md#term-float32) or [`FLOAT64`](../../../../glossary.md#term-float64).
- Move the [opcode](../../../../glossary.md#term-opcode) switch and direct `FloatVector`/`DoubleVector` method-call construction out of
  `CpuClassFileKernelGenerator.emitVectorBody` into that emitter.
- Rename the current package-private `CpuVectorEmitter` helper to package-private final
  `CpuVectorMath`. The new name reflects that it owns pure typed vector formulas and no
  `CodeBuilder`, loop, [carrier](../../../../glossary.md#term-storage-carrier), strategy, or opcode selection.
- `CpuClassFileKernelGenerator` continues to own deterministic class construction, scalar/vector
  loop choice, boundary loading/storing, local allocation, scalar tails, verification, and hidden
  class definition. Its vector body delegates each instruction exactly once.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [hidden class glossary](../../../../glossary.md#term-hidden-class))</span>
- `CpuCarrierEmitter` gains type-aware unmasked vector load, scalar broadcast, and store methods
  for exact `float[]`/`double[]` and native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment) carriers.
- Add no class per [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation) or opcode. The family emitter retains one closed switch over
  `CpuPointwiseOpcode`; generated hot code contains no opcode, type, carrier, route, or shape
  dispatch.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>

### [FLOAT32](../../../../glossary.md#term-float32) numerical realization

The new FLOAT32 [vector](../../../../glossary.md#term-vector) path uses the following exact/default rules:

| Group | FLOAT32 vector realization | Required comparison |
|---|---|---|
| `ADD`, `SUB`, `MUL`, `DIV`, [scalar](../../../../glossary.md#term-scalar) counterparts | direct `FloatVector` [operation](../../../../glossary.md#operation) in stored operand order | exact non-[NaN](../../../../glossary.md#term-nan) result bits against the corresponding generated scalar primitive operation |
| proved scalar power | exact existing positive-one, [identity](../../../../glossary.md#term-identity-meanings), one-multiply square, or one-division reciprocal sequence | exact non-NaN result bits against the same primitive sequence |
| `NEG`, `ABS`, `RECIPROCAL` | direct vector operation; reciprocal is exact typed `+1.0f / x` | exact fixed special values and zero signs; ordinary [finite](../../../../glossary.md#term-finite) result within one [binary32](../../../../glossary.md#floating-point-precision-and-rounding) ulp of scalar reference |
| [`LOG`](../../../../glossary.md#term-log), `LOG1P`, [`EXP`](../../../../glossary.md#term-expx), `EXPM1` | matching Java 26 `VectorOperators` [token](../../../../glossary.md#term-token) | at most two binary32 ulps from the completed scalar reference outside fixed [overflow](../../../../glossary.md#term-overflow)/[underflow](../../../../glossary.md#term-underflow) results |
| `SQRT` | `VectorOperators.SQRT` | at most one binary32 ulp from scalar reference and exact fixed special values |
| `RSQRT` | typed positive one divided by vector [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions) | at most two binary32 ulps from scalar reference; preserve the task-0005H classifications and zero signs |
| [`TANH`](../../../../glossary.md#term-tanh) | `VectorOperators.TANH` | at most five binary32 ulps from scalar reference |
| [`ERF`](../../../../glossary.md#term-erf) | the selected piecewise [rational approximation](../../../../glossary.md#range-reduction-and-approximation) evaluated entirely in `FloatVector` order | `max(2e-5, 2e-5 * abs(expected))` |
| `GELU_EXACT` | `0.5f*x*(1+erf(x/sqrt(2)))` using the selected FLOAT32 vector `ERF` helper | `max(2e-5, 2e-5 * abs(expected))` |

The tolerances are [conformance](../../../../glossary.md#term-conformance) bounds, not relaxed-math permission or performance claims. The
task preserves the complete exceptional-value table from [CPU](../../../../glossary.md#term-cpu) 0005H, including negative-[infinity](../../../../glossary.md#term-infinity)
[GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) producing [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero). [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero) and sign remain unspecified where CPU 0005H already
specified classification only.

### [ERF](../../../../glossary.md#term-erf) and [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) coefficient policy

- Retain the existing selected Cephes double-precision piecewise rational approximation and
  coefficient order for scalar reference and <span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [rational approximation glossary](../../../../glossary.md#range-reduction-and-approximation); [scalar glossary](../../../../glossary.md#term-scalar))</span>`DoubleVector`. The source is the official
  [Cephes double-precision documentation](https://netlib.org/cephes/doubldoc.html).
- Add dedicated [binary32](../../../../glossary.md#floating-point-precision-and-rounding) coefficient arrays to `CpuVectorMath`. Each entry is the exact one-time
  IEEE-754 binary32 rounding of the corresponding already-selected double coefficient and is
  stored as a hexadecimal float literal or another source-stable exact-bit literal. Do not cast
  the double arrays per invocation or reuse <span data-glossary-navigation="true"> ([IEEE glossary](../../../../glossary.md#term-ieee); [rounding glossary](../../../../glossary.md#term-rounding))</span>`double[]` from `FloatVector` code.
- This deliberate typed derivation is accepted only because boundary-directed, deterministic
  corpus, random differential, and independent-oracle tests must prove the FLOAT32 bounds above.
  The official <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>[Cephes single-precision documentation](https://netlib.org/cephes/singldoc.html)
  is review evidence, but this task does not silently substitute Cephes `erff`'s different
  polynomial family for the already-selected Synaptik approximation.
- Add source/provenance comments beside both retained scalar double tables and vector typed
  tables. Preserve coefficient order. No runtime lookup, mutable coefficient state, reflection,
  resource file, generic polynomial registry, or public numerical API is permitted.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [vector glossary](../../../../glossary.md#term-vector); [runtime glossary](../../../../glossary.md#term-runtime); [lookup glossary](../../../../glossary.md#term-lookup); [mutable glossary](../../../../glossary.md#term-mutable); [registry glossary](../../../../glossary.md#term-registry); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Implement each `CpuVectorMath.gelu` overload through its matching typed `erf` helper so the
  piecewise algorithm is written once per precision rather than duplicated between `ERF` and
  GELU. Preserve the existing operation order and fixed exceptional-value corrections.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Scalar and vector packages remain separate access domains. Do not introduce a technically
  public cross-package coefficient holder merely to share private arrays. Differential and
  independent-oracle tests guard the intentional private representation copies.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>

### Compatibility and [lifecycle](../../../../glossary.md#lifecycle)

- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 8 to 9 because the emitted vector class body,
  helper owner, accepted specialization matrix, and cache-compatible bytecode change.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [owner glossary](../../../../glossary.md#term-owner); [specialization glossary](../../../../glossary.md#term-specialization); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [cache glossary](../../../../glossary.md#term-cache); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- [Schema](../../../../glossary.md#term-schema) 9 rejects every schema 8 or older envelope without a migration reader.
- Preserve deterministic bytes and structural verification. FLOAT32 versus FLOAT64 boundary
  types and exact species bits remain in compatibility bytes and structural keys.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- Preserve one-through-eight connected straight-line pointwise fusion, virtual intermediates,
  one final store, arbitrary <span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [fusion glossary](../../../../glossary.md#term-fusion))</span>`start`/`end`, heap/segment/mixed carriers, scalar tails, optional
  parallel orchestration, four-candidate/one-artifact budgets, zero fixed-shape variants, zero
  unrolled variants, cold binding, and concurrent-run isolation.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [scalar glossary](../../../../glossary.md#term-scalar); [artifact glossary](../../../../glossary.md#term-artifact); [shape glossary](../../../../glossary.md#shape); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [binding glossary](../../../../glossary.md#term-binding); [run glossary](../../../../glossary.md#term-run))</span>
- Preserve capability reporting. This task changes a realization strategy, not Model semantic
  support; no occurrence becomes newly supported or unsupported.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- vectorization of `MIN`, `MAX`, `POW`, `SCALAR_MIN`, `SCALAR_MAX`, `SCALAR_CLAMP`, `FLOOR`,
  `CEIL`, `SIGN`, `RELU`, [`SIGMOID`](../../../../glossary.md#term-sigmoid), `GELU_TANH_APPROXIMATION`, `SILU`, classifications,
  comparisons, BOOL logic, <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`WHERE`, or `CAST`
- integral Vector API lanes, BFLOAT16, FLOAT16, cross-type casts, mixed-precision vector IR, vector
  masks as logical results, gather/scatter vector access, masked tails, or explicit lane
  extraction and scalar reinsertion<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api); [lanes glossary](../../../../glossary.md#term-lane); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [precision glossary](../../../../glossary.md#term-precision); [vector glossary](../../../../glossary.md#term-vector); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [masks glossary](../../../../glossary.md#term-boolean-mask); [gather glossary](../../../../glossary.md#gather); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- changing scalar numerical realization, scalar/reference result contracts, Model operations,
  compiler rewrites, public APIs, or exact/default versus relaxed numerical policy<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- broad emitter architecture, a generic emitter registry, operation-specific emitter classes,
  another IR, general utilities, service locators, reflection, annotations, or string dispatch<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- FLOAT32 materialization, fixed-shape/unrolled specialization, new fusion forms, general DAG
  decomposition, layout/indexing/reduction/heavy-family work, vendor/native routes, autotuning,
  benchmark claims, or persistence-policy changes<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [materialization glossary](../../../../glossary.md#materialization); [shape glossary](../../../../glossary.md#shape); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [layout glossary](../../../../glossary.md#layout); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- dependency, Gradle, Java-version, module-boundary, architecture, backend-conformance, integration,
  Runtime, Prepare-shared, Planning, Compiler, Engine, Metal, CUDA, or OpenBLAS-provider changes<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Planning glossary](../../../../glossary.md#term-planning); [Engine glossary](../../../../glossary.md#term-engine); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- [CPU](../../../../glossary.md#term-cpu) 0006 or any later detailed task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [CPU backend master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0005C vector and parallel portable strategies](0005c-vector-and-parallel-portable-strategies.md)<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>
- [CPU 0005H portable unary closure](0005h-portable-unary-transcendental-and-activation-closure.md)<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>

Technical evidence:

- [Java 26 `FloatVector`](https://docs.oracle.com/en/java/javase/26/docs/api/jdk.incubator.vector/jdk/incubator/vector/FloatVector.html)
- [Java 26 `DoubleVector`](https://docs.oracle.com/en/java/javase/26/docs/api/jdk.incubator.vector/jdk/incubator/vector/DoubleVector.html)
- [Java 26 `VectorOperators`](https://docs.oracle.com/en/java/javase/26/docs/api/jdk.incubator.vector/jdk/incubator/vector/VectorOperators.html)
- [Cephes double-precision functions](https://netlib.org/cephes/doubldoc.html)<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision))</span>
- [Cephes single-precision functions](https://netlib.org/cephes/singldoc.html)

## Architecture constraints

- Planning has already selected CPU ownership. CPU analysis alone selects scalar versus Vector
  realization and declares exact shared resources.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Vector glossary](../../../../glossary.md#term-vector))</span>
- Generated class construction remains CPU finalization work after shared slot assignment;
  Runtime receives only the prepared executable and never selects a type, species, route, or
  operation.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>
- The generated [hot path](../../../../glossary.md#term-hot-path) consumes no [Model](../../../../glossary.md#term-model) [`Operation`](../../../../glossary.md#operation), `CompiledNode`, [graph](../../../../glossary.md#graph), or tuning state.
- [Scalar](../../../../glossary.md#term-scalar) and [Vector API](../../../../glossary.md#term-vector-api) remain routes inside one [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) and never become separate backends.
- No [dependency direction](../../../../glossary.md#java-platform-and-build-tools) or supported public CPU surface changes.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — type-aware vector eligibility and
  preferred-species selection during CPU analysis.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [specialization](../../../../glossary.md#term-specialization) and [schema](../../../../glossary.md#term-schema) compatibility.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — unchanged closed opcode vocabulary with
  generalized vector-eligibility documentation.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — family vector instruction,
  carrier, loop, class generation, and pure vector math.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — unchanged scalar behavior plus
  approximation provenance.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — review and test of prepared
  parallel-vector FLOAT32 execution; no new production type.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32))</span>

Packages added or changed:

- No package is added, removed, or moved.
- `internal.codegen.emit` replaces one misleading helper type and adds one cohesive emitter type.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuVectorInstructionEmitter` —
  package-private family-oriented vector bytecode emission beside scalar/carrier/loop emitters.<span data-glossary-navigation="true"> ([bytecode emission glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuVectorMath` — package-private pure
  `FloatVector`/`DoubleVector` multi-instruction formulas callable by generated [nestmate](../../../../glossary.md#term-nestmate) code.
- `CpuVectorEmitter` is removed with no alias; it is unsupported CPU-private code and is replaced
  atomically by the two accurately named responsibilities.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorEmitter.java` (remove)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorInstructionEmitter.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMath.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcode.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFusedGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMathTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcodeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only paths include the completed [pointwise](../../../../glossary.md#term-pointwise) lowerer/[IR](../../../../glossary.md#term-intermediate-representation-ir)/access/[materialization](../../../../glossary.md#materialization)/reference tests;
CPU capability, route, finalizer, memory, and executable production contracts; [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation),
[DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [descriptor](../../../../glossary.md#term-descriptor), and [layout](../../../../glossary.md#layout) contracts; Config/Planning/[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) APIs;
[Gradle](../../../../glossary.md#term-gradle); architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/tests; [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance); integration; vendor providers; and later CPU
tasks.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create, remove, or modify at most 29 paths:

| Category | Maximum | Accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production/package | 13 | Exact thirteen listed paths, including the one-for-two emitter replacement |
| CPU tests | 11 | Exact eleven listed test paths |
| Explanatory documentation | 2 | CPU [backend](../../../../glossary.md#backend) guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **29** | **13 + 11 + 2 + 3** |

The task exceeds the ordinary guardrail for one documented technical reason: the [homogeneous](../../../../glossary.md#term-homogeneous)
[vector](../../../../glossary.md#term-vector) type is one [bytecode](../../../../glossary.md#term-bytecode)-shaping fact that must remain consistent across analysis, preferred
species, [carrier](../../../../glossary.md#term-storage-carrier) calls, emitted [descriptors](../../../../glossary.md#term-descriptor), helper descriptors, [specialization](../../../../glossary.md#term-specialization) [identity](../../../../glossary.md#term-identity-meanings),
[artifact](../../../../glossary.md#term-artifact) compatibility, prepared parallel execution, and numerical [conformance](../../../../glossary.md#term-conformance). Splitting those
changes would create an invalid partial vector specialization or a temporary duplicate emitter
path. The task adds exactly one net production type and no public surface.

Unused authorized paths may remain unchanged. Stop and revise the task if another production,
test, documentation, build, or [module path](../../../../glossary.md#term-module-path) is required.

## Acceptance criteria

- [`FLOAT64`](../../../../glossary.md#term-float64) preserves its completed vector behavior and all twenty-one current vector-eligible
  opcodes; [`FLOAT32`](../../../../glossary.md#term-float32) gains the same exact eligible set with the [scalar](../../../../glossary.md#term-scalar)-power `DIRECT` exception.
- CPU analysis derives one homogeneous floating vector type, exact preferred species, lane count,
  species bits, access eligibility, and scalar tail. Mixed or ineligible IR selects scalar compute.<span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous); [lane glossary](../../../../glossary.md#term-lane); [scalar tail glossary](../../../../glossary.md#term-scalar-tail); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Generated FLOAT32 [vector](../../../../glossary.md#term-vector) bodies load/store exact `float[]` and native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment)
  boundaries, support scalar broadcasts, arbitrary subranges and tails, and execute through both
  single-thread vector and parallel-vector prepared strategies.
- `CpuClassFileKernelGenerator` no longer owns an [operation](../../../../glossary.md#operation)-level vector [opcode](../../../../glossary.md#term-opcode) switch.
  `CpuVectorInstructionEmitter` owns that one family switch for both precisions, while
  `CpuVectorMath` owns only pure formulas. No per-operation class or [alias](../../../../glossary.md#aliasing-and-arena) remains.
- FLOAT32 primitive, transcendental, ERF, and GELU results meet the exact special-value and finite
  conformance matrix. Tests cover adjacent values around <span data-glossary-navigation="true"> ([ERF glossary](../../../../glossary.md#term-erf); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [finite glossary](../../../../glossary.md#term-finite); [conformance glossary](../../../../glossary.md#term-conformance); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>`-8`, `-1`, signed zeros, `+1`, and `+8`,
  subnormal/normal and overflow/underflow transitions, infinities, and NaN classification.<span data-glossary-navigation="true"> ([subnormal glossary](../../../../glossary.md#term-subnormal); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>
- FLOAT32 coefficient values have source-stable exact bits, documented Cephes provenance, fixed
  order, no per-call conversion, and independent-oracle plus scalar differential evidence.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- One-through-eight FLOAT32 vector fusion uses virtual intermediates and one final store. A chain
  containing one vector-ineligible opcode selects scalar or parallel-scalar rather than splitting,
  throwing, or extracting lanes.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion))</span>
- Generator schema is exactly 9. Schema 8 and older artifacts are incompatible misses; type and
  preferred-species facts change identity deterministically while extents, ranges, carriers,
  addresses, run identity, and parallel chunk facts remain excluded as before.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [artifacts glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [run glossary](../../../../glossary.md#term-run))</span>
- Existing capability, scalar/reference behavior, access regimes, FLOAT64-only materialization,
  fusion and specialization budgets, persistence-disabled default, unsupported rows, and every
  architecture boundary remain unchanged.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [specialization glossary](../../../../glossary.md#term-specialization); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- No code outside `backends/cpu`, no Gradle/shared-module/architecture/test-boundary path, no later
  task specification, and no unrelated refactor appears.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- A separate clean documentation-focused pass finalizes affected Javadocs/package summaries, CPU
  guide, glossary, task/master/roadmap evidence, official links, and no-change conclusions while
  reusing stabilized Java evidence unless executable behavior changes.
- [CPU](../../../../glossary.md#term-cpu) 0005A–0005I are `Complete`; CPU 0006 and every later task remain `Draft` without detailed
  specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests while developing:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcodeTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPointwisePartitionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGeneratorTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuFusedGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuVectorMathTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest
```

The focused [matrix](../../../../glossary.md#term-matrix-meanings) must cover both preferred species, all twenty-one eligible opcodes, every
proved [scalar-power realization](../../../../glossary.md#scalar-power-realization) and `DIRECT` fallback, [heap](../../../../glossary.md#term-heap)/segment/mixed carriers, [scalar](../../../../glossary.md#term-scalar)
broadcast, complete [contiguous](../../../../glossary.md#term-contiguous) runs, rejected general odometer, arbitrary `start`/`end`, [scalar
tail](../../../../glossary.md#term-scalar-tail), zero [elements](../../../../glossary.md#term-element), [vector](../../../../glossary.md#term-vector) and parallel-vector, [fused](../../../../glossary.md#specialization-and-fusion) lengths one and eight, ineligible-chain
fallback, deterministic bytes, [specialization](../../../../glossary.md#term-specialization) [identity](../../../../glossary.md#term-identity-meanings), [schema](../../../../glossary.md#term-schema) rejection, coefficient breakpoints,
special values, raw zero signs, directed [finite](../../../../glossary.md#term-finite) cases, and deterministic random differential data.

After executable Java stabilizes, run exactly one final [CPU](../../../../glossary.md#term-cpu) [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :backends:cpu:test
```

Do not rerun CPU 0005D's opt-in filesystem-[persistence](../../../../glossary.md#term-persistence) timing experiment. This task changes schema
compatibility but not its `KEEP_DISABLED` policy or timing verdict.

The clean documentation-focused pass runs after final Javadocs and Markdown:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates local Markdown links and anchors, official [Oracle](../../../../glossary.md#term-test-oracle)/Netlib references, balanced
fences, final newlines, exact 29-path [scope](../../../../glossary.md#term-scope), the 48-[opcode](../../../../glossary.md#term-opcode)/21-vector-opcode/two-[precision](../../../../glossary.md#term-precision) matrix,
schema 9, no old `CpuVectorEmitter` live reference, one new family emitter, scalar fallback,
unchanged [materialization](../../../../glossary.md#materialization)/[fusion](../../../../glossary.md#term-fusion)/specialization budgets, status [synchronization](../../../../glossary.md#term-synchronization), and the absence
of Java/test changes outside `backends/cpu` or any [Gradle](../../../../glossary.md#term-gradle)/architecture/shared-[module path](../../../../glossary.md#term-module-path).

Repository-wide validation is deferred to CPU 0009 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). The task changes no
[dependency](../../../../glossary.md#term-dependency) or authoritative [architecture contract](../../../../glossary.md#architecture-contract). Current CPU-internal generated/reference/
oracle coverage remains the appropriate [conformance](../../../../glossary.md#term-conformance) evidence before the composed [Engine](../../../../glossary.md#term-engine)/[backend](../../../../glossary.md#backend)
harness exists.

## Dependencies

- Complete CPU 0005A–0005H partition-kernel, access/broadcast, vector/parallel, materialization,
  typed pointwise, scalar-power, and unary numerical contracts.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [kernel glossary](../../../../glossary.md#kernel); [vector glossary](../../../../glossary.md#term-vector); [materialization glossary](../../../../glossary.md#materialization); [pointwise glossary](../../../../glossary.md#term-pointwise); [scalar glossary](../../../../glossary.md#term-scalar); [unary glossary](../../../../glossary.md#term-unary))</span>
- Java 26 [Class-File API](../../../../glossary.md#term-class-file-api) plus incubating `FloatVector`, `DoubleVector`, `VectorSpecies`, and the
  selected `VectorOperators` [tokens](../../../../glossary.md#term-token).
- Current exact [DataType](../../../../glossary.md#data-type--datatype), [carrier](../../../../glossary.md#term-storage-carrier), [`MemorySegment`](../../../../glossary.md#term-memorysegment), layout, staged prepare/finalization, prepared
  execution, schema, and optional artifact-store contracts.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [prepare glossary](../../../../glossary.md#prepare); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [schema glossary](../../../../glossary.md#term-schema); [artifact glossary](../../../../glossary.md#term-artifact))</span>

## Follow-up tasks

- CPU 0006 remains Draft without a detailed specification and follows this correction for portable
  layout, indexing, ordering, and explicit-state random family coverage.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- CPU 0007–0008C remain Draft for reduction/multi-pass and heavy families, general DAG
  decomposition, specialized-subgraph recognition, and bounded profitability.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- [CPU](../../../../glossary.md#term-cpu) 0009 remains the portable closure [checkpoint](../../../../glossary.md#term-checkpoint) and explicitly includes CPU 0005I.
- Integral Vector API, vector extrema/comparison/selection, FLOAT32 materialization, masked tails,
  gather, and relaxed math remain deferred pending concrete family or performance evidence. No
  detailed follow-up specification is created here.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api); [vector glossary](../../../../glossary.md#term-vector); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [FLOAT32 glossary](../../../../glossary.md#term-float32); [materialization glossary](../../../../glossary.md#materialization); [gather glossary](../../../../glossary.md#gather))</span>

## Architecture impact

Expected impact: None.

This task changes only [CPU](../../../../glossary.md#term-cpu)-owned [prepare](../../../../glossary.md#prepare)-time strategy selection and generated portable code. It
does not change [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), dependencies, shared [lifecycle](../../../../glossary.md#lifecycle), public APIs, [backend identity](../../../../glossary.md#backend-identity--backendid),
or [Runtime](../../../../glossary.md#term-runtime) behavior. Stop if implementation requires an architecture change.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, completed
CPU tasks 0005A–0005H, and
docs/planning/backends/cpu/tasks/0005i-float32-vector-parity-and-vector-emission-boundary.md in
full. Inspect every affected and review-only source/test named by task 0005I before editing.

Implement CPU 0005I exactly inside its exact 29-path map. Add truthful preferred-species FLOAT32
vector parity for the closed current eligible opcode set, preserve FLOAT64 and scalar behavior,
separate vector instruction emission from pure vector math, and prove the selected typed ERF/GELU
coefficient derivation and conformance. Stop on architecture, dependency, numerical-proof,
completed-contract, affected-file, or maximum-scope conflict. Do not implement later families,
broader vectorization, relaxed math, dependencies, build, architecture changes, commits, or pushes.

Run focused tests while developing and exactly one final CPU suite after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent/thread. That pass follows documentation-rules.md, finalizes Javadocs,
CPU guide, glossary, task/master/roadmap evidence, official links and documentation/scope checks,
and reuses successful Java evidence unless executable behavior changes or a concrete stale-evidence
risk is recorded.

Do not mark 0005I Complete until both passes and every acceptance criterion succeed. Leave CPU
0006 and every later task Draft without detailed specifications.
```

## Local decisions

- Insert 0005I before family expansion because current prepare, carrier, generator, helper, and
  compatibility contracts encode FLOAT64 assumptions that would otherwise spread into tasks
  0006–0008.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [carrier glossary](../../../../glossary.md#term-storage-carrier); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- Use one family vector instruction emitter for both precisions. One class per operation would
  recreate the discarded per-node explosion, while leaving the switch in the top-level generator
  would keep its current mixed orchestration/semantic responsibility.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [operation glossary](../../../../glossary.md#operation); [node glossary](../../../../glossary.md#node))</span>
- Keep `vectorEligible()` as the closed opcode-level necessary condition. Homogeneous data type,
  species, access, element count, and scalar-power realization remain additional prepare gates.<span data-glossary-navigation="true"> ([Homogeneous glossary](../../../../glossary.md#term-homogeneous); [data type glossary](../../../../glossary.md#data-type--datatype); [element glossary](../../../../glossary.md#term-element); [scalar-power realization glossary](../../../../glossary.md#scalar-power-realization))</span>
- Preserve the existing approximation family. Dedicated exact-bit binary32 coefficients plus
  proof are safer than per-call casts, while silently adopting another <span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`erff` algorithm would make
  scalar/vector semantics harder to compare.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Preserve scalar fallback for unsupported vector rows. Vector API availability alone is not a
  correctness or profitability claim.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api))</span>
- Bump [schema](../../../../glossary.md#term-schema) once to 9; add no compatibility bridge for unsupported private [artifacts](../../../../glossary.md#term-artifact).
- Keep the existing `vectorSpeciesBitSize` specialization field. The ordered homogeneous boundary
  types already distinguish FLOAT32 from FLOAT64, so another lane-type field would duplicate a
  code-shaping fact without improving compatibility.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- Keep CPU 0005D materialization FLOAT64-only. Direct FLOAT32 vector parity does not supply the
  cost, reuse, or transition evidence required to broaden the copy candidate matrix.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [materialization glossary](../../../../glossary.md#materialization); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Let the family instruction emitter own exactly one closed [opcode](../../../../glossary.md#term-opcode) switch for both [lane](../../../../glossary.md#term-lane) types.
  `CpuVectorMath` owns only typed pure formulas, and the class generator retains loop, carrier,
  local, store, tail, verification, and hidden-class ownership.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Known limitations

- Only homogeneous FLOAT32 and FLOAT64 pointwise IR can vectorize. Integral, BOOL, mixed-type,
  classification, comparison, selection, and cross-type conversion remain scalar or fail-closed
  according to existing contracts.<span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous); [FLOAT64 glossary](../../../../glossary.md#term-float64); [pointwise glossary](../../../../glossary.md#term-pointwise); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [BOOL glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- General odometer access, masked tails, gathers, and [FLOAT32](../../../../glossary.md#term-float32) [materialization](../../../../glossary.md#materialization) remain unsupported.
- Java Vector math operators may be scalarized internally by the JVM; this task proves correctness
  and route construction, not a hardware intrinsic or universal speedup.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- ERF/GELU remain bounded approximations with the completed tolerance and no NaN-payload or
  cross-platform bitwise promise.<span data-glossary-navigation="true"> ([ERF glossary](../../../../glossary.md#term-erf); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [tolerance glossary](../../../../glossary.md#term-tolerance); [NaN glossary](../../../../glossary.md#term-nan); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Optional generated-class persistence remains disabled by default. Native routes, tuning,
  broader fusion, and later operation families remain deferred.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [fusion glossary](../../../../glossary.md#term-fusion); [operation glossary](../../../../glossary.md#operation))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) evidence before promotion to [`Ready`](../../../../glossary.md#term-ready):

- Read `AGENTS.md`, the complete authoritative architecture contract, current architecture index,
  planning and documentation rules/profiles, CPU master plan, CPU 0005H, the live generator,
  carrier, prepare, specialization, opcode, scalar-reference contracts, focused tests, CPU guide,
  glossary, and roadmap.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [carrier glossary](../../../../glossary.md#term-storage-carrier); [prepare glossary](../../../../glossary.md#prepare); [specialization glossary](../../../../glossary.md#term-specialization); [opcode glossary](../../../../glossary.md#term-opcode))</span>
- Confirmed CPU 0005H is Complete, CPU 0006 is Draft without a detailed specification, and no
  existing Ready CPU task blocks insertion of 0005I.
- Confirmed Java 26 exposes preferred `FloatVector` species and the selected floating lane
  operators. Confirmed official Cephes documentation distinguishes double and single precision;
  this task deliberately retains the existing selected approximation and requires a proved,
  source-stable binary32 coefficient derivation rather than an unreviewed cast or algorithm swap.<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- Confirmed the change preserves the authoritative backend-owned prepare/lowering/generation and
  Runtime prepared-execution boundaries; no architecture edit is required.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Planning validation passed for exactly this task, the CPU master plan, and the roadmap. All
  repository-local Markdown targets exist, code fences are balanced, all three files have final
  newlines, and <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>`git diff --check` passed.
- Status validation found exactly one detailed [CPU](../../../../glossary.md#term-cpu) `Ready` task, CPU 0005I. CPU 0006–0009 have no
  detailed task files and remain Draft; completed CPU 0005A–0005H history remains present.
- Scope validation found exactly three planning/documentation paths and no Java, test, Gradle,
  architecture-contract, or cross-module change. No Java or Gradle test was run for this
  planning-only change.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

Implementation evidence reused by the documentation pass:

- The implementation pass changed only authorized CPU Java/test/planning paths. Its revised exact
  focused eleven-class command passed 11 suites and 62 tests with zero skips, failures, or errors.
- Its final `./gradlew :backends:cpu:test` passed 26 suites and 117 tests with zero failures or
  errors and one expected opt-in persistence-evidence skip. The persistence timing experiment was
  not rerun and its `KEEP_DISABLED` verdict remains unchanged.
- The first final-suite [run](../../../../glossary.md#term-run) exposed only one stale expectation in the newly authorized
  `CpuPointwisePartitionLoweringTest`; the implementation follow-up changed that expectation and
  no production behavior. The successful 117-test result is the post-correction evidence.
- `javap` confirmed package-private final `CpuVectorMath` and
  `CpuVectorInstructionEmitter`, each with `FloatVector` and `DoubleVector` overload or emission
  support as specified. Implementation `git diff --check` passed.
- Clean documentation context `/root` inspected the actual final diff, all changed production and
  test paths, affected generated CPU Javadoc, the CPU guide, glossary, task, master plan, and
  roadmap. It changed no executable behavior and therefore did not repeat either successful Java
  test command.

Documentation-focused evidence:

- Applied the General, API/Javadoc, Backend Guide, Developer Guide example, Planning, and glossary
  terminology rules. Finalized the affected cache/specialization, generator/emitter, carrier, IR,
  prepare, reference, and package Javadocs. Generated pages distinguish family instruction
  emission from pure typed vector mathematics and contain no live <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [cache glossary](../../../../glossary.md#term-cache); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [vector glossary](../../../../glossary.md#term-vector))</span>`CpuVectorEmitter` page.
- Finalized the CPU guide and glossary for preferred-species FLOAT32/FLOAT64 parity across exactly
  21 of 48 opcodes, homogeneous-type selection, scalar fallback and scalar tails, schema 9,
  unchanged FLOAT64-only materialization, and the absence of universal hardware-acceleration,
  speedup, or relaxed-math claims.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [homogeneous glossary](../../../../glossary.md#term-homogeneous); [schema glossary](../../../../glossary.md#term-schema))</span>
- Documented the retained Cephes double-precision approximation and coefficient order, exact
  one-time IEEE-754 binary32 rounding into source-stable hexadecimal float literals, the reason
  Cephes <span data-glossary-navigation="true"> ([IEEE glossary](../../../../glossary.md#term-ieee); [rounding glossary](../../../../glossary.md#term-rounding))</span>`erff` is not substituted, and the fixed FLOAT32 conformance bounds. Official Oracle and
  Netlib links were checked from the final documentation state.<span data-glossary-navigation="true"> ([Oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- `./gradlew :backends:cpu:javadoc` passed after final Javadocs. Its only two warnings report the
  expected incubating `jdk.incubator.vector` [module](../../../../glossary.md#term-module); there were no missing-[Javadoc](../../../../glossary.md#term-javadoc) warnings.
- `ruby /tmp/validate_0005i_markdown.rb` passed for the CPU guide, glossary, this task, CPU master
  plan, and roadmap: 700 repository-local links, 291 heading anchors, balanced fences, final
  newlines, and no trailing whitespace.
- A final `curl --head` check returned HTTP 200 for the Java 26 `FloatVector`, `DoubleVector`, and
  `VectorOperators` pages plus the Netlib Cephes double- and single-[precision](../../../../glossary.md#term-precision) pages.
- `ruby /tmp/validate_0005i_scope.rb` passed the exact 29-path authorization: 28 changed paths and
  the reviewed unchanged `CpuPointwiseOpcodeTest`, with synchronized 0005I Complete/0006 Draft
  status, no later detailed specification, final newlines, and clean whitespace.
- Manual source/generated-page inventory found exactly 48 opcode constants, 21 vector-eligible
  constants, schema 9, both precisions, package-private final instruction/math helpers, no live old<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>
  `CpuVectorEmitter` source/reference/page, [scalar](../../../../glossary.md#term-scalar) fallback and tails, and unchanged budgets.
  `git diff --check` passed on the final combined tracked diff.

Reasoned no-change conclusions:

- **Public APIs and capability:** both new types are package-private CPU internals, and the sole
  supported public `CpuCapabilityProvider` path is unchanged. The existing 48-opcode semantic
  support matrix is unchanged; CPU 0005I changes realization eligibility only, so capability
  advertisement neither broadens nor narrows.<span data-glossary-navigation="true"> ([support matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span>
- **[Materialization](../../../../glossary.md#materialization) and budgets:** no materialization-plan or specialization-budget production
  contract changed. CPU 0005D remains FLOAT64-only with direct plus at most three copy candidates,
  one selected copy, one realized artifact, zero fixed-shape variants, and zero unrolled variants.
  The completed one-through-eight straight-line fusion and five access regimes are preserved;
  FLOAT32 parity adds neither a copy candidate nor a fusion/split form.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [shape glossary](../../../../glossary.md#shape); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- **[Persistence](../../../../glossary.md#term-persistence) policy:** schema 9 makes older private envelopes incompatible because emitted bytes
  and compatibility facts changed, but optional persistence remains disabled by default and
  correctness-independent. No timing or policy input changed, so rerunning the opt-in persistence
  experiment would not answer a new policy question.
- **Modules, dependencies, architecture, and shared APIs:** every Java/test change is inside
  `backends/cpu`; no Gradle, shared-module, public Config/Planning/Compiler/Prepare/Runtime API,
  architecture contract, focused architecture page, ADR, or architecture-test path changed. The
  work remains CPU-owned prepare/finalization/code generation under the existing dependency and
  lifecycle boundaries.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [code generation glossary](../../../../glossary.md#term-code-generation); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- **[Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) and integration:** no composed Engine/backend harness or cross-backend
  behavior changed. Current CPU-internal generated/reference/oracle tests are the appropriate
  evidence at this frontier, so backend-conformance and integration paths require no update; CPU
  0009 remains the recorded portable closure checkpoint.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- **Later task specifications:** CPU 0006–0017 remain Draft master-plan rows. No later detailed
  task file was created because this task neither implements nor stabilizes those families.

## Implementation notes

- Added preferred-species FLOAT32 selection without changing the closed 48-opcode vocabulary or
  the 21-opcode eligibility flag. Preparation derives the lane count and species bits from one
  homogeneous FLOAT32 or FLOAT64 IR; mixed, ineligible, general-odometer, and too-short work keeps
  scalar fallback.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [opcode glossary](../../../../glossary.md#term-opcode); [lane glossary](../../../../glossary.md#term-lane); [homogeneous glossary](../../../../glossary.md#term-homogeneous); [FLOAT64 glossary](../../../../glossary.md#term-float64); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Added exact `float[]` and native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment) vector load/broadcast/store emission and
  preserved unmasked complete vectors plus the existing scalar body for arbitrary tails and
  parallel subranges.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>
- Replaced unsupported private `CpuVectorEmitter` atomically with package-private final
  `CpuVectorInstructionEmitter` and `CpuVectorMath`. The former owns the operation-level switch;
  the latter provides typed pure formulas with <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>`FloatVector` and `DoubleVector` overloads.
- Retained the Cephes binary64 coefficient family and added exact source-stable binary32 rounded
  tables. FLOAT32 ERF/GELU conformance is covered by directed boundaries, deterministic random
  differential data, raw special-value checks, and an independent numerical-integration oracle.<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [ERF glossary](../../../../glossary.md#term-erf); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [conformance glossary](../../../../glossary.md#term-conformance); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Advanced the generator schema from 8 to 9. Older envelopes are incompatible misses with no
  migration reader; capability reporting, scalar/reference behavior, one-through-eight fusion,
  five access regimes, four-candidate/one-artifact/zero-fixed-shape/zero-unroll budgets, optional
  persistence, and concurrent-run ownership remain unchanged.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [fusion glossary](../../../../glossary.md#term-fusion); [artifact glossary](../../../../glossary.md#term-artifact); [shape glossary](../../../../glossary.md#shape); [persistence glossary](../../../../glossary.md#term-persistence); [run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Completion summary

- Completed changes: added truthful preferred-species FLOAT32 parity for the exact existing
  21-opcode vector subset, separated vector instruction emission from pure vector math, recorded
  auditable Cephes/binary32 coefficient provenance and bounds, and advanced compatibility to
  schema 9.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [opcode glossary](../../../../glossary.md#term-opcode); [vector glossary](../../../../glossary.md#term-vector); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [provenance glossary](../../../../glossary.md#provenance); [schema glossary](../../../../glossary.md#term-schema))</span>
- Files changed or created: the exact 13 authorized CPU production/package paths, 10 of the 11
  authorized CPU test paths, <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>`docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task,
  the CPU master plan, and the roadmap; 28 changed paths inside the exact 29-path map, including
  the removed `CpuVectorEmitter.java`. Authorized `CpuPointwiseOpcodeTest.java` required review
  but no edit.
- Tests and validation: reused the implementation pass's focused 11-suite/62-test result and final
  26-suite/117-test result with one expected opt-in persistence skip; documentation validation
  passed CPU Javadoc, local Markdown links/anchors, official links, fences, final newlines, exact
  29-path scope, status, semantic inventory, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [persistence glossary](../../../../glossary.md#term-persistence); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.
- Documentation-agent review: clean documentation context `/root` independently reviewed the
  final implementation/tests and finalized all authorized Javadocs and documentation without
  changing executable behavior or rerunning successful Java tests.
- Documentation impact: CPU guide, glossary, task, CPU master plan, and roadmap now describe the
  exact current realization and retain current-versus-planned boundaries.
- Javadoc review: affected production and package contracts are current; generated CPU Javadoc
  completed with only the two expected incubator warnings.
- Glossary impact: updated existing CPU portable route, generated-kernel, specialization,
  artifact, access, strategy, and scalar-power entries; no public API term was added.<span data-glossary-navigation="true"> ([CPU portable route glossary](../../../../glossary.md#cpu-portable-route); [kernel glossary](../../../../glossary.md#kernel); [specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [scalar glossary](../../../../glossary.md#term-scalar); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Unresolved issues: None.
- Follow-up required: None. CPU 0006 and every later CPU task remain Draft without a detailed
  specification.

Status: Complete
