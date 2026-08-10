# Task 0005I: FLOAT32 vector parity and vector-emission boundary

## Status

Complete

## Goal

Close the intentional FLOAT64-only Java Vector API boundary before the portable CPU backend adds
more operation families. Add truthful preferred-species `FloatVector` execution for the exact
current pointwise opcode subset already eligible for `DoubleVector`, preserve scalar fallback for
every other opcode or access pattern, and move operation-level vector bytecode emission out of the
top-level class generator into one cohesive CPU-private family emitter.

The completed state is:

```text
typed pointwise CpuKernelIr
  -> CPU prepare selects scalar or the preferred species for the one homogeneous floating type
  -> CpuClassFileKernelGenerator owns class, loop, carrier, and store orchestration
  -> CpuVectorInstructionEmitter emits one typed FloatVector or DoubleVector instruction
  -> CpuVectorMath supplies only pure multi-instruction vector formulas
```

This task also makes the existing error-function (`ERF`) and exact/default GELU approximation
auditable. It records the Cephes provenance of the selected piecewise rational approximation,
uses an explicit binary32 coefficient derivation for `FloatVector`, and proves that derivation
against the completed scalar semantics and an independent mathematical oracle. It does not add a
public numerical framework or another CPU operation vocabulary.

## Scope

### Exact vector opcode matrix

Preserve the existing forty-eight `CpuPointwiseOpcode` values and the existing
`vectorEligible()` set. Generalize that set from “implemented only by the FLOAT64 emitter” to
“implemented by both current floating vector types” for exactly these twenty-one opcodes:

| Family | Exact opcodes |
|---|---|
| Binary arithmetic | `ADD`, `SUB`, `MUL`, `DIV` |
| Exact scalar arithmetic | `SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, `SCALAR_DIV`, `SCALAR_POW` |
| Unary | `NEG`, `ABS`, `RECIPROCAL`, `LOG`, `LOG1P`, `EXP`, `EXPM1`, `ERF`, `SQRT`, `RSQRT`, `TANH`, `GELU_EXACT` |

`SCALAR_POW` remains vector-realizable only for the already-proved `POSITIVE_ONE`, `IDENTITY`,
`SQUARE`, and `RECIPROCAL` realizations. `DIRECT` remains scalar. Every other opcode remains
vector-ineligible and selects scalar or parallel-scalar compute without failing capability.

Vector specialization requires all IR values, including virtual values and the output, to have
one exact homogeneous type: either all `FLOAT32` or all `FLOAT64`. Mixed types, canonical-BOOL
outputs, and `WHERE` conditions remain scalar even if individual floating instructions would be
vectorizable.

### Type-aware prepare selection

- Replace the hard-coded `DoubleVector` species selection with one type-aware choice derived from
  the validated homogeneous IR data type.
- `FLOAT32` uses `FloatVector.SPECIES_PREFERRED`; `FLOAT64` preserves
  `DoubleVector.SPECIES_PREFERRED`.
- Minimum element count, contiguous-run eligibility, loop step, scalar tail, parallel ranges,
  recorded species bit size, manifest, specialization identity, and generated validation use the
  selected type's exact lane count.
- `vectorSpeciesBitSize` remains the existing specialization fact. Boundary data types already
  distinguish otherwise equal FLOAT32 and FLOAT64 specializations; add no duplicate lane-type
  field.
- Preserve all five access regimes. Vector execution remains limited to complete direct
  contiguous runs and scalar/all-zero reads under the completed access rules. There is no gather
  or masked tail.
- Preserve CPU 0005D materialization exactly as its current FLOAT64-only optional candidate.
  FLOAT32 direct vector support does not silently broaden copy/materialization selection.

### Vector emission ownership

- Add package-private final `CpuVectorInstructionEmitter` in `internal.codegen.emit`. It is bound
  to one `CodeBuilder` and emits one already-validated vector IR instruction into preallocated
  topology-local vector locals for either `FLOAT32` or `FLOAT64`.
- Move the opcode switch and direct `FloatVector`/`DoubleVector` method-call construction out of
  `CpuClassFileKernelGenerator.emitVectorBody` into that emitter.
- Rename the current package-private `CpuVectorEmitter` helper to package-private final
  `CpuVectorMath`. The new name reflects that it owns pure typed vector formulas and no
  `CodeBuilder`, loop, carrier, strategy, or opcode selection.
- `CpuClassFileKernelGenerator` continues to own deterministic class construction, scalar/vector
  loop choice, boundary loading/storing, local allocation, scalar tails, verification, and hidden
  class definition. Its vector body delegates each instruction exactly once.
- `CpuCarrierEmitter` gains type-aware unmasked vector load, scalar broadcast, and store methods
  for exact `float[]`/`double[]` and native-order `MemorySegment` carriers.
- Add no class per Model operation or opcode. The family emitter retains one closed switch over
  `CpuPointwiseOpcode`; generated hot code contains no opcode, type, carrier, route, or shape
  dispatch.

### FLOAT32 numerical realization

The new FLOAT32 vector path uses the following exact/default rules:

| Group | FLOAT32 vector realization | Required comparison |
|---|---|---|
| `ADD`, `SUB`, `MUL`, `DIV`, scalar counterparts | direct `FloatVector` operation in stored operand order | exact non-NaN result bits against the corresponding generated scalar primitive operation |
| proved scalar power | exact existing positive-one, identity, one-multiply square, or one-division reciprocal sequence | exact non-NaN result bits against the same primitive sequence |
| `NEG`, `ABS`, `RECIPROCAL` | direct vector operation; reciprocal is exact typed `+1.0f / x` | exact fixed special values and zero signs; ordinary finite result within one binary32 ulp of scalar reference |
| `LOG`, `LOG1P`, `EXP`, `EXPM1` | matching Java 26 `VectorOperators` token | at most two binary32 ulps from the completed scalar reference outside fixed overflow/underflow results |
| `SQRT` | `VectorOperators.SQRT` | at most one binary32 ulp from scalar reference and exact fixed special values |
| `RSQRT` | typed positive one divided by vector square root | at most two binary32 ulps from scalar reference; preserve the task-0005H classifications and zero signs |
| `TANH` | `VectorOperators.TANH` | at most five binary32 ulps from scalar reference |
| `ERF` | the selected piecewise rational approximation evaluated entirely in `FloatVector` order | `max(2e-5, 2e-5 * abs(expected))` |
| `GELU_EXACT` | `0.5f*x*(1+erf(x/sqrt(2)))` using the selected FLOAT32 vector `ERF` helper | `max(2e-5, 2e-5 * abs(expected))` |

The tolerances are conformance bounds, not relaxed-math permission or performance claims. The
task preserves the complete exceptional-value table from CPU 0005H, including negative-infinity
GELU producing negative zero. NaN payload and sign remain unspecified where CPU 0005H already
specified classification only.

### ERF and GELU coefficient policy

- Retain the existing selected Cephes double-precision piecewise rational approximation and
  coefficient order for scalar reference and `DoubleVector`. The source is the official
  [Cephes double-precision documentation](https://netlib.org/cephes/doubldoc.html).
- Add dedicated binary32 coefficient arrays to `CpuVectorMath`. Each entry is the exact one-time
  IEEE-754 binary32 rounding of the corresponding already-selected double coefficient and is
  stored as a hexadecimal float literal or another source-stable exact-bit literal. Do not cast
  the double arrays per invocation or reuse `double[]` from `FloatVector` code.
- This deliberate typed derivation is accepted only because boundary-directed, deterministic
  corpus, random differential, and independent-oracle tests must prove the FLOAT32 bounds above.
  The official [Cephes single-precision documentation](https://netlib.org/cephes/singldoc.html)
  is review evidence, but this task does not silently substitute Cephes `erff`'s different
  polynomial family for the already-selected Synaptik approximation.
- Add source/provenance comments beside both retained scalar double tables and vector typed
  tables. Preserve coefficient order. No runtime lookup, mutable coefficient state, reflection,
  resource file, generic polynomial registry, or public numerical API is permitted.
- Implement each `CpuVectorMath.gelu` overload through its matching typed `erf` helper so the
  piecewise algorithm is written once per precision rather than duplicated between `ERF` and
  GELU. Preserve the existing operation order and fixed exceptional-value corrections.
- Scalar and vector packages remain separate access domains. Do not introduce a technically
  public cross-package coefficient holder merely to share private arrays. Differential and
  independent-oracle tests guard the intentional private representation copies.

### Compatibility and lifecycle

- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 8 to 9 because the emitted vector class body,
  helper owner, accepted specialization matrix, and cache-compatible bytecode change.
- Schema 9 rejects every schema 8 or older envelope without a migration reader.
- Preserve deterministic bytes and structural verification. FLOAT32 versus FLOAT64 boundary
  types and exact species bits remain in compatibility bytes and structural keys.
- Preserve one-through-eight connected straight-line pointwise fusion, virtual intermediates,
  one final store, arbitrary `start`/`end`, heap/segment/mixed carriers, scalar tails, optional
  parallel orchestration, four-candidate/one-artifact budgets, zero fixed-shape variants, zero
  unrolled variants, cold binding, and concurrent-run isolation.
- Preserve capability reporting. This task changes a realization strategy, not Model semantic
  support; no occurrence becomes newly supported or unsupported.

## Out of scope

- vectorization of `MIN`, `MAX`, `POW`, `SCALAR_MIN`, `SCALAR_MAX`, `SCALAR_CLAMP`, `FLOOR`,
  `CEIL`, `SIGN`, `RELU`, `SIGMOID`, `GELU_TANH_APPROXIMATION`, `SILU`, classifications,
  comparisons, BOOL logic, `WHERE`, or `CAST`
- integral Vector API lanes, BFLOAT16, FLOAT16, cross-type casts, mixed-precision vector IR, vector
  masks as logical results, gather/scatter vector access, masked tails, or explicit lane
  extraction and scalar reinsertion
- changing scalar numerical realization, scalar/reference result contracts, Model operations,
  compiler rewrites, public APIs, or exact/default versus relaxed numerical policy
- broad emitter architecture, a generic emitter registry, operation-specific emitter classes,
  another IR, general utilities, service locators, reflection, annotations, or string dispatch
- FLOAT32 materialization, fixed-shape/unrolled specialization, new fusion forms, general DAG
  decomposition, layout/indexing/reduction/heavy-family work, vendor/native routes, autotuning,
  benchmark claims, or persistence-policy changes
- dependency, Gradle, Java-version, module-boundary, architecture, backend-conformance, integration,
  Runtime, Prepare-shared, Planning, Compiler, Engine, Metal, CUDA, or OpenBLAS-provider changes
- CPU 0006 or any later detailed task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [CPU backend master plan](../master-plan.md)
- [CPU 0005C vector and parallel portable strategies](0005c-vector-and-parallel-portable-strategies.md)
- [CPU 0005H portable unary closure](0005h-portable-unary-transcendental-and-activation-closure.md)

Technical evidence:

- [Java 26 `FloatVector`](https://docs.oracle.com/en/java/javase/26/docs/api/jdk.incubator.vector/jdk/incubator/vector/FloatVector.html)
- [Java 26 `DoubleVector`](https://docs.oracle.com/en/java/javase/26/docs/api/jdk.incubator.vector/jdk/incubator/vector/DoubleVector.html)
- [Java 26 `VectorOperators`](https://docs.oracle.com/en/java/javase/26/docs/api/jdk.incubator.vector/jdk/incubator/vector/VectorOperators.html)
- [Cephes double-precision functions](https://netlib.org/cephes/doubldoc.html)
- [Cephes single-precision functions](https://netlib.org/cephes/singldoc.html)

## Architecture constraints

- Planning has already selected CPU ownership. CPU analysis alone selects scalar versus Vector
  realization and declares exact shared resources.
- Generated class construction remains CPU finalization work after shared slot assignment;
  Runtime receives only the prepared executable and never selects a type, species, route, or
  operation.
- The generated hot path consumes no Model `Operation`, `CompiledNode`, graph, or tuning state.
- Scalar and Vector API remain routes inside one CPU backend and never become separate backends.
- No dependency direction or supported public CPU surface changes.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — type-aware vector eligibility and
  preferred-species selection during CPU analysis.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — specialization and schema compatibility.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — unchanged closed opcode vocabulary with
  generalized vector-eligibility documentation.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — family vector instruction,
  carrier, loop, class generation, and pure vector math.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — unchanged scalar behavior plus
  approximation provenance.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — review and test of prepared
  parallel-vector FLOAT32 execution; no new production type.

Packages added or changed:

- No package is added, removed, or moved.
- `internal.codegen.emit` replaces one misleading helper type and adds one cohesive emitter type.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuVectorInstructionEmitter` —
  package-private family-oriented vector bytecode emission beside scalar/carrier/loop emitters.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuVectorMath` — package-private pure
  `FloatVector`/`DoubleVector` multi-instruction formulas callable by generated nestmate code.
- `CpuVectorEmitter` is removed with no alias; it is unsupported CPU-private code and is replaced
  atomically by the two accurately named responsibilities.

## Affected files

Expected CPU production/package paths:

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

Expected documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only paths include the completed pointwise lowerer/IR/access/materialization/reference tests;
CPU capability, route, finalizer, memory, and executable production contracts; Model operation,
DataType, Shape, descriptor, and layout contracts; Config/Planning/Compiler/Prepare/Runtime APIs;
Gradle; architecture/ADR/tests; backend conformance; integration; vendor providers; and later CPU
tasks.

## Maximum scope

This task may create, remove, or modify at most 29 paths:

| Category | Maximum | Accounting |
|---|---:|---|
| CPU production/package | 13 | Exact thirteen listed paths, including the one-for-two emitter replacement |
| CPU tests | 11 | Exact eleven listed test paths |
| Explanatory documentation | 2 | CPU backend guide and glossary |
| Planning/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **29** | **13 + 11 + 2 + 3** |

The task exceeds the ordinary guardrail for one documented technical reason: the homogeneous
vector type is one bytecode-shaping fact that must remain consistent across analysis, preferred
species, carrier calls, emitted descriptors, helper descriptors, specialization identity,
artifact compatibility, prepared parallel execution, and numerical conformance. Splitting those
changes would create an invalid partial vector specialization or a temporary duplicate emitter
path. The task adds exactly one net production type and no public surface.

Unused authorized paths may remain unchanged. Stop and revise the task if another production,
test, documentation, build, or module path is required.

## Acceptance criteria

- `FLOAT64` preserves its completed vector behavior and all twenty-one current vector-eligible
  opcodes; `FLOAT32` gains the same exact eligible set with the scalar-power `DIRECT` exception.
- CPU analysis derives one homogeneous floating vector type, exact preferred species, lane count,
  species bits, access eligibility, and scalar tail. Mixed or ineligible IR selects scalar compute.
- Generated FLOAT32 vector bodies load/store exact `float[]` and native-order `MemorySegment`
  boundaries, support scalar broadcasts, arbitrary subranges and tails, and execute through both
  single-thread vector and parallel-vector prepared strategies.
- `CpuClassFileKernelGenerator` no longer owns an operation-level vector opcode switch.
  `CpuVectorInstructionEmitter` owns that one family switch for both precisions, while
  `CpuVectorMath` owns only pure formulas. No per-operation class or alias remains.
- FLOAT32 primitive, transcendental, ERF, and GELU results meet the exact special-value and finite
  conformance matrix. Tests cover adjacent values around `-8`, `-1`, signed zeros, `+1`, and `+8`,
  subnormal/normal and overflow/underflow transitions, infinities, and NaN classification.
- FLOAT32 coefficient values have source-stable exact bits, documented Cephes provenance, fixed
  order, no per-call conversion, and independent-oracle plus scalar differential evidence.
- One-through-eight FLOAT32 vector fusion uses virtual intermediates and one final store. A chain
  containing one vector-ineligible opcode selects scalar or parallel-scalar rather than splitting,
  throwing, or extracting lanes.
- Generator schema is exactly 9. Schema 8 and older artifacts are incompatible misses; type and
  preferred-species facts change identity deterministically while extents, ranges, carriers,
  addresses, run identity, and parallel chunk facts remain excluded as before.
- Existing capability, scalar/reference behavior, access regimes, FLOAT64-only materialization,
  fusion and specialization budgets, persistence-disabled default, unsupported rows, and every
  architecture boundary remain unchanged.
- No code outside `backends/cpu`, no Gradle/shared-module/architecture/test-boundary path, no later
  task specification, and no unrelated refactor appears.
- A separate clean documentation-focused pass finalizes affected Javadocs/package summaries, CPU
  guide, glossary, task/master/roadmap evidence, official links, and no-change conclusions while
  reusing stabilized Java evidence unless executable behavior changes.
- CPU 0005A–0005I are `Complete`; CPU 0006 and every later task remain `Draft` without detailed
  specifications.

## Tests / validation

Run focused tests while developing:

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

The focused matrix must cover both preferred species, all twenty-one eligible opcodes, every
proved scalar-power realization and `DIRECT` fallback, heap/segment/mixed carriers, scalar
broadcast, complete contiguous runs, rejected general odometer, arbitrary `start`/`end`, scalar
tail, zero elements, vector and parallel-vector, fused lengths one and eight, ineligible-chain
fallback, deterministic bytes, specialization identity, schema rejection, coefficient breakpoints,
special values, raw zero signs, directed finite cases, and deterministic random differential data.

After executable Java stabilizes, run exactly one final CPU module suite:

```bash
./gradlew :backends:cpu:test
```

Do not rerun CPU 0005D's opt-in filesystem-persistence timing experiment. This task changes schema
compatibility but not its `KEEP_DISABLED` policy or timing verdict.

The clean documentation-focused pass runs after final Javadocs and Markdown:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates local Markdown links and anchors, official Oracle/Netlib references, balanced
fences, final newlines, exact 29-path scope, the 48-opcode/21-vector-opcode/two-precision matrix,
schema 9, no old `CpuVectorEmitter` live reference, one new family emitter, scalar fallback,
unchanged materialization/fusion/specialization budgets, status synchronization, and the absence
of Java/test changes outside `backends/cpu` or any Gradle/architecture/shared-module path.

Repository-wide validation is deferred to CPU 0009 or continuous integration. The task changes no
dependency or authoritative architecture contract. Current CPU-internal generated/reference/
oracle coverage remains the appropriate conformance evidence before the composed Engine/backend
harness exists.

## Dependencies

- Complete CPU 0005A–0005H partition-kernel, access/broadcast, vector/parallel, materialization,
  typed pointwise, scalar-power, and unary numerical contracts.
- Java 26 Class-File API plus incubating `FloatVector`, `DoubleVector`, `VectorSpecies`, and the
  selected `VectorOperators` tokens.
- Current exact DataType, carrier, `MemorySegment`, layout, staged prepare/finalization, prepared
  execution, schema, and optional artifact-store contracts.

## Follow-up tasks

- CPU 0006 remains Draft without a detailed specification and follows this correction for portable
  layout, indexing, ordering, and explicit-state random family coverage.
- CPU 0007–0008C remain Draft for reduction/multi-pass and heavy families, general DAG
  decomposition, specialized-subgraph recognition, and bounded profitability.
- CPU 0009 remains the portable closure checkpoint and explicitly includes CPU 0005I.
- Integral Vector API, vector extrema/comparison/selection, FLOAT32 materialization, masked tails,
  gather, and relaxed math remain deferred pending concrete family or performance evidence. No
  detailed follow-up specification is created here.

## Architecture impact

Expected impact: None.

This task changes only CPU-owned prepare-time strategy selection and generated portable code. It
does not change module ownership, dependencies, shared lifecycle, public APIs, backend identity,
or Runtime behavior. Stop if implementation requires an architecture change.

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
  0006–0008.
- Use one family vector instruction emitter for both precisions. One class per operation would
  recreate the discarded per-node explosion, while leaving the switch in the top-level generator
  would keep its current mixed orchestration/semantic responsibility.
- Keep `vectorEligible()` as the closed opcode-level necessary condition. Homogeneous data type,
  species, access, element count, and scalar-power realization remain additional prepare gates.
- Preserve the existing approximation family. Dedicated exact-bit binary32 coefficients plus
  proof are safer than per-call casts, while silently adopting another `erff` algorithm would make
  scalar/vector semantics harder to compare.
- Preserve scalar fallback for unsupported vector rows. Vector API availability alone is not a
  correctness or profitability claim.
- Bump schema once to 9; add no compatibility bridge for unsupported private artifacts.
- Keep the existing `vectorSpeciesBitSize` specialization field. The ordered homogeneous boundary
  types already distinguish FLOAT32 from FLOAT64, so another lane-type field would duplicate a
  code-shaping fact without improving compatibility.
- Keep CPU 0005D materialization FLOAT64-only. Direct FLOAT32 vector parity does not supply the
  cost, reuse, or transition evidence required to broaden the copy candidate matrix.
- Let the family instruction emitter own exactly one closed opcode switch for both lane types.
  `CpuVectorMath` owns only typed pure formulas, and the class generator retains loop, carrier,
  local, store, tail, verification, and hidden-class ownership.

## Known limitations

- Only homogeneous FLOAT32 and FLOAT64 pointwise IR can vectorize. Integral, BOOL, mixed-type,
  classification, comparison, selection, and cross-type conversion remain scalar or fail-closed
  according to existing contracts.
- General odometer access, masked tails, gathers, and FLOAT32 materialization remain unsupported.
- Java Vector math operators may be scalarized internally by the JVM; this task proves correctness
  and route construction, not a hardware intrinsic or universal speedup.
- ERF/GELU remain bounded approximations with the completed tolerance and no NaN-payload or
  cross-platform bitwise promise.
- Optional generated-class persistence remains disabled by default. Native routes, tuning,
  broader fusion, and later operation families remain deferred.

## Validation evidence

Planning evidence before promotion to `Ready`:

- Read `AGENTS.md`, the complete authoritative architecture contract, current architecture index,
  planning and documentation rules/profiles, CPU master plan, CPU 0005H, the live generator,
  carrier, prepare, specialization, opcode, scalar-reference contracts, focused tests, CPU guide,
  glossary, and roadmap.
- Confirmed CPU 0005H is Complete, CPU 0006 is Draft without a detailed specification, and no
  existing Ready CPU task blocks insertion of 0005I.
- Confirmed Java 26 exposes preferred `FloatVector` species and the selected floating lane
  operators. Confirmed official Cephes documentation distinguishes double and single precision;
  this task deliberately retains the existing selected approximation and requires a proved,
  source-stable binary32 coefficient derivation rather than an unreviewed cast or algorithm swap.
- Confirmed the change preserves the authoritative backend-owned prepare/lowering/generation and
  Runtime prepared-execution boundaries; no architecture edit is required.
- Planning validation passed for exactly this task, the CPU master plan, and the roadmap. All
  repository-local Markdown targets exist, code fences are balanced, all three files have final
  newlines, and `git diff --check` passed.
- Status validation found exactly one detailed CPU `Ready` task, CPU 0005I. CPU 0006–0009 have no
  detailed task files and remain Draft; completed CPU 0005A–0005H history remains present.
- Scope validation found exactly three planning/documentation paths and no Java, test, Gradle,
  architecture-contract, or cross-module change. No Java or Gradle test was run for this
  planning-only change.

Implementation evidence reused by the documentation pass:

- The implementation pass changed only authorized CPU Java/test/planning paths. Its revised exact
  focused eleven-class command passed 11 suites and 62 tests with zero skips, failures, or errors.
- Its final `./gradlew :backends:cpu:test` passed 26 suites and 117 tests with zero failures or
  errors and one expected opt-in persistence-evidence skip. The persistence timing experiment was
  not rerun and its `KEEP_DISABLED` verdict remains unchanged.
- The first final-suite run exposed only one stale expectation in the newly authorized
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
  emission from pure typed vector mathematics and contain no live `CpuVectorEmitter` page.
- Finalized the CPU guide and glossary for preferred-species FLOAT32/FLOAT64 parity across exactly
  21 of 48 opcodes, homogeneous-type selection, scalar fallback and scalar tails, schema 9,
  unchanged FLOAT64-only materialization, and the absence of universal hardware-acceleration,
  speedup, or relaxed-math claims.
- Documented the retained Cephes double-precision approximation and coefficient order, exact
  one-time IEEE-754 binary32 rounding into source-stable hexadecimal float literals, the reason
  Cephes `erff` is not substituted, and the fixed FLOAT32 conformance bounds. Official Oracle and
  Netlib links were checked from the final documentation state.
- `./gradlew :backends:cpu:javadoc` passed after final Javadocs. Its only two warnings report the
  expected incubating `jdk.incubator.vector` module; there were no missing-Javadoc warnings.
- `ruby /tmp/validate_0005i_markdown.rb` passed for the CPU guide, glossary, this task, CPU master
  plan, and roadmap: 700 repository-local links, 291 heading anchors, balanced fences, final
  newlines, and no trailing whitespace.
- A final `curl --head` check returned HTTP 200 for the Java 26 `FloatVector`, `DoubleVector`, and
  `VectorOperators` pages plus the Netlib Cephes double- and single-precision pages.
- `ruby /tmp/validate_0005i_scope.rb` passed the exact 29-path authorization: 28 changed paths and
  the reviewed unchanged `CpuPointwiseOpcodeTest`, with synchronized 0005I Complete/0006 Draft
  status, no later detailed specification, final newlines, and clean whitespace.
- Manual source/generated-page inventory found exactly 48 opcode constants, 21 vector-eligible
  constants, schema 9, both precisions, package-private final instruction/math helpers, no live old
  `CpuVectorEmitter` source/reference/page, scalar fallback and tails, and unchanged budgets.
  `git diff --check` passed on the final combined tracked diff.

Reasoned no-change conclusions:

- **Public APIs and capability:** both new types are package-private CPU internals, and the sole
  supported public `CpuCapabilityProvider` path is unchanged. The existing 48-opcode semantic
  support matrix is unchanged; CPU 0005I changes realization eligibility only, so capability
  advertisement neither broadens nor narrows.
- **Materialization and budgets:** no materialization-plan or specialization-budget production
  contract changed. CPU 0005D remains FLOAT64-only with direct plus at most three copy candidates,
  one selected copy, one realized artifact, zero fixed-shape variants, and zero unrolled variants.
  The completed one-through-eight straight-line fusion and five access regimes are preserved;
  FLOAT32 parity adds neither a copy candidate nor a fusion/split form.
- **Persistence policy:** schema 9 makes older private envelopes incompatible because emitted bytes
  and compatibility facts changed, but optional persistence remains disabled by default and
  correctness-independent. No timing or policy input changed, so rerunning the opt-in persistence
  experiment would not answer a new policy question.
- **Modules, dependencies, architecture, and shared APIs:** every Java/test change is inside
  `backends/cpu`; no Gradle, shared-module, public Config/Planning/Compiler/Prepare/Runtime API,
  architecture contract, focused architecture page, ADR, or architecture-test path changed. The
  work remains CPU-owned prepare/finalization/code generation under the existing dependency and
  lifecycle boundaries.
- **Backend conformance and integration:** no composed Engine/backend harness or cross-backend
  behavior changed. Current CPU-internal generated/reference/oracle tests are the appropriate
  evidence at this frontier, so backend-conformance and integration paths require no update; CPU
  0009 remains the recorded portable closure checkpoint.
- **Later task specifications:** CPU 0006–0017 remain Draft master-plan rows. No later detailed
  task file was created because this task neither implements nor stabilizes those families.

## Implementation notes

- Added preferred-species FLOAT32 selection without changing the closed 48-opcode vocabulary or
  the 21-opcode eligibility flag. Preparation derives the lane count and species bits from one
  homogeneous FLOAT32 or FLOAT64 IR; mixed, ineligible, general-odometer, and too-short work keeps
  scalar fallback.
- Added exact `float[]` and native-order `MemorySegment` vector load/broadcast/store emission and
  preserved unmasked complete vectors plus the existing scalar body for arbitrary tails and
  parallel subranges.
- Replaced unsupported private `CpuVectorEmitter` atomically with package-private final
  `CpuVectorInstructionEmitter` and `CpuVectorMath`. The former owns the operation-level switch;
  the latter provides typed pure formulas with `FloatVector` and `DoubleVector` overloads.
- Retained the Cephes binary64 coefficient family and added exact source-stable binary32 rounded
  tables. FLOAT32 ERF/GELU conformance is covered by directed boundaries, deterministic random
  differential data, raw special-value checks, and an independent numerical-integration oracle.
- Advanced the generator schema from 8 to 9. Older envelopes are incompatible misses with no
  migration reader; capability reporting, scalar/reference behavior, one-through-eight fusion,
  five access regimes, four-candidate/one-artifact/zero-fixed-shape/zero-unroll budgets, optional
  persistence, and concurrent-run ownership remain unchanged.

## Completion summary

- Completed changes: added truthful preferred-species FLOAT32 parity for the exact existing
  21-opcode vector subset, separated vector instruction emission from pure vector math, recorded
  auditable Cephes/binary32 coefficient provenance and bounds, and advanced compatibility to
  schema 9.
- Files changed or created: the exact 13 authorized CPU production/package paths, 10 of the 11
  authorized CPU test paths, `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task,
  the CPU master plan, and the roadmap; 28 changed paths inside the exact 29-path map, including
  the removed `CpuVectorEmitter.java`. Authorized `CpuPointwiseOpcodeTest.java` required review
  but no edit.
- Tests and validation: reused the implementation pass's focused 11-suite/62-test result and final
  26-suite/117-test result with one expected opt-in persistence skip; documentation validation
  passed CPU Javadoc, local Markdown links/anchors, official links, fences, final newlines, exact
  29-path scope, status, semantic inventory, and `git diff --check`.
- Documentation-agent review: clean documentation context `/root` independently reviewed the
  final implementation/tests and finalized all authorized Javadocs and documentation without
  changing executable behavior or rerunning successful Java tests.
- Documentation impact: CPU guide, glossary, task, CPU master plan, and roadmap now describe the
  exact current realization and retain current-versus-planned boundaries.
- Javadoc review: affected production and package contracts are current; generated CPU Javadoc
  completed with only the two expected incubator warnings.
- Glossary impact: updated existing CPU portable route, generated-kernel, specialization,
  artifact, access, strategy, and scalar-power entries; no public API term was added.
- Unresolved issues: None.
- Follow-up required: None. CPU 0006 and every later CPU task remain Draft without a detailed
  specification.

Status: Complete
