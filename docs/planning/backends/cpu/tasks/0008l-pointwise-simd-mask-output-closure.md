# Task 0008L: Pointwise SIMD Mask Output Closure

## Status

Complete

## Goal

Close the remaining dense FLOAT32/FLOAT64 pointwise single-instruction, multiple-data (SIMD)
Vector API boundary for Boolean predicate values. Generated vector and parallel-vector kernels
will publish comparison, classification, and floating-mask logical results as canonical BOOL
bytes and reload a dense canonical BOOL condition as the matching floating `VectorMask` for
`WHERE`.

```text
floating vectors -> predicate/logical VectorMask -> canonical 0/1 BOOL boundary
canonical 0/1 BOOL boundary -> bounded ByteVector -> floating VectorMask -> WHERE
same-unit private predicate -> WHERE
  remains a VectorMask and performs no BOOL store/reload
```

“Portable” means generated JVM Class-Files using the Java 26 Vector API. It is not a native,
vendor, or hardware-specific route.

## Scope

### Exact semantic and topology inventory

The current 48-opcode pointwise inventory contains exactly 13 mask-related opcodes, or 26 typed
opcode variants across FLOAT32 and FLOAT64:

| Role | Exact opcodes | Typed variants | 0008L change |
|---|---|---:|---|
| Classification producer | `IS_FINITE`, `IS_NAN`, `IS_INF` | 3 x 2 = 6 | Permit dense canonical BOOL output as well as a virtual mask. |
| Comparison producer | `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, `NOT_EQUAL` | 6 x 2 = 12 | Permit dense canonical BOOL output as well as a virtual mask. |
| Mask algebra | `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT` | 3 x 2 = 6 | Permit a final floating-lane mask result to be published; preserve BOOL-only `ByteVector` execution. |
| Mask consumer | `WHERE` | 1 x 2 = 2 | Permit a dense canonical BOOL input to be reloaded as a floating-lane mask. |

The 26 typed variants are semantic operation variants. Publication, reload, and fan-out are
generated topology roles, not additional Model operations. Apply this closed role grammar to any
otherwise eligible existing one-through-eight-instruction pointwise directed acyclic graph (DAG):

- FLOAT32/FLOAT64 predicate results may be `VIRTUAL` or dense `OUTPUT` BOOL values.
- Floating-mask logical instructions require mask-represented operands; their result may be
  `VIRTUAL` or a dense `OUTPUT` BOOL value.
- A dense `INPUT` BOOL value may become a floating mask only as the condition of `WHERE`.
- Existing scalar/all-zero BOOL input to `WHERE` remains admitted.
- A private mask used only inside one computation unit remains virtual.
- A mask already required as a graph output or unit boundary may also feed later same-unit
  `WHERE`. One live `VectorMask` feeds both its ordered BOOL store and `WHERE`; it is not
  reloaded, and this task does not create the materialization.

External dense BOOL inputs to mixed floating logical operations remain scalar. Integral
comparisons, BFLOAT16, cross-type CAST, and mixed floating topologies outside this grammar retain
their existing scalar fallback or fail-closed behavior.

### Carrier, access, strategy, and range support

| Dimension | Vector support | Fallback or boundary |
|---|---|---|
| Floating type | FLOAT32 preferred species; FLOAT64 preferred species | No BFLOAT16 or integral predicate SIMD. |
| Carrier | Exact primitive array or `MemorySegment` per boundary; every ordered mixed pattern. | No generic carrier argument or hot carrier switch. |
| Layout | `DENSE_LINEAR`, including zero and positive cold base offsets. | New mask boundaries in `LAST_AXIS_BIAS`, `BLOCK_OUTER`, or `GENERAL_ODOMETER` use scalar/parallel-scalar. No gather/scatter. |
| Strategy | `VECTOR` and `PARALLEL_VECTOR`, sharing one vector artifact. | Existing deterministic `SCALAR` and `PARALLEL_SCALAR` fallback. |
| Range | Any valid `[start,end)`; unmasked full numeric vectors plus existing scalar remainder. | Fewer than one numeric vector lane executes scalar. |
| Shape | Any fully static rank with the admitted dense boundary. | Zero extent is zero work; rank zero is one scalar element. |

Parallel-vector preserves current configured/available parallelism, minimum-elements policy,
quotient/remainder chunking, disjoint worker ranges, joining, failure behavior, and run isolation.

### Canonical byte publication and reload

Materialized BOOL remains one byte per element: false is exact byte `0`, true is exact byte `1`,
and no other external byte is valid.

The Java 26 Vector API has no valid direct floating-mask-to-byte-vector conversion.
`VectorMask.cast` requires equal lane counts, while a byte vector of the numeric vector's shape
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
   existing specialization because helper species is deterministically derived from it.
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
access retains element-width scaling and native byte order. Full chunks allocate no array/object,
call no `toArray`, execute no scalar per-lane conversion, and access no byte beyond the numeric
lane count. The existing scalar body owns the remainder. Complete canonical-input validation
remains a cold preflight before worker submission or output mutation.

### Fan-out, multi-output, alias, and zero work

- Existing lowering alone decides graph publication and unit boundaries. This task adds no
  automatic materialization or profitability policy.
- A mask materialized between units is reloaded by the later unit. No fusion crosses that boundary.
- Existing ordered multi-store publishes a graph-required same-unit mask once and the floating
  result in store order. A virtual mask receives no slot.
- Read-only inputs may alias. Every output span remains disjoint from every input and other output
  span; no in-place BOOL form is added.
- Zero elements perform no carrier access, conversion, generated call, or worker submission.
  Positive offset-dense forms retain exact cold offsets and surrounding sentinels.

### Schema, identity, capability, and preparation

- Advance the generated compatibility envelope from 60 to 61.
- Select class-identity schema 61 only for a VECTOR pointwise specialization with one FLOAT32 or
  FLOAT64 lane type and at least one dense non-scalar BOOL `INPUT` or materialized BOOL `OUTPUT`.
- Preserve schema 52 and byte-identical classes for unchanged ordinary pointwise forms, including
  virtual-mask-only `WHERE`; preserve schema 59 for BFLOAT16 and schema 60 for cross-type CAST.
  Scalar predicate forms retain their existing schema.
- Existing IR already records opcode order, types, roles, access plans, stores, and structural key.
  Existing specialization records ordered carriers, strategy, numeric species bit size, numerical
  mode, and power facts. Add no second mask IR or conversion-policy type.
- Code-shaping/cache facts are schema 61, complete canonical IR, exact ordered carrier pattern,
  VECTOR compute, and numeric preferred-species bit size.
- Extents, rank, base offsets, carrier instances, addresses, slots, graph/value/run identities,
  caller bounds, parallel ranges, chunking, workers, and reusable byte-width mask remain cold or
  invocation-local and outside class identity.
- `CpuCapabilityProvider` remains semantic and gains no strategy API. `CpuPartitionPreparer`
  owns exact eligibility and scalar fallback. Shared Prepare and Runtime remain unchanged.
- Add no workspace, materialization candidate, requirement kind, route, tuning input, persistence
  policy, or runtime decision.

### Generated-code discipline

For each specialized shape, a well-written optimal clean Java 26 Vector API implementation is the
design/review oracle. Generated bytecode preserves its semantic algorithm, full-vector/scalar-tail
loop, mask-bit/byte-vector dataflow, direct carrier accesses, store order, and avoidable-overhead
profile.

The generated hot path contains no Synaptik mask helper call, dispatch, allocation, boxing,
reflection, map/string/opcode lookup, per-element virtual call, generic carrier branch, temporary
array, or `VectorMask.toArray`. Existing unrelated chunk-level `CpuVectorMath` calls remain
unchanged and are absent from mask-only evidence forms.

## Out of scope

- General CAST SIMD or changes to CPU 0008K.
- BFLOAT16/FLOAT16 SIMD, mixed promotion, integral predicate SIMD, or non-canonical BOOL storage.
- Non-contiguous gather/scatter; vectorized `LAST_AXIS_BIAS`, `BLOCK_OUTER`, or general stride;
  negative strides, dynamic layouts, or non-injective outputs.
- Native/vendor/OpenBLAS routes, relaxed numerics, automatic materialization policy, autotuning,
  tuning-cache mutation, benchmark-selected behavior, or persistence policy.
- General partition-DAG, fusion, fan-out, publication, alias, Runtime, or shared Prepare redesign.
- FLOOR, CEIL, SIGMOID, SILU, GELU-TANH approximation, exact GELU changes, or general POW SIMD.
- Vector MSE `NONE`; CPU 0008M is the next `Ready` frontier without a detailed specification.
- Changes to Model, Compiler, Planning, Config, Backend Contract, Trace, Engine, Training, other
  backends, architecture contracts, ADRs, module dependencies, or public APIs.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially backend/Prepare/Runtime
  ownership and generated-code discipline.
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Planning Guide](../../../planning-guide.md).
- [CPU backend master plan](../master-plan.md).
- [CPU 0005C vector and parallel strategies](0005c-vector-and-parallel-portable-strategies.md).
- [CPU 0005J bounded pointwise vector coverage](0005j-bounded-pointwise-coverage-and-parity-hardening.md).
- [CPU 0007A0 generated hot-path parity](0007a0-generated-hot-path-parity-correction.md).
- [CPU 0007A1A scalar-body self-containment](0007a1a-generated-scalar-body-self-containment.md).
- [CPU 0007A1L pointwise general-loop parity](0007a1l-pointwise-general-loop-residual-parity.md).
- [CPU 0007A1O pointwise ledger reconciliation](0007a1o-pointwise-ledger-evidence-reconciliation.md).
- [CPU 0008B partition-DAG decomposition](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md).
- [CPU 0008E1 shared partition-DAG adoption](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md).
- [CPU 0008J BFLOAT16 scalar pointwise closure](0008j-bfloat16-scalar-pointwise-closure.md).
- [CPU 0008K cross-type CAST execution](0008k-cross-type-cast-execution.md).
- [CPU backend guide](../../../../backend-guide/cpu-backend.md).
- [Glossary](../../../../glossary.md).

## Architecture constraints

- Planning selects CPU ownership only. CPU analysis owns eligibility, species, specialization,
  direct carrier signature, and scalar fallback.
- Analysis selects the route and declares unchanged resources before assignment; finalization only
  realizes it. Runtime executes the immutable prepared recipe.
- Generated code receives direct typed arguments and sees no Model/Compiler graph object, storage
  discovery, route choice, graph inspection, or backend lookup.
- Prepared recipes stay immutable/reusable; each run has isolated mutable state. Existing
  caller-owned workers remain the only parallel resource.
- Exact/default predicate, logical, and `WHERE` semantics do not change.
- Any public/shared contract, dependency, lifecycle, architecture, or materialization-policy need
  is a stop condition.

## Package impact

Existing packages used:

- `.internal.prepare` — eligibility and schema choice.
- `.internal.codegen.emit` — direct conversion, vector locals, generation, and evidence.
- `.internal.cache` — schema-61 identity and compatibility.
- `.internal.executable` — unchanged canonicality, spans, overlap, ranges, workers, and failures.
- `.internal.ir` and `.internal.lowering` — unchanged roles, access, stores, DAG, and publication.
- `.internal.reference` — unchanged independent scalar oracle.

Packages added or changed:

- No package or responsibility is added or moved.

Type placement:

- No production type is expected; extend existing carrier/generator/preparer owners.
- Add at most two package-private test types under `.internal.codegen.emit`: one
  semantic/structural evidence owner and one performance owner.

## Affected files

Expected production/Javadoc paths (5):

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
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/operation-family-form-ledger-v3.tsv` (new immutable successor)
- `backends/cpu/build.gradle.kts` — guarded evidence properties only.

Expected documentation/planning paths (5):

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0008l-pointwise-simd-mask-output-closure.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Reviewed and expected unchanged: the public, cache, emitter, and preparation package summaries;
`CpuCapabilityProvider`, `CpuKernelIr`, `CpuPointwiseOpcode`, `CpuVectorInstructionEmitter`,
`CpuLoopEmitter`, `CpuScalarEmitter`, pointwise lowering production, `CpuPreparedExecutable`,
scalar reference semantics, shared Prepare, Runtime, Model/Compiler/Training APIs, architecture
tests, backend-conformance, integration tests, and root Java 26 Gradle configuration.

## Maximum scope

At most 27 paths: five production/Javadoc, 17 test/evidence/build, and five
documentation/planning. At most two test types and no production type or package may be added.

This exceeds the normal range because one generated boundary must update eligibility, direct
emission, compatibility, semantic/structural/performance evidence, immutable ledger provenance,
Javadocs, and backend documentation together, while five prior-family evidence/finalization
owners must advance their hard-coded current-envelope assertion from 60 to 61. Those five test
edits authorize assertion maintenance only, not changed prior-family generated bytes or behavior.
Stop and replan if another production owner, test type, shared/public path, access regime, route,
or architecture edit is required. A listed review path need not change merely because it is
allowed.

## Acceptance criteria

1. Inventory remains 48 opcodes and exactly 13 mask opcodes/26 F32-or-F64 typed variants;
   semantic opcodes are distinguished from publication, reload, and fan-out roles.
2. Preparation admits exactly the dense role grammar for VECTOR/PARALLEL_VECTOR, preserves virtual
   masks, and retains existing fallback/fail-closed behavior elsewhere.
3. Publication/reload use the bounded fixed-byte-species bridge, preserve lane order, access
   exactly numeric lane count, allocate nothing, and retain scalar tails.
4. Both types, arrays/segments/every mixed pattern, zero/positive offsets, both vector strategies,
   arbitrary ranges, zero/rank-zero/short ranges, aliases/overlaps, canonical rejection,
   multi-output/fan-out, later-unit reload, and concurrency are covered.
5. Graph-required same-unit fan-out stores one live mask once and reuses it for `WHERE`; no
   unnecessary materialization or reload is introduced.
6. Schema advances 60 to 61 only for changed vector bytes. Unchanged schema-52, schema-59, and
   schema-60 classes regenerate byte-identically and incompatible envelopes miss.
7. The exact 72 retained Class-File dossiers below pass complete decompilation, determinism,
   descriptor/member, tail/store-order, and forbidden-overhead checks. Lightweight normalization
   covers omitted ordered carrier combinations without retaining a Cartesian corpus.
8. Every sample-pair ratio, fork median, and aggregate in the exact 12-row performance matrix is
   `<= 1.15x` its same-algorithm optimal clean Java oracle, without retry/discard.
9. Invalid BOOL bytes, carriers, spans, inaccessible segments, or overlap fail before worker
   submission and output mutation.
10. Immutable v2 ledger remains unchanged; tested v3 records 0008L evidence and historical
    provenance.
11. No public/shared API, workspace, materialization policy, route, tuning, dependency,
    architecture, conformance, integration, or unrelated SIMD behavior changes.
12. A separate clean documentation pass finalizes Javadocs, package summaries, CPU guide, glossary
    impact, task evidence, master plan, and roadmap before Complete. CPU 0008M then becomes the sole
    `Ready` CPU frontier with no detailed specification.

## Tests / validation

### Semantic execution matrix

Use 28 explicit all-array, zero-offset, VECTOR, exact-multiple baseline forms:

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

Focused tests separately cover all three non-dense fallbacks, zero extent, rank zero, short and
arbitrary non-zero ranges, canonical rejection for array/segment, read-only input alias,
input/output and output/output overlap rejection, failure-before-publication, later-unit reload,
sentinels, concurrent calls, and unchanged BOOL-only/virtual-mask controls for both types.

### Class-File matrix

Retain 72 complete Class-File dossiers. The 28 baselines establish each changed
operator/topology/type. Carrier emission has five unique ordered boundary shapes:

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

An automated normalization check may enumerate all ordered carrier patterns but retains only these
72 dossiers. It proves each omitted pattern consists solely of already-proved boundary-local
descriptor/access substitutions and per-type segment-layout initialization. Any additional
normalized shape expands the retained matrix or stops the task. Offset, extent, tail, and parallel
orchestration are cold and do not create classes.

### Performance matrix

Measure six materially distinct dataflows for both types: classification publication, comparison
publication, unary logical publication, binary logical publication, external BOOL `WHERE`
reload, and publication-plus-`WHERE` fan-out.

| Dataflow | FLOAT32 row | FLOAT64 row |
|---|---|---|
| Classification publication | all array, zero offset, VECTOR, exact multiple, `IS_NAN` | all segment, positive offset, PARALLEL_VECTOR, tail, `IS_FINITE` |
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
VECTOR/PARALLEL_VECTOR, and exact/tail ranges for each type. Every sample-pair ratio, fork median,
and aggregate is `<= 1.15x`. Use five fresh isolated fixed-heap forks, at
least five warmup batches, adaptive minimum-25-ms batches, nine randomized AB/BA rounds, exact
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

Both evidence owners use the same fresh root. Record its exact path/checksums and fail on any
missing or duplicate row, class, decompilation, fork, sample, manifest entry, or checksum.

Documentation:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

Also validate local links/anchors, fences, final newlines, arithmetic, schema `60 -> 61`, exact
27-path ceiling/type placement, synchronized status/order, 0008L Complete, 0008M as the sole Ready
CPU frontier without a task file, empty staging, unrelated-work preservation, and no-change
conclusions. Repository-wide Java validation is deferred to CPU 0009/CI. Shared architecture,
conformance, and integration suites are not run unless a shared boundary changes, which requires
stopping and replanning.

## Dependencies

- CPU 0008K — Complete; schema-60 CAST and current pointwise machinery.
- CPU 0005C/0005J — Complete; preferred-species loops, carriers, worker ranges, virtual masks,
  BOOL-only vectors, and tails.
- CPU 0007A0/0007A1A/0007A1L/0007A1O — Complete; generated/direct parity, self-containment,
  fallback, and immutable ledger discipline.
- CPU 0008B/0008E1 — Complete; bounded multi-store DAG and shared DAG adoption.
- CPU 0008J — Complete; canonical BOOL pointwise publication/preflight and BFLOAT16 exclusions.
- Java 26.0.1 Vector/Class-File APIs through the existing build.

## Follow-up tasks

- CPU 0008M becomes the sole `Ready` frontier and owns only vector MSE `NONE`; do not create its
  specification in this completion pass.
- CPU 0008N–0008P retain Draft order/scope.
- CPU 0009 owns generated-coverage reconciliation, including CPU 0008I's gap and 0008L evidence.
- General non-contiguous mask vectorization and excluded unary SIMD gain no task here.

## Architecture impact

Expected impact: None.

Required no-change conclusions:

- Architecture contract/pages/ADRs/tests stay unchanged: ownership, lifecycle, dependencies, and
  generated-code authority do not change.
- Model operations, `DataType`, `LayoutDescriptor`, Tensor/Compiler/Training APIs, and
  capability-query contracts stay unchanged because semantics/support already exist.
- Shared Prepare/Runtime stay unchanged; current resources, slots, finalization, ranges, workers,
  publication, and run isolation suffice.
- `CpuKernelIr`, `CpuPointwiseOpcode`, `CpuAccessPlan`, pointwise lowering production,
  `CpuLoopEmitter`, `CpuScalarEmitter`, `CpuPreparedExecutable`, and scalar reference
  semantics stay behaviorally unchanged.
- Backend-conformance/integration tests stay unchanged because no shared/public behavior owner
  changes; CPU-private evidence covers this route.
- Dependencies, plugins, source sets, root Java toolchain, and ordinary tests stay unchanged; CPU
  Gradle forwards guarded evidence properties only.
- Other backends, native routes, materialization/tuning policy, and 0008M onward stay behaviorally
  unchanged; only the required 0008M planning status advances to `Ready`.
- Prior-family Conv2d, Conv3d, partition-DAG, batch-normalization-training, and finalization
  behavior and generated bytes stay unchanged; their five test owners update only the hard-coded
  current compatibility-envelope assertion required by the repository-wide 60-to-61 advance.

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
  boundaries require its store. Reuse the live mask; never materialize a private mask for 0008L.
- Packed low lane bits plus a bounded fixed-species `ByteVector` is the only planned full-chunk
  bridge. Equal-lane cast, reinterpretation, arrays, and scalar full-chunk loops are rejected.
- The 28 forms are the explicit semantic evidence basis, not the semantic inventory. The inventory
  is 13 opcodes/26 typed variants. Pairwise role probes and normalized structural proof replace
  an all-Cartesian semantic and decompilation corpus.
- Schema 61 follows changed generated bytes, not the mere presence of a predicate in semantic IR.

## Known limitations

- Only dense F32/F64 mask boundaries vectorize. External BOOL inputs to mixed floating logical
  operations and non-dense boundaries stay scalar.
- The bridge requires the numeric preferred species to have at most 64 lanes and derives one
  covering fixed byte species with at most 64 lanes. A target failing that proof falls back and
  never calls `toLong` on a wider mask.
- No scalar speedup is claimed; performance compares generated code with its same-algorithm clean
  Java oracle on the recorded host/JVM.

## Validation evidence

Pre-implementation planning and review evidence:

- Implementation-discovered scope correction: after `CpuGeneratorSchema.CURRENT_VERSION`
  advanced from 60 to 61, a focused run executed 22 tests with 10 failures solely in
  `CpuPartitionFinalizerTest`, `CpuConv2dEvidenceTest`, `CpuConv3dEvidenceTest`,
  `CpuPartitionDagGeneratedEvidenceTest`, and `CpuBatchNormTrainingEvidenceTest`; each owner
  hard-codes the repository's current envelope/schema value. These five existing test paths are
  therefore required assertion-maintenance owners. They replace five unused production
  allowances (the four package summaries and `CpuVectorInstructionEmitter`), which remain
  review-only and expected unchanged. The exact ceiling remains 27 paths: five
  production/Javadoc + 17 test/evidence/build + five documentation/planning, with at most two new
  test types and no new production type.

- Replacement planning/review context read in full: `AGENTS.md`, `ARCHITECTURE.md`, current
  architecture index, documentation rules, General/Planning profiles, Planning Guide, CPU master
  plan, this draft, completed 0005C, 0005J, 0007A0, 0007A1A, 0007A1L, 0007A1O, 0008B, 0008E1,
  0008J, and 0008K, plus directly relevant current CPU production/test/package/ledger owners.
  The roadmap CPU status/order sections were read in full.
- Local Java 26.0.1 source confirms least-significant-bit lane order, `toLong()`'s at-most-64
  precondition, `fromLong`, equal-length `cast`, and masked `ByteVector` array/segment APIs.
  Current CPU source confirms 48/13/26 inventory, virtual/BOOL-only behavior, dense boundary gap,
  canonical preflight, DAG/store fan-out representation, prior schema-60 ownership, and the
  partial schema-61 isolation change.
- Scope-correction validation recounted the three `Affected files` groups as 5 + 17 + 5 = 27,
  found all 630 local Markdown links resolving across this task, the CPU master plan, and the
  roadmap, and confirmed balanced fences and final newlines in all three files. Status/order scans
  found the planned frontier statuses internally consistent at that checkpoint. Final
  whitespace/index/path checks are recorded below and were rerun after every planning edit.
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
  paths and changed no Java, build, test, architecture, conformance, or integration source.
- Documentation review confirmed the 48-opcode/13-mask-opcode/26-typed-variant distinction; dense
  F32/F64 publication and reload; virtual-versus-materialized boundary; fixed covering byte
  species; least-significant-bit lane mapping; exact numeric-lane byte access; scalar tails;
  shared VECTOR/PARALLEL_VECTOR artifact; non-dense, mixed-logical, BFLOAT16, integral, and wider-
  than-64-lane fallback boundaries; graph-required store-once fan-out; later-unit reload; and
  schema-61 isolation from schema 52, 59, and 60.
- Javadocs on all five changed production files remain accurate after review. The changed members
  document schema 61, dense floating-mask eligibility, fixed-byte-species setup, direct array or
  segment publication/reload, numeric-lane bounds, and scalar-tail ownership. Package summaries
  remain unchanged because package responsibility, visibility, and lifecycle did not change.
- Glossary impact is real but narrow: the existing `CPU portable route` entry now distinguishes
  virtual floating masks from canonical dense BOOL boundaries and records schema-61 isolation.
  No new reusable project term was introduced, so no separate glossary heading was added.
- Public Model/Compiler/Training APIs, `CpuCapabilityProvider`, shared Prepare/Runtime contracts,
  materialization and tuning policy, dependencies, architecture pages/ADRs/tests, backend-
  conformance tests, and integration tests remain unchanged because this is a CPU-private
  generated-route eligibility and emission closure using existing roles, slots, ranges, workers,
  publication, and run isolation.
- Documentation context ran `./gradlew :backends:cpu:javadoc`: it passed with 11 actionable tasks
  (one executed, ten up-to-date) and 53 pre-existing missing-`@param` warnings in unchanged
  `CpuPartitionLowering.LoweredPartition` and `CpuPartitionPreparationPlan.ExecutionUnitPlan`
  constructors, plus the two expected incubating-Vector-module warnings. No warning names a
  0008L-changed source member.
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
  `javap` reports, and 72 unique records in the 73-line manifest. The manifest SHA-256 is
  `b7e89b8a9c07c730df5bd382640eec549260cd293e7be264163bebf343b22a57`. Complete inspection found
  no Synaptik helper, allocation, temporary array, boxing, reflection, dispatch,
  `CpuVectorMath`, or `VectorMask.toArray` in the changed mask artifacts. The fan-out artifact has
  exactly one publication, one blend, and one `toLong`, then reuses the live mask for `WHERE`.
- The retained performance command completed in 3 minutes 33 seconds. Its exact
  `12 x 5 x 9 = 540` sample pairs, 60 fork medians, and 12 median-of-fork-medians aggregates all
  passed the `<= 1.15x` gate. Sample-pair ratios ranged from `0.1715377840834143x` through
  `0.8171099525620247x`; fork medians ranged from `0.18067302308736136x` through
  `0.7176738221779834x`; and aggregates ranged from `0.24111596838095475x` through
  `0.7030305670814165x`. Individual timed sides ranged from 48,323,667 through 386,857,375 ns, so
  every retained sample exceeded 25 ms. All pre-run, warmup, sample, and post-run equality checks
  passed, with retry and discard both false.
- The performance manifest contains 80 entries and has SHA-256
  `01c84d6f6925d01a71d30879b51f74206b644d7fc9a077372e1db885aea7a55e`. The aggregate SHA-256 is
  `0ecfbb53d921f8bc79064d7cc9764c2b11a76d1abd352f0197de99b9c63b9d23`, and the retained source
  snapshot SHA-256 is `7b0635e8ac4e84e5e7b43abe0528c875d4e485a069ff3f4fa9f02333522ac082`.
- Ledger v2 remains byte-identical with SHA-256
  `41774384007bdb17b011cb8fbaae3b6928baa6ac8c04027a82540977f046031a`; ledger v3 has SHA-256
  `f1b47d1a10472b8c9c2b92bf3bc768e78c80bf9b5f286e6a4025d2098f0c7da7` and records CPU 0008L,
  schema 61, the new evidence owners, and historical provenance.
- Final scope is 23 of 27 authorized paths: five production/Javadoc, 13 test/evidence/build, and
  five documentation/planning. Exactly two test types and no production type were added. Final
  CPU Javadoc, local links/fragments, fences, final newlines, whitespace, arithmetic, staging,
  and diff checks passed. CPU 0008L is `Complete`; CPU 0008M is the sole `Ready` frontier and has
  no detailed task file.

## Implementation notes

- The implementation extends only the five planned production owners. CPU analysis admits dense
  FLOAT32/FLOAT64 mask inputs and outputs for vector compute, proves the preferred numeric lane
  count is from two through 64, and selects schema 61 only when such a boundary changes generated
  vector bytes.
- Generated publication and reload derive the smallest covering fixed `ByteVector` species,
  preserve least-significant-bit lane order, mask byte access to the numeric lane count, retain
  direct array or native-order segment access, and leave the existing scalar body responsible for
  short ranges and tails.
- Virtual masks remain `VectorMask` locals. A graph-required materialized mask that also feeds
  same-unit `WHERE` is stored once and reused live; only a later unit reloads a dense boundary.
  Existing lowering continues to decide publication and materialization.
- Build changes only forward the guarded structural/performance evidence properties. Ledger v3
  names CPU 0008L and schema 61 while retaining the immutable v2 predecessor checksum and prior
  provenance.
- Final retained evidence is recorded above and closes every acceptance criterion.

## Completion summary

- Completed changes: Added dense FLOAT32/FLOAT64 vector-mask publication and reload through a
  bounded fixed-`ByteVector` bridge for arrays, segments, and ordered mixed carriers while
  preserving virtual masks, graph-required store-once fan-out, later-unit reload, scalar tails,
  and exact fallback boundaries.
- Files changed or created: Five production/Javadoc paths, 13 test/evidence/build paths, and the
  five authorized documentation/planning paths; exactly two test types and no production type
  were added.
- Tests and validation: The exact focused command passed in 10 seconds; the final full CPU rerun
  passed in 26 seconds with 715 tests, zero failures/errors, and 18 expected opt-in skips. The
  retained 72-Class-File structural gate and complete 540-sample-pair performance gate passed at
  `/private/tmp/synaptik-cpu-0008l-1lMuloJp`, and CPU Javadoc and documentation integrity passed.
- Documentation-agent review: Clean context `01a072a9-a208-7931-9feb-90176a70e268` independently
  finalized the CPU guide, glossary impact, task evidence, CPU master plan, and roadmap after
  reviewing the five production Javadocs and executable evidence owners.
- Documentation impact: The CPU guide now explains the fixed-byte-species bridge, virtual and
  materialized boundaries, lane mapping, carrier/range/tail behavior, fallback limits, fan-out,
  selective schema identity, and generated-versus-optimal-clean-Java evidence boundary.
- Javadoc review: All five changed production Javadocs are accurate; CPU Javadoc passed. Package
  summaries remain unchanged because package ownership and responsibility did not change.
- Glossary impact: The existing CPU portable-route entry now distinguishes virtual masks from
  dense canonical BOOL boundaries and records selective schema-61 identity; no new reusable term
  required a separate heading.
- Unresolved issues: None.
- Follow-up required: CPU 0008M is the sole `Ready` CPU frontier without a detailed task file.

Status: Complete
