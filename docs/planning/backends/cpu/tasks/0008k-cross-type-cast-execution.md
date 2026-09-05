# CPU Task 0008K: Cross-Type CAST Execution

## Status

Complete

## Goal

Implement generated scalar and caller-parallel scalar execution for every Model-defined ordered
`CAST` pair among `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT64`, `INT32`, and `BOOL`. Results must
match completed Model 0025L at represented-bit level for all 36 pairs while preserving existing
typed carriers, non-negative static access geometry, bounded pointwise directed acyclic graph
(DAG), preparation, artifact-cache, finalization, and publication machinery.

Directly emitted Class-Files must match an optimal clean Java implementation of each specialized
case in semantic algorithm, hot-loop/dataflow shape, and avoidable-overhead profile. Generated
element work must not call `CastValueConversions` or another conversion bridge.

## Scope

### Exact operation matrix

- Admit all 36 ordered source/target pairs from Model 0025L. The source descriptor,
  `CastAttrs.targetDataType()`, and output descriptor must agree; input and output Shapes must be
  equal and fully static; layouts must be resolved and representable by the current non-negative
  CPU access contract; and the output must be injective.
- Preserve all six same-type pairs as exact represented-bit identities, including floating signed
  zero and every NaN sign, quiet/signaling state, and payload.
- Implement the 30 cross-type pairs through these structural evidence groups:

  | Group | Pairs | Required operation |
  |---|---:|---|
  | Same-type identity | 6 | Raw bits unchanged |
  | `BOOL` to numeric | 5 | False/true to positive numeric zero/one |
  | Numeric to `BOOL` | 5 | False only for integral zero or either floating signed zero |
  | Integral width change | 2 | Sign extension or low-32-bit narrowing |
  | Lossless floating widening | 3 | Exact finite widening and Model-defined NaN mapping |
  | Lossy floating narrowing | 3 | Direct round-to-nearest, ties-to-even (RNE) |
  | Floating to integral | 6 | Truncate toward zero, then saturate; NaN to zero |
  | Integral to floating | 6 | Direct target-format RNE |

  These groups partition all 36 pairs for review and evidence; they are not runtime tags.
- Apply every Model 0025L zero, subnormal, underflow, overflow, infinity, NaN, integral-boundary,
  and Boolean rule. In particular, `FLOAT64`/`INT64`/`INT32 -> BFLOAT16` converts directly rather
  than through FLOAT32; lossy floating NaNs become the positive canonical target quiet NaN;
  widening NaNs preserve sign, quiet/signaling state, and payload through the specified
  left-aligned fraction mappings; infinities saturate for integral targets; `INT64 -> INT32`
  retains low bits rather than saturating; and BOOL stores remain canonical zero/one.
- Emit direct primitive conversion, bit, branch, and arithmetic instructions. Inline exact direct
  BFLOAT16 RNE and NaN behavior. Generated element code must not invoke `CastValueConversions`,
  `BFloat16Bits`, a CPU reference kernel, or another Synaptik helper.

### Carriers, access geometry, ranges, and DAGs

- Support each source/target native primitive array (`double[]`, `float[]`, `short[]`, `long[]`,
  `int[]`, or `byte[]` for BOOL), native-order `MemorySegment`, and all four ordered boundary
  patterns: array/array, array/segment, segment/array, and segment/segment. Capability sees BOOL
  descriptor types, not carrier contents. Existing executable/run preflight validates every BOOL
  input byte as canonical zero/one before worker submission, output mutation, or publication. Add
  no carrier conversion buffer or `Object` bridge.
- Preserve exact current access terminology and machinery:
  - canonical contiguous and `LayoutKind.OFFSET_DENSE` values lower to `DENSE_LINEAR`; an
    offset-dense layout changes cold `baseElementOffset`, not structural class identity;
  - positive non-unit layouts use `BLOCK_OUTER` when they retain a contiguous suffix and
    `GENERAL_ODOMETER` otherwise;
  - rank-zero input uses read-only `SCALAR_ALL_ZERO`, while its injective output uses
    `DENSE_LINEAR`;
  - zero extents, partial/empty half-open ranges, checked referenced spans, arbitrary non-negative
    offsets, and overlap rejection retain current behavior.
- Negative slice steps do not imply negative storage strides. Their owning operation semantics do
  not create a negative-stride `LayoutDescriptor`. Negative storage strides remain unrepresentable
  by current Model `LayoutDescriptor` and `CpuAccessPlan` and are outside this task.
- Scalar and caller-parallel scalar plans reuse one generated scalar artifact. Non-empty parallel
  ranges own disjoint output coordinates and join before the next unit or publication. Rank zero
  is necessarily scalar because one element cannot form two non-empty ranges; matrix accounting
  marks its parallel cell structurally inapplicable rather than claiming an impossible strategy.
- Keep cross-type CAST scalar-only. Any bounded pointwise DAG containing one becomes vector-
  ineligible and selects scalar or parallel-scalar. Do not split merely to regain SIMD or use a
  scalar tail from a vector conversion body.
- Permit cross-type CAST within the existing one-to-eight-occurrence pointwise DAG when every node
  is exactly typed. Each CAST writes its exact target primitive/raw local before any consumer, so
  explicit cast chains preserve every intermediate conversion boundary. Existing fan-out,
  publication, alias, materialized split, multi-store, and fail-closed rules remain unchanged.
- CAST adds zero workspace and no new materialization, representation policy, route, or resource.

### Capability, IR, preparation, cache, and finalization

- Extend `CpuCapabilityProvider` from same-type/non-BFLOAT16 CAST to exactly all 36 pairs under the
  descriptor checks above. Capability validates BOOL data types and descriptor relationships only;
  it cannot inspect runtime array or segment bytes. Malformed descriptors, unresolved or negative
  layouts, and non-injective outputs fail before lowering or declaration. A supported occurrence
  whose bound BOOL input contains a byte other than zero or one is rejected later by the existing
  executable/run preflight before worker submission, output write, or publication.
- Retain one `CpuPointwiseOpcode.CAST`; change its validation from same-type identity to one source
  and the exact target output type. Ordered input/output data types already participate in
  `CpuKernelIr.structuralKey()` and specialization `boundaryDataTypes`; add no runtime type map,
  string dispatch, conversion-mode object, or scalar immediate.
- Keep lowering in `CpuPartitionLowering`: capability validates `CastAttrs`, while the output
  descriptor supplies the typed target. Keep strategy selection in `CpuPartitionPreparer` and
  force scalar compute for cross-type CAST; caller parallelism remains cold orchestration.
- Advance `CpuGeneratorSchema.CURRENT_VERSION` once from 59 to 60. Class-identity schema 60 applies
  if and only if pointwise IR contains cross-type CAST. Same-type non-BFLOAT16 CAST and unchanged
  older pointwise/family projections retain schema 52; unchanged BFLOAT16 pointwise, including
  same-type BFLOAT16 CAST, retains schema 59. Envelope version 60 makes schema-59 and older entries
  safe incompatible misses; add no migration, alias, or dual reader.
- Keep class identity Shape-polymorphic. Source/target types, carriers, structural access plans,
  instruction topology, stores, and scalar compute are code-shaping. Extents, offsets, concrete
  strides, addresses, slots, ranges, parallelism, workers, runs, and evidence paths stay cold.
- Preserve finalization ownership: after shared assignment, `CpuPartitionFinalizer` realizes the
  selected artifact through `CpuGeneratedKernelArtifactStore` and returns an immutable prepared
  executable. Runtime performs no generation, cache lookup, conversion selection, or dispatch.

### Auditable semantic and structural strategy

- Execute a generated Cartesian semantic matrix for every 36 ordered pairs and four carrier
  patterns against contiguous `DENSE_LINEAR`, offset-dense `DENSE_LINEAR`, positive-strided
  `BLOCK_OUTER`, positive-strided `GENERAL_ODOMETER`, and rank-zero
  `SCALAR_ALL_ZERO -> DENSE_LINEAR`. Run scalar and caller-parallel scalar for the four
  multi-element fixtures and scalar for rank zero: exactly `36 * 4 * (4 * 2 + 1) = 1,296` legal
  execution cells.
- Give every cell a compact pair-specific edge vector containing applicable signed zeros, finite
  values, extrema, subnormals, infinities, NaNs, saturation/modulo edges, and halfway ties.
  Expected results may use Model `CastValueConversions` in test/cold oracle code, but independently
  fixed raw constants must cover direct-BFLOAT16 double-rounding, NaN mapping, saturation, and
  low-bit cases. Expected values must not come from generated code or forbidden chained Java
  conversions.
- Enumerate 576 structural class forms: 36 pairs times four carrier patterns times four distinct
  access-plan pairs (`DENSE_LINEAR`, `BLOCK_OUTER`, `GENERAL_ODOMETER`, and rank-zero). Prove that
  offset-dense and contiguous share key/bytes and that parallel plans reuse scalar key/bytes.
  Generate each form twice and prove deterministic identity and bytes.
- Retain all 576 Class-Files, checksums, complete `javap -c -v -p`, descriptor/member-reference/
  forbidden-opcode reports, normalized instruction summaries, optimal clean Java oracle source,
  classes and decompilation, inventory, commands, environment facts, and verified manifest under
  one fresh `/private/tmp/synaptik-cpu-0008k-*` root.
- Prove generated classes final and field-free with one typed static entry and direct operations
  matching the clean Java group. Reject Model/CPU conversion helpers, allocation, boxing,
  reflection, method handles, `invokedynamic`, monitors, collections, strings, map/string/runtime
  opcode or type dispatch, generic `Object` carriers, and graph/layout/cache/route/resource/worker
  lookup.
- Prove representative schema-52 and schema-59 fixtures remain byte-identical. Schema 60 must
  distinguish code-shaping source/target, carrier, access, and DAG facts while compatible cold
  extents, offsets, ranges, and parallelism reuse it.

### Representative performance evidence

Run a retained five-fork generated-versus-direct matrix over these code-shaping rows:

| Row | Distinct concern |
|---|---|
| Dense array `FLOAT64 -> BFLOAT16` | Direct binary64-to-BFLOAT16 RNE and dense integer loop |
| General mixed `INT64 -> BFLOAT16` | Direct integral-to-BFLOAT16 RNE and long odometer |
| Offset-dense mixed `FLOAT64 -> INT64` | Truncation/NaN/infinity saturation with cold offset |
| Block-outer `BFLOAT16 -> FLOAT64` | Raw widening/NaN mapping with positive affine input |
| Dense segment `INT64 -> FLOAT32` | Inexact integral-to-binary32 RNE and segment access |
| General array `FLOAT32 -> BOOL` | Signed-zero/NaN truth and canonical byte output |
| Dense mixed `BOOL -> FLOAT64` | Canonical byte load and positive zero/one output |

Add one same-type raw identity control and one two-CAST rounding-sensitive fused-chain control.
Do not benchmark all 576 classes: these rows cover each algorithm group, primitive width,
BFLOAT16 path, saturation/truth branch, carrier/access family, and virtual conversion boundary;
the exhaustive semantic/Class-File matrices cover non-code-shaping permutations.

The direct side is ordinary optimal clean Java with matching typed carriers, cold geometry,
range, algorithm, stores, and loop/address form. Select/bind before timing. Use five fresh
fixed-heap C2-only forks, explicit untimed compilation stabilization, at least five equal-count
warmups, and nine seeded-randomized symmetric AB/BA measurements per row. Each retained sample
uses equal generated and direct work in adjacent generated/direct/direct/generated or reverse
blocks and estimates generated/direct as the geometric mean of its two directional ratios. This
cancels multiplicative linear temporal drift in log time because each side has the same mean
temporal position. Use adaptive batches calibrated to at least 250 ms per side, deterministic
inputs, exact pre/post verification, observed sink, and no retry/discard. Retain every timed
invocation and every sample. Every row in every fork and each median of fork medians must be
generated/direct `<= 1.15x`.

### Documentation

- A distinct clean documentation-focused context finalizes affected Javadocs/package summaries,
  CPU guide, glossary impact, evidence, task, master plan, and roadmap after executable behavior
  stabilizes.
- Documentation distinguishes Model semantics from CPU execution, explains schema 60 and
  scalar-only scope, and states that negative slice steps do not create negative storage strides.
  CPU 0008L and later work remain unimplemented.

## Out of scope

- Negative storage strides or any Model/layout architecture change. Negative slice steps do not
  authorize negative storage strides.
- SIMD/vector or parallel-vector CAST, BFLOAT16 SIMD, vector masks, or CPU 0008L–0008P work.
- New operations, cast modes, public APIs, implicit promotion, folding/elimination, Compiler or
  gradient changes, or Model semantic changes.
- Helper/fallback calls from emitted element code; allocation, boxing, reflection, maps, strings,
  runtime dispatch, or carrier conversion buffers in the hot path.
- Dynamic/unresolved geometry, non-injective output, overlap/in-place conversion, accepting
  non-canonical BOOL storage, fixed-Shape specialization, new materialization/workspace/route/native/autotuning,
  tuning-cache mutation, or relaxed numerics.
- Backend-conformance, integration, architecture-test, dependency/plugin/source-set/toolchain,
  shared Prepare/Runtime, Engine, Config, Trace, Training, NN, other-backend, CPU 0009
  implementation, or detailed CPU 0009 planning. The CPU test task necessarily forwards three
  opt-in CAST evidence properties; this is test plumbing, not an ordinary build-behavior change.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core invariants, concrete
  backend ownership, performance evidence, and CPU backend routes
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [Model 0025L cross-type CAST semantics](../../../modules/model/tasks/0025l-cross-type-cast-conversion-semantics.md)
- [CPU 0005B access plans](0005b-universal-access-plans-and-right-aligned-broadcasting.md)
- [CPU 0005C scalar/vector and parallel strategies](0005c-vector-and-parallel-portable-strategies.md)
- [CPU 0007A0 generated hot-path parity](0007a0-generated-hot-path-parity-correction.md)
- [CPU 0007A1A scalar-body self-containment](0007a1a-generated-scalar-body-self-containment.md)
- [CPU 0007A1L pointwise general-loop parity](0007a1l-pointwise-general-loop-residual-parity.md)
- [CPU 0007A1O pointwise evidence ledger](0007a1o-pointwise-ledger-evidence-reconciliation.md)
- [CPU 0008B bounded pointwise DAG](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)
- [CPU 0008E1 shared DAG adoption](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md)
- [CPU 0008J BFLOAT16 scalar closure](0008j-bfloat16-scalar-pointwise-closure.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md), with General,
  Planning, Backend-guide, API/Javadoc, and Example profiles as applicable

## Architecture constraints

- Model 0025L is the sole semantic authority. CPU may advertise only exact implemented cases.
- Planning selects CPU ownership; CPU analysis owns lowering, scalar strategy, specialization and
  declarations; finalization owns compatible generation/reuse; Runtime invokes the immutable result.
- Hot code sees typed carriers and primitive geometry, not graph/operation objects, route/cache
  policy, workers, scalar wrappers, or conversion helpers.
- Optimal specialized clean Java is the design/review oracle. Any generated algorithm, conversion-
  boundary, loop/dataflow, or avoidable-overhead deviation requires a technical reason and evidence.
- Preserve atomic partition execution, pre-write validation, exact declarations, deterministic
  unit order, caller-parallel join, publication, immutable prepared state, and run isolation.
- Stop if implementation needs Model/shared/public/dependency/architecture or ordinary build-
  behavior changes; forwarding task-specific opt-in evidence properties to the Test JVM is allowed.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.backend.cpu` — truthful capability.
- `...internal.ir` — typed CAST validation and identity.
- `...internal.lowering` — existing pointwise/access lowering.
- `...internal.codegen.emit` — direct conversion emission and generated class assembly.
- `...internal.prepare` — scalar selection and schema projection.
- `...internal.cache` — schema 60 compatibility/identity.
- `...internal.reference` — cold/test differential reference only.

No package is added, moved, removed, or exported.

Type placement:

- Add package-private `CpuCastEmitter` under `...internal.codegen.emit` as the cold
  Class-File-construction owner for direct conversion instruction sequences. It is not a runtime
  helper call. `CpuScalarEmitter` remains the pointwise instruction coordinator.
- Existing IR, lowering, preparation, specialization, and schema owners stay in place. Add no
  generic conversion service, registry, facade, map, or public type.

## Affected files

Expected production/Javadoc paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcode.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCastEmitter.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`

Expected test/evidence paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcodeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCastGeneratedKernelTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCastEvidenceTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCastPerformanceTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseLedgerEvidenceTest.java`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/operation-family-form-ledger-v2.tsv`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected documentation/planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0008k-cross-type-cast-execution.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Necessary opt-in evidence plumbing:

- `backends/cpu/build.gradle.kts` — forwards the structural-evidence root, performance enablement,
  and performance-evidence root properties to Test JVMs; no plugin, dependency, source set,
  toolchain, or ordinary test behavior changes.

Reviewed but unchanged: `CpuAccessPlan`, `CpuCarrierEmitter`, `CpuLoopEmitter`,
`CpuPartitionFinalizer`, `CpuPreparedExecutable`, Model CAST source/tests, backend-conformance,
integration/architecture tests, and Tensor/Compile/Training API references.

## Maximum scope

The expected implementation has 38 paths: 18 production/Javadoc, 15 test/evidence, and the five
named documentation/planning paths. At most 40 paths may change, allowing two additional focused
test/evidence owners if implementation inspection exposes a distinct uncovered gate. At most one
package-private production type and three focused
test/evidence types may be added; no package or public type may be added. Retained evidence under
one fresh `/private/tmp/synaptik-cpu-0008k-*` root does not count.

Stop and replan if correctness needs more than 40 paths, another production abstraction, any
Model/shared edit, negative strides, new resources, or another route/policy.

## Acceptance criteria

1. Capability through preparation admits exactly all 36 descriptor-valid pairs and fails closed
   before declaration/artifact lookup for malformed or unsupported descriptor occurrences.
   Capability tests assert BOOL type/descriptor rules without pretending to inspect runtime bytes.
2. All 1,296 legal pair/carrier/layout/strategy execution cells pass represented-bit checks;
   rank-zero parallel is explicitly inapplicable.
3. Same-type identity and every Model 0025L edge rule pass, including direct-BFLOAT16
   double-rounding counterexamples and negative counterparts.
4. Typed arrays, segments, mixed carriers, all scoped access forms/ranges, caller parallelism,
   overlap rejection, zero workspace, failure-before-publication, and run isolation pass. For every
   BOOL-source carrier family, focused execution tests inject non-zero/non-one bytes and prove the
   existing executable/run preflight rejects them before worker submission, any output write, or
   publication; output sentinels remain unchanged.
5. Cross-type CAST is scalar-only; existing same-type vector eligibility is unchanged; bounded
   scalar DAGs preserve every explicit conversion boundary.
6. Schema advances exactly 59 to 60 only for cross-type CAST. Unchanged schema-52/schema-59 bytes
   remain identical; older envelopes miss safely; cold facts do not shape class identity.
7. All 576 structural classes are deterministic, final, field-free, correctly typed, completely
   decompiled, and free of all forbidden helpers/overheads.
8. Generated groups match optimal clean Java algorithm/dataflow/overhead; any difference is
   justified with structural and performance evidence.
9. Every row in every retained fork and each fork-median aggregate passes `<= 1.15x`, with no
   retry, discard, or production selection effect.
10. The pointwise ledger updates CAST evidence without rewriting unrelated historical rows.
11. The separate documentation pass finalizes Javadocs, guide, glossary, evidence and planning.
    CPU 0008L remains next Draft and 0008L–0008P order stays unchanged.

## Tests / validation

Implementation pass:

```bash
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIrTest --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcodeTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPointwisePartitionLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuCastGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseLedgerEvidenceTest
./gradlew :backends:cpu:test --rerun-tasks
```

Retained evidence:

```bash
CPU_0008K_EVIDENCE_ROOT="$(mktemp -d '/private/tmp/synaptik-cpu-0008k-XXXXXXXX')"
printf '%s\n' "$CPU_0008K_EVIDENCE_ROOT" > "$CPU_0008K_EVIDENCE_ROOT/evidence-root.txt"
touch "$CPU_0008K_EVIDENCE_ROOT/RUN-STRUCTURAL-EVIDENCE"
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuCastEvidenceTest -Dsynaptik.cpu.cast.structuralEvidenceRoot="$CPU_0008K_EVIDENCE_ROOT" --rerun-tasks
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuCastPerformanceTest -Dsynaptik.cpu.cast.performance=true -Dsynaptik.cpu.cast.performanceEvidenceRoot="$CPU_0008K_EVIDENCE_ROOT" --rerun-tasks
```

The task records the exact value from `evidence-root.txt`. Both evidence owners use that same fresh
root. The structural owner consumes its guard after success, and validation fails on any missing
class, decompilation, fork, row, manifest entry, or checksum.

Documentation pass:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

Also check Markdown links/anchors/fences/final newlines, exact path ceiling and package placement,
schema `59 -> 60`, matrix arithmetic, evidence manifest, status/order, empty staging, unrelated
work preservation, and all no-change conclusions.

Repository-wide Java validation is deferred to CPU 0009/CI. Architecture, backend-conformance,
and integration tests are not run because no shared boundary changes.

## Dependencies

- CPU 0008J — Complete; schema-59 BFLOAT16 scalar pointwise and current machinery.
- Model 0025L — Complete; exact 36-pair semantics and scalar test oracle.
- CPU 0005B/0005C — Complete; access, carriers, strategies, and parallel ranges.
- CPU 0007A0/0007A1A/0007A1L/0007A1O — Complete; generated parity, self-containment,
  general-loop parity, and evidence ledger.
- CPU 0008B/0008E1 — Complete; bounded and shared pointwise DAG infrastructure.

## Follow-up tasks

- CPU 0008L remains the next Draft task and owns SIMD mask/output closure, not cross-type CAST SIMD.
- CPU 0008M–0008P retain Draft order and scope.
- CPU 0009 remains the generated-coverage checkpoint and owner of CPU 0008I's recorded evidence gap.
- No negative-stride prerequisite is added.

## Risks

- Direct BFLOAT16 rounding is the highest semantic risk: a superficially convenient FLOAT32
  intermediate silently double-rounds required FLOAT64/integral cases. Fixed raw counterexamples,
  generated decompilation, and clean-Java parity are mandatory controls.
- Java primitive conversion syntax is not by itself evidence for NaN payload mapping, direct
  target rounding, or every saturation boundary. Each structural group requires raw-bit tests and
  an independently reviewable emitted algorithm.
- The 576-Class-File inventory can create misleading evidence if normalized groups conceal a
  carrier/access branch. The manifest must retain every class and relate each one to its exact
  source/target, carrier, access, and oracle-equivalence row.
- Performance pressure could encourage a helper call, fixed-fixture specialization, or SIMD scope
  expansion. The forbidden-reference scan, cold-identity checks, and explicit scalar-only boundary
  are stop gates rather than optimization suggestions.

## Architecture impact

Expected impact: None.

Required no-change conclusions:

- Architecture files, ADRs, and architecture tests stay unchanged because ownership and
  dependencies do not change.
- Model, `LayoutDescriptor`, `CpuAccessPlan`, Model capabilities, and Tensor/Compile APIs stay
  unchanged because CAST semantics already exist and negative strides remain unrepresentable.
- Compiler and Training API stay unchanged because differentiation is preserved.
- Shared Prepare and Runtime stay unchanged because current slots, finalization, scalar artifact,
  ranges, and publication contracts suffice.
- Backend-conformance stays unchanged because no CAST conformance owner exists; the exhaustive
  CPU-private oracle/evidence matrix validates this first executable backend. A reusable
  cross-backend suite is a later checkpoint concern.
- Integration stays unchanged because no Engine/end-to-end public lifecycle changes.
- Build boundaries stay unchanged: dependencies, plugins, source sets, Java 26 toolchain, and
  ordinary build/test behavior are unchanged. `backends/cpu/build.gradle.kts` necessarily forwards
  three opt-in CAST evidence properties to Test JVMs; this narrow test plumbing is the only Gradle
  file change and does not enable retained evidence by default.
- Other backends and CPU 0008L–0008P stay unchanged.

Re-evaluate these against the final diff and stop if any becomes false.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean-context implementation agent for Synaptik CPU task 0008K.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, and task
0008K in full. Read completed Model 0025L, CPU 0008J, the referenced generated-parity tasks, and
every affected current CPU source/test owner. Implement exactly the Ready task.

Preserve Model semantics and current non-negative layout, preparation, finalization, cache,
scalar/caller-parallel, publication, and pointwise-DAG contracts. Emit conversions directly; do
not call conversion helpers from generated element code. Do not add negative storage strides,
SIMD, public/shared APIs, resources, routes, fallback, or unrelated refactoring. Stop on
architecture, semantic, optimal-clean-Java parity, evidence, or path-budget conflict. Do not
commit, push, stage, reset, or modify unrelated work.

After executable work and all Java/Class-File/performance gates stabilize, hand the exact diff and
evidence to a distinct clean documentation-focused agent following documentation-rules.md. Do not
mark Complete until every acceptance criterion and that pass succeed.
```

## Local decisions

- Negative storage strides were removed after source review proved current `LayoutDescriptor` and
  `CpuAccessPlan` reject them. Negative slice steps remain a distinct semantic concept.
- One existing CAST opcode plus ordered source/output types is sufficient; no runtime conversion
  vocabulary is added.
- Schema 60 is isolated to cross-type CAST so older class bytes remain stable.
- The 1,296 semantic cells are the complete legal product; the 576 structural classes remove cold
  offset/parallel duplicates; performance measures distinct code shapes rather than every class.
- Retained performance samples use a symmetric AB/BA geometric estimator after separate untimed
  compilation stabilization. This removes the one-sided temporal-position bias diagnosed in the
  original alternating-sum estimator without changing workloads, work counts, the five forks,
  nine rows, nine samples, 250 ms-per-side calibration, per-sample `1.15x` limit, or no-retry/
  no-discard policy.

## Known limitations

- Cross-type CAST has no Vector API or native realization.
- Negative strides, dynamic layouts, non-injective outputs, overlap/in-place conversion, and
  non-canonical BOOL carriers remain unsupported.
- Performance evidence is local and is not tuning input or a whole-backend claim.

## Validation evidence

- The implementation context's focused 12-owner command passed 185 tests: 184 successful and one
  expected guarded skip. The successful focused suite was not repeated by the documentation pass.
- The implementation context's pre-correction full CPU run passed 705 tests with 16 expected
  skips. After the final dense FLOAT64-to-BFLOAT16 code-shape correction, documentation context
  `01a07240-d2f5-70e3-a9d4-0cf63e2fc66b` ran
  `./gradlew :backends:cpu:test --rerun-tasks`: passed in 23 seconds across 135 suites and 708
  tests, with 16 expected skips and zero failures or errors.
- The retained structural command passed its single guarded test in 1 minute 20 seconds under
  `/private/tmp/synaptik-cpu-0008k-A4NSiC7q`. The root contains 576 classes, 576 complete `javap`
  reports, 2,304 structural reports, and 3,464 structural manifest entries. The structural
  manifest file has SHA-256
  `543375e294a0cfef45870a4d39c64af26689df1c4d2927f9daadf9cdfcd785fa`.
- The retained performance command passed seven tests in 52 minutes 52 seconds. All 405 retained
  samples passed: the global generated/direct range was
  `0.096962322231808..1.072864015536589`, the maximum fork median was
  `1.0306215940115075`, and the maximum aggregate median was `1.0291098600274287`. The performance
  manifest has 65 entries and SHA-256
  `ff14040ab0127ae4b97461e4f62e09bd5c0f16b6151a285f161789edec7f6903`.
- The retained benchmark used five fresh C2-only fixed-heap forks, nine rows, nine retained samples
  per row and fork, 64 chunks, 32 warmups, and 12,000 untimed compile-stabilization invocations.
  Each sample used symmetric generated/direct/direct/generated or direct/generated/generated/
  direct blocks, the geometric mean of directional ratios, at least 250 ms calibration per side,
  a `<= 1.15x` threshold, and no retry or discard.
- Documentation context `01a07240-d2f5-70e3-a9d4-0cf63e2fc66b` independently reviewed the final
  implementation diff, all affected production Javadocs and package summaries, Model 0025L,
  capability/IR/preparation/cache/reference boundaries, CPU guide, glossary, task, master plan,
  roadmap, build-property forwarding, retained manifests, and required no-change areas. It changed
  no executable Java behavior or test.
- Documentation validation: `./gradlew :backends:cpu:javadoc` passed; targeted local Markdown
  link/anchor, balanced-fence, final-newline, matrix/count, schema/status/order, package-placement,
  and exact-path checks passed; `git diff --check` and `git diff --cached --check` passed; staging
  remained empty and the final changed-path count remained at most 40.
- Architecture, backend-conformance, and integration suites were not run because no ownership,
  dependency, shared contract, or public end-to-end lifecycle changed. Repository-wide Java
  validation remains deferred to CPU 0009/CI as specified.

## Implementation notes

- Capability, lowering, IR validation, preparation, direct emission, cache compatibility, and the
  cold scalar differential oracle now cover all 36 ordered CAST pairs across FLOAT64, FLOAT32,
  BFLOAT16, INT64, INT32, and BOOL.
- Generated scalar and caller-parallel scalar execution covers primitive arrays, native-order
  `MemorySegment` carriers, and all mixed carrier directions over contiguous/offset-dense,
  positive block/general strided, and rank-zero access. Negative storage strides remain
  unrepresentable and out of scope; negative slice steps are a distinct operation-coordinate rule.
- Conversion matches Model 0025L exactly: identity preserves represented bits; FLOAT64/integral to
  BFLOAT16 rounds directly to nearest ties-to-even; lossy floating NaNs canonicalize; widening
  follows the exact NaN fraction mapping; floating-to-integral truncates, saturates, and maps NaN
  to zero; INT64-to-INT32 retains low bits; and BOOL output is canonical.
- Any bounded pointwise DAG containing cross-type CAST is scalar-only and commits every explicit
  target representation before the next consumer. Existing same-type vector eligibility is
  unchanged. No SIMD/native route, materialization, workspace, representation, route-selection,
  or autotuning policy changed.
- Schema 60 applies only to cross-type CAST. Unchanged schema-52 and schema-59 projections retain
  byte-identical generated classes and safely miss older incompatible envelopes.
- Every retained generated class is self-contained, final, field-free, and exposes one typed
  static entry without Synaptik helper leakage. Dense FLOAT64-to-BFLOAT16 was compacted from 392
  bytes and 39 locals to 287 bytes and 30 locals after C2 rejected the former shape as
  `hot method too big`; the semantic algorithm did not change.
- `backends/cpu/build.gradle.kts` changed only to forward the three opt-in CAST evidence properties
  to Test JVMs. Dependencies, plugins, source sets, the Java 26 toolchain, and ordinary build/test
  behavior remain unchanged.
- Final no-change conclusions: architecture contracts/pages, ADRs, and architecture tests remain
  unchanged because ownership and dependencies did not change; Model, Model capabilities,
  `LayoutDescriptor`, and the Model/shared CAST semantics remain unchanged because Model 0025L is
  authoritative and negative storage strides remain unrepresentable; shared Prepare and Runtime
  remain unchanged because current slots, finalization, ranges, publication, and run-isolation
  contracts suffice; Compiler and Training APIs remain unchanged because differentiation was not
  redesigned; backend-conformance remains unchanged because no shared CAST conformance owner
  exists and the first executable backend uses exhaustive CPU-private oracle/evidence coverage;
  integration remains unchanged because Engine and the public lifecycle did not change; other
  backends remain unchanged because CPU owns this realization; and CPU 0008L remains the next
  Draft frontier without a detailed task specification.

## Completion summary

- Completed changes: implemented and documented exact generated scalar/caller-parallel scalar
  execution for all 36 Model-defined CAST pairs, including bounded pointwise DAG conversion
  boundaries, schema-60 isolation, and exhaustive structural/performance evidence.
- Files changed or created: 36 paths in the final overall change, comprising one Gradle evidence-
  plumbing file, 15 production/Javadoc/package-summary paths, 15 test/evidence paths, and five
  documentation/planning paths.
- Tests and validation: focused 185-test evidence, retained 576-class structural evidence,
  retained 405-sample performance evidence, post-correction 708-test full CPU module validation,
  CPU Javadoc, Markdown/status/scope checks, and clean diff/cached-diff checks all passed.
- Documentation-agent review: complete in clean context
  `01a07240-d2f5-70e3-a9d4-0cf63e2fc66b` without executable behavior or test changes.
- Documentation impact: the CPU backend guide now explains all-pair conversion, carriers/access,
  scalar-only DAG behavior, negative-stride boundary, schema isolation, and deliberate exclusions.
- Javadoc review: every affected production contract and containing package summary now describes
  CAST semantics, inputs/results/failures where applicable, lifecycle ownership, strategy, and
  unsupported boundaries.
- Glossary impact: the existing CPU portable-route entry now records completed cross-type CAST and
  distinguishes it from SIMD CAST, negative strides, materialization, native routes, and tuning.
- Unresolved issues: None.
- Follow-up required: CPU 0008L remains the next Draft frontier; do not create its detailed task
  until it becomes current.

Status: Complete
