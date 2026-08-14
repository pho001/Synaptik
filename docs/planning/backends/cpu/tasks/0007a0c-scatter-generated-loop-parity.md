# Task 0007A0C: Scatter Generated-Loop Parity

## Status

Complete

## Goal

Replace the bridge-only generated output and contribution work for the completed CPU functional
scatter family with carrier-, type-, reduction-, access-, and family-specialized loops embedded
in each generated class. `SCATTER_ELEMENTS`, Gather-compatible `SCATTER_ADD`, and `SCATTER_ND`
must preserve every CPU 0006B1 semantic, validation, layout, carrier, range, resource, worker,
scratch, and lifecycle contract while two representative dense cases independently reach
measurable near-parity with equivalent direct Java loops.

This is one cohesive CPU-private correction. The three forms share `CpuScatterIr`, compact cold
geometry, deduplicated boundary roles, `CpuScatterEmitter`, canonical contribution scanning,
represented reductions, optional exact floating-product scratch, and one generated-artifact
compatibility boundary. The task changes generated realization only; completed CPU 0006B1
capability and semantics remain unchanged.

## Scope

### Mandatory pre-edit source and bytecode baseline

- Before changing production Java, create a fresh isolated `/tmp` evidence directory and capture
  representative current schema-24 generated classes for:
  - dense unique INT32-index `SCATTER_ELEMENTS + NONE` over FLOAT32 heap arrays;
  - dense duplicate-index INT32 `SCATTER_ADD` over FLOAT32 heap arrays;
  - one `SCATTER_ND` general-layout or segment/mixed-carrier form; and
  - one floating `MUL` form whose entry includes the declared scratch `MemorySegment`.
- Retain the probe source, exact compile/generate/decompile/run/summarize commands, Java and
  operating-system facts, class bytes, complete `javap -c -p` and `javap -v -p` output, sizes,
  SHA-256 checksums, raw fork samples, summaries, ratios, and verification results together.
- Record the current generated entry descriptors and constant-pool method references. The known
  current shape is a typed outer descriptor whose body only loads arguments and invokes
  `CpuScatterEmitter.execute2*`, `execute3*`, or `execute4*` through `Object` parameters; do not
  rely on this planning statement in place of the required fresh baseline.
- Time only warmed, already-validated output/contribution work. Exclude lowering, generation,
  class definition, artifact lookup, cold binding, allocation, input initialization, complete
  bounds/duplicate validation, result verification, and sink observation from both timed forms.

### Embedded specialized scatter work

- Change `CpuClassFileKernelGenerator` and `CpuScatterEmitter` so each generated entry embeds the
  selected scatter family, reduction, data type, index width, carrier accesses, contribution
  matching, represented reduction, and output store rather than calling an `Object` bridge for
  output-cell or contribution work.
- Select `SCATTER_ELEMENTS`, Gather-compatible `SCATTER_ADD`, or `SCATTER_ND`; `NONE`, `ADD`,
  `MUL`, `MIN`, or `MAX`; FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, or canonical BOOL where
  admitted; INT32 or INT64 indices; and each ordered heap/segment carrier form at generation time
  from existing typed CPU-private IR and specialization facts.
- Preserve unique boundary declaration and the exact three-role occurrence map when data,
  indices, or updates reuse the same `ValueId`. The generated class must address each semantic
  role through its mapped direct typed boundary without reconstructing generic carriers.
- Preserve the universal primitive `long start, long end` entry. Each arbitrary legal half-open
  range owns disjoint output coordinates, reads the functional base, scans matching logical
  update contributions in canonical row-major order, and writes every owned output coordinate
  exactly once.
- Generate direct typed array or native-order `MemorySegment` loads and stores. Use no hot
  `DataType`, `ScatterReduction`, family, carrier, array-versus-segment, string, map, reflection,
  graph, Runtime, route, or cache dispatch and no boxing or per-output/per-contribution allocation.
- Hoist family, reduction, ranks, boundary positions, axis/batch/tuple roles, and invariant packed-
  geometry positions out of the output and contribution loops when structural specialization
  makes them invariant. Concrete extents, offsets, stride magnitudes, ranges, and scratch offsets
  remain cold shape-polymorphic geometry.

### Dense and general generated forms

- For cold-proved dense heap-array boundaries, use the current `DENSE_HEAP_ARRAY_INT` category.
  Narrow universal bounds, bases, and other proved Java-array-index facts once before hot work,
  then retain primitive integer output/update/address state without per-contribution `l2i` or
  generic address helpers.
- Dense `SCATTER_ELEMENTS + NONE` must copy base for unaddressed outputs or the sole validated
  update for addressed outputs with no reduction path. Its unique-target timed case must not do
  duplicate detection inside the generated entry.
- Dense FLOAT32 `SCATTER_ADD` must include base once and add every duplicate contribution once in
  canonical logical update order using binary32 addition after each contribution. Its timed
  direct baseline must use the same output-domain scan and the same ordered binary32 work; an
  update-centric, pre-grouped, reordered, atomic, widened, or otherwise semantically easier
  direct algorithm is forbidden.
- Generate typed `GENERAL_LONG` forms for every supported `MemorySegment`, mixed-carrier,
  arbitrary non-negative layout, zero-stride read, offset/strided injective output, large range,
  and otherwise unproved case. Use primitive carry/reset state and only coordinate arithmetic
  genuinely required by the structural form.
- A failed dense proof selects the correct general form and never rejects a CPU 0006B1-supported
  occurrence. Preserve scalar and zero domains, empty contributions, scalar indices where legal,
  both index widths, all current families/reductions/types, and arbitrary legal ranges.

### Semantic and validation invariants

- Preserve functional behavior: inputs are never mutated; the result has the exact data Shape and
  type; an unaddressed output copies exact base bits; `NONE` replaces without base participation;
  arithmetic target groups include base exactly once and every duplicate contribution exactly
  once.
- Preserve complete deterministic logical row-major bounds validation, followed for
  `SCATTER_ELEMENTS + NONE` and `SCATTER_ND + NONE` by complete deterministic target-uniqueness
  validation, before any generated call, worker submission, scratch mutation, or output write.
- Preserve exact exception types, messages, logical ordinals, selected axes/extents, first-later
  duplicate reporting, independent empty-index/zero-output domains, and complete no-write behavior
  for every invalid binding.
- Generated work consumes only validated indices. It must not normalize, wrap, clamp, ignore,
  default, or revalidate indices; choose first/last duplicate replacement; or expose partial output.
- Preserve canonical logical contribution order independently of physical layouts and worker
  scheduling. FLOAT64/FLOAT32 add in the same format, BFLOAT16 rounds after each addition,
  INT32/INT64 arithmetic is modular, extrema use exact signed/floating rules, and `NONE` preserves
  represented bits.
- Preserve scalar/parallel-scalar bitwise parity, disjoint output ownership, read-only shared
  inputs, invocation-private coordinate state, optional disjoint scratch slices, and absence of
  atomics, locks, cross-range partials, barriers, or merge work.

### Exact floating-product scratch

- Preserve the CPU 0006B1 once-rounded abstract floating product for FLOAT64, FLOAT32, and
  BFLOAT16 `MUL`, including base participation, all finite factors, ties-to-even, overflow,
  subnormal/underflow, sign parity, signed zero, infinity, NaN, and zero-times-infinity behavior.
- Preserve exactly one declared `SCATTER_PRODUCT` workspace when and only when a non-empty
  floating-product calculation is possible. Retain checked per-range slice sizing, eight-byte
  alignment, fixed metadata, precision-derived primitive limbs, disjoint range offsets, and no
  materialization or second workspace.
- Embed the type-specialized factor classification, significand multiplication, exponent state,
  and final rounding operations in generated work. Calls are allowed only to the minimal JDK or
  `MemorySegment` primitives necessary to realize those operations; no generic scatter helper,
  `Object` access, family/type/reduction dispatcher, `BigInteger`, collection, array, cursor,
  table, or per-factor/per-output allocation may enter the hot path.
- Do not include floating `MUL` in either parity ratio. Record its scratch-bearing class shape,
  semantic differential coverage, allocation/dispatch evidence, and workspace accounting
  separately so a scratch-free benchmark cannot hide regression in this distinct contract.

### Resources, preparation, finalization, and Runtime boundary

- Preserve one lowered unit, one artifact, two through four exact buffers after input
  deduplication, one distinct injective output, no input materialization, scalar or parallel-
  scalar strategy, and only the optional exact product workspace above.
- Preserve complete carrier size/alignment/accessibility/writability validation, canonical BOOL
  input validation, full referenced-span output/input non-overlap, permitted input/input overlap,
  exact assignment validation, one artifact lookup after shared slot assignment, and prepared
  resource ownership.
- CPU analysis remains deterministic and measurement-free. CPU finalization generates or reuses
  the selected class after shared assignment. Runtime receives an immutable prepared executable
  and performs no scatter interpretation, generation, specialization, cache lookup, route choice,
  tuning, fallback, or benchmark-driven mutation.

### Generated compatibility

- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly from `24` to `25`. Generated scatter bytes
  and their compatibility envelope necessarily change because the output/contribution body moves
  into the class.
- Schema 25 records the embedded scatter family/type/reduction/carrier/access body and the existing
  optional scratch-bearing entry distinction. Treat schema 24 as an incompatible safe miss. In
  `CpuGeneratedKernelArtifactStoreTest`, write an otherwise valid envelope with version 24 and
  valid current metadata/class bytes, prove it is not loaded, and prove lookup regenerates and
  publishes the schema-25 artifact. Add no migration reader, converter, compatibility alias, or
  legacy execution bridge.
- Preserve compatible shape-polymorphic reuse. Concrete axes, batch count, tuple depth, extents,
  layout magnitudes, counts, ranges/chunks, values, validation results, scratch byte count/offset,
  carrier objects, byte offsets, slots, workers, and run identity remain cold.

### Stable automated evidence

- Add Class-File model tests for every scatter family, all five reduction modes where admitted,
  every data type, both index widths, heap-array and `MemorySegment` carriers, mixed and deduplicated
  boundary patterns, dense integer and general long forms, arbitrary layouts/ranges, scratch-free
  and scratch-bearing entries, and scalar/parallel prepared use in proportion to each risk.
- Prove generated classes contain typed loops, matching operations, reduction instructions, and
  direct stores. Prove absence of `CpuScatterEmitter.execute*` or an equivalent bridge, `Object`
  descriptors/casts, generic family/type/reduction/carrier dispatch, `DataType.values`,
  `ScatterReduction.values`, per-element allocation, boxing, reflection, string/map dispatch,
  graph/Runtime/backend/cache lookup, and avoidable helper calls. Allow only the closed necessary
  carrier/JDK/`MemorySegment` primitives asserted explicitly by the tests.
- Test both dense and general class shapes without source-text matching, fragile absolute bytecode
  offsets, or manual-only `javap`. The retained decompilation is audit evidence; automated
  Class-File assertions are the durable gate.
- Preserve or extend independent semantic differential evidence. Expected scatter results must
  come from `CpuScalarReferenceKernel` or test-local oracle code that neither calls generated
  helpers nor shares the generated reduction/scratch implementation.

### Performance acceptance protocol

- Before production edits and again after stabilization, run these two independent dense heap-
  array cases. Both use rank-one contiguous FLOAT32 data and updates, INT32 indices, axis zero,
  data extent 1,024, update extent 1,024, and therefore exactly 1,048,576 output/candidate-
  contribution examinations under the retained output-domain algorithm. Initialize
  `data[i] = (float) ((i % 251 - 125) * 0.25)` and
  `updates[i] = (float) ((i % 127 - 63) * 0.125)` in both forms:
  - unique-target `SCATTER_ELEMENTS + NONE`, with indices
    `indices[i] = (5 * i + 17) & 1023`, a deterministic permutation of every target; and
  - duplicate-index Gather-compatible `SCATTER_ADD`, with `indices[i] = i >>> 1`, so targets
    0 through 511 each receive exactly two FLOAT32 contributions in logical update order and
    targets 512 through 1,023 retain their base value.
- Use the exact same case definitions, direct algorithms, deterministic inputs, fixed heap,
  compiler/run commands, and measurement/summarization code for baseline and final evidence.
- Use five fresh JVM forks, fixed `-Xms1g -Xmx1g`, at least five randomized warmup batches per
  fork, nine randomized measurement rounds per fork, and adaptive batches lasting at least 25
  milliseconds. Verify exact inputs and outputs before and after measurement and consume an
  observed sink outside the timed region.
- Report every generated and direct per-fork median and ratio. Acceptance is independent per case:
  the median of generated per-fork medians must be no greater than `1.15x` the median of the
  corresponding direct-loop per-fork medians. Do not average cases, change shapes or work after
  measurement, widen the threshold, or substitute a passing family for a failing one.
- Keep timing outside Gradle/JUnit. Evidence is observational and cannot select or mutate
  production behavior. If either case exceeds `1.15x`, retain `In progress` or `Review needed`,
  inspect generated and just-in-time code shape, and correct the implementation.

## Out of scope

- New Model semantics, operations, attrs, reductions, types, Shapes, negative-index policy,
  capabilities, public APIs, or historical scatter forms
- Fold, ordering, random/dropout, pointwise, affine/movement, indexing, scan, aggregate, vector
  scatter/reduction, native-provider, or later semantic-family work
- New routes, resources, workspace kinds, materialization policy, public tuning controls,
  tuning-cache changes, fixed-shape classes, unrolling, algorithm substitution, relaxed numerics,
  in-place output, negative strides, dynamic Shapes, atomics, or update-centric execution
- Architecture, module-boundary, dependency, shared Model/Compiler/Planning/Prepare/Runtime,
  Config, Trace, Engine, Gradle/build, Java-version, architecture-test, backend-conformance,
  integration-test, other-backend, NN, training, or unrelated documentation changes
- Detailed specifications or implementation for CPU 0007A0D–0007A0F or CPU 0007A1
- Commit, push, staging, revert, deletion, or modification of unrelated concurrent work

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Runtime/Prepare/Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU 0006B1 portable functional scatter](0006b1-portable-functional-scatter.md)
- [CPU 0007A0 generated hot-path parity correction](0007a0-generated-hot-path-parity-correction.md)
- [CPU 0007A0A affine and movement parity](0007a0a-affine-and-movement-generated-loop-parity.md)
- [CPU 0007A0B indexing parity](0007a0b-indexing-generated-loop-parity.md)

## Architecture constraints

- Model owns scatter meaning. This task changes only CPU-private generated realization and cannot
  reinterpret, narrow, or extend a semantic contract.
- Planning selects CPU ownership only. CPU analysis owns deterministic lowering, specialization,
  strategy, and exact declarations; CPU finalization owns compatible generation/reuse after shared
  assignment; Runtime executes the immutable prepared result only.
- Generated hot code receives typed direct carriers and primitive cold geometry. No `Operation`,
  `CompiledNode`, Tensor, graph object, backend discovery, route/cache policy, validation decision,
  or benchmark evidence reaches it.
- Prepared recipes remain immutable and reusable. Concurrent runs use distinct `RunState` and
  invocation-private geometry/scratch state; generated classes retain no mutable run state.
- Complete validation-before-write, canonical contribution order, general carrier/layout/range
  support, exact resource declarations, and Runtime/Prepare ownership remain unchanged.
- Stop if parity requires a public/shared contract, architecture, dependency, semantic, route,
  resource, materialization, Runtime-policy, or broader-module change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — typed scatter family,
  contribution, reduction, scratch, carrier, and dense/general Class-File emission.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — exact schema and artifact compatibility.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — existing closed scatter family,
  reduction, occurrence-map, boundary-type/access, and scratch-signature identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — existing compact cold geometry,
  canonical mapping, and exact scratch sizing.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing deterministic plan,
  declarations, specialization, and final assignment checks.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — existing complete validation-
  before-write, direct binding, geometry packing, range/worker invocation, and scratch slicing.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent oracle evidence only.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — generated route Javadoc and
  existing direct scratch-bearing entry contract.

Packages added or changed:

- No package is added, removed, moved, exported, or made supported API.

Type placement:

- `CpuScatterEmitter` remains the sole owner of generated scatter mapping, typed reduction, and
  exact-product emission. It becomes an embedded-body emitter rather than a generic bridge owner.
- `CpuClassFileKernelGenerator` passes the existing structural `CpuKernelIr` to the focused
  emitter; it does not acquire scatter semantics or Runtime state.
- `CpuCarrierEmitter` remains the unchanged focused owner of primitive-array and native-order
  segment instruction emission; do not move scatter semantics into it.
- Existing `CpuScatterIr`, `CpuScatterLowering.Geometry`, preparation, executable validation, and
  reference types retain their current owners and contracts. Add no dispatcher, registry,
  manager, facade, generic math utility, or new public type.

## Affected files

Expected production and package-documentation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0007a0c-scatter-generated-loop-parity.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized. An expected path may remain unchanged after explicit review. Before
substituting a path, update this specification with the replacement and reason while keeping the
same 20-path ceiling and owning module/documentation area.

Review without modification and record explicit conclusions for:

- `ARCHITECTURE.md`, focused architecture pages, ADRs, and architecture tests;
- Model scatter operations, Tensor/API and Compile API documentation, Compiler, shared Planning,
  Prepare, Runtime, Config, Trace, Engine, backend conformance, integration, Gradle/build, native
  routes, other backends, NN, and training;
- `CpuScatterIr`, `CpuScatterLowering`, `CpuPartitionPreparationPlan`, `CpuPreparedExecutable`,
  `CpuScalarReferenceKernel`, `CpuContiguousWorkspace`, and their package contracts when their
  existing semantics, validation, scratch, and lifecycle Javadocs remain accurate; and
- CPU 0007A0D–0007A0F and 0007A1, which remain Draft without detailed specifications.

## Maximum scope

This task may create or modify at most 20 repository paths: 7 production/package-documentation
paths, 8 CPU test paths, and 5 documentation/planning paths listed above. Stop and revise this
specification before a 21st path, a new package, an executable path outside `backends/cpu`, or a
documentation path outside the allowlist. Unused capacity does not authorize cleanup.

## Acceptance criteria

- The complete pre-edit schema-24 source/Class-File audit and both timing baselines are retained
  before production edits; every class descriptor, constant-pool bridge, checksum, command,
  environment fact, raw fork sample, summary, and exact verification result is reproducible.
- Generated classes for all three scatter families embed carrier/type/reduction/access-specialized
  output and contribution loops and contain no `CpuScatterEmitter.execute*` or equivalent
  `Object` bridge.
- Dense heap-array forms use direct typed integer loop/address state; every segment, mixed,
  arbitrary-layout, large-range, or otherwise unproved form retains a correct typed general-long
  generated fallback.
- Every CPU 0006B1 family, reduction, type, index width, occurrence-map/deduplication, scalar/zero
  domain, layout, carrier, range, parallel, overlap, validation, and resource behavior remains
  correct against an independent oracle.
- Complete bounds then `NONE` uniqueness validation remains on the bound invoking thread before
  generated work or worker submission; every invalid case leaves output and scratch unchanged.
- Exact floating-product scratch preserves once-rounded semantics, one declared/disjoint
  workspace, checked size/alignment, and allocation-free generated work; it is verified separately
  from the scratch-free performance cases.
- Stable Class-File tests prove typed generated operations and the absence of bridges, `Object`
  carrier descriptors/casts, generic family/type/reduction dispatch, allocation, reflection,
  boxing, string/map dispatch, Runtime/backend/cache lookup, and avoidable helper calls.
- Generated compatibility advances exactly `24 -> 25`; schema 24 is an incompatible safe miss,
  no migration reader exists, and compatible concrete geometry still reuses one artifact.
- Unique replacement `SCATTER_ELEMENTS` and duplicate-index FLOAT32 `SCATTER_ADD` each pass the
  unchanged independent `<= 1.15x` median-of-fork-medians gate under the complete five-fork
  protocol and equivalent semantic/algorithmic work.
- No public API, capability, semantic, route, materialization, resource kind, architecture,
  dependency, shared-module, Runtime-tuning, build, conformance, integration, later-task, or
  unrelated change is present.
- A distinct clean documentation-focused context independently reviews the final diff and
  evidence, finalizes affected Javadocs/package summaries, CPU guide, glossary impact, task,
  master plan, and roadmap, and records every no-change conclusion before completion.

## Tests / validation

Run focused tests for every changed test owner while iterating. The final focused command must
name every changed test class and cover Class-File shape, scatter semantics, independent oracle,
validation-before-write/worker submission, arbitrary range/parallel execution, scratch slicing,
preparation/finalization, specialization, and schema/persistence compatibility.

After executable Java stabilizes, run exactly one authoritative affected-module command:

```bash
./gradlew :backends:cpu:test
```

Run the isolated baseline/final performance probe separately from Gradle/JUnit using the retained
commands and unchanged five-fork protocol. Timing is evidence only and is not an ordinary test.

The distinct clean documentation pass reuses successful Java and timing evidence unless it
changes executable Java or probe behavior, or records a concrete stale-evidence reason. It runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also inspects rendered affected Javadocs; validates local Markdown links/anchors, balanced
fences, terminal newlines, trailing whitespace, exact changed-path allowlist/count, package/type
placement, schema-25 history and schema-24 rejection, stable bytecode claims, baseline/final
protocol/ratios/checksums, status/dependency synchronization, and exact
0007A0C/0007A0D–0007A0F/0007A1 ordering.

Repository-wide, architecture, backend-conformance, and integration validation remains deferred
to CPU 0009 or continuous integration because this task changes one concrete backend and no
shared or architecture contract. Stop and replan if implementation makes that conclusion false.

## Dependencies

- [CPU 0006B1](0006b1-portable-functional-scatter.md) is `Complete` and owns the exact current
  scatter semantics, validation-before-write order, occurrence mapping, canonical contributions,
  represented reductions, arbitrary layout/carrier/range behavior, exact-product scratch,
  reference oracle, worker safety, declarations, and schema-16 history preserved here.
- [CPU 0007A0](0007a0-generated-hot-path-parity-correction.md) is `Complete` and supplies the
  shared typed carrier emission, dense integer loop/address proof, stable Class-File testing, and
  retained five-fork performance protocol.
- [CPU 0007A0A](0007a0a-affine-and-movement-generated-loop-parity.md) is `Complete` and establishes
  the current dense/general correction pattern and schema-23 history.
- [CPU 0007A0B](0007a0b-indexing-generated-loop-parity.md) is `Complete` and establishes current
  schema 24 plus embedded mixed-type indexing loops while preserving separate pre-write validation.
- Current Java 26 Class-File API, shared Prepare/Runtime, CPU worker, direct carrier binding,
  exact workspace, and generated-artifact cache contracts are complete and unchanged.

## Follow-up tasks

- CPU 0007A0D remains the next `Draft` corrective row and owns fold generated-loop parity after
  this task completes; do not create its detailed specification here.
- CPU 0007A0E and CPU 0007A0F remain ordered `Draft` corrections for ordering and random/dropout.
- CPU 0007A1 remains `Draft` and explicitly depends on CPU 0007A0F before semantic-family work
  resumes.
- CPU 0009 retains the generated-coverage and conformance checkpoint.

## Architecture impact

Expected impact: None.

The task changes only CPU-owned generated scatter code shape and current-only artifact
compatibility under the existing analysis/finalization/binding lifecycle. If implementation
requires an architecture, public/shared contract, module, dependency, semantic, route, resource,
materialization, or Runtime-policy change, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Implement Synaptik CPU task 0007A0C exactly from
docs/planning/backends/cpu/tasks/0007a0c-scatter-generated-loop-parity.md. Use a separate clean
implementation context. Do not use GSD. Do not commit, push, stage, revert, delete, or modify
unrelated work.

Read in full AGENTS.md, ARCHITECTURE.md, the current architecture plan and directly relevant
architecture/performance/Runtime-Prepare-Backend documents, documentation rules and applicable
profiles, planning guide, roadmap, CPU master plan, task 0007A0C, completed tasks
0006B1/0007A0/0007A0A/0007A0B, and every affected or directly relevant current CPU source/test/
package file. Inspect the complete worktree diff and preserve concurrent unrelated changes.

Record the required schema-24 generated-class decompilation and both performance baselines before
production edits. Then implement embedded typed scatter output/contribution/reduction/scratch
work, exact schema 25 compatibility, stable Class-File and semantic tests, focused validation,
one authoritative CPU suite, and the unchanged isolated five-fork per-case parity gates. Preserve
all CPU 0006B1 validation-before-write, canonical-order, exact-product, carrier/layout/range,
worker, declaration, and lifecycle contracts. Stop on any architecture, scope, semantic, shared-
contract, route, resource, lifecycle, or unresolved performance conflict.

After executable Java and probe evidence stabilize, hand the same uncommitted diff and exact
evidence to a distinct clean documentation-focused context following
docs/developer-guide/documentation-rules.md. That context must independently inspect final source,
tests, class shape, and evidence; finalize affected Javadocs/package summaries, CPU guide,
glossary impact, task/master/roadmap, rendered pages, Markdown, exact scope, schema, status/order,
performance evidence, and whitespace; and not repeat successful Java or timing runs unless it
changes executable/probe behavior or records a concrete stale-evidence reason. Do not mark
Complete until both contexts and every acceptance criterion succeed.

Return exact changed paths, commands/results, environment and per-fork/per-case ratios,
documentation/Javadoc/glossary impact and explicit no-change conclusions, context IDs,
unresolved issues, and exactly Status: Complete or Status: Incomplete with
Follow-up required: <specific follow-up>.
```

## Local decisions

- Keep all three scatter families and every reduction together because they share one bridge-only
  emitter, one output-domain contribution algorithm, one scratch-signature boundary, and one
  compatibility schema.
- Measure already-validated writers so timing isolates the bridge/output-contribution defect while
  production retains mandatory validation. Direct loops perform identical output-domain mapping
  and canonical contribution work.
- Require separate replacement and duplicate-add gates because represented-bit selection and
  ordered reduction have different control flow; neither passing case proves the other.
- Advance directly to schema 25 because generated scatter bytes necessarily change. Schema 24 is
  the single prior incompatible envelope and concrete geometry remains cold.
- Keep floating-product scratch outside timing and inside semantic/Class-File/resource evidence so
  the benchmark neither simplifies nor conceals its distinct exact arithmetic contract.

## Known limitations

- Performance gates cover two representative large dense heap-array cases on the recorded local
  environment. They do not claim parity for every scatter family, reduction, type, carrier,
  layout, range, index width, CPU, or Java virtual machine.
- General segment/mixed/arbitrary-layout and floating-product forms receive semantic, resource,
  and stable Class-File validation but no timing threshold in this task.
- Complete bounds and replacement-uniqueness validation intentionally remains a separate scalar
  execution-time pass; generated output work reads validated indices again.
- Only the existing one-node fully static resolved-layout CPU scatter capability is corrected.
  Multi-node fusion, dynamic layouts, vector scatter, native routes, and new semantics remain out
  of scope.

## Validation evidence

Planning context `01a00085-1280-7bf1-be61-f6e7b1cf1e4f` read the governing repository,
architecture, performance, Runtime/Prepare/backend, planning, and documentation instructions;
the CPU master plan and roadmap; completed CPU 0006B1/0007A0/0007A0A/0007A0B tasks; and directly
relevant scatter IR/lowering/emitter/generator/specialization/cache/prepare/executable/reference
source, tests, package summaries, CPU guide, glossary, and retained benchmark evidence.

The pre-edit audit confirmed schema 24 is current and `CpuScatterEmitter.emit` still emits only an
`invokestatic execute2*`, `execute3*`, or `execute4*`. A retained representative current class at
`/tmp/synaptik-generated-bytecode-audit/scatter.class` is 570 bytes with SHA-256
`b3dc572bf64a3348a8ce70fde16e0d9f097a899e1224cc32d0eaeaf88e2d82ae`; complete `javap`
shows its sole typed segment entry loads arguments and calls
`CpuScatterEmitter.execute4(Object,Object,Object,Object,long[],long,long)`. Source and helper
bytecode show generic carrier selection, `DataType`/reduction lookup, per-output logical update
scans, generic represented-bit read/write/reduce helpers, and the separate exact-product scratch
implementation behind that bridge.

Planning-only validation confirmed the required section order and `Ready` status, 20 unique
allowlisted paths, 30 unique generated heading anchors, existing targets for every local Markdown
link, balanced fences, terminal LF newlines, no carriage returns or trailing whitespace, exact
schema/status/dependency order, one CPU `Ready` row, and no detailed 0007A0D–0007A0F or 0007A1
task. The final changed-path inspection contains only this task, the CPU master plan, and the
CPU-specific roadmap synchronization in addition to preserved concurrent NN work. Tracked
planning changes pass `git diff --check`. No Java test, Javadoc, or performance command is run for
this planning-only change.

## Implementation notes

- `CpuClassFileKernelGenerator` now passes the canonical structural IR to
  `CpuScatterEmitter`. The emitter parses only generation-time structural facts and embeds the
  selected `SCATTER_ELEMENTS`, Gather-compatible `SCATTER_ADD`, or `SCATTER_ND` family, data and
  index types, reduction, occurrence mapping, carrier access, contribution matching, and output
  store in the generated class. The former `execute2*`/`execute3*`/`execute4*` generic bridges and
  their generic represented-value access/reduction helpers were removed.
- Cold-proved dense rank-one heap-array `SCATTER_ELEMENTS` and `SCATTER_ADD` entries narrow the
  universal `long` bounds and bases once, then retain primitive integer output/update/address
  state. Typed general-long emission remains for `MemorySegment`, mixed-carrier, arbitrary-layout,
  large-range, and otherwise unproved forms. A failed dense proof therefore changes generated
  shape rather than narrowing supported CPU 0006B1 behavior.
- Generated work still consumes only validated indices. Complete deterministic row-major bounds
  validation, followed by complete deterministic target-uniqueness validation for replacement
  forms, remains in cold binding before any generated call, worker submission, scratch mutation,
  or output write. Each generated range reads the functional base, scans contributions in
  canonical logical order, and writes each owned output coordinate exactly once.
- FLOAT64, FLOAT32, and BFLOAT16 `MUL` entries now embed the exact-product classification,
  sign/exponent/significand state, primitive-limb multiplication, and ties-to-even final rounding.
  They use the one existing checked, aligned, disjoint `SCATTER_PRODUCT` slice and contain no
  generic scatter helper, `BigInteger`, collection, array allocation, or per-factor/per-output
  allocation. BFLOAT16 ADD/MIN/MAX paths likewise embed direct conversion, arithmetic/extrema, and
  ties-to-even narrowing without ordinal dispatch.
- `CpuGeneratorSchema.CURRENT_VERSION` advances exactly from 24 to 25. Schema-24 envelopes are
  incompatible safe misses and regenerate schema-25 bytes; there is no migration reader,
  converter, compatibility alias, or legacy execution bridge. Concrete geometry, values, ranges,
  scratch offsets/sizes, carriers, slots, workers, and run identity remain cold compatible facts.
- Performance evidence remains observational. The two scratch-free timed class files remained
  byte-identical after the final exact-product audit/fix, so their accepted five-fork evidence was
  reused. No timing threshold or universal performance claim applies to general layouts, segment
  or mixed carriers, other families/types/reductions/index widths, or scratch-bearing forms.

## Completion summary

Completed changes:

- Replaced bridge-only generated scatter work with typed direct dense/general loops for all three
  current scatter families and every admitted represented reduction/type/carrier combination.
- Embedded exact floating-product scratch state, advanced current generated compatibility from
  schema 24 to schema 25, and preserved complete validation-before-write and Runtime/Prepare
  ownership boundaries.
- Finalized affected implementation Javadocs, package summaries, CPU guide, existing glossary
  entries, this task, the CPU master plan, and the roadmap. CPU 0007A0D is the sole next `Draft`
  corrective frontier; CPU 0007A0E–0007A0F remain `Draft` without detailed specifications, and
  CPU 0007A1 remains behind CPU 0007A0F.

Files changed or created:

- Exactly 17 repository paths changed or were created: seven production/package-documentation
  paths, five CPU test paths, and five documentation/planning paths, all in the exact 20-path
  allowlist.
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0007a0c-scatter-generated-loop-parity.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Tests and validation:

- Reused final implementation/audit focused evidence: the exact eight changed/risk-owner suites
  passed 105 tests with zero skips, failures, or errors.
- Reused authoritative `./gradlew :backends:cpu:test` evidence: 53 suites and 312 tests, zero
  failures/errors, and one existing skip. No executable Java or probe behavior changed during the
  clean documentation pass, so the Java suite was not repeated.
- Reused the retained Java 26.0.1, macOS 26.5.2 aarch64 five-fork probe. Every individual fork was
  below `1.15x`; generated/direct median-of-fork-medians ratios were `0.979533` for unique
  `SCATTER_ELEMENTS + NONE` and `0.983230` for duplicate-index FLOAT32 `SCATTER_ADD`.
- Representative final schema-25 SHA-256 checksums are
  `557ba806c649f2e758b8013cd9090c1dd18dfd3a3ef57694e1be12b39da3b61e` for general-segment
  BFLOAT16 ADD, `13f20bacbab22a25aa741adaa78405407e8ff531b525c577c3528ad3a3837c95`
  for exact MUL scratch, `bbcf2cc940567dedc581530e8177964b95de274bc5886da96493b453c3b6c59c`
  for dense Elements NONE, `8b1b17b3fa83135279eadff55134e13f1285b92938642cee4bbd65b02c4e5194`
  for dense Scatter Add, and
  `ad1e3934e5ae9f1c2f7fc1c20c47e88575ff9fd175aed26e2783585056217631`
  for general-segment Scatter-ND.
- Clean documentation context `01a000c9-bb65-7d52-82ba-29ce0c26157a` ran
  `./gradlew :backends:cpu:javadoc` successfully with only the two expected incubating-module
  warnings and inspected the rendered affected type and package pages. Final Markdown
  link/anchor/fence/newline/whitespace, exact allowlist/path-count, schema/status/order/evidence,
  checksum, and `git diff --check` gates passed.

Documentation impact:

- The CPU guide now explains the embedded direct-loop boundary, dense/general distinction,
  validation-before-write ownership, inline exact-product scratch, schema-25 compatibility, and
  the bounded meaning of the retained performance evidence.
- Existing glossary entries were updated for the changed CPU generated-kernel and specialization
  boundaries. No new reusable project term was introduced, so no new glossary heading was needed.
- Javadocs and package summaries now describe the direct scatter body, optional inline
  exact-product state, schema 25, and the separation between generated work and cold validation.

Reasoned no-change conclusions:

- Public Tensor/API and Compile API documentation, Model scatter semantics, Compiler, and shared
  Planning/Prepare/Runtime/Config/Trace/Engine remain unchanged because this correction changes
  only CPU-private generated realization after the existing semantic and lifecycle boundaries.
- `ARCHITECTURE.md`, focused architecture pages, ADRs, dependency rules, and architecture tests
  remain unchanged because no ownership, module boundary, dependency, resource kind, or Runtime
  policy changed.
- Backend conformance and integration tests remain unchanged and their repository-wide checkpoint
  remains CPU 0009/CI because no public behavior or cross-backend contract changed.
- Gradle/build configuration and the Java version remain unchanged; existing Java 26 Class-File
  and incubating Vector API configuration already covers the implementation.
- Native routes, other backends, NN, and training remain unchanged because this task neither
  changes capability advertising nor adds a peer route, public semantic, gradient, or training
  behavior. Concurrent NN planning work was preserved and excluded from the CPU allowlist count.
- `CpuScatterIr`, `CpuScatterLowering`, `CpuPartitionPreparationPlan`, `CpuPreparedExecutable`,
  `CpuScalarReferenceKernel`, `CpuContiguousWorkspace`, and their package contracts remain accurate:
  their semantic occurrence mapping, compact geometry, validation, workspace, oracle, binding,
  and lifecycle responsibilities did not change.

Unresolved issues: None.

Follow-up required: None.

Status: Complete
