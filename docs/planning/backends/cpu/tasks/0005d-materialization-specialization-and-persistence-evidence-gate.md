# Task 0005D: Materialization, Specialization, and Persistence Evidence Gate

## Status

Complete

## Goal

Add one bounded CPU-private decision before shared resource assignment: compare direct universal
access with at most one contiguous input materialization for the completed fully static FLOAT64
`ADD -> exact GELU -> MUL` proving partition. The comparison accounts for the selected copy cost,
the generated-kernel benefit, reuse and fan-out, expected repeated runs, additional memory, and
portable-route eligibility. If materialization wins, CPU analysis records the copy as an immutable
lowering fact and declares its exact workspace before shared Prepare assigns slots. The Model
graph and backend-neutral logical memory remain unchanged.

At the same time, make the current generated specialization boundary explicit and bounded.
Code-shaping structural facts remain in specialization and artifact identity; concrete extents,
offsets, strides, assignments, carriers, workers, and run identity remain cold instance facts.
Fixed-shape and unrolled variants remain disabled because this task supplies no evidence for them.

Finally, run one reproducible, opt-in development evidence suite comparing verified in-memory
generation with a verified trusted-root class-byte hit. This suite runs during CPU 0005D
implementation, not for every model and never inside ordinary prepare or Runtime. Its report
records a policy verdict against the exact threshold below. Ordinary explicitly enabled
persistence performs only bounded lookup, compatibility/integrity/corruption verification, load
on hit, and generate with optional atomic store on miss. It never benchmarks or compares timing.

The lifecycle is:

```text
CPU analysis
  -> direct candidate + at most three one-input contiguous-copy candidates
  -> hard eligibility and memory filters
  -> bounded deterministic cost comparison
  -> one selected plan and exact declarations
  -> shared CPU-blind slot assignment
  -> CPU finalization and one generated artifact
  -> per-run selected copy, when present
  -> generated consumer execution

separate opt-in development evidence suite
  -> compare no-root generation with verified trusted-root hits
  -> write a report and KEEP_DISABLED or ENABLE_ELIGIBLE verdict
```

## Scope

- Preserve the completed one-partition, one-unit, fully static FLOAT64
  `ADD -> exact GELU -> MUL` topology, exact/default numerical mode, fixed instruction order,
  four logical boundaries, two virtual intermediates, right-aligned access plans, ordered
  heap/segment carrier specializations, and all four portable execution strategies.
- Add a CPU-private immutable `MaterializationPolicy` to `CpuPartitionAnalysisInputs`. `DEFAULT`
  remains direct-only. An enabled policy carries non-negative `copyFixedCostUnits`,
  `copyCostUnitsPerElement`, `directKernelCostUnitsPerElement`, and
  `contiguousKernelCostUnitsPerElement`; positive `expectedRunCount`; non-negative
  `maximumAdditionalBytes` and `minimumNetBenefitCostUnits`; and a
  `minimumBenefitBasisPoints` value from 0 through 10,000. Cost units are dimensionless comparable
  evidence supplied on the cold analysis path; they are neither measured in prepare nor presented
  as nanoseconds.
- Derive reuse/fan-out from the lowered unit rather than accepting it as a caller assertion. The
  current topology has one use for each materialized input. The selected formula for input `i` is:

  ```text
  copyCost(i) = copyFixedCostUnits + elementCount * copyCostUnitsPerElement
  directKernelCost(i) = elementCount * directKernelCostUnitsPerElement
  contiguousKernelCost(i) = elementCount * contiguousKernelCostUnitsPerElement
  direct = expectedRunCount * useCount(i) * directKernelCost(i)
  copied = expectedRunCount * (copyCost(i) + useCount(i) * contiguousKernelCost(i))
  netBenefit = direct - copied
  benefitBasisPoints = floor(10_000 * netBenefit / direct)
  ```

  All additions and multiplications use checked `long` arithmetic. A zero direct estimate cannot
  select a copy. Selection requires positive net benefit, the configured absolute minimum, the
  configured basis-point minimum, and the memory limit.
- Enumerate candidates in deterministic order: direct, materialize `a`, materialize `b`, then
  materialize `c`. Select the lowest estimated total cost; equal costs select the earlier
  candidate, so direct wins every tie. Admit at most one materialized read boundary. Never
  materialize the output, a scalar/all-zero input, an already dense-linear input, a value with no
  consumer, or a value whose selected consumer route cannot use canonical contiguous FLOAT64
  segment access.
- Treat `GENERAL_ODOMETER`, `LAST_AXIS_BIAS`, and `BLOCK_OUTER` read inputs as eligible only when
  replacing that one access by canonical dense access preserves the exact operation semantics,
  referenced span, alias decision, and selected portable scalar/vector behavior. A selected copy
  may make an otherwise scalar `GENERAL_ODOMETER` consumer vector-eligible, but route and strategy
  selection occurs once over each complete candidate; it is not changed after assignment.
- Introduce `CpuMaterializationPlan` as the route-independent immutable selected copy fact. It
  records the source boundary index and `ValueId`, source access binding, canonical contiguous
  consumer binding, checked element/byte count, fixed analysis-local workspace requirement ID
  `0`, alignment `Double.BYTES`, derived use count, expected runs, direct/copy/contiguous costs,
  net benefit, and selection reason. An empty optional means direct access.
- When a copy is selected, append exactly one
  `PreparationResourceRequirement.Workspace(0, elementCount * Double.BYTES, Double.BYTES)` after
  the four existing buffer declarations. Return it in the same analysis result before shared
  assignment. Keep the original four graph-value buffer declarations and `LogicalMemoryPlan`
  unchanged; the workspace is CPU-private scratch and has no `ValueId`.
- Retain the selected materialization plan, workspace declaration, adjusted generated-consumer
  access bindings, route/strategy, and specialization opaquely in
  `CpuPartitionPreparationPlan`. Shared Prepare assigns the workspace without interpreting the
  CPU plan.
- Add `CpuContiguousWorkspace`, a run-owned aligned shared-arena FLOAT64
  `WorkspaceRepresentation`. It exposes one exact writable native `MemorySegment` only through
  CPU-private cold binding, owns and closes its arena once, supports zero bytes, and is accessible
  to the selected worker group.
- During finalization, resolve and validate every four-buffer assignment and the optional exact
  workspace assignment before the first artifact-store operation. A direct plan rejects an
  unexpected workspace; a copied plan requires exactly the declaration identity, assigned slot,
  byte size, alignment, and workspace selection fixed during analysis.
- During cold binding, validate the source buffer, workspace liveness/accessibility/size/alignment,
  and adjusted generated carrier pattern. The generated consumer sees the selected workspace as
  one `MEMORY_SEGMENT` argument while the original source remains a read-only executable buffer
  selection for the copy. No generated method gains a fifth boundary argument.
- Execute the selected copy exactly once per bound invocation on the invoking thread before any
  consumer chunk starts. Copy logical elements in canonical order from the already-normalized
  source binding into dense workspace elements `[0, elementCount)`, using checked primitive
  address/carry arithmetic and no cursor allocation, reflection, map lookup, division, or modulo
  per element. Zero elements touch neither source nor workspace. Copy failure prevents consumer
  execution; Runtime's existing executable failure behavior leaves the output invalid.
- Preserve parallel execution ownership: the copy completes before worker submission, the
  writable workspace is visible to every selected worker, workers only read it, and existing
  deterministic chunk/failure/interrupt/close behavior remains unchanged.
- Add `CpuSpecializationBudget` in the cache package. Its current hard limits are four complete
  candidate plans per analysis, one realized artifact per analysis, zero fixed-shape variants,
  and zero unrolled variants. The direct candidate plus three possible one-input copy candidates
  exactly fits the candidate limit. Exceeding any limit fails analysis before declaration or
  artifact work; it never silently emits an extra class.
- Keep these code-shaping structural facts in `CpuKernelSpecialization` compatibility bytes and
  structural identity: current schema, canonical adjusted `CpuKernelIr`, exact/default numerical
  mode, scalar versus vector compute, exact vector species for vector compute, ordered generated
  carrier pattern, and direct versus selected materialization source position. The selected copy
  changes the canonical consumer access structure and therefore the lowering fingerprint.
- Keep these instance facts out of specialization and generated-class identity: concrete
  compatible extents, element count, layout offsets and stride magnitudes within the same
  structure, source and workspace objects, byte addresses, `ValueId`, declaration and slot
  identities, cost values, expected runs, memory limit, worker and chunk configuration, artifact
  root, model/graph/run identity, and benchmark measurements.
- Do not add fixed-shape or unrolled code in this task. A later task may raise either zero budget
  only with a reproducible benchmark showing a benefit for an exact eligible workload class and
  must add the baked shape/unroll fact to schema, specialization, verification, and invalidation.
- Harden `CpuGeneratedKernelArtifactStore` to one bounded current-schema envelope per key. One
  realization performs at most one process-local weak-intern lookup and, when a trusted root is
  present, one envelope lookup. Reject an envelope larger than 2 MiB, metadata larger than 64 KiB,
  class bytes larger than 1 MiB, wrong schema/key/metadata/length/checksum, trailing bytes,
  malformed class shape, or incompatible entry descriptor before definition.
- Publish a miss as one complete envelope through a forced temporary file and atomic move under
  the explicit normalized trusted root; if atomic move is unavailable or any read/write/security
  operation fails, use the verified in-memory artifact and leave persistence non-critical. There
  is one current schema and no migration reader, converter, expiry, eviction, background service,
  or attacker-authentication claim.
- Keep the default finalizer constructor and `CpuPartitionAnalysisInputs.DEFAULT` persistence-free.
  A present explicit trusted root enables ordinary lookup/store behavior. Even an
  `ENABLE_ELIGIBLE` evidence verdict does not invent a default path or public policy in 0005D;
  changing composition defaults requires a later explicitly scoped task.
- Add the opt-in `CpuGeneratedKernelPersistenceEvidenceTest`. Ordinary
  `./gradlew :backends:cpu:test` must skip its measurement method through an environment/property
  assumption. The suite is run explicitly only with
  `SYNAPTIK_CPU_PERSISTENCE_EVIDENCE=1` and a focused `--tests` selection.
- Make the evidence controller fork seven fresh JVMs per mode for each of six fixed current
  specializations: scalar and preferred-species vector crossed with all-segment, all-heap, and one
  fixed mixed carrier pattern. Each fork performs 20 warm-up realizations and 50 measured
  realizations, alternates no-root/hit order, clears process-local weak interning before every
  measurement, pre-seeds and verifies the trusted-root envelope outside the timed hit sample, and
  uses identical canonical IR and specialization inputs for both modes.
- Measure elapsed nanoseconds for complete no-root emit + structural verification + definition and
  complete trusted-root envelope read + compatibility/integrity/class-shape verification +
  definition. Record per-fixture sample count, median, p95, median absolute saving, ratio, hit and
  fallback counts, generated class/envelope byte sizes, Java vendor/version, OS/architecture,
  available processors, command, schema, thresholds, and SHA-256 of the canonical report payload
  excluding the hash field itself.
- Write the untracked verdict artifact to
  `backends/cpu/build/reports/evidence/cpu-0005d-persistence.json`. The verdict is
  `ENABLE_ELIGIBLE` only when every fixture has zero hit fallbacks, hit median at most 80% of
  no-root median, median absolute saving at least 200,000 ns, and hit p95 at most 90% of no-root
  p95. Otherwise it is `KEEP_DISABLED`; insufficient samples or an execution/environment failure
  is `INCONCLUSIVE`, which also keeps the default disabled. The test validates methodology and
  report production but does not fail merely because the performance verdict is
  `KEEP_DISABLED`.
- Run that evidence suite during 0005D implementation and record its exact report hash, metrics,
  environment, and verdict in this task. Rerun it only after a material generator, generator
  schema, JDK, persistence-policy change, or deliberate performance evaluation. Never invoke it
  for a model, from ordinary prepare/runtime, or as part of the final ordinary CPU suite.
- Keep generated class-byte persistence distinct from JVM just-in-time (JIT) machine code and
  profiling: a hit reuses verified class bytes only and still defines a fresh hidden class that
  the JVM may compile independently. Keep both distinct from the future workload tuning cache,
  which records route/configuration decisions rather than executable class bytes.
- Finalize affected Javadocs, package summaries, the CPU backend guide, glossary, and synchronized
  planning records in a separate clean documentation-focused context after executable Java and
  the evidence report stabilize.

## Out of scope

- Any data type, operation, topology, numerical policy, or capability beyond the completed fully
  static resolved-layout FLOAT64 `ADD -> exact GELU -> MUL` proving partition.
- More than one materialized input, output materialization, cross-partition materialization,
  packing, reorder, opaque/prepacked layouts, transfer, pooling, workspace reuse/aliasing, or
  mutation of the Model graph or `LogicalMemoryPlan`.
- Vector gather, scatter, masked tails, new Vector species search, tiles, fixed shapes, unrolling,
  broader pointwise coverage, reductions, scans, native/vendor routes, or CPU 0005E implementation.
- Per-model/workload measurement, autotuning, tuning-cache lookup or mutation, candidate search in
  ordinary prepare, or benchmark-driven Runtime mutation. Those belong to future tools/tuning.
- A public Config contract, CPU facade, persistence root discovery, default cache directory,
  Engine/Runtime orchestration, global registry, service locator, or process-wide mutable policy.
- Changes to Model, Compiler, Planning, shared Prepare, Runtime, Config, Backend Contract, Trace,
  Engine, OpenBLAS provider, another backend, module dependencies, Gradle configuration,
  `ARCHITECTURE.md`, architecture explanations, ADRs, architecture tests, backend-conformance
  tests, or integration tests.
- Creating a detailed CPU 0005E or later specification or changing completed CPU 0005A–0005C
  history.
- Claiming that persistence preserves JIT machine code/profile, is always faster, is required for
  correctness, authenticates hostile bytes, or is the workload tuning cache.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Prepare,
  Runtime, concrete backends, performance evidence, and CPU routes.
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md).
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md).
- [ADR 0010: staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md).
- [ADR 0008: performance evidence and autotuning boundaries](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md).
- [Planning guide](../../../planning-guide.md).
- [CPU backend master plan](../master-plan.md).
- [Completed CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md).
- [Completed CPU 0005B](0005b-universal-access-plans-and-right-aligned-broadcasting.md).
- [Completed CPU 0005C](0005c-vector-and-parallel-portable-strategies.md).

## Architecture constraints

- Planning continues to choose only CPU ownership. CPU analysis owns direct/materialized candidate
  formation, route/strategy selection, specialization, and exact resource declarations.
- Shared Prepare receives one opaque selected CPU plan and backend-neutral buffer/workspace
  declarations. It assigns slots but does not inspect CPU access, copy, cost, route, carrier,
  specialization, persistence, or benchmark facts.
- Backend analysis is deterministic from explicit context and CPU inputs. It performs no timing,
  persistence I/O, artifact realization, allocation, or Runtime work.
- Finalization occurs only after every declaration is assigned. It verifies but does not change
  the selected copy, route, strategy, specialization, or resource set.
- Runtime receives one immutable executable recipe. Per-run binding receives direct buffer and
  optional workspace references; execution performs only the prepared copy and prepared generated
  kernel work, with no graph inspection, selection, lookup, persistence, or benchmark comparison.
- Benchmarking is observational development evidence and does not mutate production settings.
  Future per-model autotuning remains a separate explicit workflow and cache.
- Any required shared-module, lifecycle, dependency, build, or architecture change is a stop
  condition rather than implied scope.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — typed cold inputs, deterministic
  comparison, exact declarations, selected plan retention, and post-assignment verification.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — copy eligibility, reuse/fan-out,
  adjusted canonical access facts, and the selected `CpuMaterializationPlan`.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — unchanged sole normalized access family
  and canonical structural identity.
- `io.github.pho001.synaptik.backend.cpu.internal.memory` — run-owned contiguous workspace and
  direct source/workspace carrier access.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — bounded specialization identity,
  current-schema envelope verification, optional persistence, and development evidence seam.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — already-selected generated
  consumer realization facts.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold copy/workspace binding and
  copy-before-consumer execution.

Packages added or changed:

- No package is added and no responsibility moves between packages.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan` — new immutable
  route-independent selected-copy fact because lowering owns semantic access replacement and
  reuse/fan-out accounting.
- `io.github.pho001.synaptik.backend.cpu.internal.cache.CpuSpecializationBudget` — new immutable
  code-generation ceiling because cache/specialization owns class-explosion control.
- `io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace` — new run-owned
  CPU workspace because concrete backends own physical workspace implementations.
- `CpuPartitionAnalysisInputs.MaterializationPolicy` — nested CPU-private cold policy because it
  is consumed only by the current analysis input and does not justify public Config.
- Existing `CpuPartitionPreparationPlan` retains the optional materialization, exact declarations,
  adjusted bindings, and budget result through shared assignment.
- Existing `CpuPreparedExecutable` performs the already-selected copy and generated invocation; no
  second executable, public transfer facade, or Runtime operation kind is added.

## Affected files

Expected production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMaterializationPlan.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAccessPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStore.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuSpecializationBudget.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuContiguousWorkspace.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`

Expected test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMaterializationPlanTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelPersistenceEvidenceTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuContiguousWorkspaceTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFusedGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuShapePolymorphicArtifactTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Local scope correction: `CpuInternalPackageInventoryTest` replaces `CpuKernelIrTest` in the
12-test allocation. The three required new production types necessarily change the exact internal
type inventory, while the cohesive implementation leaves the `CpuKernelIr` contract unchanged and
does not require a change to its existing focused test. This correction authorizes the required
inventory update without increasing the 38-path ceiling and does not report implementation
completion.

Expected explanatory and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0005d-materialization-specialization-and-persistence-evidence-gate.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

The evidence report is generated under `backends/cpu/build/` and is not committed. No other path
is authorized by default.

## Maximum scope

This task may create or modify at most 38 repository paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| CPU production | 21 | Eighteen existing production/package paths plus three new types |
| CPU tests | 12 | Nine existing tests plus three new tests |
| Explanatory documentation | 2 | CPU backend guide and glossary |
| Planning/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **38** | **21 + 12 + 2 + 3** |

The untracked evidence JSON does not count as a repository path. If implementation needs another
path, a second materialization, a shared contract, Gradle wiring, or a new package, stop and revise
this specification before coding.

## Acceptance criteria

- `CpuPartitionAnalysisInputs.DEFAULT` remains direct-only, scalar/single-thread, four-segment,
  manifest-disabled, and persistence-free. Existing 0005A–0005C behavior remains compatible.
- The candidate set is deterministically bounded to direct plus at most the three single-input
  copies. Direct wins ties; exactly one complete candidate is selected and one artifact is
  realized after assignment.
- Focused tables prove every hard filter and cost term: copy cost, generated-kernel benefit,
  derived reuse/fan-out, expected repeated runs, additional memory, and route eligibility.
  Checked overflow fails before declarations or artifact access.
- Direct plans retain exactly four buffer declarations and no workspace. A selected copy retains
  those four declarations and appends exactly workspace ID `0` with checked dense byte geometry
  before assignment. The Model graph, `LogicalMemoryPlan`, boundary `ValueId` values, and virtual
  intermediates are unchanged.
- Finalization resolves all buffer/workspace assignments before one bounded artifact lookup. It
  cannot add, remove, or change the selected materialization, route, strategy, specialization, or
  resource geometry.
- Generated/reference differential tests cover direct and each eligible source position across
  general odometer, block/outer, and last-axis cases, zero elements, arbitrary ranges, all four
  strategies where eligible, heap/segment/mixed original carriers, and exact numerical/special-
  value behavior.
- A selected copy happens once before consumer work. Zero ranges touch nothing; copy failure
  prevents consumer work; parallel workers read only the completed contiguous workspace; output
  validity and worker failure semantics remain unchanged.
- The generated entry still has exactly four direct arguments. Its selected copied position is a
  segment carrier and dense access. No fifth argument, hot carrier/access/route switch, cursor,
  per-element division/modulo, reflection, lookup, or allocation is introduced.
- `CpuSpecializationBudget` enforces four candidate plans, one realized artifact, zero fixed-shape
  variants, and zero unrolled variants. Shape, extent, offset, stride magnitude, costs, expected
  runs, memory budget, resources, workers, roots, and benchmark measurements remain outside class
  identity. Every actual code-shaping fact named in Scope remains inside identity.
- The current-schema persistence envelope is size-bounded, complete, checksummed, structurally
  verified, atomically published, and safely ignored on absence, incompatibility, corruption,
  truncation, trailing data, malformed class, or I/O/security failure. No invalid bytes are
  defined and no legacy schema is read.
- Explicit-root ordinary finalization performs no more than one weak-intern lookup and one
  envelope lookup, then load-on-hit or generate/optionally-store-on-miss. No timing, comparison,
  tuning, model-specific search, or Runtime persistence work exists.
- Ordinary `./gradlew :backends:cpu:test` skips the performance measurement. The explicit evidence
  command runs the fixed six-fixture, seven-fork methodology, creates the exact JSON report, and
  records an internally consistent `ENABLE_ELIGIBLE`, `KEEP_DISABLED`, or `INCONCLUSIVE` verdict.
- The default remains disabled unless the recorded report satisfies every exact threshold. A pass
  is only evidence eligibility; 0005D still creates no default path or public/composition policy.
- Documentation and Javadoc clearly distinguish generated class bytes from JIT machine code and
  profile, and both from the future workload tuning cache. They state the benchmark rerun triggers
  and that ordinary prepare/runtime never benchmark.
- No Java, test, build, architecture, dependency, shared-module, conformance, integration,
  provider, native/vendor, or later-task change occurs outside the exact path map.
- A separate clean documentation-focused pass finalizes affected Javadocs, package summaries, CPU
  guide, glossary, task evidence, master plan, and roadmap without repeating the final CPU test
  suite or evidence benchmark unless executable Java or the benchmark harness changed afterward.
- CPU 0005A–0005C remain `Complete`; CPU 0005D becomes `Complete` only after every gate passes;
  CPU 0005E and later remain `Draft` without detailed specifications.

## Tests / validation

Run focused tests while implementing:

```bash
./gradlew :backends:cpu:test --tests '*CpuMaterializationPlanTest' --tests '*CpuPartitionPreparerTest'
./gradlew :backends:cpu:test --tests '*CpuPartitionFinalizerTest' --tests '*CpuPreparedExecutableTest'
./gradlew :backends:cpu:test --tests '*CpuGeneratedKernelArtifactStoreTest' --tests '*CpuKernelSpecializationTest'
```

After executable Java stabilizes, run exactly one final ordinary CPU suite:

```bash
./gradlew :backends:cpu:test
```

The ordinary suite must report the evidence measurement as skipped and cover materialization
selection, declaration/finalization order, copy execution/failure, specialization budgets and
identity, persistence correctness, and all preserved CPU behavior.

Then run the development evidence suite once, separately from ordinary validation:

```bash
SYNAPTIK_CPU_PERSISTENCE_EVIDENCE=1 ./gradlew :backends:cpu:test --rerun-tasks --tests '*CpuGeneratedKernelPersistenceEvidenceTest'
```

Record the exact environment, per-fixture metrics, report path/hash, and verdict. This command is
not a final module validation command and is not rerun by the documentation context unless the
harness, generator/schema/JDK/persistence policy changes or a deliberate performance evaluation
is requested.

The clean documentation-focused context reuses both sets of executable evidence unless its edits
make either stale, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also records exact commands/results for:

- all local Markdown targets and explicit anchors in the five authorized Markdown files;
- balanced fences, final newlines, and trailing whitespace;
- the exact 21-production/12-test/five-Markdown inventory and 38-path ceiling;
- no path outside `backends/cpu` except the five authorized Markdown files, and no Gradle/shared/
  architecture/conformance/integration/native/vendor/later-spec change;
- direct/materialized declaration order, exact workspace ID/geometry, assignment-before-artifact
  order, and one-artifact budget;
- structural-versus-instance identity membership and zero shape/unroll budgets;
- bounded persistence envelope sizes/lookups, corruption fallbacks, single current schema, and no
  benchmark call from production prepare/runtime;
- ordinary-suite evidence skip, explicit-suite methodology/report/verdict, and default-policy
  coherence;
- bytecode-versus-JIT-versus-workload-cache terminology; and
- task/master/roadmap status synchronization with 0005D as the sole detailed frontier.

Repository-wide validation is deferred to the portable generated-coverage closure checkpoint and
CI. Architecture, backend-conformance, integration, and other-module suites are not run because
this task changes only CPU-private behavior and documentation, with no dependency, shared
contract, public end-to-end, or conformance claim.

## Dependencies

- [CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md) is `Complete`.
- [CPU 0005B](0005b-universal-access-plans-and-right-aligned-broadcasting.md) is `Complete`.
- [CPU 0005C](0005c-vector-and-parallel-portable-strategies.md) is `Complete`.
- Current shared Prepare staged analysis/declaration/assignment/finalization and workspace
  requirement contracts.
- Current Runtime prepared workspace selection, per-run workspace ownership, cold binding,
  execution failure, output validity, and cleanup contracts.
- Current CPU normalized access, whole-partition lowering, scalar/reference/vector generation,
  worker orchestration, specialization/schema, and optional trusted-root artifact store.
- Java 26 Class-File and Vector API toolchain already configured for the CPU module.

## Follow-up tasks

- CPU 0005E remains Draft for broader exact portable pointwise types, carriers, comparisons,
  selection, casts, and semantic families over the completed access/materialization architecture.
- CPU 0005F and 0006–0017 remain Draft in existing master-plan order.
- A later explicit specialization task may consider fixed shapes or unrolling only after evidence
  crosses a separately recorded workload-specific threshold and updates every identity/schema
  boundary.
- Future tools/tuning owns per-model/workload autotuning and its distinct persistent workload
  tuning cache. It may consume compatible candidate evidence before prepare; it does not turn this
  development persistence benchmark into ordinary model work.

## Architecture impact

Expected impact: None.

This task implements CPU-private lowering, exact resource declaration, specialization, optional
class-byte persistence, and observational development evidence already permitted by the current
architecture. If implementation requires shared Prepare to interpret CPU facts, Runtime or Engine
to select/copy/benchmark, a dependency/build change, or another architecture rule, stop and report
the conflict.

## Implementation prompt

Use this prompt in a separate clean coding context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, completed CPU tasks 0005A/0005B/0005C, and this CPU
task 0005D in full. Read the focused Runtime/Prepare/backend and performance-evidence architecture
docs/ADRs, documentation rules/profiles, and directly relevant final CPU/Prepare/Runtime source,
tests, and Javadocs.

Implement CPU 0005D exactly as specified within its exact 38-path ceiling. Keep shared Prepare
CPU-blind and the Model graph unchanged. Never run the persistence benchmark from ordinary
prepare/runtime or for each model; run its explicit development evidence command once and record
the report. Preserve 0005A-C history, leave 0005E+ Draft without detailed specs, and do not make
shared-module, build, architecture, public-facade, native/vendor, tuning, later-task, commit, or
push changes. Stop on an architecture or scope conflict.

After executable Java, the single final CPU suite, and the explicit evidence report stabilize,
hand the diff and exact evidence to a separate clean documentation-focused context. That pass
must finalize affected Javadocs/package summaries, CPU guide, glossary, and synchronized planning
records, run CPU Javadoc and documentation validation, and update this task's evidence, notes,
summary, and status. It reuses successful Java and benchmark evidence unless its changes make that
evidence stale.
```

## Local decisions

- Direct plus at most one of three input copies is the complete current candidate set; direct wins
  ties and one artifact is realized.
- Materialization cost evidence is explicit dimensionless cold input. Ordinary analysis computes
  but never measures it; future tuning remains separate.
- A selected copy uses one analysis-local workspace ID `0`, native contiguous FLOAT64 storage,
  and copy-before-consumer execution. It does not create a graph value or Runtime operation kind.
- Fixed-shape and unrolled specialization budgets are exactly zero in 0005D.
- Persistence evidence uses six fixed current specializations, seven fresh JVM forks per mode, and
  the exact all-fixtures threshold. Failure or noise keeps the default disabled.
- Explicit-root ordinary persistence is bounded verified class-byte reuse. It preserves neither
  JIT machine code/profile nor route-selection evidence.

## Known limitations

- Materialization remains limited to at most one read input of the exact current FLOAT64 proving
  unit and one CPU-private native workspace. There is no cross-unit, cross-partition, output,
  packing, transfer, pooling, or broader-family materialization.
- The cost model uses explicit comparable units and expected runs; it is not a production
  autotuner or a claim that one fixed policy is optimal for every workload.
- Engine composition remains absent, so no public persistence root, worker, workspace-creator, or
  materialization policy surface is added.
- Persistence is optional and correctness-independent. Its evidence verdict is environment- and
  JDK-specific and must be rerun after the recorded triggers.
- Backend-conformance, integration, native/vendor, broader data type/operation, dynamic shape,
  relaxed math, fixed-shape, unroll, and workload-tuning claims remain deferred.

## Validation evidence

Implementation context supplied the final executable evidence. No executable Java, test, or
evidence-harness code changed afterward; clean documentation context `/root/cpu_0005d_docs`
therefore reused it as required:

- The focused combined matrix passed 11 suites and 40 tests with zero skips, failures, or errors:
  `CpuInternalPackageInventoryTest`, `CpuMaterializationPlanTest`,
  `CpuContiguousWorkspaceTest`, `CpuPartitionPreparerTest`, `CpuPartitionFinalizerTest`,
  `CpuPreparedExecutableTest`, `CpuGeneratedKernelArtifactStoreTest`,
  `CpuKernelSpecializationTest`, `CpuFusedGeneratedKernelTest`,
  `CpuShapePolymorphicArtifactTest`, and `CpuReferenceDifferentialTest`.
- `./gradlew :backends:cpu:test --tests '*CpuMaterializationPlanTest' --tests
  '*CpuPartitionPreparerTest'` passed 8 tests.
- `./gradlew :backends:cpu:test --tests '*CpuPartitionFinalizerTest' --tests
  '*CpuPreparedExecutableTest'` passed 13 tests.
- `./gradlew :backends:cpu:test --tests '*CpuGeneratedKernelArtifactStoreTest' --tests
  '*CpuKernelSpecializationTest'` passed 7 tests.
- The sole final ordinary `./gradlew :backends:cpu:test` passed 21 suites and 62 tests with one
  skipped opt-in evidence measurement and zero failures or errors; `BUILD SUCCESSFUL` completed in
  1 second with 21 actionable tasks (1 executed, 20 up-to-date).

The exact environment-prefixed evidence launch was first blocked before Gradle by sandbox access
to the wrapper lock, and its approval channel failed. The same command then ran through the
pre-approved narrow `/tmp/synaptik-clean-codex-exec` wrapper. Its first actual Gradle run failed
before any fork, sample, or report because the harness assumed repository-root working directory
and could not find its scratch directory. The harness was corrected to locate `backends/cpu` from
either working directory. No successful evidence run was repeated.

The successful explicit command represented in the report was:

```bash
SYNAPTIK_CPU_PERSISTENCE_EVIDENCE=1 ./gradlew :backends:cpu:test --rerun-tasks --tests '*CpuGeneratedKernelPersistenceEvidenceTest'
```

It passed 1 test with zero skips, failures, or errors; `BUILD SUCCESSFUL` completed in 20 seconds
with 21 executed tasks. The untracked report is
`backends/cpu/build/reports/evidence/cpu-0005d-persistence.json`. It records Oracle Corporation
Java 26.0.1, Mac OS X 26.5.2, aarch64, 16 processors, schema 4, six fixtures, seven fresh JVM forks
per mode, 20 warmups and 50 samples per fork, and 350 samples per mode/fixture. The canonical
SHA-256 excluding `reportHash` is
`0977061bba616421a23f69a7819ba0a85de9af072956d98c21a934af13ba6453`.

| Fixture | No-root / hit median (ns) | No-root / hit p95 (ns) | Saving (ns) | Ratio | Hits / fallback | Class / envelope bytes |
|---|---:|---:|---:|---:|---:|---:|
| scalar-all-segment | 583042 / 491625 | 951709 / 825042 | 91417 | 0.843207 | 350 / 0 | 1237 / 1521 |
| scalar-all-heap | 519708 / 388417 | 812500 / 534041 | 131291 | 0.747375 | 350 / 0 | 557 / 833 |
| scalar-mixed | 586292 / 496292 | 1062333 / 748625 | 90000 | 0.846493 | 350 / 0 | 1141 / 1421 |
| vector-all-segment | 689417 / 538917 | 1260750 / 879500 | 150500 | 0.781700 | 350 / 0 | 2017 / 2303 |
| vector-all-heap | 611625 / 528000 | 1098375 / 823500 | 83625 | 0.863274 | 350 / 0 | 1195 / 1473 |
| vector-mixed | 665083 / 563958 | 1227000 / 942334 | 101125 | 0.847951 | 350 / 0 | 2010 / 2292 |

The verdict is `KEEP_DISABLED`: every fixture had zero fallback, but every fixture missed the
required 200,000 ns median saving and several missed the 0.80 median-ratio threshold. The default
remains persistence-free; an explicit root only enables bounded ordinary lookup/store behavior.

Clean documentation context `/root/cpu_0005d_docs` independently reviewed the complete diff,
final source/tests, generated evidence JSON, architecture/planning contracts, completed CPU
0005A–0005C, and the General, API/Javadoc, Backend Guide, Planning, and Example profiles. It
finalized all 21 permitted production/package Javadocs plus the CPU guide, glossary, and three
planning/status records without changing executable Java or tests. Its final documentation
commands and audit results were:

- The first `./gradlew :backends:cpu:javadoc` exposed misplaced constructor tags in a
  documentation-only edit and failed before completion. After correction, the final
  `./gradlew :backends:cpu:javadoc` passed with `BUILD SUCCESSFUL` in 1 second; 11 actionable tasks
  reported 2 executed and 9 up-to-date. Its five warnings were the two incubating-Vector warnings
  plus pre-existing default-constructor/unchanged scalar-emitter documentation warnings; there was
  no Javadoc error.
- A repository-local read-only Markdown validator over the five authorized files checked 697
  local targets and 290 explicit anchors with zero errors. The same final pass found balanced
  fences, final newlines, and no trailing whitespace.
- The exact-scope validator found 21 authorized production/package paths, 12 authorized tests,
  and five authorized Markdown paths: exactly 38 changed paths, with zero extra or missing paths
  and no shared/Gradle/architecture/conformance/integration/native/vendor/later-spec path.
- Twelve source-structure checks passed for declaration/workspace order and geometry,
  assignment-before-artifact order, one-artifact and zero shape/unroll budgets, structural identity
  membership, bounded envelope/corruption fallback, no production measurement, evidence skip and
  thresholds, and default-policy coherence.
- The evidence-report hash was recomputed from the canonical payload excluding `reportHash` and
  matched `0977061bba616421a23f69a7819ba0a85de9af072956d98c21a934af13ba6453`.
- Status checks confirmed CPU 0005A–0005D `Complete`, CPU 0005E and later `Draft`, and zero later
  detailed specifications.
- `git diff --check` passed after final documentation and status edits; an all-38-path final-
  newline/trailing-whitespace audit also reported zero errors.

## Implementation notes

CPU analysis now compares direct access with the three possible one-input copies in deterministic
order, derives use count from the lowered unit, and appends one exact workspace only for the
selected copy. Finalization resolves all declarations before one artifact realization. Cold
binding keeps the original source for the copy, substitutes the workspace only in the generated
carrier pattern, and executes one canonical-order copy before inline or parallel consumer work.

The direct and selected-copy generated structures have distinct compatibility identities; two
compatible extents still share the same selected-copy bytes and loaded identity. Concrete extents,
offsets, strides, assignments, carriers, resources, costs, workers, roots, and run identity remain
instance facts. The hard budget is four candidate plans, one realized artifact, zero fixed-shape
variants, and zero unrolled variants.

Optional persistence now uses one bounded current-schema envelope and safely falls back on every
absence, incompatibility, corruption, malformed class, size, I/O, security, or publication
failure. It is correctness-independent and disabled by default. Ordinary prepare and Runtime do
not benchmark. Persisted class bytes, JVM JIT machine code/profile, and the future workload tuning
cache remain three distinct artifacts/state roles.

No Model, Compiler, Planning, shared Prepare, Runtime, Config, Backend Contract, Trace, Engine,
OpenBLAS provider, Gradle, architecture, conformance, integration, native/vendor, or later-task
change was required. `CpuInternalPackageInventoryTest` is the authorized replacement for
`CpuKernelIrTest`; the existing kernel-IR implementation and focused contract needed no change.

## Completion summary

- Completed changes: implemented and documented bounded one-input contiguous materialization,
  copy-before-consumer workspace execution, explicit specialization budgets, adjusted structural
  identity, bounded optional current-schema class-byte persistence, and the opt-in evidence gate.
- Files changed or created: exactly the authorized 21 CPU production/package paths, 12 CPU test
  paths, and five Markdown paths; 38 total at the task ceiling, with no path outside the map.
- Tests and validation: reused the final 11-suite/40-test focused matrix, focused 8-, 13-, and
  7-test runs, sole final ordinary 21-suite/62-test CPU pass with one evidence skip, and sole
  successful 1-test explicit evidence report described above. CPU Javadoc, Markdown, exact-scope,
  structural/instance, declaration/order, persistence, status, terminology, and whitespace gates
  passed in `/root/cpu_0005d_docs`.
- Documentation-agent review: `/root/cpu_0005d_docs` independently finalized affected Javadocs,
  package summaries, CPU guide, glossary, task evidence, CPU master plan, and roadmap.
- Documentation impact: current materialization lifecycle, original/adjusted carrier patterns,
  budgets, persistence envelope, evidence verdict/rerun triggers, and class-byte/JIT/tuning-cache
  distinctions are explicit without changing architecture authority.
- Javadoc review: every permitted affected internal contract now documents inputs, results,
  geometry, identity membership, ownership, lifecycle, failure, and persistence boundaries.
- Glossary impact: CPU specialization, artifact store, preparation/executable, access plan, and
  materialization terminology now reflects completed 0005D behavior.
- Unresolved issues: None.
- Follow-up required: None. CPU 0005E and later remain Draft without detailed specifications.

Status: Complete
