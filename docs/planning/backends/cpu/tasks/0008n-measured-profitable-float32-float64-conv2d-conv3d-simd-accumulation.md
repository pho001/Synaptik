# Task 0008N: Measured profitable FLOAT32/FLOAT64 Conv2d/Conv3d SIMD accumulation

## Status

Incomplete (bounded scalar stop)

## Goal

Determine, implement only if proved profitable, and document one exact CPU-private SIMD form for
existing grouped Conv2d and Conv3d. The only candidate is **output-width lane parallelism**:
one preferred-species lane owns one consecutive output-width cell and retains that cell's current
increasing `input-channel -> kernel-depth (Conv3d) -> kernel-height -> kernel-width` accumulation
order. This is not horizontal vector reduction.

The planning-time source audit establishes semantic feasibility but not a reliable retained
measurement: current Conv emitters are scalar-only and their historical performance harnesses do
not exercise the proposed Vector API body or the required protocol. No benchmark number is
invented. Consequently this task is a bounded implementation-time spike with a conditional
implementation outcome, not a promise that SIMD becomes selectable.

## Scope

### Candidate and semantic boundary

The spike may select `VECTOR`/`PARALLEL_VECTOR` only when all facts hold:

- the portable IR is existing direct `CpuConv2dIr` or `CpuConv3dIr`, with no external Conv2d ADD
  or ADD-RELU epilogue, and all input, weight, optional intrinsic bias, and result types are the
  same `FLOAT32` or same `FLOAT64`;
- every boundary has a dense row-major resolved layout, non-negative cold storage offset, native
  byte order for segments, valid accessible span/writability/overlap proof, and direct array,
  `MemorySegment`, or ordered mixed carrier; no packing or materialization is admitted;
- preferred species has more than one lane; width stride and width dilation are both one; static
  output width contains at least one full species whose every lane is in bounds for every kernel
  width contribution; and ordinary output-cell parallelism selects its existing disjoint ranges;
- all geometry/address/range arithmetic is already representable by lowering and binding checks.

For each interior width block, initialize vector lanes from zero or the scalar intrinsic bias
broadcast. For each channel and kernel coordinate in the existing order, load the contiguous input
width vector, broadcast that one scalar weight, and do exactly `sum = sum + input * weight` per
lane. Store the contiguous output vector. Scalar prologue, padding-touching cells, tails, and any
worker-boundary fragment use the current scalar cell body in increasing output ordinal order.
This preserves each cell's FLOAT32/FLOAT64 multiply then left-associated add, NaN/infinity,
signed-zero, subnormal, overflow, and conceptual-positive-zero padding behavior. It uses neither
FMA, reassociation, widening, horizontal reduction, lane extraction, masked/gather/scatter access,
nor a per-lane callback.

NCHW/NCDHW output width is contiguous; input width is contiguous only under the stated width
geometry. Weight layouts `[Cout,Cin/group,Kh,Kw]` and `[Cout,Cin/group,Kd,Kh,Kw]` supply a scalar
weight broadcast shared by lanes. Batches, groups, channels, depth, height, arbitrary positive
non-width strides, and padding remain ordinary cold loop facts. Conv1d remains its exact virtual
singleton Conv2d composition; it gains no emitter, IR, capability, or separate evidence form.

### Conditional stop outcomes

Before production selection, create an untracked temporary spike/evidence root outside the
repository. If any of these conditions fails, do not add schema 63 or production vector code:

1. a clean Java 26 Vector API oracle cannot reproduce scalar raw-result semantics for the finite
   special-value and geometry rows;
2. arrays, native-order segments, or an ordered mixed-carrier representative cannot use direct
   preferred-species loads/stores with the same algorithm and no prohibited hot overhead;
3. every fixed eligibility-branch row fails to make the required measured gain over its current
   scalar generated kernel under the gate below; or
4. the necessary change escapes the listed owners, changes a shared/public contract, or exposes an
   accumulation-order ambiguity.

On a stop, retain immutable spike evidence, leave all Conv execution `SCALAR`/`PARALLEL_SCALAR`,
do not claim SIMD, and update this task/master-plan status with the exact failed condition and a
Draft follow-up only if it is independently actionable. If every gate passes, implement the one
form described here; scalar fallback remains mandatory for every ineligible occurrence and range.

## Out of scope

- BFLOAT16/FLOAT16, mixed promotion, Conv2d external epilogues, Conv3d suffixes, Conv1d-specific
  code, non-dense/gather/scatter/masked-vector access, packing, unrolling, fixed-shape classes,
  autotuning, native routes, or benchmark-selected Runtime policy.
- Any reduction across kernel/channel lanes, partial sums, workspace, combine kernel, FMA, relaxed
  math, changed Model accumulation semantics, public API, Compiler/Training behavior, shared
  Prepare/Runtime contracts, architecture/ADR/dependency/build-toolchain change.

## Architecture references and constraints

- [Architecture contract](../../../../../ARCHITECTURE.md), especially CPU route ownership,
  staged preparation, immutable recipes/run isolation, and generated-code parity.
- [CPU master plan](../master-plan.md), [CPU backend guide](../../../../backend-guide/cpu-backend.md),
  [Conv2d foundation](0008-portable-grouped-nchw-conv2d-execution-foundation.md),
  [Conv dimensional closure](0008a-portable-channels-first-dimensional-convolution-closure.md),
  [Vector MSE](0008m-vector-mse-none.md), and [Planning Guide](../../../planning-guide.md).

CPU analysis alone selects eligibility, preferred species, strategy, schema, and carriers;
finalization realizes its immutable decision; shared Prepare and Runtime do not learn convolution
or vectors. Existing complete-output-cell range partitioning remains the only parallelism. Direct
generated entries retain typed carriers, `long[] geometry`, `long start`, and `long end`; no graph,
operation, layout, worker, cache, or dispatch lookup may occur in a hot loop.

The generated class must match an optimal clean Java specialized oracle in semantic algorithm,
hot-loop/dataflow shape, and avoidable-overhead profile. A final, field-free,
constructor-free generated class has direct typed entry only. Prepared executables remain reusable;
concurrent runs retain independent carriers, geometry, ranges, and `RunState`.

## Package impact

No package was added or moved. The following impact map was conditional on passing the admission
gate; none of these production or test changes remains after the bounded scalar stop:

- `...internal.ir`: existing Conv IR gains only a route-private output-width realization fact if it
  is required to distinguish scalar from schema-63 bytes; scalar IR encoding remains byte-stable.
- `...internal.prepare`: exact candidate predicate, range-safe strategy selection, and selective
  schema choice.
- `...internal.codegen.emit`: Conv2d/Conv3d direct vector body plus current scalar fallback;
  carrier/vector helpers only if their existing responsibility is sufficient.
- `...internal.cache`: schema-63 validation/compatibility/class-identity support.
- `...internal.codegen.emit` tests: package-private `CpuConvSimdEvidenceTest`,
  `CpuConvSimdPerformanceOracle`, and `CpuConvSimdPerformanceTest` are the sole new test types.

## Schema and cache identity decision

Schema 63 was conditional and was not retained. Had the implementation gate passed, it would have
denoted exactly `OUTPUT_WIDTH_LANE_PARALLEL_V1`: same-typed dense FLOAT32/FLOAT64 direct
Conv2d/Conv3d with width stride/dilation one, preferred species, and at least one statically proved
interior full-width block. The distinguishing code-shaping fact would have been that realization,
represented by schema 63 together with `Compute.VECTOR`; it would not have been merely an execution
preference. The existing canonical Conv family fingerprint would have supplied rank, types, bias,
stride/padding/dilation/groups, access plans, and ordered roles; carrier pattern and preferred-
species bits would have remained identity facts. Parallel reuse would have used the same artifact.
Extents, offsets, carrier objects/addresses, start/end, ranges, workers, graph, slots, and runs would
have remained cold facts.

Schema 62 Conv MSE and every scalar Conv class retain their current projections and bytes.
`CURRENT_VERSION` remains 62 because no schema-63 bytes exist. A future Conv vector realization
requires a newly justified schema decision; this stopped task does not reserve or establish schema
63 as a production compatibility identity.

## Affected files and maximum scope

The conditional implementation ceiling was at most 27 paths: eight production/Javadoc owners,
`backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/{cache/CpuGeneratorSchema.java,cache/CpuKernelSpecialization.java,prepare/CpuPartitionPreparer.java,codegen/emit/CpuClassFileKernelGenerator.java,codegen/emit/CpuConv2dEmitter.java,codegen/emit/CpuConv3dEmitter.java,ir/CpuConv2dIr.java,ir/CpuConv3dIr.java}`;
the following 16 test/evidence paths; and these three planning paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv2dEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv2dGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv2dPerformanceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv3dEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv3dGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv3dPerformanceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConvSimdEvidenceTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConvSimdPerformanceOracle.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConvSimdPerformanceTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuConv1dCompositionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuConv2dLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuConv3dLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`

Thus the conditional ceiling arithmetic is exactly
`8 production + 16 test/evidence + 3 planning = 27` paths. The bounded stop removed every transient
production and test change, so the retained repository diff is exactly the three planning paths.
Existing source sets would have run the new tests; no Gradle edit was authorized. No Model, shared,
public, guide, glossary, architecture, conformance, integration, or Gradle path changed.

## Exact finite validation matrices

Named rows must assert their stated count and uniqueness. They are representative bounded coverage,
not a claim to test every geometry combination.

### Semantic matrix: 48 rows

```text
2 ranks (Conv2d, Conv3d) * 2 types *
  (unit interior exact-width, unit interior tail, padded prologue/interior/epilogue,
   grouped, depthwise, batch>1, stride-height/depth with width=1, dilation-height/depth with width=1,
   scalar width-stride fallback, scalar width-dilation fallback,
   short-width fallback, non-dense fallback) = 48
```

Each rank/type has arrays, all-segment, and one ordered mixed-carrier representative distributed
across the twelve rows; positive offsets and valid non-zero subranges occur in each carrier form.
The stated count is `2 * 2 * 12 = 48` rows.
The matrix includes bias/no-bias, `VECTOR` and `PARALLEL_VECTOR`, range starts that bisect a width
row, zero work, input/output sentinel checks, aliases/rejections using the existing binding owner,
and finite ordinary plus `+0`, `-0`, infinities, NaN, overflow, and subnormal values. It compares
each lane/output raw bits to the scalar direct oracle where semantics define bits.

### Structural/Class-File matrix: 24 dossiers

```text
2 ranks * 2 types * 2 bias forms *
  (array/array/array + segment/segment/segment + array/segment/array) = 24
```

For a bias form, the bias carrier follows the row's declared array/segment form; the mixed row
uses the inverse form for bias. Thus the arithmetic is `2 * 2 * 2 * 3 = 24` dossiers.

For every dossier retain class bytes, SHA-256, deterministic second emission, `javap -c -v -p`,
descriptor/member inventory, normalized call-owner record, and scalar schema-51/52 controls.
Inspect typed Vector loads/stores, scalar weight broadcast, vector multiply then add, vector scalar
fallback cells/tails, and absence of reduction/FMA/mask/gather/scatter/temp array/per-lane access,
allocation, boxing, reflection, `invokedynamic`, collection/map/string dispatch, monitor,
method-handle, graph/layout/operation/cache/worker lookup, or Synaptik hot helper. Decompilation
and the compiled direct Java oracle must show the same outer cell/block, ordered accumulation, and
scalar fallback shape.

### Performance matrix: 16 branch rows, two comparisons each

| Per rank/type rows | Carrier | Geometry | Strategy/range |
|---|---|---|---|
| 1 | all arrays | interior exact width | VECTOR |
| 2 | all segments | interior tail + positive offsets | PARALLEL_VECTOR |
| 3 | mixed input/weight/output | padding and scalar borders | VECTOR |
| 4 | all arrays | grouped/depthwise interior | PARALLEL_VECTOR |

```text
2 ranks * 2 types * 4 eligibility branches = 16 rows
16 rows * 5 independent fresh forks * 9 retained symmetric pairs * 2 comparisons = 1,440 pairs
1,440 pairs * 4 individual timed sides = 5,760 retained timings
16 rows * 5 forks * 2 comparisons = 160 fork medians
16 rows * 2 comparisons = 32 median-of-fork-medians aggregates
```

Use a shape-polymorphic workload with at least `8192 * preferredLanes` interior output-width cells
per timed row; the direct oracle has the exact generated carrier signature, cold geometry/ranges,
scalar borders and tails, and output consumption. It is compiled by `javac` before timing; no
fixture trip count, bridge, helper, fallback/reference call, allocation, boxing, or reflection is
allowed.

The 16 rows are the complete fixed set of selectable eligibility branches: rank/type crossed with
the array exact-interior, segment tail/offset, mixed padding/border, and grouped/depthwise branch.
The static predicate may not add another carrier, geometry, or strategy branch unless this matrix
and its arithmetic are expanded first. It remains shape-polymorphic within a covered branch; there
is no runtime benchmark selection.

Each fork measures three equal-work implementations: current scalar generated (`S`), candidate
vector generated (`V`), and the independently compiled direct Java vector oracle (`D`). Use Java
26, recorded CPU/OS/JDK facts, `-Xms1g -Xmx1g -XX:-TieredCompilation -Xbatch`, and deterministic
row/order seed. Run five warmup symmetric `V-S-S-V`/`S-V-V-S` pairs and five warmup symmetric
`V-D-D-V`/`D-V-V-D` pairs. After warmup, double one shared iteration count until the two-invocation
aggregate for **each** of `S`, `V`, and `D` is at least 50 ms. Retain exactly nine randomized
symmetric pairs for each comparison: `V-S-S-V` or `S-V-V-S` proves profitability, and
`V-D-D-V` or `D-V-V-D` proves generated/oracle parity. Every individual one of the 5,760 retained
timings must be at least 25 ms. No retry, discard, replacement, outlier filter, threshold change,
or asymmetric work is allowed.

For every retained pair, fork median, and aggregate, require `V/S <= 0.90x`: at least a 10% speedup
over the current scalar generated kernel. Ten percent is deliberately above ordinary timing noise
while remaining small enough for a portable Vector API increment; it is a fixed gate, not a tuning
input. Separately require `V/D <= 1.15x` for every retained pair, fork median, and aggregate.
Both gates must pass for all 16 branch rows before the corresponding predicate branches are
selectable; otherwise narrow the predicate to the passing complete branch set or take the bounded
scalar stop. Write immutable raw timings, calibration, checksums, source/class/oracle snapshots,
commands, environment, summaries, and complete SHA-256 manifest under one untracked evidence root.

## Acceptance criteria

1. Either all stated gates prove the exact output-width form and it is the only selectable SIMD
   Conv form, or the task records a bounded stop without production SIMD/schema change.
2. Any successful lane result is observationally equivalent to existing ordered scalar Conv2d or
   Conv3d; scalar borders, tails, invalid geometry, non-dense access, BFLOAT16, mixed types, and
   unsupported epilogues retain current scalar behavior.
3. Existing complete-output-cell range partitioning, validation, carrier accessibility, overlap
   rejection, cold offsets, subranges, worker failure/join behavior, and run isolation are unchanged.
4. The 48 semantic rows, 24 dossiers, and both the 1,440-pair/5,760-timing/160-median/
   32-aggregate performance gates pass before selection: every `V/S` is `<= 0.90x` and every
   `V/D` is `<= 1.15x`; the direct oracle and forbidden-overhead evidence pass.
5. Had the admission gate passed, any new schema identity would have been selective and scalar
   schema-51/52 and prior-family bytes would have remained unchanged. Because the spike stopped,
   no schema-63 bytes exist and schema 62 remains current.
6. A separate clean documentation-focused pass reviews final Javadocs/planning/glossary impact,
   records no-change conclusions for public/shared/architecture/docs not changed, and does not
   rerun successful executable tests unless it changes Java.

## Tests / validation

The implementation context records all actual commands and results. The conditional implementation
protocol was:

```bash
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv1dCompositionLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConv2dGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConv3dGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConvSimdEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConvSimdPerformanceTest -Dsynaptik.cpu.convSimd.performance=true -Dsynaptik.cpu.convSimd.performanceEvidenceRoot="$CPU_0008N_EVIDENCE_ROOT" --rerun-tasks
./gradlew :backends:cpu:test --rerun-tasks
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

Also validate Markdown links/anchors/fences, heading uniqueness, final newlines, whitespace, the
`2*2*2*3=24`, `2*2*4=16`, `16*5*9*2=1440`, `1440*4=5760`, `16*5*2=160`, and `16*2=32`
arithmetic; both fixed ratio gates; status/order; exact `8+16+3=27` path ceiling; empty index; and
no tracked spike evidence. Repository-wide, architecture, conformance, and integration validation
is deferred to CPU 0009/CI unless a forbidden shared boundary changes.

## Dependencies and follow-up tasks

- CPU 0008, 0008A, 0005C/0005I, 0007A0/0007A1A, 0008L, and 0008M are Complete.
- CPU 0008N1 is Draft and next. It owns the generated nested width-block/dataflow parity re-spike;
  no detailed task specification exists yet. CPU 0008O follows 0008N1 and owns stable-reduction
  numerical eligibility. CPU 0008P remains Draft and must not advance. CPU 0009 inventories
  successful or stopped evidence.

## Architecture impact

Expected impact: None. Stop if Model accumulation meaning, shared preparation/finalization,
Runtime, public API, dependency, resource, architecture contract, or test boundary must change.

## Implementation prompt

```text
You are the clean-context implementation agent for CPU task 0008N. Read AGENTS.md,
ARCHITECTURE.md, the Planning Guide, CPU master plan, this task, direct Conv2d/Conv3d/Conv1d,
carrier/vector/cache/preparer/finalizer/reference/test owners, and Model convolution semantics.
Implement exactly this bounded spike. Do not commit, stage, push, reset, or alter unrelated work.
Stop without SIMD if any semantic, Vector API, scope, or profitability gate fails; otherwise
implement only the specified output-width lane-parallel form and its matrices. Then hand the final
diff and executable evidence to a separate clean documentation-focused agent. Do not mark Complete
before that pass and all required evidence are recorded.
```

## Local decisions

- Output-width lane parallelism is the sole semantically permissible candidate because each lane
  owns a complete scalar accumulation; kernel/channel horizontal reduction is forbidden.
- At task-definition time, planning evidence was insufficient to pre-authorize schema 63. The
  completed implementation spike applied the fixed gate without tuning production behavior and
  stopped before admitting any production vector form or schema identity.

## Known limitations

Same-typed dense FLOAT32/FLOAT64 Conv2d/Conv3d were the only forms eligible for this stopped task;
they remain scalar. Conv1d composition, BFLOAT16, mixed carriers outside the three direct forms,
non-dense access, width stride/dilation other than one, and all other SIMD axes also remain scalar.
Any future vector admission is separate work beginning with CPU 0008N1's bounded re-spike.

## Validation evidence

Planning context: this clean documentation/planning task at commit
`15f83a2269480d45ca4ef45104f4d72850daafb4`.

- Read the required architecture, planning, documentation, CPU guide/glossary, prior CPU 0008G,
  0008G1, and 0008M tasks; current Conv emitters/IR/lowering/reference/tests, carrier/vector,
  schema/cache, preparer/finalizer, and Model Conv semantics.
- Confirmed scalar emitters traverse the required accumulation order and existing preparation
  explicitly excludes Conv from generic vector eligibility.
- At planning time no temporary vector benchmark existed; the implementation-time evidence below
  supersedes that pre-implementation observation. The task's bounded stop conditions prevented a
  one-row diagnostic from being represented as final matrix acceptance.

Implementation context, 2026-09-06, Java 26.0.1 on macOS arm64:

- Retained untracked evidence root:
  `/tmp/synaptik-cpu-0008n-spike-20260906a`.
- Compiled the direct oracle independently with
  `javac --add-modules jdk.incubator.vector -d
  /tmp/synaptik-cpu-0008n-spike-20260906a/oracle-classes
  /tmp/synaptik-cpu-0008n-spike-20260906a/src/DirectConvOracle.java`, and compiled the candidate
  separately into `candidate-classes`. The combined `SpikeHarness` command and complete output are
  retained in `results/checkpoint.txt`; it completed in `11.195285458 s`.
- The independently compiled source-level candidate and direct Vector API oracle passed raw-bit
  checks for representative FLOAT32/FLOAT64 Conv2d/Conv3d array rows. Its diagnostic medians were
  respectively `V/S = 0.269914616, 0.442504668, 0.488461170, 0.551605512` and
  `V/D = 1.007456389, 0.988221544, 0.990847646, 1.008247078`. This established technical source-
  level viability only; one Conv3d FLOAT32 timing was below the final 25 ms minimum.
- A transient schema-63 generated Conv2d/Conv3d implementation was then measured before admission.
  The exact command was
  `JAVA_TOOL_OPTIONS=-Dsynaptik.cpu.convSimd.performance=true ./gradlew :backends:cpu:test
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConvSimdPerformanceTest
  --rerun-tasks`; the failed test completed in `14.34 s`.
  The generated Conv2d FLOAT32 all-array exact-interior row used 64 iterations, twelve warmup
  triplets, and nine retained symmetric pairs. Every individual timing was at least
  `61,244,791 ns`. It passed generated vector/current generated scalar at median
  `V/S = 0.265215012`, but failed generated vector/direct clean Java at
  `V/D = 1.775769426`, above the fixed `1.15` maximum. The exact failed XML and transient compiled
  emitter/test snapshots are retained under `results/`; their recorded SHA-256 values are in
  `results/manifest.sha256`.
- The actual generated diagnostic was intentionally bounded and stopped at the first decisive
  failed admission gate. It is not represented as the unexecuted 16-row/five-fork final matrix.
  No ratio threshold was changed, no sample was retried or discarded, and no other branch is
  claimed to pass or fail.
- Removed the transient Conv selection, schema-63 compatibility, emitters, and test harness only
  through `apply_patch`. `CpuGeneratorSchema.CURRENT_VERSION` remains 62, existing Conv preparation
  remains `SCALAR`/`PARALLEL_SCALAR`, and no production or test Java path remains changed.
- After the stop,
  `./gradlew :backends:cpu:test --tests ...CpuConv1dCompositionLoweringTest --tests
  ...CpuConv2dLoweringTest --tests ...CpuConv3dLoweringTest --tests
  ...CpuConv2dGeneratedKernelTest --tests ...CpuConv3dGeneratedKernelTest --tests
  ...CpuPartitionPreparerTest --tests ...CpuPartitionFinalizerTest` passed 51 tests across the
  seven named classes. Gradle reused its compatible cache after the production sources returned
  byte-for-byte to their tracked state.
- `./gradlew :backends:cpu:javadoc`, `git diff --check`, and `git diff --cached --check` passed.
  `git status --short` showed only these three planning paths, with the task file untracked; no
  production, test, schema, build, architecture, conformance, integration, or evidence path is
  tracked. `shasum -a 256 -c results/manifest.sha256` passed all four retained manifest entries.
  Repository-wide validation remains deferred as prescribed because no executable or shared
  boundary change remains.

Documentation-focused review context, 2026-09-06, mandatory separate clean context:

- `ruby /tmp/validate_cpu_0008n_markdown.rb` passed for the task, CPU master plan, and roadmap. It
  resolved every repository-local link and heading anchor and confirmed unique headings, balanced
  fences, final newlines, no carriage returns, and no trailing whitespace.
- `shasum -a 256 -c manifest.sha256`, run from
  `/tmp/synaptik-cpu-0008n-spike-20260906a/results`, passed all four retained entries. An initial
  invocation from the repository root failed only because the manifest paths are relative to its
  own directory; rerunning from that directory resolved all four files and verified their hashes.
- The final static consistency command passed the `48`, `24`, `16`, `1,440`, `5,760`, `160`, `32`,
  and conditional `27` arithmetic; both fixed ratios and the `61,244,791 ns` minimum; CPU 0008N
  Incomplete, CPU 0008N1 Draft-next, and CPU 0008O-after-N1 order; current schema 62; scalar-only
  Conv preparation; empty index; and the exact three-path planning-only worktree scope.
- Final `git diff --check` and `git diff --cached --check` passed. Final `git status --short` showed
  only the CPU master plan, this untracked task, and the roadmap. No Java test or Javadoc command
  was rerun because this review changed no executable Java or Javadoc.

## Implementation notes

Inspection of the transient emitter snapshot shows that the attempted generated realization
decoded complete output coordinates for each width block and repeatedly performed long
geometry/address work inside that block path. That emitted shape differs from the optimal clean
Java nested-loop dataflow, which advances outer coordinates through enclosing loops and carries
width-local address state into the vector block. The measured generated body still beat the
existing scalar generated body, but it took `1.775769426x` the independently compiled direct vector
loop in the first actual-generated diagnostic row. The observed dataflow mismatch is the concrete
performance problem and a plausible explanation for the gap; this one diagnostic does not isolate
the causal share of each decode or address operation. Admitting the body would violate the
generated-code parity requirement, so the conditional stop fired before the full matrices and
before any production selection.

## Completion summary

- Completed changes: executed and retained the bounded source-level and actual-generated spike;
  rejected the generated realization at the fixed direct-oracle gate; restored scalar-only Conv
  production and schema 62.
- Files changed or created: this task, the CPU master plan, and the roadmap only. The evidence root
  is untracked and outside the repository.
- Tests and validation: source prototype raw-bit checks passed four representative rows; source
  diagnostic ratios are recorded above. The actual generated Conv2d FLOAT32 array diagnostic
  failed exactly `V/D = 1.775769426 > 1.15` after passing `V/S = 0.265215012 <= 0.90`; the final
  16-row/five-fork matrix was not run after this decisive stop. This review's Markdown/link/
  anchor/heading/fence/newline/whitespace, manifest, arithmetic, ratio, status/order, schema/route,
  exact-scope/index, and diff checks all passed.
- Documentation-agent review: this mandatory separate clean-context documentation-focused review
  read the required architecture, documentation, and planning contracts; inspected the complete
  retained diff and evidence; finalized exactly this task, the CPU master plan, and the roadmap;
  and did not rerun successful Java tests or Javadoc because no executable Java or Javadoc changed.
- Documentation impact: planning status, order, evidence, conditional schema language, and the
  bounded-stop explanation only. Public/shared API and guides remain unchanged because the
  transient implementation was removed and no user-visible contract or workflow changed.
- Javadoc review: no change; all transient Java edits were removed, so existing production and
  public/shared API Javadocs describe the retained scalar-only implementation.
- Glossary impact: no change; the stopped prototype introduced no retained project term or
  production contract.
- Architecture/ADR impact: no change; backend ownership, lifecycle, dependencies, and generated-
  code parity requirements are unchanged, so neither `ARCHITECTURE.md`, a focused architecture
  guide, nor an architecture decision record requires revision.
- Tests/build impact: no retained Java test or build change; the implementation context's successful
  focused Java and Javadoc evidence was reused. Architecture tests, backend conformance tests,
  integration tests, and repository-wide validation require no rerun because no executable,
  dependency, shared-boundary, or build behavior remains changed.
- Unresolved issues: the attempted generated width-block body does not match the optimal direct
  Java loop/dataflow overhead profile.
- Follow-up required: CPU 0008N1 must first design and prove a generated nested width-block loop
  that hoists coordinate/geometry work to match the direct Java oracle, then restart the complete
  fixed semantic, structural, and five-fork performance matrices without weakening either gate.

Status: Incomplete
Follow-up required: CPU 0008N1 generated width-block loop/dataflow parity re-spike.
