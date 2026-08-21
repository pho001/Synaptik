# Task 0007A1C: Generated/Direct Evidence Closure

## Status

Review needed

## Goal

Close the remaining bounded evidence gaps for every currently generatable portable CPU family by
mapping each operation/family/form to equivalent direct primitive Java, emitted/decompiled
Class-File shape, forbidden-reference checks, and realistic parity evidence.

This is evidence-first corrective work. It distinguishes semantic/structural coverage from
measured performance, does not promise universal parity across algorithms, resources, layouts,
JVMs, or machines, and permits production correction only for a demonstrated bounded defect.

## Current inventory and conclusion

The current generator has ten computation categories: pointwise; affine copy; movement (`PAD`,
`TILE`, `CONCAT`, `STACK`, `UNFOLD_AXIS`, `UNFOLD2D`, `SLICE_UPDATE`); indexing (`GATHER`,
`GATHER_ELEMENTS`, `GATHER_ND`, `ONE_HOT`); scatter (`SCATTER_ELEMENTS`, Gather-compatible
`SCATTER_ADD`, `SCATTER_ND`); fold (`FOLD_AXIS`, `FOLD2D`); ordering (`SORT`, `ARGSORT`, `TOP_K`);
explicit state (`INITIAL_STATE`, FLOAT64/FLOAT32 `DROPOUT`); `CUM_SUM`/`CUM_PROD`; and ordinary
`MIN`/`MAX`/`ALL`/`ANY`/`SUM`/`MEAN`/`PROD` aggregates.

| Category | Retained direct/decompilation evidence | Remaining gap |
|---|---|---|
| Pointwise | Dense scalar ADD, Vector ADD, fused ADD -> GELU_EXACT -> MUL; ten scalar activation, four vector activation, scalar-tail member-reference evidence; semantic tests cover all 48 opcodes/admitted types. | General-long, segment/mixed, non-floating vector, and virtual-mask performance are partial; all opcodes need an explicit code-shape mapping. |
| Affine/movement | FLOAT64 CONTIGUOUS, FLOAT32 TILE, INT32 SLICE_UPDATE direct gates; dense/general Class-File and semantic tests cover affine and all seven movement forms. | PAD, composition, windows, and general-long/segment movement are untimed. |
| Indexing | Dense FLOAT32 GATHER_ELEMENTS and BOOL ONE_HOT gates; Class-File tests cover all families, index widths, general layouts, and segment/mixed carriers. | GATHER, GATHER_ND, and general-long/segment performance are missing. |
| Scatter | Dense replacement/addition, all three mappings, one general mixed addition, and exact floating-product scratch gates; schema-31 tests span reductions/types/index widths. | Add one non-additive general-long reduction so closure does not rely only on FLOAT32 addition outside dense arrays. |
| Fold | Dense FLOAT32 FOLD_AXIS gate; structural/semantic tests cover both families, types, ranges, and mixed/general carriers. | FOLD2D and general-long/segment performance are missing. |
| Ordering | Dense FLOAT32 SORT and TOP_K gates; structural/semantic tests cover all families/types/directions/general carriers. | ARGSORT and general-long/segment performance are missing. |
| Random | Dense FLOAT64/FLOAT32 DROPOUT gates; decompilation covers heap/segment INITIAL_STATE and dense/segment/mixed DROPOUT. | General mixed DROPOUT is untimed; two-word INITIAL_STATE is unsuitable for throughput parity. |
| Scan | Dense FLOAT32 CUM_SUM plus BFLOAT16 CUM_SUM/CUM_PROD gates; tests cover five types and all modes. | A non-BFLOAT16 general segment reverse/exclusive scan is missing. |
| Extrema/Boolean aggregate | Dense full MIN/MAX for five numeric types and ALL/ANY gates; tests cover all forms/layouts/carriers. | General single-/multi-axis forms are untimed. |
| Numerical aggregate | Thirteen dense single-axis gates cover every current kind/type; Class-File examples include dense and general scratch forms. | General carrier/layout and multi-axis forms were deliberately unmeasured. |

Current tests already establish six primitive-array carrier forms plus `MemorySegment`, mixed
patterns, five pointwise access regimes, dense-int/general-long addressing, partial/empty ranges,
scalar/vector tails, external parallel ranges, multi-output entries, and zero-/scratch-bearing
entries. The closure samples independent equivalence categories rather than their combinatorial
cross-product.

## Scope

### Evidence-first workflow

- Before any production edit, create exactly one fresh directory with
  `mktemp -d /private/tmp/synaptik-cpu-0007a1c-XXXXXXXX`. Retain probe source/classes, generated
  classes, full `javap`, Class-File reports, member references, commands, environment, raw samples,
  summaries, and SHA-256 manifests only there.
- Record HEAD, complete dirty paths, Java/JVM/Gradle/OS/architecture/CPU facts, schema 31, and
  checksums for available retained evidence. Preserve every existing uncommitted change.
- Freeze probe and comparator source before timing. A correctness repair must be recorded and
  rehashed before final measurement; cases, thresholds, and comparators cannot change after a
  performance result is seen.
- Produce a machine-readable ledger mapping every current operation/family/form to its semantic
  test owner, retained/new Class-File category, retained/new performance category or explicit
  `STRUCTURAL_ONLY` reason, and represented types/carriers/access/address/range/vector/scratch axes.
- Reuse retained evidence only when its files, checksum, and schema verify. Otherwise replace it or
  record an unresolved gap; prose alone is not executable evidence.

### Fixed new matrix

Each timed case uses at least 262,144 output/visited scalars unless slice-oriented by nature.
Dimensions may be increased before the probe is frozen to reach the 25 ms sample floor, never
decreased or simplified.

| ID | Fixed generated case | Carrier/access and direct comparator |
|---|---|---|
| P-SCALAR-GENERAL | FLOAT32 scalar fused `DIV -> SIGMOID -> MUL`, `[512,512]` | Mixed array/segment, broadcast input, strided output, `GENERAL_LONG`; one fused scalar comparator with identical odometer/formula/narrowing/store work. |
| P-VECTOR-SEGMENT | preferred-species FLOAT32 `LOG1P -> TANH`, 1,048,576 values | Segment input/output; direct Vector API loop with identical species, lane operators, bound, tail, and segment access. |
| P-INTEGRAL-MIXED | preferred-species INT64 `ADD -> MAX -> CAST`, 1,048,576 values | Mixed `long[]`/segment; direct `LongVector` loop and tail with identical modular/signed semantics. |
| A-GENERAL | BFLOAT16 PERMUTE/SLICE affine copy, `[256,32,32]` | Segment to `short[]`, nonzero offsets/non-unit strides; direct raw-short composed mapping and long odometer. |
| M-PAD | BFLOAT16 PAD, at least 262,144 outputs | General segment-to-array; direct output mapping and identical exact padding bits. |
| M-CONCAT | INT32 four-occurrence CONCAT with one repeated input | Mixed unique boundaries; direct occurrence mapping and canonical coordinates. |
| M-STACK | BOOL four-occurrence STACK with one repeated input | Mixed canonical byte carriers; direct inserted-axis/occurrence mapping. |
| M-UNFOLD-AXIS | FLOAT32 non-unit-step/dilated UNFOLD_AXIS | Segment input, strided heap output; direct window mapping and identical addresses. |
| M-UNFOLD2D | FLOAT64 padded/dilated UNFOLD2D | Mixed general-long; direct im2col mapping and identical padding/carrier work. |
| I-GATHER | FLOAT64 GATHER with INT64 indices | Segment data, heap indices, strided segment output; identical validated direct writer, validation excluded on both sides. |
| I-GATHER-ND | FLOAT32 batch-one, tuple-depth-two GATHER_ND with non-scalar suffix | Mixed general-long with INT32 indices; identical tuple/suffix writer. |
| S-GENERAL-MIN | INT64 `SCATTER_ND + MIN`, duplicate tuples/non-scalar suffix | General segment/heap mix; identical range-owned copy-then-update, target filter, signed MIN, and update order. |
| F-FOLD2D | FLOAT32 padded/dilated overlapping FOLD2D | Mixed general-long; identical output-owned mapping and canonical represented addition. |
| O-ARGSORT | descending stable INT64 ARGSORT, `[4096,64]` | Segment input/strided segment output/scratch; identical stable merge, comparison/tie, slices, and stores. |
| R-DROPOUT-GENERAL | FLOAT32 DROPOUT, 1,048,576 values | Mixed general-long roles; exact V1 mapping, threshold, scaling, mask, state, and addresses. |
| C-SCAN-GENERAL | exclusive reverse INT64 CUM_PROD, `[1024,1024]`, axis one | Offset/strided segments; identical reverse/exclusive slice loop and modular multiplication. |
| X-MIN-MULTI | BFLOAT16 multi-axis MIN with kept dimensions | Segment input/strided `short[]` output; identical domain order, first-NaN/signed-zero policy, raw store. |
| X-ANY-SINGLE | BOOL single-axis ANY | Mixed carriers, zero-stride read/strided output; identical canonical fold/address work. |
| N-MEAN-GENERAL | FLOAT32 single-axis MEAN, `[128,2048]` | Segment input/strided heap output/exact scratch; identical exact-sum/rational-rounding state and access. |
| N-PROD-MULTI | BFLOAT16 multi-axis PROD with kept dimensions | Mixed general-long/exact scratch; identical significand/exponent/special-value state and one conversion. |

Retained cases cover the remaining dense algorithm categories. `INITIAL_STATE` is
`STRUCTURAL_ONLY`: its fixed two-word work would mostly measure call overhead. Computation-free
internal affine views are also structural-only and require no generated timing. No other distinct
current family/form may be omitted without leaving the task incomplete.

### Comparator, anti-DCE, and measurement rules

- Generated/direct sides use separate bit-identical preallocated inputs, outputs, geometry, and
  scratch, with identical represented types, algorithms, semantic work, carrier/layout/address
  work, ranges, state reset, and output mutation.
- Direct code is ordinary static primitive Java. It cannot call generated/emitter/reference/
  Synaptik helpers, collections/boxing, `BigInteger`, easier sorting/grouping/hashing, or easier
  numerical semantics. Vector cases use equivalent direct Vector API; scalar cases use scalar Java.
- Exact numerical/scatter product comparators use identical abstract algorithms and scratch.
  Ordering uses identical stable merge/scratch. Random uses the exact V1 mapping. Generation,
  definition, lowering, preparation, binding, allocation, initialization, validation, and full
  correctness scans are excluded symmetrically from timing.
- Both sides use pre-resolved exact matching `MethodHandle` values; timed work contains no lookup,
  reflection, or `invokeWithArguments`.
- After every invocation, sample raw bits from every output/state role into a local checksum;
  publish it once per batch to a static volatile sink and verify it after the fork. Sampling is
  identical on both sides. Verify exact outputs, immutable inputs, untouched regions, canaries,
  and scratch invariants before warmup and after every measured sample.
- Use deterministic initialization and fixed-seed randomized case/side order.
- Run five isolated forks with `-Xms1g -Xmx1g`, at least five randomized warmup rounds, nine
  randomized measured rounds, and adaptive batches lasting at least 25 ms. Report each fork's
  generated/direct median and ratio plus the median of fork medians; never average cases.
- Every fork and aggregate for every new case must be `<= 1.15x`. Retained thresholds/results stay
  unchanged. There is no minimum speedup because no known adverse baseline is assumed.
- If algorithm/resource equivalence cannot be preserved or noise prevents a fixed gate, retain
  `Review needed`/`Incomplete`; do not weaken, pool, or simplify the case.
- Measurements are release evidence only and never production configuration, cache identity,
  preparation, route selection, Runtime behavior, or tuning input.

### Class-File and forbidden-reference gates

- Generate deterministic representatives for every new row and each current family/form not
  distinct there, including all 48 pointwise opcodes mapped by scalar/vector emitted category,
  both scan kinds/four modes, aggregate kinds/forms, scatter reduction categories, both index
  widths, every admitted type, and both scratch signatures.
- Retain `.class`, exact descriptor/size/SHA-256, `javap -c -p`, `javap -v -p`, constant-pool and
  member-reference inventories, and Java 26 Class-File model reports. Equal specializations must
  regenerate byte-identically.
- Each class has zero fields, exactly one static entry, no constructor/secondary method, and the
  exact typed carrier/optional scratch/`long[]` geometry/`long start,end` descriptor.
- Reject `Object`, bridges, method-handle/type constants, `invokedynamic`, dynamic constants,
  bootstraps, reflection, graph/operation/runtime dispatch, collections/boxing, normal hot-path
  allocation, or Model/Compiler/Runtime/Prepare/reference/emitter/generator calls.
- All non-pointwise and scalar pointwise classes have no Synaptik-owned references. Vector
  pointwise classes may reference only their exact typed chunk-level `CpuVectorMath` method;
  scalar tails have none.
- Assert matching primitive array opcodes, native-order typed segment `get`/`set`, exact Vector API
  species/operations, and exact scratch presence/access. Use Class-File semantic categories, not
  source searches, pool indices, absolute bytecode offsets, hidden names, or JIT assembly.

### Correction and schema policy

- No production change is expected. A correction requires a frozen reproducible pre-edit failure,
  retained class/decompilation/timing, a named faulty emitted shape, and a stable test failing first.
- Correct at most two cohesive code-shaping owners. More independent failures, a third owner, or
  IR/lowering/resource redesign requires stop/replan and separate ordered work.
- If emitted bytes change, increment schema exactly once from 31 to 32. Schema-31 artifacts become
  incompatible safe misses; add no migration, alias, converter, partial reuse, or dual-schema path.
- If emitted bytes do not change, schema remains 31. Tests/evidence/docs/Javadoc never increment it.
  A production change claimed byte-neutral must regenerate every affected category byte-identically.
- Stop before changing semantics, thresholds, eligibility, capability, public API, resources,
  materialization, routes, preparation/Runtime, parallel ownership, numerical order, dependencies,
  architecture, Gradle/build, another module, or later work.

## Out of scope

- Universal cross-product/machine/JVM/species/size parity or duplicate reruns of verified evidence.
- Timing metadata-only/zero-work views, validation failures, or two-word INITIAL_STATE throughput.
- Easier direct algorithms/resources, weakened gates, toleranced exact results, or using one
  representative for a distinct algorithm.
- Production benchmark tooling, JMH/public APIs, Config, autotuning, tuning caches, runtime
  profiling/selection, or benchmark-driven mutation.
- New semantics or CPU 0007A2+, native routes, fusion expansion, conformance/integration, other
  modules/backends, architecture/build, commit/push/stage/revert/delete, or unrelated refactoring.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`performance evidence and tuning`](../../../../architecture/performance-evidence-and-tuning.md)
- [`runtime, Prepare, and backend boundary`](../../../../architecture/runtime-prepare-backend-boundary.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`general profile`](../../../../developer-guide/documentation/general-style.md)
- [`planning profile`](../../../../developer-guide/documentation/planning-style.md)
- [`backend-guide profile`](../../../../developer-guide/documentation/backend-guide-style.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007A0`](0007a0-generated-hot-path-parity-correction.md)
- [`CPU 0007A0A`](0007a0a-affine-and-movement-generated-loop-parity.md)
- [`CPU 0007A0B`](0007a0b-indexing-generated-loop-parity.md)
- [`CPU 0007A0C`](0007a0c-scatter-generated-loop-parity.md)
- [`CPU 0007A0D`](0007a0d-fold-generated-loop-parity.md)
- [`CPU 0007A0E`](0007a0e-ordering-generated-loop-parity.md)
- [`CPU 0007A0F`](0007a0f-random-and-dropout-generated-loop-parity.md)
- [`CPU 0007A1`](0007a1-portable-ordinary-numerical-aggregate-reductions.md)
- [`CPU 0007A1A`](0007a1a-generated-scalar-body-self-containment.md)
- [`CPU 0007A1B`](0007a1b-scatter-algorithmic-parity.md)

## Architecture constraints

- Model owns semantics; only CPU-private realization/evidence may change.
- CPU analysis owns lowering/specialization/strategy/declarations before assignment; finalization
  owns generation/reuse afterward; Runtime only executes immutable prepared work.
- Generated classes stay typed, direct, allocation-free in normal hot work, and free of mutable
  static run state. Runtime identities/carriers/addresses/slots/measurements stay outside identity.
- Concurrent runs retain distinct `RunState`, geometry, buffers, and scratch.
- Benchmark evidence may accept/reject/diagnose, never select or mutate production behavior.

## Package impact

Existing packages used:

- `internal.codegen.emit` — family generation, verification, and new test-only closure matrix.
- `internal.cache` — schema/artifact compatibility only if emitted bytes change.
- `internal.ir`, `.lowering`, `.prepare`, `.route.portable`, `.executable`, `.reference` — read-only
  inventory and semantic/resource owners; no contract change is allowed.

No production package, export, supported API, or module is added/moved. New test
`CpuGeneratedDirectEvidenceClosureTest` belongs beside emitters because it owns cross-family
Class-File evidence, not lowering or Runtime behavior.

## Affected and allowed files

Expected evidence-only paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedDirectEvidenceClosureTest.java` (new)
- `docs/backend-guide/cpu-backend.md`
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`
- `docs/glossary.md` only if a reusable term changes; expected result is no change.

Optional correction allowlist; at most two code-shaping owners may change:

- `CpuClassFileKernelGenerator.java`, `CpuLoopEmitter.java`, `CpuCarrierEmitter.java`,
  `CpuScalarEmitter.java`, `CpuVectorInstructionEmitter.java`, `CpuAffineCopyEmitter.java`,
  `CpuDataMovementEmitter.java`, `CpuIndexingEmitter.java`, `CpuScatterEmitter.java`,
  `CpuFoldEmitter.java`, `CpuOrderingEmitter.java`, `CpuRandomEmitter.java`, `CpuScanEmitter.java`,
  `CpuAggregateEmitter.java`, `CpuExactSumEmitter.java`, or `CpuExactProductEmitter.java` under
  `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/`.

Optional compatibility/Javadoc consequences are exactly `internal/cache/CpuGeneratorSchema.java`,
`internal/codegen/emit/package-info.java`, and `internal/cache/package-info.java`.

Optional tests are limited to the corrected family's current `Cpu*GeneratedKernelTest.java`, plus
`CpuGeneratedKernelArtifactStoreTest.java`, `CpuKernelSpecializationTest.java`,
`CpuPartitionPreparerTest.java`, and `CpuPartitionFinalizerTest.java` when schema changes. No other
source, test, documentation, architecture, build, or planning path is allowed.

## Maximum scope

At most 18 repository paths:

- 5 production/package paths: no more than two code-shaping owners, schema, two package summaries;
- 7 tests: new matrix, no more than two family owners, up to four compatibility owners;
- 6 documentation/planning paths; and
- 18 total.

Evidence files are outside the repository. A third owner, another test owner/module, IR/lowering/
resource change, or larger total is a stop/replan condition.

## Acceptance criteria

- The ledger enumerates every current operation/family/form and maps semantic, Class-File, and
  retained/new performance evidence or a justified structural-only result.
- All twenty new cases pass exact semantics and every fork/aggregate `<= 1.15x`.
- Retained evidence verifies by checksum/schema or is replaced/classified unresolved.
- Every selected class passes descriptor, member, forbidden construct/reference, scratch,
  determinism, decompilation, and checksum gates.
- Combined evidence spans all six primitive-array carrier types, both index widths, segment/mixed
  carriers, five pointwise regimes, dense/general addressing, scalar/vector/tails, partial/empty
  ranges, multi-output, zero scratch, and exact sum/product scratch.
- Only INITIAL_STATE and computation-free views are structural-only absent a new unresolved gap.
- Production changes occur only for a frozen demonstrated failure and stay within two owners.
- Schema stays 31 if bytes are unchanged, or advances once to 32 with schema-31 safe misses and no
  migration when bytes change.
- Architecture/API/capability/semantics/resources/routes/Prepare/Runtime/dependencies/build/later
  work and unrelated dirty paths remain unchanged.
- A distinct clean documentation context finalizes affected Javadocs/package summaries, guide,
  glossary impact, task evidence/status, master plan, and roadmap with reasoned no-change results.

## Tests / validation

After executable Java stabilizes, run one focused matrix and one authoritative module suite:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedDirectEvidenceClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuFusedGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuVectorMathTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAffineCopyGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuDataMovementGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuIndexingGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuScatterGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuFoldGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuOrderingGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuRandomGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuScanGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateGeneratedKernelTest
./gradlew :backends:cpu:test
```

Run the frozen probe outside JUnit from the evidence directory. Record exact `javac` and five
`java -Xms1g -Xmx1g` commands. For every class run:

```bash
javap -c -p <generated-class-file>
javap -v -p <generated-class-file>
sha256sum <generated-class-file>
```

The distinct documentation context reuses stable Java/timing evidence unless executable/probe
behavior changes, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also validates changed Markdown links/anchors/fences/newlines/whitespace; exact allowlist/count;
package/type placement; schema policy; 0007A1B Complete -> 0007A1C current status -> corrective
frontier -> 0007A2 Draft; exactly one Ready corrective task; no later detailed correction or
0007A2 specification; empty staged diff; preserved dirty paths. Repository/architecture/
conformance/integration validation remains deferred to CPU 0009 or CI because no shared contract
may change; otherwise stop and replan.

## Dependencies

- CPU 0007A0–0007A0F: Complete retained family parity/decompilation foundation.
- CPU 0007A1: Complete numerical algorithms/scratch/general Class-File and thirteen dense gates.
- CPU 0007A1A: Complete scalar self-containment/member-reference evidence.
- CPU 0007A1B: Complete current schema 31 scatter algorithms/six gates/general evidence.
- Current Java 26 generator, specialization/schema/store, every family owner, preparation/
  finalization/execution/reference, and generated-family tests define the inventory.

## Follow-up tasks

- CPU 0007A1D retains the stable schema-32 invocation-local segment-layout implementation but is
  Incomplete because every one of its 13 required performance targets failed the final fork.
- CPU 0007A1E, CPU 0007A1F, and CPU 0007A1G are Complete. Their retained evidence closes the
  movement general-address, canonical-BOOL STACK/ANY, and FOLD2D/dropout groups without closing
  this full task.
- CPU 0007A1H through CPU 0007A1L are Complete and close the bounded numerical-aggregate,
  indexing, cumulative-scan, affine-copy, and pointwise-general clusters. Detailed Ready
  [CPU 0007A1M](0007a1m-scatter-min-residual-parity.md) is the sole next task and owns persistent
  `S-GENERAL-MIN`. `X-MIN-MULTI` remains Draft and unassigned.
- CPU 0007A2 remains Draft and blocked until all corrections close the frozen failures and CPU
  0007A1C can become Complete.
- CPU 0009 remains the later repository-wide portable conformance checkpoint.
- More than two correction owners requires a separately planned insertion before CPU 0007A2; do
  not pre-create it here.

## Architecture impact

Expected impact: None. A required architecture/dependency/public/shared/resource/route/Runtime/
tuning change is a stop condition, not implied scope.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1C. Work on the existing
uncommitted diff without committing, pushing, staging, reverting, deleting, or modifying unrelated
work. Do not use a GSD skill/workflow.

Read in full AGENTS.md, ARCHITECTURE.md, linked architecture/performance/planning/documentation
pages, CPU master plan/guide, task 0007A1C and dependencies 0007A0 through 0007A1B, every current
family IR/lowering/emitter/generator/specialization/schema, related tests, and retained evidence.
Inspect and preserve the complete dirty diff.

Create the one evidence directory, freeze the ledger/probe/direct comparators, add the stable
Class-File closure test, and run the exact twenty-case/five-fork gates. Treat work as evidence-first.
Change production only for a retained reproducible failure within correction/schema/allowlist/stop
rules. Never weaken semantics, resources, thresholds, comparators, architecture, or scope.

After executable work stabilizes, hand diff/task/evidence/checksums/commands/results to a distinct
clean documentation context. It independently inspects final source/tests/classes/evidence,
finalizes Javadocs/package summaries/guide/glossary/task/master/roadmap, and runs documentation/
scope/schema/status/whitespace gates without repeating stable Java/timing absent executable/probe
change or a recorded stale-evidence reason. Do not mark Complete until every gate passes.
```

## Local decisions

- Close equivalence categories, not the cross-product; distinct algorithms get distinct cases.
- New timing emphasizes general-long/segment/mixed paths because retained timing is mostly dense.
- Exact handles and identical output sampling make dispatch/anti-DCE symmetric.
- Fixed two-word INITIAL_STATE and computation-free views are structural-only.
- Two correction owners is the maximum; broader failure is new ordered work.
- Schema changes only for emitted bytes.

## Known limitations

- Ratios apply only to fixed cases, environment, JVM, and protocol; never universal or selecting.
- Not every carrier/layout/type/worker/alignment/species/size cross-product is timed; existing
  semantic/Class-File tests cover independent axes.
- Entry timing does not separately benchmark worker-group orchestration; range correctness and
  scalar/parallel reuse remain automated lifecycle evidence.
- Unverifiable retained evidence must be replaced or unresolved.
- Native/future families/runtime end-to-end/repository conformance remain outside this frontier.
- The first frozen performance fork failed 17 of 20 rows, so forks two through five, aggregate
  ratios, exhaustive Class-File member inventories/reports, the focused Java matrix, the full CPU
  suite, and Javadoc remain intentionally unexecuted under the task's stop rule.
- `schema-after.txt` is malformed (`schema=` without a value). It cannot prove the post-fork
  schema. Current source independently remains at schema 31, but the malformed artifact remains an
  unresolved evidence claim rather than a pass.

## Validation evidence

Planning context inspected governing instructions; architecture, performance, runtime/prepare/
backend, planning, and documentation contracts; complete CPU master plan/roadmap; CPU guide and
glossary; completed 0007A0–0007A1B plans and retained evidence; the current dirty diff; all current
generated family emitters/generator/specialization/schema/IR inventory; and related generated tests.

Source confirms ten categories, six primitive-array forms plus segment, five pointwise regimes,
dense-int/general-long, scalar/vector tails, external parallel ranges, zero/scratch entries, and
schema 31. Tests provide broad semantic/structural coverage, while retained performance is mostly
dense. The fixed twenty-case matrix closes the identified general/form gaps and keeps
INITIAL_STATE/views honestly structural-only.

Planning changed exactly this task, CPU master plan, and roadmap in context
`01a01b6b-fc28-7951-91d5-08c7913b27b9`. Local-link existence passed and the three changed files
contain no local anchor links; fences are balanced, final newlines are present, and trailing
whitespace is absent. CPU 0007A1B is Complete, 0007A1C is the sole Ready CPU row, CPU 0007A2 is
Draft, and no 0007A2/later detailed CPU task exists. The staged diff is empty; the pre-existing
dirty work is preserved. `git diff --check` and `git diff --cached --check` pass. No Java test,
Javadoc, generation, decompilation, or benchmark command ran because this is planning-only.

Mandatory documentation/planning context `/root` independently inspected the final test, current
emitters/schema, frozen source, ledger, raw first-fork output, summaries, checksums, generated
classes, and retained `javap` output. `sha256sum -c` passed for all three frozen sources, the
ledger, and all 20 generated classes. An `awk` recount of `fork-1-summary.csv` confirmed three
passing and 17 failing rows with the expected passing names. A targeted `find` confirmed no
fork-two-through-five, aggregate, exhaustive, focused-suite, or Javadoc result file exists.

For the four final planning paths, the local Markdown target check passed; there are no local
anchor links to validate; fences are balanced; final newlines are present; and trailing
whitespace is absent. Status checks found exactly one CPU master-plan `Ready` row and exactly one
detailed `Ready` CPU task, both CPU 0007A1D. Current source reports schema 31. `git diff --check`
and `git diff --cached --check` passed, the staged diff is empty, and `git status --short` confirms
the pre-existing dirty CPU implementation/test/documentation work plus the preserved new closure
test and the four planning paths. Java tests, timing, full Class-File reporting, and Javadoc were
not rerun because this pass changed no executable Java or Javadoc and the stop rule deliberately
left those gates open.

Implementation context `01a01dcf-2ad5-75c1-9e60-94cb4c69a659` subsequently completed the bounded
CPU 0007A1D code shape at schema 32. Its focused matrix passed 119 tests, the authoritative CPU
suite passed 54 suites/345 tests with zero failures/errors and one skip, and frozen semantics
again passed 20/20. The three controls passed. The final required fork nevertheless failed all 13
A1D targets; forks two through five correctly did not run and no aggregate exists. A diagnostic
version eliminating native-order/with-order construction entirely still left ten A1D targets
above `1.15x`, proving that layout construction was only one partial cause. Clean documentation/
planning context `/root` retained A1D as Incomplete, preserved every acceptance gate, and selected
only the four-row `CpuDataMovementEmitter` cluster as detailed Ready CPU 0007A1E.

## Implementation notes

- The implementation context created and froze one 20-row semantic/performance probe, its direct
  comparators, the complete operation/family/form ledger, generated representatives, checksums,
  `javap` output, and the stable repository test
  `CpuGeneratedDirectEvidenceClosureTest` in evidence directory
  `/private/tmp/synaptik-cpu-0007a1c-oMAmuVhF`.
- Frozen semantic comparison passed all 20 cases exactly (`VERIFIED,20`). The first isolated fork
  produced 17 ratios above `1.15x`. Only `P-VECTOR-SEGMENT` (`0.985088753x`),
  `P-INTEGRAL-MIXED` (`0.266417978x`), and `O-ARGSORT` (`0.871210032x`) passed.
- The failing rows were `P-SCALAR-GENERAL`, `A-GENERAL`, `M-PAD`, `M-CONCAT`, `M-STACK`,
  `M-UNFOLD-AXIS`, `M-UNFOLD2D`, `I-GATHER`, `I-GATHER-ND`, `S-GENERAL-MIN`, `F-FOLD2D`,
  `R-DROPOUT-GENERAL`, `C-SCAN-GENERAL`, `X-MIN-MULTI`, `X-ANY-SINGLE`, `N-MEAN-GENERAL`, and
  `N-PROD-MULTI`.
- No production code-shaping owner changed in the implementation context. Current source still
  declares schema 31. The recorded `schema-after.txt` value is malformed and is therefore not
  accepted as evidence.
- The explicit more-than-two-code-shaping-owners stop condition fired. The implementation context
  correctly stopped without changing production Java, weakening or refreezing the probe, running
  forks two through five, calculating aggregate ratios, or running the remaining Class-File,
  focused/full Java, or Javadoc gates.
- Evidence-driven correction order starts with the shared native-order segment-access bytecode
  shape in CPU 0007A1D. That implementation is retained as a stable prerequisite but did not close
  parity. The next correction is CPU 0007A1E's four-row movement general-address-loop cluster;
  BOOL movement/aggregate and fold/dropout remain later ordered groups.
- CPU 0007A1I subsequently closes both indexing rows at schema 37. Frozen semantics remain
  `VERIFIED,20`; `I-GATHER` and `I-GATHER-ND` pass all five forks with medians `0.853338331x` and
  `0.745954874x`, and all three controls pass. Exactly five rows remain persistent in the fresh
  accepted evidence. CPU 0007A1J is the sole next Ready owner cluster.
- CPU 0007A1J subsequently closes `C-SCAN-GENERAL` at schema 38. Frozen semantics remain
  `VERIFIED,20`; the scan row passes all five accepted forks with median `0.978732901x`, and all
  three controls pass. Two complete samples were rejected because one unrelated row exceeded the
  gate in one fork; their data is retained but does not contribute accepted evidence. Exactly four
  rows remain persistent, and CPU 0007A1K is the sole next Ready owner cluster.
- CPU 0007A1K subsequently closes `A-GENERAL` at schema 39. Frozen semantics remain
  `VERIFIED,20`; the affine-copy row passes five accepted forks with median `0.653888342x`, and
  all three controls pass. One complete sample was rejected because an unrelated `M-CONCAT` fork
  exceeded the gate and contributes no accepted evidence. Exactly `P-SCALAR-GENERAL`,
  `S-GENERAL-MIN`, and `X-MIN-MULTI` remain, and CPU 0007A1L is the sole next Ready owner cluster.
- CPU 0007A1L subsequently closes `P-SCALAR-GENERAL` at schema 40. Frozen semantics remain
  `VERIFIED,20`; the pointwise row and all three controls pass every accepted fork and median,
  leaving exactly `S-GENERAL-MIN` and `X-MIN-MULTI`. CPU 0007A1M is the sole next Ready owner
  cluster.
- CPU 0007A1M subsequently closes `S-GENERAL-MIN` at schema 41. Frozen semantics remain
  `VERIFIED,20`; the scatter row passes five accepted forks at `0.984900063x`, `0.988888234x`,
  `0.983823803x`, `0.978065816x`, and `0.992400680x`, with median `0.984900063x`, and all three
  controls pass. The unchanged full probe now leaves only `X-MIN-MULTI`, so CPU 0007A1N is the
  sole next Ready owner cluster. A1C remains incomplete until that residual and the original
  closure gates are reconciled.
- Architecture and ADR no-change conclusion: no production owner, lifecycle, module boundary,
  dependency direction, backend ownership, Prepare/Runtime handoff, route, resource, capability,
  or semantic contract changed, so `ARCHITECTURE.md`, focused architecture pages, and ADRs remain
  accurate.
- Test/build/API no-change conclusion: the preserved closure test is the only new Java artifact in
  this context; no dependency, Gradle/toolchain, public API, or executable contract changed, so
  architecture tests, backend conformance, integration tests, build files, and API reference pages
  require no edit. Their suites remain intentionally unexecuted under the stop rule.
- Guide/glossary/Javadoc no-change conclusion: an incomplete local benchmark is planning evidence,
  not a supported backend claim or reusable term. The CPU guide and glossary therefore remain
  unchanged. Current package/type Javadocs still describe schema-31 behavior and unchanged
  execution contracts; Javadoc was not rerun because no Javadoc changed.

## Completion summary

- Completed changes: froze and retained the 20-case comparison and ledger, passed exact semantics,
  ran the first isolated performance fork, preserved the new structural closure test, stopped at
  the required owner boundary, and replanned the corrective dependency chain.
- Files changed or created by the implementation result: new
  `CpuGeneratedDirectEvidenceClosureTest.java`; this task and synchronized planning records are
  finalized by the documentation pass. No production Java changed in context 0007A1C.
- Tests and validation: original A1C semantics passed 20/20 and its first fork produced three
  passing and 17 failing ratios before the owner stop. Fresh A1I evidence preserves
  `VERIFIED,20`, passes 48 focused tests and 354 uncached CPU tests, and closes both indexing rows
  plus all three controls across five forks. Fresh A1J evidence closes the scan row, and fresh
  A1K evidence passes 47 focused tests and 355 of 356 CPU tests and closes the affine-copy row;
  all retain five accepted target/control forks. Fresh A1L evidence closes the pointwise row at
  median `1.006301612x`; fresh A1M evidence revalidates `VERIFIED,20`, passes 59 focused and 358
  full CPU tests with one expected skip, and closes the scatter row at median `0.984900063x`.
  A1C remains incomplete because `X-MIN-MULTI` still fails all five accepted forks.
- Documentation-agent review: mandatory clean documentation/planning context `/root`; General and
  Planning profiles were primary, with Backend Guide, API/Javadoc, and Example profiles applied to
  the no-change review.
- Documentation impact: this task, CPU master plan, roadmap, and the immediate detailed corrective
  task are synchronized. The CPU backend guide and glossary receive accepted schema-41 scatter
  behavior and evidence, while retaining the incomplete broader closure boundary.
- Javadoc review: the A1M documentation pass finalized affected scatter/schema/cache/prepare
  Javadocs for schema 41 and retained complete guards, arbitrary legal ranges, zero workspace, and
  the typed general fallback; final CPU Javadoc passes.
- Glossary impact: existing CPU portable-route, specialization, and artifact definitions advance
  to schema 41; no new reusable term was introduced.
- Unresolved issues: fresh accepted A1M evidence leaves exactly one persistent row,
  `X-MIN-MULTI`. Original closure gates intentionally stopped by this task remain open until that
  row closes and the evidence is reconciled.
- Follow-up required: CPU 0007A1E–CPU 0007A1M are Complete. Execute Ready CPU 0007A1N; after the
  final persistent frozen failure closes, resume the unchanged closure protocol and reconsider
  0007A1C completion.

Status: Incomplete
Follow-up required: CPU 0007A1N must close X-MIN-MULTI and the original closure evidence must be reconciled before CPU 0007A2.
