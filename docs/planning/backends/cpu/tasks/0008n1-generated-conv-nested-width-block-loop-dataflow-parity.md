# Task 0008N1: Generated Conv nested width-block loop/dataflow parity re-spike

## Status

Complete

## Goal

Replace the rejected per-cell/per-block flat-ordinal decode with generated Conv2d/Conv3d nested
loops that match an optimal specialized clean Java Vector API oracle in semantic algorithm,
loop/dataflow shape, and avoidable-overhead profile. Select the existing output-width SIMD form
only after the fixed evidence gates pass. Stage A may generate a test/evidence-only provisional
schema-63 class with an identity distinct from scalar while schema 62 remains current; it cannot
be a production route or artifact. Otherwise retain the bounded scalar stop and schema 62.

## Scope

The candidate is same-typed dense direct `FLOAT32`/`FLOAT64` Conv2d/Conv3d, optional intrinsic
bias, arrays, native-order segments, or ordered mixed carriers; Conv2d external epilogues are
excluded. Width stride and dilation are one. Every preferred-species lane owns one output-width
cell and retains exactly increasing `channel -> kd -> kh -> kw` accumulation order (`kd` is absent
for Conv2d): broadcast scalar bias/weight, vector-load input, multiply, then add. FMA,
reassociation, horizontal reduction, mask, gather, scatter, lane extraction, callback, allocation,
boxing, reflection, and dispatch are forbidden.

Generate `n`, group/output-channel, output-height, output-width loops for Conv2d, and `n`,
group/output-channel, output-depth, output-height, output-width loops for Conv3d. The one cold
prologue decodes `start` in reverse row-major order: first `ow`, then `oh` (and `od` for Conv3d),
then global `oc`, then `n`; it derives `group = oc / outputChannelsPerGroup` and initializes every
row/base address from those coordinates. `start == end` returns before any carrier access. The
half-open worker range is authoritative even when it begins or ends mid-row: before every scalar
cell or vector block, require the current ordinal to be below `end`.

An inlined scalar cell advances `ow` by one and increments its width-local input/output bases. At
`ow == outputWidth`, it resets `ow` and width bases, increments `oh`, and carries/reset through
`od` (Conv3d), `oc`, and `n` in that order, recomputing only the new enclosing-row bases. It is
used from a mid-row start through the next legal interior/vector boundary, for padding borders,
tails, and for a mid-row end. A full block is legal only when its lane count is within both the
current row and `[currentOrdinal, end)`, and every lane is interior for every `kw`. It advances
`ow`, current ordinal, and each input/output width base by exactly the preferred lane count; the
next block reuses those incremented locals. Scalar padding borders, range fragments, and tails use
that same inlined scalar cell body with no Synaptik hot-helper leakage. Preserve batch,
groups/depthwise, bias, and non-width stride/dilation. Conv1d composition and BF16 remain
scalar/deferred.

Preload dynamic extents, strides, offsets, and bases from `long[] geometry` into locals once per
invocation or the corresponding enclosing loop. They are cold facts, never class identity. Full
interior width blocks contain no division/modulo, repeated geometry-array load, complete-coordinate
reconstruction, range-object work, helper call, or dispatch. Existing cold lowering/binding checks
must prove `0 <= start <= end <= outputElements`, coordinate products/sums, element and byte
addresses, and segment byte-width multiplication without `long` overflow before entry. Heap-array
`int` conversion is emitted only after the existing proof that every used address fits the selected
array carrier's legal index; otherwise that occurrence remains scalar/general or is rejected by
the existing owner. Segment addresses stay checked `long` element/byte addresses in native order.
No generated increment may wrap; its loop guard and the cold proofs jointly bound it. Ordered mixed
carriers remain direct. Retain accessibility, overlap, failure/join, reusable executable, and
run-isolation behavior.

Stage A may add only the narrowly admitted provisional `classIdentitySchema=63` specialization
and generator path needed to generate and inspect that distinct candidate class in focused tests
or outside-repository evidence. `CpuGeneratorSchema.CURRENT_VERSION` remains 62; production
`CpuPartitionPreparer` must not select `VECTOR` or `PARALLEL_VECTOR` for Conv, and the artifact
store/cache must neither publish nor reload schema-63 classes. The candidate must use its own
schema-63 identity bytes, binary name, and key; it must never reuse scalar schema-52 bytes,
schema-62 compatibility metadata, or either scalar cache key. Focused Stage A generation bypasses
artifact-store/cache realization entirely. Only after Stage A and every
inherited Stage B gate pass may the same task advance `CURRENT_VERSION` to 63 and enable
production preparer selection and compatible schema-63 artifact publication/reload.

## Exclusions

- Model, Compiler, Training, shared Prepare, Runtime, public API, dependency, architecture,
  Gradle, native-route, autotuning, or fixed-shape-class changes.
- Conv1d-specific SIMD, BF16, mixed promotion, non-dense access, packing/materialization, width
  stride/dilation other than one, horizontal/partial reductions, and external Conv suffixes.
- Threshold changes, invented measurements, retries/discards, or branch selection after a stop.

## Architecture constraints

[`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) is authoritative. CPU analysis alone owns route
predicate, carrier form, preferred species, strategy, and selective schema decision; CPU
finalization alone realizes the immutable artifact after shared slots. Planning, shared Prepare,
Runtime, Model, Compiler, and Training do not learn vector or Conv-route details. Direct generated
entries retain typed carriers, `long[] geometry`, `long start`, and `long end`; hot work may not
perform graph, operation, layout, worker, cache, or route lookup. Stop for an architecture decision
if a shared/public boundary would need to change.

## Package impact

No package is added or moved. Conditional work remains in
`io.github.pho001.synaptik.backend.cpu.internal`: `ir` only for a code-shaping realization fact;
`prepare` for exact eligibility/range-safe strategy; `cache` for schema/artifact compatibility;
and `codegen.emit` for nested Conv/direct carrier/vector emission. Tests remain in their existing
`cache`, `codegen.emit`, `lowering`, and `prepare` packages. Scalar IR encoding remains byte-stable.

## Schema decision

Schema 62 remains `CpuGeneratorSchema.CURRENT_VERSION` throughout Stage A. Stage A may admit only
an explicit provisional `classIdentitySchema=63` `Compute.VECTOR` Conv specialization to the
generator and focused test/evidence path, so actual vector bytes have an identity, binary name,
and structural key distinct from scalar. This is not production selection: `CpuPartitionPreparer`
cannot select Conv `VECTOR` or `PARALLEL_VECTOR`; the artifact store/cache has no schema-63
publication, reload, or production compatibility; and no schema-63 envelope is written or read.
The provisional candidate must never emit under schema 62 or reuse scalar schema bytes,
compatibility metadata, binary names, or cache keys; its focused generation bypasses the
artifact-store/cache realization path entirely.

Schema 63 becomes current and production-selectable only after Stage A and every inherited Stage B
gate pass. Its exact static identity facts are direct Conv rank (2d or 3d), the
`OUTPUT_WIDTH_LANE_PARALLEL_V1` nested-loop realization version, `Compute.VECTOR`, same typed
FLOAT32 or FLOAT64 input/weight/optional-bias/result roles, intrinsic-bias presence, no external
epilogue, compile-time Conv stride/padding/dilation/groups attributes, dense resolved access
regime, ordered carrier pattern, and preferred-species bit size. These facts select the generated
bytes. Concrete extents/output width, offsets, strides/bases from geometry, carrier objects and
addresses, start/end, worker ranges/counts, slots, run state, and the invocation's interior-block
proof are dynamic facts and do not identify schema 63. On any Stage A or Stage B failure, remove
every provisional production-source admission change, retain only outside-repository evidence and
this planning record, preserve scalar bytes, and leave `CURRENT_VERSION` at 62. If the current
constructors cannot represent the provisional identity without broader production changes, add
only a scoped test-fixture seam within the inherited sixteen test/evidence paths or stop and
re-plan; never emit the candidate under schema 62.

## Affected files and defensible maximum scope

The conditional ceiling is exactly `9 production + 16 test/evidence + 3 planning = 28` paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv2dEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuConv3dEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuConv2dIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuConv3dIr.java`
- The sixteen focused test/evidence paths in [CPU 0008N](0008n-measured-profitable-float32-float64-conv2d-conv3d-simd-accumulation.md#affected-files-and-maximum-scope), including its three Conv SIMD evidence/performance owners.
- This task, [`../master-plan.md`](../master-plan.md), and [`../../../roadmap.md`](../../../roadmap.md).

Carrier direct-vector support is included because it may be necessary to remove hot overhead; all
shared/public owners are excluded, making this the smallest defensible ceiling.

## Acceptance criteria

1. Both ranks have the stated nested progression, single cold start decode, exact end/zero-work
   handling, scalar fragments, and incremented width-local state; vector interiors contain no
   prohibited ordinal/geometry/range/helper/dispatch work.
2. Every lane preserves existing scalar ordered accumulation and special-value semantics. All
   borders, tails, fragments, and unsupported forms retain scalar behavior.
3. Stage A uses an actual generated Conv2d F32 array exact-interior provisional schema-63 class,
   whose identity, bytes, binary name, and key are distinct from the current generated scalar,
   versus that scalar and an independently `javac`-compiled direct vector oracle in the fixed
   one-row fresh-fork protocol below. `CURRENT_VERSION` remains 62; production Conv selection and
   schema-63 artifact-store publication/reload remain disabled. Every retained ratio requires
   `V/S <= 0.90` and `V/D <= 1.15`; failure is a bounded stop, removal of provisional
   production-source admission, and schema-62 retention.
4. Only after Stage A passes, execute inherited 0008N's 48 semantic rows, 24 structural dossiers,
   16 branch rows, 5 forks, 1,440 pairs, 5,760 timings, 160 fork medians, and 32 aggregates under
   the same gates. No retry, discard, replacement, threshold change, or unrecorded expansion.
5. Retain actual generated class bytes, `javap`/decompilation, direct-oracle class, raw timings,
   calibration, commands, environment, checksums, and manifest outside the repository. Only after
   the complete Stage B pass may `CURRENT_VERSION` advance to 63 and production preparer/cache
   compatibility select, publish, or reload schema 63.
6. Complete separate clean implementation and documentation-focused passes; the latter reviews
   Javadocs/planning/glossary and records no-change conclusions without rerunning successful Java
   suites unless it changes executable Java.

## Validation

Stage A is a separate admission experiment, not one row or fork of the inherited matrix. It fixes
one actual-generated `Conv2d/FLOAT32/all-arrays/interior-exact-width/VECTOR` workload, independent
`javac` compilation of the direct vector oracle, Java 26, recorded CPU/OS/JDK facts,
`-Xms1g -Xmx1g -XX:-TieredCompilation -Xbatch`, and a deterministic row/order seed. Run five
independent fresh JVM forks. In every fork, run five warmup symmetric `V-S-S-V`/`S-V-V-S` pairs and
five warmup symmetric `V-D-D-V`/`D-V-V-D` pairs. Double one shared iteration count until the
two-invocation aggregate for each of `S`, `V`, and `D` is at least 50 ms. Then retain exactly nine
randomized symmetric pairs for each comparison, with no retry, discard, replacement, or asymmetric
work; every one of the 360 timed sides is at least 25 ms.

Thus Stage A retains `1 row * 5 forks * 9 pairs * 2 comparisons = 90 pairs`,
`90 * 4 = 360 timings`, `1 * 5 * 2 = 10` fork medians, and `1 * 2 = 2` median-of-fork-median
aggregates. Require `V/S <= 0.90` and `V/D <= 1.15` for every retained pair, fork median, and
aggregate. Retain the command, JVM/CPU/OS facts, seed/order, warmup, calibration, all timings and
medians, actual generated schema-63 class bytes and identity/key evidence, and independently
compiled oracle. Confirm `CURRENT_VERSION == 62`, scalar-only production Conv preparation, no
schema-63 artifact publication/reload, and no scalar-schema bytes or keys reused by the candidate.
Only a full Stage A pass permits the separate inherited 48/24/16-row matrix to begin.

After Stage A, run inherited 0008N exactly: 48 raw-bit semantic rows; 24 Class-File dossiers; and
16 branch rows with five fresh forks, nine retained symmetric pairs per comparison, four timings
per pair, minimum 25 ms per timing, and every pair/median/aggregate at `V/S <= 0.90` and
`V/D <= 1.15`. Inspect vector direct-carrier forms, broadcasts, multiply/add, forbidden overhead,
and nested-loop parity through bytecode/decompilation.

Run focused CPU tests, retained performance test with explicit evidence root, CPU Javadoc,
`git diff --check`, and `git diff --cached --check`. Check links/anchors/fences/headings/newlines/
whitespace, `2*2*12=48`, `2*2*2*3=24`, `2*2*4=16`, `16*5*9*2=1440`, `1440*4=5760`,
`16*5*2=160`, `16*2=32`, status/order/schema/scope, provisional-versus-production cache boundary,
and empty index. CPU 0009/CI retains broad validation unless a forbidden boundary changes.

## Dependencies

- CPU 0008N is Incomplete after its bounded scalar stop; its evidence is factual input, not
  acceptance evidence.
- CPU 0008, 0008A, 0005C/0005I, 0007A0/0007A1A, 0008L, and 0008M are Complete.
- Java 26 Vector API and current scalar Conv lowering/preparation/carrier/generated owners exist.

## Follow-ups

- CPU 0008O remains Draft and depends on 0008N1; it must not start first.
- CPU 0008P and CPU 0009 remain later. A stop may add a follow-up only if independently actionable;
  it does not advance 0008O.

## Implementation prompt

```text
You are the clean-context implementation agent for CPU 0008N1. Read AGENTS.md,
ARCHITECTURE.md, the Planning Guide, CPU master plan, CPU 0008N and this task, current Conv2d,
Conv3d, carrier, geometry/preparer/finalizer, cache/schema, lowering/reference/test owners, and
the retained 0008N spike evidence. Do not commit, stage, push, reset, or touch unrelated work.
Implement only the nested-loop width-block re-spike: decode start once, carry coordinates/base
addresses, use scalar boundary fragments, and keep vector interiors free of ordinal decode,
geometry loads, helper/dispatch/range work. Preserve semantics and safety. Run Stage A first; on
failure remove all provisional production-source admission, retain evidence outside the repository,
and leave schema 62. Stage A may generate only a distinct provisional schema-63 candidate through
test/evidence admission: production preparer selection and artifact publication/reload remain
disabled, and Stage A bypasses artifact-store/cache realization. Never emit it under schema 62.
Only after Stage A and every inherited gate pass may the implementation advance
`CURRENT_VERSION` to 63 and enable production schema-63 selection, publication/reload, and cache
compatibility. Hand the final diff and evidence to a distinct clean documentation-focused agent.
Do not mark complete before that pass.
```

## Limitations

Production schema 63 selects output-width `VECTOR` or `PARALLEL_VECTOR` only for direct,
same-typed, dense `FLOAT32`/`FLOAT64` Conv2d/Conv3d with width stride and width dilation equal to
one, a preferred species wider than one lane, and at least one full in-bounds width block.
Non-width stride and dilation remain eligible. Scalar code handles borders, worker-range fragments,
and tails. Arrays, native-order segments, and ordered mixed carriers are supported. BF16, external
Conv2d epilogues, mixed element types, non-dense access, short/no-interior-block shapes, width
stride or dilation other than one, non-width SIMD axes, and Conv1d composition remain scalar.
Broad validation remains CPU 0009/CI scope.

## Evidence

Focused audit: current `CpuConv2dEmitter` is scalar-only, decodes output coordinates per ordinal
with `lrem`/`ldiv`, and loads geometry/address facts in its inner body; current `CpuConv3dEmitter`
is scalar-only. `CpuCarrierEmitter` already has direct typed vector array/segment load/store forms,
native segment order, and proved `intAddress` arrays. Current Conv IR keeps extents, offsets,
carriers, and ranges cold.

`/tmp/synaptik-cpu-0008n-spike-20260906a/results/stop.txt` records `BOUNDED_SCALAR_STOP` for
actual generated Conv2d/FLOAT32/all-arrays/interior-exact-width/VECTOR: 64 iterations, 12 warmup
triplets, nine retained symmetric pairs, minimum `61,244,791 ns`, `V/S=0.265215012`, and failed
`V/D=1.775769426 > 1.15`. XML records the same failure. The four-entry manifest covers transient
Conv2d/Conv3d emitters, performance test, and XML. `javap` shows schema-63/VECTOR branches and
vector setup in both transient emitters. This establishes the rejected design only; it neither
proves a branch-wide result nor assigns the gap to an individual operation.

## Implementation notes

Generated Conv2d/Conv3d now uses nested output-coordinate loops with scalar borders/fragments and
output-width vector interiors. Selection, carrier realization, and persisted-artifact compatibility
remain CPU-owned; no shared Prepare, Runtime, public API, dependency, or architecture boundary
changed.

## Completion summary

Stage B semantic/structural evidence at
`/private/tmp/synaptik-cpu-0008n1-stage-b-20260906-owned` passed 48 semantic rows and 24 structural
dossiers. Full performance evidence at
`/private/tmp/synaptik-cpu-0008n1-stage-b-performance-20260906/cpu-conv-simd-performance/full-1788687867286-pid89091`
passed 16 rows, five fresh forks, 1,440 pairs, 5,760 positive finite sides (minimum
`30,135,541 ns`), 160 fork medians, 32 aggregates, and 1,803 verified manifest hashes. Maximum
pair ratios were `V/S=0.525369171498` and `V/D=0.959638121325`; maximum aggregate ratios were
`V/S=0.403144684278` and `V/D=0.932440944620`, within the `0.90` and `1.15` limits.

Production uses `CURRENT_VERSION=63`; eligible direct Conv2d/Conv3d routes select `VECTOR` or
`PARALLEL_VECTOR`, schema-63 artifacts publish/reload, stale schema 62 is rejected, and scalar Conv
retains historical schema 52. The focused integration suite passed 80 tests, CPU Javadoc passed,
backend conformance reported `NO-SOURCE`, and diff checks passed. Javadoc reported 53 pre-existing
missing-`@param` warnings in unrelated records.

The additional full CPU suite was not green: 751 tests yielded 725 passes, 24 intentional opt-in
skips, and two unrelated failures. The historical default-running 0008B F64 NEG DAG case was
semantically successful, but all fork-zero timing attempts were `1.1788` through `1.2251`, above
`1.15`; the unchanged worker-close race failed once and passed in isolation. These require separate
default-timing and worker-lifecycle reliability follow-up. No broad-suite pass is claimed; broad
validation remains CPU 0009/CI scope.

The historical Stage A root at `/private/tmp/synaptik-cpu-0008n1-stage-a-20260906b` remains failed
structural evidence: its `stop.txt` records `decision=BOUNDED_SCALAR_STOP`,
`structural_result=FAIL`, and `stage_b=NOT_RUN_AFTER_DECISIVE_STAGE_A_STOP`. It is explicitly
superseded by the final Stage B `C2_F32_ARRAY` row, which strictly contains the intended Stage A
proof. The passing structural dossier and full performance evidence contain the byte-identical
final vector class SHA-256
`6f5a183436eeeff189b1bb29812f65afe1c99f62872394276c8886e05e063c65` under schema-63 key
`c05914f7b0aa747f48e0acc1a0015ef15157f03b7078e64b60d546d7b11a356f`. Its final performance row
passed five forks, 90 pairs, and 360 positive finite sides with minimum `39,300,417 ns`, maximum
pair `V/S=0.150326995238421` and `V/D=0.896607513278599`, and aggregate
`V/S=0.14074463702810028` and `V/D=0.8497156751688318`. No semantic or performance suite was
rerun during evidence reconciliation.

CPU 0008O is restored as the next Draft task. The historical default-timing and worker-close
lifecycle reliability observations remain separate backlog concerns and do not block 0008N1.

Status: Complete
