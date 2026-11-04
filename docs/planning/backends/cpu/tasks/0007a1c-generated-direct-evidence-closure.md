# Task 0007A1C: Generated/Direct Evidence Closure

## Reader context

This is a closure dossier, not one uninterrupted successful benchmark. It starts with a frozen
comparison whose semantics passed but many timing rows failed, records a stop at the permitted
code-owner boundary, and then incorporates separately planned corrections and a replacement
evidence ledger. A row label names one fixed generated/direct case; a schema number identifies
generated-byte compatibility, not the age of a Tensor value or the current release.

Correct semantics, Class-File shape, and local performance were independent gates. Temporary
directories, hashes, disassembly, and accepted/rejected fork records are provenance for the
assertions below, not evidence recreated by this reader-context addition. The current
code-generation explanation helps decode the member/loop checks; it does not widen those local
timings into a universal parity promise.

Current reading alongside this historical record:

- [CPU generated-code ownership](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit)
  — separates generation-time emitters, emitted typed work, and JVM compilation.
- [CPU artifact compatibility and lifetime](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcache)
  — distinguishes reusable code bytes and loaded artifacts from numerical values and JIT state.

## Status

Complete

## Goal

Close the remaining bounded evidence gaps for every currently generatable portable [CPU](../../../../glossary.md#term-cpu) family by
mapping each [operation](../../../../glossary.md#operation)/family/form to equivalent direct primitive Java, emitted/decompiled
Class-File shape, forbidden-reference checks, and realistic parity evidence.

This is evidence-first corrective work. It distinguishes semantic/structural coverage from
measured performance, does not promise universal parity across algorithms, resources, [layouts](../../../../glossary.md#layout),
JVMs, or machines, and permits production correction only for a demonstrated bounded defect.

## Current inventory and conclusion

The current generator has ten computation categories: [pointwise](../../../../glossary.md#term-pointwise); affine copy; movement ([`PAD`](../../../../glossary.md#padding),
`TILE`, `CONCAT`, `STACK`, `UNFOLD_AXIS`, `UNFOLD2D`, `SLICE_UPDATE`); indexing ([`GATHER`](../../../../glossary.md#gather),
`GATHER_ELEMENTS`, `GATHER_ND`, `ONE_HOT`); scatter (`SCATTER_ELEMENTS`, Gather-compatible
`SCATTER_ADD`, `SCATTER_ND`); fold (`FOLD_AXIS`, `FOLD2D`); ordering (`SORT`, `ARGSORT`, `TOP_K`);
explicit state (`INITIAL_STATE`, [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32) [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout)); `CUM_SUM`/`CUM_PROD`; and ordinary
`MIN`/`MAX`/`ALL`/`ANY`/`SUM`/[`MEAN`](../../../../glossary.md#term-mean)/`PROD` aggregates.

| Category | Retained direct/[decompilation](../../../../glossary.md#term-decompilation) evidence | Remaining gap |
|---|---|---|
| Pointwise | Dense [scalar](../../../../glossary.md#term-scalar) ADD, [Vector](../../../../glossary.md#term-vector) ADD, [fused](../../../../glossary.md#specialization-and-fusion) ADD -> GELU_EXACT -> MUL; ten scalar [activation](../../../../glossary.md#term-activation), four vector activation, scalar-tail member-reference evidence; semantic tests cover all 48 opcodes/admitted types. | General-long, segment/mixed, non-floating vector, and virtual-[mask](../../../../glossary.md#term-boolean-mask) performance are partial; all opcodes need an explicit code-shape mapping. |
| Affine/movement | FLOAT64 [CONTIGUOUS](../../../../glossary.md#term-contiguous), FLOAT32 TILE, [INT32](../../../../glossary.md#data-type--datatype) SLICE_UPDATE direct gates; dense/general Class-File and semantic tests cover affine and all seven movement forms. | PAD, composition, windows, and general-long/segment movement are untimed. |
| Indexing | Dense FLOAT32 GATHER_ELEMENTS and [BOOL](../../../../glossary.md#data-type--datatype) ONE_HOT gates; Class-File tests cover all families, index widths, general [layouts](../../../../glossary.md#layout), and segment/mixed carriers. | GATHER, GATHER_ND, and general-long/segment performance are missing. |
| Scatter | Dense replacement/addition, all three mappings, one general mixed addition, and exact floating-product scratch gates; [schema](../../../../glossary.md#term-schema)-31 tests span reductions/types/index widths. | Add one non-additive general-long reduction so closure does not rely only on FLOAT32 addition outside dense arrays. |
| Fold | Dense FLOAT32 FOLD_AXIS gate; structural/semantic tests cover both families, types, ranges, and mixed/general carriers. | FOLD2D and general-long/segment performance are missing. |
| Ordering | Dense FLOAT32 SORT and TOP_K gates; structural/semantic tests cover all families/types/directions/general carriers. | ARGSORT and general-long/segment performance are missing. |
| Random | Dense FLOAT64/FLOAT32 DROPOUT gates; decompilation covers [heap](../../../../glossary.md#term-heap)/segment INITIAL_STATE and dense/segment/mixed DROPOUT. | General mixed DROPOUT is untimed; two-word INITIAL_STATE is unsuitable for [throughput](../../../../glossary.md#term-throughput) parity. |
| [Scan](../../../../glossary.md#term-cumulative-scan) | Dense FLOAT32 CUM_SUM plus [BFLOAT16](../../../../glossary.md#term-bfloat16) CUM_SUM/CUM_PROD gates; tests cover five types and all modes. | A non-BFLOAT16 general segment reverse/exclusive scan is missing. |
| [Extrema](../../../../glossary.md#saturation-pole-and-extremum)/[Boolean](../../../../glossary.md#data-type--datatype) aggregate | Dense full MIN/MAX for five numeric types and ALL/ANY gates; tests cover all forms/layouts/carriers. | General single-/multi-[axis](../../../../glossary.md#term-axis) forms are untimed. |
| Numerical aggregate | Thirteen dense single-axis gates cover every current kind/type; Class-File examples include dense and general scratch forms. | General [carrier](../../../../glossary.md#term-storage-carrier)/layout and multi-axis forms were deliberately unmeasured. |

Current tests already establish six primitive-array carrier forms plus [`MemorySegment`](../../../../glossary.md#term-memorysegment), mixed
patterns, five pointwise access regimes, dense-int/general-long addressing, partial/empty ranges,
scalar/vector tails, external parallel ranges, multi-output entries, and zero-/scratch-bearing
entries. The closure [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) independent equivalence categories rather than their combinatorial
cross-product.

## [Scope](../../../../glossary.md#term-scope)

### Evidence-first workflow

- Before any production edit, create exactly one fresh directory with
  `mktemp -d /private/tmp/synaptik-cpu-0007a1c-XXXXXXXX`. Retain probe source/classes, generated
  classes, full `javap`, Class-File reports, member references, commands, environment, raw samples,
  summaries, and SHA-256 manifests only there.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [SHA-256 glossary](../../../../glossary.md#term-sha-256))</span>
- Record HEAD, complete dirty paths, Java/JVM/Gradle/OS/architecture/CPU facts, schema 31, and
  checksums for available retained evidence. Preserve every existing uncommitted change.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [Gradle glossary](../../../../glossary.md#term-gradle); [CPU glossary](../../../../glossary.md#term-cpu); [schema glossary](../../../../glossary.md#term-schema))</span>
- Freeze probe and comparator source before timing. A correctness repair must be recorded and
  rehashed before final measurement; cases, thresholds, and comparators cannot change after a
  performance result is seen.
- Produce a machine-readable ledger mapping every current operation/family/form to its semantic
  test owner, retained/new Class-File category, retained/new performance category or explicit<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [owner glossary](../../../../glossary.md#term-owner))</span>
  `STRUCTURAL_ONLY` reason, and represented types/carriers/access/address/range/[vector](../../../../glossary.md#term-vector)/scratch [axes](../../../../glossary.md#term-axis).
- Reuse retained evidence only when its files, checksum, and schema verify. Otherwise replace it or
  record an unresolved gap; prose alone is not executable evidence.<span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum))</span>

### Fixed new [matrix](../../../../glossary.md#term-matrix-meanings)

Each timed case uses at least 262,144 output/visited [scalars](../../../../glossary.md#term-scalar) unless [slice](../../../../glossary.md#slice)-oriented by nature.
[Dimensions](../../../../glossary.md#dimension) may be increased before the probe is frozen to reach the 25 ms [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) floor, never
decreased or simplified.

| ID | Fixed generated case | [Carrier](../../../../glossary.md#term-storage-carrier)/access and direct comparator |
|---|---|---|
| P-SCALAR-GENERAL | [FLOAT32](../../../../glossary.md#term-float32) scalar [fused](../../../../glossary.md#specialization-and-fusion) `DIV -> SIGMOID -> MUL`, `[512,512]` | Mixed array/segment, broadcast input, strided output, `GENERAL_LONG`; one fused scalar comparator with identical odometer/formula/[narrowing](../../../../glossary.md#term-narrowing)/store work. |
| P-[VECTOR](../../../../glossary.md#term-vector)-SEGMENT | preferred-species FLOAT32 `LOG1P -> TANH`, 1,048,576 values | Segment input/output; direct [Vector API](../../../../glossary.md#term-vector-api) loop with identical species, [lane](../../../../glossary.md#term-lane) operators, bound, tail, and segment access. |
| P-INTEGRAL-MIXED | preferred-species [INT64](../../../../glossary.md#data-type--datatype) `ADD -> MAX -> CAST`, 1,048,576 values | Mixed `long[]`/segment; direct `LongVector` loop and tail with identical modular/signed semantics. |
| A-GENERAL | [BFLOAT16](../../../../glossary.md#term-bfloat16) PERMUTE/SLICE affine copy, `[256,32,32]` | Segment to `short[]`, nonzero offsets/non-unit [strides](../../../../glossary.md#term-stride); direct raw-short composed mapping and long odometer. |
| M-[PAD](../../../../glossary.md#padding) | BFLOAT16 PAD, at least 262,144 outputs | General segment-to-array; direct output mapping and identical exact [padding](../../../../glossary.md#padding) bits. |
| M-CONCAT | [INT32](../../../../glossary.md#data-type--datatype) four-[occurrence](../../../../glossary.md#term-occurrence) CONCAT with one repeated input | Mixed unique boundaries; direct occurrence mapping and [canonical](../../../../glossary.md#term-canonical-form) [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). |
| M-STACK | [BOOL](../../../../glossary.md#data-type--datatype) four-occurrence STACK with one repeated input | Mixed canonical byte carriers; direct inserted-[axis](../../../../glossary.md#term-axis)/occurrence mapping. |
| M-UNFOLD-AXIS | FLOAT32 non-unit-step/dilated UNFOLD_AXIS | Segment input, strided [heap](../../../../glossary.md#term-heap) output; direct window mapping and identical addresses. |
| M-UNFOLD2D | [FLOAT64](../../../../glossary.md#term-float64) padded/dilated UNFOLD2D | Mixed general-long; direct [im2col](../../../../glossary.md#term-im2col) mapping and identical padding/carrier work. |
| I-[GATHER](../../../../glossary.md#gather) | FLOAT64 GATHER with INT64 indices | Segment data, heap indices, strided segment output; identical validated direct writer, [validation](../../../../glossary.md#term-validation) excluded on both sides. |
| I-[GATHER-ND](../../../../glossary.md#gather-nd) | FLOAT32 [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-one, tuple-depth-two GATHER_ND with non-scalar suffix | Mixed general-long with INT32 indices; identical tuple/suffix writer. |
| S-GENERAL-MIN | INT64 `SCATTER_ND + MIN`, duplicate tuples/non-scalar suffix | General segment/heap mix; identical range-owned copy-then-update, [target](../../../../glossary.md#term-target) filter, signed MIN, and update order. |
| F-FOLD2D | FLOAT32 padded/dilated overlapping FOLD2D | Mixed general-long; identical output-owned mapping and canonical represented addition. |
| O-ARGSORT | descending stable INT64 ARGSORT, `[4096,64]` | Segment input/strided segment output/scratch; identical stable merge, comparison/tie, slices, and stores. |
| R-[DROPOUT](../../../../glossary.md#dropout--inverted-dropout)-GENERAL | FLOAT32 DROPOUT, 1,048,576 values | Mixed general-long roles; exact V1 mapping, threshold, scaling, [mask](../../../../glossary.md#term-boolean-mask), state, and addresses. |
| C-[SCAN](../../../../glossary.md#term-cumulative-scan)-GENERAL | exclusive reverse INT64 CUM_PROD, `[1024,1024]`, axis one | Offset/strided segments; identical reverse/exclusive slice loop and modular multiplication. |
| X-MIN-MULTI | BFLOAT16 multi-axis MIN with kept dimensions | Segment input/strided `short[]` output; identical domain order, first-[NaN](../../../../glossary.md#term-nan)/[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) policy, raw store. |
| X-ANY-SINGLE | BOOL single-axis ANY | Mixed carriers, zero-stride read/strided output; identical canonical fold/address work. |
| N-[MEAN](../../../../glossary.md#term-mean)-GENERAL | FLOAT32 single-axis MEAN, `[128,2048]` | Segment input/strided heap output/exact scratch; identical exact-sum/rational-[rounding](../../../../glossary.md#term-rounding) state and access. |
| N-PROD-MULTI | BFLOAT16 multi-axis PROD with kept dimensions | Mixed general-long/exact scratch; identical [significand](../../../../glossary.md#term-significand)/[exponent](../../../../glossary.md#term-exponent)/special-value state and one conversion. |

Retained cases cover the remaining dense algorithm categories. `INITIAL_STATE` is
`STRUCTURAL_ONLY`: its fixed two-word work would mostly measure call overhead. Computation-free
internal affine [views](../../../../glossary.md#term-view-meanings) are also structural-only and require no generated timing. No other distinct
current family/form may be omitted without leaving the task incomplete.

### Comparator, anti-[DCE](../../../../glossary.md#term-dead-code-elimination-dce), and measurement rules

- Generated/direct sides use separate bit-identical preallocated inputs, outputs, geometry, and
  scratch, with identical represented types, algorithms, semantic work, carrier/layout/address
  work, ranges, state reset, and output mutation.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout))</span>
- Direct code is ordinary static primitive Java. It cannot call generated/emitter/reference/
  Synaptik helpers, collections/boxing, `BigInteger`, easier sorting/grouping/hashing, or easier
  numerical semantics. Vector cases use equivalent direct Vector API; scalar cases use scalar Java.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [Vector API glossary](../../../../glossary.md#term-vector-api); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Exact numerical/scatter product comparators use identical abstract algorithms and scratch.
  Ordering uses identical stable merge/scratch. Random uses the exact V1 mapping. Generation,
  definition, lowering, preparation, binding, allocation, initialization, validation, and full
  correctness scans are excluded symmetrically from timing.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation))</span>
- Both sides use pre-resolved exact matching [`MethodHandle`](../../../../glossary.md#native-numerical-library-and-instruction-architecture) values; timed work contains no lookup,
  reflection, or <span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>`invokeWithArguments`.
- After every invocation, sample raw bits from every output/state role into a local checksum;
  publish it once per batch to a static volatile sink and verify it after the fork. Sampling is
  identical on both sides. Verify exact outputs, immutable inputs, untouched regions, canaries,
  and scratch invariants before warmup and after every measured sample.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [checksum glossary](../../../../glossary.md#term-checksum); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [immutable glossary](../../../../glossary.md#term-immutable); [warmup glossary](../../../../glossary.md#term-warmup))</span>
- Use deterministic initialization and fixed-[seed](../../../../glossary.md#term-seed) randomized case/side order.
- [Run](../../../../glossary.md#term-run) five isolated [forks](../../../../glossary.md#term-fork-meanings) with `-Xms1g -Xmx1g`, at least five randomized warmup rounds, nine
  randomized measured rounds, and adaptive batches lasting at least 25 ms. Report each fork's
  generated/direct median and ratio plus the median of fork medians; never average cases.<span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>
- Every [fork](../../../../glossary.md#term-fork-meanings) and aggregate for every new case must be `<= 1.15x`. Retained thresholds/results stay
  unchanged. There is no minimum speedup because no known adverse baseline is assumed.
- If algorithm/resource equivalence cannot be preserved or noise prevents a fixed gate, retain
  `Review needed`/`Incomplete`; do not weaken, pool, or simplify the case.
- Measurements are release evidence only and never production configuration, cache identity,
  preparation, route selection, Runtime behavior, or tuning input.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

### Class-File and forbidden-reference gates

- Generate deterministic representatives for every new row and each current family/form not
  distinct there, including all 48 pointwise opcodes mapped by scalar/vector emitted category,
  both scan kinds/four modes, aggregate kinds/forms, scatter reduction categories, both index
  widths, every admitted type, and both scratch signatures.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [scan glossary](../../../../glossary.md#term-cumulative-scan); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Retain `.class`, exact [descriptor](../../../../glossary.md#term-descriptor)/size/[SHA-256](../../../../glossary.md#term-sha-256), `javap -c -p`, `javap -v -p`, constant-pool and
  member-reference inventories, and Java 26 Class-File model reports. Equal specializations must
  regenerate byte-identically.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Each class has zero fields, exactly one static entry, no constructor/secondary method, and the
  exact typed carrier/optional scratch/<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`long[]` geometry/`long start,end` descriptor.
- Reject `Object`, bridges, method-handle/type [constants](../../../../glossary.md#term-constant), `invokedynamic`, dynamic constants,
  bootstraps, reflection, graph/operation/runtime dispatch, collections/boxing, normal hot-path
  allocation, or Model/Compiler/Runtime/Prepare/reference/emitter/generator calls.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [operation glossary](../../../../glossary.md#operation); [runtime glossary](../../../../glossary.md#term-runtime); [dispatch glossary](../../../../glossary.md#term-dispatch); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare))</span>
- All non-pointwise and scalar pointwise classes have no Synaptik-owned references. Vector
  pointwise classes may reference only their exact typed chunk-level `CpuVectorMath` method;
  scalar tails have none.
- Assert matching primitive array opcodes, native-order typed segment `get`/`set`, exact Vector API
  species/operations, and exact scratch presence/access. Use Class-File semantic categories, not
  source searches, pool indices, absolute bytecode offsets, hidden names, or JIT assembly.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api); [bytecode glossary](../../../../glossary.md#term-bytecode); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit))</span>

### Correction and [schema](../../../../glossary.md#term-schema) policy

- No production change is expected. A correction requires a frozen reproducible pre-edit failure,
  retained class/decompilation/timing, a named faulty emitted shape, and a stable test failing first.<span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation); [shape glossary](../../../../glossary.md#shape))</span>
- Correct at most two cohesive code-shaping owners. More independent failures, a third owner, or
  IR/lowering/resource redesign requires stop/replan and separate ordered work.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering))</span>
- If emitted bytes change, increment schema exactly once from 31 to 32. Schema-31 artifacts become
  incompatible safe misses; add no migration, alias, converter, partial reuse, or dual-schema path.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- If emitted bytes do not change, schema remains 31. Tests/evidence/docs/Javadoc never increment it.
  A production change claimed byte-neutral must regenerate every affected category byte-identically.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Stop before changing semantics, thresholds, eligibility, capability, public API, resources,
  materialization, routes, preparation/Runtime, parallel ownership, numerical order, dependencies,
  architecture, Gradle/build, another module, or later work.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [materialization glossary](../../../../glossary.md#materialization); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Universal cross-product/machine/[JVM](../../../../glossary.md#term-java-virtual-machine-jvm)/species/size parity or duplicate reruns of verified evidence.
- Timing [metadata](../../../../glossary.md#term-metadata)-only/zero-work [views](../../../../glossary.md#term-view-meanings), [validation](../../../../glossary.md#term-validation) failures, or two-word INITIAL_STATE [throughput](../../../../glossary.md#term-throughput).
- Easier direct algorithms/resources, weakened gates, toleranced exact results, or using one
  representative for a distinct algorithm.
- Production benchmark tooling, JMH/public APIs, Config, autotuning, tuning caches, runtime
  profiling/selection, or benchmark-driven mutation.<span data-glossary-navigation="true"> ([JMH glossary](../../../../glossary.md#term-java-microbenchmark-harness-jmh); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [runtime profiling glossary](../../../../glossary.md#runtime-profiling))</span>
- New semantics or CPU 0007A2+, native routes, fusion expansion, conformance/integration, other
  modules/backends, architecture/build, commit/push/stage/revert/delete, or unrelated refactoring.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [fusion glossary](../../../../glossary.md#term-fusion); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module); [backends glossary](../../../../glossary.md#backend); [commit glossary](../../../../glossary.md#term-commit))</span>

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

- [Model](../../../../glossary.md#term-model) owns semantics; only [CPU](../../../../glossary.md#term-cpu)-private realization/evidence may change.
- CPU analysis owns lowering/specialization/strategy/declarations before assignment; finalization
  owns generation/reuse afterward; Runtime only executes immutable prepared work.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Generated classes stay typed, direct, allocation-free in normal hot work, and free of mutable
  static run state. Runtime identities/carriers/addresses/slots/measurements stay outside identity.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [run state glossary](../../../../glossary.md#run-state--runstate); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Concurrent runs retain distinct [`RunState`](../../../../glossary.md#run-state--runstate), geometry, [buffers](../../../../glossary.md#term-buffer), and scratch.
- Benchmark evidence may accept/reject/diagnose, never select or mutate production behavior.

## Package impact

Existing packages used:

- `internal.codegen.emit` — family generation, verification, and new test-only closure [matrix](../../../../glossary.md#term-matrix-meanings).
- `internal.cache` — [schema](../../../../glossary.md#term-schema)/[artifact](../../../../glossary.md#term-artifact) compatibility only if emitted bytes change.
- `internal.ir`, `.lowering`, `.prepare`, `.route.portable`, `.executable`, `.reference` — read-only
  inventory and semantic/resource owners; no contract change is allowed.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>

No production package, export, supported [API](../../../../glossary.md#term-application-programming-interface-api), or [module](../../../../glossary.md#term-module) is added/moved. New test
`CpuGeneratedDirectEvidenceClosureTest` belongs beside emitters because it owns cross-family
Class-File evidence, not [lowering](../../../../glossary.md#lowering) or [Runtime](../../../../glossary.md#term-runtime) behavior.

## Affected and allowed files

Expected evidence-only paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedDirectEvidenceClosureTest.java` (new)
- `docs/backend-guide/cpu-backend.md`
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`
- `docs/glossary.md` only if a reusable term changes; expected result is no change.

Optional correction allowlist; at most two code-shaping [owners](../../../../glossary.md#term-owner) may change:

- `CpuClassFileKernelGenerator.java`, `CpuLoopEmitter.java`, `CpuCarrierEmitter.java`,
  `CpuScalarEmitter.java`, `CpuVectorInstructionEmitter.java`, `CpuAffineCopyEmitter.java`,
  `CpuDataMovementEmitter.java`, `CpuIndexingEmitter.java`, `CpuScatterEmitter.java`,
  `CpuFoldEmitter.java`, `CpuOrderingEmitter.java`, `CpuRandomEmitter.java`, `CpuScanEmitter.java`,
  `CpuAggregateEmitter.java`, `CpuExactSumEmitter.java`, or `CpuExactProductEmitter.java` under
  `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/`.

Optional compatibility/[Javadoc](../../../../glossary.md#term-javadoc) consequences are exactly `internal/cache/CpuGeneratorSchema.java`,
`internal/codegen/emit/package-info.java`, and `internal/cache/package-info.java`.

Optional tests are limited to the corrected family's current `Cpu*GeneratedKernelTest.java`, plus
`CpuGeneratedKernelArtifactStoreTest.java`, `CpuKernelSpecializationTest.java`,
`CpuPartitionPreparerTest.java`, and `CpuPartitionFinalizerTest.java` when [schema](../../../../glossary.md#term-schema) changes. No other
source, test, documentation, architecture, build, or [planning](../../../../glossary.md#term-planning) path is allowed.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 18 repository paths:

- 5 production/package paths: no more than two code-shaping [owners](../../../../glossary.md#term-owner), [schema](../../../../glossary.md#term-schema), two package summaries;
- 7 tests: new [matrix](../../../../glossary.md#term-matrix-meanings), no more than two family owners, up to four compatibility owners;
- 6 documentation/[planning](../../../../glossary.md#term-planning) paths; and
- 18 total.

Evidence files are outside the repository. A third [owner](../../../../glossary.md#term-owner), another test owner/[module](../../../../glossary.md#term-module), [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/
resource change, or larger total is a stop/replan condition.

## Acceptance criteria

- The ledger enumerates every current operation/family/form and maps semantic, Class-File, and
  retained/new performance evidence or a justified structural-only result.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- All twenty new cases pass exact semantics and every [fork](../../../../glossary.md#term-fork-meanings)/aggregate `<= 1.15x`.
- Retained evidence verifies by [checksum](../../../../glossary.md#term-checksum)/[schema](../../../../glossary.md#term-schema) or is replaced/classified unresolved.
- Every selected class passes descriptor, member, forbidden construct/reference, scratch,
  determinism, decompilation, and checksum gates.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [decompilation glossary](../../../../glossary.md#term-decompilation))</span>
- Combined evidence spans all six primitive-array carrier types, both index widths, segment/mixed
  carriers, five pointwise regimes, dense/general addressing, scalar/vector/tails, partial/empty
  ranges, multi-output, zero scratch, and exact sum/product scratch.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [pointwise glossary](../../../../glossary.md#term-pointwise); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector))</span>
- Only INITIAL_STATE and computation-free [views](../../../../glossary.md#term-view-meanings) are structural-only absent a new unresolved gap.
- Production changes occur only for a frozen demonstrated failure and stay within two [owners](../../../../glossary.md#term-owner).
- Schema stays 31 if bytes are unchanged, or advances once to 32 with schema-31 safe misses and no
  migration when bytes change.
- Architecture/API/capability/semantics/resources/routes/Prepare/Runtime/dependencies/build/later
  work and unrelated dirty paths remain unchanged.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- A distinct clean documentation context finalizes affected Javadocs/package summaries, guide,
  glossary impact, task evidence/status, master plan, and roadmap with reasoned no-change results.

## Tests / [validation](../../../../glossary.md#term-validation)

After executable Java stabilizes, [run](../../../../glossary.md#term-run) one focused [matrix](../../../../glossary.md#term-matrix-meanings) and one authoritative [module](../../../../glossary.md#term-module) suite:

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

Run the frozen probe outside [JUnit](../../../../glossary.md#term-junit) from the evidence directory. Record exact `javac` and five
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
package/type placement; [schema](../../../../glossary.md#term-schema) policy; 0007A1B Complete -> 0007A1C current status -> corrective
frontier -> 0007A2 Draft; exactly one [Ready](../../../../glossary.md#term-ready) corrective task; no later detailed correction or
0007A2 specification; empty staged diff; preserved dirty paths. Repository/architecture/
[conformance](../../../../glossary.md#term-conformance)/integration validation remains deferred to [CPU](../../../../glossary.md#term-cpu) 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci) because no shared contract
may change; otherwise stop and replan.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0007A0–0007A0F: Complete retained family parity/[decompilation](../../../../glossary.md#term-decompilation) foundation.
- CPU 0007A1: Complete numerical algorithms/scratch/general Class-File and thirteen dense gates.
- CPU 0007A1A: Complete [scalar](../../../../glossary.md#term-scalar) self-containment/member-reference evidence.
- CPU 0007A1B: Complete current [schema](../../../../glossary.md#term-schema) 31 scatter algorithms/six gates/general evidence.
- Current Java 26 generator, specialization/schema/store, every family owner, preparation/
  finalization/execution/reference, and generated-family tests define the inventory.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [owner glossary](../../../../glossary.md#term-owner))</span>

## Follow-up tasks

- CPU 0007A1D retains the useful schema-32 invocation-local segment-layout implementation. Its
  original 13-target performance criterion did not pass. Later bounded corrections, rather than
  A1D alone, supplied the accumulated A1C closure; a subsequent review closed A1D as<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [layout glossary](../../../../glossary.md#layout); [target glossary](../../../../glossary.md#term-target))</span>
  `Superseded`, not as a passing task.
- [CPU](../../../../glossary.md#term-cpu) 0007A1E through [CPU 0007A1N](0007a1n-multi-axis-min-residual-parity.md) are Complete. Their
  accumulated evidence closes every frozen performance row, but the original ledger gate remains
  open because rows other than INITIAL_STATE and computation-free views retain explicit<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings))</span>
  `STRUCTURAL_ONLY:covered-by-emitted-category` classifications.
- [CPU 0007A1O](0007a1o-pointwise-ledger-evidence-reconciliation.md) is Complete. Its bounded
  replacement ledger, pointwise equivalence-category evidence, stable regression test, and final
  re-audit close this task's last original gate.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [regression glossary](../../../../glossary.md#term-regression))</span>
- Planning CPU 0007A2 is the next follow-up. It remains Draft without a detailed specification,
  and there is no Ready CPU task after this closure.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Ready glossary](../../../../glossary.md#term-ready))</span>
- CPU 0009 remains the later repository-wide portable [conformance](../../../../glossary.md#term-conformance) [checkpoint](../../../../glossary.md#term-checkpoint).

## Architecture impact

Expected impact: None. A required architecture/[dependency](../../../../glossary.md#term-dependency)/public/shared/resource/route/[Runtime](../../../../glossary.md#term-runtime)/
tuning change is a stop condition, not implied [scope](../../../../glossary.md#term-scope).

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
- Exact handles and identical output [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) make [dispatch](../../../../glossary.md#term-dispatch)/anti-[DCE](../../../../glossary.md#term-dead-code-elimination-dce) symmetric.
- Fixed two-word INITIAL_STATE and computation-free [views](../../../../glossary.md#term-view-meanings) are structural-only.
- Two correction [owners](../../../../glossary.md#term-owner) is the maximum; broader failure is new ordered work.
- [Schema](../../../../glossary.md#term-schema) changes only for emitted bytes.

## Known limitations

- Ratios apply only to fixed cases, environment, [JVM](../../../../glossary.md#term-java-virtual-machine-jvm), and protocol; never universal or selecting.
- Not every carrier/layout/type/worker/alignment/species/size cross-product is timed; existing
  semantic/Class-File tests cover independent axes.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [alignment glossary](../../../../glossary.md#term-alignment); [axes glossary](../../../../glossary.md#term-axis))</span>
- Entry timing does not separately benchmark worker-group orchestration; range correctness and
  scalar/parallel reuse remain automated lifecycle evidence.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Unverifiable retained evidence must be replaced or unresolved.
- Native/future families/[runtime](../../../../glossary.md#term-runtime) end-to-end/repository [conformance](../../../../glossary.md#term-conformance) remain outside this frontier.
- The first frozen performance [fork](../../../../glossary.md#term-fork-meanings) failed 17 of 20 rows and its original malformed
  `schema-after.txt` remains rejected historical evidence. Later correction bundles and the final
  A1N bundle replace those missing closure results; they do not retroactively treat the malformed
  file or stopped first run as passing evidence.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context inspected governing instructions; architecture, performance, [runtime](../../../../glossary.md#term-runtime)/[prepare](../../../../glossary.md#prepare)/
[backend](../../../../glossary.md#backend), planning, and documentation contracts; complete [CPU](../../../../glossary.md#term-cpu) master plan/roadmap; CPU guide and
glossary; completed 0007A0–0007A1B plans and retained evidence; the current dirty diff; all current
generated family emitters/generator/[specialization](../../../../glossary.md#term-specialization)/[schema](../../../../glossary.md#term-schema)/[IR](../../../../glossary.md#term-intermediate-representation-ir) inventory; and related generated tests.

Source confirms ten categories, six primitive-array forms plus segment, five [pointwise](../../../../glossary.md#term-pointwise) regimes,
dense-int/general-long, [scalar](../../../../glossary.md#term-scalar)/[vector](../../../../glossary.md#term-vector) tails, external parallel ranges, zero/scratch entries, and
schema 31. Tests provide broad semantic/structural coverage, while retained performance is mostly
dense. The fixed twenty-case [matrix](../../../../glossary.md#term-matrix-meanings) closes the identified general/form gaps and keeps
INITIAL_STATE/[views](../../../../glossary.md#term-view-meanings) honestly structural-only.

Planning changed exactly this task, CPU master plan, and roadmap in context
`01a01b6b-fc28-7951-91d5-08c7913b27b9`. Local-link existence passed and the three changed files
contain no local anchor links; fences are balanced, final newlines are present, and trailing
whitespace is absent. CPU 0007A1B is Complete, 0007A1C is the sole [Ready](../../../../glossary.md#term-ready) CPU row, CPU 0007A2 is
Draft, and no 0007A2/later detailed CPU task exists. The staged diff is empty; the pre-existing
dirty work is preserved. `git diff --check` and `git diff --cached --check` pass. No Java test,
[Javadoc](../../../../glossary.md#term-javadoc), generation, [decompilation](../../../../glossary.md#term-decompilation), or benchmark command ran because this is planning-only.

Mandatory documentation/planning context `/root` independently inspected the final test, current
emitters/schema, frozen source, ledger, raw first-[fork](../../../../glossary.md#term-fork-meanings) output, summaries, checksums, generated
classes, and retained `javap` output. `sha256sum -c` passed for all three frozen sources, the
ledger, and all 20 generated classes. An `awk` recount of `fork-1-summary.csv` confirmed three
passing and 17 failing rows with the expected passing names. A targeted `find` confirmed no
fork-two-through-five, aggregate, exhaustive, focused-suite, or Javadoc result file exists.

For the four final planning paths, the local Markdown [target](../../../../glossary.md#term-target) check passed; there are no local
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
A1D targets; forks two through five correctly did not [run](../../../../glossary.md#term-run) and no aggregate exists. A diagnostic
version eliminating native-order/with-order construction entirely still left ten A1D targets
above `1.15x`, proving that [layout](../../../../glossary.md#layout) construction was only one partial cause. Clean documentation/
planning context `/root` retained A1D as Incomplete, preserved every acceptance gate, and selected
only the four-row `CpuDataMovementEmitter` cluster as detailed Ready CPU 0007A1E.

## Implementation notes

- The implementation context created and froze one 20-row semantic/performance probe, its direct
  comparators, the complete operation/family/form ledger, generated representatives, checksums,<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
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
  focused/full Java, or Javadoc gates.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Evidence-driven correction order starts with the shared native-order segment-access bytecode
  shape in CPU 0007A1D. That implementation is retained as a stable prerequisite but did not close
  parity. The next correction is CPU 0007A1E's four-row movement general-address-loop cluster;
  BOOL movement/aggregate and fold/dropout remain later ordered groups.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [BOOL glossary](../../../../glossary.md#data-type--datatype); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- [CPU](../../../../glossary.md#term-cpu) 0007A1I subsequently closes both indexing rows at [schema](../../../../glossary.md#term-schema) 37. Frozen semantics remain
  `VERIFIED,20`; `I-GATHER` and `I-GATHER-ND` pass all five [forks](../../../../glossary.md#term-fork-meanings) with medians `0.853338331x` and
  `0.745954874x`, and all three controls pass. Exactly five rows remain persistent in the fresh
  accepted evidence. CPU 0007A1J is the sole next Ready owner cluster.
- CPU 0007A1J subsequently closes `C-SCAN-GENERAL` at schema 38. Frozen semantics remain
  `VERIFIED,20`; the [scan](../../../../glossary.md#term-cumulative-scan) row passes all five accepted forks with [median](../../../../glossary.md#term-median) `0.978732901x`, and all
  three controls pass. Two complete samples were rejected because one unrelated row exceeded the
  gate in one fork; their data is retained but does not contribute accepted evidence. Exactly four
  rows remain persistent, and CPU 0007A1K is the sole next Ready owner cluster.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- CPU 0007A1K subsequently closes `A-GENERAL` at schema 39. Frozen semantics remain
  `VERIFIED,20`; the affine-copy row passes five accepted forks with median `0.653888342x`, and
  all three controls pass. One complete sample was rejected because an unrelated `M-CONCAT` fork
  exceeded the gate and contributes no accepted evidence. Exactly `P-SCALAR-GENERAL`,
  `S-GENERAL-MIN`, and `X-MIN-MULTI` remain, and CPU 0007A1L is the sole next [Ready](../../../../glossary.md#term-ready) [owner](../../../../glossary.md#term-owner) cluster.
- CPU 0007A1L subsequently closes `P-SCALAR-GENERAL` at schema 40. Frozen semantics remain
  `VERIFIED,20`; the pointwise row and all three controls pass every accepted fork and median,
  leaving exactly <span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise))</span>`S-GENERAL-MIN` and `X-MIN-MULTI`. CPU 0007A1M is the sole next Ready owner
  cluster.
- CPU 0007A1M subsequently closes `S-GENERAL-MIN` at schema 41. Frozen semantics remain
  `VERIFIED,20`; the scatter row passes five accepted forks at `0.984900063x`, `0.988888234x`,
  `0.983823803x`, `0.978065816x`, and `0.992400680x`, with median `0.984900063x`, and all three
  controls pass. The unchanged full probe then left only `X-MIN-MULTI`, which CPU 0007A1N owns.
- CPU 0007A1N closes `X-MIN-MULTI` at schema 42. The unchanged sources and all 14 probe classes
  remain byte-identical to A1M, semantics report `VERIFIED,20`, and five accepted sequential forks
  contain no ratio failure: all 100 row/fork ratios and all 20 medians are at most `1.15x`.
  `X-MIN-MULTI` measures `0.818775794x`, `0.811182115x`, `0.811942840x`, `0.810720356x`, and
  `0.810918826x`, with median `0.811182115x`. One whole sample is retained and rejected because
  unrelated `M-CONCAT` measured `2.537201532x`; it contributes no accepted ratio.
- The final schema-42 bundle retains deterministic generated bytes, complete `javap -c` and
  `javap -v` for all twenty classes, and a target inspection proving descriptor/member/forbidden-
  construct/scratch/fallback/hot-loop requirements. All twenty class files are byte-identical
  across semantic verification, five accepted forks, and the rejected sample. The focused matrix
  passed 62 tests in six suites, and the authoritative CPU suite passed 360 tests in 54 suites
  with zero failures or errors and one expected opt-in persistence-evidence skip.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [descriptor glossary](../../../../glossary.md#term-descriptor); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Original-gate reconciliation: exact semantics, all twenty fixed performance cases, every
  per-fork and aggregate threshold, complete accumulated generated/decompiled structural evidence,
  the required carrier/type/access/range/vector/scratch axes, the two-owner-per-correction boundary,
  schema policy, documentation pass, and repository scope are satisfied. A1C itself changed no
  production owner and remained at schema 31; each separately planned correction advanced its
  schema once, ending at current schema 42.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [vector glossary](../../../../glossary.md#term-vector); [axes glossary](../../../../glossary.md#term-axis); [scope glossary](../../../../glossary.md#term-scope))</span>
- The immutable original ledger's extra structural-only labels remain recorded as historical
  evidence. A1O does not rewrite them; its versioned, checksummed, automatically validated
  replacement closes that final accounting gate with current generated/direct evidence.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Architecture and ADR no-change conclusion: no production owner, lifecycle, module boundary,
  dependency direction, backend ownership, Prepare/Runtime handoff, route, resource, capability,
  or semantic contract changed, so <span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [lifecycle glossary](../../../../glossary.md#lifecycle); [module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [backend ownership glossary](../../../../glossary.md#backend-ownership); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [handoff glossary](../../../../glossary.md#term-handoff))</span>`ARCHITECTURE.md`, focused architecture pages, and ADRs remain
  accurate.
- Test/build/API no-change conclusion: the preserved closure test is the only new Java artifact in
  this context; no dependency, Gradle/toolchain, public API, or executable contract changed, so
  architecture tests, backend conformance, integration tests, build files, and API reference pages
  require no edit. Their suites remain intentionally unexecuted under the stop rule.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [artifact glossary](../../../../glossary.md#term-artifact); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Guide/glossary/Javadoc no-change conclusion: an incomplete local benchmark is planning evidence,
  not a supported backend claim or reusable term. The CPU guide and glossary therefore remain
  unchanged. Current package/type Javadocs still describe schema-31 behavior and unchanged
  execution contracts; Javadoc was not rerun because no Javadoc changed.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- CPU 0007A1O subsequently preserves that immutable original ledger and supplies a versioned,
  repository-tested replacement. Implementation context `01a032f9-66c9-73f1-8960-8c39c97c830d`
  and initial audit context `01a032f1-a956-7ca0-9a18-4ce3f585208b` retained evidence at
  `/private/tmp/synaptik-cpu-0007a1o-IPbQzuJi`. The original verifies at the required SHA-256 and
  79 physical lines; the replacement preserves 48 pointwise plus 30 non-pointwise rows, assigns
  fresh evidence to exactly 40 pointwise representatives, and leaves only <span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256))</span>`INITIAL_STATE`
  structural-only.
- Mandatory clean documentation-focused context `01a0331e-335a-7573-a8b4-7ad481d464d8`
  independently reviewed the final test/resource, retained corpus, actual XML, final diff, and
  governing General and Planning profiles before closing A1C and A1O.
- A1O reports `VERIFIED,44`, five accepted 44-case fixed-[heap](../../../../glossary.md#term-heap) forks, worst [fork](../../../../glossary.md#term-fork-meanings)
  `1.071655171x`, worst median `1.056510985x` (both `P-IS_NAN`), deterministic bytes for all 44
  classes, four controls byte-identical to A1N, complete `javap -c/-v`, `AUDITED,44`, and a
  verified 384-entry manifest. Its truthfulness audit classifies all 48 pointwise rows as 36
  no-operation-branch, five formula-conditional, and seven value-selection cases. The stable test
  fixes exact formula, branch, type/result, realization, value-flow, carrier/access/range, store,
  forbidden-hot-path, provenance, inventory, checksum, and schema expectations.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest); [provenance glossary](../../../../glossary.md#provenance); [checksum glossary](../../../../glossary.md#term-checksum))</span>
- The focused A1O command passed 23 tests. The final CPU XML contains 55 suites and 361 tests: 360
  passed, one expected opt-in persistence test skipped, and zero failures/errors. No executable
  Java changed afterward. Current schema remains 42, production bytes are unchanged, staging is
  empty, and the final worktree contains exactly the six A1O paths.
- Final original-gate reconciliation: the checksummed original plus tested replacement closes the
  complete ledger inventory and structural-only limit; A1N closes all twenty fixed semantic and
  performance rows; retained/fresh evidence and manifests close provenance; deterministic
  Class-File, complete disassembly, descriptors, members, forbidden constructs, and scratch gates
  close generated shape; accumulated evidence spans every required carrier/type/access/range/
  vector/scratch axis; each production correction retained its two-owner/schema boundary; A1O
  changes no production/schema/contract; and the mandatory documentation pass closes planning,
  link, scope, status, staging, and whitespace gates. Every original acceptance criterion now
  passes.
- Final documentation no-change conclusion: this closure adds test evidence only. Production and
  public/semantic contracts are unchanged, so architecture/ADRs/architecture tests, public/API
  Javadocs, glossary, guides/examples, capability, conformance/integration, build, and other-module
  documentation require no change. CPU Javadoc and Java suites were not rerun by the final
  documentation context. At this closure A1D still needed review; its subsequent `Superseded`
  disposition does not waive its historical failed local 13-target performance criterion.

## Completion summary

- Completed changes: froze the original ledger and twenty-case corpus, diagnosed and corrected the
  bounded generated-code defects through A1N, preserved the historical evidence, and used A1O's
  tested replacement to close the final pointwise evidence-accounting gap.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Files changed or created in the final A1O closure: the ledger test/resource plus A1O, this task,
  CPU master plan, and roadmap; exactly six worktree paths. The documentation pass changed no
  executable Java or test.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Tests and [validation](../../../../glossary.md#term-validation): accumulated `VERIFIED,20` and accepted twenty-row A1N evidence plus A1O
  `VERIFIED,44`, `AUDITED,44`, five accepted 44-case forks, deterministic/control checks, complete
  disassembly, the 384-entry checksum manifest, 23 focused tests, and final 55-suite/361-test CPU
  XML with 360 passes and one expected skip. Final documentation/static checks passed.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum); [manifest glossary](../../../../glossary.md#term-manifest))</span>
- Documentation-agent review: the mandatory clean documentation-focused A1O finalization reviewed
  the final diff, test/resource, retained evidence, XML, architecture/planning contracts, and
  Planning profile, then finalized only the four permitted planning paths.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: A1C and A1O are Complete; CPU master plan and roadmap now identify Draft
  CPU 0007A2 planning as the next follow-up and show no Ready CPU task.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Architecture/API/Javadoc/glossary/guide/capability/build/conformance/integration/other-module
  impact: no change because A1O adds test evidence only and changes no production/public/semantic
  contract.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module))</span>
- Unresolved issues: None for A1C. CPU 0007A1D's own historical local performance criterion
  failed; subsequent review closed it as `Superseded`, not as a passing result.
- Follow-up required: plan CPU 0007A2 in a later change; its detailed specification is not created
  within A1O's fully occupied six-path scope.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

Status: Complete
