# Task 0007A0F: Random and Dropout Generated-Loop Parity

## Status

Complete

## Goal

Replace the bridge-only generated work for the completed CPU explicit-state random family with
carrier-, represented-type-, family-, access-, state-, value-, mask-, and loop-specialized bodies
embedded in each generated class. `INITIAL_STATE` and FLOAT64/FLOAT32 `DROPOUT` must preserve
every CPU 0006D semantic, carrier, layout, range, replay, resource, validation, parallelism, and
lifecycle contract while independently gated dense heap-array FLOAT64 and FLOAT32 dropout cases
reach reproducible near-parity with equivalent direct primitive implementations.

This is one cohesive CPU-private generated-code correction. Initializer and next-state prologues,
the CPU-private V1 counter mapping, keep-mask production, finite-precision scaling, and five-
boundary invocation share `CpuRandomIr`, `CpuRandomEmitter`, one generated artifact compatibility
boundary, and the same explicit-state replay contract. The task changes generated code shape only;
completed CPU 0006D capability, semantics, state representation, mapping, route, and observable
behavior remain unchanged.

## Scope

### Mandatory pre-edit source and Class-File baseline

- Before changing production Java, create one fresh isolated directory under `/private/tmp` and
  retain the current schema-27 source, generated classes, semantic checks, and raw timing for the
  two exact performance cases below.
- Also retain representative current classes for heap-array INITIAL_STATE, all-`MemorySegment`
  FLOAT64 DROPOUT, and mixed-carrier non-dense FLOAT32 DROPOUT so the baseline covers the zero-
  input initializer, three writable output roles, both value types, every carrier category, and
  both addressing categories.
- Retain probe and summarizer source; exact compile, generation, definition, decompilation,
  execution, and summary commands; complete environment facts; generated class bytes; complete
  `javap -c -p` and `javap -v -p`; entry descriptors; class sizes; SHA-256 values; constant-pool
  member references; semantic verification; raw samples; per-fork summaries; and aggregate
  summaries.
- The planning source baseline is:
  - heap FLOAT64 DROPOUT entry descriptor `([D[J[D[B[J[JJJ)V`;
  - heap FLOAT32 DROPOUT entry descriptor `([F[J[F[B[J[JJJ)V`;
  - heap INITIAL_STATE entry descriptor `([J[JJJ)V`;
  - all-segment DROPOUT entry descriptor
    `(Ljava/lang/foreign/MemorySegment;Ljava/lang/foreign/MemorySegment;`
    `Ljava/lang/foreign/MemorySegment;Ljava/lang/foreign/MemorySegment;`
    `Ljava/lang/foreign/MemorySegment;[JJJ)V`, and segment INITIAL_STATE descriptor
    `(Ljava/lang/foreign/MemorySegment;[JJJ)V`;
  - generated `CpuRandomEmitter.emit` loads the one or five carrier parameters, fills absent
    positions with null, and calls
    `CpuRandomEmitter.execute(Object,Object,Object,Object,Object,long[],long,long,int,long,long,long)`;
  - the bridge resolves initializer versus dropout from a family ordinal, resolves the dropout
    value type through `DataType.values()`, tests every carrier with `instanceof`, and calls
    generic state/logical-address and read/write helpers during generated work; and
  - the state prologue and element ranges are separate invocations, with the `[0,0)` prologue
    writing initializer or next state and non-empty dropout ranges writing value and mask only.
- Treat those observations as planning evidence, not retained implementation evidence. Reproduce
  the exact current descriptors, bytes, hashes, member references, semantics, and timing freshly
  before any production edit. If the fresh baseline differs materially, stop and update this
  specification before implementation.
- Time only the warmed state prologue plus complete element-range mapping, threshold, value, and
  mask work. Exclude lowering, generation, verification, class definition, artifact lookup,
  preparation/finalization, cold binding, geometry packing, carrier allocation, input/output
  initialization, expected-result construction, verification, summarization, and sink observation
  from both generated and direct timed forms.

### Embedded typed initializer and dropout bodies

- Change `CpuClassFileKernelGenerator` and `CpuRandomEmitter` so every generated entry embeds the
  exact state prologue and, for dropout, the counter mapping, uniform conversion, threshold,
  finite-precision scaling, canonical mask store, and value store. It must not call
  `CpuRandomEmitter.execute` or an equivalent generic helper bridge.
- Pass the existing structural random IR to the focused emitter. Select at generation time:
  - family: `INITIAL_STATE` or `DROPOUT`;
  - value type: INT64 initializer state, FLOAT64 dropout, or FLOAT32 dropout;
  - ordered boundary role and direct carrier form for state input/output, value input/output, and
    canonical BOOL mask independently;
  - baked initializer words or baked dropout probability raw bits and the unchanged V1 numeric,
    mask, and state policies;
  - proved dense heap-array integer addressing or typed general long addressing; and
  - prologue-only versus non-empty element-range control flow.
- Preserve the universal primitive `long start, long end` entry. `[0,0)` remains the sole state
  prologue sentinel. Every non-empty dropout range uses global row-major logical ordinals, and
  arbitrary legal disjoint ranges remain valid without changing replay.
- Hoist probability, denominator, state key/counter, key offset, element count, boundary ranks,
  packed-layout positions, bases, and other invocation-invariant values into primitive locals
  before the element loop. Compute `keyOffset(key)` once per non-empty generated invocation, not
  per element. Do not read or write next state from non-empty range calls.
- Generated hot bodies must contain no `Object` descriptor or cast, `DataType.values`, family/type/
  carrier ordinal dispatch, array-versus-segment test, reflection, map/string dispatch, graph/
  Runtime/backend/route/cache lookup, boxing, object array, or per-invocation/per-range/per-element
  allocation.
- Eliminate avoidable hot helper calls. Dense heap-array classes should have no method reference.
  General `MemorySegment` forms may reference only the exact predefined native-order primitive
  layout fields and matching primitive `MemorySegment.get`/`set` methods. Any additional primitive
  bit conversion must be closed, justified in evidence, and admitted by the class-specific member-
  reference allowlist. Repeated `ValueLayout` or byte-order construction is forbidden.

### Exact CPU 0006D semantics

- Preserve `SYNAPTIK_CPU_SPLITMIX64_COUNTER_V1` exactly: constants, modulo Java `long` arithmetic,
  key translation, `mix64`, `word(key,counter,i)`, top-53-bit binary64 conversion, and keep rule
  `uniform53(word) >= probability` remain unchanged.
- Logical element `i` remains the global row-major ordinal, independent of physical layout,
  carrier, range, worker, and scheduling. Exactly one logical draw is associated with every value
  element, including probability positive or negative zero and dropped exceptional inputs.
- INITIAL_STATE writes raw key then counter bits to logical lanes zero and one and performs no
  draw or advancement. DROPOUT retains the input key and writes `counter + N` modulo `2^64`
  exactly once in the prologue, including empty `N == 0` execution.
- Keep-mask storage remains canonical BOOL byte `1` or `0`. A dropped value remains represented
  positive zero even for negative zero, NaN, or infinity input.
- Compute `denominator = 1.0d - probability` once per non-empty invocation. FLOAT64 performs one
  binary64 division. FLOAT32 widens the exact binary32 input, performs the same binary64 division,
  and narrows once. Do not substitute reciprocal multiplication, fused arithmetic, probability or
  draw narrowing, vector math, or a changed evaluation order.
- Preserve the five independent V1 vectors and exact threshold-boundary cases from CPU 0006D.
  Tests must derive expected words, uniforms, masks, scaled values, and next state independently
  of production emitter helpers.

### Dense and general carrier/layout forms

- Dense heap-array dropout forms use `DENSE_HEAP_ARRAY_INT`. Narrow universal bounds, bases,
  element count, state lanes, and every proved array address once before hot work; keep primitive
  integer value/mask address state without repeated `l2i`, exact-arithmetic helpers, rank
  decisions, division, or modulo when dense geometry makes them unnecessary.
- Generate typed `GENERAL_LONG` forms for every supported `MemorySegment`, mixed-carrier,
  arbitrary non-negative layout, zero-stride value input, offset/strided injective output, large
  range, or otherwise unproved case. Retain only the checked long coordinate/address arithmetic
  required by each selected boundary mapping.
- A failed dense proof selects the general typed form and never narrows CPU 0006D support. The
  value input, state input, value output, mask output, and next-state output independently use
  their matching primitive array or native-order `MemorySegment` carrier.
- Exhaust the 32 ordered array/segment carrier patterns for each dropout value type in stable
  descriptor and bridge-rejection assertions. Semantic execution must cover all-array, all-
  segment, and mixed patterns that independently toggle each of the five boundary roles, plus
  dense, offset, positive-strided, zero-stride-read, and injective non-dense-write layouts.
- Scalar and parallel-scalar prepared plans reuse the same compatible scalar artifact. Generated
  classes retain no mutable state; all counter, address, and loop state is invocation-private.

### Completed resources, validation, replay, and lifecycle

- Preserve exactly one lowered unit and one generated artifact. INITIAL_STATE boundaries remain
  `[stateOutput]`; DROPOUT boundaries remain `[value,state,output,keepMask,nextState]` with the
  first two reads and last three writes. Preserve zero workspace, zero materialization, and scalar
  or parallel-scalar strategy.
- Preserve the one generated prologue before any worker submission. Empty dropout runs only the
  prologue and submits no element work. Non-empty scalar execution invokes `[0,N)` after the
  prologue; parallel execution invokes deterministic disjoint global-ordinal ranges afterward.
- Preserve complete physical overlap rejection for every dropout input/output and output/output
  pair before prologue, mutation, generated invocation, or worker submission. Preserve permitted
  input/input overlap and INITIAL_STATE's complete writable-output validation.
- Preserve carrier type, size, byte alignment, accessibility, read-only/writability, complete
  referenced-span, assignment, and range validation. Preserve distinct state lanes, writable
  layout injectivity, arbitrary legal disjoint ranges, deterministic replay, counter wrap,
  concurrent-run isolation, and failure timing.
- Preserve scalar Shape, zero extent, singleton, probability boundary, signed-zero probability
  and values, finite values, NaNs, infinities, counter wrap, every legal layout/carrier pattern,
  and bitwise scalar/parallel parity.
- Preserve capability, lowering, route, specialization budget, preparation, finalization,
  executable binding, reference, optional persistence, and immutable prepared ownership unless an
  allowlisted Javadoc or package summary must be finalized for the embedded body.
- CPU analysis remains deterministic and measurement-free. Finalization generates or reuses the
  selected class only after shared slot assignment. Runtime receives an immutable prepared
  executable and performs no random interpretation, generation, specialization, lookup,
  fallback, tuning, or benchmark-driven mutation.

### Structural identity and concrete invocation state

- Structural identity continues to include family, CPU-private generator/configuration constants
  and policy IDs, value and boundary types, ordered roles/access plans, baked initializer words or
  probability raw bits, finite-precision/mask/state policy, carrier pattern, scalar compute mode,
  zero-scratch entry shape, and dense-versus-general emitted code shape.
- Concrete dropout key/counter state, extents, element count, offsets, stride magnitudes, carrier-
  relative bases, carrier instances, slots, addresses, ranges/chunking, workers, graph/run
  identity, and artifact root remain cold invocation or lifecycle facts. They must not enter class
  identity merely because the emitted body consumes them.
- Compatible concrete geometries and states that have the same structural facts must reuse one
  class. Any fact that changes emitted bytes or method descriptor must not alias.

### Generated compatibility transition

- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly from `27` to `28`. Schema 28 is the first
  compatibility version whose random classes embed typed initializer/state, mapping, threshold,
  value, and mask bodies.
- Treat schema 27 as an incompatible safe miss. Extend the artifact-store regression with an
  otherwise valid schema-27 envelope and current metadata/class bytes, prove it is not loaded,
  then prove lookup regenerates and publishes schema-28 bytes. Add no migration reader,
  converter, compatibility alias, or legacy bridge.
- Preserve current-only persistence, deterministic class identity, shape-polymorphic reuse,
  single-flight behavior, strong prepared ownership, and corruption rejection.

### Stable semantic and Class-File evidence matrix

- Add Java Class-File model assertions for INITIAL_STATE and both dropout types; both probability
  zero signs, exact threshold neighbors, ordinary probability, and counter wrap; all 32 ordered
  carrier patterns per dropout type; dense integer and general long forms; representative offset,
  strided, zero-stride, empty, scalar, partial-range, and scalar/parallel artifact reuse forms.
- Assert exact direct descriptors, embedded prologue/mix/uniform/threshold/value/mask loops,
  ordered three-output stores, state isolation, and dense/general addressing. Reject
  `CpuRandomEmitter.execute` and equivalent bridges, `Object` descriptors/casts,
  `DataType.values`, generic family/type/carrier dispatch, allocations, boxing, reflection,
  random services/classes, map/string dispatch, graph/Runtime/backend/cache references, and
  avoidable helpers through Java Class-File model inspection rather than source matching.
- Use a closed explicit member-reference allowlist for every representative class. Dense classes
  reject every member reference. Segment forms admit only their exact predefined layout fields and
  matching typed `MemorySegment.get`/`set` calls, plus a separately justified closed primitive
  conversion if direct emission is impossible.
- Retain complete `javap` as audit evidence only. Durable gates use the Java Class-File API and
  stable structural assertions, not absolute bytecode offsets, planning-observed sizes, hashes,
  or source-text tests.
- Preserve independent semantic differentials against `CpuScalarReferenceKernel` or a test-local
  oracle that does not call or share the generated emitter mapping implementation. Cover the five
  CPU 0006D vectors, raw probability bits, threshold equality/successor, every state word,
  FLOAT64/FLOAT32 scaling order, raw input/output bits, canonical masks, modulo next state, general
  layouts, all carrier roles, partial/disjoint ranges, empty/scalar/singleton cases, complete
  overlap/failure timing, allowed input alias, and scalar/parallel replay.

### Independent fixed performance cases

Use two independent dense heap-array cases. Each uses shape `[64,16384]` (1,048,576 logical
elements), the complete logical range `[0,1048576)`, probability `0.25d`, key
`0x0123456789abcdef`, and counter `0xfedcba9876543210`. Both generated and direct forms execute
one next-state prologue followed by one complete element range and write value, canonical mask,
and next-state outputs.

Initialize input element `i` once per fork with `selector = i % 257`:

- selector `0` is a quiet NaN whose sign alternates by `(i / 257) & 1`; FLOAT64 uses raw bits
  `0x7ff8000000000000L | ((i + 1) & 0x0007ffffffffffffL)` plus that sign bit, and FLOAT32 uses
  `0x7fc00000 | ((i + 1) & 0x003fffff)` plus that sign bit;
- selector `1` is represented negative zero, selector `2` is positive infinity, and selector `3`
  is negative infinity; and
- every other value is `((i * 37L + 19L) % 4093L - 2046L) * 0.125`, rounded once to the case's
  represented type.

The FLOAT64 case uses `double[]` value input/output, `long[]` state/next-state, and `byte[]` mask.
The FLOAT32 case uses the matching `float[]` value input/output with the same other carrier roles.
Verify unchanged input/state, exact raw output bits, canonical mask bytes, and exact next-state
words before and after timing.

The direct body must use only primitive locals and arrays and must implement the exact same V1
key offset, per-global-ordinal word mapping, top-53 conversion, binary64 threshold, denominator,
FLOAT64 division or FLOAT32 widen/divide/narrow-once order, canonical mask, and modulo prologue.
It must not call production random/emitter/reference helpers, precompute random words or masks,
omit any output, fuse away the prologue, use another generator, use reciprocal multiplication,
or exploit fixed input values.

### Reproducible performance protocol and gate

- Use one standalone direct-Java probe outside Gradle and JUnit. Run exactly five fresh JVM forks
  with fixed `-Xms1g -Xmx1g`. Each fork runs at least five warmup batches and nine measurement
  rounds per case, with adaptive repetitions making every timed sample at least 25 ms.
- Randomize case order and generated/direct order within every batch with deterministic seed
  `0x5A17D00D7A0F28L`, reset identically in each fork. Consume an observable sink only outside
  timed regions. Report both cases separately.
- Report generated/direct medians and ratio for every fork. For each case compute the aggregate as
  the median of its generated per-fork medians divided by the median of its direct per-fork
  medians. Every FLOAT64 fork ratio and its aggregate must be `<= 1.15x`; every FLOAT32 fork ratio
  and its aggregate must independently be `<= 1.15x`.
- Never average forks, drop an outlier, let one type or aggregate hide a failing fork, widen the
  threshold, change the fixed state/probability/shape/data/protocol after baseline, or replace the
  equivalent direct body.
- Use identical probe/summarizer source, commands, direct bodies, inputs, heap, seed, warmup/
  round/fork/adaptive protocol, verification, and sink rules for baseline and final evidence.
  Record source SHA-256 and generated-class SHA-256 for both phases.
- Performance evidence is observational and must not select or mutate ordinary preparation or
  Runtime behavior. If either type has one failing fork or aggregate, keep status `In progress` or
  `Review needed`, inspect generated and just-in-time compiled shape, and correct implementation.
- Initializer, general layouts, segment/mixed carriers, partial ranges, empty cases, and parallel
  orchestration remain semantic/Class-File gates rather than broader timing claims.

## Out of scope

- New Model semantics, generator algorithms or versions, operations, attributes, types, Shapes,
  probability rules, capability rows, public APIs, gradients, or cross-backend/cross-version
  bitstream promises
- BFLOAT16/FLOAT16/integral/BOOL dropout values, vector random execution, other distributions,
  public RNG configuration, cryptographic/statistical certification, fusion, fixed-shape
  specialization, unrolling, or relaxed numerical policy
- New routes, workspaces, materialization, replay/random buffers, mutable generator state,
  representation policy, tuning controls/cache behavior, public configuration, or Runtime mutation
- Pointwise, affine/movement, indexing, scatter, fold, ordering, scan, aggregate, reduction,
  normalization, linear algebra, native-provider, or later semantic-family implementation
- Architecture, module/dependency, Model/Compiler/Planning/shared Prepare/Runtime, Config, Trace,
  Engine, other backend, NN, Training, Gradle/build, Java version, architecture-test, backend-
  conformance, integration-test, or unrelated documentation changes
- Detailed specification or implementation for CPU 0007A1 or any later task
- Object bridges, runtime family/type/carrier dispatch, reflection, boxing, hot allocation,
  generic helper execution, hidden materialization, or unsupported scope in generated hot bodies
- Commit, push, staging, revert, deletion, or modification of unrelated concurrent work

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU 0006D explicit-state RNG and dropout](0006d-portable-explicit-state-rng-and-dropout.md)
- [CPU 0007A0 generated hot-path correction](0007a0-generated-hot-path-parity-correction.md)
- [CPU 0007A0A affine/movement parity](0007a0a-affine-and-movement-generated-loop-parity.md)
- [CPU 0007A0B indexing parity](0007a0b-indexing-generated-loop-parity.md)
- [CPU 0007A0C scatter parity](0007a0c-scatter-generated-loop-parity.md)
- [CPU 0007A0D fold parity](0007a0d-fold-generated-loop-parity.md)
- [CPU 0007A0E ordering parity](0007a0e-ordering-generated-loop-parity.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- Model owns raw state meaning, ordered roles, probability domain, logical draw count, keep rule,
  ideal inverted-dropout meaning, and portable state advancement. CPU owns only its private V1
  realization, structural IR, generated code, route, resources, and invocation.
- Planning selects CPU ownership only. CPU analysis deterministically lowers and selects the
  supported route and declares exact buffers before CPU-blind assignment; CPU finalization
  generates or reuses the exact artifact only after assignments exist.
- Runtime hot execution receives direct typed carrier references and primitive geometry only. It
  performs no Operation/CompiledNode or graph inspection, map/string lookup, reflection, backend
  discovery, route selection, cache work, random service access, or generator mutation.
- Prepared recipes remain immutable and reusable. Explicit state plus logical ordinal completely
  determines every draw, and concurrent runs keep invocation-private state under distinct
  `RunState` values.
- Work remains inside `backends/cpu` plus its existing CPU guide, glossary, and planning records;
  it adds no dependency, package, public type, resource kind, or architecture rule.
- Capability remains no broader than the completed CPU 0006D matrix. Stop if direct generation
  requires a Model/compiler/shared Prepare/Runtime change, new architecture, changed semantics or
  state mapping, another route/resource, or any non-allowlisted repository path.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — focused direct random Class-File
  emission and generator dispatch.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — current-only schema and structural
  specialization identity.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — exact random family and policy identity.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — immutable selected random
  realization passed to generation.
- Existing `.lowering`, `.prepare`, `.executable`, `.memory`, and `.reference` packages remain the
  unchanged geometry, lifecycle, binding, and independent-oracle owners.

Packages added or changed:

- No package is added, removed, moved, exported, or made supported API. Only existing CPU-private
  packages may receive focused implementation/Javadoc changes from the allowlist.

Type placement:

- `CpuRandomEmitter` remains the sole focused owner of typed generated initializer and dropout
  bytecode. It must no longer own or expose a generic execution bridge used by generated classes.
- `CpuClassFileKernelGenerator` continues only to dispatch the already identified random IR to the
  focused emitter.
- `CpuRandomIr` remains the structural semantic/code-shaping identity. `CpuRandomLowering.Geometry`
  remains cold concrete invocation geometry and must not be copied into structural identity.
- Add no random utility, service, registry, manager, facade, interpreter, or new public type.

## Affected files

Exact implementation allowlist:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuRandomEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuRandomIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuRandomGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuRandomIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0007a0f-random-and-dropout-generated-loop-parity.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Existing `CpuRandomLowering`, `CpuPartitionPreparationPlan`, `CpuPreparedExecutable` production
code, `CpuScalarReferenceKernel`, capability code, their other focused tests, build files,
architecture documents, shared modules, and completed task records are inspection-only. If a
production change to an inspection-only owner is required, stop and replan rather than exchanging
an allowlisted path silently.

## Maximum scope

This task may modify at most 22 repository paths, all drawn from the exact allowlist above. It may
create no production or test type and no later task specification. Temporary probe/evidence files
must remain inside the one fresh isolated `/private/tmp` directory and are not repository paths.

If implementation requires any non-allowlisted repository path, more than 22 changed paths, a new
type/package, a shared/public contract, or broader semantics, stop before editing and request a
planning update. Do not use a follow-up to hide an incomplete random family or failing parity gate.

## Acceptance criteria

- Fresh schema-27 source, initializer/dense/general generated classes, semantics, and both fixed
  timing cases are retained before production edits with exact source, commands, environment,
  descriptors, bytes, SHA-256, complete `javap`, member references, raw samples, and summaries.
- Every admitted INITIAL_STATE and FLOAT64/FLOAT32 DROPOUT specialization embeds typed state,
  mapping, threshold, value, and mask work without a generic bridge or runtime dispatch.
- Dense heap-array forms use cold-proved primitive integer state; every admitted segment, mixed-
  carrier, arbitrary-layout, zero-stride-read, strided-output, large-range, or unproved form
  retains correct typed general-long execution.
- V1 vectors, uniform/threshold rules, raw initializer/state words, global logical draws, modulo
  counter advancement, FLOAT64/FLOAT32 numerical order, canonical masks, empty/scalar behavior,
  arbitrary ranges, and scalar/parallel replay remain independently proved.
- Existing capability, exact ordered declarations, five-boundary/three-output binding, one-time
  prologue, zero workspace/materialization, complete overlap validation, worker ownership,
  preparation/finalization, immutable ownership, and optional persistence remain unchanged.
- Structural identity is distinct from concrete invocation geometry/state exactly as specified;
  compatible states/geometries reuse one class and byte-changing facts do not alias.
- Schema advances exactly from 27 to 28; schema-27 persistence is an incompatible safe miss that
  regenerates schema-28 bytes; no migration or legacy bridge is added.
- Stable Java Class-File assertions cover the required family/type/probability/carrier/access/
  layout/range/lifecycle matrix and reject Object/generic dispatch, allocation, forbidden lookup,
  random services, generic bridges, and avoidable helper references.
- Every FLOAT64 fork and its aggregate independently pass `<= 1.15x`; every FLOAT32 fork and its
  aggregate independently pass `<= 1.15x`, using unchanged equivalent direct bodies and exact
  pre/post verification.
- Baseline and final evidence remain under one fresh isolated `/private/tmp` directory and contain
  every required source, command, environment fact, class file, decompilation, checksum, member-
  reference report, semantic result, raw timing sample, and summary.
- Only allowlisted paths change, no more than 22 paths change, no GSD artifact exists, the index
  remains unstaged, and no CPU 0007A1 or later detailed task specification exists.
- A distinct clean documentation-focused context independently reviews final source, tests,
  Class-File evidence, and timing evidence; finalizes affected Javadocs/package summaries, CPU
  guide, glossary impact, this task, master plan, and roadmap; and records reasoned no-change
  conclusions before completion.

## Tests / validation

Run focused tests for every changed test owner while iterating. The final focused command must name
every changed test class and cover direct Class-File shape, initializer and both dropout types,
five V1 vectors, probability and threshold boundaries, dense/general forms, every ordered carrier
pattern, all-segment and role-toggled mixed carriers, layouts/ranges, raw special values, exact
state/mask/value semantics, overlap timing, input alias, scalar/parallel replay, preparation/
finalization/executable lifecycle, specialization, and schema/persistence compatibility.

After executable Java stabilizes, run exactly one authoritative affected-module command:

```bash
./gradlew :backends:cpu:test
```

Run the isolated baseline/final probe separately from Gradle/JUnit with the exact five-fork,
two-case protocol. Timing is retained observational evidence, not an ordinary unit test.

The distinct clean documentation pass reuses successful Java and timing evidence unless it
changes executable Java/probe behavior or records a concrete stale-evidence reason. It runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also inspects rendered affected Javadocs and validates local Markdown links and heading
anchors, balanced fences, terminal newlines, carriage returns, trailing whitespace, exact changed-
path allowlist/count, package/type placement, schema-28 history and schema-27 rejection, generated
descriptors/member-reference allowlists, retained baseline/final source/checksums/protocol/ratios,
task/master/roadmap status and dependency order, absence of later specs, preservation of
concurrent scope, unstaged index, and absence of GSD artifacts.

Repository-wide, architecture, backend-conformance, and integration validation remains deferred
to CPU 0009 or continuous integration because this task changes one concrete backend realization
and no shared or architecture contract. Stop and replan if implementation makes that conclusion
false.

## Dependencies

- [CPU 0006D](0006d-portable-explicit-state-rng-and-dropout.md) is `Complete` and owns the exact
  random/dropout semantic, generator, type, boundary, layout, carrier, range, resource, replay,
  validation, reference, worker, lifecycle, and schema-19 history preserved here.
- [CPU 0007A0](0007a0-generated-hot-path-parity-correction.md) is `Complete` and supplies shared
  typed carrier emission, dense integer proof, stable Class-File testing, and the retained five-
  fork performance-evidence pattern.
- [CPU 0007A0A](0007a0a-affine-and-movement-generated-loop-parity.md),
  [0007A0B](0007a0b-indexing-generated-loop-parity.md),
  [0007A0C](0007a0c-scatter-generated-loop-parity.md), and
  [0007A0D](0007a0d-fold-generated-loop-parity.md) are `Complete` and establish direct typed
  dense/general body, carrier, state, scratch, and schema-transition patterns.
- [CPU 0007A0E](0007a0e-ordering-generated-loop-parity.md) is `Complete` and establishes current
  schema 27, closed member-reference evidence, per-fork plus aggregate gates, and the latest
  implementation/documentation evidence pattern.
- Current Java 26 Class-File API, shared Prepare/Runtime, CPU worker, direct carrier binding, and
  generated-artifact store contracts are complete and unchanged.

## Follow-up tasks

- CPU 0007A1 is the sole next ordered `Ready` semantic-family task and explicitly depends on
  completion of CPU 0007A0F. It remains without a detailed specification.
- CPU 0009 retains the generated-coverage and conformance checkpoint, including the intentional
  BFLOAT16 dropout and cross-backend bitstream boundaries.

## Architecture impact

Expected impact: None.

The task changes only CPU-owned generated random code shape and current-only artifact compatibility
under the existing analysis/finalization/binding lifecycle. If implementation requires an
architecture, public/shared contract, module, dependency, semantic, generator, route, resource,
materialization, or Runtime-policy change, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Implement Synaptik CPU task 0007A0F exactly from
docs/planning/backends/cpu/tasks/0007a0f-random-and-dropout-generated-loop-parity.md. Do not use
GSD. Do not commit, push, stage, revert, delete, or modify unrelated work.

Read in full AGENTS.md, ARCHITECTURE.md, the directly relevant architecture/performance/Runtime-
Prepare-Backend documents, documentation rules and applicable profiles, planning guide, roadmap,
CPU master plan, task 0007A0F, completed tasks 0006D and 0007A0 through 0007A0E, and every affected
or directly relevant random IR/lowering/generator/emitter/specialization/route/preparation/
finalization/executable/reference/test/cache/schema/build/guide/Javadoc file. Inspect the complete
worktree and index first and preserve concurrent unrelated changes.

Capture the required fresh schema-27 source/Class-File/semantic/performance baseline under one
isolated /private/tmp directory before production edits. Implement exact typed dense/general
INITIAL_STATE and FLOAT64/FLOAT32 DROPOUT bodies, schema-28 compatibility, stable Class-File and
semantic tests, focused validation, one authoritative CPU suite, and both fixed five-fork per-
fork/aggregate parity gates. Stop on any architecture, scope, semantic, generator, shared-
contract, route, resource, lifecycle, or unresolved performance conflict.

After executable Java and probe evidence stabilize, hand the same uncommitted diff and exact
evidence to a distinct clean documentation-focused context following
docs/developer-guide/documentation-rules.md. That context must independently inspect final source,
tests, generated class shape, and evidence; finalize affected Javadocs/package summaries, CPU
guide, glossary impact, task/master/roadmap, rendered pages, Markdown, exact scope, schema,
status/order, and performance evidence; and not repeat successful Java or timing runs unless it
changes executable/probe behavior or records a concrete stale-evidence reason. Do not mark
Complete until both contexts and every acceptance criterion succeed.

Return exact changed paths, commands/results, environment, class descriptors/sizes/SHA-256/member
references, every FLOAT64 and FLOAT32 per-fork/aggregate ratio, documentation/Javadoc/glossary
impact and reasoned no-change conclusions, context IDs, unresolved issues, and exactly Status:
Complete or Status: Incomplete with Follow-up required: <specific follow-up>.
```

## Local decisions

- Keep INITIAL_STATE and DROPOUT together because one bridge-only emitter owns both prologue
  forms, and direct dropout generation must preserve the exact same state representation and
  lifecycle. Only dense dropout receives timing gates.
- Advance exactly to schema 28 because generated random bodies change while schema 27 already
  identifies embedded ordering bodies. A schema-27 bridge-only random artifact must never satisfy
  a schema-28 direct-body request.
- Use two 1,048,576-element dense heap-array cases with one exact state and probability to compare
  equivalent full value/mask/state work while retaining shape-polymorphic production generation.
- Require both every per-fork ratio and each type's aggregate to pass `<= 1.15x`; one type or
  aggregate cannot conceal a failing process.
- Keep initializer, general layouts, segment/mixed carriers, partial ranges, empty cases, and
  parallel orchestration in stable semantic/Class-File gates. The two named performance claims do
  not generalize beyond their exact cases.
- Preserve current structural identity and keep concrete explicit dropout state and invocation
  geometry cold; direct generation is not permission for fixed-state or fixed-shape artifacts.

## Known limitations

- Timing gates cover only the exact dense heap-array FLOAT64 and FLOAT32 dropout cases on the
  recorded environment. They make no parity claim for initializer, general layouts, segments,
  mixed carriers, partial ranges, parallel orchestration, every probability/state, CPU, or JVM.
- General and unmeasured forms receive stable semantic and Class-File validation but no numerical
  performance threshold.
- Only the completed one-node fully static resolved-layout CPU random capability is corrected.
  BFLOAT16, dynamic geometry, fusion, vector/native random routes, other distributions, and public
  or cross-backend bitstream contracts remain unsupported or deferred exactly as before.
- The CPU-private V1 mapping and fixed finite-precision order remain intentionally unchanged; this
  task neither certifies statistical quality nor claims cross-version replay.

## Validation evidence

Implementation, independent-review, and clean documentation evidence:

- Clean implementation context `01a0194d-ff2e-78d3-9815-f0830065185b` retained all baseline and
  final evidence under `/private/tmp/synaptik-cpu-0007a0f-implementation.A9MuK9`. The recorded
  environment is OpenJDK 26.0.1+8-34 on macOS 26.5.2 arm64, Darwin 25.5.0, repository commit
  `8b46df52d36fc5d08c4e4dd78b3ad8633f2160be`. CPU/processor sysctl facts were unavailable in the
  managed sandbox and that limitation is retained verbatim in `environment.txt`.
- The unchanged performance source SHA-256 values are
  `b1dd6cd76912d7e9b7ec94c2d1d1ff719ad479a0ac4cd63850f71b52c63736c0` for
  `RandomPerformanceProbe.java` and
  `175e877e6e1f9a5a75f8f2332bfad7f49071a5095a479193f0fb17c5d3b03787` for the summarizer.
  `commands.txt` records the exact compile, generation, definition, decompilation, five-fork,
  summary, focused-test, and authoritative-test commands.
- Fresh schema-27 baseline classes had the planned direct descriptors and bridge-only shape:
  heap INITIAL_STATE `([J[JJJ)V`, 485 bytes,
  `2c00e7d8d8603d864d7557d9a5db5b1f61b2c26342268286a2821969c31767e2`; segment INITIAL_STATE
  `(Ljava/lang/foreign/MemorySegment;[JJJ)V`, 516 bytes,
  `e70f76ef598892b686f6de46340110f33a01f5c21595d026f90e2f3136c3a921`; dense FLOAT64 DROPOUT
  `([D[J[D[B[J[JJJ)V`, 485 bytes,
  `284036b8e6dc2fca771cbccd3ac39f8946e1d8623bff16c9c3c9a2d4abdf16dc`; dense FLOAT32 DROPOUT
  `([F[J[F[B[J[JJJ)V`, 485 bytes,
  `94b15d30a10e435798d8abc5d0366c5ac02c6aeb61d369673d386dd6104159ba`; all-segment FLOAT64
  DROPOUT, 640 bytes, `8b488a6b8161031b6d6e84a0ff3817a3999f5a8a28f9a41d41111bb06a4116e0`;
  and mixed general FLOAT32 DROPOUT
  `([FLjava/lang/foreign/MemorySegment;[FLjava/lang/foreign/MemorySegment;[J[JJJ)V`, 547 bytes,
  `e1f92c60eadb67f0ed72a326f12065b37d48965c49a6d7c8d19f24f002564c47`. Every baseline class
  had exactly one member reference to the five-Object `CpuRandomEmitter.execute` bridge.
  Representative initializer/all-segment/mixed semantic checks passed.
- Baseline FLOAT64 fork ratios were `2.829098412x`, `2.788332667x`, `2.780533770x`,
  `2.791680264x`, and `2.832401595x`, aggregate `2.806925374x`. Baseline FLOAT32 fork ratios were
  `2.843347014x`, `2.903399294x`, `2.904503815x`, `2.908505695x`, and `2.896042990x`, aggregate
  `2.871508342x`.
- Final schema-28 descriptors are unchanged. Heap INITIAL_STATE is 345 bytes,
  `29cd15f6c53c9745b14c64401a25bf167f11fece316699e6a069974837f43512`; segment INITIAL_STATE is
  650 bytes, `d67ade9bf4ae7204a6530580cf00a79b68aa9003856d764c13735b9edc5b7cad`; dense FLOAT64 DROPOUT is
  965 bytes, `9fa7a02bcdca03a23f9de862d62d86cbc841bbad5f887f7435b844994f40fa63`; dense FLOAT32 DROPOUT is
  964 bytes, `d962513c4829d80a9d59d2b6771a1cd0865ae22f2f34ce38393d8beb75e3862d`; all-segment FLOAT64
  DROPOUT is 2,151 bytes, `9bffc22fdb607ab890b1de96ec2b2692be1a6e8c38858eef00ffe99d4c6d12f5`;
  and mixed general FLOAT32 DROPOUT is 1,777 bytes,
  `9d50d5be1c536464f8b1452b7547eb832c292856990c063e0012346e8f5f80f0`.
- Complete retained `javap -c -p` and `javap -v -p` show direct state, mix, uniform, threshold,
  value, mask, and loop instructions without `Object` descriptors, generic bridges, allocation,
  or runtime dispatch. Dense heap classes have zero member references. Segment forms reference
  only exact `ValueLayout.JAVA_LONG_UNALIGNED`, `JAVA_DOUBLE_UNALIGNED`, `JAVA_BYTE`, and the
  matching typed `MemorySegment.get/set` methods needed by their carrier roles; the mixed class
  uses only LONG and BYTE segment accesses because its FLOAT32 carriers are arrays.
- Final FLOAT64 fork ratios were `0.242432568x`, `0.240238314x`, `0.245355311x`,
  `0.242999122x`, and `0.241854555x`, aggregate `0.241451022x`. Final FLOAT32 fork ratios were
  `1.046430134x`, `1.063729192x`, `1.052403558x`, `1.060950642x`, and `1.053413773x`, aggregate
  `1.058174348x`. Every fork and both aggregates passed `<= 1.15x` with unchanged fixed inputs,
  direct bodies, state, probability, Shape, heap, seed, warmup, measurement, adaptive, verification,
  case/form randomization, and sink rules.
- The final focused five-owner command passed 5 suites and 50 tests with zero failures, errors, or
  skips. The exactly one authoritative `./gradlew :backends:cpu:test` passed 53 suites and 323
  tests with zero failures/errors and one existing expected opt-in persistence skip. Executable
  Java did not change afterward.
- Independent review orchestration context `01a0196a-0930-7521-972f-84021b85519b` and isolated
  review context `/root/cpu_0007a0f_review` found no production defect and changed tests only. Its
  `CpuClassFileKernelGeneratorTest` run passed 4 tests; its combined generator/random run passed
  15 tests. These are review-only focused results and are not combined with the authoritative
  323-test CPU-module result. Review added durable exact descriptor assertions and strengthened
  all 32 ordered carrier-pattern/member-reference gates for each dropout value type.
- Review regenerated the six representative classes under `review-evidence/generated`. Every
  class is byte-identical to the corresponding final class: heap INITIAL_STATE 345 bytes,
  `29cd15f6c53c9745b14c64401a25bf167f11fece316699e6a069974837f43512`; segment INITIAL_STATE
  650 bytes, `d67ade9bf4ae7204a6530580cf00a79b68aa9003856d764c13735b9edc5b7cad`;
  dense FLOAT64 965 bytes, `9fa7a02bcdca03a23f9de862d62d86cbc841bbad5f887f7435b844994f40fa63`;
  dense FLOAT32 964 bytes, `d962513c4829d80a9d59d2b6771a1cd0865ae22f2f34ce38393d8beb75e3862d`;
  all-segment FLOAT64 2,151 bytes,
  `9bffc22fdb607ab890b1de96ec2b2692be1a6e8c38858eef00ffe99d4c6d12f5`; and mixed-general
  FLOAT32 1,777 bytes, `9d50d5be1c536464f8b1452b7547eb832c292856990c063e0012346e8f5f80f0`.
  No production bytes changed in review, so the accepted timing evidence remained current. The
  retained JIT diagnostic independently supports the unusually low fixed-case FLOAT64 ratio and
  the reviewer found no benchmark flaw; this remains observational evidence, not a speedup claim.
- Clean documentation context `01a01973-5ecf-75b0-ae50-09aebdd3ddf7` independently reviewed the
  final source, every changed CPU test, complete representative Class-File evidence, retained
  probe sources/commands/environment/checksums/descriptors/member references, raw and summarized
  timing, review diagnostic, and planning state. It changed no executable Java, Java test, or
  probe behavior and therefore reused all stabilized Java and timing evidence without rerunning
  it.
- The documentation pass finalized the five affected production/package Javadocs and the CPU
  guide's schema-28 direct random/dropout boundary, dense/general carrier forms, `[0,0)` explicit-
  state prologue and range contract, structural-versus-invocation identity, schema-27 safe-miss
  persistence transition, limitations, and fixed-host/JVM performance evidence. Glossary review
  required no edit because direct generated-loop emission and schema 28 introduce no new reusable
  project term or change to the existing bounded-replay/dropout definitions.
- Reasoned no-change review: `CpuRandomIr` and its package summary remain accurate because the
  structural V1 identity and cold geometry boundary did not change; `CpuPortableRoutePlan` and
  its package summary remain accurate because route selection and encoded-IR ownership did not
  change. `CpuRandomLowering`, `CpuPartitionPreparationPlan`, preparation/finalization,
  `CpuPreparedExecutable`, and `CpuScalarReferenceKernel` remain accurate because capability,
  lowering, one-unit/one-artifact/zero-workspace declarations, exact five-boundary binding,
  overlap validation, worker orchestration, lifecycle, and independent reference semantics were
  preserved. No capability, shared/public API, dependency, module boundary, build, architecture,
  ADR, architecture-test, backend-conformance, integration, Compiler, Runtime, Engine, other
  backend, NN, or Training change is needed.
- `./gradlew :backends:cpu:javadoc` passed in the documentation context: 11 tasks completed, one
  executed and ten up-to-date, with only the two expected incubating-vector-module warnings and
  no Javadoc warning or error. The rendered `CpuGeneratorSchema`, `CpuClassFileKernelGenerator`,
  and `CpuRandomEmitter` type pages and the cache and emitter package-summary pages were inspected
  for the finalized contract text and method/constructor `@param`, `@return`, and `@throws`
  sections.
- Local Markdown link and heading-anchor resolution, balanced fences, terminal newlines, carriage
  returns, and trailing whitespace passed for the CPU guide, this task, CPU master plan, and
  roadmap. Java package/type placement, schema-28/current history, tested schema-27 incompatible
  safe-miss language, exact descriptors and member-reference allowlists, retained evidence
  inventory, status/dependency order, absence of CPU 0007A1 or later detailed task specifications,
  empty index, and absence of GSD artifacts also passed. Final `git diff --check` passed.
- The CPU task owns exactly 15 changed or created paths, all within the 22-path exact
  implementation allowlist: five production/package Javadocs, six Java test files, the CPU guide,
  this task, CPU master plan, and roadmap. The allowlisted glossary remains a reasoned CPU
  no-change; its concurrent NN-only work is not part of this task. All concurrent NN source,
  tests, API documentation, glossary, planning, and roadmap content was preserved and excluded
  from the CPU path count and conclusions.

## Implementation notes

Schema 28 embeds typed initializer and dropout state prologues, direct CPU-private V1 mix/uniform/
threshold work, exact FLOAT64 or widen-divide-narrow-once FLOAT32 values, canonical mask stores,
and global-ordinal loops. Dense heap-array forms narrow bases and ranges once and keep integer
address state; arbitrary layouts, zero-stride reads, segments, and mixed carriers retain typed
general-long geometry. Existing lowering, route, preparation, finalization, executable binding,
overlap validation, worker ownership, reference implementation, and lifecycle owners were left
unchanged.

## Completion summary

CPU 0007A0F now embeds schema-28 direct typed INITIAL_STATE and FLOAT64/FLOAT32 DROPOUT generated
bodies while preserving the completed explicit-state random semantics and surrounding lowering,
route, preparation, execution, reference, lifecycle, and architecture contracts. Stable semantic,
Class-File, descriptor, carrier/member-reference, cache-rejection, focused, authoritative-module,
and fixed-case five-fork evidence passed; independent review found no production defect, changed
tests only, and regenerated all six representative classes byte-identically.

The distinct clean documentation context finalized the five affected production/package Javadocs,
CPU guide, task, CPU master plan, and roadmap; reviewed glossary impact and directly relevant
unchanged contracts; inspected the rendered Javadocs; and passed the prescribed Javadoc,
Markdown, exact-scope, evidence, ordering, index, artifact, and whitespace validation without
changing executable Java or tests. CPU 0007A1 is the sole next ordered `Ready` frontier and remains
without a detailed task specification. There are no unresolved issues; follow-up is CPU 0007A1
planning and the existing CPU 0009 checkpoint.

Status: Complete
