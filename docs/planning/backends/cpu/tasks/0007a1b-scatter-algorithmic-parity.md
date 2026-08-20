# Task 0007A1B: Scatter Algorithmic Parity

## Status

Complete

## Goal

Replace the avoidable roughly `O(outputCount * updateScalarCount)` generated scatter traversal
with the lowest-complexity deterministic algorithm that preserves the completed CPU scatter
contract for every supported dense and general form.

The implementation must first retain a reproducible adverse output/update-ratio baseline. For
replacement, `ADD`, `MIN`, `MAX`, and integral `MUL`, each invocation range then copies its owned
base output cells once and visits the complete logical update domain once, applying only updates
whose target ordinal belongs to that range. Floating `MUL` remains a separately bounded exact-
product form unless evidence proves an equally safe algorithm without changing its established
one-scratch-slice-per-range resource contract.

This task changes CPU-private generated algorithm and current-only artifact compatibility. It
does not change scatter semantics, validation, public capability, routes, runtime policy, or
autotuning behavior.

## Scope

### Mandatory pre-edit diagnosis and retained baseline

- Create exactly one fresh evidence directory with `mktemp -d
  /private/tmp/synaptik-cpu-0007a1b-XXXXXXXX` before any production edit. All probe source,
  compiled probe classes, generated classes, `javap` output, checksums, raw samples, summaries,
  commands, and environment facts must remain beneath that directory.
- Record repository HEAD, dirty-path inventory, Java/JVM/Gradle/operating-system/architecture/CPU
  facts, current `CpuGeneratorSchema.CURRENT_VERSION`, and SHA-256 checksums for the retained
  evidence corpus. Preserve the existing uncommitted CPU 0007A1/0007A1A work.
- Generate and decompile representative current schema-30 classes for all of:
  - dense heap-array `SCATTER_ELEMENTS + NONE` with unique targets;
  - dense heap-array `SCATTER_ELEMENTS + ADD` with duplicate targets;
  - dense heap-array Gather-compatible `SCATTER_ADD` with duplicate indices;
  - dense heap-array `SCATTER_ND + NONE` with unique tuples and a non-scalar suffix;
  - one truthful typed `GENERAL_LONG` segment/mixed-carrier scatter with nonzero offsets,
    non-unit strides, and an injective strided output; and
  - one floating `MUL` entry with exact-product scratch.
- Retain complete `javap -c -p` and `javap -v -p`, class size, descriptor, SHA-256, constant-pool
  member references, and stable Class-File model inspection for each representative class. The
  diagnosis must demonstrate from emitted loop nesting that current scratch-free work is
  approximately `O(ownedOutputCount * updateScalarCount)` and must separately describe the same
  grouping traversal used by floating `MUL`.
- Before production edits, run the fixed adverse-ratio benchmark matrix below against semantically
  identical direct Java. Retain every fork and summary even when the generated baseline is slow.
  Timing includes only warmed generated/direct writer work after the same validation assumptions;
  it excludes generation, class loading, preparation, binding, allocation, index/duplicate
  validation, result checking, and sink observation.

### Required algorithm and safe form split

- For `SCATTER_ELEMENTS`, Gather-compatible `SCATTER_ADD`, and `SCATTER_ND` with `NONE`, `ADD`,
  `MIN`, `MAX`, or integral `MUL`, emit this range-owned two-phase algorithm:
  1. copy base values for exactly the invocation's legal half-open output ordinal range
     `[start, end)` through the selected data/output layouts and typed carriers; then
  2. traverse the complete logical update scalar domain in canonical row-major order, derive each
     complete target coordinate and logical row-major output ordinal, and apply the update only
     when that ordinal is inside `[start, end)`.
- Scalar complexity for those forms must be `O(rangeLength + updateScalarCount)`. For `W` disjoint
  ranges, total work must be bounded by `O(outputCount + W * updateScalarCount)`, with `W` equal to
  the already prepared bounded range count. No worker may traverse the output domain once per
  update or the update domain once per output.
- Each range writes only its owned output coordinates. Parallel ranges may share only read-only
  inputs and immutable geometry. They use no atomics, locks, barriers inside generated work,
  shared mutable table, cross-range partial result, or merge phase.
- Update traversal and reduction order remain canonical logical row-major order for every target.
  Gather-compatible `SCATTER_ADD` retains same-format FLOAT64/FLOAT32 addition, BFLOAT16 rounding
  after each contribution, and modular integral addition. Extrema and integral multiplication
  retain their established represented-value rules.
- `NONE` remains invalid for duplicate complete targets. Bounds validation and then replacement-
  uniqueness validation still complete before generated work or worker submission. If a generated
  entry is invoked directly after bypassing that required validation, its ordered overwrite is
  last-assignment-wins; this is a Class-File algorithm invariant for deterministic raw-entry tests,
  not a new observable Model or CPU contract.
- Copy-then-update is **not** universally legal for floating `MUL` under the current resource
  contract. Interleaved target updates cannot round intermediate products and cannot retain exact
  product state for every output without output-proportional state. Therefore FLOAT64, FLOAT32,
  and BFLOAT16 `MUL` must retain output-owned complete target grouping with one reusable exact
  accumulator per range unless the implementation proves a lower-complexity alternative that:
  - preserves one abstract unchanged-format product rounded once;
  - uses exactly the already declared one fixed-capacity scratch slice per selected range;
  - introduces no output/update table, sort, second workspace, changed workspace size, hidden heap
    state, or allocation; and
  - passes every semantic, resource, Class-File, range, worker, and performance gate here.
  If those conditions cannot all be met, retaining the existing `O(rangeLength *
  updateScalarCount)` floating-product traversal is required and must be documented as the safe
  form split rather than disguised as full-family linear complexity.
- The implementation may retain distinct dense rank-one and typed general-long emitters, or share
  generation-time structure, but both must realize the same range-owned algorithm for eligible
  scratch-free forms. A failed dense proof selects the truthful general form and never rejects a
  currently supported occurrence.

### Preserved semantic and lifecycle contracts

- Preserve exactly the current three semantic families, ordered `[data, indices, updates]` roles,
  first-occurrence input deduplication, result Shape/type, supported data/index types, reductions,
  static Shape requirement, resolved layout rules, and distinct injective output requirement.
- Preserve unaddressed base bits, unique `NONE` replacement bits, base participation exactly once
  for reductions, every duplicate update exactly once, current special-value behavior, signed
  zero, NaN boundary, BFLOAT16 conversion/rounding, modular overflow, and floating exact-product
  exponent/significand/overflow/underflow state.
- Preserve complete carrier size/alignment/accessibility/writability checks, full-span
  output/input overlap rejection before value validation, allowed input/input aliasing, canonical
  BOOL validation, complete bounds validation, replacement uniqueness, exception types/messages/
  ordinals, and no-write/no-worker failure timing.
- Preserve independent validation for empty indices, empty updates, zero output, zero selected
  extents, scalar forms, and empty invocation ranges. An empty range writes nothing; a nonempty
  range copies and updates only its interval.
- Preserve arbitrary legal `[start, end)` calls, including nonzero starts, adjacent partial calls,
  repeated calls, and the worker partitions chosen by `CpuPreparedExecutable`. Do not require a
  full-output call for correctness.
- Preserve all current primitive-array, `MemorySegment`, and mixed carrier combinations; nonzero
  layout offsets; zero-stride and repeated reads; non-unit input strides; injective strided output;
  both index widths; and long-address geometry without unchecked narrowing.
- Preserve one unit, one generated class/artifact, two through four deduplicated buffer
  declarations, no materialization, scalar/parallel-scalar selection, immutable prepared recipe,
  invocation-private geometry, and no shared mutation or race.
- Preserve zero workspace for every scratch-free form and exactly one `SCATTER_PRODUCT` workspace
  with unchanged per-range slice sizing/alignment/count for eligible floating `MUL`. Analysis must
  still declare every resource before slot assignment, and finalization must not add or reinterpret
  a requirement.
- Preserve the current typed static entry descriptors: ordered direct typed carriers, optional
  scratch `MemorySegment`, `long[]` geometry, and primitive `long start, long end`. Do not add
  fields, constructors, secondary methods, `Object` parameters, bridges, method handles,
  `invokedynamic`, dynamic constants, bootstrap methods, or project-owned runtime helper calls.
- CPU analysis and finalization remain deterministic and measurement-free. Runtime executes the
  prepared result only. Benchmark results must not select algorithms, mutate preparation policy,
  enter cache identity as measurements, or create runtime/autotuning behavior.

### Fixed performance protocol and gates

- Use deterministic data, index, and update initialization documented in probe source. Every case
  verifies input immutability and exact raw output bits against its direct implementation before
  and after timing. Each direct implementation must use the same carrier/layout work, target
  mapping, ordered reduction, range ownership, and safe algorithm form as the generated entry; it
  may not use pre-grouping, sorting, hashing, atomics, vector scatter, a generated/helper call, or
  easier numerical semantics.
- The fixed adverse matrix is:

  | Case | Shape and form | Targets | Purpose |
  |---|---|---|---|
  | E-NONE-dense | rank-one FLOAT32 data/output extent 65,536; INT32 indices and updates extent 1,024; indices `((61 * i) + 17) & 65535`, which are unique | unique | dense replacement, output/update ratio 64 |
  | E-ADD-dense | same extents; `SCATTER_ELEMENTS + ADD`; indices `i & 255` | duplicate | ordered same-format duplicate reduction |
  | SA-ADD-dense | FLOAT32 data/output extent 65,536; scalar-index-vector extent 1,024 and Gather-compatible updates extent 1,024; indices `i & 255` | duplicate | Gather-compatible mapping |
  | ND-NONE-dense | FLOAT32 data/output `[8192, 8]`; INT32 indices `[128, 1]`; updates `[128, 8]`; unique first-axis tuple permutation | unique | tuple mapping and suffix scalars, ratio 64 |
  | E-ADD-general | FLOAT32 `SCATTER_ELEMENTS + ADD`, logical data/output `[4096, 8]`, updates/indices `[512, 8]`, nonzero offsets and non-unit strides, mixed array/segment carriers, injective strided output | duplicate | truthful `GENERAL_LONG` work |
  | E-MUL-exact | FLOAT32 `SCATTER_ELEMENTS + MUL`, data/output extent 2,048 and updates extent 256 with duplicate targets | duplicate | unchanged exact-product grouped form and scratch |

- The implementation may correct a mathematically invalid initializer while planning the probe,
  but may not reduce any output/update ratio, remove duplicates/uniqueness, replace a general
  carrier/layout, or otherwise make a case easier. Any correction and rationale must be recorded
  before baseline measurement and then held identical for final measurement.
- Run baseline and final probes in five isolated JVM forks with fixed `-Xms1g -Xmx1g`, at least
  five randomized warmup rounds, nine randomized measured rounds, and adaptive batches whose
  samples last at least 25 ms. Report generated/direct median and ratio for every fork plus the
  median of fork medians. Do not average cases.
- Final acceptance is strict per case: every individual fork and aggregate generated/direct ratio
  is `<= 1.15x`. The five scratch-free cases must also demonstrate the required complexity from
  emitted loop structure and a final adverse-ratio generated median at least `4.00x` faster than
  that same case's retained pre-edit generated median. The exact-product case is exempt from the
  `4.00x` improvement only because it retains the proved safe grouped split; it still must pass
  every `<= 1.15x` direct-algorithm gate.
- If noise or the environment prevents a case from meeting its fixed gate, retain `In progress` or
  `Review needed`; do not change thresholds, shapes, algorithm, samples, or production policy to
  manufacture a pass.

### Generated compatibility and stable evidence

- If generated scatter bytes change, advance `CpuGeneratorSchema.CURRENT_VERSION` exactly once
  from 30 to 31. Schema-30 envelopes become incompatible safe misses even when their metadata and
  class bytes are otherwise valid. Add no migration reader, alias, converter, or legacy bridge.
- If no generated bytes change, do not increment the schema. Because this task's required
  scratch-free algorithm changes emitted loop shape, failure to advance normally indicates an
  incomplete implementation and must be explained rather than silently accepted.
- Add stable Class-File model tests covering every family, admitted reduction category, every data
  type, both index widths, dense/general access, arrays/segments/mixed/deduplicated carriers,
  scratch-free/scratch-bearing descriptors, arbitrary ranges, and scalar/parallel prepared use in
  proportion to risk.
- Automated shape tests must prove the copy loop precedes one update traversal for eligible forms,
  target-range filtering, typed direct load/reduce/store operations, no nested output-per-update
  traversal, and the preserved grouped exact-product form. Tests must inspect Class-File models and
  member references, not source text or fragile absolute bytecode offsets.
- Decompile final representatives and compare descriptors, fields/methods, member references,
  class sizes, and checksums with baseline. Record exact differences and confirm that no forbidden
  member reference or class-shape expansion entered generated code.

## Out of scope

- Changing Model scatter meaning, `NONE` uniqueness, index policy, Shape/type eligibility,
  operation kinds/attributes, Tensor APIs, compiler gradients, capability advertisement, or public
  CPU APIs.
- Output-proportional grouping tables, target hash maps, sort/group passes, atomics, locks, vector
  scatter, per-output exact-product scratch, a second workspace, materialization, in-place output,
  negative-stride expansion, dynamic Shapes, or unresolved layouts.
- New routes, native/vendor work, fusion, public configuration, tuning candidates/caches,
  benchmarking tools, runtime profiling/selection, autotuning, or benchmark-driven mutation.
- General generated/direct evidence closure owned by CPU 0007A1C; unrelated pointwise, movement,
  indexing, fold, ordering, random, scan, aggregate, normalization, or later-family work.
- Architecture, dependency, shared-module, Gradle/build, backend-conformance, integration, other-
  backend, Engine, NN, training, or unrelated documentation changes.
- Creating a CPU 0007A1C or later detailed specification; commit, push, staging, revert, deletion,
  or modification of unrelated work.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`performance evidence and tuning`](../../../../architecture/performance-evidence-and-tuning.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`general documentation profile`](../../../../developer-guide/documentation/general-style.md)
- [`planning documentation profile`](../../../../developer-guide/documentation/planning-style.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0006B1 portable functional scatter`](0006b1-portable-functional-scatter.md)
- [`CPU 0007A0C scatter generated-loop parity`](0007a0c-scatter-generated-loop-parity.md)
- [`CPU 0007A1 ordinary numerical aggregates`](0007a1-portable-ordinary-numerical-aggregate-reductions.md)
- [`CPU 0007A1A generated scalar-body self-containment`](0007a1a-generated-scalar-body-self-containment.md)

## Architecture constraints

- Model owns scatter semantics. This task changes only CPU-private generated realization and may
  neither narrow nor broaden the semantic or capability contract.
- CPU analysis owns lowering, specialization, strategy, and exact resource declarations before
  slot assignment. CPU finalization owns current-schema generation/reuse afterward. Runtime sees
  only immutable prepared execution and performs no generation, selection, validation policy, or
  tuning.
- Prepared executions remain immutable and reusable. Concurrent runs have distinct `RunState`,
  geometry, buffers, and run-owned scratch; generated classes retain no mutable run state.
- Performance evidence is fixed and observational. It can accept, reject, or diagnose the code
  change but cannot become production configuration, route selection, cache mutation, or runtime
  behavior.
- Stop if correctness or the fixed gates require a public/shared contract, architecture or
  dependency change, new resource kind, changed floating-product workspace, runtime policy,
  capability change, or a broader task.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — selected dense/general scatter
  loop emission and existing exact-product emission boundary.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — current-only generated compatibility.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` and `.lowering` — reviewed unchanged owners
  of structural scatter identity and cold mapping/resource geometry.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` and `.route.portable` — reviewed
  unchanged deterministic strategy/declaration/finalization owners.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — reviewed unchanged validation,
  range partitioning, worker ownership, direct binding, and scratch-slice owner.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent semantic oracle only.

Packages added or changed:

- No package is added, removed, moved, exported, or made supported API.
- Production behavior changes are expected only in `internal.codegen.emit`; current-schema
  metadata changes in `internal.cache`.

Type placement:

- `CpuScatterEmitter` remains the sole generated scatter algorithm owner. It may gain focused
  private generation methods but no runtime helper type.
- `CpuExactProductEmitter` remains the generation-time exact-product owner and must not become an
  execution service or change its numerical contract.
- `CpuGeneratorSchema` remains the sole current schema owner.
- `CpuScatterIr`, `CpuScatterLowering`, `CpuPartitionPreparationPlan`,
  `CpuPartitionPreparer`, `CpuPartitionFinalizer`, `CpuPreparedExecutable`,
  `CpuPortableRoutePlan`, and `CpuScalarReferenceKernel` retain their owners. If the new algorithm
  requires a contract change in one of them, stop unless the exact allowlisted Javadoc/test-only
  consequence below is sufficient.

## Affected files

Expected production and package-contract paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`

Expected test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md` only if an existing reusable scatter/generated-kernel term changes
- this task specification
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

The implementation may omit an allowlisted path when no edit is needed. `CpuExactProductEmitter`,
`CpuScatterIr`, `CpuScatterLowering`, `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`,
`CpuPartitionFinalizer`, `CpuPreparedExecutable`, `CpuPortableRoutePlan`, and
`CpuScalarReferenceKernel` may be edited only for directly necessary Javadoc or stable test seams
without changing their established behavior/resources. No new production Java type is permitted.

## Maximum scope

This task may create or modify at most:

- 6 production Java/package-contract paths, with no new production type;
- 7 Java test paths;
- 5 documentation/planning paths; and
- 18 repository paths total.

Evidence files live only beneath the one fresh `/private/tmp` directory and do not count as
repository paths. If implementation requires a new type, another module, more than 18 paths, a
changed resource contract, or a public/shared change, stop and propose a follow-up or replan.

## Acceptance criteria

- The pre-edit evidence reproducibly demonstrates the current adverse-ratio algorithm and retains
  complete commands, class evidence, direct comparisons, raw samples, summaries, environment, and
  checksums in one fresh `/private/tmp` directory.
- Every scratch-free scatter family/reduction/type keeps one base-copy traversal of `[start,end)`
  followed by one complete canonical update-scalar traversal with target-range filtering; scalar
  and total parallel complexity satisfy the bounds in Scope.
- Floating `MUL` either retains the explicitly documented output-owned safe split with unchanged
  exact scratch, or a lower-complexity implementation proves every stricter legality condition in
  Scope. No intermediate rounding or output-proportional hidden state is accepted.
- Unique and duplicate semantics, ordered reductions, raw represented results, exact product,
  arbitrary layouts/carriers/ranges, empty domains, workers, concurrency, repeat invocation,
  input deduplication, and overlap/validation failure timing pass independent differential tests.
- Public duplicate replacement remains a pre-write failure. Stable raw-entry evidence proves
  deterministic last-assignment-wins only when the required validator is intentionally bypassed,
  and documentation does not present that as public semantics.
- Resource tests prove exact unchanged buffer declarations, no materialization, zero scratch for
  scratch-free forms, and one unchanged predeclared per-range `SCATTER_PRODUCT` workspace only for
  eligible floating `MUL`.
- Stable Class-File tests and retained `javap` prove exact typed descriptors, current class/member
  shape, dense/general algorithms, no forbidden nested traversal in scratch-free forms, preserved
  grouped exact-product form, and absence of forbidden bridges/dispatch/allocation/member refs.
- Schema advances exactly 30 to 31 after emitted bytes change; schema 30 is a safe miss, corrupt
  entries fail safely, compatible geometry still reuses one schema-31 artifact, and no migration
  path exists.
- Every performance case independently passes every fork and aggregate `<= 1.15x`; every
  scratch-free adverse case also passes the `>= 4.00x` pre-edit-generated improvement gate.
- No runtime/autotuning behavior, capability, public API, architecture, dependency, shared module,
  route, build, conformance/integration owner, later task, or unrelated dirty path changes.
- A distinct clean documentation-focused context reviews the final implementation, tests,
  generated-class evidence, performance corpus, and affected contracts; finalizes Javadocs,
  package summaries, CPU guide, glossary impact, task evidence/status, master plan, and roadmap;
  and records reasoned no-change conclusions before completion.

## Tests / validation

Run focused tests for every changed test owner during implementation. After executable Java and
tests stabilize, run exactly one authoritative CPU module suite:

```bash
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuScatterGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest
./gradlew :backends:cpu:test
```

Run the fixed pre-edit and final five-fork probe outside Gradle/JUnit from the one evidence
directory. Record exact `javac`/`java` commands and, for every representative class:

```bash
javap -c -p <generated-class-file>
javap -v -p <generated-class-file>
sha256sum <generated-class-file>
```

The distinct documentation-focused context reuses successful Java/performance evidence unless it
changes executable Java or probe behavior or records a concrete stale-evidence reason. After its
edits it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also inspects rendered affected Javadocs; validates changed Markdown links/anchors, balanced
fences, terminal newlines, trailing whitespace, exact allowlist/count, package/type placement,
schema-30-to-31 history, status/order/dependencies, evidence checksums, preserved unrelated dirty
paths, and absence of a detailed CPU 0007A1C specification.

This is a high-risk single-module algorithm correction. Repository-wide, architecture,
backend-conformance, and integration suites remain deferred to CPU 0009 or continuous integration
because no shared contract or dependency may change. If implementation makes that conclusion
false, stop and replan before running or changing another tier.

## Dependencies

- [CPU 0006B1](0006b1-portable-functional-scatter.md) is Complete and owns current scatter
  semantics, validation order, resources, layouts/carriers/ranges, parallel ownership, and exact
  floating-product scratch.
- [CPU 0007A0C](0007a0c-scatter-generated-loop-parity.md) is Complete and owns schema-25 typed
  embedded scatter loops plus the direct-Class-File and five-fork evidence foundation.
- [CPU 0007A1](0007a1-portable-ordinary-numerical-aggregate-reductions.md) is Complete and extracted
  the shared generation-time exact-product emitter without changing scatter semantics.
- [CPU 0007A1A](0007a1a-generated-scalar-body-self-containment.md) is Complete and supplies the
  current schema-30 typed generated-code/member-reference baseline.
- Current Java 26 Class-File API, CPU carrier/access geometry, preparation/finalization, worker,
  exact workspace, reference, and artifact-store contracts remain complete and unchanged.

## Follow-up tasks

- CPU 0007A1C remains Draft and owns bounded generated/direct evidence closure across general
  layouts, `MemorySegment`, and previously unmeasured families/variants. Do not create its detailed
  specification here.
- CPU 0007A2 and later CPU work remain ordered Draft behind CPU 0007A1C.
- CPU 0009 retains repository-wide portable generated-coverage/conformance closure.

## Architecture impact

Expected impact: None.

The architecture already places lowering, generated algorithm selection, exact resource
declaration, and current-only artifact compatibility inside CPU prepare/finalization while Runtime
executes immutable prepared work. If implementation requires a changed architecture contract,
focused architecture page, architecture decision record, architecture test, dependency, public or
shared lifecycle contract, resource kind, or runtime/tuning policy, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1B. Work on the existing
uncommitted diff without committing, pushing, staging, reverting, deleting, or modifying unrelated
work. Do not use a GSD skill or workflow.

Read in full AGENTS.md, ARCHITECTURE.md, the current architecture and performance-evidence pages,
planning guide/roadmap, CPU master plan, task 0007A1B, completed tasks 0006B1/0007A0C/0007A1/
0007A1A, and every directly relevant final scatter IR/lowering/emitter/preparation/finalization/
execution/reference/test/Javadoc contract. Inspect and preserve the existing dirty diff.

Before production edits, create the one fresh /private/tmp evidence directory and retain the
required adverse-ratio baseline, generated classes, complete decompilation/member references,
commands, environment, samples, summaries, and checksums. Then implement this specification's
range-owned copy-then-update algorithm and explicit floating-MUL safe split exactly within its
allowlist and maximum scope. Run the focused tests, one authoritative CPU suite, final Class-File
evidence, and every fixed five-fork per-case gate. Stop on architecture/shared-contract/resource/
scope conflict, a failed fixed gate, or inability to preserve semantics and arbitrary ranges.

After executable work stabilizes, hand the complete diff, task, evidence directory, commands, and
results to a distinct clean documentation-focused context following the documentation rules. That
context must independently inspect final source/tests/classes/evidence; finalize affected Javadocs,
package summaries, CPU guide, glossary impact, task/master/roadmap evidence and status; run the
documentation/planning/scope/schema/whitespace gates; and not repeat stable Java or timing runs
without executable/probe change or a recorded stale-evidence reason. Do not mark Complete until all
implementation, performance, and documentation gates pass.
```

## Local decisions

- Range-owned copy-then-update is the lowest-complexity safe scratch-free algorithm because it
  retains disjoint output ownership while replacing a scan per output with one scan per bounded
  worker range.
- Update-domain partitioning without target ownership is rejected because duplicate targets can
  race or require atomics/merge and can change ordered floating addition.
- Floating exact product keeps the output-owned safe split because one reusable per-range
  accumulator cannot hold exact state for interleaved target groups. Enlarging scratch is a
  resource-contract change, not an implicit optimization.
- `NONE` uniqueness remains public semantics. Raw writer last-assignment-wins is tested only to
  make bypassed-validation code shape deterministic and is never advertised as accepted behavior.
- Dense and general forms share the same algorithmic contract; only their address representation
  differs.
- Evidence gates are release criteria and documentation inputs, never production tuning inputs.

## Known limitations

- Total parallel work includes one complete update traversal per selected output-owning range;
  this is `O(output + workers * updates)`, not a claim of work-optimal `O(output + updates)` for
  arbitrary worker counts.
- Floating `MUL` remains approximately `O(rangeLength * updateScalarCount)` unless the strict
  unchanged-resource alternative is proved. Its exact semantics and resource truth take priority
  over a false whole-family complexity claim.
- Performance claims apply only to the fixed retained cases, environment, JVM, and protocol.
  CPU 0007A1C owns broader evidence, not this task.
- Complete bounds and replacement-uniqueness validation remains a separate pre-write pass and is
  deliberately excluded from writer-only timing.

## Validation evidence

Planning context read the governing repository, architecture, performance-evidence, planning, and
documentation rules; the CPU master plan and roadmap; completed tasks 0006B1, 0007A0C, 0007A1,
and 0007A1A; current scatter IR/lowering/emitter/generator/schema/preparation/finalization/
execution/reference owners; and directly relevant scatter tests and Model semantics. The dirty
worktree contains the completed uncommitted CPU 0007A1/0007A1A implementation and documentation,
which this planning change preserves.

The current generated dense and general scratch-free emitters use an output-owned outer range and
scan the complete update domain for every output. Scalar work is approximately
`O(rangeLength * updateScalarCount)`. A range-owned base copy plus one canonical update scan is
legal for replacement, ADD, MIN/MAX, and integral MUL and remains race-free for arbitrary legal
ranges because each worker filters targets to its disjoint ordinal interval. It is not universally
legal for once-rounded floating MUL with one exact accumulator slice per range; that form requires
the explicit safe split above. Current Model/CPU `NONE` rejects duplicate targets before generated
work, so last-wins is not public behavior.

Planning context `01a01b19-283c-79e2-a407-5ac8bc17446f` changed exactly this new task, the CPU
master plan, and the CPU-specific roadmap synchronization. Canonical task headings and `Ready`
status, the sole CPU `Ready` row, 0007A1A -> 0007A1B -> 0007A1C dependency/status order, absence
of stale Draft/no-spec claims for 0007A1B, continued Draft status for 0007A1C and later work,
absence of a detailed 0007A1C file, balanced fences, terminal newlines, trailing whitespace,
repository-local Markdown targets and heading anchors, and exact three-path planning scope all
passed. `git diff --check` and `git diff --cached --check` passed. Final `git status --short`
confirmed the preserved uncommitted CPU 0007A1/0007A1A implementation and documentation plus this
planning frontier; no existing dirty implementation path was edited or reverted by this context.
No Java, test, Javadoc, benchmark, or generated-class command was run because this is a
planning-only change.

Implementation context `01a01b36-512a-78a0-b474-1f6b6e41bfeb` created and retained the complete
read-only evidence corpus at `/private/tmp/synaptik-cpu-0007a1b-pB7Ju0wX`. The frozen probe
SHA-256 is `9a4076c5480e4fb43f9ba78eac215ca6f6fb6c41b597cfec17a71b6dc7e0f823`.
Its protocol used five isolated JVM forks with `-Xms1g -Xmx1g`, five randomized warmup rounds,
nine randomized measured rounds, adaptive batches lasting at least 25 ms, preserved pre-edit
schema-30 class bytes loaded as hidden classes, and identical frozen direct code and carriers.

Final generated/direct ratios for forks one through five, followed by the aggregate
median-of-fork-medians, were:

| Case | Fork 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | Aggregate |
|---|---:|---:|---:|---:|---:|---:|
| E-NONE-dense | 1.040 | 1.034 | 0.998 | 0.984 | 1.040 | 1.034 |
| E-ADD-dense | 1.022 | 1.025 | 1.009 | 1.028 | 1.008 | 1.022 |
| SA-ADD-dense | 1.014 | 1.016 | 1.017 | 1.020 | 1.019 | 1.017 |
| ND-NONE-dense | 1.001 | 1.013 | 1.015 | 1.012 | 1.008 | 1.012 |
| E-ADD-general | 0.554 | 0.563 | 0.559 | 0.569 | 0.565 | 0.563 |
| E-MUL-exact | 0.841 | 0.850 | 0.864 | 0.855 | 0.815 | 0.850 |

All 30 fork ratios and all six aggregates passed `<= 1.15`. Baseline-to-final generated median
improvements were `1795.9x`, `1966.0x`, `1925.0x`, `1388.1x`, and `6652.7x` for the five
scratch-free cases in table order and `2.09x` for E-MUL-exact. The scratch-free cases therefore
all exceeded `4x`; the grouped exact-product safe split passed its separate direct-parity gate.

The six final generated classes have zero fields, one typed static method, unchanged descriptors,
and no forbidden project/runtime/reflection/map/bootstrap/bridge references. Their retained
Class-File evidence is:

| Class | Descriptor | Bytes, schema 30 -> 31 | Final SHA-256 |
|---|---|---:|---|
| E-NONE-dense | `([F[I[F[F[JJJ)V` | 924 -> 862 | `5148e561327064c008b0f9477405e8cd4acad409792e1350cc5ad963e1ca9df3` |
| E-ADD-dense | `([F[I[F[F[JJJ)V` | 930 -> 868 | `7a74e162f2d664eacd1792d18b6298af3417a3542907058739564f6b12b92bb1` |
| SA-ADD-dense | `([F[I[F[F[JJJ)V` | 930 -> 868 | `42e3b38a797092a3713435e28a274d23bd099a0f51bd84c983491d453ee76358` |
| ND-NONE-dense | `([F[I[F[F[JJJ)V` | 1185 -> 1347 | `a5466d2f48e426ae29cf467b6a6ffaee197450dcb8017db08835931b4e2b33bd` |
| E-ADD-general | `(Ljava/lang/foreign/MemorySegment;[I[FLjava/lang/foreign/MemorySegment;[JJJ)V` | 1615 -> 1650 | `7b6197ad4e40cf38b431d072261136e6b7ee43356b5585dfbe9703c5cd10849b` |
| E-MUL-exact | `([F[I[F[FLjava/lang/foreign/MemorySegment;[JJJ)V` | 4893 -> 4560 | `a925080905fce7713c8b8ec6cf6720924b7cbaabd29752fe34e60f82a625f61c` |

The implementation focused scatter test passed all 16 tests. The exact seven-owner focused matrix
passed, and the authoritative `./gradlew :backends:cpu:test` run passed 53 suites and 343 tests
with zero failures, zero errors, and one existing opt-in persistence-evidence skip. No executable
Java or probe behavior changed in documentation context
`01a01b5c-abf0-7f81-902f-e6d47d585503`, so those successful commands were not repeated.
That clean documentation context applied the General, API/Javadoc, Backend Guide, Developer Guide,
Example, and Planning profiles; inspected final source, tests, package contracts, guide, glossary,
planning state, and the retained evidence; and finalized the schema-31 algorithm/resource boundary.
It ran `./gradlew :backends:cpu:javadoc`: BUILD SUCCESSFUL with two expected incubating-Vector
warnings; 11 tasks were actionable, two executed, and nine were up-to-date. Targeted `rg` review of
the rendered `CpuScatterEmitter`, `CpuGeneratorSchema`, emitter-package, and cache-package pages
confirmed the finalized copy-then-update, grouped-product, schema-31, and schema-30-safe-miss text.
A targeted Ruby validator over the CPU guide, this task, CPU master plan, and roadmap passed every
repository-local Markdown target/anchor, balanced-fence, terminal-newline, and trailing-whitespace
check. Explicit shell assertions passed the exact 14-path task allowlist, package/type placement,
schema 31, 0007A1B Complete / 0007A1C Draft order, and absence of a detailed 0007A1C task.
`git diff --check` and `git diff --cached --check` passed. Final `git status --short` and the
42-path cumulative dirty inventory confirmed that completed 0007A1/0007A1A and unrelated existing
work remained un-staged and preserved.

## Implementation notes

- Scratch-free generated scatter now uses a range-owned two-phase body: copy the selected base
  interval once, then traverse the complete logical update-scalar domain once and filter targets
  to that interval. Dense and general-long paths preserve the same algorithm and canonical update
  order while retaining their respective address representations.
- Floating exact `MUL` retains output-owned complete target grouping and the existing one-slice-
  per-range `SCATTER_PRODUCT` workspace. `CpuExactProductEmitter` gained a focused generation-time
  local-header seam so the grouped path initializes only reachable limbs without changing result,
  workspace size, alignment, ownership, or entry descriptor.
- Direct raw replacement entries are deterministically last-assignment-wins only when the required
  uniqueness validator is deliberately bypassed. Public execution still rejects duplicate
  replacement targets before any write or worker submission.
- Generated compatibility advances exactly from schema 30 to schema 31. Preserved schema-30
  envelopes are incompatible safe misses; compatible schema-31 geometry reuses one artifact, and
  no migration reader, alias, converter, or legacy bridge was added.

## Completion summary

- Completed changes: replaced scratch-free output-per-update grouping with range-owned copy-then-
  update; retained exact floating-product grouping and resource shape; advanced compatibility to
  schema 31; and added stable algorithm, raw-entry, schema, cache, specialization, preparation,
  and finalization coverage.
- Files changed or created for 0007A1B: 14 allowlisted paths—five production/package-contract
  paths, five test paths, and four documentation/planning paths. They are
  `CpuScatterEmitter.java`, `CpuExactProductEmitter.java`, `CpuGeneratorSchema.java`, the emitter
  and cache `package-info.java` files; `CpuScatterGeneratedKernelTest.java`,
  `CpuGeneratedKernelArtifactStoreTest.java`, `CpuKernelSpecializationTest.java`,
  `CpuPartitionPreparerTest.java`, and `CpuPartitionFinalizerTest.java`; the CPU backend guide,
  this task, the CPU master plan, and the roadmap.
- Tests and validation: reused the stabilized 16-test scatter run, passing seven-owner focused
  matrix, 53-suite/343-test CPU run, six-class Class-File audit, and all fixed performance gates;
  the clean documentation pass completed Javadoc, rendered-page, Markdown, scope, schema/status,
  package/type-placement, and whitespace validation.
- Documentation-agent review: context `01a01b5c-abf0-7f81-902f-e6d47d585503` finalized affected
  Javadocs, package summaries, CPU guide, planning evidence/status, and glossary conclusion without
  changing executable Java or probe behavior.
- Documentation impact: the CPU guide and emitter/cache contracts now explain range-owned
  copy-then-update, dynamic target access through dense/general layouts, the grouped floating exact-
  product exception, range ownership, schema 31, and the strictly bounded performance evidence.
- Javadoc review: `CpuScatterEmitter`, the scatter-specific `CpuExactProductEmitter` methods and
  local-state constructor seam, `CpuGeneratorSchema`, and affected emitter/cache package summaries
  now state accurate inputs, mutation, failure, algorithm, resource, and compatibility boundaries.
- Glossary impact: no change. The task changes CPU-private algorithmic realization and schema
  history, but introduces no reusable project term and changes no existing term's meaning;
  functional-scatter target groups, generated kernels, specialization, and scratch retain their
  established definitions.
- Architecture/API/build/conformance impact: no change. Public scatter semantics and capability,
  resource kind/shape, route, Runtime/Prepare ownership, module dependencies, architecture pages/
  tests, Gradle configuration, backend-conformance, integration, other backends, Engine, NN, and
  training remain unchanged. CPU 0009/CI retains the broader checkpoint.
- Reviewed unchanged owners: `CpuScatterIr`, `CpuScatterLowering`,
  `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`, `CpuPartitionFinalizer`,
  `CpuPreparedExecutable`, `CpuPortableRoutePlan`, `CpuScalarReferenceKernel`, and their package
  contracts remain accurate because geometry, validation, declaration, binding, oracle, worker,
  and lifecycle responsibilities did not change.
- Unresolved issues: None.
- Follow-up required: None for 0007A1B. CPU 0007A1C is the next ordered `Draft` frontier and remains
  without a detailed specification.

Status: Complete
