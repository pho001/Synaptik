# Task 0007A1M: Scatter MIN [residual](../../../../glossary.md#scan-residual-port-and-history-demand) parity

## Status

Complete

## Goal

Correct exactly the persistent `S-GENERAL-MIN` generated/direct [residual](../../../../glossary.md#scan-residual-port-and-history-demand) through its cohesive
functional-scatter code-shaping [owner](../../../../glossary.md#term-owner), `CpuScatterEmitter`. Preserve the frozen [CPU](../../../../glossary.md#term-cpu) 0007A1C [INT64](../../../../glossary.md#data-type--datatype)
`SCATTER_ND + MIN` copy-then-update semantics, duplicate tuples, non-[scalar](../../../../glossary.md#term-scalar) suffix, mixed carriers,
general [layouts](../../../../glossary.md#layout), arbitrary legal [half-open](../../../../glossary.md#term-half-open-interval) output ranges, and unchanged typed general fallback
while passing the frozen five-[fork](../../../../glossary.md#term-fork-meanings) `<= 1.15x` gate.

## [Scope](../../../../glossary.md#term-scope)

- Use the [immutable](../../../../glossary.md#term-immutable) A1C probe sources and accepted A1L evidence bundle at
  `/private/tmp/synaptik-cpu-0007a1l-PY33M3lJ`.
- Correct only `S-GENERAL-MIN`: [INT64](../../../../glossary.md#data-type--datatype) `SCATTER_ND` with reduction `MIN`, data/result [Shape](../../../../glossary.md#shape)
  `[16384,16]`, index Shape `[4096,1]`, update Shape `[4096,16]`, duplicate tuples, batch
  dimensions zero, ordered carriers <span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [dimensions glossary](../../../../glossary.md#dimension))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment), `long[]`, `long[]`, `MemorySegment`, and
  offset two-strided data/result layouts.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout))</span>
- Preserve the range-owned scratch-free algorithm: copy each owned output interval from data,
  then traverse all logical updates in canonical row-major order, filter targets to the owned
  interval, and apply signed INT64 <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major))</span>`MIN` in encounter order. Preserve duplicate-target behavior,
  exact represented values, input immutability, canaries, disjoint range ownership, complete
  validation before generated work, scalar or parallel-scalar reuse, and zero workspace.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Treat the frozen optimal direct primitive Java loop as the generated-code design and review
  oracle. A guarded body may exploit only completely proved power-of-two row count, suffix width,
  carrier, offset, stride, range, and tuple/update geometry to remove avoidable general address
  reconstruction while retaining the same copy-then-update algorithm and work.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [carrier glossary](../../../../glossary.md#term-storage-carrier); [stride glossary](../../../../glossary.md#term-stride))</span>
- Require complete runtime geometry, ordered-range, start-address, and sentinel guards before any
  narrowed body. Every unproved scatter family, reduction, type, carrier pattern, mapping, batch
  dimension, tuple depth, suffix, extent, stride, offset, range, or address must retain the
  existing typed general-long implementation.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [batch dimension glossary](../../../../glossary.md#batch-dimension); [tuple depth glossary](../../../../glossary.md#term-tuple-depth); [extent glossary](../../../../glossary.md#term-extent))</span>
- Keep the generated [target](../../../../glossary.md#term-target) final and field-free with exactly one typed static `invoke`, no
  constructor or bridge, and no allocation, reflection, dynamic construct, generic `Object`
  descriptor, Runtime operation dispatch, collection/boxing, or new Synaptik hot-path call.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Advance the generator schema exactly once if emitted bytes change, propagate it to current
  schema assertions, and treat schema 40 and older envelopes as incompatible safe misses without
  migration or dual-schema reuse.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Retain immutable inputs, exact commands, environment, semantics, generated classes, complete
  `javap -c` and `javap -v`, structural/hot-loop reports, five accepted forks, rejected samples,
  XML summaries, context identifiers, inventories, and checksums.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Changing the frozen probe, case, Shapes, carriers, comparator, JVM flags, fork protocol,
  threshold, direct oracle, or semantics.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [fork glossary](../../../../glossary.md#term-fork-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Correcting `X-MIN-MULTI`, reopening any completed row, or closing [CPU](../../../../glossary.md#term-cpu) 0007A1C.
- Changing another production code-shaping owner, scatter IR/lowering, family eligibility,
  reduction meaning, validation, resources, materialization, orchestration, route selection,
  Prepare/Runtime behavior, capabilities, public API, dependencies, build structure, conformance,
  or integration.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation); [materialization glossary](../../../../glossary.md#materialization); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Adding a generic fixed-Shape scatter framework, per-update address table, scratch, atomics,
  sorting/grouping, target reordering, vector/native implementation, dynamic Shape/layout support,
  tuning input, or universal performance claim.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [vector glossary](../../../../glossary.md#term-vector); [layout glossary](../../../../glossary.md#layout))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0006B1 functional scatter`](0006b1-portable-functional-scatter.md)
- [`CPU 0007A0C scatter generated parity`](0007a0c-scatter-generated-loop-parity.md)
- [`CPU 0007A1B scatter algorithmic parity`](0007a1b-scatter-algorithmic-parity.md)
- [`CPU 0007A1C evidence closure`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1L pointwise residual parity`](0007a1l-pointwise-general-loop-residual-parity.md)

## Architecture constraints

- [Model](../../../../glossary.md#term-model) owns scatter and signed [INT64](../../../../glossary.md#data-type--datatype) `MIN` semantics. This task changes only CPU-private
  generated realization and evidence.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- CPU analysis retains scatter lowering, validation, strategy, resource declaration, and route
  selection before shared assignment; CPU finalization realizes one artifact afterward; Runtime
  invokes only prepared direct carriers, cold geometry, and range bounds.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation); [artifact glossary](../../../../glossary.md#term-artifact); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Generated code must preserve the optimal clean Java copy-then-update algorithm, update order,
  hot-loop/dataflow shape, and avoidable-overhead profile while keeping the typed general-long
  form for every unproved case.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape))</span>
- Benchmark observations accept or reject this bounded generated body and never select or mutate
  production settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — existing functional-scatter
  emission and generated-kernel tests.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema) compatibility when bytes change.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing schema assertion and
  current-version package wording only if the version advances.

Packages added or changed: None.

Type placement:

- `CpuScatterEmitter` remains the sole authorized production algorithm/code-shaping [owner](../../../../glossary.md#term-owner).
- No helper, public type, package, or [module](../../../../glossary.md#term-module) may be added.

## Affected files

Expected implementation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- directly affected existing [cache](../../../../glossary.md#term-cache)/code-generation/[prepare](../../../../glossary.md#prepare) package Javadocs only as required;
- `CpuScatterGeneratedKernelTest.java` and `CpuGeneratedDirectEvidenceClosureTest.java` only for
  stable semantic/Class-File assertions;
- `CpuGeneratedKernelArtifactStoreTest.java`, `CpuKernelSpecializationTest.java`,
  `CpuPartitionPreparerTest.java`, and `CpuPartitionFinalizerTest.java` only when schema advances;
  and<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- CPU backend guide, glossary, this task, CPU 0007A1C, CPU master plan, roadmap, and the sole next
  detailed residual task created only after this task completes.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most one production algorithm/code-shaping [owner](../../../../glossary.md#term-owner) may change: `CpuScatterEmitter`. With
[schema](../../../../glossary.md#term-schema)/package propagation, scatter/closure tests, four schema assertion owners, and
documentation/[planning](../../../../glossary.md#term-planning), the overall change must remain at or below 18 repository paths. If a
second production owner, new helper, [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/resource change, another [module](../../../../glossary.md#term-module), or a broader
path set is required, stop and replan.

## Acceptance criteria

- Frozen sources recompile byte-identically and exact semantics remain `VERIFIED,20`.
- `S-GENERAL-MIN` passes `<= 1.15x` in every one of five fresh isolated forks and in the
  cross-fork median.
- Controls `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every [fork](../../../../glossary.md#term-fork-meanings) and [median](../../../../glossary.md#term-median).
- The unchanged full probe exits nonzero only for `X-MIN-MULTI`.
- Exact output values, duplicate-tuple update order, signed [INT64](../../../../glossary.md#data-type--datatype) `MIN`, input immutability,
  canaries, general mapping, arbitrary legal range ownership, and zero-resource contracts remain
  unchanged.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Target Class-File and complete disassembly prove the completely guarded typed scatter
  copy-then-update body and retained fallback, with no avoidable general coordinate reconstruction,
  repeated geometry lookup, division/remainder, generic dispatch, allocation, sorting/grouping,
  atomics, or helper call in the proved loops.<span data-glossary-navigation="true"> ([Target glossary](../../../../glossary.md#term-target); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Focused scatter/closure/[schema](../../../../glossary.md#term-schema) [owners](../../../../glossary.md#term-owner) pass, followed by one uncached [CPU](../../../../glossary.md#term-cpu) suite.
- A distinct clean documentation context finalizes affected Javadocs, guide/glossary impact,
  evidence/status records, and documentation validation without rerunning stable Java tests.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) the focused `CpuScatterGeneratedKernelTest`, `CpuGeneratedDirectEvidenceClosureTest`, and four
[schema](../../../../glossary.md#term-schema)-assertion [owners](../../../../glossary.md#term-owner), followed once by:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Recompile and [checksum](../../../../glossary.md#term-checksum) the frozen sources, run `VERIFIED,20`, retain the [target](../../../../glossary.md#term-target) and all required
control/deferred generated classes and complete disassembly, run five sequential fresh [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) [forks](../../../../glossary.md#term-jvm-fork)
with the unchanged `-Xms1g -Xmx1g` protocol, and regenerate the evidence inventory/checksums. A
complete [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) is accepted only if the target, controls, and every non-deferred row pass; retain
and reject the whole sample otherwise.

The documentation pass runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also checks Markdown links/anchors/fences/final newlines, schema propagation, exact path bounds,
exactly one next [Ready](../../../../glossary.md#term-ready) task, A1C incomplete, A1M Complete only after every gate, [CPU](../../../../glossary.md#term-cpu) 0007A2
Draft/blocked, and empty staging. Repository-wide, architecture, [conformance](../../../../glossary.md#term-conformance), and integration
suites remain deferred to CPU 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci) because no shared contract may change.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0007A1C supplies the [immutable](../../../../glossary.md#term-immutable) twenty-row corpus and remains incomplete.
- CPU 0006B1 and CPU 0007A0C supply functional-scatter semantics and typed generated bodies.
- CPU 0007A1B supplies the range-owned scratch-free copy-then-update algorithm.
- CPU 0007A1D supplies stable invocation-local segment [layout](../../../../glossary.md#layout) handling.
- CPU 0007A1L is Complete at [schema](../../../../glossary.md#term-schema) 40 and supplies the accepted five-[fork](../../../../glossary.md#term-fork-meanings) [residual](../../../../glossary.md#scan-residual-port-and-history-demand) frontier.

## Follow-up tasks

- [CPU 0007A1N](0007a1n-multi-axis-min-residual-parity.md) is the sole next [`Ready`](../../../../glossary.md#term-ready) task and owns
  the retained `X-MIN-MULTI` [residual](../../../../glossary.md#scan-residual-port-and-history-demand) through the aggregate code-shaping boundary.
- [Resume](../../../../glossary.md#schema-serialization-and-checkpoint) [CPU](../../../../glossary.md#term-cpu) 0007A1C closure only after the final persistent row closes.
- CPU 0007A2 remains Draft and blocked behind CPU 0007A1C and all corrective residual work.

## Architecture impact

Expected impact: None.

If the task requires an architecture, [dependency](../../../../glossary.md#term-dependency), public [API](../../../../glossary.md#term-application-programming-interface-api), semantic, numerical-policy,
resource, route, [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime), build, [IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering), [conformance](../../../../glossary.md#term-conformance), or integration change, stop
and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1M. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, CPU task 0007A1M, CPU tasks
0006B1/0007A0C/0007A1B/0007A1C/0007A1L, the relevant scatter/cache/package source and tests, and
the complete A1L evidence bundle. Implement exactly the bounded S-GENERAL-MIN correction through
CpuScatterEmitter. Preserve the immutable probe, INT64 SCATTER_ND + MIN semantics, duplicate
tuples, non-scalar suffix, mixed carriers, general layouts, range-owned copy-then-update order,
arbitrary legal output ranges, zero workspace, typed fallback, and optimal direct-clean-Java
algorithm/dataflow shape. Stop before changing a second production owner or any
IR/lowering/resource/architecture/lifecycle contract.

Run the specified focused/full CPU, semantic/Class-File/five-fork, forbidden-reference, schema,
scope, and preservation gates. Hand the stabilized diff and exact evidence to a distinct clean
documentation context. Do not mark Complete until the target and all controls pass every gate.
Do not commit, push, or stage.
```

## Local decisions

- Accepted A1L [forks](../../../../glossary.md#term-fork-meanings) measure `S-GENERAL-MIN` at `23.171699417x`, `23.581771604x`,
  `23.399670635x`, `23.000646875x`, and `23.311442548x`, with [median](../../../../glossary.md#term-median) `23.311442548x`; the row is
  persistent rather than a one-fork anomaly.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- This row is selected before `X-MIN-MULTI` by the frozen residual order and because its existing
  range-owned copy-then-update algorithm has one sole code-shaping owner. Multi-axis extrema has
  a distinct selected-domain aggregate owner and algorithm risk.<span data-glossary-navigation="true"> ([residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [owner glossary](../../../../glossary.md#term-owner); [axis glossary](../../../../glossary.md#term-axis); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- No other residual shares the functional-scatter owner or copy-then-update algorithm, so this
  remains a one-row task.

## Known limitations

- Ratios apply only to the frozen case, JVM, host, and protocol; they are not a universal scatter,
  general-layout, or MIN performance claim or production tuning input.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [host glossary](../../../../glossary.md#term-host); [layout glossary](../../../../glossary.md#layout))</span>
- This task does not close A1C or authorize [CPU](../../../../glossary.md#term-cpu) 0007A2.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `01a0252d-3ccd-7b42-bb52-8ba5c46b4de8` produced the immutable evidence
  bundle at <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`/private/tmp/synaptik-cpu-0007a1m-7rhBZfwg` from repository base
  `ba34acb330ac77d835a5ceecd197ad78fe976724`. The documentation context verified the bundle's
  SHA-256 manifest, complete reports, five raw fork outputs, generated classes, full <span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [manifest glossary](../../../../glossary.md#term-manifest); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>`javap -c`
  and `javap -v`, frozen sources, XML, implementation patch, context, environment, and unchanged-
  owner inventory.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Frozen inputs and compiled probe classes are byte-identical and exact semantics pass as
  `VERIFIED,20`. The implementation-focused command passed 59 tests in six suites. The one and
  only full CPU run passed 358 tests in 54 suites with zero failures or errors and one expected
  opt-in persistence-evidence skip. No Java test, semantics probe, timing fork, benchmark, or full
  build was rerun by the documentation context because it changed no executable Java behavior.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [run glossary](../../../../glossary.md#term-run); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Five fresh accepted forks measured `S-GENERAL-MIN` at `0.984900063x`, `0.988888234x`,
  `0.983823803x`, `0.978065816x`, and `0.992400680x`; [median](../../../../glossary.md#term-median) `0.984900063x`. The three required
  controls and the already-closed `P-SCALAR-GENERAL` row passed every fork and median. Every full
  probe leaves only `X-MIN-MULTI`, at `8.975696234x`, `8.989556087x`, `8.984536082x`,
  `9.118772524x`, and `9.076912061x`; no rejected [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) occurred.
- Complete [target](../../../../glossary.md#term-target) inspection proves a final, field-free generated class with one typed static
  `invoke(MemorySegment,long[],long[],MemorySegment,long[],long,long)`. Guards cover the exact
  carrier, 50-value geometry, ordered range, start/end, start coordinates, offsets, strides,
  types, and sentinels. The proved path performs a direct linear copy followed by 4,096 tuple and
  16-column suffix loops, loads one index per tuple, filters targets to the owned interval, and
  applies signed <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [strides glossary](../../../../glossary.md#term-stride); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>`Math.min(long,long)` in encounter order. It has no division/remainder,
  allocation, reflection, dynamic constant/invocation, generic <span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>`Object` dispatch, collections,
  boxing, sorting/grouping, atomics, or Synaptik hot call; the unchanged typed general-long body
  begins at the failed-guard target in the same method.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [guard glossary](../../../../glossary.md#term-guard))</span>
- Documentation context `01a02555-44df-7991-adb3-2593b5bc0f97` applied the General, API/Javadoc, Backend Guide, and Planning
  profiles, finalized directly affected internal Javadocs/package summaries, the CPU guide,
  glossary, task evidence, A1C, master plan, roadmap, and the sole next detailed task. It ran CPU
  Javadoc exactly once after final Java documentation edits and passed the recorded Markdown,
  scope, status, staging, and whitespace gates.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning); [scope glossary](../../../../glossary.md#term-scope))</span>

## Implementation notes

- `CpuScatterEmitter` recognizes only the frozen [INT64](../../../../glossary.md#data-type--datatype) `SCATTER_ND + MIN` structure and admits the
  direct path only after complete runtime geometry and range proof. It preserves the existing
  range-owned copy-then-update algorithm, duplicate encounter order, arbitrary legal half-open
  ranges, direct typed carriers, signed long selection, and zero workspace. Every unproved form
  retains the unchanged typed general-long implementation.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [half-open glossary](../../../../glossary.md#term-half-open-interval))</span>
- Generated compatibility advances exactly once from schema 40 to schema 41. Schema 40 and older
  envelopes are incompatible safe misses; no migration, alias, converter, or dual-schema reuse
  was added.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Architecture/current-plan/ADR/architecture-test conclusion: the bounded CPU-private generated
  body changes no architecture rule, module boundary, dependency direction, or shared contract,
  so no architecture contract, focused architecture page, ADR, or architecture test changes.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu); [module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [architecture contract glossary](../../../../glossary.md#architecture-contract); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- API/capability/lifecycle conclusion: public APIs, capability reporting, scatter IR/lowering,
  validation, resources, materialization, orchestration, route selection, Prepare/Runtime
  ownership, and unproved forms are unchanged. Backend conformance and integration tests require
  no edit.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation); [materialization glossary](../../../../glossary.md#materialization); [Prepare glossary](../../../../glossary.md#prepare); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Build/documentation conclusion: no dependency, Gradle/toolchain, shared-module, native/vector,
  example, or public API documentation change is required. The CPU guide and glossary advance
  existing generated-kernel terms to the bounded schema-41 behavior; no new reusable term was
  introduced.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [vector glossary](../../../../glossary.md#term-vector); [kernel glossary](../../../../glossary.md#kernel))</span>

## Completion summary

- Completed changes: added the completely guarded frozen [INT64](../../../../glossary.md#data-type--datatype) `SCATTER_ND + MIN` direct
  copy-then-update body at schema 41, retained the typed fallback, synchronized schema assertions,
  and finalized directly affected internal Javadocs, package summaries, CPU guide, glossary, and
  planning records.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: ten implementation-owned production/test paths plus the CPU guide,
  glossary, this task, CPU 0007A1C, CPU master plan, roadmap, and new CPU 0007A1N task, for 17
  combined repository paths. No executable Java or test changed in the documentation pass.
- Tests and [validation](../../../../glossary.md#term-validation): reused `VERIFIED,20`, six suites/59 focused tests, 54 suites/358 CPU tests
  with one expected skip, deterministic generated-class/Class-File evidence, and five accepted
  target/control forks. The documentation pass ran CPU Javadoc exactly once and passed all
  required documentation, scope, status, staging, and whitespace checks.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean documentation context `01a02555-44df-7991-adb3-2593b5bc0f97` independently verified source,
  tests, retained evidence, Javadocs, guide/glossary impact, and planning synchronization.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Documentation impact: internal emitter/schema/cache/code-generation/prepare Javadocs, CPU guide,
  glossary, this task, A1C, CPU master plan, roadmap, and the sole next detailed task are
  synchronized. No public API or architecture documentation changed.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Javadoc review: affected internal Javadocs now state the complete guard, direct loop, arbitrary-
  range, schema compatibility, zero-resource, and typed-fallback boundaries.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>
- Glossary impact: existing CPU portable-route, specialization, and artifact entries advance to
  schema 41; no new reusable term was introduced.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Unresolved issues: only `X-MIN-MULTI` remains above the frozen parity gate.
- Follow-up required: execute Ready CPU 0007A1N. CPU 0007A1C remains incomplete and CPU 0007A2
  remains Draft/blocked until that final residual closes.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>

Status: Complete
