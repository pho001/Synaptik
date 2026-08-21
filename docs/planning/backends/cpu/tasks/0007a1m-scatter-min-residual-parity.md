# Task 0007A1M: Scatter MIN residual parity

## Status

Ready

## Goal

Correct exactly the persistent `S-GENERAL-MIN` generated/direct residual through its cohesive
functional-scatter code-shaping owner, `CpuScatterEmitter`. Preserve the frozen CPU 0007A1C INT64
`SCATTER_ND + MIN` copy-then-update semantics, duplicate tuples, non-scalar suffix, mixed carriers,
general layouts, arbitrary legal half-open output ranges, and unchanged typed general fallback
while passing the frozen five-fork `<= 1.15x` gate.

## Scope

- Use the immutable A1C probe sources and accepted A1L evidence bundle at
  `/private/tmp/synaptik-cpu-0007a1l-PY33M3lJ`.
- Correct only `S-GENERAL-MIN`: INT64 `SCATTER_ND` with reduction `MIN`, data/result Shape
  `[16384,16]`, index Shape `[4096,1]`, update Shape `[4096,16]`, duplicate tuples, batch
  dimensions zero, ordered carriers `MemorySegment`, `long[]`, `long[]`, `MemorySegment`, and
  offset two-strided data/result layouts.
- Preserve the range-owned scratch-free algorithm: copy each owned output interval from data,
  then traverse all logical updates in canonical row-major order, filter targets to the owned
  interval, and apply signed INT64 `MIN` in encounter order. Preserve duplicate-target behavior,
  exact represented values, input immutability, canaries, disjoint range ownership, complete
  validation before generated work, scalar or parallel-scalar reuse, and zero workspace.
- Treat the frozen optimal direct primitive Java loop as the generated-code design and review
  oracle. A guarded body may exploit only completely proved power-of-two row count, suffix width,
  carrier, offset, stride, range, and tuple/update geometry to remove avoidable general address
  reconstruction while retaining the same copy-then-update algorithm and work.
- Require complete runtime geometry, ordered-range, start-address, and sentinel guards before any
  narrowed body. Every unproved scatter family, reduction, type, carrier pattern, mapping, batch
  dimension, tuple depth, suffix, extent, stride, offset, range, or address must retain the
  existing typed general-long implementation.
- Keep the generated target final and field-free with exactly one typed static `invoke`, no
  constructor or bridge, and no allocation, reflection, dynamic construct, generic `Object`
  descriptor, Runtime operation dispatch, collection/boxing, or new Synaptik hot-path call.
- Advance the generator schema exactly once if emitted bytes change, propagate it to current
  schema assertions, and treat schema 40 and older envelopes as incompatible safe misses without
  migration or dual-schema reuse.
- Retain immutable inputs, exact commands, environment, semantics, generated classes, complete
  `javap -c` and `javap -v`, structural/hot-loop reports, five accepted forks, rejected samples,
  XML summaries, context identifiers, inventories, and checksums.

## Out of scope

- Changing the frozen probe, case, Shapes, carriers, comparator, JVM flags, fork protocol,
  threshold, direct oracle, or semantics.
- Correcting `X-MIN-MULTI`, reopening any completed row, or closing CPU 0007A1C.
- Changing another production code-shaping owner, scatter IR/lowering, family eligibility,
  reduction meaning, validation, resources, materialization, orchestration, route selection,
  Prepare/Runtime behavior, capabilities, public API, dependencies, build structure, conformance,
  or integration.
- Adding a generic fixed-Shape scatter framework, per-update address table, scratch, atomics,
  sorting/grouping, target reordering, vector/native implementation, dynamic Shape/layout support,
  tuning input, or universal performance claim.

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

- Model owns scatter and signed INT64 `MIN` semantics. This task changes only CPU-private
  generated realization and evidence.
- CPU analysis retains scatter lowering, validation, strategy, resource declaration, and route
  selection before shared assignment; CPU finalization realizes one artifact afterward; Runtime
  invokes only prepared direct carriers, cold geometry, and range bounds.
- Generated code must preserve the optimal clean Java copy-then-update algorithm, update order,
  hot-loop/dataflow shape, and avoidable-overhead profile while keeping the typed general-long
  form for every unproved case.
- Benchmark observations accept or reject this bounded generated body and never select or mutate
  production settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — existing functional-scatter
  emission and generated-kernel tests.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema compatibility when bytes change.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing schema assertion and
  current-version package wording only if the version advances.

Packages added or changed: None.

Type placement:

- `CpuScatterEmitter` remains the sole authorized production algorithm/code-shaping owner.
- No helper, public type, package, or module may be added.

## Affected files

Expected implementation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- directly affected existing cache/code-generation/prepare package Javadocs only as required;
- `CpuScatterGeneratedKernelTest.java` and `CpuGeneratedDirectEvidenceClosureTest.java` only for
  stable semantic/Class-File assertions;
- `CpuGeneratedKernelArtifactStoreTest.java`, `CpuKernelSpecializationTest.java`,
  `CpuPartitionPreparerTest.java`, and `CpuPartitionFinalizerTest.java` only when schema advances;
  and
- CPU backend guide, glossary, this task, CPU 0007A1C, CPU master plan, roadmap, and the sole next
  detailed residual task created only after this task completes.

## Maximum scope

At most one production algorithm/code-shaping owner may change: `CpuScatterEmitter`. With
schema/package propagation, scatter/closure tests, four schema assertion owners, and
documentation/planning, the overall change must remain at or below 18 repository paths. If a
second production owner, new helper, IR/lowering/resource change, another module, or a broader
path set is required, stop and replan.

## Acceptance criteria

- Frozen sources recompile byte-identically and exact semantics remain `VERIFIED,20`.
- `S-GENERAL-MIN` passes `<= 1.15x` in every one of five fresh isolated forks and in the
  cross-fork median.
- Controls `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every fork and median.
- The unchanged full probe exits nonzero only for `X-MIN-MULTI`.
- Exact output values, duplicate-tuple update order, signed INT64 `MIN`, input immutability,
  canaries, general mapping, arbitrary legal range ownership, and zero-resource contracts remain
  unchanged.
- Target Class-File and complete disassembly prove the completely guarded typed scatter
  copy-then-update body and retained fallback, with no avoidable general coordinate reconstruction,
  repeated geometry lookup, division/remainder, generic dispatch, allocation, sorting/grouping,
  atomics, or helper call in the proved loops.
- Focused scatter/closure/schema owners pass, followed by one uncached CPU suite.
- A distinct clean documentation context finalizes affected Javadocs, guide/glossary impact,
  evidence/status records, and documentation validation without rerunning stable Java tests.

## Tests / validation

Run the focused `CpuScatterGeneratedKernelTest`, `CpuGeneratedDirectEvidenceClosureTest`, and four
schema-assertion owners, followed once by:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Recompile and checksum the frozen sources, run `VERIFIED,20`, retain the target and all required
control/deferred generated classes and complete disassembly, run five sequential fresh JVM forks
with the unchanged `-Xms1g -Xmx1g` protocol, and regenerate the evidence inventory/checksums. A
complete sample is accepted only if the target, controls, and every non-deferred row pass; retain
and reject the whole sample otherwise.

The documentation pass runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also checks Markdown links/anchors/fences/final newlines, schema propagation, exact path bounds,
exactly one next Ready task, A1C incomplete, A1M Complete only after every gate, CPU 0007A2
Draft/blocked, and empty staging. Repository-wide, architecture, conformance, and integration
suites remain deferred to CPU 0009 or CI because no shared contract may change.

## Dependencies

- CPU 0007A1C supplies the immutable twenty-row corpus and remains incomplete.
- CPU 0006B1 and CPU 0007A0C supply functional-scatter semantics and typed generated bodies.
- CPU 0007A1B supplies the range-owned scratch-free copy-then-update algorithm.
- CPU 0007A1D supplies stable invocation-local segment layout handling.
- CPU 0007A1L is Complete at schema 40 and supplies the accepted five-fork residual frontier.

## Follow-up tasks

- `X-MIN-MULTI` remains Draft and unassigned until accepted A1M evidence selects its exact bounded
  owner-cohesive correction.
- Resume CPU 0007A1C closure only after the final persistent row closes.
- CPU 0007A2 remains Draft and blocked behind CPU 0007A1C and all corrective residual work.

## Architecture impact

Expected impact: None.

If the task requires an architecture, dependency, public API, semantic, numerical-policy,
resource, route, Prepare/Runtime, build, IR, lowering, conformance, or integration change, stop
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

- Accepted A1L forks measure `S-GENERAL-MIN` at `23.171699417x`, `23.581771604x`,
  `23.399670635x`, `23.000646875x`, and `23.311442548x`, with median `23.311442548x`; the row is
  persistent rather than a one-fork anomaly.
- This row is selected before `X-MIN-MULTI` by the frozen residual order and because its existing
  range-owned copy-then-update algorithm has one sole code-shaping owner. Multi-axis extrema has
  a distinct selected-domain aggregate owner and algorithm risk.
- No other residual shares the functional-scatter owner or copy-then-update algorithm, so this
  remains a one-row task.

## Known limitations

- Ratios apply only to the frozen case, JVM, host, and protocol; they are not a universal scatter,
  general-layout, or MIN performance claim or production tuning input.
- This task does not close A1C or authorize CPU 0007A2.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
