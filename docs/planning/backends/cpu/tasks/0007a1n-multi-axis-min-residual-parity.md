# Task 0007A1N: Multi-axis MIN residual parity

## Status

Ready

## Goal

Correct exactly the final persistent `X-MIN-MULTI` generated/direct residual through its cohesive
ordinary-aggregate code-shaping owner, `CpuAggregateEmitter`. Preserve the frozen CPU 0007A1C
BFLOAT16 multi-axis `MIN` semantics, kept Dimensions, first-NaN and signed-zero selection,
segment input, offset two-strided `short[]` output, canonical selected-domain order, arbitrary
legal complete-output-cell ranges, zero workspace, and unchanged typed general-long fallback while
passing the frozen five-fork `<= 1.15x` gate.

## Scope

- Use the immutable twenty-row probe sources and accepted A1M evidence bundle at
  `/private/tmp/synaptik-cpu-0007a1m-7rhBZfwg`.
- Correct only `X-MIN-MULTI`: BFLOAT16 ordinary `MIN` over input Shape `[64,64,64]`, selected axes
  `[0,2]`, `keepDimensions == true`, output Shape `[1,64,1]`, ordered carriers `MemorySegment` and
  `short[]`, dense input, and offset two-strided output.
- Preserve one complete output cell per range ordinal. For each kept middle-axis coordinate, visit
  the selected `[axis 0, axis 2]` domain in canonical logical row-major order, select the first
  represented NaN, choose negative zero when both zero signs occur, otherwise select the smaller
  represented BFLOAT16 value, and store the selected raw 16-bit representation once.
- Treat the frozen optimal direct primitive Java loop as the generated-code design and review
  oracle. A guarded body may exploit only completely proved power-of-two extents, carrier, offset,
  stride, selected axes, kept-Dimension form, range, and address geometry to remove avoidable
  general coordinate reconstruction while retaining identical domain visits and dataflow.
- Require complete runtime geometry, ordered-range, start-address, and sentinel guards before any
  narrowed body. Every unproved aggregate kind, type, form, axes, rank, carrier pattern, extent,
  layout, range, or address must retain the existing typed general-long implementation.
- Keep the generated target final and field-free with exactly one typed static `invoke`, no
  constructor or bridge, and no allocation, reflection, dynamic construct, generic `Object`
  descriptor, Runtime semantic dispatch, collection/boxing, or new Synaptik hot-path call.
- Advance the generator schema exactly once if emitted bytes change, propagate it to current
  schema assertions, and treat schema 41 and older envelopes as incompatible safe misses without
  migration or dual-schema reuse.
- Retain immutable inputs, exact commands, environment, semantics, generated classes, complete
  `javap -c` and `javap -v`, structural/hot-loop reports, five accepted forks, rejected samples,
  XML summaries, context identifiers, inventories, and checksums.

## Out of scope

- Changing the frozen probe, case, Shapes, carriers, comparator, JVM flags, fork protocol,
  threshold, direct oracle, or semantics.
- Reopening any completed row or marking CPU 0007A1C Complete before all its original closure
  gates are explicitly reconciled after this residual passes.
- Changing another production code-shaping owner, aggregate IR/lowering, family eligibility,
  reduction meaning, validation, resources, materialization, orchestration, route selection,
  Prepare/Runtime behavior, capabilities, public API, dependencies, build structure, conformance,
  or integration.
- Adding a generic fixed-Shape aggregate framework, scratch, partial reductions, combine state,
  selected-domain reordering, vector/native implementation, dynamic Shape/layout support, tuning
  input, or universal performance claim.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007A ordinary extrema`](0007a-portable-ordinary-extrema-and-boolean-reductions.md)
- [`CPU 0007A1C evidence closure`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1F BOOL aggregate residual`](0007a1f-bool-movement-and-aggregate-residual-parity.md)
- [`CPU 0007A1H numerical aggregate residual`](0007a1h-numerical-aggregate-residual-parity.md)
- [`CPU 0007A1M scatter MIN residual`](0007a1m-scatter-min-residual-parity.md)

## Architecture constraints

- Model owns multi-axis reduction and BFLOAT16 `MIN` semantics. This task changes only CPU-private
  generated realization and evidence.
- CPU analysis retains aggregate lowering, validation, strategy, resource declaration, and route
  selection before shared assignment; CPU finalization realizes one artifact afterward; Runtime
  invokes only prepared direct carriers, cold geometry, and range bounds.
- Generated code must preserve the optimal clean Java selected-domain algorithm, visit order,
  first-NaN/signed-zero policy, hot-loop/dataflow shape, and avoidable-overhead profile while
  keeping the typed general-long form for every unproved case.
- Benchmark observations accept or reject this bounded generated body and never select or mutate
  production settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — existing ordinary-aggregate
  emission and generated-kernel tests.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema compatibility when bytes change.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing schema assertion and
  current-version package wording only if the version advances.

Packages added or changed: None.

Type placement:

- `CpuAggregateEmitter` remains the sole authorized production algorithm/code-shaping owner.
- No helper, public type, package, or module may be added.

## Affected files

Expected implementation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- directly affected existing cache/code-generation/prepare package Javadocs only as required;
- `CpuAggregateGeneratedKernelTest.java` and `CpuGeneratedDirectEvidenceClosureTest.java` only for
  stable semantic/Class-File assertions;
- `CpuGeneratedKernelArtifactStoreTest.java`, `CpuKernelSpecializationTest.java`,
  `CpuPartitionPreparerTest.java`, and `CpuPartitionFinalizerTest.java` only when schema advances;
  and
- CPU backend guide, glossary, this task, CPU 0007A1C, CPU master plan, and roadmap.

## Maximum scope

At most one production algorithm/code-shaping owner may change: `CpuAggregateEmitter`. With
schema/package propagation, aggregate/closure tests, four schema assertion owners, and
documentation/planning, the overall change must remain at or below 18 repository paths. If a
second production owner, new helper, IR/lowering/resource change, another module, or a broader
path set is required, stop and replan.

## Acceptance criteria

- Frozen sources recompile byte-identically and exact semantics remain `VERIFIED,20`.
- `X-MIN-MULTI` passes `<= 1.15x` in every one of five fresh isolated forks and in the cross-fork
  median.
- Controls `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every fork and median.
- The unchanged full probe has no ratio failure.
- Exact represented output bits, canonical selected-domain order, first-NaN and signed-zero
  behavior, input immutability, canaries, general mapping, arbitrary legal range ownership, and
  zero-resource contracts remain unchanged.
- Target Class-File and complete disassembly prove the completely guarded typed multi-axis
  BFLOAT16 `MIN` body and retained fallback, with no avoidable general coordinate reconstruction,
  repeated geometry lookup, division/remainder, generic dispatch, allocation, helper call, or
  selected-domain reordering in the proved loops.
- Focused aggregate/closure/schema owners pass, followed by one uncached CPU suite.
- A distinct clean documentation context finalizes affected Javadocs, guide/glossary impact,
  evidence/status records, and documentation validation without rerunning stable Java tests.

## Tests / validation

Run the focused `CpuAggregateGeneratedKernelTest`, `CpuGeneratedDirectEvidenceClosureTest`, and
four schema-assertion owners, followed once by:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Recompile and checksum the frozen sources, run `VERIFIED,20`, retain the target and all required
control generated classes and complete disassembly, run five sequential fresh JVM forks with the
unchanged `-Xms1g -Xmx1g` protocol, and regenerate the evidence inventory/checksums. A complete
sample is accepted only if the target, controls, and every row pass; retain and reject the whole
sample otherwise.

The documentation pass runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also checks Markdown links/anchors/fences/final newlines, schema propagation, exact path bounds,
no second Ready task, A1C status against its original closure gates, A1N Complete only after every
gate, CPU 0007A2 Draft/blocked until A1C closes, and empty staging. Repository-wide,
architecture, conformance, and integration suites remain deferred to CPU 0009 or CI because no
shared contract may change.

## Dependencies

- CPU 0007A1C supplies the immutable twenty-row corpus and remains incomplete.
- CPU 0007A and CPU 0007A1A supply ordinary extrema semantics and the self-contained typed
  aggregate selection body.
- CPU 0007A1D supplies stable invocation-local segment layout handling.
- CPU 0007A1F and CPU 0007A1H supply the accepted guarded aggregate-loop precedent.
- CPU 0007A1M is Complete at schema 41 and supplies the accepted final-residual frontier.

## Follow-up tasks

- Reconcile and close CPU 0007A1C only after this final persistent row and all original closure
  evidence gates pass.
- CPU 0007A2 remains Draft and blocked behind CPU 0007A1C and all corrective residual work.

## Architecture impact

Expected impact: None.

If the task requires an architecture, dependency, public API, semantic, numerical-policy,
resource, route, Prepare/Runtime, build, IR, lowering, conformance, or integration change, stop
and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1N. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, CPU task 0007A1N, CPU tasks
0007A/0007A1C/0007A1F/0007A1H/0007A1M, the relevant aggregate/cache/package source and tests,
and the complete A1M evidence bundle. Implement exactly the bounded X-MIN-MULTI correction through
CpuAggregateEmitter. Preserve the immutable probe, BFLOAT16 multi-axis MIN semantics, kept
Dimensions, first-NaN/signed-zero policy, segment input, strided short-array output, canonical
selected-domain order, arbitrary legal output-cell ranges, zero workspace, typed fallback, and
optimal direct-clean-Java algorithm/dataflow shape. Stop before changing a second production owner
or any IR/lowering/resource/architecture/lifecycle contract.

Run the specified focused/full CPU, semantic/Class-File/five-fork, forbidden-reference, schema,
scope, and preservation gates. Hand the stabilized diff and exact evidence to a distinct clean
documentation context. Do not mark Complete until the target, controls, and complete unchanged
probe pass every gate. Do not commit, push, or stage.
```

## Local decisions

- Accepted A1M forks measure `X-MIN-MULTI` at `8.975696234x`, `8.989556087x`, `8.984536082x`,
  `9.118772524x`, and `9.076912061x`; the row is persistent rather than a one-fork anomaly.
- This is the only residual in every accepted A1M fork. Its selected-domain ordinary aggregate
  algorithm and sole code-shaping owner are distinct from A1M's functional-scatter algorithm.
- The direct oracle uses one kept-axis output loop and nested selected-axis loops with raw BFLOAT16
  selection. The task must match that algorithm and work, not merely produce equivalent values.

## Known limitations

- Ratios apply only to the frozen case, JVM, host, and protocol; they are not a universal
  aggregate, BFLOAT16, multi-axis, or MIN performance claim.
- Passing this task does not automatically close A1C; the original closure evidence and status
  must be reconciled explicitly before CPU 0007A2 can advance.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
