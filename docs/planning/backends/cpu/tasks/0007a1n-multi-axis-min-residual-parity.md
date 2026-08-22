# Task 0007A1N: Multi-axis MIN residual parity

## Status

Complete

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

- CPU 0007A1C supplies the immutable twenty-row corpus and remains Incomplete after final
  reconciliation because its frozen ledger violates the original structural-only limit.
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

- Implementation context `01a02611-1b99-7890-b618-6d7b659b1853` produced the retained evidence
  bundle at `/private/tmp/synaptik-cpu-0007a1n-26FYzb9w` from repository base
  `47569dbedcf3474d36378fcf2a5d9ec4c66ac95a`. The documentation context verified the bundle's
  checksum manifest before relying on its reports, immutable inputs, generated classes, complete
  disassembly, raw samples, XML, and implementation patch.
- The frozen `ClosureCases.java`, `ClosureProbe.java`, and `ExactClosure.java` sources and all 14
  compiled probe classes are byte-identical to A1M. Exact semantic verification reports
  `VERIFIED,20`. The focused implementation command passed 62 tests in six suites. The one
  authoritative CPU run passed 360 tests in 54 suites with zero failures or errors and one
  expected opt-in persistence-evidence skip. No Java test, semantic probe, performance fork,
  generation benchmark, or decompilation command was rerun by the documentation context.
- Five fresh accepted sequential JVM forks were forks 1, 2, 3, 5, and 6. `X-MIN-MULTI` measured
  `0.818775794x`, `0.811182115x`, `0.811942840x`, `0.810720356x`, and `0.810918826x`, with median
  `0.811182115x`. Every one of the twenty rows and every cross-fork median was at most `1.15x`.
  Fork 4 was rejected as one whole sample because unrelated `M-CONCAT` measured `2.537201532x`;
  its raw output and generated classes remain retained and were excluded from accepted ratios.
- All twenty generated classes are byte-identical across final semantic verification, the five
  accepted forks, and the rejected fork. Complete `javap -c` and `javap -v` output is retained for
  all twenty classes. The target is final and field-free with exactly one static
  `invoke(MemorySegment, short[], long[], long, long)` method. Its guards cover the ordered range
  and all 36 required geometry values before the first specialized store. The proved offsets
  `394..625` contain the direct output-cell/axis-0/axis-2 traversal, direct unaligned short loads,
  raw first-NaN and negative-zero MIN selection, and one raw short store after 4,096 visits. They
  contain no division, remainder, allocation, generic dispatch, or Synaptik helper call. Failed
  guards branch to offset 628, the unchanged typed general-long realization.
- Generated compatibility advances exactly once from schema 41 to schema 42. Schema-41 and older
  envelopes are incompatible safe misses, and the retained negative persistence case regenerates
  a schema-42 envelope without migration, aliasing, or dual-schema reuse.
- CPU Javadoc and the final static documentation, planning, schema, scope, staging, checksum, and
  whitespace checks are recorded in the completion summary after the documentation pass.

## Implementation notes

- `CpuAggregateEmitter` is the only changed production algorithm/code-shaping owner. It recognizes
  only BFLOAT16 `MIN`, input Shape `[64,64,64]`, axes `[0,2]`, kept output `[1,64,1]`, domain
  4,096, `MemorySegment` input, offset two-strided `short[]` output, and the complete retained
  geometry. Range and all geometry/address sentinels are checked before the specialized body can
  write.
- One range ordinal owns one middle-axis output cell. Axis 0 is the outer selected loop and axis 2
  the inner loop. The accumulator starts from the first represented factor, preserves the first
  raw BFLOAT16 NaN, chooses negative zero when both zero signs occur, otherwise selects the smaller
  represented value, and stores its raw 16 bits once. The body uses no workspace, materialization,
  partial/combine state, selected-domain buffer, or reordered visits.
- Every unproved aggregate kind, type, form, axes, rank, carrier pattern, extent, layout, range, or
  address retains the existing typed general-long body in the same generated method.
- Architecture/current-plan/ADR conclusion: the bounded CPU-private emitted body changes no
  architecture authority, module boundary, dependency direction, or shared contract, so
  `ARCHITECTURE.md`, focused architecture pages, ADRs, and architecture tests require no edit.
- API/lifecycle/test/build conclusion: public APIs, capabilities, aggregate IR/lowering,
  validation ownership, resources, materialization policy, route selection, Prepare/Runtime
  ownership, conformance/integration coverage, Gradle/toolchain configuration, examples, and
  other modules are unchanged. Existing focused aggregate and full CPU coverage supplied the
  executable evidence; no test or build source required documentation-pass edits.

## Completion summary

- Completed changes: added the completely guarded frozen BFLOAT16 multi-axis MIN primitive body,
  retained the typed general-long fallback, advanced current-only generated compatibility to
  schema 42, synchronized schema assertions, and finalized affected internal Javadocs, the CPU
  guide, glossary, task evidence, A1C closure, CPU master plan, and roadmap.
- Files changed or created: ten implementation-owned production/test paths plus the CPU backend
  guide, glossary, this task, CPU 0007A1C, CPU master plan, and roadmap, for 16 repository paths.
  No executable Java or test changed in the documentation pass.
- Validation: retained `VERIFIED,20`, six suites/62 focused tests, 54 suites/360 CPU tests with one
  expected skip, deterministic generated-Class-File evidence, complete disassembly, five accepted
  twenty-row forks, and one retained rejected whole sample. The documentation pass ran CPU
  Javadoc exactly once and passed the required static checks.
- Documentation impact: affected emitter/schema/cache/code-generation/prepare Javadocs, the CPU
  backend guide, glossary, this task, A1C, master plan, and roadmap are synchronized to schema 42.
  No public API, architecture contract, ADR, example, or other-module documentation changed.
- Planning result: A1N is Complete, but A1C remains Incomplete because its immutable ledger labels
  non-INITIAL_STATE, non-view pointwise rows `STRUCTURAL_ONLY:covered-by-emitted-category`, contrary
  to A1C's original structural-only limit. CPU 0007A2 therefore remains Draft/blocked; no CPU task
  is Ready in this pass.
- Unresolved issues: none for CPU 0007A1N. A1C requires an evidence-authorized ledger correction
  or replacement and a fresh closure audit before CPU 0007A2 planning can advance.
- Follow-up: reconcile the A1C ledger in a separate evidence-authorized task; only after A1C closes
  should a detailed CPU 0007A2 specification be created and considered for Ready.

Status: Complete
