# Task 0007A1L: Pointwise general-loop residual parity

## Status

Complete

## Goal

Correct exactly the persistent `P-SCALAR-GENERAL` generated/direct residual through the cohesive
pointwise loop-shaping boundary in `CpuClassFileKernelGenerator` and `CpuLoopEmitter`. Preserve the
frozen CPU 0007A1C FLOAT32 `DIV -> SIGMOID -> MUL` semantics, mixed carriers, broadcast input,
strided output, arbitrary legal half-open ranges, and unchanged general-long fallback while
passing the frozen five-fork `<= 1.15x` gate.

## Scope

- Use the immutable A1C probe sources and the accepted A1K evidence bundle at
  `/private/tmp/synaptik-cpu-0007a1k-tSVmcoQh`.
- Correct only `P-SCALAR-GENERAL`: the fused FLOAT32 `DIV -> SIGMOID -> MUL` chain over Shape
  `[512,512]`, with ordered carriers `MemorySegment`, `float[]`, `MemorySegment`, `float[]`;
  dense first and third inputs; a last-axis broadcast second input; and an offset, two-strided
  general output.
- Preserve exact FLOAT32 division and multiplication and the existing stable SIGMOID formula,
  including its binary64 exponential work, sign branch, and one final narrowing. Do not substitute
  a different activation, approximation, vector body, formula order, or numerical policy.
- Preserve arbitrary legal half-open output ranges, input immutability, canaries, output-domain
  ownership, complete overlap rejection, scalar or parallel-scalar reuse, and zero workspace.
- Treat the frozen optimal direct primitive Java loop as the generated-code design and review
  oracle. The guarded body may derive the row and column from an integer ordinal using the proved
  power-of-two Shape, then use direct typed carrier loads/stores and the existing scalar semantic
  emitter.
- Require complete runtime geometry, range, and sentinel guards before entering any narrowed body.
  Every unproved pointwise topology, opcode sequence, carrier pattern, access form, extent, stride,
  offset, or range must retain the existing typed general-long state machine.
- Keep the generated target final and field-free with exactly one typed static `invoke`, no
  constructor or bridge, and no allocation, reflection, dynamic construct, generic `Object`
  descriptor, Runtime operation dispatch, or new Synaptik hot-path call.
- Advance the generator schema exactly once if emitted bytes change, propagate it to current
  schema assertions, and treat schema 39 and older envelopes as incompatible safe misses without
  migration or dual-schema reuse.
- Retain immutable inputs, exact commands, environment, semantics, generated classes, complete
  `javap -c` and `javap -v`, structural/hot-loop reports, five accepted forks, rejected samples,
  XML summaries, context identifiers, inventories, and checksums.

## Out of scope

- Changing the frozen probe, case, Shape, carriers, comparator, JVM flags, fork protocol,
  threshold, direct oracle, or semantics.
- Correcting `S-GENERAL-MIN`, `X-MIN-MULTI`, or any completed row.
- Changing `CpuScalarEmitter` formulas, `CpuCarrierEmitter`, another family emitter, pointwise IR
  or lowering, fusion eligibility, access-plan meaning, validation, resources, materialization,
  orchestration, route selection, Prepare/Runtime behavior, capabilities, public API,
  dependencies, build structure, conformance, or integration.
- Adding a generic fixed-Shape pointwise framework, per-element address table, workspace,
  materialization route, vector/native implementation, dynamic Shape/layout support, tuning
  input, or universal performance claim.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0005B access plans`](0005b-universal-access-plans-and-right-aligned-broadcasting.md)
- [`CPU 0005H unary closure`](0005h-portable-unary-transcendental-and-activation-closure.md)
- [`CPU 0007A0 generated parity`](0007a0-generated-hot-path-parity-correction.md)
- [`CPU 0007A1C evidence closure`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1K affine-copy residual parity`](0007a1k-affine-copy-residual-parity.md)

## Architecture constraints

- Model owns pointwise meaning and numerical semantics. This task changes only CPU-private
  generated loop realization and evidence.
- CPU analysis retains fusion, access-plan lowering, strategy, resource declaration, and route
  selection before shared assignment; CPU finalization realizes one artifact afterward; Runtime
  only invokes prepared direct carriers, cold geometry, and range bounds.
- Generated code must preserve the optimal clean Java algorithm and hot-loop/dataflow shape while
  keeping the general-long form for all unproved cases.
- Benchmark observations accept or reject this bounded generated body and never select or mutate
  production settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — existing pointwise class
  assembly, access-loop state machines, scalar semantic emission, and generated-kernel tests.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema compatibility when bytes change.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing schema assertion and
  current-version package wording only if the version advances.

Packages added or changed: None.

Type placement:

- `CpuClassFileKernelGenerator` remains the pointwise body assembler and may select only the
  completely proved frozen form before delegating its address state.
- `CpuLoopEmitter` remains the owner of generated pointwise loop/address state.
- `CpuScalarEmitter` remains the unchanged owner of the `DIV`, stable `SIGMOID`, and `MUL`
  formulas.
- No helper, public type, package, or module may be added.

## Affected files

Expected implementation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLoopEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- directly affected existing cache and prepare package Javadocs only as required;
- `CpuPointwiseGeneratedKernelTest.java` and `CpuGeneratedDirectEvidenceClosureTest.java` only for
  stable semantic/Class-File assertions;
- `CpuGeneratedKernelArtifactStoreTest.java`, `CpuKernelSpecializationTest.java`,
  `CpuPartitionPreparerTest.java`, and `CpuPartitionFinalizerTest.java` only when schema advances;
  and
- CPU backend guide, glossary, this task, CPU 0007A1C, CPU master plan, roadmap, and the sole next
  detailed residual task created only after this task completes.

## Maximum scope

At most two production code-shaping owners may change: the pointwise body assembler and loop
state owner named above. With schema/package propagation, pointwise/closure tests, four schema
assertion owners, and documentation/planning, the overall change must remain at or below 18
repository paths. If `CpuScalarEmitter`, another production owner, IR/lowering/resource code, a
new helper, another module, or a broader path set is required, stop and replan.

## Acceptance criteria

- Frozen sources recompile byte-identically and exact semantics remain `VERIFIED,20`.
- `P-SCALAR-GENERAL` passes `<= 1.15x` in every one of five fresh isolated forks and in the
  cross-fork median.
- Controls `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every fork and median.
- The unchanged full probe exits nonzero only for `S-GENERAL-MIN` and `X-MIN-MULTI`.
- Exact output bits, input immutability, canaries, broadcast behavior, strided output mapping,
  arbitrary legal range ownership, and zero-resource contracts remain unchanged.
- Target Class-File and complete disassembly prove the completely guarded typed pointwise loop and
  retained fallback, with no avoidable general odometer, repeated geometry lookup, coordinate
  division/remainder, generic dispatch, allocation, or helper call in the proved address loop.
  Calls required by the unchanged stable SIGMOID formula remain permitted.
- Focused pointwise/closure/schema owners pass, followed by one uncached CPU suite.
- A distinct clean documentation context finalizes affected Javadocs, guide/glossary impact,
  evidence/status records, and documentation validation without rerunning stable Java tests.

## Tests / validation

Run the focused `CpuPointwiseGeneratedKernelTest`, `CpuGeneratedDirectEvidenceClosureTest`, and
four schema-assertion owners, followed once by:

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
exactly one next Ready task, A1C incomplete, A1L Complete only after every gate, CPU 0007A2
Draft/blocked, and empty staging. Repository-wide, architecture, conformance, and integration
suites remain deferred to CPU 0009 or CI because no shared contract may change.

## Dependencies

- CPU 0007A1C supplies the immutable twenty-row corpus and remains incomplete.
- CPU 0005B and CPU 0007A0 supply the general access state machine and generated pointwise parity
  foundation.
- CPU 0005H supplies the exact stable SIGMOID formula and exceptional-value behavior.
- CPU 0007A1D supplies stable invocation-local segment layout handling.
- CPU 0007A1K is Complete at schema 39 and supplies the accepted five-fork residual frontier.

## Follow-up tasks

- Detailed [CPU 0007A1M scatter-MIN residual parity](0007a1m-scatter-min-residual-parity.md) is
  the sole next Ready task and owns `S-GENERAL-MIN`. `X-MIN-MULTI` remains Draft and unassigned.
- Resume CPU 0007A1C closure only after both remaining persistent rows close.
- CPU 0007A2 remains Draft and blocked behind CPU 0007A1C and all corrective residual work.

## Architecture impact

Expected impact: None.

If the task requires an architecture, dependency, public API, semantic, numerical-policy,
resource, route, Prepare/Runtime, build, IR, lowering, conformance, or integration change, stop
and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1L. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, CPU task 0007A1L, CPU tasks
0005B/0005H/0007A0/0007A1C/0007A1K, the relevant pointwise loop/generator/cache/package source and
tests, and the complete A1K evidence bundle. Implement exactly the bounded
P-SCALAR-GENERAL correction through CpuClassFileKernelGenerator and CpuLoopEmitter. Preserve the
immutable probe, exact FLOAT32 DIV/stable-SIGMOID/MUL formula, mixed carriers, last-axis
broadcast, strided output, arbitrary legal ranges, zero resources, typed fallback, and optimal
direct-clean-Java algorithm/dataflow shape. Stop before changing CpuScalarEmitter, a third
production owner, or any IR/lowering/resource/architecture/lifecycle contract.

Run the specified focused/full CPU, semantic/Class-File/five-fork, forbidden-reference, schema,
scope, and preservation gates. Hand the stabilized diff and exact evidence to a distinct clean
documentation context. Do not mark Complete until the target and all controls pass every gate.
Do not commit, push, or stage.
```

## Local decisions

- Accepted A1K forks measure `P-SCALAR-GENERAL` at `2.641514165x`, `2.609853653x`,
  `2.607747319x`, `2.638802714x`, and `2.642773957x`, with median `2.638802714x`; the row is
  persistent rather than a one-fork anomaly.
- This row is selected before the two remaining residuals because it is first in the frozen
  residual order and has one cohesive pointwise loop-shaping boundary. The unchanged scalar
  semantic emitter already matches the direct formula; the defect evidence is confined to the
  generated general address state around that formula.
- Scatter MIN and multi-axis aggregate MIN retain distinct copy/update/reduction and selected-
  domain algorithms and must not enter this task.

## Known limitations

- Ratios apply only to the frozen case, JVM, host, and protocol; they are not a universal
  pointwise, general-layout, or activation performance claim.
- This task does not close A1C or authorize CPU 0007A2.

## Validation evidence

- Implementation context `01a02438-4fe2-7b40-b70e-2664d29a4095` produced the immutable evidence
  bundle at `/private/tmp/synaptik-cpu-0007a1l-PY33M3lJ` from repository base
  `d51dcfc8c00a1886596dd95545f99d1dc607c7fb`. The complete manifests, checksums, frozen-source and
  probe preservation reports, XML summaries, five raw fork outputs and summary, generated target
  class, complete `javap -c` and `javap -v`, direct oracle, schema transition, implementation
  patch, context, commands, environment, and unchanged-owner reports were independently reviewed.
- Frozen semantics passed exactly (`VERIFIED,20`). The focused invocation passed 62 tests in six
  suites. The one and only full CPU rerun passed 356 of 357 tests in 54 suites with zero failures
  or errors and one expected skip. No Java suite, semantic probe, timing fork, benchmark, or
  generated-class corpus was rerun by the documentation context.
- Five accepted isolated forks measured `P-SCALAR-GENERAL` at `1.006810905x`, `1.005707592x`,
  `1.006301612x`, `1.004126974x`, and `1.006512287x`; median `1.006301612x`. All three controls
  passed every fork and median. Every accepted full fork leaves exactly `S-GENERAL-MIN` and
  `X-MIN-MULTI`; no rejected sample occurred in this final run.
- Complete target disassembly proves a final, field-free class with one typed static
  `invoke(MemorySegment, float[], MemorySegment, float[], long[], long, long)`. Complete guards
  precede the ordinal loop; its address body has shifts and masks but no integer or long division
  or remainder, performs direct `JAVA_FLOAT_UNALIGNED` loads and a strided `float[]` store, and
  preserves FLOAT32 division/multiplication around the sign-branch binary64
  `StrictMath.exp` sigmoid with one final narrowing. The unchanged typed general-long body begins
  at the failed-guard target in the same method.
- Documentation context `/root/doc_0007a1l` independently finalized affected internal Javadocs,
  package summaries, the CPU guide, glossary, and synchronized planning without changing
  executable Java or tests. It ran CPU Javadoc exactly once after edits and passed Markdown,
  schema/current-only wording, status/order, exact-scope, frozen-evidence, implementation-path
  preservation, staging, and whitespace checks recorded in the completion summary.

## Implementation notes

- `CpuClassFileKernelGenerator` recognizes only the exact frozen scalar general pointwise IR:
  FLOAT32 `[512,512]`, `DIV -> SIGMOID -> MUL`, ordered carriers `MemorySegment`, `float[]`,
  `MemorySegment`, `float[]`, dense first/third inputs, a last-axis-broadcast second input, and one
  offset two-strided output. `CpuLoopEmitter` admits the direct body only after the complete
  24-value geometry, ordered range, start coordinate/address, extent, stride, and sentinel proof.
- The proved loop narrows the bounded ordinal and end once, derives row/column with shift/mask
  operations, and performs direct typed carrier access. It preserves the existing stable sigmoid:
  FLOAT32 division, sign branch, binary64 `StrictMath.exp`, one final FLOAT32 narrowing, then
  FLOAT32 multiplication. Every legal half-open range is supported; empty ranges return without
  work, and every failed proof retains the unchanged typed general-long state machine.
- Generated compatibility advances exactly once to schema 40. Schema 39 and older envelopes are
  incompatible safe misses; no migration, alias, converter, or dual-schema reuse was added.
- Architecture/current-plan/ADR/architecture-test conclusion: the bounded CPU-private generated
  body changes no architecture rule, module boundary, dependency direction, or shared contract,
  so no architecture contract, explanatory architecture page, ADR, or architecture test changes.
- API/capability/lifecycle conclusion: public APIs and capability reporting, IR/lowering,
  materialization and resource declarations, route selection, Prepare/Runtime ownership, and
  unproved forms are unchanged. Backend conformance and integration tests therefore need no edit.
- Build/documentation conclusion: no dependency, Gradle/toolchain, shared-module, native/vector,
  example, or public API documentation change is required. The CPU guide and glossary record only
  the bounded current schema-40 behavior; affected internal Javadocs preserve its exact guard,
  numerical, range, addressing, and fallback boundaries.

## Completion summary

- Completed changes: added the completely guarded frozen pointwise ordinal body at schema 40,
  retained the typed fallback, synchronized schema assertions, and finalized directly affected
  Javadocs, package summaries, CPU guide, glossary, and planning records.
- Files changed or created: ten implementation-owned production/test paths plus eight
  documentation/Javadoc/planning paths, for 18 combined paths. No executable Java or test changed
  in the documentation pass.
- Tests and validation: reused `VERIFIED,20`, six suites/62 focused tests, 54 suites/357 CPU tests
  with 356 passes and one expected skip, complete Class-File inspection, and five accepted target
  and control forks. The documentation pass ran CPU Javadoc exactly once and passed all required
  documentation, scope, preservation, staging, and whitespace checks.
- Documentation impact: internal generator/loop/schema/cache/code-generation/prepare Javadocs,
  CPU guide, glossary, this task, A1C, CPU master plan, roadmap, and the sole next detailed task
  are synchronized. No new public API or architecture term was introduced.
- Unresolved issues: exactly `S-GENERAL-MIN` and `X-MIN-MULTI` remain.
- Follow-up required: execute Ready CPU 0007A1M, then plan only the remaining evidence-supported
  owner-cohesive `X-MIN-MULTI` correction. CPU 0007A2 remains Draft and blocked until A1C closes.

Status: Complete
