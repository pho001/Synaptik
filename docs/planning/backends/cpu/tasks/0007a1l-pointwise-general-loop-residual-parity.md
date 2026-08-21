# Task 0007A1L: Pointwise general-loop residual parity

## Status

Ready

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

- `S-GENERAL-MIN` and `X-MIN-MULTI` remain Draft and unassigned until accepted A1L evidence
  selects the next bounded owner-cohesive task.
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

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
