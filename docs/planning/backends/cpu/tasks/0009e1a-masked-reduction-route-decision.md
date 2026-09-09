# Task 0009E1A: Masked Reduction Route Decision

## Status

Complete

## Goal

Decide whether the complete current CPU masked floating `SUM`/`MEAN` family should retain its typed generated route or be replaced by finite direct typed Java. Choose direct Java only when it lowers total implementation and verification complexity. This task has no benchmark or performance-parity admission gate.

## Scope

Inspect and decide one cohesive, static, one-node masked-reduction family: `AggregateReductionKind.SUM` or `MEAN` with `MaskedReductionAttrs`; same-typed `FLOAT64`, `FLOAT32`, or `BFLOAT16` data and output; one normalized axis removed from the output; and one canonical `BOOL` mask whose directional right-aligned broadcast equals the data Shape.

The decision covers the full current resolved-layout/carrier matrix: dense-linear and general-odometer data, mask, and injective output layouts; heap, `MemorySegment`, and ordered mixed typed carriers; scalar and disjoint complete-output-cell ranges; and current cold materialization/rejection behavior. Preserve ordered `[data, mask, output]` boundaries. False mask values must be tested before data load or classification; only true positions advance the invocation-local selected count and enter the existing exact floating SUM state. `MEAN` finalizes that selected exact state using the selected count. Empty selection, special value, rounding, signed-zero, and canonical-mask-byte behavior remain exactly current.

The current route comprises `CpuMaskedReductionIr`, `CpuMaskedReductionLowering.Geometry`, `CpuMaskedReductionEmitter`, `CpuExactSumEmitter`, generated artifact finalization, and `CpuPreparedExecutable` binding/invocation. It packs cold data/mask/output bases plus geometry, uses one aligned exact-state slice per concurrently used output-cell range when outputs exist, and has no mask copy, count workspace, partial/combine state, or masked-only materialization. Preparation validates complete spans, carrier access/alignment, canonical mask bytes, injective output, and forbidden output/read or workspace/buffer overlap before writes or worker submission.

Compare the generated emitter's typed carrier/layout bodies, direct mask-address derivation, exact-state seam, artifact identity, class-file inspection surface, and current 90 `MASKED_SUM` plus 90 `MASKED_MEAN` inventory/semantic owners against the complete direct replacement and retirement proof. A direct route, if selected, must have finite typed entry points and cold selection/binding; no generic carrier dispatch may appear in its element loop. A retained route must leave production code and tests unchanged and record why replacement costs more overall.

## Out of scope

Advanced reductions, `LOG_SUM_EXP`, statistics, norms, `SOFTMAX`, `LOG_SOFTMAX`, normalizations, losses, ordinary aggregates, scans, arg-extrema, partial integral reductions, vectors, native routes, Model or Compiler semantics, gradients, shared Prepare/Runtime contracts, architecture, modules, Gradle, benchmarks, and performance claims. Do not create or detail E1B, E1C, E2, E3, F, or G. Legacy is read-only behavioral evidence; do not copy its source.

## Architecture references

Read [the architecture contract](../../../../../ARCHITECTURE.md), [the current architecture plan](../../../../architecture/current-architecture-plan.md), [the Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md), [the CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [CPU 0009E1](0009e1-partial-integral-reduction-direct-java-migration.md), and completed [CPU 0007C](0007c-portable-masked-reduction-coverage.md).

## Architecture constraints

`ARCHITECTURE.md` is authoritative. CPU Prepare owns lowering, route selection, exact-resource declaration, and finalization after shared slot assignment; Runtime executes immutable prepared recipes only. Preserve borrowed carriers, run-owned workspace, one `RunState` per active run, cold validation before mutation, and prepared-region ownership. Generated code must keep an optimal clean-Java implementation of the same specialized case as its review oracle, including early mask branch, typed carrier/layout work, selected-count state, exact-state algorithm, and final store. Do not add a public/shared dependency, resource kind, generic execution facade, or architecture rule; stop and report any need for one.

## Package impact

Existing packages used:

- `internal.ir` — masked structural identity.
- `internal.lowering` — resolved directional broadcast and exact-state geometry.
- `internal.prepare` — cold route/resource/finalization facts.
- `internal.executable` — immutable recipe, validation, typed invocation, and range-owned scratch.
- `internal.codegen.emit` — current generated typed entry and exact-state emission.

Packages added or changed:

- None if generation is retained. A selected direct route may change only existing CPU-private preparation/execution packages and retire masked-only emitter selection after proof.

Type placement:

- No production type is authorized by default. If a finite direct route needs one new private type, it belongs in `internal.executable` and counts against the maximum scope.

## Affected files

Expected decision-only paths are this task, the CPU master plan, parent CPU 0009, and the roadmap. If direct Java is selected, expected CPU paths are limited to masked-specific members of `CpuMaskedReductionIr`, `CpuMaskedReductionLowering`, `CpuPartitionPreparer`, `CpuPartitionFinalizer`, `CpuPreparedExecutable`, `CpuMaskedReductionEmitter`, `CpuClassFileKernelGenerator`, generated coverage inventory/evidence resources, and focused IR/lowering/emitter/preparer/finalizer/executable/semantic-closure tests. Review affected package Javadocs and the CPU guide only if executable ownership or behavior changes.

## Maximum scope

This task may create or modify at most 24 CPU production/test/resource paths, this task, and the three synchronized planning records. It may add at most one CPU-private production type and no public, shared, Model, Compiler, Runtime, Prepare, architecture, build, conformance, or integration path. Stop and create a separately scoped follow-up if direct replacement needs a second selected route, a new workspace kind, a generic carrier bridge, or exceeds these limits.

## Acceptance criteria

- Record an evidence-based route decision for the entire admitted family. Retain generation when its complete typed matrix is less costly to keep; select direct Java only when its replacement, binding, validation, and retirement proof is demonstrably smaller.
- Preserve the exact current type, axis/removing Shape, right-aligned directional `BOOL` broadcast, selected-count/numerical state, dense/general layout, heap/segment/mixed carrier, and output-cell range surfaces. Unsupported or malformed forms remain cold rejections or their established fallback; no silent widening is allowed.
- A selected direct route has finite typed heap/segment/mixed entry points chosen cold, retains current pre-write validation and private exact-state slices, and contains no per-element `Object` branch, allocation, boxing, reflection, map/string dispatch, synchronization, `Operation`/`CompiledNode`, or avoidable virtual dispatch.
- A retained generated route preserves mask-before-data-load order, selected-count semantics, exact-state finalization, specialization/cache identity, and invocation. Do not retain a permanent selected direct/generated duplicate.
- If generation remains selected, preserve the optimal-clean-Java oracle requirement and inspect generated members/class files proportionately; if retired, prove no selected masked route, inventory, or evidence consumer still requires it.
- Focused tests cover three floating types, both kinds, scalar/full/right-aligned singleton masks, dense/general layouts, heap/segment/mixed carriers, selected-count edges, special values, range determinism, malformed no-write failures, workspace isolation, and independent semantic/inventory evidence.
- A separate clean documentation-focused pass finalizes affected Javadocs, explanatory documentation, glossary impact, planning status, and documentation validation in the same overall change. A retained no-code decision records justified no-change conclusions.

## Tests / validation

After executable work stabilizes, run:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuMaskedReductionGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuMaskedAdvancedReductionSemanticClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuSpecializedGeneratedMatrixTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest
```

Run generated-Class-File/member inspection or its automated equivalent only if generation remains selected or is changed. Run direct-route/no-selected-reference retirement tests only if direct Java is selected. The documentation pass checks links, anchors, fences, terminology, glossary impact, exact paths, status/order/dependencies, whitespace, and `git diff --check`; run `:backends:cpu:javadoc` only after Java/Javadoc changes. No benchmark, representative timing, performance-parity ratio, repository-wide, backend-conformance, or integration suite is a gate. Repository-wide closure remains CPU 0009G/CI.

## Dependencies

- CPU 0009E1 is Complete and retained its separate dense-array integral partial route.
- CPU 0007C supplies current masked semantics, private owners, schema/inventory lineage, and the generated exact-state route; it is evidence, not permission to widen scope.
- CPU 0009D4 is Complete. CPU 0009 and parent E remain incomplete.

## Follow-up tasks

- E1B advanced reduction is the next Draft summary only after this task is Complete.
- E1C, E2, E3, F, and G remain ordered summary-only Draft work. Do not create detailed specifications for them here.

## Architecture impact

Expected impact: None.

If this task requires architecture changes, stop and report the issue.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009E1A. Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, Planning Guide, documentation rules and General/Planning profiles, CPU master plan, parent CPU 0009, completed CPU 0009E1 and 0007C, and this task. Implement exactly this route decision; select finite direct typed Java only if it lowers total implementation and verification complexity, otherwise retain generation with concrete evidence. Stop on an architecture or scope conflict. Do not commit or push. Hand the final diff and focused-test evidence to a separate clean documentation-focused agent.
```

## Local decisions

The inspected masked family is cohesive despite its three boundaries: directional mask addressing, false exclusion, selected count, and exact-state finalization form one operation-specific loop contract. It therefore receives one detailed task rather than splitting by carrier, layout, or kind. Its 180 current masked semantic/inventory owners and shared `CpuExactSumEmitter` seam make generated-route retention the initial hypothesis, not a pre-approved outcome.

## Known limitations

The current family remains static one-axis axis-removing floating masked `SUM`/`MEAN`; it does not authorize dynamic or symbolic layouts, keep-dimension or multi-axis masked forms, vector/native routes, changed Model semantics, or a performance claim. Exact-state workspace is range-private and parallelism is only across complete output cells.

## Validation evidence

The route remains generated. The current emitter is already the finite typed specialization for
the complete admitted family: its cold specialization fixes the three carrier roles, type,
layout regime, and artifact identity, while its generated entry has direct carrier loads/stores
and no Synaptik runtime helper reference. Replacing it requires a second finite direct-Java
matrix for FLOAT64/FLOAT32/BFLOAT16, heap/segment/mixed carriers, dense/general layouts, and
SUM/MEAN; cold route selection/binding; the current exact-state and selected-count seam; and
replacement, invocation, inventory, and no-selected-generated-route retirement proof. That is
strictly more implementation and verification work than retaining the one selected route.

The implementation context reviewed `CpuMaskedReductionIr`,
`CpuMaskedReductionLowering.Geometry`, `CpuMaskedReductionEmitter`,
`CpuExactSumEmitter`, generator dispatch, preparer/finalizer wiring, immutable executable
binding/validation, the independent scalar reference, focused semantic tests, and the
coverage inventory. The emitter loads the BOOL mask, branches on false, then loads and
classifies data only for true values; it increments a primitive invocation-local selected count
only on that true branch and passes it to exact mean finalization. Lowering/preparation retain
ordered data/mask/output boundaries, injective output validation, complete output-cell ranges,
and one aligned range-private exact-state slice with no mask/count/partial workspace.

`./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIrTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuMaskedReductionGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuMaskedAdvancedReductionSemanticClosureTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuSpecializedGeneratedMatrixTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest`
passed: 119 tests, zero failures, errors, or skips. The generated-kernel Class-File test
independently verifies deterministic field-free public static entry shape, no Synaptik member
references or bootstrap methods, and mask-load/false-branch/data-load order. The semantic
closure test invokes every inventory owner against its independent oracle; the checked
inventory has 90 `MASKED_SUM` and 90 `MASKED_MEAN` owners. No benchmark was run or claimed.

Clean documentation-focused context `/root` independently reviewed the stable four-path planning
diff against the architecture contract, Planning Guide, General and Planning profiles, CPU master,
parent 0009, completed E1/E1A/0007C evidence, current masked source, and focused tests. It
confirmed local links, headings, fenced commands, terminology, exact paths, status/dependency
order, whitespace, and `git diff --check`. No Java/Javadoc change is needed: no executable API or
implementation contract changed. No glossary change is needed: this decision introduces no new
reusable term or changed term boundary. The focused 119-test evidence is reused because no Java
changed after that successful run; no benchmark is claimed.

## Implementation notes

No production or test change is warranted: generation is retained, so no direct route, duplicate
binding, retirement consumer, or new Javadoc contract is introduced. E1B advanced reduction is
now the sole next summary frontier; it remains Draft and receives no detailed specification here.

## Completion summary

- Completed changes: Retained the complete current generated masked FLOAT64/FLOAT32/BFLOAT16
  SUM/MEAN route after proportional source, generated-member/Class-File, semantic/inventory,
  and focused invocation review; synchronized task/frontier status.
- Files changed or created: This task and the synchronized CPU master plan, parent CPU 0009,
  and roadmap only. No production, test, resource, Javadoc, public, shared, or architecture
  path changed.
- Tests and validation: The required focused eight-class CPU command passed 119 tests with zero
  failures, errors, or skips. Generated-Class-File inspection and the independent semantic
  closure cover mask-before-data-load, artifact invocation, typed carrier forms, and all 180
  masked inventory owners. No benchmark or repository-wide suite was run.
- Documentation impact: The separate clean documentation-focused pass finalized this planning-only
  diff, including Markdown/link/anchor/fence, status/order, terminology, glossary, exact-path,
  whitespace, and `git diff --check` validation. No Javadoc change is needed because executable
  behavior and its contracts are unchanged; no glossary change is needed because terminology did
  not change.
- Architecture impact: None.
- Unresolved issues: None for E1A. Parent CPU 0009 and parent E remain incomplete; E1B is the
  sole next summary frontier and remains Draft.
- Follow-up required: E1B remains the sole next Draft summary frontier; no E1B specification is
  created by this task.

Status: Complete
