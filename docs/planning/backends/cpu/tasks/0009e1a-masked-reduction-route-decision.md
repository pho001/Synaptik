# Task 0009E1A: [Masked Reduction](../../../../glossary.md#masked-reduction) Route Decision

## Reader context

This task records retaining generated masked floating `SUM`/`MEAN` as one specialized family.
`CpuMaskedReductionIr` identifies the case; `CpuMaskedReductionLowering.Geometry` resolves
directional broadcast and addresses; `CpuMaskedReductionEmitter` uses `CpuExactSumEmitter` for exact
accumulation. The mask is tested before data loading or classification, and only selected positions
advance the invocation-local count used by `MEAN`.

Here, CPU analysis selects structural carrier/layout forms and finalization constructs an immutable
`CpuPreparedExecutable` recipe. Runtime later calls its CPU-owned binder on current run storage to
validate ordered data/mask/output boundaries and create a per-run bound invocation. Its exact-state
scratch belongs to the run; borrowed input/output carriers do not become persistent model state.

Class-File checks are structural oracles for entry shape and branch order. Independent scalar
comparisons across the recorded 180 inventory owners supply semantic evidence.

Current-topic navigation (not a replacement for the historical record):

- [current directional masking and selected counts](../../../../reference/tensor/masked-index-reductions.md#masked-sum-and-mean-expressions) — Explains why the mask cannot enlarge the data Shape and why excluded values never contribute.
- [backend ownership versus implementation route](../../../../design/notes/cpu-kernel-strategy.md#strategy) — Separates Planning ownership, CPU route choice, finalization, and Runtime invocation.

## Status

Complete

## Goal

Decide whether the complete current [CPU](../../../../glossary.md#term-cpu) masked floating `SUM`/[`MEAN`](../../../../glossary.md#term-mean) family should retain its typed generated route or be replaced by [finite](../../../../glossary.md#term-finite) direct typed Java. Choose direct Java only when it lowers total implementation and verification complexity. This task has no benchmark or performance-parity [admission](../../../../glossary.md#term-admission) gate.

## [Scope](../../../../glossary.md#term-scope)

Inspect and decide one cohesive, static, one-[node](../../../../glossary.md#node) masked-reduction family: `AggregateReductionKind.SUM` or [`MEAN`](../../../../glossary.md#term-mean) with `MaskedReductionAttrs`; same-typed [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), or [`BFLOAT16`](../../../../glossary.md#term-bfloat16) data and output; one [normalized axis](../../../../glossary.md#normalized-axis) removed from the output; and one [canonical](../../../../glossary.md#term-canonical-form) [`BOOL`](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask) whose directional right-aligned broadcast equals the data [Shape](../../../../glossary.md#shape).

The decision covers the full current resolved-[layout](../../../../glossary.md#layout)/[carrier](../../../../glossary.md#term-storage-carrier) [matrix](../../../../glossary.md#term-matrix-meanings): dense-[linear](../../../../glossary.md#term-linear-meanings) and general-odometer data, mask, and [injective](../../../../glossary.md#term-injective) output layouts; [heap](../../../../glossary.md#term-heap), [`MemorySegment`](../../../../glossary.md#term-memorysegment), and ordered mixed typed carriers; [scalar](../../../../glossary.md#term-scalar) and disjoint complete-output-cell ranges; and current cold [materialization](../../../../glossary.md#materialization)/rejection behavior. Preserve ordered `[data, mask, output]` boundaries. False mask values must be tested before data load or classification; only true positions advance the invocation-local selected count and enter the existing exact floating SUM state. `MEAN` finalizes that selected exact state using the selected count. Empty selection, special value, [rounding](../../../../glossary.md#term-rounding), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), and canonical-mask-byte behavior remain exactly current.

The current route comprises `CpuMaskedReductionIr`, `CpuMaskedReductionLowering.Geometry`, `CpuMaskedReductionEmitter`, `CpuExactSumEmitter`, generated [artifact](../../../../glossary.md#term-artifact) finalization, and `CpuPreparedExecutable` [binding](../../../../glossary.md#term-binding)/invocation. It packs cold data/mask/output bases plus geometry, uses one aligned exact-state [slice](../../../../glossary.md#slice) per concurrently used output-cell range when outputs exist, and has no mask copy, count workspace, partial/combine state, or masked-only materialization. Preparation validates complete spans, carrier access/[alignment](../../../../glossary.md#term-alignment), canonical mask bytes, injective output, and forbidden output/read or workspace/[buffer](../../../../glossary.md#term-buffer) overlap before writes or worker submission.

Compare the generated emitter's typed carrier/layout bodies, direct mask-address derivation, exact-state seam, artifact [identity](../../../../glossary.md#term-identity-meanings), class-file inspection surface, and current 90 `MASKED_SUM` plus 90 `MASKED_MEAN` inventory/semantic [owners](../../../../glossary.md#term-owner) against the complete direct replacement and retirement proof. A direct route, if selected, must have [finite](../../../../glossary.md#term-finite) typed entry points and cold selection/binding; no generic carrier [dispatch](../../../../glossary.md#term-dispatch) may appear in its [element](../../../../glossary.md#term-element) loop. A retained route must leave production code and tests unchanged and record why replacement costs more overall.

## Out of [scope](../../../../glossary.md#term-scope)

Advanced reductions, `LOG_SUM_EXP`, statistics, norms, [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax), `LOG_SOFTMAX`, normalizations, [losses](../../../../glossary.md#loss), ordinary aggregates, scans, arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum), partial integral reductions, [vectors](../../../../glossary.md#term-vector), native routes, [Model](../../../../glossary.md#term-model) or [Compiler](../../../../glossary.md#term-compiler) semantics, [gradients](../../../../glossary.md#term-gradient), shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) contracts, architecture, [modules](../../../../glossary.md#term-module), [Gradle](../../../../glossary.md#term-gradle), benchmarks, and performance claims. Do not create or detail E1B, E1C, E2, E3, F, or G. Legacy is read-only behavioral evidence; do not copy its source.

## Architecture references

Read [the architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, [the current architecture plan](../../../../architecture/current-architecture-plan.md), [the Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [documentation rules](../../../../developer-guide/documentation-rules.md), [the CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>, parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [CPU 0009E1](0009e1-partial-integral-reduction-direct-java-migration.md), and completed [CPU 0007C](0007c-portable-masked-reduction-coverage.md).

## Architecture constraints

`ARCHITECTURE.md` is authoritative. [CPU](../../../../glossary.md#term-cpu) [Prepare](../../../../glossary.md#prepare) owns [lowering](../../../../glossary.md#lowering), route selection, exact-resource declaration, and finalization after shared slot assignment; [Runtime](../../../../glossary.md#term-runtime) executes [immutable](../../../../glossary.md#term-immutable) prepared recipes only. Preserve [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) carriers, [run](../../../../glossary.md#term-run)-owned workspace, one [`RunState`](../../../../glossary.md#run-state--runstate) per active run, cold [validation](../../../../glossary.md#term-validation) before mutation, and prepared-region [ownership](../../../../glossary.md#term-ownership-meanings). Generated code must keep an optimal clean-Java implementation of the same specialized case as its review [oracle](../../../../glossary.md#term-test-oracle), including early [mask](../../../../glossary.md#term-boolean-mask) branch, typed [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout) work, selected-count state, exact-state algorithm, and final store. Do not add a public/shared [dependency](../../../../glossary.md#term-dependency), resource kind, generic execution [facade](../../../../glossary.md#term-facade), or architecture rule; stop and report any need for one.

## Package impact

Existing packages used:

- `internal.ir` — masked structural [identity](../../../../glossary.md#term-identity-meanings).
- `internal.lowering` — resolved directional broadcast and exact-state geometry.
- `internal.prepare` — cold route/resource/finalization facts.
- `internal.executable` — [immutable](../../../../glossary.md#term-immutable) recipe, [validation](../../../../glossary.md#term-validation), typed invocation, and range-owned scratch.
- `internal.codegen.emit` — current generated typed entry and exact-state emission.

Packages added or changed:

- None if generation is retained. A selected direct route may change only existing [CPU](../../../../glossary.md#term-cpu)-private preparation/execution packages and retire masked-only emitter selection after proof.

Type placement:

- No production type is authorized by default. If a [finite](../../../../glossary.md#term-finite) direct route needs one new private type, it belongs in `internal.executable` and counts against the maximum [scope](../../../../glossary.md#term-scope).

## Affected files

Expected decision-only paths are this task, the [CPU](../../../../glossary.md#term-cpu) master plan, parent CPU 0009, and the roadmap. If direct Java is selected, expected CPU paths are limited to masked-specific members of `CpuMaskedReductionIr`, `CpuMaskedReductionLowering`, `CpuPartitionPreparer`, `CpuPartitionFinalizer`, `CpuPreparedExecutable`, `CpuMaskedReductionEmitter`, `CpuClassFileKernelGenerator`, generated coverage inventory/evidence resources, and focused [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/emitter/preparer/finalizer/executable/semantic-closure tests. Review affected package Javadocs and the CPU guide only if executable [ownership](../../../../glossary.md#term-ownership-meanings) or behavior changes.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 24 [CPU](../../../../glossary.md#term-cpu) production/test/resource paths, this task, and the three synchronized [planning](../../../../glossary.md#term-planning) records. It may add at most one CPU-private production type and no public, shared, [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), architecture, build, [conformance](../../../../glossary.md#term-conformance), or integration path. Stop and create a separately scoped follow-up if direct replacement needs a second selected route, a new workspace kind, a generic [carrier](../../../../glossary.md#term-storage-carrier) bridge, or exceeds these limits.

## Acceptance criteria

- Record an evidence-based route decision for the entire admitted family. Retain generation when its complete typed [matrix](../../../../glossary.md#term-matrix-meanings) is less costly to keep; select direct Java only when its replacement, [binding](../../../../glossary.md#term-binding), [validation](../../../../glossary.md#term-validation), and retirement proof is demonstrably smaller.
- Preserve the exact current type, [axis](../../../../glossary.md#term-axis)/removing [Shape](../../../../glossary.md#shape), right-aligned directional [`BOOL`](../../../../glossary.md#data-type--datatype) broadcast, selected-count/numerical state, dense/general [layout](../../../../glossary.md#layout), [heap](../../../../glossary.md#term-heap)/segment/mixed [carrier](../../../../glossary.md#term-storage-carrier), and output-cell range surfaces. Unsupported or malformed forms remain cold rejections or their established fallback; no silent [widening](../../../../glossary.md#term-widening) is allowed.
- A selected direct route has [finite](../../../../glossary.md#term-finite) typed heap/segment/mixed entry points chosen cold, retains current pre-write validation and private exact-state slices, and contains no per-[element](../../../../glossary.md#term-element) `Object` branch, allocation, boxing, reflection, map/string [dispatch](../../../../glossary.md#term-dispatch), [synchronization](../../../../glossary.md#term-synchronization), [`Operation`](../../../../glossary.md#operation)/`CompiledNode`, or avoidable virtual dispatch.
- A retained generated route preserves [mask](../../../../glossary.md#term-boolean-mask)-before-data-load order, selected-count semantics, exact-state finalization, [specialization](../../../../glossary.md#term-specialization)/[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), and invocation. Do not retain a permanent selected direct/generated duplicate.
- If generation remains selected, preserve the optimal-clean-Java [oracle](../../../../glossary.md#term-test-oracle) requirement and inspect generated members/class files proportionately; if retired, prove no selected masked route, inventory, or evidence consumer still requires it.
- Focused tests cover three floating types, both kinds, [scalar](../../../../glossary.md#term-scalar)/full/right-aligned [singleton](../../../../glossary.md#term-singleton) masks, dense/general layouts, heap/segment/mixed carriers, selected-count edges, special values, range determinism, malformed no-write failures, workspace isolation, and independent semantic/inventory evidence.
- A separate clean documentation-focused pass finalizes affected Javadocs, explanatory documentation, glossary impact, [planning](../../../../glossary.md#term-planning) status, and documentation validation in the same overall change. A retained no-code decision records justified no-change conclusions.

## Tests / [validation](../../../../glossary.md#term-validation)

After executable work stabilizes, [run](../../../../glossary.md#term-run):

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

Run generated-Class-File/member inspection or its automated equivalent only if generation remains selected or is changed. Run direct-route/no-selected-reference retirement tests only if direct Java is selected. The documentation pass checks links, anchors, fences, terminology, glossary impact, exact paths, status/order/dependencies, whitespace, and `git diff --check`; run `:backends:cpu:javadoc` only after Java/[Javadoc](../../../../glossary.md#term-javadoc) changes. No benchmark, representative timing, performance-parity ratio, repository-wide, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), or integration suite is a gate. Repository-wide closure remains [CPU](../../../../glossary.md#term-cpu) 0009G/[CI](../../../../glossary.md#term-continuous-integration-ci).

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0009E1 is Complete and retained its separate dense-array integral partial route.
- CPU 0007C supplies current masked semantics, private [owners](../../../../glossary.md#term-owner), [schema](../../../../glossary.md#term-schema)/inventory lineage, and the generated exact-state route; it is evidence, not permission to widen [scope](../../../../glossary.md#term-scope).
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

The inspected masked family is cohesive despite its three boundaries: directional [mask](../../../../glossary.md#term-boolean-mask) addressing, false exclusion, selected count, and exact-state finalization form one [operation](../../../../glossary.md#operation)-specific loop contract. It therefore receives one detailed task rather than splitting by [carrier](../../../../glossary.md#term-storage-carrier), [layout](../../../../glossary.md#layout), or kind. Its 180 current masked semantic/inventory [owners](../../../../glossary.md#term-owner) and shared `CpuExactSumEmitter` seam make generated-route retention the initial hypothesis, not a pre-approved outcome.

## Known limitations

The current family remains static one-[axis](../../../../glossary.md#term-axis) axis-removing floating masked `SUM`/[`MEAN`](../../../../glossary.md#term-mean); it does not authorize dynamic or symbolic [layouts](../../../../glossary.md#layout), keep-[dimension](../../../../glossary.md#dimension) or multi-axis masked forms, [vector](../../../../glossary.md#term-vector)/native routes, changed [Model](../../../../glossary.md#term-model) semantics, or a performance claim. Exact-state workspace is range-private and parallelism is only across complete output cells.

## [Validation](../../../../glossary.md#term-validation) evidence

The route remains generated. The current emitter is already the [finite](../../../../glossary.md#term-finite) typed [specialization](../../../../glossary.md#term-specialization) for
the complete admitted family: its cold specialization fixes the three [carrier](../../../../glossary.md#term-storage-carrier) roles, type,
[layout](../../../../glossary.md#layout) regime, and [artifact](../../../../glossary.md#term-artifact) [identity](../../../../glossary.md#term-identity-meanings), while its generated entry has direct carrier loads/stores
and no Synaptik [runtime](../../../../glossary.md#term-runtime) helper reference. Replacing it requires a second finite direct-Java
[matrix](../../../../glossary.md#term-matrix-meanings) for [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16), [heap](../../../../glossary.md#term-heap)/segment/mixed carriers, dense/general layouts, and
SUM/[MEAN](../../../../glossary.md#term-mean); cold route selection/[binding](../../../../glossary.md#term-binding); the current exact-state and selected-count seam; and
replacement, invocation, inventory, and no-selected-generated-route retirement proof. That is
strictly more implementation and verification work than retaining the one selected route.

The implementation context reviewed `CpuMaskedReductionIr`,
`CpuMaskedReductionLowering.Geometry`, `CpuMaskedReductionEmitter`,
`CpuExactSumEmitter`, generator [dispatch](../../../../glossary.md#term-dispatch), preparer/finalizer wiring, [immutable](../../../../glossary.md#term-immutable) executable
binding/validation, the independent [scalar](../../../../glossary.md#term-scalar) reference, focused semantic tests, and the
coverage inventory. The emitter loads the [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask), branches on false, then loads and
classifies data only for true values; it increments a primitive invocation-local selected count
only on that true branch and passes it to exact mean finalization. [Lowering](../../../../glossary.md#lowering)/preparation retain
ordered data/mask/output boundaries, [injective](../../../../glossary.md#term-injective) output validation, complete output-cell ranges,
and one aligned range-private exact-state [slice](../../../../glossary.md#slice) with no mask/count/partial workspace.

`./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIrTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuMaskedReductionGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuMaskedAdvancedReductionSemanticClosureTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuSpecializedGeneratedMatrixTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest`
passed: 119 tests, zero failures, errors, or skips. The generated-[kernel](../../../../glossary.md#kernel) Class-File test
independently verifies deterministic field-free public static entry shape, no Synaptik member
references or bootstrap methods, and mask-load/false-branch/data-load order. The semantic
closure test invokes every inventory [owner](../../../../glossary.md#term-owner) against its independent [oracle](../../../../glossary.md#term-test-oracle); the checked
inventory has 90 `MASKED_SUM` and 90 `MASKED_MEAN` owners. No benchmark was [run](../../../../glossary.md#term-run) or claimed.

Clean documentation-focused context `/root` independently reviewed the stable four-path [planning](../../../../glossary.md#term-planning)
diff against the [architecture contract](../../../../glossary.md#architecture-contract), Planning Guide, General and Planning profiles, [CPU](../../../../glossary.md#term-cpu) master,
parent 0009, completed E1/E1A/0007C evidence, current masked source, and focused tests. It
confirmed local links, headings, fenced commands, terminology, exact paths, status/[dependency](../../../../glossary.md#term-dependency)
order, whitespace, and `git diff --check`. No Java/[Javadoc](../../../../glossary.md#term-javadoc) change is needed: no executable [API](../../../../glossary.md#term-application-programming-interface-api) or
implementation contract changed. No glossary change is needed: this decision introduces no new
reusable term or changed term boundary. The focused 119-test evidence is reused because no Java
changed after that successful run; no benchmark is claimed.

## Implementation notes

No production or test change is warranted: generation is retained, so no direct route, duplicate
[binding](../../../../glossary.md#term-binding), retirement consumer, or new [Javadoc](../../../../glossary.md#term-javadoc) contract is introduced. E1B advanced reduction is
now the sole next summary frontier; it remains Draft and receives no detailed specification here.

## Completion summary

- Completed changes: Retained the complete current generated masked FLOAT64/FLOAT32/BFLOAT16
  SUM/MEAN route after proportional source, generated-member/Class-File, semantic/inventory,
  and focused invocation review; synchronized task/frontier status.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [MEAN glossary](../../../../glossary.md#term-mean))</span>
- Files changed or created: This task and the synchronized CPU master plan, parent CPU 0009,
  and roadmap only. No production, test, resource, Javadoc, public, shared, or architecture
  path changed.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Tests and validation: The required focused eight-class CPU command passed 119 tests with zero
  failures, errors, or skips. Generated-Class-File inspection and the independent semantic
  closure cover mask-before-data-load, artifact invocation, typed carrier forms, and all 180
  masked inventory owners. No benchmark or repository-wide suite was run.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [mask glossary](../../../../glossary.md#term-boolean-mask); [artifact glossary](../../../../glossary.md#term-artifact); [carrier glossary](../../../../glossary.md#term-storage-carrier); [run glossary](../../../../glossary.md#term-run))</span>
- Documentation impact: The separate clean documentation-focused pass finalized this planning-only
  diff, including Markdown/link/anchor/fence, status/order, terminology, glossary, exact-path,
  whitespace, and <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>`git diff --check` validation. No Javadoc change is needed because executable
  behavior and its contracts are unchanged; no glossary change is needed because terminology did
  not change.
- Architecture impact: None.
- Unresolved issues: None for E1A. Parent CPU 0009 and parent E remain incomplete; E1B is the
  sole next summary frontier and remains Draft.
- Follow-up required: E1B remains the sole next Draft summary frontier; no E1B specification is
  created by this task.

Status: Complete
