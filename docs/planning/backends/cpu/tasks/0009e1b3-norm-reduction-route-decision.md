# Task 0009E1B3: Norm Reduction Route Decision

## Status

Complete

## Goal

Decide whether the current CPU `L1_NORM` and `L2_NORM` advanced reductions retain their specialized generated route or are replaced by finite direct typed Java. Choose direct Java only when its total implementation and verification cost is lower; this task has no benchmark, performance gate, or performance claim.

## Scope

Inspect exactly two static one-input/one-output meanings: `AggregateReductionKind.L1_NORM` and `AggregateReductionKind.L2_NORM`, each with `MultiAxisReductionAttrs`. Admit only identical `FLOAT64`, `FLOAT32`, or `BFLOAT16` input/output types; normalized, ordered, distinct, non-negative axes (including the empty list); and retained or removed selected dimensions. The reduced output Shape must exactly match the selected axes. The current route admits only resolved non-negative input and injective output layouts, whether dense-linear or general-odometer; typed heap, `MemorySegment`, and ordered mixed boundary carriers; and scalar or disjoint parallel-scalar ranges that own complete output cells.

Preserve `CpuAdvancedReductionIr` and `CpuAdvancedReductionLowering.Geometry`: two borrowed boundary carriers, cold packed bases/static geometry, checked selected-domain and output counts, one final typed store for each owned cell, no materialization, and no partial/combine state. L1 declares the existing aligned exact-state workspace only when output cells exist, one disjoint slice for each concurrently selected complete-cell range. L2 declares zero workspace. Cold validation must establish the complete route, carrier spans/access/alignment, output injectivity, input/output and workspace/buffer non-overlap, range ownership, and resource sufficiency before any write or worker submission.

Reject cold every other partition or aggregate kind; non-`MultiAxisReductionAttrs`; wrong input/output arity, identity, type, or reduced Shape; non-identical or non-floating type; unnormalized or invalid axes; unresolved, negative, inaccessible, undersized, or misaligned layout/carrier facts; non-injective output; overlapping input/output or workspace/buffer spans; invalid range or worker facts; missing/incorrect L1 exact state; and any request for L2 scratch. Rejection leaves outputs, scratch, and canaries unchanged and submits no worker work.

The numerical owners remain distinct. L1 must sum absolute *represented* values through its existing exact-state workspace and narrow once, ties-to-even, at the final represented seam. L2 must use its current scaled sum-of-squares state and final square root, with zero workspace, so finite inputs avoid avoidable square overflow and underflow. Preserve current empty-domain and point-domain behavior, signed-zero result behavior, canonical-NaN-over-infinity priority, finite/infinite results, final narrowing, and one-final-store semantics exactly as source and focused tests establish; do not infer a new numerical rule.

Compare `CpuNormEmitter`, `CpuExactSumEmitter` for L1, shared typed carrier/address/classification helpers, generated artifact/cache identity, Prepare declaration/finalization, immutable prepared binding/invocation, independent reference oracle, and focused inventory/evidence owners with a complete finite direct replacement and retirement proof. A direct replacement must cold-select finite typed array/segment and dense/general forms, bind immutable range-owning entry points, and have no generic per-element carrier, type, layout, kind, reflection, map/string, or virtual semantic dispatch. If generation remains, retain a matching optimal clean-Java oracle with the same L1 exact-state or L2 scaled-squares dataflow, special handling, narrowing, and store shape; the generated hot loop must have no avoidable allocation, boxing, reflection, helper semantic dispatch, or per-element virtual indirection.

## Out of scope

Completed `LOG_SUM_EXP`, `VARIANCE`, and `STANDARD_DEVIATION`; `SOFTMAX`/`LOG_SOFTMAX`; normalization; loss; ordinary or masked reductions; Model/Compiler semantics or gradients; architecture, shared Prepare/Runtime, build, or resource-contract changes; vector/native routes; materialization; partial/combine execution; and legacy source copying. Do not create detailed E1C or later tasks. Legacy `pre-rewrite` is read-only behavioral evidence only.

## Architecture references

Read [the architecture contract](../../../../../ARCHITECTURE.md), [current architecture plan](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md), CPU master plan, parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [E1](0009e1-partial-integral-reduction-direct-java-migration.md), [E1A](0009e1a-masked-reduction-route-decision.md), [E1B1](0009e1b1-log-sum-exp-route-decision.md), [E1B2](0009e1b2-statistical-reduction-route-decision.md), and [0007D](0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md).

## Architecture constraints

`ARCHITECTURE.md` is authoritative. CPU Prepare owns lowering, finite route selection, exact resource declaration, and finalization after shared slot assignment; Runtime invokes immutable prepared work only. Preserve borrowed carriers, run-owned resources, one `RunState` per active run, cold validation before writes/submission, and prepared-region ownership. Do not add a public/shared dependency, generic executor, workspace/resource kind, architecture rule, or build change; stop and report any need for one.

## Package impact

Existing packages only:

- `internal.ir` and `internal.lowering` retain advanced-reduction identity and static geometry.
- `internal.codegen.emit` retains the generated norm and L1 exact-state emitter seams if generation is selected.
- `internal.prepare`, `internal.executable`, and `internal.cache` retain cold selection, exact resource finalization, artifact identity, and immutable invocation.
- `internal.reference` retains the independent mathematical oracle.

A retained route adds no type. A selected direct route may add at most one narrowly named CPU-private norm executable owner in an existing package; it must not absorb log-sum-exp or statistical algorithms.

## Affected files

Expected decision/planning paths:

- this task;
- `docs/planning/backends/cpu/master-plan.md`;
- parent `docs/planning/backends/cpu/tasks/0009-portable-generated-coverage-closure-checkpoint.md`;
- `docs/planning/roadmap.md`.

Review-only implementation seams: `CpuCapabilityProvider`, `CpuAdvancedReductionIr`, `CpuAdvancedReductionLowering`, `CpuNormEmitter`, `CpuExactSumEmitter`, `CpuClassFileKernelGenerator`, generated artifact/cache, preparation/finalization, `CpuPreparedExecutable`, `CpuAdvancedReductionReferenceKernel`, and focused capability, IR/lowering/generated/reference/prepare/executable/inventory/evidence tests.

## Maximum scope

The implementation, tests, documentation, and planning may modify or create at most 30 repository paths: at most 16 production/Javadoc paths, ten test paths, and the four named planning paths. At most one new CPU-private direct production type is permitted.

A 31st path, second production type, shared advanced-geometry/resource-contract change, materialization, partial/combine state, shared/public/build/architecture/conformance/integration path, or generic element dispatch is a stop-and-replan condition.

## Acceptance criteria

- The final decision covers the complete current L1/L2 matrix, selects retained generation or a finite direct typed route, gives concrete implementation-plus-verification cost evidence, and makes no performance claim.
- The selected route preserves `MultiAxisReductionAttrs` axes/keep behavior; identical floating types; static Shape/layout/carrier/range admission; complete-cell scalar/disjoint-parallel ownership; truthful L1 exact-state and L2 zero-workspace declarations; and cold no-write/no-submission failure behavior.
- L1 preserves exact represented absolute summation, final ties-to-even represented narrowing, and its exact-state resource isolation. L2 preserves scaled sum-of-squares and final square root without workspace. Both preserve current empty/point-domain, signed-zero, NaN/infinity priority, finite overflow/underflow avoidance, narrowing, and exactly-one-store behavior.
- A direct route uses finite cold typed carrier/layout selection and immutable binding with no generic per-element dispatch. Retire generated selection only after artifact/cache, invocation, inventory, evidence, and no-selected-reference proof are updated.
- A retained route changes no executable path and records the typed carrier/layout, L1 resource, L2 state, emitter/helper, artifact/binding, oracle, and retirement costs that make replacement larger. It supplies a matching optimal clean-Java oracle and proportionate generated member/Class-File and hot-loop/dataflow hygiene evidence.
- Focused semantic, special-value, type/narrowing, layout/carrier, range/canary, L1 scratch isolation, L2 zero-workspace, preparation/finalization, immutable invocation, reference, inventory, and route-retirement tests pass for the selected route.
- A separate documentation-focused clean context finalizes affected Javadocs, explanatory documentation, glossary impact, planning status, and documentation validation in the same overall change.

## Tests / validation

After executable stabilization, the implementation context runs focused CPU tests selected from the current norm owners, including capability, advanced IR/lowering/generated/reference, specialized matrix/direct-evidence/coverage inventory, and affected prepare/finalizer/executable tests. Record exact commands and results here. Do not run benchmarks or repository-wide tests.

The documentation context reuses stable Java evidence unless it changes executable Java; it checks Markdown links, anchors, fences, terminology/glossary impact, changed-path scope, statuses, dependencies, and frontier consistency, then runs:

```bash
git diff --check
git status --short -uall
```

Repository-wide, architecture, backend-conformance, integration, and performance validation are deferred to CPU 0009 final checkpoint or CI because this task must not change a shared boundary.

## Dependencies and follow-up tasks

- E1, E1A, E1B1, and E1B2 are Complete; E1B2 retains the corrected statistical generated route.
- Complete 0007D supplies the existing advanced IR/lowering, norm emitter, L1 exact-state resource, L2 scaled-squares state, reference oracle, and focused evidence.
- E1C, E2, E3, F, and G remain ordered Draft summaries only. E1C follows this decision; do not create its detailed specification here.

## Architecture impact

Expected impact: None. Stop and report any public, dependency, resource, module, or architecture conflict rather than changing that boundary.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU task 0009E1B3. Work on the existing worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, current architecture plan, Planning Guide, CPU master plan, parent CPU 0009, completed E1/E1A/E1B1/E1B2/0007D, this task, current norm advanced-reduction source and focused tests, documentation rules, and General/Planning profiles. Implement exactly this route decision. Choose finite direct typed Java only if total implementation and verification cost is lower; otherwise retain generation with concrete evidence. Preserve the complete L1/L2 boundary and distinct numerical owners. Stop on an architecture or scope conflict. Hand any stabilized executable diff and exact test evidence to a separate documentation-focused clean context before marking Complete. Do not run benchmarks or repository-wide tests. Do not commit, push, or stage.
```

## Local decisions

E1B3 owns both norms because they share advanced geometry but not numerical state: L1 is exact represented absolute summation with a per-range exact-state slice, while L2 is scaled squares and final square root with no workspace. They follow the completed zero-workspace log-sum-exp and corrected statistical decisions; softmax remains a later owner.

## Known limitations

This is a route decision, not a numerical redesign, public API change, or performance result. Only current static scalar/parallel-scalar resolved-layout forms are in scope; dynamic Shapes, vectorization, fusion, materialization, partial/combine execution, and native routes remain out.

## Validation evidence

Implementation evidence finalized by the separate documentation-focused context.

The implementation review retains the bounded generated route for both `L1_NORM` and
`L2_NORM`. This is an implementation-plus-verification-cost decision, not a performance claim.
`CpuNormEmitter` already produces the finite typed entries for FLOAT64, FLOAT32, and BFLOAT16
over the cold-selected heap, `MemorySegment`, and mixed carrier forms and the packed
dense-linear/general-odometer geometry. Its immutable generated entry consumes the existing
complete-output-cell range binding. L1 additionally reuses the existing per-range exact-state
slice and `CpuExactSumEmitter`; L2 uses its existing scaled-squares state and declares no
workspace. A direct replacement would need a new finite typed executable owner, duplicate both
different numerical bodies (including L1's exact represented absolute sum and one ties-to-even
narrowing), and prove the full carrier/layout/range matrix. It would also require retirement
proof and corresponding changes for generator selection, artifact identity/cache, finalization,
immutable invocation, inventory, and generated-evidence owners. That is greater total work than
retaining the verified route.

Source, generated-Class-File, and test-owner inspection found that the selected route retains
the required hot-loop shape: typed carrier loads and final typed store are emitted directly;
generated artifacts have one typed static entry and no Synaptik member references, fields,
bootstrap methods, reflection, collection/map ownership, or per-element semantic dispatch.
The generated L1 body resets and uses only its invocation-private exact-state slice, classifies
special values, sums represented absolute factors exactly, narrows once, and stores once. The L2
body uses scaled squares followed by the final square root, has zero scratch, preserves the
current special-value priority, and stores once. `CpuAdvancedReductionReferenceKernel` remains a
separate clean-Java oracle: it does not consume lowering geometry, packed invocation data,
generated helpers, or exact-state workspace. This evidence makes no literal Class-File, JIT, or
performance assertion.

The implementation context ran this focused command on 2026-09-09:

```bash
./gradlew :backends:cpu:test --rerun-tasks \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAdvancedReductionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAdvancedReductionGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuSpecializedGeneratedMatrixTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedDirectEvidenceClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuAdvancedReductionReferenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelPersistenceEvidenceTest
```

The 13 selected Gradle XML reports record 168 tests, zero failures, zero errors, and one expected
opt-in skip (`CpuGeneratedKernelPersistenceEvidenceTest`). No benchmark or repository-wide suite
ran. The selected owners cover capability admission, IR/lowering, finite and special values,
type narrowing, layouts/carriers, complete-cell ranges and canaries, L1 scratch isolation, L2
zero-workspace, preparation/finalization, immutable invocation, independent reference behavior,
artifact/cache, inventory, and generated-entry evidence.

No executable Java changed. The current Java and Javadoc remain accurate; no explanatory,
glossary, architecture, build, conformance, or integration update is indicated because the route
is CPU-private and all public/shared contracts remain unchanged. This separate documentation
context independently finalized that assessment, planning consistency, and task status.

## Implementation notes

Retain generated execution for `L1_NORM` and `L2_NORM` unchanged. Their shared cold typed
carrier/layout and immutable artifact/binding mechanics do not make their numerical owners
interchangeable: L1 owns exact represented absolute summation with isolated exact-state slices;
L2 owns scaled squares and final square root with zero workspace. Direct migration would increase
implementation and verification cost while requiring generator-route retirement across existing
artifact, cache, finalizer, invocation, inventory, and evidence seams. No performance claim or
benchmark gate supports this decision.

## Completion summary

- Completed changes: Finalized the retained bounded-generated-execution decision for
  `L1_NORM` and `L2_NORM`. A finite direct replacement would duplicate their distinct numerical
  bodies and typed carrier/layout/range binding, then require generated-route retirement proof
  across artifact/cache, finalization, immutable invocation, inventory, and evidence owners.
- Files changed or created: This task and the three synchronized CPU planning records only.
- Tests and validation: Reused the implementation context's 2026-09-09 focused CPU evidence:
  168 tests, zero failures or errors, and one expected opt-in persistence-evidence skip across
  the 13 recorded XML reports. This documentation context inspected the norm emitter, exact-state
  emitter, advanced IR/lowering, generator, immutable invocation and scratch binding, reference
  oracle, and focused test owners; it also checked local links, anchors, fences, terminology,
  task dependencies/frontier, exact changed-path scope, `git diff --check`, and `git status
  --short -uall`. It did not rerun Java tests because executable Java did not change.
- Documentation-agent review: This clean documentation-focused context finalized the route
  decision under the General and Planning profiles without changing executable code.
- Documentation impact: The four planning records now record the completed norm decision and
  advance the summary frontier to E1C. No explanatory documentation update is needed because
  CPU-private execution and public/shared behavior are unchanged.
- Javadoc review: No Javadoc change is needed: the inspected CPU-private contracts accurately
  describe the existing generated route and this decision changes neither a Java API nor an
  implementation contract.
- Glossary impact: No glossary change is needed; this decision introduces no reusable term or
  changes an existing term's meaning.
- No-change review: `ARCHITECTURE.md` and explanatory architecture documentation remain correct
  because module ownership and dependency direction are unchanged. Gradle/build configuration,
  architecture tests, backend-conformance tests, and integration tests need no update because
  there is no executable, shared-boundary, backend-contract, or end-to-end behavior change.
- Unresolved issues: None for E1B3. CPU 0009 remains Ready and incomplete; E1C is the sole next
  Draft summary frontier.
- Follow-up required: None for E1B3.

Status: Complete
