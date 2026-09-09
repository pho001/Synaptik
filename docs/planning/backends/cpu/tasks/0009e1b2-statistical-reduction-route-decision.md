# Task 0009E1B2: Statistical Reduction Route Decision

## Status

Complete

## Goal

Decide whether the current CPU `VARIANCE` and `STANDARD_DEVIATION` reductions retain their specialized generated route or are replaced by finite direct typed Java. Select direct Java only when its total implementation and verification cost is lower. This task has no benchmark, performance gate, or performance claim.

## Scope

Inspect exactly the two current one-input/one-output statistical meanings: `AggregateReductionKind.VARIANCE` and `AggregateReductionKind.STANDARD_DEVIATION`, each with `StatisticalReductionAttrs`. Admission is identical `FLOAT64`, `FLOAT32`, or `BFLOAT16` input and output; non-null, normalized, ordered, distinct non-negative selected axes (including an empty list); either retained or removed selected dimensions; and non-negative `correction`. The reduced output Shape must be exact, and checked selected-domain count `N` must exceed `correction`.

The decision covers the current static resolved non-negative layouts: dense-linear or general-odometer input, and dense-linear or general-odometer injective output; heap, `MemorySegment`, and ordered mixed typed input/output carriers; two borrowed boundary carriers; and scalar or disjoint parallel-scalar ranges that each own complete output cells. It preserves the shared `CpuAdvancedReductionIr` / `CpuAdvancedReductionLowering.Geometry` boundary: cold packed input/output bases and static geometry, two passes, no materialized input or partial/combine state, exactly one final typed store for every owned cell, and exact scratch state for every concurrently selected range when output cells exist.

Preserve the current corrected two-pass algorithm exactly: first pass classifies values and uses the per-range `CpuExactSumEmitter` state to form the exact represented mean rounded at the current represented seam; second pass uses compensated deviation and square sums, subtracts `deviations² / N`, clamps a negative numerator to positive zero, divides by `N - correction`, and applies `Math.sqrt` only for standard deviation. Empty selected domains and `N <= correction` are rejected cold for statistics; an empty axes list is a point domain and produces zero only when its `N = 1` satisfies `N > correction` (therefore, under the current rule, only with correction zero). Canonical NaN input or any infinity produces canonical NaN; negative-zero-only variance is positive zero; overflowing finite squares produce positive infinity unless a special-value rule overrides it. Preserve FLOAT32/BFLOAT16 narrowing, including canonical NaN narrowing.

Compare `CpuStatisticalReductionEmitter`, `CpuExactSumEmitter`, shared typed carrier/address and selected-domain helpers, generator/artifact identity, Prepare declaration/finalization, immutable prepared binding/invocation, independent reference oracle, and current semantic/inventory/evidence owners with a complete finite direct replacement and retirement proof. A direct replacement must cold-select finite typed array/segment and dense/general forms and bind an immutable range-owning entry; it must not perform generic carrier, type, layout, kind, reflection, map/string, or virtual semantic dispatch per element. If generation remains, retain a matching optimal clean-Java oracle: the same two passes, exact-mean state and compensation order, special handling, narrowing, and one store, with no avoidable allocation, boxing, reflection, helper semantic dispatch, or per-element virtual indirection in the generated hot loop.

## Out of scope

`LOG_SUM_EXP` (Complete E1B1), `L1_NORM`/`L2_NORM` (Draft E1B3), softmax/log-softmax, normalization, loss, masked or ordinary reductions, public Model/Compiler semantics or gradients, architecture/build/shared-boundary changes, new resource kinds, vector/native routes, benchmarks, and performance assertions. Do not create detailed E1B3 or later specifications. Legacy `pre-rewrite` is read-only behavioral evidence; do not copy source, packages, or dependencies.

## Architecture references and constraints

Read [the architecture contract](../../../../../ARCHITECTURE.md), [current architecture plan](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md), CPU master plan, parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [E1](0009e1-partial-integral-reduction-direct-java-migration.md), [E1A](0009e1a-masked-reduction-route-decision.md), [E1B1](0009e1b1-log-sum-exp-route-decision.md), and [0007D](0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md).

`ARCHITECTURE.md` is authoritative. CPU Prepare alone owns lowering, finite route selection, resource declaration, and finalization after shared slot assignment; Runtime only invokes immutable prepared work. Preserve borrowed caller carriers, run-owned scratch, one `RunState` for an active run, cold validation before mutation or worker submission, and prepared-region ownership. Do not add a public/shared dependency, generic executor, workspace/resource kind, architecture rule, or build change; stop and report any need for one.

## Package impact

Existing packages only:

- `internal.ir` and `internal.lowering` retain structural identity and static geometry.
- `internal.codegen.emit` retains the generated statistical body and exact-state/helper seams if generation is selected.
- `internal.prepare`, `internal.executable`, and `internal.cache` retain cold selection, resource finalization, artifact identity, and immutable invocation.
- `internal.reference` retains the independent oracle.

A retained route adds no type. A selected direct route may add at most one narrowly named CPU-private statistical executable owner in an existing package; it must not absorb norm or log-sum-exp algorithms.

## Affected files and maximum scope

Expected planning paths are this task, CPU master plan, parent CPU 0009, and roadmap. Review-only seams are `CpuCapabilityProvider`, `CpuAdvancedReductionIr`, `CpuAdvancedReductionLowering`, `CpuStatisticalReductionEmitter`, `CpuExactSumEmitter`, `CpuNormEmitter`, `CpuClassFileKernelGenerator`, preparation/finalization and `CpuPreparedExecutable`, `CpuAdvancedReductionReferenceKernel`, and focused capability/IR/lowering/generated/reference/prepare/executable/inventory/evidence tests.

Implementation may modify or create at most 30 repository paths: at most 16 production/Javadoc, ten test, and these four planning paths. At most one new CPU-private direct production type is permitted. A 31st path, a second new production type, changed shared advanced geometry/resource contract, materialization, partial/combine state, shared/public/build/architecture/conformance/integration path, or generic element dispatch is a stop-and-replan condition.

## Acceptance criteria

- The final decision covers the complete current two-kind matrix and says whether generation is retained or a finite direct typed route replaces it, with concrete implementation-plus-verification cost evidence and no performance claim.
- The chosen route preserves `StatisticalReductionAttrs` axes, keep-dimensions, correction, `N > correction` admission, Shape/layout/carrier/range boundary, complete-cell ownership, run-owned aligned exact-state slice isolation, and cold failure before writes/submission.
- It preserves corrected exact-mean two-pass semantics, empty/point-domain distinction, singleton, invalid denominator, canonical NaN/infinity/signed-zero/overflow behavior, represented narrowing, and one final output store exactly as current source and tests require.
- A direct route has finite cold typed carrier/layout selection and immutable binding with no generic per-element dispatch; generation is retired only after all selected artifact, cache, invocation, inventory, evidence, and no-selected-reference proof is updated.
- A retained route changes no executable path and records the typed carrier/layout, scratch, emitter/helper, artifact/binding, oracle, and retirement costs that make replacement larger.
- Focused semantic, special-value, correction/rejection, type/narrowing, layout/carrier, range/canary, scratch isolation, preparation/finalization, invocation, reference, inventory, and route-retirement tests pass. Retained generation receives proportionate class-file/member and hot-loop inspection against the clean-Java oracle; direct code receives equivalent source and call-site hygiene review.
- Before `Complete`, a separate documentation-focused clean context finalizes affected Javadocs, explanatory documentation, glossary impact, planning status, and documentation validation.

## Tests / validation

After executable stabilization, the implementation context runs focused CPU tests selected from the existing statistical owners, including capability, advanced IR/lowering/generated/reference, specialized matrix/direct-evidence/coverage inventory, and affected prepare/finalizer/executable tests. Record exact commands and results here. Do not run benchmarks or repository-wide tests.

The documentation context reuses stable Java evidence unless it changes executable Java; it checks Markdown links, anchors, fences, terminology/glossary impact, changed-path scope, statuses, dependencies, and frontier consistency, then runs:

```bash
git diff --check
git status --short -uall
```

Repository-wide, architecture, backend-conformance, integration, and performance validation are deferred to CPU 0009 final checkpoint or CI because this task must not change a shared boundary.

## Dependencies and follow-up tasks

- E1, E1A, and E1B1 are Complete; E1B1 retains the zero-workspace log-sum-exp route.
- Complete 0007D supplies the existing advanced IR/lowering, statistical generated emitter, exact-state workspace, reference oracle, and focused evidence.
- E1B3 remains Draft and depends on this decision; E1C, E2, E3, F, and G remain ordered Draft summaries only.

## Architecture impact

Expected impact: None. Stop and report any public, dependency, resource, module, or architecture conflict rather than changing that boundary.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU task 0009E1B2. Work on the existing worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, current architecture plan, Planning Guide, CPU master plan, parent CPU 0009, completed E1/E1A/E1B1/0007D, this task, current statistical advanced-reduction source and focused tests, documentation rules, and General/Planning profiles. Implement exactly this route decision. Choose finite direct typed Java only if total implementation and verification cost is lower; otherwise retain generation with concrete evidence. Preserve the complete current VARIANCE/STANDARD_DEVIATION boundary. Stop on an architecture or scope conflict. Hand stabilized executable diff and exact test evidence to a separate documentation-focused clean context before marking Complete. Do not run benchmarks or repository-wide tests. Do not commit, push, or stage.
```

## Local decisions

E1B2 owns both statistics because they share the corrected exact-mean state, two-pass traversal, denominator correction, and only differ in the final square root. It follows E1B1 because the logarithmic route has zero workspace; it precedes E1B3 because norms have different numerical state and finishing rules despite shared geometry.

## Known limitations

This is a route decision, not a numerical redesign, public API change, or performance result. Only current static scalar/parallel-scalar resolved-layout forms are in scope; dynamic Shapes, vectorization, fusion, materialization, partial/combine execution, and native routes remain out.

## Validation evidence

The implementation context's unchanged executable evidence was independently reviewed by the
mandatory clean-context documentation-finalization pass on 2026-09-09. No executable Java changed
after that run, so the pass reused rather than reran the focused CPU command.

The implementation review retained generated execution for both `VARIANCE` and
`STANDARD_DEVIATION`. No executable source or test changed. The retained route is the lower total
implementation-and-verification-cost option: `CpuStatisticalReductionEmitter` already generates
the exact two-pass typed body for all three represented types and every cold-selected heap,
`MemorySegment`, and mixed carrier form. Its generated entry consumes the existing packed
dense/general geometry, complete-cell range, and per-range exact-state slice. Replacing it would
require a new finite typed direct owner plus a second maintenance representation of the large
exact-sum/mean state algorithm, or moving that algorithm out of the generation-only emitter;
either option would additionally require a complete direct-entry carrier/layout matrix and
retirement proof for the generator dispatch, schema/artifact identity, artifact store, finalizer,
immutable binding, invocation, inventory, and evidence owners. That is more implementation and
verification work, with no task-authorized performance result. This is a cost decision only, not a
performance claim.

Source and generated-Class-File review found that the retained body keeps the required algorithm:
it resets one invocation-private exact state per owned output cell, forms the represented exact
mean in the first pass, then performs compensated deviations and squares in the second pass,
applies the correction expression and optional `Math.sqrt`, resolves canonical-NaN/infinity
priority, narrows once, and stores once. `CpuGeneratedDirectEvidenceClosureTest` parses generated
`VARIANCE` and `STANDARD_DEVIATION` representatives and verifies one typed static entry, no
fields, `Object`, bootstrap methods, method handles, reflection, or collection/map member
references; its representative inspection also verifies segment-layout construction is hoisted
before segment access. `CpuAdvancedReductionGeneratedKernelTest` verifies numerical-emitter
ownership and the finite, special-value, correction, type, carrier, and point-domain cases. The
independent reference oracle separately implements the corrected two-pass mathematical order
without lowering, packed geometry, generated helpers, or exact-state workspace. This is
proportionate generated-body and oracle evidence under the architecture rule; it does not claim
literal Class-File identity, JIT assembly identity, or a performance result.

The implementation context ran the following focused command on 2026-09-09:

```bash
./gradlew :backends:cpu:test \
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

The corresponding Gradle XML files under `backends/cpu/build/test-results/test/` record exactly
168 tests, zero failures, zero errors, and one skip. The skip is the expected opt-in
`CpuGeneratedKernelPersistenceEvidenceTest`; no benchmark or repository-wide suite ran.
The selected owners cover capability admission, IR/lowering, special values and narrowing,
carrier/layout/range and scratch isolation, preparation/finalization, immutable invocation,
reference behavior, generated artifact/cache, and generated/inventory evidence.

The documentation-finalization pass reviewed the Planning and General profiles; this task, the
CPU master plan, parent CPU 0009 task, and roadmap; current generated emitter/exact-state,
lowering/geometry, generator/schema/artifact-cache, finalization/binding, reference-oracle, and
focused-test seams; and the final changed-path scope. Markdown links, anchors, fences,
terminology, statuses, dependencies, and the E1B frontier are consistent. `git diff --check` and
`git status --short -uall` passed during this pass. Javadoc, explanatory documentation, glossary,
architecture documentation, Gradle/build configuration, backend-conformance tests, and integration
tests require no update: this retains an unchanged CPU-private generated route behind existing
public, shared Prepare, Runtime, and backend contracts, and introduces no term, dependency,
resource kind, build behavior, or cross-backend/end-to-end behavior.
## Implementation notes

Retain the bounded generated statistical route unchanged. The current
lowering declares exactly two borrowed boundary buffers and the existing aligned exact-state
workspace only when output cells exist; finalization preserves artifact identity, and binding
packs cold bases/geometry before invoking immutable typed entries. Complete output-cell ranges
receive disjoint run-owned scratch slices, and cold overlap validation precedes writes or worker
submission. Direct replacement and generator retirement would add a finite carrier/layout route
matrix and duplicate the statistical state algorithm, binding, artifact/cache, invocation,
inventory, and retirement proof; that has greater implementation and verification cost. This is
not a performance claim and has no benchmark gate.

## Completion summary

- Completed changes: Finalized the decision to retain the current generated `VARIANCE` and
  `STANDARD_DEVIATION` route because direct replacement has greater implementation and
  verification cost; no executable behavior changed.
- Files changed or created: This task, the CPU master plan, parent CPU 0009 task, and the global
  roadmap.
- Tests and validation: Reused the implementation context's focused 13-class CPU command; its
  XML results total 168 tests, zero failures, zero errors, and one expected opt-in skip. The
  documentation pass checked planning consistency, Markdown links/anchors/fences, terminology,
  changed paths, `git diff --check`, and `git status --short -uall`; no benchmark,
  repository-wide, architecture, conformance, or integration suite ran.
- Documentation-agent review: The mandatory independent clean-context documentation-finalization
  pass completed this review without changing executable Java.
- Documentation impact: No Javadoc, explanatory, glossary, architecture, or build update is
  needed because the CPU-private route and all public/shared contracts remain unchanged.
- Javadoc review: Current Javadoc remains accurate; no Java API or implementation contract changed.
- Glossary impact: None; no reusable terminology changed or was introduced.
- Unresolved issues: None.
- Follow-up required: E1B3 norms is the sole next Draft summary; do not create its detailed task
  until it reaches the implementation frontier.

Status: Complete
