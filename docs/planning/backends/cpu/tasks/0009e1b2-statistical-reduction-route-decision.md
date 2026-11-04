# Task 0009E1B2: Statistical Reduction Route Decision

## Status

Complete

## Goal

Decide whether the current [CPU](../../../../glossary.md#term-cpu) [`VARIANCE`](../../../../glossary.md#term-variance) and `STANDARD_DEVIATION` reductions retain their specialized generated route or are replaced by [finite](../../../../glossary.md#term-finite) direct typed Java. Select direct Java only when its total implementation and verification cost is lower. This task has no benchmark, performance gate, or performance claim.

## [Scope](../../../../glossary.md#term-scope)

Inspect exactly the two current one-input/one-output statistical meanings: `AggregateReductionKind.VARIANCE` and `AggregateReductionKind.STANDARD_DEVIATION`, each with `StatisticalReductionAttrs`. [Admission](../../../../glossary.md#term-admission) is identical [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), or [`BFLOAT16`](../../../../glossary.md#term-bfloat16) input and output; non-null, normalized, ordered, distinct non-negative selected [axes](../../../../glossary.md#term-axis) (including an empty list); either retained or removed selected [dimensions](../../../../glossary.md#dimension); and non-negative `correction`. The reduced output [Shape](../../../../glossary.md#shape) must be exact, and checked selected-domain count `N` must exceed `correction`.

The decision covers the current static resolved non-negative [layouts](../../../../glossary.md#layout): dense-[linear](../../../../glossary.md#term-linear-meanings) or general-odometer input, and dense-linear or general-odometer [injective](../../../../glossary.md#term-injective) output; [heap](../../../../glossary.md#term-heap), [`MemorySegment`](../../../../glossary.md#term-memorysegment), and ordered mixed typed input/output carriers; two [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) boundary carriers; and [scalar](../../../../glossary.md#term-scalar) or disjoint parallel-scalar ranges that each own complete output cells. It preserves the shared `CpuAdvancedReductionIr` / `CpuAdvancedReductionLowering.Geometry` boundary: cold packed input/output bases and static geometry, two passes, no materialized input or partial/combine state, exactly one final typed store for every owned cell, and exact scratch state for every concurrently selected range when output cells exist.

Preserve the current corrected two-pass algorithm exactly: first pass classifies values and uses the per-range `CpuExactSumEmitter` state to form the exact represented [mean](../../../../glossary.md#term-mean) rounded at the current represented seam; second pass uses compensated deviation and square sums, subtracts `deviations² / N`, clamps a negative numerator to [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), divides by `N - correction`, and applies `Math.sqrt` only for [standard deviation](../../../../glossary.md#term-standard-deviation). Empty selected domains and `N <= correction` are rejected cold for statistics; an empty axes list is a point domain and produces zero only when its `N = 1` satisfies `N > correction` (therefore, under the current rule, only with correction zero). Canonical NaN input or any [infinity](../../../../glossary.md#term-infinity) produces [canonical NaN](../../../../glossary.md#term-canonical-nan); negative-zero-only [variance](../../../../glossary.md#term-variance) is positive zero; overflowing [finite](../../../../glossary.md#term-finite) squares produce [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero) unless a special-value rule overrides it. Preserve FLOAT32/BFLOAT16 [narrowing](../../../../glossary.md#term-narrowing), including canonical NaN narrowing.

Compare `CpuStatisticalReductionEmitter`, `CpuExactSumEmitter`, shared typed [carrier](../../../../glossary.md#term-storage-carrier)/address and selected-domain helpers, generator/[artifact](../../../../glossary.md#term-artifact) [identity](../../../../glossary.md#term-identity-meanings), [Prepare](../../../../glossary.md#prepare) declaration/finalization, [immutable](../../../../glossary.md#term-immutable) prepared [binding](../../../../glossary.md#term-binding)/invocation, independent reference [oracle](../../../../glossary.md#term-test-oracle), and current semantic/inventory/evidence [owners](../../../../glossary.md#term-owner) with a complete finite direct replacement and retirement proof. A direct replacement must cold-select finite typed array/segment and dense/general forms and bind an immutable range-owning entry; it must not perform generic carrier, type, layout, kind, reflection, map/string, or virtual semantic [dispatch](../../../../glossary.md#term-dispatch) per [element](../../../../glossary.md#term-element). If generation remains, retain a matching optimal clean-Java oracle: the same two passes, exact-mean state and compensation order, special handling, narrowing, and one store, with no avoidable allocation, boxing, reflection, helper semantic dispatch, or per-element virtual indirection in the generated hot loop.

## Out of [scope](../../../../glossary.md#term-scope)

`LOG_SUM_EXP` (Complete E1B1), `L1_NORM`/`L2_NORM` (Draft E1B3), [softmax](../../../../glossary.md#softmax--log-softmax)/[log-softmax](../../../../glossary.md#softmax--log-softmax), [normalization](../../../../glossary.md#term-normalization), [loss](../../../../glossary.md#loss), masked or ordinary reductions, public [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) semantics or [gradients](../../../../glossary.md#term-gradient), architecture/build/shared-boundary changes, new resource kinds, [vector](../../../../glossary.md#term-vector)/native routes, benchmarks, and performance assertions. Do not create detailed E1B3 or later specifications. Legacy `pre-rewrite` is read-only behavioral evidence; do not copy source, packages, or dependencies.

## Architecture references and constraints

Read [the architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, [current architecture plan](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [documentation rules](../../../../developer-guide/documentation-rules.md), [CPU](../../../../glossary.md#term-cpu) master plan, parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [E1](0009e1-partial-integral-reduction-direct-java-migration.md), [E1A](0009e1a-masked-reduction-route-decision.md), [E1B1](0009e1b1-log-sum-exp-route-decision.md), and [0007D](0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md).

`ARCHITECTURE.md` is authoritative. CPU [Prepare](../../../../glossary.md#prepare) alone owns [lowering](../../../../glossary.md#lowering), [finite](../../../../glossary.md#term-finite) route selection, resource declaration, and finalization after shared slot assignment; [Runtime](../../../../glossary.md#term-runtime) only invokes [immutable](../../../../glossary.md#term-immutable) prepared work. Preserve [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) caller carriers, [run](../../../../glossary.md#term-run)-owned scratch, one [`RunState`](../../../../glossary.md#run-state--runstate) for an active run, cold [validation](../../../../glossary.md#term-validation) before mutation or worker submission, and prepared-region [ownership](../../../../glossary.md#term-ownership-meanings). Do not add a public/shared [dependency](../../../../glossary.md#term-dependency), generic executor, workspace/resource kind, architecture rule, or build change; stop and report any need for one.

## Package impact

Existing packages only:

- `internal.ir` and `internal.lowering` retain structural [identity](../../../../glossary.md#term-identity-meanings) and static geometry.
- `internal.codegen.emit` retains the generated statistical body and exact-state/helper seams if generation is selected.
- `internal.prepare`, `internal.executable`, and `internal.cache` retain cold selection, resource finalization, [artifact](../../../../glossary.md#term-artifact) identity, and [immutable](../../../../glossary.md#term-immutable) invocation.
- `internal.reference` retains the independent [oracle](../../../../glossary.md#term-test-oracle).

A retained route adds no type. A selected direct route may add at most one narrowly named [CPU](../../../../glossary.md#term-cpu)-private statistical executable [owner](../../../../glossary.md#term-owner) in an existing package; it must not absorb [norm](../../../../glossary.md#term-norm) or [log-sum-exp](../../../../glossary.md#log-sum-exp) algorithms.

## Affected files and maximum [scope](../../../../glossary.md#term-scope)

Expected [planning](../../../../glossary.md#term-planning) paths are this task, [CPU](../../../../glossary.md#term-cpu) master plan, parent CPU 0009, and roadmap. Review-only seams are `CpuCapabilityProvider`, `CpuAdvancedReductionIr`, `CpuAdvancedReductionLowering`, `CpuStatisticalReductionEmitter`, `CpuExactSumEmitter`, `CpuNormEmitter`, `CpuClassFileKernelGenerator`, preparation/finalization and `CpuPreparedExecutable`, `CpuAdvancedReductionReferenceKernel`, and focused capability/[IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/generated/reference/[prepare](../../../../glossary.md#prepare)/executable/inventory/evidence tests.

Implementation may modify or create at most 30 repository paths: at most 16 production/[Javadoc](../../../../glossary.md#term-javadoc), ten test, and these four planning paths. At most one new CPU-private direct production type is permitted. A 31st path, a second new production type, changed shared advanced geometry/resource contract, [materialization](../../../../glossary.md#materialization), partial/combine state, shared/public/build/architecture/[conformance](../../../../glossary.md#term-conformance)/integration path, or generic [element](../../../../glossary.md#term-element) [dispatch](../../../../glossary.md#term-dispatch) is a stop-and-replan condition.

## Acceptance criteria

- The final decision covers the complete current two-kind [matrix](../../../../glossary.md#term-matrix-meanings) and says whether generation is retained or a [finite](../../../../glossary.md#term-finite) direct typed route replaces it, with concrete implementation-plus-verification cost evidence and no performance claim.
- The chosen route preserves `StatisticalReductionAttrs` [axes](../../../../glossary.md#term-axis), keep-[dimensions](../../../../glossary.md#dimension), correction, `N > correction` [admission](../../../../glossary.md#term-admission), [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/[carrier](../../../../glossary.md#term-storage-carrier)/range boundary, complete-cell [ownership](../../../../glossary.md#term-ownership-meanings), [run](../../../../glossary.md#term-run)-owned aligned exact-state [slice](../../../../glossary.md#slice) isolation, and cold failure before writes/submission.
- It preserves corrected exact-[mean](../../../../glossary.md#term-mean) two-pass semantics, empty/point-domain distinction, [singleton](../../../../glossary.md#term-singleton), invalid denominator, [canonical NaN](../../../../glossary.md#term-canonical-nan)/[infinity](../../../../glossary.md#term-infinity)/[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)/[overflow](../../../../glossary.md#term-overflow) behavior, represented [narrowing](../../../../glossary.md#term-narrowing), and one final output store exactly as current source and tests require.
- A direct route has finite cold typed carrier/layout selection and [immutable](../../../../glossary.md#term-immutable) [binding](../../../../glossary.md#term-binding) with no generic per-[element](../../../../glossary.md#term-element) [dispatch](../../../../glossary.md#term-dispatch); generation is retired only after all selected [artifact](../../../../glossary.md#term-artifact), [cache](../../../../glossary.md#term-cache), invocation, inventory, evidence, and no-selected-reference proof is updated.
- A retained route changes no executable path and records the typed carrier/layout, scratch, emitter/helper, artifact/binding, [oracle](../../../../glossary.md#term-test-oracle), and retirement costs that make replacement larger.
- Focused semantic, special-value, correction/rejection, type/narrowing, layout/carrier, range/[canary](../../../../glossary.md#term-canary), scratch isolation, preparation/finalization, invocation, reference, inventory, and route-retirement tests pass. Retained generation receives proportionate class-file/member and hot-loop inspection against the clean-Java oracle; direct code receives equivalent source and call-site hygiene review.
- Before `Complete`, a separate documentation-focused clean context finalizes affected Javadocs, explanatory documentation, glossary impact, [planning](../../../../glossary.md#term-planning) status, and documentation [validation](../../../../glossary.md#term-validation).

## Tests / [validation](../../../../glossary.md#term-validation)

After executable stabilization, the implementation context runs focused [CPU](../../../../glossary.md#term-cpu) tests selected from the existing statistical [owners](../../../../glossary.md#term-owner), including capability, advanced [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/generated/reference, specialized [matrix](../../../../glossary.md#term-matrix-meanings)/direct-evidence/coverage inventory, and affected [prepare](../../../../glossary.md#prepare)/finalizer/executable tests. Record exact commands and results here. Do not [run](../../../../glossary.md#term-run) benchmarks or repository-wide tests.

The documentation context reuses stable Java evidence unless it changes executable Java; it checks Markdown links, anchors, fences, terminology/glossary impact, changed-path [scope](../../../../glossary.md#term-scope), statuses, dependencies, and frontier consistency, then runs:

```bash
git diff --check
git status --short -uall
```

Repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration, and performance validation are deferred to CPU 0009 final [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci) because this task must not change a shared boundary.

## Dependencies and follow-up tasks

- E1, E1A, and E1B1 are Complete; E1B1 retains the zero-workspace [log-sum-exp](../../../../glossary.md#log-sum-exp) route.
- Complete 0007D supplies the existing advanced [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering), statistical generated emitter, exact-state workspace, reference [oracle](../../../../glossary.md#term-test-oracle), and focused evidence.
- E1B3 remains Draft and depends on this decision; E1C, E2, E3, F, and G remain ordered Draft summaries only.

## Architecture impact

Expected impact: None. Stop and report any public, [dependency](../../../../glossary.md#term-dependency), resource, [module](../../../../glossary.md#term-module), or architecture conflict rather than changing that boundary.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU task 0009E1B2. Work on the existing worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, current architecture plan, Planning Guide, CPU master plan, parent CPU 0009, completed E1/E1A/E1B1/0007D, this task, current statistical advanced-reduction source and focused tests, documentation rules, and General/Planning profiles. Implement exactly this route decision. Choose finite direct typed Java only if total implementation and verification cost is lower; otherwise retain generation with concrete evidence. Preserve the complete current VARIANCE/STANDARD_DEVIATION boundary. Stop on an architecture or scope conflict. Hand stabilized executable diff and exact test evidence to a separate documentation-focused clean context before marking Complete. Do not run benchmarks or repository-wide tests. Do not commit, push, or stage.
```

## Local decisions

E1B2 owns both statistics because they share the corrected exact-[mean](../../../../glossary.md#term-mean) state, two-pass traversal, denominator correction, and only differ in the final [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions). It follows E1B1 because the logarithmic route has zero workspace; it precedes E1B3 because norms have different numerical state and finishing rules despite shared geometry.

## Known limitations

This is a route decision, not a numerical redesign, public [API](../../../../glossary.md#term-application-programming-interface-api) change, or performance result. Only current static [scalar](../../../../glossary.md#term-scalar)/parallel-scalar resolved-[layout](../../../../glossary.md#layout) forms are in [scope](../../../../glossary.md#term-scope); dynamic [Shapes](../../../../glossary.md#shape), vectorization, [fusion](../../../../glossary.md#term-fusion), [materialization](../../../../glossary.md#materialization), partial/combine execution, and native routes remain out.

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context's unchanged executable evidence was independently reviewed by the
mandatory clean-context documentation-finalization pass on 2026-09-09. No executable Java changed
after that [run](../../../../glossary.md#term-run), so the pass reused rather than reran the focused [CPU](../../../../glossary.md#term-cpu) command.

The implementation review retained generated execution for both [`VARIANCE`](../../../../glossary.md#term-variance) and
`STANDARD_DEVIATION`. No executable source or test changed. The retained route is the lower total
implementation-and-verification-cost option: `CpuStatisticalReductionEmitter` already generates
the exact two-pass typed body for all three represented types and every cold-selected [heap](../../../../glossary.md#term-heap),
[`MemorySegment`](../../../../glossary.md#term-memorysegment), and mixed [carrier](../../../../glossary.md#term-storage-carrier) form. Its generated entry consumes the existing packed
dense/general geometry, complete-cell range, and per-range exact-state [slice](../../../../glossary.md#slice). Replacing it would
require a new [finite](../../../../glossary.md#term-finite) typed direct [owner](../../../../glossary.md#term-owner) plus a second maintenance representation of the large
exact-sum/[mean](../../../../glossary.md#term-mean) state algorithm, or moving that algorithm out of the generation-only emitter;
either option would additionally require a complete direct-entry carrier/[layout](../../../../glossary.md#layout) [matrix](../../../../glossary.md#term-matrix-meanings) and
retirement proof for the generator [dispatch](../../../../glossary.md#term-dispatch), [schema](../../../../glossary.md#term-schema)/[artifact](../../../../glossary.md#term-artifact) [identity](../../../../glossary.md#term-identity-meanings), artifact store, finalizer,
[immutable](../../../../glossary.md#term-immutable) [binding](../../../../glossary.md#term-binding), invocation, inventory, and evidence owners. That is more implementation and
verification work, with no task-authorized performance result. This is a cost decision only, not a
performance claim.

Source and generated-Class-File review found that the retained body keeps the required algorithm:
it resets one invocation-private exact state per owned output cell, forms the represented exact
mean in the first pass, then performs compensated deviations and squares in the second pass,
applies the correction expression and optional `Math.sqrt`, resolves [canonical](../../../../glossary.md#term-canonical-form)-[NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity)
priority, narrows once, and stores once. `CpuGeneratedDirectEvidenceClosureTest` parses generated
`VARIANCE` and `STANDARD_DEVIATION` representatives and verifies one typed static entry, no
fields, `Object`, bootstrap methods, method handles, reflection, or collection/map member
references; its representative inspection also verifies segment-layout construction is hoisted
before segment access. `CpuAdvancedReductionGeneratedKernelTest` verifies numerical-emitter
[ownership](../../../../glossary.md#term-ownership-meanings) and the finite, special-value, correction, type, carrier, and point-domain cases. The
independent reference [oracle](../../../../glossary.md#term-test-oracle) separately implements the corrected two-pass mathematical order
without [lowering](../../../../glossary.md#lowering), packed geometry, generated helpers, or exact-state workspace. This is
proportionate generated-body and oracle evidence under the architecture rule; it does not claim
literal Class-File identity, [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) assembly identity, or a performance result.

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

The corresponding [Gradle](../../../../glossary.md#term-gradle) XML files under `backends/cpu/build/test-results/test/` record exactly
168 tests, zero failures, zero errors, and one skip. The skip is the expected opt-in
`CpuGeneratedKernelPersistenceEvidenceTest`; no benchmark or repository-wide suite ran.
The selected owners cover capability [admission](../../../../glossary.md#term-admission), [IR](../../../../glossary.md#term-intermediate-representation-ir)/lowering, special values and [narrowing](../../../../glossary.md#term-narrowing),
carrier/layout/range and scratch isolation, preparation/finalization, immutable invocation,
reference behavior, generated artifact/[cache](../../../../glossary.md#term-cache), and generated/inventory evidence.

The documentation-finalization pass reviewed the [Planning](../../../../glossary.md#term-planning) and General profiles; this task, the
CPU master plan, parent CPU 0009 task, and roadmap; current generated emitter/exact-state,
lowering/geometry, generator/schema/artifact-cache, finalization/binding, reference-oracle, and
focused-test seams; and the final changed-path [scope](../../../../glossary.md#term-scope). Markdown links, anchors, fences,
terminology, statuses, dependencies, and the E1B frontier are consistent. `git diff --check` and
`git status --short -uall` passed during this pass. [Javadoc](../../../../glossary.md#term-javadoc), explanatory documentation, glossary,
architecture documentation, Gradle/build configuration, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) tests, and integration
tests require no update: this retains an unchanged CPU-private generated route behind existing
public, shared [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), and backend contracts, and introduces no term, [dependency](../../../../glossary.md#term-dependency),
resource kind, build behavior, or cross-backend/end-to-end behavior.
## Implementation notes

Retain the bounded generated statistical route unchanged. The current
[lowering](../../../../glossary.md#lowering) declares exactly two [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) boundary [buffers](../../../../glossary.md#term-buffer) and the existing aligned exact-state
workspace only when output cells exist; finalization preserves [artifact](../../../../glossary.md#term-artifact) [identity](../../../../glossary.md#term-identity-meanings), and [binding](../../../../glossary.md#term-binding)
packs cold bases/geometry before invoking [immutable](../../../../glossary.md#term-immutable) typed entries. Complete output-cell ranges
receive disjoint [run](../../../../glossary.md#term-run)-owned scratch slices, and cold overlap [validation](../../../../glossary.md#term-validation) precedes writes or worker
submission. Direct replacement and generator retirement would add a [finite](../../../../glossary.md#term-finite) [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout) route
[matrix](../../../../glossary.md#term-matrix-meanings) and duplicate the statistical state algorithm, binding, artifact/[cache](../../../../glossary.md#term-cache), invocation,
inventory, and retirement proof; that has greater implementation and verification cost. This is
not a performance claim and has no benchmark gate.

## Completion summary

- Completed changes: Finalized the decision to retain the current generated [`VARIANCE`](../../../../glossary.md#term-variance) and
  `STANDARD_DEVIATION` route because direct replacement has greater implementation and
  verification cost; no executable behavior changed.
- Files changed or created: This task, the CPU master plan, parent CPU 0009 task, and the global
  roadmap.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Tests and validation: Reused the implementation context's focused 13-class CPU command; its
  XML results total 168 tests, zero failures, zero errors, and one expected opt-in skip. The
  documentation pass checked planning consistency, Markdown links/anchors/fences, terminology,
  changed paths, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [planning glossary](../../../../glossary.md#term-planning))</span>`git diff --check`, and `git status --short -uall`; no benchmark,
  repository-wide, architecture, conformance, or integration suite ran.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- Documentation-agent review: The mandatory independent clean-context documentation-finalization
  pass completed this review without changing executable Java.
- Documentation impact: No Javadoc, explanatory, glossary, architecture, or build update is
  needed because the CPU-private route and all public/shared contracts remain unchanged.
- [Javadoc](../../../../glossary.md#term-javadoc) review: Current Javadoc remains accurate; no Java [API](../../../../glossary.md#term-application-programming-interface-api) or implementation contract changed.
- Glossary impact: None; no reusable terminology changed or was introduced.
- Unresolved issues: None.
- Follow-up required: E1B3 norms is the sole next Draft summary; do not create its detailed task
  until it reaches the implementation frontier.

Status: Complete
