# Task 0009E1B1: [Log-Sum-Exp](../../../../glossary.md#log-sum-exp) Route Decision

## Reader context

This task records retaining generated LOG_SUM_EXP because a finite direct replacement would add
typed implementation, binding, and retirement verification work. `CpuLogSumExpEmitter` owns the
numerical body; `CpuAdvancedReductionIr` identifies the operation, while
`CpuAdvancedReductionLowering.Geometry` describes static addressing and complete-output-cell ranges.
The shared geometry does not make statistical or norm algorithms part of this decision.

Here, Prepare selects and finalizes typed heap/`MemorySegment` entries; Runtime invokes immutable
prepared work over borrowed boundary carriers with zero scratch. The structural oracle checks
generated entry shape, member ownership, and hot-loop hygiene. `CpuAdvancedReductionReferenceKernel`
supplies the independent semantic oracle for numerical results. Neither structural inspection nor
the retained-route cost comparison establishes performance.

Current-topic navigation (not a replacement for the historical record):

- [current selected-domain reduction meanings](../../../../reference/tensor/reductions.md#multi-axis-and-statistical-reduction-expressions) — Separates axes, correction, norms, and numerical semantics from historical CPU algorithm choices.
- [backend ownership versus implementation route](../../../../design/notes/cpu-kernel-strategy.md#strategy) — Separates Planning ownership, CPU route choice, finalization, and Runtime invocation.

## Status

Complete

## Goal

Decide whether the current [CPU](../../../../glossary.md#term-cpu) `LOG_SUM_EXP` floating reduction should retain its specialized generated route or be replaced by [finite](../../../../glossary.md#term-finite) direct typed Java. Select direct Java only when it lowers the total implementation and verification cost; this task has no benchmark or performance gate.

## [Scope](../../../../glossary.md#term-scope)

Inspect exactly one static, one-input/one-output advanced-reduction meaning: `AggregateReductionKind.LOG_SUM_EXP` with `MultiAxisReductionAttrs`, identical [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), or [`BFLOAT16`](../../../../glossary.md#term-bfloat16) input/output type, normalized ordered selected [axes](../../../../glossary.md#term-axis), and either retained or removed selected [dimensions](../../../../glossary.md#dimension). The decision covers the current resolved non-negative dense-[linear](../../../../glossary.md#term-linear-meanings) and general-odometer [layouts](../../../../glossary.md#layout), [heap](../../../../glossary.md#term-heap), [`MemorySegment`](../../../../glossary.md#term-memorysegment), and mixed typed carriers, [injective](../../../../glossary.md#term-injective) output, and [scalar](../../../../glossary.md#term-scalar) or disjoint complete-output-cell ranges.

Preserve the current shared `CpuAdvancedReductionIr` and `CpuAdvancedReductionLowering.Geometry` boundary: two [buffers](../../../../glossary.md#term-buffer), cold packed bases/geometry, complete output-cell [ownership](../../../../glossary.md#term-ownership-meanings), static selected-domain count, no [materialization](../../../../glossary.md#materialization), no partial or combine state, and zero [log-sum-exp](../../../../glossary.md#log-sum-exp) scratch. Preserve the existing two-pass maximum-shift and compensated exponential-sum algorithm, [singleton](../../../../glossary.md#term-singleton) represented-value path, empty-domain [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero), [canonical NaN](../../../../glossary.md#term-canonical-nan), [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), [narrowing](../../../../glossary.md#term-narrowing), and cold [validation](../../../../glossary.md#term-validation) before mutation or worker submission.

Compare `CpuLogSumExpEmitter` together with its shared [carrier](../../../../glossary.md#term-storage-carrier)/address/classification helpers, generated [artifact](../../../../glossary.md#term-artifact) [identity](../../../../glossary.md#term-identity-meanings) and invocation, finalization, executable [binding](../../../../glossary.md#term-binding), reference [oracle](../../../../glossary.md#term-test-oracle), and focused inventory/evidence [owners](../../../../glossary.md#term-owner) against a complete [finite](../../../../glossary.md#term-finite) direct replacement and retirement proof. A selected direct route must cold-select finite typed array/segment, dense/general entry forms and must not introduce generic per-[element](../../../../glossary.md#term-element) carrier, kind, layout, or reflection [dispatch](../../../../glossary.md#term-dispatch). A retained route changes no executable production/test path and records why replacement costs more overall.

The task must not decide [variance](../../../../glossary.md#term-variance), [standard deviation](../../../../glossary.md#term-standard-deviation), [L1 norm](../../../../glossary.md#norms-elementary-functions-and-probability-distributions), or [L2 norm](../../../../glossary.md#norms-elementary-functions-and-probability-distributions). Those owners share geometry but have materially separate corrected/exact-state or [norm](../../../../glossary.md#term-norm) numerical algorithms and remain ordered E1B2/E1B3 summaries.

## Out of [scope](../../../../glossary.md#term-scope)

[`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM`, masked and ordinary reductions, [softmax](../../../../glossary.md#softmax--log-softmax)/[log-softmax](../../../../glossary.md#softmax--log-softmax), normalizations, [losses](../../../../glossary.md#loss), [vector](../../../../glossary.md#term-vector)/native routes, [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) semantics or [gradients](../../../../glossary.md#term-gradient), shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) contracts, architecture, build files, benchmarks, and new performance claims. Do not create detailed E1B2, E1B3, E1C, E2, E3, F, or G specifications. Legacy is read-only behavioral evidence; do not copy its source.

## Architecture references

Read [the architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, [the current architecture plan](../../../../architecture/current-architecture-plan.md), [the Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [documentation rules](../../../../developer-guide/documentation-rules.md), the [CPU](../../../../glossary.md#term-cpu) master plan, parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [CPU 0009E1](0009e1-partial-integral-reduction-direct-java-migration.md), completed [CPU 0009E1A](0009e1a-masked-reduction-route-decision.md), and completed [CPU 0007D](0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md).

## Architecture constraints

`ARCHITECTURE.md` is authoritative. [CPU](../../../../glossary.md#term-cpu) [Prepare](../../../../glossary.md#prepare) owns [lowering](../../../../glossary.md#lowering), route selection, exact-resource declaration, and finalization after shared slot assignment; [Runtime](../../../../glossary.md#term-runtime) invokes [immutable](../../../../glossary.md#term-immutable) prepared work only. Preserve [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) carriers, [run](../../../../glossary.md#term-run)-owned resources, one [`RunState`](../../../../glossary.md#run-state--runstate) per active run, cold [validation](../../../../glossary.md#term-validation) before writes, and prepared-region [ownership](../../../../glossary.md#term-ownership-meanings). A retained generated loop remains subject to the optimal-clean-Java [oracle](../../../../glossary.md#term-test-oracle) and hot-loop hygiene: no avoidable allocation, boxing, reflection, string/map [dispatch](../../../../glossary.md#term-dispatch), semantic dispatch, or per-[element](../../../../glossary.md#term-element) virtual indirection. Do not add a public/shared [dependency](../../../../glossary.md#term-dependency), resource kind, generic executor, architecture rule, or build change; stop and report any need for one.

## Package impact

Existing packages used:

- `internal.ir` — shared advanced-reduction structural [identity](../../../../glossary.md#term-identity-meanings).
- `internal.lowering` — shared static output-cell geometry.
- `internal.codegen.emit` — generated [log-sum-exp](../../../../glossary.md#log-sum-exp) body and its current private helper seams.
- `internal.prepare`, `internal.executable`, and `internal.cache` — cold route realization, [artifact](../../../../glossary.md#term-artifact) finalization, and invocation.
- `internal.reference` — independent mathematical [oracle](../../../../glossary.md#term-test-oracle).

Packages added or changed:

- None for a retained route. A selected direct route may add one narrowly named [CPU](../../../../glossary.md#term-cpu)-private direct log-sum-exp [owner](../../../../glossary.md#term-owner) in an existing `internal` package only after its [finite](../../../../glossary.md#term-finite) entry/type map is documented in this task and the CPU master plan.

Type placement:

- No new type is authorized for a retained route.
- A selected direct owner, if needed, belongs beside the existing CPU-private executable route; it must not be public or shared and must not absorb statistical or [norm](../../../../glossary.md#term-norm) algorithms.

## Affected files

Expected decision/[planning](../../../../glossary.md#term-planning) paths:

- this task, `docs/planning/backends/cpu/master-plan.md`, parent [CPU](../../../../glossary.md#term-cpu) 0009, and `docs/planning/roadmap.md`.

Review-only implementation seams:

- `CpuAdvancedReductionIr`, `CpuAdvancedReductionLowering`, `CpuLogSumExpEmitter`, its shared `CpuNormEmitter` helper seam, `CpuGeneratedKernel`, generated [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), [partition](../../../../glossary.md#partition) preparation/finalization, `CpuPreparedExecutable`, `CpuAdvancedReductionReferenceKernel`, and their focused capability/[IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/generated/reference/[prepare](../../../../glossary.md#prepare)/executable/inventory/evidence tests.

## Maximum [scope](../../../../glossary.md#term-scope)

The implementation, tests, documentation, and [planning](../../../../glossary.md#term-planning) may modify or create at most 28 repository paths: at most 15 production/[Javadoc](../../../../glossary.md#term-javadoc) paths, eight test paths, and the four named planning paths. At most one new [CPU](../../../../glossary.md#term-cpu)-private direct [log-sum-exp](../../../../glossary.md#log-sum-exp) production type is permitted.

A 29th path, a second new production type, any change to shared advanced geometry that changes statistics or norms, a new workspace/resource kind, [materialization](../../../../glossary.md#materialization), partial/combine state, a shared/public/build/architecture/[conformance](../../../../glossary.md#term-conformance)/integration path, or a direct route with generic [element](../../../../glossary.md#term-element) [dispatch](../../../../glossary.md#term-dispatch) is a stop-and-replan condition.

## Acceptance criteria

- The final decision covers the complete current `LOG_SUM_EXP` [matrix](../../../../glossary.md#term-matrix-meanings) and states whether the generated route is retained or a [finite](../../../../glossary.md#term-finite) direct typed route replaces it; it gives a concrete implementation-plus-verification cost comparison and makes no performance claim.
- `LOG_SUM_EXP` preserves the current attribute/type/[layout](../../../../glossary.md#layout)/[carrier](../../../../glossary.md#term-storage-carrier)/range [admission](../../../../glossary.md#term-admission) and exact rejection boundary; statistics and norms remain untouched and independently ordered.
- The selected implementation preserves the exact two-pass max-shift, compensated-sum, [singleton](../../../../glossary.md#term-singleton), empty-domain, special-value, represented [narrowing](../../../../glossary.md#term-narrowing), one-final-store, complete-cell range, and cold-failure-before-write behavior documented by the current source and tests.
- If direct is selected, finite cold route selection and typed entry [descriptors](../../../../glossary.md#term-descriptor) cover the current carrier/layout forms without generic per-[element](../../../../glossary.md#term-element) [dispatch](../../../../glossary.md#term-dispatch); every selected generated [artifact](../../../../glossary.md#term-artifact), [cache](../../../../glossary.md#term-cache) route, invocation path, inventory row, and test/evidence reference is retired or updated.
- If generation is retained, production Java and executable tests remain unchanged and the task records the retained emitter/helper, typed carrier/layout, artifact, [binding](../../../../glossary.md#term-binding), [oracle](../../../../glossary.md#term-test-oracle), and retirement costs that make direct replacement larger.
- Focused semantic, special-value, layout/carrier, range/[canary](../../../../glossary.md#term-canary), [prepare](../../../../glossary.md#prepare)/finalization, executable, reference, inventory, and route-retirement tests pass for the selected route. Generated code, when retained, receives proportionate class-file/member/hot-loop inspection against a matching optimal clean-Java oracle; direct code receives equivalent source/call-site hygiene review.
- A separate documentation-focused clean context finalizes any affected Javadocs, explanatory documentation, glossary impact, [planning](../../../../glossary.md#term-planning) status, and documentation [validation](../../../../glossary.md#term-validation) in the same overall change. If a retained decision changes only planning, it records the explicit no-change conclusion for Javadocs, glossary, and executable documentation.

## Tests / [validation](../../../../glossary.md#term-validation)

The implementation context runs focused tests selected from the existing [owners](../../../../glossary.md#term-owner) after executable code stabilizes, including the advanced [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/generated/reference suites and the affected capability, [prepare](../../../../glossary.md#prepare)/finalizer, executable, inventory, and evidence suites. It must record exact commands and results in this task; do not [run](../../../../glossary.md#term-run) a repository-wide suite or benchmarks.

The documentation-focused context reuses stable Java evidence unless it changes executable Java. It validates Markdown links, anchors, fences, terminology, whitespace, exact changed paths, status/[dependency](../../../../glossary.md#term-dependency)/frontier consistency, and then runs:

```bash
git diff --check
git status --short -uall
```

Repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration, and performance validation remain deferred to [CPU](../../../../glossary.md#term-cpu) 0009's final [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes no shared boundary.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0009E1A is Complete and retains the masked generated route.
- Complete CPU 0007D supplies the existing advanced [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering), five-kind emitter split, reference [oracle](../../../../glossary.md#term-test-oracle), static floating [matrix](../../../../glossary.md#term-matrix-meanings), and historical generated evidence.
- Existing [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) contracts already [carry](../../../../glossary.md#term-carry) the two-[buffer](../../../../glossary.md#term-buffer), zero-workspace [log-sum-exp](../../../../glossary.md#log-sum-exp) route without modification.

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0009E1B2 remains a Draft summary for [variance](../../../../glossary.md#term-variance) and [standard deviation](../../../../glossary.md#term-standard-deviation); it depends on this decision because its corrected two-pass exact-[mean](../../../../glossary.md#term-mean) workspace route must not be folded into it.
- CPU 0009E1B3 remains a Draft summary for [L1](../../../../glossary.md#term-l1)/[L2](../../../../glossary.md#term-l2) norms; it follows E1B2 and keeps its separate exact-absolute-sum/scaled-squares numerical [ownership](../../../../glossary.md#term-ownership-meanings).
- CPU 0009E1C, E2, E3, F, and G remain ordered summaries only.

## Architecture impact

Expected impact: None.

If the decision requires a public/shared contract, a [dependency](../../../../glossary.md#term-dependency), a resource kind, a [module](../../../../glossary.md#term-module) change, or an architecture rule, stop and report the conflict rather than changing that boundary.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0009E1B1. Work on the existing worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, CPU master plan, parent CPU 0009, completed E1/E1A/0007D, this task, current advanced-reduction source and focused tests, documentation rules, and the General and Planning profiles.

Implement exactly this route decision. Choose finite direct typed Java only if it lowers total implementation and verification complexity; otherwise retain generation with concrete evidence. Preserve the complete current LOG_SUM_EXP boundary and do not decide statistics or norms. Stop on an architecture or scope conflict. Hand any stabilized executable diff and test evidence to a separate documentation-focused clean context before marking the task Complete. Do not run benchmarks or repository-wide tests. Do not commit, push, or stage.
```

## Local decisions

- Split E1B by numerical [ownership](../../../../glossary.md#term-ownership-meanings), not by shared geometry. `LOG_SUM_EXP` owns a two-pass max-shift/compensated exponential sum with zero workspace; statistics own corrected exact-[mean](../../../../glossary.md#term-mean) state and a distinct second pass; norms own exact absolute summation or scaled squares. A single migration decision would exceed the useful review and verification boundary.
- The first detailed child is [log-sum-exp](../../../../glossary.md#log-sum-exp) because it is the only zero-workspace advanced member and therefore exposes the direct-versus-generated route question without statistical scratch or [norm](../../../../glossary.md#term-norm) state decisions.

## Known limitations

- This is a route decision, not a new numerical algorithm, public [API](../../../../glossary.md#term-application-programming-interface-api), or performance result.
- Only the already supported static resolved-[layout](../../../../glossary.md#layout) [scalar](../../../../glossary.md#term-scalar)/parallel-scalar [CPU](../../../../glossary.md#term-cpu) route is in [scope](../../../../glossary.md#term-scope); dynamic [Shapes](../../../../glossary.md#shape), vectorization, [fusion](../../../../glossary.md#term-fusion), [materialization](../../../../glossary.md#materialization), partial/combine execution, and native routes remain outside it.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context, 2026-09-09: retained the current generated `LOG_SUM_EXP` route after
reviewing the advanced [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering), emitter/shared traversal and [carrier](../../../../glossary.md#term-storage-carrier) helpers, generator,
[artifact](../../../../glossary.md#term-artifact) [identity](../../../../glossary.md#term-identity-meanings), preparation/finalization, typed executable [binding](../../../../glossary.md#term-binding), reference [oracle](../../../../glossary.md#term-test-oracle),
inventory/evidence [owners](../../../../glossary.md#term-owner), and focused tests. A [finite](../../../../glossary.md#term-finite) direct replacement would need to duplicate
three represented types, four [heap](../../../../glossary.md#term-heap)/segment carrier pairs, dense and general selected-domain loops,
keep/remove-[dimension](../../../../glossary.md#dimension) mapping, cold typed selection and validation, [immutable](../../../../glossary.md#term-immutable) invocation, and
route/[cache](../../../../glossary.md#term-cache)/inventory retirement proof. Retention instead preserves one specialized generated
entry per existing carrier pattern, with no executable-path change.

The following focused command passed: `./gradlew :backends:cpu:test --tests
io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest --tests
io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest --tests
io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIrTest --tests
io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAdvancedReductionLoweringTest --tests
io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAdvancedReductionGeneratedKernelTest
--tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuAdvancedReductionReferenceTest
--tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuSpecializedGeneratedMatrixTest
--tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedDirectEvidenceClosureTest
--tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageCheckpointTest
--tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedStructuralOracleCatalogTest`.
It ran 69 tests with zero failures, errors, or skips. The generated-class tests and the direct
evidence closure inspect field-free typed static entry [shape](../../../../glossary.md#shape), closed member [ownership](../../../../glossary.md#term-ownership-meanings), no bootstrap
methods, no [runtime](../../../../glossary.md#term-runtime) numerical helper, and no object/reflection/method-handle/collection [dispatch](../../../../glossary.md#term-dispatch).
Source inspection of `CpuLogSumExpEmitter` confirms direct primitive carrier loads, dense and
odometer selected-domain traversal, two max-shift passes, primitive [compensated sum](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor), and exactly
one final typed store per owned output cell. No benchmark was [run](../../../../glossary.md#term-run) or claimed.

Documentation-focused finalization context, 2026-09-09: applied the General and [Planning](../../../../glossary.md#term-planning)
documentation profiles to this planning-only change. Independently reviewed the current advanced
IR/lowering, [log-sum-exp](../../../../glossary.md#log-sum-exp) emitter and helpers, generator, preparation/finalization and executable
binding seams, reference oracle, generated inventory/evidence owners, and focused tests. This
confirms the retained selected generated forms are finite and typed for exactly [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32),
and [BFLOAT16](../../../../glossary.md#term-bfloat16) across dense/general, heap/segment/mixed, [scalar](../../../../glossary.md#term-scalar)/parallel complete-cell forms. They
use zero workspace, the two-pass max-shift plus compensated exponential sum, the existing
[singleton](../../../../glossary.md#term-singleton), empty, special-value, and [narrowing](../../../../glossary.md#term-narrowing) behavior, and one final store. The current hot
loop has no Synaptik numerical helper or reflective/object dispatch. The implementation context's
ten-class [CPU](../../../../glossary.md#term-cpu) command is reused unchanged: it passed 69 tests with zero failures, errors, or
skips, and no Java changed afterward. No [Javadoc](../../../../glossary.md#term-javadoc), glossary, architecture, public [API](../../../../glossary.md#term-application-programming-interface-api), build,
resource, executable-test, [conformance](../../../../glossary.md#term-conformance), or integration update is needed because this finalization
changes only planning status and evidence; the existing contracts remain accurate.

## Implementation notes

Decision: retain generation. `CpuLogSumExpEmitter` is the numerical [owner](../../../../glossary.md#term-owner) selected by
`CpuClassFileKernelGenerator`; `CpuAdvancedReductionIr` and
`CpuAdvancedReductionLowering.Geometry` retain the two-[buffer](../../../../glossary.md#term-buffer), zero-workspace, complete-cell
contract. The existing cold preparation/finalization and [immutable](../../../../glossary.md#term-immutable) typed invocation bind the exact
array/segment [carrier](../../../../glossary.md#term-storage-carrier) pattern. The reference [oracle](../../../../glossary.md#term-test-oracle) remains independent and generated inventory/
evidence [ownership](../../../../glossary.md#term-ownership-meanings) remains valid. A direct route would add parallel production/[binding](../../../../glossary.md#term-binding)/test paths
before retiring all of those owners, so it has higher total implementation and verification cost.

No production Java, [Javadoc](../../../../glossary.md#term-javadoc), glossary, executable tests, inventory, [schema](../../../../glossary.md#term-schema), [cache](../../../../glossary.md#term-cache), [Prepare](../../../../glossary.md#prepare),
[Runtime](../../../../glossary.md#term-runtime), [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), architecture, build, [conformance](../../../../glossary.md#term-conformance), or integration path changed. Statistics
and norms remain untouched. Direct replacement would duplicate selection, binding, invocation,
inventory, and retirement proof before removing the current generated owners. This separate
documentation-focused context finalized the [planning](../../../../glossary.md#term-planning)-only diff, no-change Javadoc/glossary
conclusion, links, status [synchronization](../../../../glossary.md#term-synchronization), and recorded Java evidence.

## Completion summary

Retained the complete current generated `LOG_SUM_EXP` route. Focused semantic, special-value,
[carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout), range/[canary](../../../../glossary.md#term-canary), [lowering](../../../../glossary.md#lowering), reference, generated-entry, inventory, and evidence checks
passed. The separate documentation-focused pass independently verified the selected generated
route, final [planning](../../../../glossary.md#term-planning) evidence, terminology, links, status [synchronization](../../../../glossary.md#term-synchronization), and no-change
conclusions. No executable change requires [Javadoc](../../../../glossary.md#term-javadoc) generation.

Status: Complete
