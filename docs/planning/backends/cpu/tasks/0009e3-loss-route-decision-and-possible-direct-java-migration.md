# Task 0009E3: [Loss](../../../../glossary.md#loss) Route Decision and Possible Direct-Java Migration

## Reader context

This task records three independent retained-generated decisions: MSE, dense categorical
cross-entropy with logits, and index categorical cross-entropy with logits. `CpuLossIr` fixes
schema-58 specialization identity; axis, extents, layouts, addresses, and ignore values remain cold
geometry. `CpuLossEmitter` owns generated numerical loops, while `CpuLossReferenceKernel` supplies
the semantic oracle. Class-File evidence checks prohibited operations and dependencies across the
792-class matrix; it is not a substitute for numerical comparison.

Here, Model owns loss meaning and Compiler owns capture, inference, and gradients. CPU Prepare
lowers and binds typed carriers; `CpuLossInputValidator` checks the complete index domain before
writes or submission. Finalization publishes immutable prepared invocation. These routes declare
zero workspace. A direct replacement would need complete family implementation and generated
artifact/cache/inventory retirement, not merely alternative loops.

Current-topic navigation (not a replacement for the historical record):

- [current MSE input and reduction roles](../../../../reference/tensor/statistics-losses.md#mean-squared-error-loss-expressions) — Explains prediction versus target, exact-shape requirements, and NONE/SUM/MEAN result boundaries.
- [current dense and index categorical targets](../../../../reference/tensor/statistics-losses.md#categorical-cross-entropy-with-logits-expressions) — Distinguishes target data, class-axis semantics, and typed ignore-index metadata.

## Status

Complete

## Goal

Choose, separately for each current static [CPU](../../../../glossary.md#term-cpu) [loss](../../../../glossary.md#loss) family, the route with demonstrably lower total implementation, verification, and retirement cost: retain bounded generated [bytecode](../../../../glossary.md#term-bytecode) or migrate its static loops and orchestration to [finite](../../../../glossary.md#term-finite) direct typed Java. This is a correctness and maintainability decision, not a performance campaign.

## [Scope](../../../../glossary.md#term-scope)

Decide the complete current first-class [loss](../../../../glossary.md#loss) [matrix](../../../../glossary.md#term-matrix-meanings):

| Family | Forms | Result and domain |
|---|---|---|
| `MEAN_SQUARED_ERROR` | nine ordered [`BFLOAT16`](../../../../glossary.md#term-bfloat16)/[`FLOAT32`](../../../../glossary.md#term-float32)/[`FLOAT64`](../../../../glossary.md#term-float64) input pairs; `NONE`, `SUM`, [`MEAN`](../../../../glossary.md#term-mean) | promoted floating result; [element](../../../../glossary.md#term-element) domain |
| `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | nine ordered floating logits/[target](../../../../glossary.md#term-target) pairs; normalized static [class axis](../../../../glossary.md#class-axis); `NONE`, `SUM`, `MEAN` | promoted floating result; non-class [sample domain](../../../../glossary.md#sample-domain) |
| `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | three floating logits types, [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype) target, absent/present same-type ignore value; normalized static class axis; `NONE`, `SUM`, `MEAN` | logits type result; non-class sample domain |

Preserve static [Shapes](../../../../glossary.md#shape); resolved non-negative [layouts](../../../../glossary.md#layout); type promotion/result [narrowing](../../../../glossary.md#term-narrowing); legal ordered role-to-unique-boundary [aliasing](../../../../glossary.md#term-aliasing) for equal [MSE](../../../../glossary.md#mean-squared-error--mse)/dense input roles; typed [heap](../../../../glossary.md#term-heap) arrays, native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment)s, and ordered mixed carriers; dense-[linear](../../../../glossary.md#term-linear-meanings)/general-odometer access; checked span/address/range arithmetic; [injective](../../../../glossary.md#term-injective) output writes; input-only aliasing; and output/input or output/output overlap rejection.

Preserve increasing element MSE subtraction/square; categorical increasing [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)/class stable classification, shifted exponential, [log-sum-exp](../../../../glossary.md#log-sum-exp), and contribution passes; dense zero-target exclusion before non-[finite](../../../../glossary.md#term-finite) multiplication; index ignore-before-bounds-or-logits-read; left-associated `SUM`/`MEAN` accumulation and one final typed division. Preserve [binary32](../../../../glossary.md#floating-point-precision-and-rounding)/[binary64](../../../../glossary.md#floating-point-precision-and-rounding) boundaries, special values, empty `NONE`, empty `SUM` [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), empty `MEAN` [NaN](../../../../glossary.md#term-nan), `C == 0`, all-ignored index domains, and index non-ignored-count mean.

Preserve independent [scalar](../../../../glossary.md#term-scalar)/caller-parallel `NONE` ranges; one complete ordered scalar range for `SUM`/`MEAN`; zero workspace; [immutable](../../../../glossary.md#term-immutable) prepared invocation; cold complete index [validation](../../../../glossary.md#term-validation) before mutation/submission; and current [fail-closed](../../../../glossary.md#term-fail-closed) [admission](../../../../glossary.md#term-admission), [lowering](../../../../glossary.md#lowering), [binding](../../../../glossary.md#term-binding), preparation, finalization, [cache](../../../../glossary.md#term-cache)/[artifact](../../../../glossary.md#term-artifact), [publication](../../../../glossary.md#term-publication), and no-write behavior. Do not add fallback, [decomposition](../../../../glossary.md#term-decomposition) recognition, interpreter, [materialization](../../../../glossary.md#materialization), or dual maintained routes. A direct decision covers a family’s complete admitted matrix, not a fast subset.

## Out of [scope](../../../../glossary.md#term-scope)

New [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/public APIs, [loss](../../../../glossary.md#loss) semantics or [gradients](../../../../glossary.md#term-gradient), new loss kinds, dynamic [shapes](../../../../glossary.md#shape), probability-input loss, label smoothing, [weights](../../../../glossary.md#term-weight), [masks](../../../../glossary.md#term-boolean-mask), [broadcasting](../../../../glossary.md#broadcasting)/casting, external [fusion](../../../../glossary.md#term-fusion), [vector](../../../../glossary.md#term-vector)/native/[OpenBLAS](../../../../glossary.md#term-openblas) routes, reduction trees/partial-combine workspace, [autotuning](../../../../glossary.md#autotuning--model-autotuning), build/[dependency](../../../../glossary.md#term-dependency)/[module](../../../../glossary.md#term-module)/architecture changes, [conformance](../../../../glossary.md#term-conformance)/integration work, [CPU](../../../../glossary.md#term-cpu) 0009F or later task specifications, and a benchmark rerun or performance/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit)/speed claim.

## Architecture references

Read [ARCHITECTURE.md](../../../../../ARCHITECTURE.md), [current architecture plan](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [documentation rules](../../../../developer-guide/documentation-rules.md), [CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [CPU 0008I](0008i-portable-loss-family-execution.md), completed E1 through E2, current [loss](../../../../glossary.md#loss) [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) contracts, and directly affected [CPU](../../../../glossary.md#term-cpu) source, tests, Javadocs, [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache) inventory, and evidence.

## Architecture constraints

`ARCHITECTURE.md` is authoritative. [Model](../../../../glossary.md#term-model) owns [loss](../../../../glossary.md#loss) meaning and [Compiler](../../../../glossary.md#term-compiler) owns [capture](../../../../glossary.md#term-capture-meanings), [inference](../../../../glossary.md#term-inference), and [gradients](../../../../glossary.md#term-gradient). [CPU](../../../../glossary.md#term-cpu) [Prepare](../../../../glossary.md#prepare) owns cold [lowering](../../../../glossary.md#lowering), route selection, exact resource declaration, and post-slot finalization; [Runtime](../../../../glossary.md#term-runtime) invokes [immutable](../../../../glossary.md#term-immutable) prepared work only. Preserve [atomic](../../../../glossary.md#resource-transaction-and-rollback) prepared-region/[publication](../../../../glossary.md#term-publication) boundaries, [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) carriers, isolated [`RunState`](../../../../glossary.md#run-state--runstate), and [validation](../../../../glossary.md#term-validation) before mutation/submission. No shared/public contract, generic loss executor, resource kind, [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), architecture, or build change is authorized; stop and report any need for one.

Generated code that remains must retain an optimal clean-Java [oracle](../../../../glossary.md#term-test-oracle) with matching semantic algorithm, hot-loop/[dataflow](../../../../glossary.md#term-dataflow) [shape](../../../../glossary.md#shape), and avoidable-overhead profile as `AGENTS.md` requires.

## Package impact

Existing [CPU](../../../../glossary.md#term-cpu)-private packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.ir`
- `io.github.pho001.synaptik.backend.cpu.internal.lowering`
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`
- `io.github.pho001.synaptik.backend.cpu.internal.executable`
- `io.github.pho001.synaptik.backend.cpu.internal.cache`
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable`
- `io.github.pho001.synaptik.backend.cpu.internal.reference`

Packages added or changed: None.

Type placement:

- Retained generation adds no production type.
- If direct Java wins, add only a narrowly named family-private executable in `internal.executable` (for example `CpuMseDirectExecutable` or `CpuCategoricalCrossEntropyDirectExecutable`) owning [finite](../../../../glossary.md#term-finite) typed cold-selected entries and [immutable](../../../../glossary.md#term-immutable) range invocation; never a generic dispatcher or service.

## Affected files

Expected [planning](../../../../glossary.md#term-planning) paths are this task, [CPU](../../../../glossary.md#term-cpu) master plan, parent CPU 0009, and roadmap. Review-only seams: `CpuCapabilityProvider`; [loss](../../../../glossary.md#loss) [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering); `CpuLossEmitter` and class-file generation; `CpuLossInputValidator`; preparation/finalization/[prepared executable](../../../../glossary.md#prepared-executable--preparedexecutable); portable route; generated [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache)/[schema](../../../../glossary.md#term-schema); clean-Java [oracle](../../../../glossary.md#term-test-oracle); generated coverage inventory/evidence; and focused capability, semantic, rejection, structural, reference, and invocation tests.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 34 paths: 17 production/[Javadoc](../../../../glossary.md#term-javadoc), 13 test/resource, and four [planning](../../../../glossary.md#term-planning) paths; at most three narrowly named [CPU](../../../../glossary.md#term-cpu)-private direct executable types, one per family. A 35th path, fourth type, shared/public/resource/build/architecture/[conformance](../../../../glossary.md#term-conformance)/integration change, generic [element](../../../../glossary.md#term-element) [dispatch](../../../../glossary.md#term-dispatch), partial/combine state, or incomplete family replacement is a stop-and-replan condition.

## Acceptance criteria

- Record an auditable non-performance route decision for [MSE](../../../../glossary.md#mean-squared-error--mse), dense categorical, and index categorical [loss](../../../../glossary.md#loss), covering each complete [matrix](../../../../glossary.md#term-matrix-meanings) and every current [admission](../../../../glossary.md#term-admission)/rejection fact.
- Preserve all scoped numerical, [layout](../../../../glossary.md#layout), [carrier](../../../../glossary.md#term-storage-carrier), range, workspace, [validation](../../../../glossary.md#term-validation), [immutable](../../../../glossary.md#term-immutable)-invocation, preparation/finalization/[publication](../../../../glossary.md#term-publication), and cold no-write semantics.
- Select direct Java only when [finite](../../../../glossary.md#term-finite) typed implementation plus complete generated-path retirement is demonstrably strictly less costly than retention. Equality, uncertainty, incomplete evidence, or an unsupported candidate retains generation.
- A selected direct family has cold-selected typed carrier/layout paths without per-[element](../../../../glossary.md#term-element) type/carrier/layout/kind [dispatch](../../../../glossary.md#term-dispatch), allocation, boxing, reflection, map/string [lookup](../../../../glossary.md#term-lookup), or avoidable virtual semantic dispatch. Fully retire its emitter/generator selection, generated [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache)/[schema](../../../../glossary.md#term-schema) projection, invocation bridge, inventory, and generated-only evidence; no fallback or parallel implementation remains.
- A retained family records why emitter/Class-File review, typed [binding](../../../../glossary.md#term-binding), validator, cache/inventory, invocation, [oracle](../../../../glossary.md#term-test-oracle), and focused-evidence costs make replacement larger. Its Class-File/hot-loop review proves no hidden helper, allocation, boxing, reflection, generic/map/string, or virtual semantic dispatch; literal-[bytecode](../../../../glossary.md#term-bytecode), [JIT](../../../../glossary.md#term-just-in-time-compilation-jit), and performance assertions are unnecessary.
- Focused tests cover forms/types/reductions; dense/general layouts; [heap](../../../../glossary.md#term-heap)/segment/mixed carriers; legal [aliases](../../../../glossary.md#aliasing-and-arena); empty/special/[narrowing](../../../../glossary.md#term-narrowing) cases; ranges; canaries/no-write failure; index prevalidation; [lowering](../../../../glossary.md#lowering)/[prepare](../../../../glossary.md#prepare)/finalization/invocation; artifact/cache; oracle; and selected-route retirement or retention inventory.
- A separate clean documentation-focused context finalizes affected Javadocs, explanatory/[planning](../../../../glossary.md#term-planning) status, glossary impact, links, and documentation validation in the same overall change.

## Route-selection decision rule

First establish that both candidates preserve the full family boundary. For each family record (1) retained-generation emitter/Class-File review, cold typed [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout) [binding](../../../../glossary.md#term-binding), [validation](../../../../glossary.md#term-validation), [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache)/inventory, invocation, [oracle](../../../../glossary.md#term-test-oracle), and focused-evidence costs; (2) [finite](../../../../glossary.md#term-finite) direct-Java implementation and cold selection/binding costs; and (3) complete generated-route retirement plus semantic, rejection, invocation, cache, inventory, oracle, and hot-loop verification costs. Select direct Java only when (2) plus (3) is demonstrably strictly lower than (1). This supplies no [metric](../../../../glossary.md#term-metric), benchmark, or performance conclusion.

## Tests / [validation](../../../../glossary.md#term-validation)

After executable stabilization, [run](../../../../glossary.md#term-run) once the focused [loss](../../../../glossary.md#loss)-[owner](../../../../glossary.md#term-owner) [matrix](../../../../glossary.md#term-matrix-meanings): [CPU](../../../../glossary.md#term-cpu) capability/package inventory; loss [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering); `CpuLossInputValidator`; generated or direct loss semantic closure, evidence, coverage inventory, and [artifact](../../../../glossary.md#term-artifact) [shape](../../../../glossary.md#shape); `CpuLossReferenceKernel`; affected preparer, finalizer, prepared-executable, artifact-[cache](../../../../glossary.md#term-cache), and [persistence](../../../../glossary.md#term-persistence)-evidence tests. Add automated retirement checks before deleting a selected generated path. Record exact [Gradle](../../../../glossary.md#term-gradle) commands and XML counts. Do not run benchmarks or repository-wide validation.

The documentation context reuses stable Java evidence unless it changes executable Java; it checks links, anchors, fences, terminology/glossary, exact path [scope](../../../../glossary.md#term-scope), dependencies/status/frontier, then runs:

```bash
git diff --check
git status --short -uall
```

Architecture, [conformance](../../../../glossary.md#term-conformance), integration, and repository-wide validation defer to CPU 0009G/[CI](../../../../glossary.md#term-continuous-integration-ci) because this task must not change shared or end-to-end boundaries.

Historical performance is non-gating: CPU 0009 loss [fork](../../../../glossary.md#term-fork-meanings) 0 completed 792 rows with 19 ratios above 1.15 and is `NON_PASSING` (maximum `1.3861164205039096` at row 536, `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS-BFLOAT16-INT32-MEAN-false-roles0_1-0`). Forks 1--4 were user-closed without rerun. Preserve these facts; neither is a pass, and E3 does not rerun them.

## Dependencies

- E1 through E2 are Complete retained-generated route decisions.
- [CPU](../../../../glossary.md#term-cpu) 0008I supplies the current [loss](../../../../glossary.md#loss) semantics, [lowering](../../../../glossary.md#lowering), validator, 792-class inventory, generated route, clean-Java [oracle](../../../../glossary.md#term-test-oracle), structural evidence, and historical performance facts.

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0009F remains Draft and follows E3. Do not create or detail it here.
- CPU 0009G remains the later support/correctness/hygiene [checkpoint](../../../../glossary.md#term-checkpoint).

## Architecture impact

Expected impact: None. Stop and report any public, [dependency](../../../../glossary.md#term-dependency), resource, [module](../../../../glossary.md#term-module), or architecture conflict.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009E3. Do not commit, push, stage, reset, revert, delete, or modify unrelated work; do not use GSD. Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, Planning Guide, CPU master plan, CPU 0009, CPU 0008I, completed E1 through E2, this task, current loss source/tests/evidence, documentation rules, and General/Planning profiles. Implement exactly this per-family route decision. Choose direct finite typed Java only when complete implementation, verification, and retirement is strictly cheaper; otherwise retain generation with concrete source/Class-File/hot-loop evidence. Preserve the confirmed boundary and historical NON_PASSING fork-0/user-closed-forks facts. Stop on conflict. Hand the stabilized diff and exact test evidence to a separate clean documentation context. No benchmarks/repository-wide tests/commit/push/stage.
```

## Local decisions

All three families retain bounded generated execution. This is three independent decisions, not
one family-wide shortcut:

| Family | Decision | Non-performance cost result |
|---|---|---|
| `MEAN_SQUARED_ERROR` | Retain generated | The existing emitter owns the complete nine ordered floating-pair [matrix](../../../../glossary.md#term-matrix-meanings), three reductions, legal equal-role [alias](../../../../glossary.md#aliasing-and-arena) mapping, dense-int and general-affine bodies, three [carrier](../../../../glossary.md#term-storage-carrier) modes, promoted [binary32](../../../../glossary.md#floating-point-precision-and-rounding)/[binary64](../../../../glossary.md#floating-point-precision-and-rounding) arithmetic, and cold range [binding](../../../../glossary.md#term-binding) in one [schema](../../../../glossary.md#term-schema)-58 [artifact](../../../../glossary.md#term-artifact). A [finite](../../../../glossary.md#term-finite) direct replacement would have to reproduce those typed carrier/[layout](../../../../glossary.md#layout) entries and add a new cold selection/binding route, then retire the generated emitter selection, helper [lookup](../../../../glossary.md#term-lookup), artifact/[cache](../../../../glossary.md#term-cache) projection, 252-row inventory portion, and generated-only structural evidence. The replacement and retirement proof is strictly larger than reviewing the existing generated body and its retained [oracle](../../../../glossary.md#term-test-oracle)/evidence. |
| `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | Retain generated | The emitter already specializes all nine ordered floating pairs, `NONE`/`SUM`/[`MEAN`](../../../../glossary.md#term-mean), legal aliases, carrier combinations, and both [contiguous](../../../../glossary.md#term-contiguous) and general geometry while retaining the three-pass stable classification/[log-sum-exp](../../../../glossary.md#log-sum-exp)/contribution traversal, zero-[target](../../../../glossary.md#term-target) non-finite exclusion, and one final reduction division. A direct [owner](../../../../glossary.md#term-owner) would duplicate the highest-complexity typed carrier/layout matrix plus exact float/double/[BFLOAT16](../../../../glossary.md#term-bfloat16) [narrowing](../../../../glossary.md#term-narrowing) and the dense loop-carried denominator [shape](../../../../glossary.md#shape). It would also require complete generated-route/cache/inventory/evidence retirement for 252 current structural forms. That work is demonstrably larger than the existing emitter/Class-File/oracle review. |
| `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | Retain generated | The current route includes three logits representations, two exact index widths, absent/present ignore forms, three reductions, carrier combinations, contiguous/general geometry, cold complete pre-write index [validation](../../../../glossary.md#term-validation), `C == 0`/all-ignored handling, and non-ignored-count mean. A finite direct implementation would need the same complete typed target/carrier binding and preserve validation-before-submission while replacing the generated helper bridge and retiring 288 inventory forms, cache/artifact projection, and generated evidence. It is strictly more implementation and verification work than retention. |

The common generated boundary is already intentionally bounded: `CpuLossIr` freezes only
family/type/reduction/ignore/role/carrier/range facts in schema-58 [identity](../../../../glossary.md#term-identity-meanings), while [axis](../../../../glossary.md#term-axis), [rank](../../../../glossary.md#term-rank),
[extent](../../../../glossary.md#term-extent), layouts, addresses, ignore value, and range remain cold geometry. `CpuLossLowering`,
preparation, finalization, artifact cache, and [immutable](../../../../glossary.md#term-immutable) executable binding use that same
identity. Replacing only the hot loops would leave a prohibited dual maintained route; replacing
the complete route would add the direct route's selection and typed invocation work before its
generated-route retirement proof. Neither candidate is strictly cheaper.

## Known limitations

No E3 result may claim speed, generated/direct parity, [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) behavior, or whole-[backend](../../../../glossary.md#backend) performance closure. The prior [fork](../../../../glossary.md#term-fork-meanings)-0 failure and user-closed forks remain historical evidence only.

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context inspected the complete [loss](../../../../glossary.md#loss) boundary: `CpuLossIr`,
`CpuLossLowering`, `CpuLossEmitter`, `CpuLossInputValidator`, `CpuLossReferenceKernel`,
`CpuPartitionPreparer`, `CpuPartitionFinalizer`, `CpuPreparedExecutable`,
`CpuGeneratedKernel`, `CpuClassFileKernelGenerator`, the portable route and generated-[artifact](../../../../glossary.md#term-artifact)
[cache](../../../../glossary.md#term-cache) seams, plus their focused capability, semantic, structural, [lifecycle](../../../../glossary.md#lifecycle), cache, and
[persistence](../../../../glossary.md#term-persistence)-evidence tests.

The retained generated route has a [schema](../../../../glossary.md#term-schema)-58, 792-class structural [matrix](../../../../glossary.md#term-matrix-meanings): [MSE](../../../../glossary.md#mean-squared-error--mse) 252, dense
categorical 252, and index categorical 288. The actual `CpuLossEvidenceTest` Class-File
inspection checks the loss artifact's private direct helpers, primitive array loads/stores and
branches, stable categorical `StrictMath` calls, and cold-geometry byte stability. Its complete
evidence protocol rejects generated allocation, boxing, `invokedynamic`, monitors, collections,
reflection, method handles, strings, Synaptik helper/fallback calls, [graph](../../../../glossary.md#graph)/[layout](../../../../glossary.md#layout)/cache/route/
resource/worker [lookups](../../../../glossary.md#term-method-handle-lookup), and generic semantic [dispatch](../../../../glossary.md#term-dispatch) from all 792 generated classes. The
retained clean-Java [oracle](../../../../glossary.md#term-test-oracle) has the same increasing MSE traversal and categorical
classification/shifted-exponential/[log-sum-exp](../../../../glossary.md#log-sum-exp)/contribution passes, arithmetic widths, [narrowing](../../../../glossary.md#term-narrowing),
stores, and reduction [ownership](../../../../glossary.md#term-ownership-meanings). Manual source review found no deviation from that oracle in the
emitter's selected dense-int/general-affine bodies. This is a structural and semantic conclusion,
not a literal-[bytecode](../../../../glossary.md#term-bytecode), [JIT](../../../../glossary.md#term-just-in-time-compilation-jit), generated/direct parity, or speed claim.

After stabilization, this context ran the focused loss-[owner](../../../../glossary.md#term-owner) suite once on 2026-09-10:

```bash
./gradlew :backends:cpu:test --rerun-tasks \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuLossGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuLossSemanticClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuLossEvidenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageCheckpointTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageEvidenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedDirectEvidenceClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernelShapeTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedStructuralOracleCatalogTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuLossInputValidatorTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelPersistenceEvidenceTest
```

The 16 XML reports record 193 tests, zero failures, zero errors, and two expected opt-in skips:
the unarmed 792-class structural-artifact writer and the unconfigured persistent-artifact
evidence protocol. All ordinary selected tests passed. They cover capability/package inventory,
loss [admission](../../../../glossary.md#term-admission) and [lowering](../../../../glossary.md#lowering) through production preparation, forms/types/reductions, dense/general
layouts, [heap](../../../../glossary.md#term-heap)/segment/mixed carriers, legal [aliases](../../../../glossary.md#aliasing-and-arena), empty/special/narrowing cases, independent
ranges and [scalar](../../../../glossary.md#term-scalar) reductions, canaries/no-write failure behavior, complete index prevalidation,
finalization, [immutable](../../../../glossary.md#term-immutable) invocation, artifact/cache [shape](../../../../glossary.md#shape), generated coverage, and the frozen
reference/structural-oracle seam. No benchmark, old five-[fork](../../../../glossary.md#term-fork-meanings) [run](../../../../glossary.md#term-run), repository-wide suite,
architecture, [conformance](../../../../glossary.md#term-conformance), or integration suite ran.

Fork 0 remains historical `NON_PASSING`: 19 of 792 ratios exceeded 1.15, with maximum
`1.3861164205039096` at row 536
`INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS-BFLOAT16-INT32-MEAN-false-roles0_1-0`.
Forks 1--4 remain user-closed without rerun. Neither fact gates this route decision or is
promoted to a performance claim.

## Implementation notes

No production, test, [Javadoc](../../../../glossary.md#term-javadoc), generated [artifact](../../../../glossary.md#term-artifact), [cache](../../../../glossary.md#term-cache), inventory, or architecture path changed:
all three complete family [matrices](../../../../glossary.md#term-matrix-meanings) retain their existing bounded generated route. This independent
documentation review finalized the [planning](../../../../glossary.md#term-planning) record and synchronized its parent, master-plan, and
roadmap status. It found no Javadoc or explanatory-document change necessary because no [API](../../../../glossary.md#term-application-programming-interface-api),
behavior, terminology, architecture boundary, or build contract changed; the glossary likewise
needs no update. Architecture, build, architecture-test, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration
paths remain unchanged because this decision neither changes a [dependency](../../../../glossary.md#term-dependency) boundary nor observable
end-to-end behavior.

## Completion summary

Completed changes: separately retained bounded generated execution for [MSE](../../../../glossary.md#mean-squared-error--mse), dense categorical,
and index categorical [losses](../../../../glossary.md#loss) after full-[matrix](../../../../glossary.md#term-matrix-meanings) non-performance cost decisions; no partial
migration or dual route was introduced. This documentation-focused review corrected the
implementation-context label, finalized this task, and synchronized the parent, [CPU](../../../../glossary.md#term-cpu) master plan,
and roadmap.

Paths changed: this task; `docs/planning/backends/cpu/master-plan.md`;
`docs/planning/backends/cpu/tasks/0009-portable-generated-coverage-closure-checkpoint.md`; and
`docs/planning/roadmap.md`.

[Validation](../../../../glossary.md#term-validation): independently inspected the current loss [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/emitter/validator/reference,
preparation/finalization/invocation, generated-[artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache), inventory, focused tests, and XML
reports. Reused the recorded focused 16-report result of 193 tests, zero failures/errors, and two
expected opt-in skips; no Java suite, benchmark, or [fork](../../../../glossary.md#term-fork-meanings) was rerun because no executable Java
changed. Checked [planning](../../../../glossary.md#term-planning) links, headings, fences, terminology/glossary impact, path [scope](../../../../glossary.md#term-scope),
status/frontier consistency, final newlines, `git diff --check`, and `git status --short -uall`.
Documentation-focused context: mandatory clean-context E3 documentation review.

No-change documentation/[Javadoc](../../../../glossary.md#term-javadoc) conclusions: no production or test Javadoc, explanatory
documentation, glossary, architecture, build, architecture-test, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), or
integration update is warranted: the task records a CPU-private route decision only and changes
none of their contracts.

Unresolved issues/follow-up: the historical loss performance evidence remains `NON_PASSING`;
fork 0 has 19 of 792 ratios above 1.15, with maximum `1.3861164205039096` at row 536
`INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS-BFLOAT16-INT32-MEAN-false-roles0_1-0`; forks 1--4
remain user-closed without rerun. These facts are not a performance claim or a route-decision
gate. CPU 0009 remains [Ready](../../../../glossary.md#term-ready) and incomplete. CPU 0009F is the sole next Draft summary frontier;
it is not detailed here.

Status: Complete
