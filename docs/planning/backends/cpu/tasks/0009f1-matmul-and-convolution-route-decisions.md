# Task 0009F1: [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) and [Convolution](../../../../glossary.md#term-channels-first-convolution-layers) Route Decisions

## Status

Complete

## Goal

Make four independent, complete [CPU](../../../../glossary.md#term-cpu) route decisions from current source and tests: [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), the
visible [Conv1d composition](../../../../glossary.md#conv1d-composition), direct [Conv2d](../../../../glossary.md#convolution--conv2d), and direct [Conv3d](../../../../glossary.md#conv3d--ncdhw). Retain dense specialized generated
compute unless a complete [finite](../../../../glossary.md#term-finite) direct-Java replacement is demonstrably strictly cheaper across
implementation, verification, and retirement. Finite direct Java is permitted only for a simpler
surrounding composition/orchestration boundary when strictly cheaper. Equality or uncertainty
retains generation. This task creates no performance gate or benchmark campaign.

## [Scope](../../../../glossary.md#term-scope)

Audit and record one source- and test-backed decision for each separate row below. No row borrows
another row's decision, reference implementation, or verifier.

| Route | Current capability and semantic boundary | Current selected machinery | Complete decision boundary |
| --- | --- | --- | --- |
| [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) | Static resolved numeric [vector](../../../../glossary.md#term-vector)/[matrix](../../../../glossary.md#term-matrix-meanings)/right-broadcast-batched MATMUL; non-negative [layouts](../../../../glossary.md#layout), [injective](../../../../glossary.md#term-injective) output, exact promotion, [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) unit M/N [normalization](../../../../glossary.md#term-normalization), right [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) broadcast, and legal empty zero work. Exact recognized suffix: optional rank-one result-typed ADD in either order plus at most [ReLU](../../../../glossary.md#term-relu), [sigmoid](../../../../glossary.md#term-sigmoid), [tanh](../../../../glossary.md#term-tanh), exact [GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu), tanh-approximate GELU, [SiLU](../../../../glossary.md#sigmoid-linear-unit--silu), or typed clamp. | `CpuMatmulLowering` makes full-K geometry/access plans; `CpuMatmulCandidateSelector` chooses direct [scalar](../../../../glossary.md#term-scalar), N-vector, 2x2 scalar, or 2x2 N-vector; `CpuMatmulEmitter` emits typed bodies; `CpuMatmulReferenceKernel` is the clean Java [oracle](../../../../glossary.md#term-test-oracle). Output work units own [half-open](../../../../glossary.md#term-half-open-interval) ranges; scalar/vector and caller parallelism bind cold. | Four realizations, types/carriers, dense/general layouts, batches/broadcasts, epilogues, ranges/parallelism, empty/special behavior, [aliases](../../../../glossary.md#aliasing-and-arena)/overlap, cold [binding](../../../../glossary.md#term-binding), [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache)/invocation/inventory. No generic [epilogue](../../../../glossary.md#term-epilogue) framework. |
| Conv1d visible composition | Only exact [NCW](../../../../glossary.md#term-ncw) `EXPAND_DIMS(axis 2) -> CONV2D -> SQUEEZE(axis 2)`. [Singleton](../../../../glossary.md#term-singleton) height stays virtual. [Conv2d](../../../../glossary.md#convolution--conv2d) supplies its groups/depthwise, batch, intrinsic [bias](../../../../glossary.md#term-bias), geometry, promotion, layouts, carriers, [validation](../../../../glossary.md#term-validation), empty behavior, overlap checks, ranges, and zero workspace. It is not a direct Conv1d [operation](../../../../glossary.md#operation), capability, artifact family, or generated body. | `CpuConv1dCompositionLowering` validates/delegates; no Conv1d [IR](../../../../glossary.md#term-intermediate-representation-ir), emitter, [cache key](../../../../glossary.md#term-cache-key), artifact, or inventory row exists. | Composition/orchestration only. A direct surrounding change must preserve the visible contract and be strictly cheaper; it must not create a direct Conv1d [kernel](../../../../glossary.md#kernel) or common abstraction. |
| Conv2d direct body | Static grouped [NCHW](../../../../glossary.md#term-nchw) floating cross-correlation: [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16) inputs, [weights](../../../../glossary.md#term-weight), optional intrinsic rank-one bias, ordered promotion, batch/groups/depthwise, positive [stride](../../../../glossary.md#term-stride)/[dilation](../../../../glossary.md#term-dilation), non-negative [padding](../../../../glossary.md#padding), arrays/native-order segments/ordered mixed carriers, checked scalar/vector half-open output ranges, border/tail/empty/special scalar semantics, aliases/overlap, and zero workspace/[materialization](../../../../glossary.md#materialization). External suffix is none, same-type right-broadcast ADD, or ADD-RELU; others [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). | `CpuConv2dLowering`, `CpuConv2dIr`, `CpuConv2dEmitter`, `CpuConv2dReferenceKernel`; [schema](../../../../glossary.md#term-schema)-63 vector/parallel-vector only for eligible dense FLOAT32/FLOAT64 width blocks, scalar otherwise. | Complete scalar/vector generated family plus suffix, geometry/binding, artifact [identity](../../../../glossary.md#term-identity-meanings)/cache, invocation, inventory, validation and retirement; no [Conv3d](../../../../glossary.md#conv3d--ncdhw) merger or Conv1d fallback. |
| Conv3d direct body | Static grouped [NCDHW](../../../../glossary.md#conv3d--ncdhw) floating cross-correlation with the same types/promotion/bias/carriers, batch/groups/depthwise, padding/stride/dilation, zero workspace/materialization, checked empty/special/alias/overlap behavior, and half-open output ranges. It has no external Conv3d epilogue; general DAGs fail closed. | `CpuConv3dLowering`, `CpuConv3dIr`, `CpuConv3dEmitter`, `CpuConv3dReferenceKernel`; schema-63 vector/parallel-vector only for eligible dense FLOAT32/FLOAT64 width blocks, scalar otherwise. | Separate complete direct body, including no-epilogue boundary, geometry/binding, artifact/cache, invocation, inventory, validation and retirement. |

All rows keep [lowering](../../../../glossary.md#lowering)/selection cold; `CpuPartitionPreparer` declares exact resources,
`CpuPartitionFinalizer` validates assignments then realizes selected artifacts, and
`CpuPreparedExecutable` invokes only prepared work with [run](../../../../glossary.md#term-run)-owned range/worker state. [Runtime](../../../../glossary.md#term-runtime) does
not select a route.

## Out of [scope](../../../../glossary.md#term-scope)

- Common [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[convolution](../../../../glossary.md#term-channels-first-convolution-layers) abstraction, universal verifier/[oracle](../../../../glossary.md#term-test-oracle)/inventory, or direct Conv1d [operation](../../../../glossary.md#operation).
- Benchmark, five-[fork](../../../../glossary.md#term-fork-meanings) campaign, threshold, speed/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit) claim, packing/[materialization](../../../../glossary.md#materialization)/native route.
- Model, Compiler, Training, Prepare, Runtime, public API, capability, workspace, schema,
  dependency, architecture, or Gradle change.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [schema glossary](../../../../glossary.md#term-schema); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- 0009F2 [pooling](../../../../glossary.md#term-pooling), 0009F3 attention/[batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training), and [CPU](../../../../glossary.md#term-cpu) 0009G's broad [checkpoint](../../../../glossary.md#term-checkpoint).

## Architecture references and constraints

`ARCHITECTURE.md` is authoritative. [CPU](../../../../glossary.md#term-cpu) analysis/[prepare](../../../../glossary.md#prepare) owns [admission](../../../../glossary.md#term-admission), [lowering](../../../../glossary.md#lowering), selection,
[specialization](../../../../glossary.md#term-specialization), and exact resource declaration; finalization follows shared assignment and creates
[immutable](../../../../glossary.md#term-immutable) prepared work; [Runtime](../../../../glossary.md#term-runtime) only invokes it. Keep [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout)/geometry/route/worker facts
cold. Hot loops add no allocation, boxing, reflection, string/map [dispatch](../../../../glossary.md#term-dispatch), [synchronization](../../../../glossary.md#term-synchronization), [graph](../../../../glossary.md#graph)
interpretation, or avoidable virtual dispatch; parallelism is outside them. Preserve checked
addresses, [aliases](../../../../glossary.md#aliasing-and-arena)/failure-before-write, reuse, and [run](../../../../glossary.md#term-run) isolation. Stop on an architecture conflict.

## Package impact

No package is added or moved. Existing [`lowering`](../../../../glossary.md#lowering), `ir`, `codegen.emit`, `reference`, [`prepare`](../../../../glossary.md#prepare),
`executable`, and [`cache`](../../../../glossary.md#term-cache) [owners](../../../../glossary.md#term-owner) remain separate. Tests remain in their matching [CPU](../../../../glossary.md#term-cpu) packages. A
selected surrounding Conv1d change stays in its composition [owner](../../../../glossary.md#term-owner) and cannot create an umbrella
route package/type.

## Affected files and maximum [scope](../../../../glossary.md#term-scope)

[Planning](../../../../glossary.md#term-planning) files changed by this task are exactly:

- `docs/planning/backends/cpu/tasks/0009f1-matmul-and-convolution-route-decisions.md`
- `docs/planning/backends/cpu/tasks/0009f-hybrid-route-decisions-for-specialized-compute-families.md`
- `docs/planning/backends/cpu/tasks/0009-portable-generated-coverage-closure-checkpoint.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

If all rows retain current routes and evidence is sound, only those five planning paths change. A
strictly cheaper direct surrounding-composition migration may change at most 13 additional [CPU](../../../../glossary.md#term-cpu)
source/test/evidence/[Javadoc](../../../../glossary.md#term-javadoc) paths (18 total) and must retire the complete selected subfamily. A
dense replacement exceeding that limit is not cheaper; retain it rather than staging dual routes.

## Acceptance criteria

1. Records separate complete decisions for MATMUL, Conv1d composition, direct Conv2d, and direct
   Conv3d, including types, layouts, carriers, groups/batches/broadcasts, ranges/parallelism,
   epilogues/composition, workspace, validation, empty/special behavior, aliases/overlap, cold
   binding, artifact/cache/inventory.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Conv1d composition glossary](../../../../glossary.md#conv1d-composition); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [layouts glossary](../../../../glossary.md#layout); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [validation glossary](../../../../glossary.md#term-validation); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [binding glossary](../../../../glossary.md#term-binding); [artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache))</span>
2. Retains dense generated compute when direct replacement costs more, is equal, or uncertain;
   selects finite direct Java only for a strictly cheaper complete surrounding replacement. No
   performance result is a gate.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>
3. For each retained generated hot path, verifies a family-specific clean optimal Java oracle with
   equivalent semantic algorithm, loop/dataflow shape, stores, and avoidable-overhead profile, plus
   proportionate Class-File/decompilation inspection for hidden helpers, allocation, boxing,
   reflection, dispatch, and route leakage. This is not a universal verifier.<span data-glossary-navigation="true"> ([hot path glossary](../../../../glossary.md#term-hot-path); [oracle glossary](../../../../glossary.md#term-test-oracle); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape); [decompilation glossary](../../../../glossary.md#term-decompilation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
4. A migration retires the complete selected subfamily: lowering/selection, binding, invocation,
   artifact/cache/schema/inventory/evidence and call sites. No selected fallback or dual
   maintenance remains; otherwise retain the row.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [schema glossary](../../../../glossary.md#term-schema))</span>
5. Run focused CPU validation once after executable stabilization. Retained-only decisions reuse
   current focused evidence after independent inspection and do not rerun Java. No broad,
   architecture, conformance, integration, benchmark, or five-fork run absent shared-boundary work.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
6. Require a clean implementation context followed by a distinct clean documentation-focused
   context; the latter finalizes affected documentation/Javadoc/glossary impact and planning
   validation without duplicating stable executable tests.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

For a changed route, [run](../../../../glossary.md#term-run) one focused command after stabilization covering the relevant
`CpuMatmulLoweringTest`, `CpuMatmulCandidateSelectorTest`, `CpuMatmulGeneratedKernelTest`,
`CpuMatmulSemanticClosureTest`, `CpuMatmulReferenceTest`, `CpuConv1dCompositionLoweringTest`,
`CpuConv2dLoweringTest`, `CpuConv3dLoweringTest`, `CpuConv2dGeneratedKernelTest`,
`CpuConv3dGeneratedKernelTest`, `CpuConvolutionSemanticClosureTest`, matching evidence tests, and
affected preparation/finalization/executable/[cache](../../../../glossary.md#term-cache) tests. Include selected-route/retirement,
[aliases](../../../../glossary.md#aliasing-and-arena)/canaries/failure-before-write, and inventory evidence as applicable. Retention-only work
inspects current sources, tests, clean Java references, Class-File/[decompilation](../../../../glossary.md#term-decompilation), [specialization](../../../../glossary.md#term-specialization),
[artifact](../../../../glossary.md#term-artifact) load/generate/[persistence](../../../../glossary.md#term-persistence), and `generated-coverage-inventory.tsv` without a duplicate
[Gradle](../../../../glossary.md#term-gradle) run. [Conv2d](../../../../glossary.md#convolution--conv2d)/[Conv3d](../../../../glossary.md#conv3d--ncdhw) include [schema](../../../../glossary.md#term-schema)-63/0008N1 evidence; [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) includes four-realization and
bounded-[epilogue](../../../../glossary.md#term-epilogue) evidence; Conv1d proves no independent artifact/inventory route.

The documentation context checks links, headings/fences, terminology/glossary, statuses,
dependencies/frontier, exact [scope](../../../../glossary.md#term-scope), final newlines, and new-file whitespace, then runs:

```bash
git diff --check
git status --short -uall
```

## Dependencies and follow-up tasks

- Depends on [Ready](../../../../glossary.md#term-ready) parent 0009F and completed [CPU](../../../../glossary.md#term-cpu) 0008F, 0008, 0008A, and 0008N1.
- 0009F2 remains Draft and depends on F1; F3 remains Draft and depends on F2; 0009G remains Draft
  and follows F3.
- A strictly cheaper complete migration that exceeds scope becomes one ordered Draft follow-up; it
  cannot conceal an incomplete decision.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Architecture impact

Expected impact: None. Stop if audit requires a new [operation](../../../../glossary.md#operation)/public [API](../../../../glossary.md#term-application-programming-interface-api)/[module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency)/shared
[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) [lifecycle](../../../../glossary.md#lifecycle)/workspace/architecture change.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009F1. Read AGENTS.md,
ARCHITECTURE.md, current architecture plan, Planning Guide, CPU master plan, CPU 0009, parent
0009F, this task, CPU 0008F/0008/0008A/0008N1, and current MATMUL/Conv1d/Conv2d/Conv3d source,
tests, Javadocs, references, Class-File evidence, preparation/finalization/invocation,
artifact/cache/inventory owners. Implement only this specification. Make four independent complete
decisions; retain generated dense compute on equality or uncertainty. Stop on an architecture or
scope conflict. Do not commit, stage, push, use GSD, run Gradle for retention-only work, add a
benchmark gate, or run broad validation absent a shared boundary. Hand the stabilized diff and
evidence to a distinct clean documentation-focused context following documentation-rules.md. Do
not mark Complete until that pass is recorded.
```

## Local decisions

- The task is decision-first and does not presume a migration.
- **[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) — retain the generated family.** `CpuMatmulLowering`,
  `CpuMatmulCandidateSelector`, `CpuMatmulIr`, and `CpuMatmulEmitter` implement four distinct
  full-K forms (direct scalar, N-vector, 2x2 scalar, and 2x2 N-vector), with the closed ordered
  ADD/terminal suffix.  Their cold geometry covers the normalized rank-one M/N cases, batches and
  right-broadcast batches, non-negative arbitrary layouts, exact numeric promotion, typed array
  and segment carriers, legal empty work, aliases/overlap validation, and per-form half-open
  ranges.  <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [layouts glossary](../../../../glossary.md#layout); [numeric promotion glossary](../../../../glossary.md#numeric-promotion); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [validation glossary](../../../../glossary.md#term-validation); [half-open glossary](../../../../glossary.md#term-half-open-interval))</span>`CpuPartitionPreparer` selects the form and workers cold; finalization realizes the
  selected keyed artifact, and <span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>`CpuPreparedExecutable` packs checked bindings before invocation.
  Replacing this with direct Java would have to reimplement all four forms, carrier/layout
  binding, epilogues, range ownership, artifact/cache/inventory retirement, and the corresponding
  negative and semantic coverage.  That is more than the existing retained family and is not a
  strictly cheaper complete route.  No migration or generated-route retirement is selected.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [carrier glossary](../../../../glossary.md#term-storage-carrier); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [cache glossary](../../../../glossary.md#term-cache))</span>
- **Conv1d visible composition — retain the existing [finite](../../../../glossary.md#term-finite) composition.**
  `CpuConv1dCompositionLowering` accepts only the two private `EXPAND_DIMS(axis 2)` branches,
  [`CONV2D`](../../../../glossary.md#convolution--conv2d), and `SQUEEZE(axis 2)`; it proves the singleton height is an address-preserving virtual
  view and delegates to the already selected Conv2d family.  It creates no Conv1d IR, emitter,
  specialization, artifact/cache entry, inventory row, workspace, or vector route.  Its sole
  prepared unit inherits Conv2d's grouped/batched geometry, intrinsic bias, promotion, carriers,
  aliases/overlap checks, empty behavior, and half-open ranges.  A direct surrounding change would
  either duplicate those cold and generated-route responsibilities or introduce the prohibited
  direct Conv1d family; neither is strictly cheaper.  No migration is selected.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [view glossary](../../../../glossary.md#term-view-meanings); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [specialization glossary](../../../../glossary.md#term-specialization); [bias glossary](../../../../glossary.md#term-bias))</span>
- **Direct Conv2d body — retain the generated family.** `CpuConv2dLowering`,
  `CpuConv2dIr`, and `CpuConv2dEmitter` keep grouped/depthwise NCHW cross-correlation, ordered
  FLOAT64/FLOAT32/BFLOAT16 promotion, optional intrinsic bias, array/native-order-segment/mixed
  carriers, checked geometry and aliases/overlap before writes, empty/special scalar behavior,
  zero workspace, half-open output-cell ranges, and only <span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>`NONE`/same-type right-broadcast
  `ADD`/`ADD_RELU` external suffixes.  `CpuPartitionPreparer` cold-selects the scalar or
  schema-63 width-vector realization only for the proved dense same-type FLOAT32/FLOAT64 subset;
  current generator-envelope schema is 66, while schema 63 remains the convolution-vector class
  identity.  A direct replacement would need all scalar/vector bodies and suffixes plus lowering,
  identity, artifact/cache, invocation, inventory, and test/evidence retirement.  It is not
  strictly cheaper, so no migration or selected fallback exists.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [identity glossary](../../../../glossary.md#term-identity-meanings); [lowering glossary](../../../../glossary.md#lowering))</span>
- **Direct [Conv3d](../../../../glossary.md#conv3d--ncdhw) body — retain the generated family.** `CpuConv3dLowering`, `CpuConv3dIr`, and
  `CpuConv3dEmitter` independently own the NCDHW rank-five version: grouped/depthwise,
  batch-aware floating cross-correlation with ordered promotion, intrinsic bias, geometry,
  layouts/carriers, aliases/overlap, empty/special behavior, zero workspace, and half-open
  output-cell ranges.  It deliberately has no external epilogue.  Its cold selection and
  finalization use the same bounded artifact lifecycle but a separate IR/emitter/geometry; the
  schema-63 vector identity is restricted to the proved dense same-type FLOAT32/FLOAT64
  width-block subset.  A direct replacement would retain all of those obligations and additionally
  retire the rank-specific generated lifecycle and evidence, so it is not demonstrably strictly
  cheaper.  No migration is selected.<span data-glossary-navigation="true"> ([NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [rank glossary](../../../../glossary.md#term-rank); [epilogue glossary](../../../../glossary.md#term-epilogue); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Conv1d composition is separate because it owns rank-edit orchestration, not independent
  generated compute; it is neither evidence for nor a fallback of direct Conv2d/Conv3d.<span data-glossary-navigation="true"> ([Conv1d composition glossary](../../../../glossary.md#conv1d-composition))</span>

## Known limitations

- No benchmark, [JIT](../../../../glossary.md#term-just-in-time-compilation-jit), or general [CPU](../../../../glossary.md#term-cpu)-[backend](../../../../glossary.md#backend) conclusion is made.
- [Conv2d](../../../../glossary.md#convolution--conv2d) ADD/ADD-RELU and [Conv3d](../../../../glossary.md#conv3d--ncdhw) no-external-[epilogue](../../../../glossary.md#term-epilogue) stay distinct current contracts.

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context read the required architecture, [planning](../../../../glossary.md#term-planning), parent/completed tasks,
documentation profiles, and the current family lowerers, IRs, emitters, references, preparer,
finalizer, executable, [artifact](../../../../glossary.md#term-artifact) store, focused tests, and inventory. The audit found no
architecture, [API](../../../../glossary.md#term-application-programming-interface-api), [dependency](../../../../glossary.md#term-dependency), workspace, [schema](../../../../glossary.md#term-schema), or build change.

- [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) evidence inspected: `CpuMatmulReferenceKernel` and `CpuMatmulReferenceTest` supply the
  typed full-K, increasing-K, single-store clean-Java oracle (including arbitrary non-negative
  strides, empty work, promotion, and integral wrapping). <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [strides glossary](../../../../glossary.md#term-stride))</span>`CpuMatmulGeneratedKernelTest` covers
  the four realization bodies, typed/segment carriers, epilogues, tails, and generated-body
  hygiene; `CpuMatmulSemanticClosureTest` constructs, defines, and invokes all 390 MATMUL
  inventory rows against its independent test-local oracle. `CpuMatmulLoweringTest` and
  `CpuMatmulCandidateSelectorTest` cover [admission](../../../../glossary.md#term-admission) and cold selection. The current
  `CpuMatmulEmitter` preserves full-K accumulators and one final represented store per owned
  output or microtile lane; its repository-resident generated-kernel test checks the relevant
  generated-body hygiene. This decision does not depend on a temporary evidence directory or on
  a materialization-candidate record.<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane); [kernel glossary](../../../../glossary.md#kernel); [materialization glossary](../../../../glossary.md#materialization))</span>
- Conv1d evidence inspected: `CpuConv1dCompositionLowering` and
  `CpuConv1dCompositionLoweringTest` prove the exact topology, virtual values, one Conv2d unit,
  zero workspace/materialization, wrong-axis failure, and scalar-only selection even when vector
  configuration is requested. Repository search found no Conv1d IR/emitter/reference/cache or
  generated-coverage inventory family. Therefore there is no generated Conv1d route to retire.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [axis glossary](../../../../glossary.md#term-axis); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [cache glossary](../../../../glossary.md#term-cache))</span>
- [Conv2d](../../../../glossary.md#convolution--conv2d) evidence inspected: `CpuConv2dReferenceKernel`, `CpuConv2dLoweringTest`,
  `CpuConv2dGeneratedKernelTest`, `CpuConv2dEvidenceTest`, and
  `CpuConvolutionSemanticClosureTest` cover the direct and bounded-suffix boundary. The clean
  oracle uses the same logical NCHW output-cell traversal, initialized accumulator, ordered kernel
  traversal, and one store. <span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [accumulator glossary](../../../../glossary.md#accumulator-type))</span>`CpuConv2dEvidenceTest` generates and parses representative classes and
  checks a final field-free class with one static `invoke` method; its constant-pool scan rejects
  Synaptik helpers, reflection, invoke dispatch, <span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>`java.util`, and boxing. The retained
  CPU 0008N1 and current <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>`CpuPartitionPreparer` establish the separate schema-63 vector identity
  and nested width-block route; scalar routes retain schema 52. Current evidence tests assert
  envelope schema 66, distinct from the convolution-vector class identity.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- [Conv3d](../../../../glossary.md#conv3d--ncdhw) evidence inspected: `CpuConv3dReferenceKernel`, `CpuConv3dLoweringTest`,
  `CpuConv3dGeneratedKernelTest`, `CpuConv3dEvidenceTest`, `CpuConvSimdEvidenceTest`, and
  `CpuConvolutionSemanticClosureTest` cover the rank-specific direct boundary and schema-63
  width-vector subset. The reference has the same NCDHW output-cell/kernel traversal and one
  store. <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw))</span>`CpuConv3dEvidenceTest` performs the same parsed Class-File hygiene checks over dense,
  grouped/bias, depthwise BF16, segment, mixed-layout, and parallel representatives.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [BF16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [layout glossary](../../../../glossary.md#layout))</span>
- The generator [dispatch](../../../../glossary.md#term-dispatch) (`CpuClassFileKernelGenerator`), artifact store
  (`CpuGeneratedKernelArtifactStore`), finalizer (`CpuPartitionFinalizer`), and prepared
  invocation (`CpuPreparedExecutable`) were inspected. Selection is cold in preparation,
  finalization realizes only the selected keyed artifact after assignment validation, the store
  verifies compatible generated/persisted bytes, and invocation binds typed carriers and
  run-owned ranges without selecting a route. The generated coverage inventory was inspected for
  MATMUL/Conv2d/Conv3d ownership; no Conv1d generated row exists.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

This clean documentation-focused context independently inspected the retained source and test
[owners](../../../../glossary.md#term-owner). Reuse without rerun is permitted by the Planning Guide and documentation rules because no
executable Java changed after the recorded focused evidence and this review found no concrete
discrepancy or cross-check risk. Accordingly, no [Gradle](../../../../glossary.md#term-gradle) command was [run](../../../../glossary.md#term-run). It checked the F1 and
parent/master/roadmap links, headings, fences, statuses, dependencies, frontiers, exact five-path
[scope](../../../../glossary.md#term-scope), final newlines, and terminology.

No-change conclusions: the reviewed Java/Javadocs remain accurate because neither public nor
implementation behavior changed; explanatory and architecture documentation need no revision
because no boundary or workflow changed; the glossary needs no entry because this records no new
or altered reusable term; build, architecture, [conformance](../../../../glossary.md#term-conformance), integration, and executable-test
owners need no change because no source, dependency, capability, or end-to-end behavior changed.

## Implementation notes

All four decisions retain their current routes. Consequently no Java, test, [cache](../../../../glossary.md#term-cache), inventory, or
route-retirement edit is warranted, and no generated/direct dual route was introduced. This task's
only implementation is the finalized F1 decision/evidence record.

## Completion summary

Completed changes: independently finalized four retained-route decisions and replaced the
non-durable temporary-path evidence wording with repository-resident source/test evidence.

Changed paths: this task plus its parent 0009F, [CPU](../../../../glossary.md#term-cpu) master plan, parent CPU 0009, and roadmap.

[Validation](../../../../glossary.md#term-validation): independently inspected the current lowerers, IRs, emitters, clean-Java references,
preparer/finalizer/executable/[artifact](../../../../glossary.md#term-artifact) [owners](../../../../glossary.md#term-owner), focused tests, and generated-coverage inventory;
reviewed Javadocs, explanatory documentation, glossary, architecture, build, [conformance](../../../../glossary.md#term-conformance), and
integration impact; reused existing focused Java evidence as permitted; and passed the final
[planning](../../../../glossary.md#term-planning) checks (`git diff --check` and `git status --short -uall`).

Unresolved/follow-up: no unresolved issue. 0009F remains [Ready](../../../../glossary.md#term-ready) and incomplete; 0009F2 [pooling](../../../../glossary.md#term-pooling) is
the sole next Draft summary frontier. F3 and 0009G remain later Draft and are not detailed here.

Status: Complete
