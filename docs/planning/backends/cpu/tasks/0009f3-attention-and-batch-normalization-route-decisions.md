# Task 0009F3: Attention and [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[Normalization](../../../../glossary.md#term-normalization) Route Decisions

## Reader context

This task records separate retention decisions for schema-57 attention, schema-49 BatchNorm
inference, and schema-50 BatchNorm training. The attention lowerer, intermediate representation,
emitter, and independent reference own masked complete-row attention and range-private score/weight
scratch. `CpuBatchNormInferenceLowering` and its family owners implement five-input/one-output
arbitrary-axis normalization without workspace. The training family instead publishes five ordered
outputs and owns complete-channel, three-pass work with exact-state scratch.

Here, clean-Java semantic oracles establish numerical behavior, while structural comparisons and
generated-class inspection examine loop/dataflow/store shape and prohibited machinery. Preparation
declares resources and validates boundaries; finalization realizes artifacts after assignment;
prepared invocation does not select routes. Attention and training scratch are run-owned temporary
workspace, not persistent model state. The attention mapping-count correction changes evidence
accounting, not generated behavior.

Current-topic navigation (not a replacement for the historical record):

- [current attention roles and result forms](../../../../reference/tensor/attention.md#scaled-dot-product-attention-expressions) — Names query/key/value, eligibility, and explicit one-output versus two-output requests.
- [current BatchNorm transition and saved-output roles](../../../../reference/tensor/statistics-losses.md#batch-normalization-training-and-statistic-transition-expressions) — Explains five outputs, biased versus unbiased variance, and pure next-statistic values rather than mutation.

## Status

Complete

## Goal

Make independent, complete [CPU](../../../../glossary.md#term-cpu) route decisions for current scaled-dot-product attention, BatchNorm
[inference](../../../../glossary.md#term-inference), and BatchNorm training/[statistic](../../../../glossary.md#term-statistic) transition. Use [finite](../../../../glossary.md#term-finite) typed direct Java only when its
implementation, verification, and complete selected-generated-route retirement are demonstrably
strictly cheaper; equality or uncertainty retains generation. No common route/verifier, performance
gate, or benchmark campaign is authorized.

## [Scope](../../../../glossary.md#term-scope)

Each row is a separate decision; a matching conclusion is not a shared abstraction.

| Route | Exact current capability [matrix](../../../../glossary.md#term-review-matrix) | Required complete decision |
| --- | --- | --- |
| Attention | `SCALED_DOT_PRODUCT_ATTENTION`: query/key/value floating inputs; optional right-broadcast [canonical](../../../../glossary.md#term-canonical-form)-[BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask); top-left causal eligibility; optional scale; one value result or two ordered result/[weights](../../../../glossary.md#term-weight) outputs. `CpuAttentionLowering`/`Ir`/`Emitter`/`ReferenceKernel` own static broadcast-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), query/key/value geometry; [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) promoted result; array/native-order segment/mixed carriers; normalized non-negative [layouts](../../../../glossary.md#layout); deduplicated [aliases](../../../../glossary.md#aliasing-and-arena); [scalar](../../../../glossary.md#term-scalar)/parallel complete-row ranges; and per-range score/weight scratch. `S==0` result-only is zero-range; frozen stable [normalization](../../../../glossary.md#term-normalization) governs empty/special/fully ineligible cases. Invalid input/output overlap fails before worker submission. Cold [binding](../../../../glossary.md#term-binding) validates each distinct mask [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) before output writes; finalization realizes selected [schema](../../../../glossary.md#term-schema)-57 [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache) after assignment; typed prepared invocation receives carriers, packed geometry, range, and scratch. Inventory covers actual one/two-output, mask/causal/scale, role-alias, [carrier](../../../../glossary.md#term-storage-carrier)/layout, and range rows. | Cover all listed facts, including [NaN](../../../../glossary.md#term-nan)/infinite/[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) and canonical-mask [validation](../../../../glossary.md#term-validation), scratch and range [ownership](../../../../glossary.md#term-ownership-meanings), cold binding/finalization/invocation, and artifact/cache/schema/inventory truth. |
| BatchNorm [inference](../../../../glossary.md#term-inference) | `BATCH_NORM_INFERENCE`: five inputs—input, scale, [bias](../../../../glossary.md#term-bias), running [mean](../../../../glossary.md#term-mean), running [variance](../../../../glossary.md#term-variance)—and one output. `CpuBatchNormInferenceLowering`/`Ir`/`Emitter`/`ReferenceKernel` own arbitrary static [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis) at [rank](../../../../glossary.md#term-rank) >=2, promoted BFLOAT16/FLOAT32/FLOAT64 arithmetic, positive [finite](../../../../glossary.md#term-finite) [epsilon](../../../../glossary.md#epsilon), broadcast channel [vectors](../../../../glossary.md#term-vector), [injective](../../../../glossary.md#term-injective) output, arrays/segments/mixed carriers, dense/general non-negative layouts, and cold channel-range or flattened non-channel range ownership. Channel statistics are hoisted; empty output/channel work is legal; workspace/[materialization](../../../../glossary.md#materialization) is zero. Any output/input overlap fails before writes. Cold selection, artifact/cache/schema realization, and typed invocation retain exact carrier/layout/alias/range inventory facts. | Decide this stateless five/one arbitrary-axis route only; cover empty/special/epsilon validation, broadcast/layout/carrier addressing, aliases/overlap, zero workspace, cold selection/invocation, and artifact/cache/schema/inventory. |
| BatchNorm training/[statistic](../../../../glossary.md#term-statistic) transition | `BATCH_NORM_TRAINING`: five inputs and five ordered outputs—normalized affine output, next running mean, next running variance, saved mean, saved inverse [standard deviation](../../../../glossary.md#term-standard-deviation). `CpuBatchNormTrainingLowering`/`Ir`/`Emitter`/`ReferenceKernel` own arbitrary static channel axis, complete-channel range ownership, three non-channel passes, promoted BFLOAT16/FLOAT32/FLOAT64 arithmetic, finite [momentum](../../../../glossary.md#term-momentum) in `[0,1]`, positive finite epsilon, and corrected biased/unbiased transition. Each non-empty range has range-private exact-state scratch; only empty channel work has zero limbs/[slice](../../../../glossary.md#slice). Five injective outputs and all input/output/output overlap and mutation/[publication](../../../../glossary.md#term-publication) constraints fail before publication writes. Cold analysis declares exact scratch and five [buffers](../../../../glossary.md#term-buffer); finalization realizes selected generated artifact/schema/cache after assignment; typed prepared invocation publishes five outputs with [run](../../../../glossary.md#term-run)-owned workspace and no route selection. Inventory records axis/type/carrier/layout/range/scratch/alias/artifact facts. | Decide the complete five/five transition independently of inference; cover empty/non-empty exact-state workspace and [lifetime](../../../../glossary.md#term-lifetime), pass/range ownership, special/validation behavior, all publication and mutation constraints, cold binding/finalization/invocation, and artifact/cache/schema/inventory truth. |

Retained generated hot paths require their own optimal clean-Java [oracle](../../../../glossary.md#term-test-oracle) for semantic algorithm,
loop/[dataflow](../../../../glossary.md#term-dataflow)/store [shape](../../../../glossary.md#shape), and avoidable-overhead profile, plus proportionate actual Class-File or
[decompilation](../../../../glossary.md#term-decompilation) inspection for helpers, allocation, boxing, reflection, map/string [dispatch](../../../../glossary.md#term-dispatch), worker
management, and route/cache leakage. This is not a universal structural verifier. A migrated row
must completely retire selected [lowering](../../../../glossary.md#lowering)/[IR](../../../../glossary.md#term-intermediate-representation-ir)/emitter selection, cold binding, invocation,
artifact/cache/schema/inventory, and generated-only evidence with no fallback or dual route.

## Out of [scope](../../../../glossary.md#term-scope)

- Shared attention/normalization/reduction abstractions or verifiers; changed capability/numerics,
  Model/Compiler/Training/public APIs, dynamic Shapes, native/vector expansion, or shared
  Prepare/Runtime, workspace, materialization, package, module, or architecture changes.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Shapes glossary](../../../../glossary.md#shape); [vector glossary](../../../../glossary.md#term-vector); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [materialization glossary](../../../../glossary.md#materialization); [module glossary](../../../../glossary.md#term-module))</span>
- Benchmark/five-fork campaigns, performance/JIT claims, Gradle/executable/Javadoc edits during
  planning, broad/repository validation absent a changed shared boundary, and CPU 0009G.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [Gradle glossary](../../../../glossary.md#term-gradle); [Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning); [validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture references and constraints

`ARCHITECTURE.md` is authoritative. [CPU](../../../../glossary.md#term-cpu) analysis/[prepare](../../../../glossary.md#prepare) owns cold [lowering](../../../../glossary.md#lowering), route selection, and
exact resources; finalization constructs [immutable](../../../../glossary.md#term-immutable) prepared work after slots; [Runtime](../../../../glossary.md#term-runtime) cold-binds
typed carriers and invokes only prepared work. Preserve [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) inputs, [run](../../../../glossary.md#term-run)-owned workspace,
[publication](../../../../glossary.md#term-publication) [ownership](../../../../glossary.md#term-ownership-meanings) and failure cleanup. Hot loops add no allocation, boxing, reflection,
[synchronization](../../../../glossary.md#term-synchronization), map/string [dispatch](../../../../glossary.md#term-dispatch), or avoidable virtual dispatch; parallelism stays outside
family ranges. Stop before edits if complete work needs an architecture, [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency),
public [API](../../../../glossary.md#term-application-programming-interface-api), shared [lifecycle](../../../../glossary.md#lifecycle), resource-contract, or capability change.

## Package impact

Only existing [CPU](../../../../glossary.md#term-cpu) `internal.lowering`, `internal.ir`, `internal.codegen.emit`,
`internal.reference`, `internal.prepare`, `internal.executable`, `internal.cache`, and their
mirrored tests are used. No package/type is added or moved; direct replacement stays in its family
[owner](../../../../glossary.md#term-owner) and cannot introduce shared [normalization](../../../../glossary.md#term-normalization)/reduction infrastructure.

## Affected files and maximum [scope](../../../../glossary.md#term-scope)

[Planning](../../../../glossary.md#term-planning) paths are exactly this task, parent F, parent 0009, [CPU](../../../../glossary.md#term-cpu) master plan, and roadmap. If all
routes retain, only those five paths change. A migration may add at most 13 focused CPU
source/test/evidence/[Javadoc](../../../../glossary.md#term-javadoc) paths (18 total), including complete retirement; otherwise it is not
strictly cheaper and must retain or stop with a bounded follow-up.

## Acceptance criteria

1. Records three independent complete decisions and all matrix facts above: attention one/two
   output, mask/causal/broadcast/static geometry/types/carriers/layouts/scratch/ranges; inference
   five/one arbitrary channel axis; training five/five transition, exact-state workspace, range
   ownership and mutation/publication constraints; plus empty/special/validation/alias/overlap/
   cold/artifact/inventory facts.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [mask glossary](../../../../glossary.md#term-boolean-mask); [layouts glossary](../../../../glossary.md#layout); [inference glossary](../../../../glossary.md#term-inference); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [publication glossary](../../../../glossary.md#term-publication); [validation glossary](../../../../glossary.md#term-validation); [alias glossary](../../../../glossary.md#aliasing-and-arena); [artifact glossary](../../../../glossary.md#term-artifact))</span>
2. Direct Java is selected only with source/test-backed proof of strictly cheaper complete work;
   greater/equal/uncertain cost retains generation. Retention has a family-specific oracle and
   Class-File/decompilation evidence; a migration has complete no-fallback retirement.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [decompilation glossary](../../../../glossary.md#term-decompilation))</span>
3. After executable stabilization, runs the focused command once. Retained-only work independently
   inspects stable owner evidence without rerunning Java. No benchmarks/five forks/repository-wide
   run occur absent a genuine shared boundary.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [run glossary](../../../../glossary.md#term-run))</span>
4. Uses distinct clean implementation and documentation contexts. The documentation context
   finalizes affected Javadocs/docs/glossary impact in the same change and validates planning/docs
   without duplicating stable Java tests.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

For executable changes, [run](../../../../glossary.md#term-run) once after stabilization:

```bash
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.CpuAttentionCapabilityTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAttentionLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIrTest --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormTrainingIrTest --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuAttentionReferenceKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuBatchNormInferenceReferenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuBatchNormTrainingReferenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAttentionSemanticClosureTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAttentionGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBatchNormInferenceGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBatchNormTrainingGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAttentionEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBatchNormInferenceEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBatchNormTrainingEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageCheckpointTest
```

Record semantic/[oracle](../../../../glossary.md#term-test-oracle)-body [dataflow](../../../../glossary.md#term-dataflow), actual Class-File/[decompilation](../../../../glossary.md#term-decompilation), [aliases](../../../../glossary.md#aliasing-and-arena)/canaries/
failure-before-write, ranges/scratch, cold [binding](../../../../glossary.md#term-binding)/finalization/invocation, and [cache](../../../../glossary.md#term-cache)/[schema](../../../../glossary.md#term-schema)/
inventory truth. The documentation context applies General and [Planning](../../../../glossary.md#term-planning) profiles; checks links,
headings/anchors, fences, terminology/glossary, dependencies/sole frontier, exact [scope](../../../../glossary.md#term-scope), final
newline/new-file whitespace; then runs:

```bash
git diff --check
git status --short -uall
```

## Dependencies and follow-up tasks

Depends on [Ready](../../../../glossary.md#term-ready) parent 0009F, Complete F2, [CPU](../../../../glossary.md#term-cpu) 0008H, CPU 0007F1, and CPU 0007F2. CPU 0009G
remains Draft after F3 and is not detailed here.

## Architecture impact

Expected impact: None.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009F3. Read AGENTS.md,
ARCHITECTURE.md, current architecture plan, Planning Guide, CPU master, CPU 0009, parent 0009F,
this task, CPU 0008H, CPU 0007F1/F2, and current Model/Compiler/CPU family source, tests,
Javadocs, references, Class-File evidence, workspace/preparation/finalization/invocation,
artifact/cache/schema/inventory owners. Implement only this specification. Make three independent
complete decisions; direct Java requires strictly cheaper complete work, otherwise retain. Stop on
architecture/scope conflict. Do not commit, stage, push, use GSD, add benchmark gates, or run broad
validation absent a shared boundary. Hand the stabilized diff and focused evidence to a distinct
clean documentation-focused context following documentation-rules.md; do not mark Complete until
its review is recorded.
```

## Local decisions

Decision — all three independent decisions are **retain current generated route**.

- **Scaled-dot-product attention — retain [schema](../../../../glossary.md#term-schema)-57 generation.** `CpuAttentionLowering`,
  `CpuAttentionIr`, `CpuAttentionEmitter`, and `CpuAttentionReferenceKernel` are a single
  selected route, not interchangeable implementation fragments. They retain both one-output and
  ordered two-output entry shapes; right-broadcast canonical-BOOL mask and top-left causal
  eligibility; finite explicit/default scale; promoted BFLOAT16/FLOAT32/FLOAT64 arithmetic;
  static broadcast-batch/query/key/value geometry; normalized layouts; typed array,
  native-order-segment, and mixed carriers; and first-occurrence role aliases. The row owns stable
  score classification, weight formation, output stores, NaN/infinite/signed-zero outcomes,
  complete-row ranges, and aligned range-private score/weight scratch. <span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask); [finite glossary](../../../../glossary.md#term-finite); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [layouts glossary](../../../../glossary.md#layout); [occurrence glossary](../../../../glossary.md#term-occurrence); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [weight glossary](../../../../glossary.md#term-weight); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`S == 0` result-only work
  remains a legal zero range. Cold binding validates every distinct mask coordinate and every
  input/output/workspace overlap before workers or writes; finalization realizes only the selected
  artifact and cache identity after assignment; prepared invocation receives typed carriers,
  packed geometry, range, and scratch without selecting a route. A direct finite typed Java body
  would have to reproduce that boundary and then retire the lowering, IR, emitter selection,
  binding, invocation, schema-57 specialization/artifact/cache, inventory, and generated evidence.
  That is greater work than retention, not strictly cheaper. No fallback or dual route is selected.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [lowering glossary](../../../../glossary.md#lowering); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [specialization glossary](../../../../glossary.md#term-specialization))</span>

- **BatchNorm [inference](../../../../glossary.md#term-inference) — retain schema-49 generation.** `CpuBatchNormInferenceLowering`,
  `CpuBatchNormInferenceIr`, `CpuBatchNormInferenceEmitter`, and
  `CpuBatchNormInferenceReferenceKernel` separately own the five-input/one-output arbitrary-axis
  formula. The retained body preserves channel-hoisted channel-range or flattened non-channel
  range ownership; promoted BFLOAT16/FLOAT32/FLOAT64 arithmetic; finite positive epsilon;
  broadcast vectors; dense/general non-negative layouts; typed array/segment/mixed carriers;
  empty output or channel work; and the frozen special-value and signed-zero behavior. It declares
  zero workspace and materialization, rejects every input/output overlap before writes, then
  cold-selects and invokes the compatible generated artifact using exact carrier/layout/alias/range
  facts. A direct replacement would duplicate the complete formula/access/range/validation body
  and retire all lowerer/IR/emitter, cold binding/invocation, schema-49 artifact/cache, inventory,
  and evidence paths. That complete migration is greater work than retention, not strictly
  cheaper. No fallback or dual route is selected.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [channel glossary](../../../../glossary.md#term-channel); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [epsilon glossary](../../../../glossary.md#epsilon); [vectors glossary](../../../../glossary.md#term-vector); [materialization glossary](../../../../glossary.md#materialization); [carrier glossary](../../../../glossary.md#term-storage-carrier); [validation glossary](../../../../glossary.md#term-validation))</span>

- **BatchNorm training/[statistic](../../../../glossary.md#term-statistic) transition — retain schema-50 generation.**
  `CpuBatchNormTrainingLowering`, `CpuBatchNormTrainingIr`,
  `CpuBatchNormTrainingEmitter`, and `CpuBatchNormTrainingReferenceKernel` retain the independent
  five-input/five-output transition. Each channel range owns its complete three-pass reduction,
  corrected biased/unbiased variance computation, momentum transition, epsilon/inverse-standard-
  deviation order, four scalar-statistic publications, and normalized-affine output pass. Every
  non-empty range receives a private exact-state scratch slice; empty channel work has no limbs or
  slice. The route preserves promoted BFLOAT16/FLOAT32/FLOAT64 arithmetic, arbitrary axis and
  layout/carrier addressing, special values, all five injective outputs, and every input/output,
  output/output, and workspace overlap rejection before publication writes. Cold analysis declares
  the five buffers and exact workspace; finalization realizes the selected schema-50 artifact/cache
  after assignment; typed prepared invocation publishes all five outputs without route selection.
  A direct replacement must duplicate all of this and fully retire the selected lowerer/IR/emitter,
  binding/invocation, schema/cache/artifact, inventory, and generated-only evidence. It is greater
  work than retention and not demonstrably strictly cheaper. No fallback or dual route is selected.<span data-glossary-navigation="true"> ([variance glossary](../../../../glossary.md#term-variance); [momentum glossary](../../../../glossary.md#term-momentum); [scalar glossary](../../../../glossary.md#term-scalar); [slice glossary](../../../../glossary.md#slice); [injective glossary](../../../../glossary.md#term-injective); [publication glossary](../../../../glossary.md#term-publication); [buffers glossary](../../../../glossary.md#term-buffer))</span>

## Known limitations

No performance, [JIT](../../../../glossary.md#term-just-in-time-compilation-jit), or whole-[backend](../../../../glossary.md#backend) conclusion is made. Attention is not generalized [softmax](../../../../glossary.md#softmax--log-softmax);
neither BatchNorm route becomes Layer/[RMS normalization](../../../../glossary.md#rms-normalization) or a generic reduction.

## [Validation](../../../../glossary.md#term-validation) evidence

The separate clean documentation-focused context read the required architecture and [planning](../../../../glossary.md#term-planning)
contracts, documentation rules and General/Planning profiles, [CPU](../../../../glossary.md#term-cpu) 0009/F/F1/F2/F3, and relevant
attention and BatchNorm contracts, sources, and tests. The three retained routes have independent
clean-Java [oracle](../../../../glossary.md#term-test-oracle) and generated evidence [owners](../../../../glossary.md#term-owner): attention is [schema](../../../../glossary.md#term-schema) 57, [inference](../../../../glossary.md#term-inference) BatchNorm is
schema 49, and training BatchNorm is schema 50. Complete direct replacement would duplicate each
family's [lowering](../../../../glossary.md#lowering)/[IR](../../../../glossary.md#term-intermediate-representation-ir)/emitter, cold validation and [binding](../../../../glossary.md#term-binding), invocation, [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache)/inventory,
and retirement proof, so it is not strictly cheaper. No fallback or dual route is selected.

The coordinating implementation context ran the focused [Gradle](../../../../glossary.md#term-gradle) command once with
`JAVA_TOOL_OPTIONS` propagating `synaptik.cpu.attention.structuralEvidenceRoot`; it completed
`BUILD SUCCESSFUL` in 1m16s (22 tasks). XML records one passing
`emitsAndScansExactElevenThousandEightHundredEightyInventory` test in 66.768 seconds, with zero
failures, errors, or skips. This pass independently inspected
`/private/tmp/synaptik-cpu-0009f3-attention-gradle.bRpgDy`: its [manifest](../../../../glossary.md#term-manifest) validates and has
[SHA-256](../../../../glossary.md#term-sha-256) `b75ad1aee6c2503a3cc2aad1a0477e63b85e9f0e65f8d7eded83829ffe282e39`; it contains 11,880
class files. `summary.txt` agrees on 11,880 classes, 992 rows, 456 skeletons, 312 fragments,
inventory [hash](../../../../glossary.md#term-hash) `1d0dc5b397429da4e876252b82b05bf0e9e8a63b8100c9638aadfebea78921a2`, and `javap` hash
`6b827dee6c1a953c4e3ac7a20834632e62795c239b5fd609539ed4e36fd3ec84`.

The attention correction is limited to the inventory assertion and its comment: the loop has 57
legal mappings—27 distinct-role, 27 two-role-[alias](../../../../glossary.md#aliasing-and-arena), and three all-role-alias
(`27 + 9 + 9 + 9 + 3`)—rather than 58. It changes neither a generated route nor numerical
behavior. Separately generated BatchNorm evidence was inspected: inference has eight classes and
manifest hash `d29041359f916941395bea4f3855fc2983c738fc88c27f5275a640c259728467`; training has
eight classes and manifest hash
`995ad93566225b7dfb452b0836edbd732d8fab9201c80b74434b92c7e304176e`.

No Java test was rerun by this documentation pass. No benchmark, timing mode, performance [fork](../../../../glossary.md#term-fork-meanings),
broad validation, [architecture test](../../../../glossary.md#term-architecture-test), [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) test, or [integration test](../../../../glossary.md#term-integration-test) ran: the
correction changes no shared boundary, conformance contract, or end-to-end behavior. [Javadoc](../../../../glossary.md#term-javadoc),
glossary, architecture, production source, [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/Training APIs, shared [lifecycle](../../../../glossary.md#lifecycle), build,
conformance, and integration documentation were reviewed and need no change because neither
behavior nor reusable terminology changed.

## Implementation notes

No Java migration is selected. The documentation pass finalized the parent/master/roadmap status
records. The attention assertion now matches the legal 57-mapping inventory; no generated route is
changed.

## Completion summary

Completed independent retained-route decisions: [schema](../../../../glossary.md#term-schema)-57 attention, schema-49 BatchNorm
[inference](../../../../glossary.md#term-inference), and schema-50 BatchNorm training remain generated because complete direct replacement,
verification, and retirement are not strictly cheaper. Corrected the attention inventory from 58
to the legal 57-row role-[alias](../../../../glossary.md#aliasing-and-arena) [partition](../../../../glossary.md#partition). No production [Javadoc](../../../../glossary.md#term-javadoc), glossary, architecture, build,
[conformance](../../../../glossary.md#term-conformance), or integration update is needed. [CPU](../../../../glossary.md#term-cpu) 0009G is the sole next Draft frontier.

Status: Complete
