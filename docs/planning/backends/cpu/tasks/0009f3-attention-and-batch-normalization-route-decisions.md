# Task 0009F3: Attention and Batch-Normalization Route Decisions

## Status

Complete

## Goal

Make independent, complete CPU route decisions for current scaled-dot-product attention, BatchNorm
inference, and BatchNorm training/statistic transition. Use finite typed direct Java only when its
implementation, verification, and complete selected-generated-route retirement are demonstrably
strictly cheaper; equality or uncertainty retains generation. No common route/verifier, performance
gate, or benchmark campaign is authorized.

## Scope

Each row is a separate decision; a matching conclusion is not a shared abstraction.

| Route | Exact current capability matrix | Required complete decision |
| --- | --- | --- |
| Attention | `SCALED_DOT_PRODUCT_ATTENTION`: query/key/value floating inputs; optional right-broadcast canonical-BOOL mask; top-left causal eligibility; optional scale; one value result or two ordered result/weights outputs. `CpuAttentionLowering`/`Ir`/`Emitter`/`ReferenceKernel` own static broadcast-batch, query/key/value geometry; BFLOAT16/FLOAT32/FLOAT64 promoted result; array/native-order segment/mixed carriers; normalized non-negative layouts; deduplicated aliases; scalar/parallel complete-row ranges; and per-range score/weight scratch. `S==0` result-only is zero-range; frozen stable normalization governs empty/special/fully ineligible cases. Invalid input/output overlap fails before worker submission. Cold binding validates each distinct mask coordinate before output writes; finalization realizes selected schema-57 artifact/cache after assignment; typed prepared invocation receives carriers, packed geometry, range, and scratch. Inventory covers actual one/two-output, mask/causal/scale, role-alias, carrier/layout, and range rows. | Cover all listed facts, including NaN/infinite/signed-zero and canonical-mask validation, scratch and range ownership, cold binding/finalization/invocation, and artifact/cache/schema/inventory truth. |
| BatchNorm inference | `BATCH_NORM_INFERENCE`: five inputs—input, scale, bias, running mean, running variance—and one output. `CpuBatchNormInferenceLowering`/`Ir`/`Emitter`/`ReferenceKernel` own arbitrary static channel axis at rank >=2, promoted BFLOAT16/FLOAT32/FLOAT64 arithmetic, positive finite epsilon, broadcast channel vectors, injective output, arrays/segments/mixed carriers, dense/general non-negative layouts, and cold channel-range or flattened non-channel range ownership. Channel statistics are hoisted; empty output/channel work is legal; workspace/materialization is zero. Any output/input overlap fails before writes. Cold selection, artifact/cache/schema realization, and typed invocation retain exact carrier/layout/alias/range inventory facts. | Decide this stateless five/one arbitrary-axis route only; cover empty/special/epsilon validation, broadcast/layout/carrier addressing, aliases/overlap, zero workspace, cold selection/invocation, and artifact/cache/schema/inventory. |
| BatchNorm training/statistic transition | `BATCH_NORM_TRAINING`: five inputs and five ordered outputs—normalized affine output, next running mean, next running variance, saved mean, saved inverse standard deviation. `CpuBatchNormTrainingLowering`/`Ir`/`Emitter`/`ReferenceKernel` own arbitrary static channel axis, complete-channel range ownership, three non-channel passes, promoted BFLOAT16/FLOAT32/FLOAT64 arithmetic, finite momentum in `[0,1]`, positive finite epsilon, and corrected biased/unbiased transition. Each non-empty range has range-private exact-state scratch; only empty channel work has zero limbs/slice. Five injective outputs and all input/output/output overlap and mutation/publication constraints fail before publication writes. Cold analysis declares exact scratch and five buffers; finalization realizes selected generated artifact/schema/cache after assignment; typed prepared invocation publishes five outputs with run-owned workspace and no route selection. Inventory records axis/type/carrier/layout/range/scratch/alias/artifact facts. | Decide the complete five/five transition independently of inference; cover empty/non-empty exact-state workspace and lifetime, pass/range ownership, special/validation behavior, all publication and mutation constraints, cold binding/finalization/invocation, and artifact/cache/schema/inventory truth. |

Retained generated hot paths require their own optimal clean-Java oracle for semantic algorithm,
loop/dataflow/store shape, and avoidable-overhead profile, plus proportionate actual Class-File or
decompilation inspection for helpers, allocation, boxing, reflection, map/string dispatch, worker
management, and route/cache leakage. This is not a universal structural verifier. A migrated row
must completely retire selected lowering/IR/emitter selection, cold binding, invocation,
artifact/cache/schema/inventory, and generated-only evidence with no fallback or dual route.

## Out of scope

- Shared attention/normalization/reduction abstractions or verifiers; changed capability/numerics,
  Model/Compiler/Training/public APIs, dynamic Shapes, native/vector expansion, or shared
  Prepare/Runtime, workspace, materialization, package, module, or architecture changes.
- Benchmark/five-fork campaigns, performance/JIT claims, Gradle/executable/Javadoc edits during
  planning, broad/repository validation absent a changed shared boundary, and CPU 0009G.

## Architecture references and constraints

`ARCHITECTURE.md` is authoritative. CPU analysis/prepare owns cold lowering, route selection, and
exact resources; finalization constructs immutable prepared work after slots; Runtime cold-binds
typed carriers and invokes only prepared work. Preserve borrowed inputs, run-owned workspace,
publication ownership and failure cleanup. Hot loops add no allocation, boxing, reflection,
synchronization, map/string dispatch, or avoidable virtual dispatch; parallelism stays outside
family ranges. Stop before edits if complete work needs an architecture, module, dependency,
public API, shared lifecycle, resource-contract, or capability change.

## Package impact

Only existing CPU `internal.lowering`, `internal.ir`, `internal.codegen.emit`,
`internal.reference`, `internal.prepare`, `internal.executable`, `internal.cache`, and their
mirrored tests are used. No package/type is added or moved; direct replacement stays in its family
owner and cannot introduce shared normalization/reduction infrastructure.

## Affected files and maximum scope

Planning paths are exactly this task, parent F, parent 0009, CPU master plan, and roadmap. If all
routes retain, only those five paths change. A migration may add at most 13 focused CPU
source/test/evidence/Javadoc paths (18 total), including complete retirement; otherwise it is not
strictly cheaper and must retain or stop with a bounded follow-up.

## Acceptance criteria

1. Records three independent complete decisions and all matrix facts above: attention one/two
   output, mask/causal/broadcast/static geometry/types/carriers/layouts/scratch/ranges; inference
   five/one arbitrary channel axis; training five/five transition, exact-state workspace, range
   ownership and mutation/publication constraints; plus empty/special/validation/alias/overlap/
   cold/artifact/inventory facts.
2. Direct Java is selected only with source/test-backed proof of strictly cheaper complete work;
   greater/equal/uncertain cost retains generation. Retention has a family-specific oracle and
   Class-File/decompilation evidence; a migration has complete no-fallback retirement.
3. After executable stabilization, runs the focused command once. Retained-only work independently
   inspects stable owner evidence without rerunning Java. No benchmarks/five forks/repository-wide
   run occur absent a genuine shared boundary.
4. Uses distinct clean implementation and documentation contexts. The documentation context
   finalizes affected Javadocs/docs/glossary impact in the same change and validates planning/docs
   without duplicating stable Java tests.

## Tests / validation

For executable changes, run once after stabilization:

```bash
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.CpuAttentionCapabilityTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAttentionLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIrTest --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormTrainingIrTest --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuAttentionReferenceKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuBatchNormInferenceReferenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuBatchNormTrainingReferenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAttentionSemanticClosureTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAttentionGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBatchNormInferenceGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBatchNormTrainingGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAttentionEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBatchNormInferenceEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBatchNormTrainingEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageCheckpointTest
```

Record semantic/oracle-body dataflow, actual Class-File/decompilation, aliases/canaries/
failure-before-write, ranges/scratch, cold binding/finalization/invocation, and cache/schema/
inventory truth. The documentation context applies General and Planning profiles; checks links,
headings/anchors, fences, terminology/glossary, dependencies/sole frontier, exact scope, final
newline/new-file whitespace; then runs:

```bash
git diff --check
git status --short -uall
```

## Dependencies and follow-up tasks

Depends on Ready parent 0009F, Complete F2, CPU 0008H, CPU 0007F1, and CPU 0007F2. CPU 0009G
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

- **Scaled-dot-product attention — retain schema-57 generation.** `CpuAttentionLowering`,
  `CpuAttentionIr`, `CpuAttentionEmitter`, and `CpuAttentionReferenceKernel` are a single
  selected route, not interchangeable implementation fragments. They retain both one-output and
  ordered two-output entry shapes; right-broadcast canonical-BOOL mask and top-left causal
  eligibility; finite explicit/default scale; promoted BFLOAT16/FLOAT32/FLOAT64 arithmetic;
  static broadcast-batch/query/key/value geometry; normalized layouts; typed array,
  native-order-segment, and mixed carriers; and first-occurrence role aliases. The row owns stable
  score classification, weight formation, output stores, NaN/infinite/signed-zero outcomes,
  complete-row ranges, and aligned range-private score/weight scratch. `S == 0` result-only work
  remains a legal zero range. Cold binding validates every distinct mask coordinate and every
  input/output/workspace overlap before workers or writes; finalization realizes only the selected
  artifact and cache identity after assignment; prepared invocation receives typed carriers,
  packed geometry, range, and scratch without selecting a route. A direct finite typed Java body
  would have to reproduce that boundary and then retire the lowering, IR, emitter selection,
  binding, invocation, schema-57 specialization/artifact/cache, inventory, and generated evidence.
  That is greater work than retention, not strictly cheaper. No fallback or dual route is selected.

- **BatchNorm inference — retain schema-49 generation.** `CpuBatchNormInferenceLowering`,
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
  cheaper. No fallback or dual route is selected.

- **BatchNorm training/statistic transition — retain schema-50 generation.**
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
  work than retention and not demonstrably strictly cheaper. No fallback or dual route is selected.

## Known limitations

No performance, JIT, or whole-backend conclusion is made. Attention is not generalized softmax;
neither BatchNorm route becomes Layer/RMS normalization or a generic reduction.

## Validation evidence

The separate clean documentation-focused context read the required architecture and planning
contracts, documentation rules and General/Planning profiles, CPU 0009/F/F1/F2/F3, and relevant
attention and BatchNorm contracts, sources, and tests. The three retained routes have independent
clean-Java oracle and generated evidence owners: attention is schema 57, inference BatchNorm is
schema 49, and training BatchNorm is schema 50. Complete direct replacement would duplicate each
family's lowering/IR/emitter, cold validation and binding, invocation, artifact/cache/inventory,
and retirement proof, so it is not strictly cheaper. No fallback or dual route is selected.

The coordinating implementation context ran the focused Gradle command once with
`JAVA_TOOL_OPTIONS` propagating `synaptik.cpu.attention.structuralEvidenceRoot`; it completed
`BUILD SUCCESSFUL` in 1m16s (22 tasks). XML records one passing
`emitsAndScansExactElevenThousandEightHundredEightyInventory` test in 66.768 seconds, with zero
failures, errors, or skips. This pass independently inspected
`/private/tmp/synaptik-cpu-0009f3-attention-gradle.bRpgDy`: its manifest validates and has
SHA-256 `b75ad1aee6c2503a3cc2aad1a0477e63b85e9f0e65f8d7eded83829ffe282e39`; it contains 11,880
class files. `summary.txt` agrees on 11,880 classes, 992 rows, 456 skeletons, 312 fragments,
inventory hash `1d0dc5b397429da4e876252b82b05bf0e9e8a63b8100c9638aadfebea78921a2`, and `javap` hash
`6b827dee6c1a953c4e3ac7a20834632e62795c239b5fd609539ed4e36fd3ec84`.

The attention correction is limited to the inventory assertion and its comment: the loop has 57
legal mappings—27 distinct-role, 27 two-role-alias, and three all-role-alias
(`27 + 9 + 9 + 9 + 3`)—rather than 58. It changes neither a generated route nor numerical
behavior. Separately generated BatchNorm evidence was inspected: inference has eight classes and
manifest hash `d29041359f916941395bea4f3855fc2983c738fc88c27f5275a640c259728467`; training has
eight classes and manifest hash
`995ad93566225b7dfb452b0836edbd732d8fab9201c80b74434b92c7e304176e`.

No Java test was rerun by this documentation pass. No benchmark, timing mode, performance fork,
broad validation, architecture test, backend-conformance test, or integration test ran: the
correction changes no shared boundary, conformance contract, or end-to-end behavior. Javadoc,
glossary, architecture, production source, Model/Compiler/Training APIs, shared lifecycle, build,
conformance, and integration documentation were reviewed and need no change because neither
behavior nor reusable terminology changed.

## Implementation notes

No Java migration is selected. The documentation pass finalized the parent/master/roadmap status
records. The attention assertion now matches the legal 57-mapping inventory; no generated route is
changed.

## Completion summary

Completed independent retained-route decisions: schema-57 attention, schema-49 BatchNorm
inference, and schema-50 BatchNorm training remain generated because complete direct replacement,
verification, and retirement are not strictly cheaper. Corrected the attention inventory from 58
to the legal 57-row role-alias partition. No production Javadoc, glossary, architecture, build,
conformance, or integration update is needed. CPU 0009G is the sole next Draft frontier.

Status: Complete
