# Task 0009F3: Attention and Batch-Normalization Route Decisions

## Status

Ready

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

Implementation draft—retain all three current generated routes independently:

- **Attention:** retain schema-57 generation. One/two outputs, mask/causal eligibility, stable
  score/weight/output flow, broadcast/general layouts, typed carriers, aliases, range-private
  scratch, canonical-mask cold validation, pre-write overlap failure, and prepared artifact/cache
  invocation are one complete boundary. Equivalent direct bodies plus their proof and complete
  retirement are greater work, not strictly cheaper. `CpuAttentionReferenceKernel` and actual
  Class-File/decompilation evidence remain mandatory; no fallback/dual route is selected.
- **Inference BatchNorm:** retain generation. Its five/one arbitrary-axis formula, channel-hoisted
  range forms, broadcast vectors, promotion, carrier/layout matrix, empty/special validation, zero
  workspace, overlap proof, cold binding, artifact/cache, and invocation would all be duplicated
  before generated-route retirement. It is therefore not strictly cheaper; its reference and
  generated-class evidence remain independent. No fallback/dual route is selected.
- **Training BatchNorm/statistic transition:** retain generation. Its five/five three-pass,
  exact-state scratch, corrected statistic transition, momentum/epsilon, five-output publication,
  and mutation/overlap contract requires distinct direct-route and full-retirement proof. It is not
  demonstrably strictly cheaper. Its reference and generated-class evidence remain independent; no
  fallback/dual route is selected.

## Known limitations

No performance, JIT, or whole-backend conclusion is made. Attention is not generalized softmax;
neither BatchNorm route becomes Layer/RMS normalization or a generic reduction.

## Validation evidence

The planning context read the required architecture/planning/documentation sources and current
Model/Compiler attention and BatchNorm contracts; CPU lowerers, IRs, emitters, references, tests,
workspace/preparation/finalization/executable paths, artifact/cache/schema, and inventory. It
found no architecture uncertainty. No executable or Javadoc source changed during planning, so no
Gradle command was run. Completion evidence is empty until separate implementation and
documentation contexts finish.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
