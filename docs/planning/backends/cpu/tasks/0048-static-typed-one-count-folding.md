# Task CPU0048: Proven static typed-one count folding

## Status

Complete — MAIN accepted 2026-10-31 after independent source/docs approval and final CPU/Javadoc
and named join PASS. CPU frontier closed; CPU0049–0052 stay Draft/held.

## Change class

Class C — CPU-private generated dataflow/resource elimination; clean implementation and
independent code/numerical/resource/Javadoc/documentation review.

## Goal and scope

Replace a proved static SUM of exact typed positive ones with the same represented count,
without executing private expansion/materialization/reduction work. A logical splat means one
immutable typed scalar repeated at every logical coordinate, not a sampled buffer value.

- Use only explicit (EXPLICIT) exact typed facts already projected in `PrepareContext.constants()`.
  Recognize direct splats and same-dtype `EXPAND` with optional `CONTIGUOUS` before ordinary SUM.
  All relevant source/domain/result shapes must be fully static; do not specialize a sampled B.
- Cover supported FLOAT32/FLOAT64/BFLOAT16/FLOAT16/INT32/INT64 SUM full, single/multiple-axis,
  keep/remove-dimension and SUM-to-Shape forms using existing normalized-axis rules.
- Preserve output shape/layout, typed empty/point semantics, modular integral results and the
  current CPU once-rounded exact floating SUM result, including low narrowing and overflow.
  This is a count proof, not general constant propagation or a new reduction algorithm.

## Non-goals

No Compiler/Model/shared Prepare/Runtime/Engine production change, supported API, module edge,
lifecycle, native ABI, Scan-body lowering/tuning, new capability, general affine propagation,
masked/data-dependent counts, other constants/reductions, benchmark campaign or speed/parity claim.
No build edits, historical timing selectors, automatic successor, stage/commit/push or rollout.

## Contracts

- [Root: Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Backend owner: [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  — CPU-local lowering/resource declaration; finalization generates only after slot assignment.
- Numerical owner: [Exact and discrete floor](../../../../architecture/contracts/foundational-modules.md#exact-and-discrete-floor),
  [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives)
  and [Aggregate floor](../../../../architecture/contracts/foundational-modules.md#aggregate-floor)
  — mapping, contributor membership, empty identities and rounding remain intact.
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  — explicit splat provenance, ordinary count algebra and saved/publication boundaries remain unchanged.
- [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)
  and [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  — immutable projected facts and exact declarations before slots; no handoff/owner change.

Current source/tests refine CPU algorithms, not Model authority; stop on a missing/conflicting contract.

## Dependencies and integration

- Depends on: CPU0047 Complete; accepted CPU0076 Complete as the current integrated CPU baseline.
- Conflicts with: CPU lowering/portable IR/emission/cache/resource writers; CPU0049–0052; shared
  CPU guide/master/roadmap writers. Writes serialized; final docs delegated exclusively by MAIN.
- Parallel group: None; serial implementation and independent read-only review.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty tree;
  MAIN's 2580-file `/private/tmp/synaptik-cpu0048.G18rM6/MAIN-BASELINE.json` is the conservation base.
- Integration order: accepted CPU0047/CPU0076 → CPU0048 implementation/focused checks → independent
  review and final Javadocs → executor's one final CPU test/Javadoc → MAIN named consumer join/acceptance.
- Integration validation: the named conformance/Engine checkpoint below; MAIN conserves accepted
  unrelated work using its baseline and `before-cpu.tar`. No full-root or performance campaign.
- Shared-document integration owner: MAIN; independent reviewer proposes/finalizes targeted docs
  under exclusive delegation, reusing executable evidence. Workers do not edit master/roadmap.

Launch verified by MAIN, 2026-10-31: Ready frontier, Complete predecessors/reviews/gates, contracts,
unchanged HEAD/index, protected dirty base and inactive conflicts; CPU0048-only GO after plan APPROVE.

## Source-grounded seam and files

CPU paths below are relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.

- `internal/lowering/CpuAggregateLowering.java`: reuse `normalizedAxes`/existing geometry checks;
  prove every contributor is typed +1 through the bounded chain and compute exact count bits cold.
  `CpuPartitionLowering.lower` dispatches that proof before ordinary family lowering.
- `internal/lowering/CpuPartitionDagDecomposer.java`: select the complete private chain as one seed
  before ordinary affine/aggregate seeds; retain node membership/dependencies and original value IDs.
  Use full DAG consumers and logical-memory publication/external-consumer obligations, not adjacency
  alone. Refuse absorption across a public, saved, fan-out or outside-consumer intermediate.
- Added `internal/ir/CpuConstantWriteIr.java` in package
  `io.github.pho001.synaptik.backend.cpu.internal.ir`: exact typed `CpuKernelIr.ScalarImmediate`
  bits and structural output writes, without computational inputs, contributor loops or scratch.
- Wired through existing `internal/ir/CpuPortableKernelIr.java`,
  `CpuPartitionLowering.LoweredPartition.kernelIr`, `internal/route/portable/CpuPortableRoutePlan.java`,
  `internal/prepare/CpuPartitionPreparer.java` and `CpuPartitionPreparationPlan.java` only as needed.
  Existing `internal/codegen/emit/CpuClassFileKernelGenerator.java` emits direct typed stores with
  existing carrier/loop machinery; no new emitter/executor/framework or public abstraction.
- `internal/prepare/CpuPartitionFinalizer.java`: MAIN-authorized existing-composite predicate for
  a single constant-write unit; ValueId-based selection keeps original sources out of its entry.
- `internal/cache/CpuKernelSpecialization.java` unchanged: the IR fingerprint keys baked typed
  literals/code shape in both identity streams; retain unaffected/fallback keys and ClassFiles.
  No global schema/projection bump; affected-versus-unaffected controls remain required.
- Tests under the corresponding CPU test root: new `internal/lowering/CpuStaticTypedOneCountFoldingTest.java`,
  `internal/prepare/CpuStaticTypedOneCountPreparationTest.java` and
  `internal/codegen/emit/CpuConstantWriteGeneratedKernelTest.java`; reuse aggregate/DAG fixtures and
  independent exact rational/integral oracles. Update `CpuInternalPackageInventoryTest` for the one type.
- Add `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuStaticTypedOneCountConformanceTest.java`;
  extend `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineCpuComposedValidationIntegrationTest.java`.
  `LossGradientRules.restore` and `NormalizationGradientRules.count` in Compiler are read-only witnesses:
  respectively `oneBase.expand(domain).contiguous().sum()` and `oneLike(tensor).sum(axes,true)`.

## Acceptance criteria

- Static explicit +1 witnesses fold, including actual Compiler-generated mean-loss and normalization
  gradients. Full/axis/multi-axis/SUM-to-Shape, scalar, selected-zero, unselected-zero, empty-axis and
  domain-one cases preserve exact represented results and output metadata, including public outputs.
- Check ties-to-even around BF16 256/258, F16 2048/2050 and F16 finite/overflow boundary, and F32/F64
  precision boundaries; test integral modular counts and checked shape/count overflow behavior.
  Large-count metadata/oracle tests must not allocate huge arrays or iterate billions of contributors.
  A host `elementCount` cast is insufficient without proof of identical current-route rounding.
- Inspect analysis declarations AND prepared memory/units AND generated ClassFiles: absorbed private
  EXPAND/CONTIGUOUS values have no buffer/fill/copy/executable; SUM has no contributor loop or exact-state
  workspace. Constant writes use only required output stores, no runtime input read/dummy arithmetic,
  hidden helper dispatch, boxing, reflection or per-element allocation. Compare with clean direct
  Java specialized constant stores; prove removed work structurally, without a timing claim.
- Preserve required original source declarations/initialized representations: `GraphPreparation`
  validates every compile constant source. A scalar +1 source may remain initialized; it is not
  the private domain-sized expansion. A nonscalar explicit source's mandated initialization also
  remains. Do not erase source obligations or claim their bytes/fill were removed.
- Public/saved/fan-out/external intermediates remain materialized via unchanged fallback; final
  count publication and downstream uses remain valid. Test barriers at each absorbed stage.
- Negative controls: absent facts; mutable host leaves initially full of ones; same shape/label/
  factory or equal-looking Tensor; non-one/wrong-typed facts; casts/arithmetic/masks; unsupported
  layouts/topologies. No payload read, leaf equality or mutable-value assumption proves constancy.
  Change actual storage/fresh run inputs after preparation and verify recomputation/current results.
- Dynamic/named-B or otherwise unproved counts keep their current count algebra, resource topology,
  admission/failure behavior and reusable run binding. Existing Scan paths/defaults remain untouched.

## Validation

Executor owns focused checks and one final affected-module test/Javadoc gate after final docs.
Review reuses executable evidence and checks links/anchors/size/status/diff; no Java rerun.
Use fixed JDK26/offline dependencies and no opted-in performance properties or evidence-root overrides:

```bash
export JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home
CPU0048_INIT=/private/tmp/synaptik-cpu0048.G18rM6/no-campaigns.gradle
./gradlew --offline --no-daemon -I "$CPU0048_INIT" :backends:cpu:test \
  --tests '*CpuStaticTypedOneCountFoldingTest' --tests '*CpuStaticTypedOneCountPreparationTest' \
  --tests '*CpuConstantWriteGeneratedKernelTest' --tests '*CpuAggregateLoweringTest' \
  --tests '*CpuAggregateGeneratedKernelTest'
./gradlew --offline --no-daemon -I "$CPU0048_INIT" :backends:cpu:test :backends:cpu:javadoc
```

The external init excludes ONLY `CpuPartitionDagGeneratedEvidenceTest.fiveFixedHeapForksAndAggregateStayWithinGate`
and `.freshFusionEvidenceRetainsEverySampleAndBothRequiredComparisons` in package
`io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`; disclose those historical unconditional
timing exclusions and normal opt-in skips. This is not an unfiltered-default PASS; do not edit build/tests
to suppress them. If the init is unavailable, ask MAIN to restore it before the module gate.

MAIN runs this named join once after integration, with the same JDK/offline/no-performance environment:

```bash
./gradlew --offline --no-daemon :testing:backend-conformance:test \
  --tests '*CpuStaticTypedOneCountConformanceTest' --tests '*CpuOrdinaryDynamicBatchConformanceTest' \
  :testing:integration-tests:test --tests '*EngineCpuComposedValidationIntegrationTest'
git diff --check
```

## Documentation and review impact

Update affected implementation/type Javadocs and `docs/backend-guide/cpu-backend.md`'s ordinary
aggregate explanation in the same change; explain constant proof, resource removal/source exception,
observable barriers and fallback. Apply General/Backend/API-Javadoc profiles; glossary search for
logical splat/compile constant/materialization requires no new term unless meaning actually changes.
Independent Class C review checks numerical proof, resource closure, mutable/artifact controls and
final docs without duplicating successful suites; MAIN owns final acceptance/status closure.
No ADR/authority/architecture-test change is intended because no owner, edge or lifecycle changes.

## Result

- Implemented exact static typed-one SUM folding and `CpuConstantWriteIr`; private domain storage,
  copies/contributor loops/scratch removed, original scalar/nonscalar source initialization retained.
- Changed the ten named CPU production files, inventory and three new focused CPU tests, plus
  conformance/Engine fixtures; no shared production, API, architecture, build or Scan change.
- Frozen-source independent BOUNDED APPROVE, zero unresolved P1/P2; provisional intermediate-plan
  and trailing-zero overflow findings closed, one-unit finalizer regression covered.
- Focused command above plus `--tests '*CpuInternalPackageInventoryTest'`: PASS, 48 tests,
  0 failures/errors/skips (`EXEC-focused-7.log`); filtered scope, not a full/default module PASS.
- Pure integer oracle covers precision/low overflow and BF16 double rounding; actual CPU SUM,
  resource/source/carrier, ClassFile/direct-Java and selective identity controls passed.
- Independent Javadoc/CPU-guide pass uses General/Backend/API-Javadoc/Planning profiles; existing
  logical-splat/compile-constant/materialization meanings unchanged, no glossary entry needed.
- Final CPU/Javadoc PASS, 6m17s: 236 suites, 1461 tests, 1405 executed, 56 skipped, 0 failures/errors;
  94 existing Javadoc warnings, exactly two historical timing exclusions (`EXEC-final-cpu-gate.log`).
- MAIN named join PASS, exit 0/7s: count/dynamic-B conformance 4/4, Engine 6/6, 0 failures/errors/skips;
  command above plus `--max-workers=1 --console=plain` (`MAIN-named-join.log`). No tests rerun by reviewer.
- MAIN accepted protection: 2566 unchanged files/20 authorized paths, HEAD/index unchanged;
  diff check PASS. No benchmark claim, commit/push or automatic successor.
- Scope/hashes/logs: `/private/tmp/synaptik-cpu0048.G18rM6/`; exact final evidence `MAIN-VALIDATION.json`.

Status: Complete
