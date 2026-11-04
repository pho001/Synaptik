# Task CPU0048: Proven static typed-one count folding

## Reader context

`CpuAggregateLowering` consumes explicit immutable facts from `PrepareContext.constants()`,
not values sampled from a run's input storage. A leaf Tensor has no producing expression;
that alone does not mean it has storage or is constant. `CpuConstantWriteIr` describes output
stores, which finalization turns into code after slot assignment. Shared Prepare still retains
and initializes the original constant sources; only proved private expanded intermediates vanish.

For a finite static domain, let $D$ be the selected contributor coordinates for one output cell
and $n=|D|$. Exact positive-one contributions give $\sum_{i\in D}1=n$; the existing result
representation determines rounding or integral modular bits. A `[2,3]` domain reduced over its
second axis has $n=3$ twice, hence `[3,3]`. This is not a proof about mutable inputs containing ones.
See [static typed-one folding](../../../../backend-guide/cpu-backend.md#static-typed-one-count-folding)
for the source-initialization exception and direct-to-result rounding.

## Status

Complete — MAIN accepted 2026-10-31 after independent source/docs approval and final [CPU](../../../../glossary.md#term-cpu)/[Javadoc](../../../../glossary.md#term-javadoc)
and named join PASS. CPU frontier closed; CPU0049–0052 stay Draft/held.

## Change class

Class C — [CPU](../../../../glossary.md#term-cpu)-private generated [dataflow](../../../../glossary.md#term-dataflow)/resource elimination; clean implementation and
independent code/numerical/resource/[Javadoc](../../../../glossary.md#term-javadoc)/documentation review.

## Goal and [scope](../../../../glossary.md#term-scope)

Replace a proved static SUM of exact typed positive ones with the same represented count,
without executing private expansion/[materialization](../../../../glossary.md#materialization)/reduction work. A [logical splat](../../../../glossary.md#logical-splat) means one
[immutable](../../../../glossary.md#term-immutable) typed [scalar](../../../../glossary.md#term-scalar) repeated at every logical [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), not a sampled [buffer](../../../../glossary.md#term-buffer) value.

- Use only explicit (EXPLICIT) exact typed facts already projected in `PrepareContext.constants()`.
  Recognize direct splats and same-dtype <span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype))</span>`EXPAND` with optional [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) before ordinary SUM.
  All relevant source/domain/result shapes must be fully static; do not specialize a sampled B.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape))</span>
- Cover supported FLOAT32/FLOAT64/BFLOAT16/FLOAT16/INT32/INT64 SUM full, single/multiple-axis,
  keep/remove-dimension and SUM-to-Shape forms using existing normalized-axis rules.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [axis glossary](../../../../glossary.md#term-axis); [dimension glossary](../../../../glossary.md#dimension); [SUM-to-Shape glossary](../../../../glossary.md#sum-to-shape))</span>
- Preserve output shape/layout, typed empty/point semantics, modular integral results and the
  current CPU once-rounded exact floating SUM result, including low narrowing and overflow.
  This is a count proof, not general constant propagation or a new reduction algorithm.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu); [narrowing glossary](../../../../glossary.md#term-narrowing); [overflow glossary](../../../../glossary.md#term-overflow); [constant glossary](../../../../glossary.md#term-constant))</span>

## Non-goals

No [Compiler](../../../../glossary.md#term-compiler)/[Model](../../../../glossary.md#term-model)/shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[Engine](../../../../glossary.md#term-engine) production change, supported [API](../../../../glossary.md#term-application-programming-interface-api), [module](../../../../glossary.md#term-module) edge,
[lifecycle](../../../../glossary.md#lifecycle), native [ABI](../../../../glossary.md#term-application-binary-interface-abi), [Scan](../../../../glossary.md#term-cumulative-scan)-body [lowering](../../../../glossary.md#lowering)/tuning, new capability, general affine propagation,
masked/data-dependent counts, other [constants](../../../../glossary.md#term-constant)/reductions, benchmark campaign or speed/parity claim.
No build edits, historical timing selectors, automatic successor, stage/[commit](../../../../glossary.md#term-commit)/push or rollout.

## Contracts

- [Root: Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span> and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Backend](../../../../glossary.md#backend) [owner](../../../../glossary.md#term-owner): [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  — CPU-local lowering/resource declaration; finalization generates only after slot assignment.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- Numerical owner: [Exact and discrete floor](../../../../architecture/contracts/foundational-modules.md#exact-and-discrete-floor),
  [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives)
  and [Aggregate floor](../../../../architecture/contracts/foundational-modules.md#aggregate-floor)<span data-glossary-navigation="true"> ([Aggregate floor glossary](../../../../glossary.md#term-aggregate-floor))</span>
  — mapping, contributor membership, empty identities and rounding remain intact.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding))</span>
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>
  — explicit splat provenance, ordinary count algebra and saved/publication boundaries remain unchanged.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [publication glossary](../../../../glossary.md#term-publication))</span>
- [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>
  and [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  — immutable projected facts and exact declarations before slots; no handoff/owner change.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [handoff glossary](../../../../glossary.md#term-handoff))</span>

Current source/tests refine [CPU](../../../../glossary.md#term-cpu) algorithms, not [Model](../../../../glossary.md#term-model) authority; stop on a missing/conflicting contract.

## Dependencies and integration

- Depends on: CPU0047 Complete; accepted CPU0076 Complete as the current integrated [CPU](../../../../glossary.md#term-cpu) baseline.
- Conflicts with: CPU lowering/portable IR/emission/cache/resource writers; CPU0049–0052; shared
  CPU guide/master/roadmap writers. Writes serialized; final docs delegated exclusively by MAIN.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [cache glossary](../../../../glossary.md#term-cache))</span>
- Parallel group: None; serial implementation and independent read-only review.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty tree;
  MAIN's 2580-file `/private/tmp/synaptik-cpu0048.G18rM6/MAIN-BASELINE.json` is the conservation base.
- Integration order: accepted CPU0047/CPU0076 → CPU0048 implementation/focused checks → independent
  review and final Javadocs → executor's one final CPU test/Javadoc → MAIN named consumer join/acceptance.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Integration validation: the named conformance/Engine checkpoint below; MAIN conserves accepted
  unrelated work using its baseline and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`before-cpu.tar`. No full-root or performance campaign.
- Shared-document integration owner: MAIN; independent reviewer proposes/finalizes targeted docs
  under exclusive delegation, reusing executable evidence. Workers do not edit master/roadmap.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

Launch verified by MAIN, 2026-10-31: [Ready](../../../../glossary.md#term-ready) frontier, Complete predecessors/reviews/gates, contracts,
unchanged HEAD/index, protected dirty base and inactive conflicts; CPU0048-only GO after plan APPROVE.

## Source-grounded seam and files

[CPU](../../../../glossary.md#term-cpu) paths below are relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.

- `internal/lowering/CpuAggregateLowering.java`: reuse `normalizedAxes`/existing geometry checks;
  prove every contributor is typed +1 through the bounded chain and compute exact count bits cold.
  `CpuPartitionLowering.lower` dispatches that proof before ordinary family [lowering](../../../../glossary.md#lowering).
- `internal/lowering/CpuPartitionDagDecomposer.java`: select the complete private chain as one seed
  before ordinary affine/aggregate seeds; retain node membership/dependencies and original value IDs.
  Use full DAG consumers and logical-memory publication/external-consumer obligations, not adjacency
  alone. Refuse absorption across a public, saved, fan-out or outside-consumer intermediate.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [node glossary](../../../../glossary.md#node); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [publication glossary](../../../../glossary.md#term-publication); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- Added `internal/ir/CpuConstantWriteIr.java` in package
  `io.github.pho001.synaptik.backend.cpu.internal.ir`: exact typed `CpuKernelIr.ScalarImmediate`
  bits and structural output writes, without computational inputs, contributor loops or scratch.
- Wired through existing `internal/ir/CpuPortableKernelIr.java`,
  `CpuPartitionLowering.LoweredPartition.kernelIr`, `internal/route/portable/CpuPortableRoutePlan.java`,
  `internal/prepare/CpuPartitionPreparer.java` and `CpuPartitionPreparationPlan.java` only as needed.
  Existing `internal/codegen/emit/CpuClassFileKernelGenerator.java` emits direct typed stores with
  existing carrier/loop machinery; no new emitter/executor/framework or public abstraction.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `internal/prepare/CpuPartitionFinalizer.java`: MAIN-authorized existing-composite predicate for
  a single constant-write unit; ValueId-based selection keeps original sources out of its entry.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [ValueId glossary](../../../../glossary.md#logical-valueid-versus-physical-memory-slot))</span>
- `internal/cache/CpuKernelSpecialization.java` unchanged: the IR fingerprint keys baked typed
  literals/code shape in both identity streams; retain unaffected/fallback keys and ClassFiles.
  No global schema/projection bump; affected-versus-unaffected controls remain required.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings); [schema glossary](../../../../glossary.md#term-schema))</span>
- Tests under the corresponding CPU test root: new `internal/lowering/CpuStaticTypedOneCountFoldingTest.java`,
  `internal/prepare/CpuStaticTypedOneCountPreparationTest.java` and
  `internal/codegen/emit/CpuConstantWriteGeneratedKernelTest.java`; reuse aggregate/DAG fixtures and
  independent exact rational/integral oracles. Update `CpuInternalPackageInventoryTest` for the one type.
- Add `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuStaticTypedOneCountConformanceTest.java`;
  extend `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineCpuComposedValidationIntegrationTest.java`.
  `LossGradientRules.restore` and `NormalizationGradientRules.count` in Compiler are read-only witnesses:
  respectively <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>`oneBase.expand(domain).contiguous().sum()` and `oneLike(tensor).sum(axes,true)`.

## Acceptance criteria

- Static explicit +1 witnesses fold, including actual Compiler-generated mean-loss and normalization
  gradients. Full/axis/multi-axis/SUM-to-Shape, scalar, selected-zero, unselected-zero, empty-axis and
  domain-one cases preserve exact represented results and output metadata, including public outputs.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [mean glossary](../../../../glossary.md#term-mean); [loss glossary](../../../../glossary.md#loss); [normalization glossary](../../../../glossary.md#term-normalization); [gradients glossary](../../../../glossary.md#term-gradient); [axis glossary](../../../../glossary.md#term-axis); [SUM-to-Shape glossary](../../../../glossary.md#sum-to-shape); [scalar glossary](../../../../glossary.md#term-scalar); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Check ties-to-even around BF16 256/258, F16 2048/2050 and F16 finite/overflow boundary, and F32/F64
  precision boundaries; test integral modular counts and checked shape/count overflow behavior.
  Large-count metadata/oracle tests must not allocate huge arrays or iterate billions of contributors.
  A host <span data-glossary-navigation="true"> ([ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [BF16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [finite glossary](../../../../glossary.md#term-finite); [overflow glossary](../../../../glossary.md#term-overflow); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [precision glossary](../../../../glossary.md#term-precision); [shape glossary](../../../../glossary.md#shape); [oracle glossary](../../../../glossary.md#term-test-oracle); [host glossary](../../../../glossary.md#term-host))</span>`elementCount` cast is insufficient without proof of identical current-route [rounding](../../../../glossary.md#term-rounding).
- Inspect analysis declarations AND prepared memory/units AND generated ClassFiles: absorbed private
  EXPAND/CONTIGUOUS values have no buffer/fill/copy/executable; SUM has no contributor loop or exact-state
  workspace. Constant writes use only required output stores, no runtime input read/dummy arithmetic,
  hidden helper dispatch, boxing, reflection or per-element allocation. Compare with clean direct
  Java specialized constant stores; prove removed work structurally, without a timing claim.<span data-glossary-navigation="true"> ([CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [buffer glossary](../../../../glossary.md#term-buffer); [Constant glossary](../../../../glossary.md#term-constant); [runtime glossary](../../../../glossary.md#term-runtime); [dispatch glossary](../../../../glossary.md#term-dispatch); [element glossary](../../../../glossary.md#term-element))</span>
- Preserve required original source declarations/initialized representations: [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation)
  validates every compile constant source. A scalar +1 source may remain initialized; it is not
  the private domain-sized expansion. A nonscalar explicit source's mandated initialization also
  remains. Do not erase source obligations or claim their bytes/fill were removed.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- Public/saved/fan-out/external intermediates remain materialized via unchanged fallback; final
  count publication and downstream uses remain valid. Test barriers at each absorbed stage.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [publication glossary](../../../../glossary.md#term-publication))</span>
- Negative controls: absent facts; mutable host leaves initially full of ones; same shape/label/
  factory or equal-looking Tensor; non-one/wrong-typed facts; casts/arithmetic/masks; unsupported
  layouts/topologies. No payload read, leaf equality or mutable-value assumption proves constancy.
  Change actual storage/fresh run inputs after preparation and verify recomputation/current results.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [Tensor glossary](../../../../glossary.md#tensor); [masks glossary](../../../../glossary.md#term-boolean-mask); [layouts glossary](../../../../glossary.md#layout); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [run glossary](../../../../glossary.md#term-run))</span>
- Dynamic/named-B or otherwise unproved counts keep their current count algebra, resource topology,
  admission/failure behavior and reusable run binding. Existing Scan paths/defaults remain untouched.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [admission glossary](../../../../glossary.md#term-admission); [binding glossary](../../../../glossary.md#term-binding); [Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>

## [Validation](../../../../glossary.md#term-validation)

Executor owns focused checks and one final affected-[module](../../../../glossary.md#term-module) test/[Javadoc](../../../../glossary.md#term-javadoc) gate after final docs.
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

MAIN runs this named join once after integration, with the same [JDK](../../../../glossary.md#term-java-development-kit-jdk)/offline/no-performance environment:

```bash
./gradlew --offline --no-daemon :testing:backend-conformance:test \
  --tests '*CpuStaticTypedOneCountConformanceTest' --tests '*CpuOrdinaryDynamicBatchConformanceTest' \
  :testing:integration-tests:test --tests '*EngineCpuComposedValidationIntegrationTest'
git diff --check
```

## Documentation and review impact

Update affected implementation/type Javadocs and `docs/backend-guide/cpu-backend.md`'s ordinary
aggregate explanation in the same change; explain [constant](../../../../glossary.md#term-constant) proof, resource removal/source exception,
observable barriers and fallback. Apply General/[Backend](../../../../glossary.md#backend)/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc) profiles; glossary search for
[logical splat](../../../../glossary.md#logical-splat)/[compile](../../../../glossary.md#compile) constant/[materialization](../../../../glossary.md#materialization) requires no new term unless meaning actually changes.
Independent Class C review checks numerical proof, resource closure, [mutable](../../../../glossary.md#term-mutable)/[artifact](../../../../glossary.md#term-artifact) controls and
final docs without duplicating successful suites; MAIN owns final acceptance/status closure.
No [ADR](../../../../glossary.md#term-architecture-decision-record-adr)/authority/architecture-test change is intended because no [owner](../../../../glossary.md#term-owner), edge or [lifecycle](../../../../glossary.md#lifecycle) changes.

## Result

- Implemented exact static typed-one SUM folding and `CpuConstantWriteIr`; private domain storage,
  copies/contributor loops/scratch removed, original scalar/nonscalar source initialization retained.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Changed the ten named CPU production files, inventory and three new focused CPU tests, plus
  conformance/Engine fixtures; no shared production, API, architecture, build or Scan change.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Frozen-source independent BOUNDED APPROVE, zero unresolved P1/P2; provisional intermediate-plan
  and trailing-zero overflow findings closed, one-unit finalizer regression covered.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [regression glossary](../../../../glossary.md#term-regression))</span>
- Focused command above plus `--tests '*CpuInternalPackageInventoryTest'`: PASS, 48 tests,
  0 failures/errors/skips (`EXEC-focused-7.log`); filtered [scope](../../../../glossary.md#term-scope), not a full/default [module](../../../../glossary.md#term-module) PASS.
- Pure integer oracle covers precision/low overflow and BF16 double rounding; actual CPU SUM,
  resource/source/carrier, ClassFile/direct-Java and selective identity controls passed.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [precision glossary](../../../../glossary.md#term-precision); [BF16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [rounding glossary](../../../../glossary.md#term-rounding); [carrier glossary](../../../../glossary.md#term-storage-carrier); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Independent Javadoc/CPU-guide pass uses General/Backend/API-Javadoc/Planning profiles; existing
  logical-splat/compile-constant/materialization meanings unchanged, no glossary entry needed.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning); [compile glossary](../../../../glossary.md#compile); [constant glossary](../../../../glossary.md#term-constant); [materialization glossary](../../../../glossary.md#materialization))</span>
- Final CPU/Javadoc PASS, 6m17s: 236 suites, 1461 tests, 1405 executed, 56 skipped, 0 failures/errors;
  94 existing Javadoc warnings, exactly two historical timing exclusions (`EXEC-final-cpu-gate.log`).
- MAIN named join PASS, exit 0/7s: count/dynamic-B conformance 4/4, Engine 6/6, 0 failures/errors/skips;
  command above plus `--max-workers=1 --console=plain` (`MAIN-named-join.log`). No tests rerun by reviewer.
- MAIN accepted protection: 2566 unchanged files/20 authorized paths, HEAD/index unchanged;
  diff check PASS. No benchmark claim, commit/push or automatic successor.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>
- Scope/[hashes](../../../../glossary.md#term-hash)/[logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings): `/private/tmp/synaptik-cpu0048.G18rM6/`; exact final evidence `MAIN-VALIDATION.json`.

Status: Complete
