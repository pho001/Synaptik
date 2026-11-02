# Task Prepare0013: Ordinary dynamic-batch CPU forward and first-order backward

## Status

Complete — MAIN accepted 2026-10-24; JAVA FROZEN 2026-10-24 19:34:25 UTC.
MAIN verified all dependencies Complete and accepted authority before the sole serial launch.
Final code/Javadoc/performance and shared-document reviews APPROVE, zero open P1/P2;
MAIN's native-enabled root join passed. No successor is authorized.

## Change class

Class C — sizing/admission/publication, lifetime and generated code. One serial clean executor
and independent clean code/documentation review are mandatory.
Size exception: this cohesive cross-owner vertical retains its exact type/file map and eight
acceptance criteria in one brief rather than splitting the resource ABI and its consumers.

## Goal

Deliver one complete low-level Engine+CPU capability: compile/prepare once with named symbolic B,
FLOAT32 grad-eligible X `[B,4]`, W `[4,3]`, publish Y=X@W and L=mean(Y), and obtain actual first-order
dX/dW from L with scalar-one seed. Execute fresh values at 64→40→1→64, plus qualified B=0.
Compiler binds metadata → Prepare fixes identities → CPU recipes → Runtime run → actual Engine results.

## Scope and non-goals

- One named B, fixed features/parameters, rank ≤2 boundaries plus AD views; generated scalar,
  single-range CPU. Fixed-shape closure helpers remain legal.
- Qualify the actual final optimized closure, not just forward capability: MATMUL; full/selected SUM
  and MEAN including logical-one count SUM; DIV and required splats; EXPAND_DIMS, EXPAND, CONTIGUOUS,
  PERMUTE and SUM-to-Shape. Record surviving helpers after exact optimization; preserve AD formulas.
- Keep existing partition/unit topology and Invocation; no new DAG, executor, session, public facade,
  B_max, per-B fullprepare/cache, representative/fake-B lowering, run-time Model lowerer or shared currentB.
- No Scan/NN/Training/cold/wire/native/mixed/wider-dtype/variable-T/higher-order/vector/parallel/tuning
  enablement or performance campaign. Retain existing paths; migrate static creator consumers.

## Contracts

Read AGENTS/root index and these headings; stop on conflict.

- `ARCHITECTURE.md`: Core lifecycle; Core invariants; Scope-indexed normative contracts;
  Module-ownership routing; Dependency rules; Testing requirements.
- `docs/architecture/contracts/runtime-prepare-engine.md`: `modules/runtime`;
  Accepted symbolic-batch resource contract; `modules/prepare`; Prepare lifecycle; Run lifecycle;
  `modules/engine`; Public reusable prepared execution.
- `docs/architecture/contracts/compiler-autograd.md`: `modules/compiler`; Compile artifacts;
  Accepted symbolic-batch metadata binding; Compiler-owned automatic differentiation.
- `docs/architecture/contracts/backend-execution.md`: CPU backend routes;
  Accepted ordinary symbolic-batch CPU contract.
- `docs/architecture/contracts/foundational-modules.md`: Profile-free numerical semantics;
  Profile-free capability queries.
- Signature map: `docs/planning/modules/prepare/dynamic-batch-geometry-contract-proposal.md`,
  Exact proposed signatures and semantics; Complete type and replacement map; Concrete reuse and empty policy.

## Dependencies and integration

- Depends on: MAIN-accepted Prepare0012A; Complete Prepare0012, Prepare0011, Runtime0016,
  Engine0021, Compiler0013, CPU0037; MAIN verified at admission.
- Conflicts with: all geometry/creator ABI, Compiler bound predicates/layout, CPU lowering/identity/binder,
  Engine admission/publication and shared-doc writes; all writers serialized.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted dirty
  baseline plus MAIN-accepted Prepare0012A delta, not merged predecessors. Pre-Java snapshot:
  `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/pre-java.json`;
  preserve unrelated dirty work, index and HEAD.
- Integration order: accepted authority → verified Ready brief/MAIN admission → complete clean vertical
  → independent Class C code/docs review → MAIN executable-stable root join → bounded final A/B → acceptance.
- Integration validation: module evidence/review and MAIN root architecture/conformance/integration join below.
- Shared-document integration owner: MAIN; worker drafts scoped docs/Javadoc, independent review finalizes;
  MAIN integrates shared plans/glossary. No commit/push or own agents.

## Exact added types and write map

Prefix `io.github.pho001.synaptik.`; source roots are the named module's `src/main/java`.
Exactly nine additions, no moves/removals/edges:

1. `compiler.BatchBindingPlan` — public final/private construction.
2. `compiler.BatchBindingPlan.Binding` — public static final/private construction.
3. `runtime.memory.PreparedMemoryPlan.ByteSize` — public nested interface.
4. `runtime.memory.PreparedMemoryPlan.FixedByteSize` — public nested record.
5. `runtime.run.RunMemoryGeometry` — public final/private construction.
6. `backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize` — public nested record.
7. `backend.cpu.internal.ir.CpuAccessPlan.BatchBinding` — public nested record.
8. `backend.cpu.internal.lowering.CpuMatmulLowering.BatchGeometry` — public nested record.
9. `backend.cpu.internal.lowering.CpuAggregateLowering.BatchGeometry` — public nested record/ByteSize.

Allowed existing files (braces enumerate files):

- Compiler root: `{CompileDiagnostics,DeferredGraphConstraint,StaticResultLogicalLayoutClosure}.java`:
  private typed proof, symbolic layouts/lazy descriptors; retain count algebra and eight components.
- Runtime: `memory/PreparedMemoryPlan.java`, `resource/PreparedRepresentationPlan.java`,
  `run/{RunState,RunStateCreation,PreparedExecutionRunner}.java`, `execution/PreparedExecutable.java`:
  exact reviewed overloads/mode, creator SAM replacement, state-aware compatibility and cleanup.
- Prepare: `analysis/{PrepareContext,PreparationResourceRequirement}.java`,
  `{GraphPreparation,BackendPartitionFinalization,BackendPartitionFinalizationHandoff,PreparedScheduleContribution}.java`:
  symbolic mode/structural declarations; static max-size aggregation retained, no recipe evaluation.
- CPU root: `CpuCapabilityProvider.java`; internal `memory/CpuRepresentationRecipes.java`,
  `ir/{CpuAccessPlan,CpuMatmulIr,CpuAggregateIr,CpuAffineCopyIr,CpuPortableKernelIr}.java`,
  `lowering/{CpuPartitionLowering,CpuAffineLayoutLowering,CpuMatmulLowering,CpuAggregateLowering}.java`,
  `prepare/{CpuPartitionPreparationPlan,CpuPartitionPreparer,CpuPartitionFinalizer,CpuPreparedScheduleAssembler}.java`,
  `executable/{CpuPreparedExecutable,CpuPreparedPartitionExecutable}.java`,
  `route/portable/CpuPortableRoutePlan.java`, `cache/CpuKernelSpecialization.java`,
  `codegen/emit/{CpuClassFileKernelGenerator,CpuCarrierEmitter,CpuLoopEmitter,CpuAffineCopyEmitter,CpuMatmulEmitter,CpuAggregateEmitter}.java`:
  numeric units/fixed routes/scratch; retain CpuExactSumEmitter's five-limb code.
- Engine root: `{Engine,PreparedExecution,AdvancedEngine,EngineBackendComposition,CpuEngineBackendComposition,MetalEngineBackendComposition,RunResult}.java`:
  retain signatures; metadata/preflight/actual descriptors and existing MemorySegmentStorage prefix
  borrowing. Metal rejects dynamic use; CompiledGraph/AdvancedRunResult/host exporter stay read-only.
- Mechanical static SAM migration: CPU factories/assembler above; Metal
  `modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedScheduleAssembler.java`, plus test
  creator consumers across Runtime/Prepare/CPU/Metal/Engine/testing; retain native semantics.
- Build: `modules/{backend/cpu,engine}/build.gradle.kts` only to forward the new test-only boolean
  `synaptik.cpu.dynamicBatch.performance` (default false); no dependency/configuration changes.
- Tests: extend existing CompileDiagnosticsTest, PrepareContextTest, BackendPartitionFinalizationHandoffTest,
  PreparedMemoryPlanTest, RunStateCreationTest, PreparedExecutableTest, CPU access/MATMUL/aggregate/
  preparation/cache tests and EngineTypedLifecycleTest. Add BatchBindingPlanTest, RunMemoryGeometryTest,
  CPU `internal.codegen.emit.{CpuOrdinaryDynamicBatchTest,CpuOrdinaryDynamicBatchCleanJavaOracle,CpuOrdinaryDynamicBatchTimingTest}`,
  Engine `{EngineOrdinaryDynamicBatchTest,EngineOrdinaryDynamicBatchTimingTest}`, conformance
  `CpuOrdinaryDynamicBatchConformanceTest` and integration `EngineCpuOrdinaryDynamicBatchIntegrationTest`.
  Update testing/architecture-tests inventories; precisely nine additions, no further production scope.

## Acceptance criteria

1. Genuine symbolic compile/provider/Prepare admission for the final closure; no fake B.
   Compile/prepare/provider/generator counts, code identity, carriers/routes, slots and schedule remain
   unchanged across 64→40→1→64. Forward and actual first-order backward both execute.
2. All-one values: Y=L=4, dX=1/B, dW=1/3 with existing primitive/rounding semantics. Actual publication
   Y bytes 768/480/12/768, dX 1024/640/16/1024, dW always 48, L always 4. Check against numeric
   references under existing rounding; use nonuniform values to expose stride/K errors.
   dW has M/K/N=4/B/3, never a B×parameter bank. Record actual retained helper/resource declarations.
3. B=0 gives empty Y/dX, scalar NaN empty MEAN and nonempty typed-zero dW; no whole-run early skip.
   Exercise actual AD count/expand/divide without reading absent dY. Check SUM-to-Shape B=1 entry
   copy/singleton and `[F,0]` strides `[0,1]`; same artifacts/slots. Exact F32 SUM/MEAN has five limbs
   for every admitted int-limited domain, including 0/1, with conditional 0/48 workspace bytes.
4. Explicit requiresBatchSize remains true for all-constant-byte `X[B,0].sum()`; absent-B runner,
   creation and RunState overloads reject before creator invocation/ownership acceptance. Check fixed
   constructors/accessors, run-bound accessor failure, present/absent-B mismatch and foreign-plan geometry.
5. Pre-borrow semantic/numeric/storage checks cover consistent B including unused declarations; unknown/
   output-only symbols, predicates, rank/fixed axes/type/grad flags, missing/noncanonical/offset layouts,
   absent/dead/short/inaccessible storage and dense read views. CPU analysis proves write injectivity;
   no per-run enumeration, DAG traversal, inference or eager intermediate descriptors. Check int element/stride/span and
   long byte/aggregate/padding/multiplicity overflow, even with zero counts, before any borrow/allocation.
6. Exact usable prefix/bytes/alignment, fresh zero-sized objects and actual-count initialization;
   physical alignment/read-only/writable-overlap failures before execution. Inject allocation, null/duplicate
   creator, cold-bind and publication failures: no partial result/caller close; reverse attempt-all
   rollback, distinct suppression and successful retry. Retain lease/close precedence and materialization
   after prepared close. Different-B runs/results share no writable storage/mutable arrays. Gradient
   descriptors derive from generated values, not target flags; retain authenticated distinct aliases.
7. Both specialization identity streams mark changed baked meaning with `ordinary-batch-v1`, exclude
   actual B/bytes/slots/addresses and reject incompatible artifacts. Retain static/native keys/generated
   bytes/routes; dynamic false answers stay false outside the declared closure and carrier domain.
8. Semantic tests plus Class-File inspection compare the optimal specialized clean Java algorithm and
   hot-loop/dataflow, including exact five-limb SUM/MEAN (never naive float accumulation). No hidden
   per-element recipe work, allocation/boxing/reflection/dispatch; direct typed cold-bound invocation.

## Validation

Use JDK26 and focused tests, then final modules once; Javadoc after final reviewer edits:

```bash
./gradlew :modules:compiler:test :modules:runtime:test :modules:prepare:test :backends:cpu:test :modules:engine:test :backends:metal:compileTestJava :modules:compiler:javadoc :modules:runtime:javadoc :modules:prepare:javadoc :backends:cpu:javadoc :modules:engine:javadoc :backends:metal:javadoc --max-workers=2 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false
```

MAIN root join, existing artifact/no rebuild:

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false
```

MAIN verified dylib SHA-256 `03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a`.
MAIN root session23020 exited 0: BUILD SUCCESSFUL in 40s, 119 tasks (30 executed, 89 up-to-date).
Active settings-mapped XML totals: 694 suites, 4885 tests, zero failures/errors, 51 skips; counts
include up-to-date reuse, not entirely new execution. Exclude stale old Training72/Checkpoint57 XML.
Skips: 39 CPU and one Engine performance opt-in, ten MetalPointwiseDispatchObserverNativeTest cases
requiring unset `SYNAPTIK_METAL_TEST_OBSERVER`, and
`EngineModelAutotuningIntegrationTest.absentCpuHandoffFallsBackAndReturnedPreparationRuns`'s assumption.
Configured MetalMixedTrainingSucceeds and ordinary Metal native integration cases passed; no
separately verified Metal checkpoint-resume claim. Root log: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/main-root-join.log`.
Final bounded A/B (property forwarded); no competing measurements:

```bash
./gradlew :backends:cpu:test --tests '*CpuOrdinaryDynamicBatchTimingTest' :modules:engine:test --tests '*EngineOrdinaryDynamicBatchTimingTest' --no-parallel --max-workers=1 --console=plain -Dsynaptik.cpu.dynamicBatch.performance=true
```

Tiny: B=1/40/64,F4,H3; medium: B64,F128,H128, all ones: mean=128,dW=1/128,dX=1/B.
≤60s total including warmup/repeats; serial tasks/forks. Prefer 3 short independent serial JVM repeats
if budget permits; report JVM/options/forks/repeats, not in-JVM samples mislabeled as forks.
Generated/direct match invocation B, carriers/algorithm/exact five-limb SUM/MEAN, never naive sums.
Static/dynamic match B, algorithm, scalar single-range route/carriers, allocations/result lifetime/
materialization; static prepare untimed. Separate cold generation/classload/JIT, admission/kernel costs;
report ratios/variance/gaps. No matrix or identical-JIT/zero-overhead claim.
Timing-method qualification: Engine F128/H128 retains its existing static tiled versus dynamic
direct selection as an explicitly UNMATCHED route-policy diagnostic; matched tiny rows plus
B64/F128/H1 assert equal actual direct realizations. Report allocation/topology differences rather
than attributing end-to-end gaps solely to B propagation. CPU forced-direct F128/H128 stays matched.

## Documentation and review impact

Finalize affected Javadoc (params/results/units/nullability/failures/ownership/threading/migration).
Target `docs/api/{compile-api,public-api}.md`,
`docs/backend-guide/{cpu-backend,partition-preparer}.md`, `docs/user-guide/running-models.md` and
`docs/architecture/runtime-prepare-backend-boundary.md`; synchronize implemented versus future support.
MAIN owns glossary/authority support-status/plans. Independent clean Class C code/docs review:
zero open P1/P2, source/Javadoc/examples/links/type inventory/static retention; reuse successful tests.
Validate final docs/Javadoc and `git diff --check`, including new-file whitespace.
Final shared integration covers the root, three affected scoped contracts, ADR0050, five glossary
entries, Prepare master plan, roadmap, dynamic-batch plan and this brief; guides/Javadoc reuse writer evidence.

## Result

Implemented ordinary FLOAT32 forward/actual first-order backward, B=0 and public seeded scalar loss;
exactly nine production additions, no new dependency, executor or semantics. The Y+L witness retains
16 units, 19 buffers/four workspaces across B; dW stays parameter-shaped. Six static units retain keys/bytes.
Final CPU command `./gradlew :backends:cpu:test --max-workers=2 --console=plain` with all three
performance flags false passed in 5m36s: 1243 tests/39 skips. Reused affected-module evidence:
Compiler 365, Runtime 159, Prepare 93, Engine 141/1 skip; combined 2001 tests/40 skips, zero failures/errors.
Post-fix focused CPU 4, Engine 4 and public integration 1 pass, zero skips. Six Javadoc tasks pass;
CPU regenerated after final edits. Guides/example reuse writer evidence; shared support status reconciled.
Bounded JDK26.0.1 timing uses one JVM/harness, three in-JVM samples, not forks; setup reported separately.
Original harness predates final MATMUL repair. One authorized post-fix probe passed numerical checks
in a 2.061s timed body (last JIT event 2.374s JVM uptime); recorded total bodies remain below 60s.
Probe generated/direct sample ratios: tiny MATMUL 0.968–1.337, medium 1.008–1.035, exact MEAN 1.398–1.414.
Joint callsite/warmup/input confounds, original outliers, unmatched medium Engine routes and differing
allocation topology remain; no general parity, negligible overhead, causal isolation or B-tax claim.
Final A/B and shared-document reviews APPROVE, zero open P1/P2; MAIN accepted 2026-10-24.
Native-enabled root passed in 40s: 4885 tests/51 qualified skips, zero failures/errors, with reuse
(30 tasks executed/89 up-to-date). Final links/whitespace, 69 frozen hashes, native hash and ownership guards pass.
Exact worker commands, 70 changed paths, static/native hashes and caveats:
`/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/0013-completion.md`; 69 frozen hashes:
`/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/0013-source-hashes.json` (verified unchanged).
Status: Complete
