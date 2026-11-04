# Task Prepare0013: Ordinary dynamic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [CPU](../../../../glossary.md#term-cpu) forward and first-order backward

> **Editorial historical reading key.**
>
> This historical vertical implements one immutable symbolic-B recipe across fresh runs, not per-B
> recompilation. Compiler binds descriptor obligations; Prepare fixes declarations and slots; CPU
> binds numeric geometry; Runtime creates isolated run resources; Engine validates caller storage
> before borrowing and publishes actual descriptors. Caller storage remains usable through result
> close; preparation owns persistent resources, while materialized host values are detached. See
> [current metadata
> binding](../../../../reference/compiler/batch-binding.md#ordinary-named-batch-metadata-binding) and
> the [current
> lifecycle](../../../../reference/engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle).
>
> Preserve the recorded reused-test evidence and timing confounds; neither proves general performance
> parity or broader backend/workflow support.
>
> **Mathematical reading example.** In the all-one witness, let $B$ be a positive integer row count,
> $X\in\mathbb{R}^{B\times4}$ the input, $W\in\mathbb{R}^{4\times3}$ the weights, $Y=XW$, and $L$ the
> mean of all $3B$ entries of $Y$. In real arithmetic, each output is $4$, each output derivative is
> $1/(3B)$, each input derivative sums three contributions to $1/B$, and each weight derivative sums
> $B$ contributions to $1/3$. For $B=2$, these derivatives are $1/2$ and $1/3$. Actual floating-point
> evaluation retains the archive's rounding rules. This positive-domain explanation does not apply at
> $B=0$: empty MEAN, empty input gradients and nonempty zero weight gradients follow their separate
> policies.

## Status

Complete — MAIN accepted 2026-10-24; JAVA FROZEN 2026-10-24 19:34:25 UTC.
MAIN verified all dependencies Complete and accepted authority before the sole serial launch.
Final code/[Javadoc](../../../../glossary.md#term-javadoc)/performance and shared-document reviews APPROVE, zero open P1/P2;
MAIN's native-enabled root join passed. No successor is authorized.

## Change class

Class C — sizing/[admission](../../../../glossary.md#term-admission)/[publication](../../../../glossary.md#term-publication), [lifetime](../../../../glossary.md#term-lifetime) and generated code. One serial clean executor
and independent clean code/documentation review are mandatory.
Size exception: this cohesive cross-[owner](../../../../glossary.md#term-owner) vertical retains its exact type/file map and eight
acceptance criteria in one brief rather than splitting the resource [ABI](../../../../glossary.md#term-application-binary-interface-abi) and its consumers.

## Goal

Deliver one complete low-level [Engine](../../../../glossary.md#term-engine)+[CPU](../../../../glossary.md#term-cpu) capability: [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare) once with named symbolic B,
[FLOAT32](../../../../glossary.md#term-float32) grad-eligible X `[B,4]`, W `[4,3]`, publish Y=X@W and L=[mean](../../../../glossary.md#term-mean)(Y), and obtain actual first-order
dX/dW from L with [scalar](../../../../glossary.md#term-scalar)-one [seed](../../../../glossary.md#term-seed). Execute fresh values at 64→40→1→64, plus qualified B=0.
[Compiler](../../../../glossary.md#term-compiler) binds [metadata](../../../../glossary.md#term-metadata) → Prepare fixes [identities](../../../../glossary.md#term-identity-meanings) → CPU recipes → [Runtime](../../../../glossary.md#term-runtime) [run](../../../../glossary.md#term-run) → actual Engine results.

## [Scope](../../../../glossary.md#term-scope) and non-goals

- One named B, fixed features/parameters, rank ≤2 boundaries plus AD views; generated scalar,
  single-range CPU. Fixed-shape closure helpers remain legal.<span data-glossary-navigation="true"> ([features glossary](../../../../glossary.md#term-software-feature); [parameters glossary](../../../../glossary.md#term-parameter); [rank glossary](../../../../glossary.md#term-rank); [AD glossary](../../../../glossary.md#autograd); [views glossary](../../../../glossary.md#term-view-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [CPU glossary](../../../../glossary.md#term-cpu); [shape glossary](../../../../glossary.md#shape))</span>
- Qualify the actual final optimized closure, not just forward capability: MATMUL; full/selected SUM
  and MEAN including logical-one count SUM; DIV and required splats; EXPAND_DIMS, EXPAND, CONTIGUOUS,
  PERMUTE and SUM-to-Shape. Record surviving helpers after exact optimization; preserve AD formulas.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [MEAN glossary](../../../../glossary.md#term-mean); [CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [SUM-to-Shape glossary](../../../../glossary.md#sum-to-shape))</span>
- Keep existing partition/unit topology and Invocation; no new DAG, executor, session, public facade,
  B_max, per-B fullprepare/cache, representative/fake-B lowering, run-time Model lowerer or shared currentB.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [topology glossary](../../../../glossary.md#term-topology); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [facade glossary](../../../../glossary.md#term-facade); [cache glossary](../../../../glossary.md#term-cache); [lowering glossary](../../../../glossary.md#lowering); [run glossary](../../../../glossary.md#term-run); [Model glossary](../../../../glossary.md#term-model))</span>
- No Scan/NN/Training/cold/wire/native/mixed/wider-dtype/variable-T/higher-order/vector/parallel/tuning
  enablement or performance campaign. Retain existing paths; migrate static creator consumers.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [dtype glossary](../../../../glossary.md#data-type--datatype); [vector glossary](../../../../glossary.md#term-vector))</span>

## Contracts

Read AGENTS/root index and these headings; stop on conflict.

- `ARCHITECTURE.md`: Core lifecycle; Core invariants; Scope-indexed normative contracts;
  Module-ownership routing; Dependency rules; Testing requirements.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative); [Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Dependency glossary](../../../../glossary.md#term-dependency))</span>
- `docs/architecture/contracts/runtime-prepare-engine.md`: `modules/runtime`;
  Accepted symbolic-batch resource contract; <span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>`modules/prepare`; [Prepare](../../../../glossary.md#prepare) [lifecycle](../../../../glossary.md#lifecycle); [Run](../../../../glossary.md#term-run) lifecycle;
  `modules/engine`; Public reusable [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution).
- `docs/architecture/contracts/compiler-autograd.md`: `modules/compiler`; Compile artifacts;
  Accepted symbolic-batch metadata binding; Compiler-owned automatic differentiation.<span data-glossary-navigation="true"> ([Compile artifacts glossary](../../../../glossary.md#compile-artifacts); [metadata glossary](../../../../glossary.md#term-metadata); [binding glossary](../../../../glossary.md#term-binding); [Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>
- `docs/architecture/contracts/backend-execution.md`: CPU backend routes;
  Accepted ordinary symbolic-batch CPU contract.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
- `docs/architecture/contracts/foundational-modules.md`: Profile-free numerical semantics;
  Profile-free capability queries.<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>
- [Signature](../../../../glossary.md#term-signature) map: `docs/planning/modules/prepare/dynamic-batch-geometry-contract-proposal.md`,
  Exact proposed signatures and semantics; Complete type and replacement map; Concrete reuse and empty policy.

## Dependencies and integration

- Depends on: MAIN-accepted Prepare0012A; Complete Prepare0012, Prepare0011, Runtime0016,
  Engine0021, Compiler0013, CPU0037; MAIN verified at admission.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>
- Conflicts with: all geometry/creator ABI, Compiler bound predicates/layout, CPU lowering/identity/binder,
  Engine admission/publication and shared-doc writes; all writers serialized.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [Compiler glossary](../../../../glossary.md#term-compiler); [layout glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [identity glossary](../../../../glossary.md#term-identity-meanings); [Engine glossary](../../../../glossary.md#term-engine); [publication glossary](../../../../glossary.md#term-publication))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted dirty
  baseline plus MAIN-accepted Prepare0012A delta, not merged predecessors. Pre-Java snapshot:<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
  `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/pre-java.json`;
  preserve unrelated dirty work, index and HEAD.
- Integration order: accepted authority → verified Ready brief/MAIN admission → complete clean vertical
  → independent Class C code/docs review → MAIN executable-stable root join → bounded final A/B → acceptance.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Integration [validation](../../../../glossary.md#term-validation): [module](../../../../glossary.md#term-module) evidence/review and MAIN root architecture/[conformance](../../../../glossary.md#term-conformance)/integration join below.
- Shared-document integration owner: MAIN; worker drafts scoped docs/Javadoc, independent review finalizes;
  MAIN integrates shared plans/glossary. No commit/push or own agents.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Javadoc glossary](../../../../glossary.md#term-javadoc); [commit glossary](../../../../glossary.md#term-commit))</span>

## Exact added types and write map

Prefix `io.github.pho001.synaptik.`; source roots are the named [module](../../../../glossary.md#term-module)'s `src/main/java`.
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

- [Compiler](../../../../glossary.md#term-compiler) root: `{CompileDiagnostics,DeferredGraphConstraint,StaticResultLogicalLayoutClosure}.java`:
  private typed proof, symbolic layouts/lazy descriptors; retain count algebra and eight components.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- [Runtime](../../../../glossary.md#term-runtime): `memory/PreparedMemoryPlan.java`, `resource/PreparedRepresentationPlan.java`,
  `run/{RunState,RunStateCreation,PreparedExecutionRunner}.java`, `execution/PreparedExecutable.java`:
  exact reviewed overloads/mode, creator SAM replacement, state-aware compatibility and cleanup.
- [Prepare](../../../../glossary.md#prepare): `analysis/{PrepareContext,PreparationResourceRequirement}.java`,
  `{GraphPreparation,BackendPartitionFinalization,BackendPartitionFinalizationHandoff,PreparedScheduleContribution}.java`:
  symbolic mode/structural declarations; static max-size aggregation retained, no recipe evaluation.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- [CPU](../../../../glossary.md#term-cpu) root: `CpuCapabilityProvider.java`; internal `memory/CpuRepresentationRecipes.java`,
  `ir/{CpuAccessPlan,CpuMatmulIr,CpuAggregateIr,CpuAffineCopyIr,CpuPortableKernelIr}.java`,
  `lowering/{CpuPartitionLowering,CpuAffineLayoutLowering,CpuMatmulLowering,CpuAggregateLowering}.java`,
  `prepare/{CpuPartitionPreparationPlan,CpuPartitionPreparer,CpuPartitionFinalizer,CpuPreparedScheduleAssembler}.java`,
  `executable/{CpuPreparedExecutable,CpuPreparedPartitionExecutable}.java`,
  `route/portable/CpuPortableRoutePlan.java`, `cache/CpuKernelSpecialization.java`,
  `codegen/emit/{CpuClassFileKernelGenerator,CpuCarrierEmitter,CpuLoopEmitter,CpuAffineCopyEmitter,CpuMatmulEmitter,CpuAggregateEmitter}.java`:
  numeric units/fixed routes/scratch; retain CpuExactSumEmitter's five-limb code.<span data-glossary-navigation="true"> ([limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs))</span>
- [Engine](../../../../glossary.md#term-engine) root: `{Engine,PreparedExecution,AdvancedEngine,EngineBackendComposition,CpuEngineBackendComposition,MetalEngineBackendComposition,RunResult}.java`:
  retain signatures; metadata/preflight/actual descriptors and existing MemorySegmentStorage prefix
  borrowing. Metal rejects dynamic use; CompiledGraph/AdvancedRunResult/host exporter stay read-only.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [metadata glossary](../../../../glossary.md#term-metadata); [preflight glossary](../../../../glossary.md#term-preflight); [borrowing glossary](../../../../glossary.md#term-borrowing); [host glossary](../../../../glossary.md#term-host))</span>
- Mechanical static SAM migration: CPU factories/assembler above; [Metal](../../../../glossary.md#term-metal)
  `modules/backend/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedScheduleAssembler.java`, plus test
  creator consumers across Runtime/Prepare/CPU/Metal/Engine/testing; retain native semantics.
- Build: `modules/{backend/cpu,engine}/build.gradle.kts` only to forward the new test-only [boolean](../../../../glossary.md#data-type--datatype)
  `synaptik.cpu.dynamicBatch.performance` (default false); no [dependency](../../../../glossary.md#term-dependency)/configuration changes.
- Tests: extend existing CompileDiagnosticsTest, PrepareContextTest, BackendPartitionFinalizationHandoffTest,
  PreparedMemoryPlanTest, RunStateCreationTest, PreparedExecutableTest, CPU access/MATMUL/aggregate/
  preparation/cache tests and EngineTypedLifecycleTest. Add BatchBindingPlanTest, RunMemoryGeometryTest,
  CPU <span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [cache glossary](../../../../glossary.md#term-cache))</span>`internal.codegen.emit.{CpuOrdinaryDynamicBatchTest,CpuOrdinaryDynamicBatchCleanJavaOracle,CpuOrdinaryDynamicBatchTimingTest}`,
  Engine `{EngineOrdinaryDynamicBatchTest,EngineOrdinaryDynamicBatchTimingTest}`, [conformance](../../../../glossary.md#term-conformance)
  `CpuOrdinaryDynamicBatchConformanceTest` and integration `EngineCpuOrdinaryDynamicBatchIntegrationTest`.
  Update testing/architecture-tests inventories; precisely nine additions, no further production scope.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

1. Genuine symbolic compile/provider/Prepare admission for the final closure; no fake B.
   Compile/prepare/provider/generator counts, code identity, carriers/routes, slots and schedule remain
   unchanged across 64→40→1→64. Forward and actual first-order backward both execute.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [Prepare glossary](../../../../glossary.md#prepare); [admission glossary](../../../../glossary.md#term-admission))</span>
2. All-one values: Y=L=4, dX=1/B, dW=1/3 with existing primitive/rounding semantics. Actual publication
   Y bytes 768/480/12/768, dX 1024/640/16/1024, dW always 48, L always 4. Check against numeric
   references under existing rounding; use nonuniform values to expose stride/K errors.
   dW has M/K/N=4/B/3, never a B×parameter bank. Record actual retained helper/resource declarations.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [publication glossary](../../../../glossary.md#term-publication); [parameter glossary](../../../../glossary.md#term-parameter))</span>
3. B=0 gives empty Y/dX, scalar NaN empty MEAN and nonempty typed-zero dW; no whole-run early skip.
   Exercise actual AD count/expand/divide without reading absent dY. Check SUM-to-Shape B=1 entry
   copy/singleton and <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [NaN glossary](../../../../glossary.md#term-nan); [MEAN glossary](../../../../glossary.md#term-mean); [run glossary](../../../../glossary.md#term-run); [AD glossary](../../../../glossary.md#autograd); [SUM-to-Shape glossary](../../../../glossary.md#sum-to-shape); [singleton glossary](../../../../glossary.md#term-singleton))</span>`[F,0]` [strides](../../../../glossary.md#term-stride) `[0,1]`; same artifacts/slots. Exact F32 SUM/MEAN has five limbs
   for every admitted int-limited domain, including 0/1, with conditional 0/48 workspace bytes.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
4. Explicit requiresBatchSize remains true for all-[constant](../../../../glossary.md#term-constant)-byte `X[B,0].sum()`; absent-B runner,
   creation and RunState overloads reject before creator invocation/ownership acceptance. Check fixed
   constructors/accessors, run-bound accessor failure, present/absent-B mismatch and foreign-plan geometry.<span data-glossary-navigation="true"> ([RunState glossary](../../../../glossary.md#run-state--runstate); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
5. Pre-borrow semantic/numeric/storage checks cover consistent B including unused declarations; unknown/
   output-only symbols, predicates, rank/fixed axes/type/grad flags, missing/noncanonical/offset layouts,
   absent/dead/short/inaccessible storage and dense read views. CPU analysis proves write injectivity;
   no per-run enumeration, DAG traversal, inference or eager intermediate descriptors. Check int element/stride/span and
   long byte/aggregate/padding/multiplicity overflow, even with zero counts, before any borrow/allocation.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axes glossary](../../../../glossary.md#term-axis); [layouts glossary](../../../../glossary.md#layout); [views glossary](../../../../glossary.md#term-view-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [inference glossary](../../../../glossary.md#term-inference); [eager glossary](../../../../glossary.md#term-eager-evaluation); [descriptors glossary](../../../../glossary.md#term-descriptor); [element glossary](../../../../glossary.md#term-element); [padding glossary](../../../../glossary.md#padding); [overflow glossary](../../../../glossary.md#term-overflow))</span>
6. Exact usable prefix/bytes/alignment, fresh zero-sized objects and actual-count initialization;
   physical alignment/read-only/writable-overlap failures before execution. Inject allocation, null/duplicate
   creator, cold-bind and publication failures: no partial result/caller close; reverse attempt-all
   rollback, distinct suppression and successful retry. Retain lease/close precedence and materialization
   after prepared close. Different-B runs/results share no writable storage/mutable arrays. Gradient
   descriptors derive from generated values, not target flags; retain authenticated distinct aliases.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [rollback glossary](../../../../glossary.md#term-rollback); [lease glossary](../../../../glossary.md#term-lease); [materialization glossary](../../../../glossary.md#materialization); [mutable glossary](../../../../glossary.md#term-mutable); [Gradient glossary](../../../../glossary.md#term-gradient); [target glossary](../../../../glossary.md#term-target); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
7. Both [specialization](../../../../glossary.md#term-specialization) [identity](../../../../glossary.md#term-identity-meanings) streams mark changed baked meaning with `ordinary-batch-v1`, exclude
   actual B/bytes/slots/addresses and reject incompatible artifacts. Retain static/native keys/generated
   bytes/routes; dynamic false answers stay false outside the declared closure and carrier domain.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
8. Semantic tests plus Class-File inspection compare the optimal specialized clean Java algorithm and
   hot-loop/dataflow, including exact five-limb SUM/MEAN (never naive float accumulation). No hidden
   per-element recipe work, allocation/boxing/reflection/dispatch; direct typed cold-bound invocation.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [dispatch glossary](../../../../glossary.md#term-dispatch); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>

## [Validation](../../../../glossary.md#term-validation)

Use JDK26 and focused tests, then final [modules](../../../../glossary.md#term-module) once; [Javadoc](../../../../glossary.md#term-javadoc) after final reviewer edits:

```bash
./gradlew :modules:compiler:test :modules:runtime:test :modules:prepare:test :backends:cpu:test :modules:engine:test :backends:metal:compileTestJava :modules:compiler:javadoc :modules:runtime:javadoc :modules:prepare:javadoc :backends:cpu:javadoc :modules:engine:javadoc :backends:metal:javadoc --max-workers=2 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false
```

MAIN root join, existing [artifact](../../../../glossary.md#term-artifact)/no rebuild:

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false
```

MAIN verified [dylib](../../../../glossary.md#term-dylib) [SHA-256](../../../../glossary.md#term-sha-256) `03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a`.
MAIN root session23020 exited 0: BUILD SUCCESSFUL in 40s, 119 tasks (30 executed, 89 up-to-date).
Active settings-mapped XML totals: 694 suites, 4885 tests, zero failures/errors, 51 skips; counts
include up-to-date reuse, not entirely new execution. Exclude stale old Training72/Checkpoint57 XML.
Skips: 39 [CPU](../../../../glossary.md#term-cpu) and one [Engine](../../../../glossary.md#term-engine) performance opt-in, ten MetalPointwiseDispatchObserverNativeTest cases
requiring unset `SYNAPTIK_METAL_TEST_OBSERVER`, and
`EngineModelAutotuningIntegrationTest.absentCpuHandoffFallsBackAndReturnedPreparationRuns`'s assumption.
Configured MetalMixedTrainingSucceeds and ordinary [Metal](../../../../glossary.md#term-metal) native integration cases passed; no
separately verified Metal [checkpoint](../../../../glossary.md#term-checkpoint)-[resume](../../../../glossary.md#schema-serialization-and-checkpoint) claim. Root [log](../../../../glossary.md#term-log): `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/main-root-join.log`.
Final bounded A/B (property forwarded); no competing measurements:

```bash
./gradlew :backends:cpu:test --tests '*CpuOrdinaryDynamicBatchTimingTest' :modules:engine:test --tests '*EngineOrdinaryDynamicBatchTimingTest' --no-parallel --max-workers=1 --console=plain -Dsynaptik.cpu.dynamicBatch.performance=true
```

Tiny: B=1/40/64,F4,H3; medium: B64,F128,H128, all ones: [mean](../../../../glossary.md#term-mean)=128,dW=1/128,dX=1/B.
≤60s total including [warmup](../../../../glossary.md#term-warmup)/repeats; serial tasks/[forks](../../../../glossary.md#term-fork-meanings). Prefer 3 short independent serial [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) repeats
if budget permits; report JVM/options/forks/repeats, not in-JVM [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) mislabeled as forks.
Generated/direct match invocation B, carriers/algorithm/exact five-[limb](../../../../glossary.md#exact-superaccumulator-and-limbs) SUM/MEAN, never naive sums.
Static/dynamic match B, algorithm, [scalar](../../../../glossary.md#term-scalar) single-range route/carriers, allocations/result [lifetime](../../../../glossary.md#term-lifetime)/
[materialization](../../../../glossary.md#materialization); static [prepare](../../../../glossary.md#prepare) untimed. Separate cold generation/classload/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit), [admission](../../../../glossary.md#term-admission)/[kernel](../../../../glossary.md#kernel) costs;
report ratios/[variance](../../../../glossary.md#term-variance)/gaps. No [matrix](../../../../glossary.md#term-matrix-meanings) or identical-JIT/zero-overhead claim.
Timing-method [qualification](../../../../glossary.md#term-qualification): Engine F128/H128 retains its existing static tiled versus dynamic
direct selection as an explicitly UNMATCHED route-policy diagnostic; matched tiny rows plus
B64/F128/H1 assert equal actual direct realizations. Report allocation/[topology](../../../../glossary.md#term-topology) differences rather
than attributing end-to-end gaps solely to B propagation. CPU forced-direct F128/H128 stays matched.

## Documentation and review impact

Finalize affected [Javadoc](../../../../glossary.md#term-javadoc) (params/results/units/[nullability](../../../../glossary.md#term-nullability)/failures/[ownership](../../../../glossary.md#term-ownership-meanings)/threading/migration).
[Target](../../../../glossary.md#term-target) `docs/api/{compile-api,public-api}.md`,
`docs/backend-guide/{cpu-backend,partition-preparer}.md`, `docs/user-guide/running-models.md` and
`docs/architecture/runtime-prepare-backend-boundary.md`; synchronize implemented versus future support.
MAIN owns glossary/authority support-status/plans. Independent clean Class C code/docs review:
zero open P1/P2, source/Javadoc/examples/links/type inventory/static retention; reuse successful tests.
Validate final docs/Javadoc and `git diff --check`, including new-file whitespace.
Final shared integration covers the root, three affected scoped contracts, ADR0050, five glossary
entries, [Prepare](../../../../glossary.md#prepare) master plan, roadmap, dynamic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) plan and this brief; guides/Javadoc reuse writer evidence.

## Result

Implemented ordinary [FLOAT32](../../../../glossary.md#term-float32) forward/actual first-order backward, B=0 and public seeded [scalar](../../../../glossary.md#term-scalar) [loss](../../../../glossary.md#loss);
exactly nine production additions, no new [dependency](../../../../glossary.md#term-dependency), executor or semantics. The Y+L witness retains
16 units, 19 [buffers](../../../../glossary.md#term-buffer)/four workspaces across B; dW stays [parameter](../../../../glossary.md#term-parameter)-shaped. Six static units retain keys/bytes.
Final [CPU](../../../../glossary.md#term-cpu) command `./gradlew :backends:cpu:test --max-workers=2 --console=plain` with all three
performance flags false passed in 5m36s: 1243 tests/39 skips. Reused affected-[module](../../../../glossary.md#term-module) evidence:
[Compiler](../../../../glossary.md#term-compiler) 365, [Runtime](../../../../glossary.md#term-runtime) 159, [Prepare](../../../../glossary.md#prepare) 93, [Engine](../../../../glossary.md#term-engine) 141/1 skip; combined 2001 tests/40 skips, zero failures/errors.
Post-fix focused CPU 4, Engine 4 and public integration 1 pass, zero skips. Six [Javadoc](../../../../glossary.md#term-javadoc) tasks pass;
CPU regenerated after final edits. Guides/example reuse writer evidence; shared support status reconciled.
Bounded JDK26.0.1 timing uses one [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)/harness, three in-JVM [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), not [forks](../../../../glossary.md#term-fork-meanings); setup reported separately.
Original harness predates final [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) repair. One authorized post-fix probe passed numerical checks
in a 2.061s timed body (last [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) event 2.374s JVM uptime); recorded total bodies remain below 60s.
Probe generated/direct sample ratios: tiny MATMUL 0.968–1.337, medium 1.008–1.035, exact [MEAN](../../../../glossary.md#term-mean) 1.398–1.414.
Joint callsite/[warmup](../../../../glossary.md#term-warmup)/input confounds, original outliers, unmatched medium Engine routes and differing
allocation [topology](../../../../glossary.md#term-topology) remain; no general parity, negligible overhead, causal isolation or B-tax claim.
Final A/B and shared-document reviews APPROVE, zero open P1/P2; MAIN accepted 2026-10-24.
Native-enabled root passed in 40s: 4885 tests/51 qualified skips, zero failures/errors, with reuse
(30 tasks executed/89 up-to-date). Final links/whitespace, 69 frozen [hashes](../../../../glossary.md#term-hash), native [hash](../../../../glossary.md#term-hash) and [ownership](../../../../glossary.md#term-ownership-meanings) guards pass.
Exact worker commands, 70 changed paths, static/native hashes and caveats:
`/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/0013-completion.md`; 69 frozen hashes:
`/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/0013-source-hashes.json` (verified unchanged).
Status: Complete
