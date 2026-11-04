# Task 0018: Explicit Numerical-Profile Propagation Spine

## Status

Complete

Implementation `ce7a7dfa67c0dce5cd7b41b2a5bc90ae305393c3` plus documentation/[Javadoc](../../../../glossary.md#term-javadoc)
remediation `07a01b9c13c2ceea0922b9af82da3e6e08375306` passed the required [module](../../../../glossary.md#term-module)
test/Javadoc [checkpoint](../../../../glossary.md#term-checkpoint), [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), architecture tests, the actual
`EngineNumericalProfileIntegrationTest`, full build, documentation [validation](../../../../glossary.md#term-validation), and diff checks.
Independent Class C review returned final `APPROVE` with zero findings.

## Change class

Class C — one public [Engine](../../../../glossary.md#term-engine) construction option, one clean-cutover public advanced factory, a
[Planning](../../../../glossary.md#term-planning) [API](../../../../glossary.md#term-application-programming-interface-api)/[dependency edge](../../../../glossary.md#term-dependency-edge), three cross-[module](../../../../glossary.md#term-module) [immutable](../../../../glossary.md#term-immutable) records, and [backend](../../../../glossary.md#backend) plan/[cache](../../../../glossary.md#term-cache)
[identity](../../../../glossary.md#term-identity-meanings) change together. A clean implementation context and independent targeted API,
documentation, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), and cache review are mandatory.

## Goal and exact [API](../../../../glossary.md#term-application-programming-interface-api)

Transport one [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph)-wide `NumericalProfile` from [Engine](../../../../glossary.md#term-engine) construction through every
ordinary and advanced forward/backward [compile](../../../../glossary.md#compile), profile-qualified [ownership](../../../../glossary.md#term-ownership-meanings) query, nine-component
[Compiler](../../../../glossary.md#term-compiler) [artifacts](../../../../glossary.md#term-artifact), six-component [Prepare](../../../../glossary.md#prepare) projection, and [backend](../../../../glossary.md#backend) plan/[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings). Make no
numerical result or relaxed capability current.

The clean-cutover [shapes](../../../../glossary.md#shape) are exact:

```java
public record OperationCapabilityQuery(NumericalProfile numericalProfile, Operation operation,
        List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) { ... }
public static CompileArtifacts compile(
        CompileMode mode, NumericalProfile numericalProfile, List<Tensor> forwardOutputs,
        Optional<FunctionalGradientRequest> functionalGradientRequest,
        GraphOptimizationConfig optimizationConfig, BackendIntent backendIntent,
        PartitionScoringConfig partitionScoringConfig, List<BackendCapabilityProvider> capabilityProviders,
        List<BackendAvailabilitySnapshot> availabilitySnapshots)
public record CompileArtifacts(CompileMode mode, NumericalProfile numericalProfile,
        CompiledGraphModel graph, List<PlannedPartition> partitions, LogicalMemoryPlan memory,
        PublicationPlan publication, CompileConstantPlan constants, CompileDiagnostics diagnostics,
        DerivativeGraphMetadata derivatives) { ... }
public record PrepareContext<I extends BackendAnalysisInputs>(
        NumericalProfile numericalProfile, PartitionDag partitionDag, List<GraphValue> values,
        List<LogicalMemoryRequirement> memoryRequirements, Map<ValueId, ScalarValue> constants,
        I backendInputs) { ... }
public synchronized Engine.Builder numericalProfile(NumericalProfile numericalProfile)
public static AdvancedEngine takeOwnership(CpuBackendIntegration cpuIntegration, NumericalProfile numericalProfile)
```

[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext)'s [partition](../../../../glossary.md#partition)/[node](../../../../glossary.md#node) convenience constructor becomes seven [arguments](../../../../glossary.md#term-argument) with profile
first. `GraphCompiler`'s complete package-private entry adds profile immediately after mode.
There are no old-[arity](../../../../glossary.md#term-arity) overloads, one-argument advanced factory, [aliases](../../../../glossary.md#aliasing-and-arena), or hidden defaults.

## [Lifecycle](../../../../glossary.md#lifecycle), null, and default contract

- A fresh `Engine.Builder` holds `STRICT_IEEE`. `numericalProfile` rejects null with message
  `numericalProfile` before builder-state validation, returns the same open builder, may be called
  before or after registrations, and uses the last explicit value. Spent/closed builders reject a
  non-null call without mutation. <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`build()` [captures](../../../../glossary.md#term-capture-meanings) the value exactly once into the new [Engine](../../../../glossary.md#term-engine).
- `Engine.standard()` keeps its current public shape and builder ownership path and is strict by
  that default. The profile is immutable for the Engine lifetime and cannot vary by graph,
  compile, prepare, session, tuning trial, or run.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [lifetime glossary](../../../../glossary.md#term-lifetime); [graph glossary](../../../../glossary.md#graph); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run))</span>
- `AdvancedEngine.takeOwnership(integration, profile)` validates `cpuIntegration` then
  `numericalProfile`; null transfers no ownership. Every package-private constructor/test seam also
  requires an explicit profile. Advanced `compile(...)` and every ordinary forward, explicit-seed
  backward, scalar-objective backward, one-shot, tuning, and session path use the captured field.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective))</span>
- New record components are non-null in declaration order, retain the enum singleton, and
  participate in record equality, hashing, and text. Compiler and Prepare retain the exact value;
  neither interprets it. No Runtime, RunState, prepared-execution, run-result, session, or Trace
  production type gains a profile field or per-run lookup.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [RunState glossary](../../../../glossary.md#run-state--runstate); [Trace glossary](../../../../glossary.md#trace); [lookup glossary](../../../../glossary.md#term-lookup))</span>

## [Backend identity](../../../../glossary.md#backend-identity--backendid) and [fail-closed](../../../../glossary.md#term-fail-closed) boundary

- `CpuCapabilityProvider` and `MetalCapabilityProvider` return false for every [`ACCELERATOR`](../../../../glossary.md#term-accelerator) query
  before existing occurrence checks. Both preparers independently reject a non-strict context
  before route/candidate construction. Existing strict matrices and owner selection are unchanged.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [matrices glossary](../../../../glossary.md#term-matrix-meanings); [owner glossary](../../../../glossary.md#term-owner))</span>
- Add `NumericalProfile numericalProfile` as the first `CpuPartitionPreparationPlan` component and require every unit [specialization](../../../../glossary.md#term-specialization) to agree. Replace `CpuKernelSpecialization.NumericalMode` and
  `CpuOpenBlasTuningBatch.NumericalMode` in their current component positions outright with
  `NumericalProfile`; production creates only strict values. Profile enters specialization class
  identity/compatibility, OpenBLAS workload identity, local tuning equality/compatibility, and
  complete-plan projection/variant identity. Use stable names or backend-local wire values, never
  ordinal. Advance generator envelope 66→67, OpenBLAS candidate/policy 1→2, and local/complete
  value schemas 1→2; old and cross-profile decisions are misses/rejections.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [backend glossary](../../../../glossary.md#backend); [schemas glossary](../../../../glossary.md#term-schema))</span>
- Give `MetalNegPreparationPlan` a final profile, first constructor [argument](../../../../glossary.md#term-argument), and accessor; require plan/context agreement. Add profile after route-policy version in tuning `Compatibility` and to
  the workload digest with an explicit wire value. Advance workload, candidate, compatibility,
  route-policy, and codec versions 4→5; native ABI and MPSGraph node schema stay 4. Old and
  cross-profile decisions fail closed.<span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest); [wire value glossary](../../../../glossary.md#term-wire-value); [codec glossary](../../../../glossary.md#term-codec); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [node glossary](../../../../glossary.md#node); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- These changes transport and isolate [identity](../../../../glossary.md#term-identity-meanings) only. Do not enable an `ACCELERATOR` CPU or Metal
  occurrence, alter generated/native arithmetic, choose fast math, or reinterpret Model semantics.
  CPU 0017 and Metal 0015 were the sole Draft semantic-realization owners during this task.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [Model glossary](../../../../glossary.md#term-model))</span>

## Dependencies and integration

- Depends on: [Model](../../../../glossary.md#term-model) 0027; Config 0006 at `37e9e9db0bdb0b11564e3696079b0d6a5e0c8109`
- Conflicts with: Planning capability-query/API dependency, Compiler artifacts/port, Prepare
  context/projection, Engine construction/compile lifecycle, CPU or Metal plan/cache identity<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [Compiler glossary](../../../../glossary.md#term-compiler); [artifacts glossary](../../../../glossary.md#term-artifact); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [compile glossary](../../../../glossary.md#compile); [lifecycle glossary](../../../../glossary.md#lifecycle); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Parallel group: None
- Common base revision: `37e9e9db0bdb0b11564e3696079b0d6a5e0c8109`
- Integration order: one [atomic](../../../../glossary.md#resource-transaction-and-rollback) clean cutover; no partial merge
- Integration [validation](../../../../glossary.md#term-validation): numerical-profile propagation spine [checkpoint](../../../../glossary.md#term-checkpoint)
- Shared-document integration [owner](../../../../glossary.md#term-owner): Main planner

## Exact implementation and callsite inventory

Production/[API](../../../../glossary.md#term-application-programming-interface-api) [owners](../../../../glossary.md#term-owner):

- `modules/planning/build.gradle.kts`,
  `modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/OperationCapabilityQuery.java`,
  and `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/PlanningDependencyContractTest.java`.
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/`:
  `GraphCompilationPort.java`, `GraphCompiler.java`, and `CompileArtifacts.java`.
- `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/`: `GraphPreparation.java` and
  `analysis/PrepareContext.java`.
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/`: `Engine.java` and
  `AdvancedEngine.java`; update all four port calls and every package-private constructor.
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`:
  `CpuCapabilityProvider.java`, `CpuLocalWorkloadTuning.java`, `CpuCompletePlanTuning.java`,
  `internal/prepare/{CpuPartitionPreparer,CpuPartitionPreparationPlan}.java`,
  `internal/cache/{CpuKernelSpecialization,CpuGeneratorSchema}.java`,
  `internal/lowering/{CpuRepresentationPlanner,CpuConv1dCompositionLowering,
  CpuPool1dCompositionLowering,CpuPartitionDagDecomposer}.java`,
  `internal/route/nativeblas/openblas/{CpuBackendComposition,CpuOpenBlasRouteSelector,
  CpuOpenBlasTuningBatch}.java`, and every production query constructor in `internal/lowering/`.
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/`:
  `MetalCapabilityProvider.java`, `MetalNegPartitionPreparer.java`, `MetalNegPreparationPlan.java`,
  `MetalNegRouteCandidateGenerator.java`, `MetalNegTuningBatch.java`, `MetalNegTuningCodec.java`,
  and `MetalNegTuningDecision.java`.

Exhaustive clean-cutover callers are every base-revision match for `new OperationCapabilityQuery`,
`GraphCompilationPort.compile`, `new CompileArtifacts`, `new PrepareContext`, `new AdvancedEngine`,
and `AdvancedEngine.takeOwnership` under executable `modules/**`, `backends/**`, `testing/**`, and
[backend](../../../../glossary.md#backend) evidence. Production projections preserve their source profile; standalone tests/evidence
pass `STRICT_IEEE` unless explicitly testing [identity](../../../../glossary.md#term-identity-meanings). Update [shape](../../../../glossary.md#shape) tests
`BackendCapabilityContractTest`, `GraphCompilationPortTest`/`GraphCompilationPortPublicShapeTest`,
`CompileArtifactsTest`, `PrepareContextTest`, `AdvancedEnginePublicShapeTest`,
`EngineTypedPublicShapeTest`, and `PlanningDependencyContractTest`. Search those executable roots
for zero old-[arity](../../../../glossary.md#term-arity) calls and zero `NumericalMode` declarations/uses; completed briefs remain
historical evidence, not callers.

## Migration order

1. Change [Planning](../../../../glossary.md#term-planning)'s Config edge to `api`, query [shape](../../../../glossary.md#shape)/tests, and all explicit query callers.
2. Change [Compiler](../../../../glossary.md#term-compiler) complete port/internal entry/[artifact](../../../../glossary.md#term-artifact) shapes and all constructors.
3. Change [Prepare](../../../../glossary.md#prepare) context/projection and migrate every constructor, preserving source profiles.
4. Add [Engine](../../../../glossary.md#term-engine) [capture](../../../../glossary.md#term-capture-meanings)/default/factory cutover and route all four [compile](../../../../glossary.md#compile) paths through the field.
5. Add [CPU](../../../../glossary.md#term-cpu) then [Metal](../../../../glossary.md#term-metal) plan/[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), version bumps, cross-profile rejection, and strict guards.
6. Reconcile Javadocs/current docs, [run](../../../../glossary.md#term-run) the exact stale-call search, then validate as one change.

## Acceptance criteria

1. Exact public [shapes](../../../../glossary.md#shape) above pass reflection/[Javadoc](../../../../glossary.md#term-javadoc) checks; [Planning](../../../../glossary.md#term-planning) exposes Config through `api`.
2. Null order/messages, immutable capture, builder reassignment/state behavior, and factory
   ownership-on-success behavior are covered without compatibility shims.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [capture glossary](../../../../glossary.md#term-capture-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
3. Default builder and `Engine.standard()` compile strict. Explicit strict/accelerator identity is
   observed unchanged in advanced and ordinary forward/backward queries and artifacts; sessions
   reuse their Engine's prepared profile without run-time selection.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [accelerator glossary](../../../../glossary.md#term-accelerator); [identity glossary](../../../../glossary.md#term-identity-meanings); [artifacts glossary](../../../../glossary.md#term-artifact); [Engine glossary](../../../../glossary.md#term-engine); [run glossary](../../../../glossary.md#term-run))</span>
4. Query, artifact, and context equality/hash/text differ only when otherwise-equal values have a
   different profile; Prepare projection retains the exact artifact profile.<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash); [Prepare glossary](../../../../glossary.md#prepare))</span>
5. CPU/Metal strict capability and execution remain unchanged; accelerator compilation is
   fail-closed before a backend route is authorized. No relaxed result or route is claimed.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [backend route glossary](../../../../glossary.md#backend-route))</span>
6. CPU/Metal plans, specializations, route/tuning compatibility, and encoded decisions include
   profile. Same-profile encoding is deterministic; different profiles never share identity;
   pre-change and malformed/unknown encodings reject or miss safely.
7. Runtime, RunState, session/run result, and Trace production shapes remain profile-free. No hot
   run loop branches, allocates, boxes, reflects, synchronizes, or looks up profile policy.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [RunState glossary](../../../../glossary.md#run-state--runstate); [run result glossary](../../../../glossary.md#run-result--runresult); [Trace glossary](../../../../glossary.md#trace))</span>
8. Add `EngineNumericalProfileIntegrationTest`: an actual `Engine.standard()` strict CPU graph
   compiles, prepares, runs, and materializes expected output; an explicit strict builder session
   repeats successfully; an explicit accelerator CPU Engine fails compilation with no run or
   relaxed fallback. Existing advanced integration callers use the exact two-argument factory.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [argument glossary](../../../../glossary.md#term-argument))</span>
9. Reconcile `ARCHITECTURE.md`; `docs/architecture/contracts/{foundational-modules,
   compiler-autograd,runtime-prepare-engine,backend-execution}.md`; `docs/architecture/{
   module-boundaries,dependency-rules,lifecycle,runtime-prepare-backend-boundary,
   partition-scoring}.md`; `docs/api/{compile-api,public-api,tensor-api,runtime-api}.md`;
   `docs/backend-guide/{capability-provider,cpu-backend,metal-backend,partition-preparer}.md`;
   `docs/user-guide/{backend-selection,compiling-graphs}.md`; `docs/glossary.md`; affected package Javadocs; and accepted `docs/design/decisions/0019-explicit-numerical-profiles.md`.
10. At 0018 completion, Model 0026, CPU 0017, and Metal 0015 remained Draft and no other task
    became Ready. Their later planning promotion is outside this completed implementation.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Ready glossary](../../../../glossary.md#term-ready))</span>

## [Validation](../../../../glossary.md#term-validation) and review

```bash
./gradlew :modules:planning:test :modules:planning:javadoc :modules:compiler:test \
  :modules:compiler:javadoc :modules:prepare:test :modules:prepare:javadoc \
  :modules:engine:test :modules:engine:javadoc :backends:cpu:test :backends:cpu:javadoc \
  :backends:metal:test :backends:metal:javadoc
./gradlew :testing:backend-conformance:test :testing:architecture-tests:test --rerun-tasks
./gradlew :testing:integration-tests:test --tests '*EngineNumericalProfileIntegrationTest'
./gradlew build
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Inspect generated Javadocs for all changed public records/methods and inspect the actual [Engine](../../../../glossary.md#term-engine) [smoke](../../../../glossary.md#term-smoke-test)
output. The independent Class C review must cover [API](../../../../glossary.md#term-application-programming-interface-api)/callsite completeness, [ownership](../../../../glossary.md#term-ownership-meanings)/null order,
strict [fail-closed behavior](../../../../glossary.md#admission-preflight-and-fail-closed-validation), [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) isolation, [dependency](../../../../glossary.md#term-dependency) visibility, [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) absence,
and documentation. Reuse validation only when review changes no executable code.

## Result

Implemented the [atomic](../../../../glossary.md#resource-transaction-and-rollback) profile-propagation spine and independently approved it with zero findings.

- Planning queries, Compiler artifacts, Prepare projection, Engine/AdvancedEngine capture, and
  every executable caller now retain one exact graph-wide profile. <span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [artifacts glossary](../../../../glossary.md#term-artifact); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph))</span>`STRICT_IEEE` remains the
  builder and `Engine.standard()` default; Runtime, sessions, run state/results, and Trace remain
  profile-free.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [run state glossary](../../../../glossary.md#run-state--runstate); [Trace glossary](../../../../glossary.md#trace))</span>
- [CPU](../../../../glossary.md#term-cpu) and [Metal](../../../../glossary.md#term-metal) capability/preparation remained [fail-closed](../../../../glossary.md#term-fail-closed) for [`ACCELERATOR`](../../../../glossary.md#term-accelerator), while their plans,
  specializations, workload/candidate decisions, codecs, and local/complete cache compatibility
  became profile-separated with the required version advances.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Required module tests/Javadocs, backend conformance, architecture tests, the actual Engine
  numerical-profile integration smoke, full build, Markdown validation, and diff checks passed.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [smoke glossary](../../../../glossary.md#term-smoke-test); [validation glossary](../../../../glossary.md#term-validation))</span>
- The performance gate found identical normalized hashes for the base/current generated executable
  body. The base showed the same ambient rejects, the current focused gate passed twice, and no
  performance test or threshold was modified.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span>
- Documentation/[Javadoc](../../../../glossary.md#term-javadoc) remediation landed at `07a01b9c`; final independent Class C review returned
  `APPROVE` with zero findings.
