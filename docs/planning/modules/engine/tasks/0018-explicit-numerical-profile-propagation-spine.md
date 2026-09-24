# Task 0018: Explicit Numerical-Profile Propagation Spine

## Status

Complete

Implementation `ce7a7dfa67c0dce5cd7b41b2a5bc90ae305393c3` plus documentation/Javadoc
remediation `07a01b9c13c2ceea0922b9af82da3e6e08375306` passed the required module
test/Javadoc checkpoint, backend conformance, architecture tests, the actual
`EngineNumericalProfileIntegrationTest`, full build, documentation validation, and diff checks.
Independent Class C review returned final `APPROVE` with zero findings.

## Change class

Class C — one public Engine construction option, one clean-cutover public advanced factory, a
Planning API/dependency edge, three cross-module immutable records, and backend plan/cache
identity change together. A clean implementation context and independent targeted API,
documentation, dependency, lifecycle, and cache review are mandatory.

## Goal and exact API

Transport one immutable graph-wide `NumericalProfile` from Engine construction through every
ordinary and advanced forward/backward compile, profile-qualified ownership query, nine-component
Compiler artifacts, six-component Prepare projection, and backend plan/cache identity. Make no
numerical result or relaxed capability current.

The clean-cutover shapes are exact:

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

`PrepareContext`'s partition/node convenience constructor becomes seven arguments with profile
first. `GraphCompiler`'s complete package-private entry adds profile immediately after mode.
There are no old-arity overloads, one-argument advanced factory, aliases, or hidden defaults.

## Lifecycle, null, and default contract

- A fresh `Engine.Builder` holds `STRICT_IEEE`. `numericalProfile` rejects null with message
  `numericalProfile` before builder-state validation, returns the same open builder, may be called
  before or after registrations, and uses the last explicit value. Spent/closed builders reject a
  non-null call without mutation. `build()` captures the value exactly once into the new Engine.
- `Engine.standard()` keeps its current public shape and builder ownership path and is strict by
  that default. The profile is immutable for the Engine lifetime and cannot vary by graph,
  compile, prepare, session, tuning trial, or run.
- `AdvancedEngine.takeOwnership(integration, profile)` validates `cpuIntegration` then
  `numericalProfile`; null transfers no ownership. Every package-private constructor/test seam also
  requires an explicit profile. Advanced `compile(...)` and every ordinary forward, explicit-seed
  backward, scalar-objective backward, one-shot, tuning, and session path use the captured field.
- New record components are non-null in declaration order, retain the enum singleton, and
  participate in record equality, hashing, and text. Compiler and Prepare retain the exact value;
  neither interprets it. No Runtime, RunState, prepared-execution, run-result, session, or Trace
  production type gains a profile field or per-run lookup.

## Backend identity and fail-closed boundary

- `CpuCapabilityProvider` and `MetalCapabilityProvider` return false for every `ACCELERATOR` query
  before existing occurrence checks. Both preparers independently reject a non-strict context
  before route/candidate construction. Existing strict matrices and owner selection are unchanged.
- Add `NumericalProfile numericalProfile` as the first `CpuPartitionPreparationPlan` component and require every unit specialization to agree. Replace `CpuKernelSpecialization.NumericalMode` and
  `CpuOpenBlasTuningBatch.NumericalMode` in their current component positions outright with
  `NumericalProfile`; production creates only strict values. Profile enters specialization class
  identity/compatibility, OpenBLAS workload identity, local tuning equality/compatibility, and
  complete-plan projection/variant identity. Use stable names or backend-local wire values, never
  ordinal. Advance generator envelope 66→67, OpenBLAS candidate/policy 1→2, and local/complete
  value schemas 1→2; old and cross-profile decisions are misses/rejections.
- Give `MetalNegPreparationPlan` a final profile, first constructor argument, and accessor; require plan/context agreement. Add profile after route-policy version in tuning `Compatibility` and to
  the workload digest with an explicit wire value. Advance workload, candidate, compatibility,
  route-policy, and codec versions 4→5; native ABI and MPSGraph node schema stay 4. Old and
  cross-profile decisions fail closed.
- These changes transport and isolate identity only. Do not enable an `ACCELERATOR` CPU or Metal
  occurrence, alter generated/native arithmetic, choose fast math, or reinterpret Model semantics.
  CPU 0017 and Metal 0015 were the sole Draft semantic-realization owners during this task.

## Dependencies and integration

- Depends on: Model 0027; Config 0006 at `37e9e9db0bdb0b11564e3696079b0d6a5e0c8109`
- Conflicts with: Planning capability-query/API dependency, Compiler artifacts/port, Prepare
  context/projection, Engine construction/compile lifecycle, CPU or Metal plan/cache identity
- Parallel group: None
- Common base revision: `37e9e9db0bdb0b11564e3696079b0d6a5e0c8109`
- Integration order: one atomic clean cutover; no partial merge
- Integration validation: numerical-profile propagation spine checkpoint
- Shared-document integration owner: Main planner

## Exact implementation and callsite inventory

Production/API owners:

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
backend evidence. Production projections preserve their source profile; standalone tests/evidence
pass `STRICT_IEEE` unless explicitly testing identity. Update shape tests
`BackendCapabilityContractTest`, `GraphCompilationPortTest`/`GraphCompilationPortPublicShapeTest`,
`CompileArtifactsTest`, `PrepareContextTest`, `AdvancedEnginePublicShapeTest`,
`EngineTypedPublicShapeTest`, and `PlanningDependencyContractTest`. Search those executable roots
for zero old-arity calls and zero `NumericalMode` declarations/uses; completed briefs remain
historical evidence, not callers.

## Migration order

1. Change Planning's Config edge to `api`, query shape/tests, and all explicit query callers.
2. Change Compiler complete port/internal entry/artifact shapes and all constructors.
3. Change Prepare context/projection and migrate every constructor, preserving source profiles.
4. Add Engine capture/default/factory cutover and route all four compile paths through the field.
5. Add CPU then Metal plan/cache identity, version bumps, cross-profile rejection, and strict guards.
6. Reconcile Javadocs/current docs, run the exact stale-call search, then validate as one change.

## Acceptance criteria

1. Exact public shapes above pass reflection/Javadoc checks; Planning exposes Config through `api`.
2. Null order/messages, immutable capture, builder reassignment/state behavior, and factory
   ownership-on-success behavior are covered without compatibility shims.
3. Default builder and `Engine.standard()` compile strict. Explicit strict/accelerator identity is
   observed unchanged in advanced and ordinary forward/backward queries and artifacts; sessions
   reuse their Engine's prepared profile without run-time selection.
4. Query, artifact, and context equality/hash/text differ only when otherwise-equal values have a
   different profile; Prepare projection retains the exact artifact profile.
5. CPU/Metal strict capability and execution remain unchanged; accelerator compilation is
   fail-closed before a backend route is authorized. No relaxed result or route is claimed.
6. CPU/Metal plans, specializations, route/tuning compatibility, and encoded decisions include
   profile. Same-profile encoding is deterministic; different profiles never share identity;
   pre-change and malformed/unknown encodings reject or miss safely.
7. Runtime, RunState, session/run result, and Trace production shapes remain profile-free. No hot
   run loop branches, allocates, boxes, reflects, synchronizes, or looks up profile policy.
8. Add `EngineNumericalProfileIntegrationTest`: an actual `Engine.standard()` strict CPU graph
   compiles, prepares, runs, and materializes expected output; an explicit strict builder session
   repeats successfully; an explicit accelerator CPU Engine fails compilation with no run or
   relaxed fallback. Existing advanced integration callers use the exact two-argument factory.
9. Reconcile `ARCHITECTURE.md`; `docs/architecture/contracts/{foundational-modules,
   compiler-autograd,runtime-prepare-engine,backend-execution}.md`; `docs/architecture/{
   module-boundaries,dependency-rules,lifecycle,runtime-prepare-backend-boundary,
   partition-scoring}.md`; `docs/api/{compile-api,public-api,tensor-api,runtime-api}.md`;
   `docs/backend-guide/{capability-provider,cpu-backend,metal-backend,partition-preparer}.md`;
   `docs/user-guide/{backend-selection,compiling-graphs}.md`; `docs/glossary.md`; affected package Javadocs; and accepted `docs/design/decisions/0019-explicit-numerical-profiles.md`.
10. At 0018 completion, Model 0026, CPU 0017, and Metal 0015 remained Draft and no other task
    became Ready. Their later planning promotion is outside this completed implementation.

## Validation and review

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

Inspect generated Javadocs for all changed public records/methods and inspect the actual Engine smoke
output. The independent Class C review must cover API/callsite completeness, ownership/null order,
strict fail-closed behavior, schema/cache isolation, dependency visibility, Runtime/Trace absence,
and documentation. Reuse validation only when review changes no executable code.

## Result

Implemented the atomic profile-propagation spine and independently approved it with zero findings.

- Planning queries, Compiler artifacts, Prepare projection, Engine/AdvancedEngine capture, and
  every executable caller now retain one exact graph-wide profile. `STRICT_IEEE` remains the
  builder and `Engine.standard()` default; Runtime, sessions, run state/results, and Trace remain
  profile-free.
- CPU and Metal capability/preparation remained fail-closed for `ACCELERATOR`, while their plans,
  specializations, workload/candidate decisions, codecs, and local/complete cache compatibility
  became profile-separated with the required version advances.
- Required module tests/Javadocs, backend conformance, architecture tests, the actual Engine
  numerical-profile integration smoke, full build, Markdown validation, and diff checks passed.
- The performance gate found identical normalized hashes for the base/current generated executable
  body. The base showed the same ambient rejects, the current focused gate passed twice, and no
  performance test or threshold was modified.
- Documentation/Javadoc remediation landed at `07a01b9c`; final independent Class C review returned
  `APPROVE` with zero findings.
