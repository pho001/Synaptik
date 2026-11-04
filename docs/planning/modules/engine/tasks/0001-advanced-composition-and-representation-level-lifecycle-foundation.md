# Task 0001: Advanced Composition and Representation-Level [Lifecycle](../../../../glossary.md#lifecycle) Foundation

> **Editorial historical reading key.**
>
> This task records the first advanced CPU-only integration boundary. The caller transfers one opened
> CPU integration to AdvancedEngine; Engine passes Tensor expressions and explicit compile settings to
> Compiler, obtains CPU preparation collaborations for shared Prepare, and supplies ordered
> BufferRepresentation inputs to Runtime’s runner. Compiler owns graph and derivative semantics; CPU
> owns lowering and physical compatibility; Engine coordinates admission and outward ownership rather
> than implementing either.
>
> Borrowed wrappers and their backing storage remain caller responsibilities through result closure.
> The result owns completed run state, not detached host values. The original methodless prepared
> handle, HostTensorStorage borrow signature, and results-before-adapter shutdown describe this
> historical stage. See the [current advanced
> lifecycle](../../../../reference/engine/advanced.md#advanced-engine-lifecycle) and [current Runtime
> prepared owner](../../../../reference/runtime/prepared-execution.md#current-prepared-execution) for
> explicit prepared closure and Runtime’s unique lease authority.

## Status

Complete

## Goal

Establish the first [Engine](../../../../glossary.md#term-engine)-owned, explicitly composed [compile](../../../../glossary.md#compile) -> [prepare](../../../../glossary.md#prepare) -> [run](../../../../glossary.md#term-run) [lifecycle](../../../../glossary.md#lifecycle) over the
current public [Compiler](../../../../glossary.md#term-compiler), Prepare, [Runtime](../../../../glossary.md#term-runtime), and supported [CPU](../../../../glossary.md#term-cpu) seams. The result is an advanced
integration [API](../../../../glossary.md#term-application-programming-interface-api) for tests and low-level composition, not the ordinary user experience.

The smallest truthful current composition owns exactly one `CpuBackendIntegration`. It compiles
with that adapter's capability and availability facts, accepts preparation only for the resulting
single non-empty maximal CPU [partition](../../../../glossary.md#partition), delegates complete schedule construction to that same CPU
adapter, and runs with representation-level `BufferRepresentation` inputs. It does not claim that
several [backend](../../../../glossary.md#backend)-owned complete-schedule assemblers can be combined.

The public lifecycle [shape](../../../../glossary.md#shape) is:

```java
try (AdvancedEngine engine = AdvancedEngine.takeOwnership(cpuIntegration)) {
    AdvancedCompiledGraph compiled = engine.compile(
            mode,
            forwardOutputs,
            functionalGradientRequest,
            optimizationConfig,
            backendIntent,
            partitionScoringConfig);
    AdvancedPreparedExecution prepared = engine.prepare(compiled);
    try (AdvancedRunResult result = engine.run(prepared, callerRepresentations)) {
        int publicationCount = result.resultCount();
    }
}
```

`AdvancedCompiledGraph` and `AdvancedPreparedExecution` are [opaque](../../../../glossary.md#term-opaque) Engine-owned handles. They
prevent ordinary callers from extracting and running inward-layer recipes independently of the
Engine [owner](../../../../glossary.md#term-owner). Typed logical input [binding](../../../../glossary.md#term-binding), published-value access, and [host](../../../../glossary.md#term-host) [materialization](../../../../glossary.md#materialization) remain
tasks 0003 and 0004.

## [Scope](../../../../glossary.md#term-scope)

- Replace the placeholder `EngineModule` with public final `AdvancedEngine`, an `AutoCloseable`
  advanced integration facade in <span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>`io.github.pho001.synaptik.engine`.
- Add exactly one public construction path,
  `AdvancedEngine.takeOwnership(CpuBackendIntegration)`. A successful call transfers ownership of
  the exact open CPU adapter to the Engine; a null argument fails before transfer. There is no
  borrowed-adapter mode in this task.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [Engine glossary](../../../../glossary.md#term-engine); [argument glossary](../../../../glossary.md#term-argument); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Add `AdvancedEngine.compile(...)` with the current standalone [`CompileMode`](../../../../glossary.md#compile-mode--compilemode), ordered forward
  [`Tensor`](../../../../glossary.md#tensor) outputs, optional `FunctionalGradientRequest`, [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig),
  [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), and [`PartitionScoringConfig`](../../../../glossary.md#partition-scoring-configuration--partitionscoringconfig) inputs. Delegate once to
  `GraphCompilationPort.compile(...)` with the owned CPU capability provider and availability
  snapshot as singleton ordered lists.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Return a new [opaque](../../../../glossary.md#term-opaque) `AdvancedCompiledGraph` bound by exact owner identity to this Engine and the
  exact <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>`CompileArtifacts`. Do not expose the [artifacts](../../../../glossary.md#term-artifact) through a public accessor.
- Add `AdvancedEngine.prepare(AdvancedCompiledGraph)`. Require an open Engine and the exact owner,
  then call `GraphPreparation.prepare(...)` with the owned [CPU](../../../../glossary.md#term-cpu) adapter's
  `preparations(artifacts)` and `scheduleAssembler()` results.
- Return a new opaque `AdvancedPreparedExecution` bound by exact owner identity to this Engine and
  the exact [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution). Do not expose the [Runtime](../../../../glossary.md#term-runtime) recipe through a public accessor.
- Add `AdvancedEngine.borrow(HostTensorStorage)` as the explicit representation-level bridge to
  the owned CPU adapter. Return the existing public nominal <span data-glossary-navigation="true"> ([nominal glossary](../../../../glossary.md#term-nominal))</span>`BufferRepresentation`; caller
  storage and the non-owning wrapper remain caller-owned.
- Add `AdvancedEngine.run(AdvancedPreparedExecution, List<BufferRepresentation>)`. Require an open
  Engine and exact owner, then invoke one retained stateless [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner).
- Wrap a successful Runtime result in public final `AdvancedRunResult`. Expose only
  `resultCount()`, `isClosed()`, and [idempotent](../../../../glossary.md#term-idempotent) [thread-safe](../../../../glossary.md#term-thread-safe) `close()` in this task; expose no
  representation, Tensor, host value, publication map, or <span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [publication glossary](../../../../glossary.md#term-publication))</span>[`RunState`](../../../../glossary.md#run-state--runstate) access.
- Track every open `AdvancedRunResult` so Engine closure closes results in reverse successful-run
  order before closing the owned CPU integration. Borrowed caller input representations and their
  storage are never closed by Engine or Runtime.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Use one package-private `EngineBackendComposition` collaboration and one package-private
  `CpuEngineBackendComposition` realization to isolate current Engine orchestration from CPU
  construction details and provide a focused failure-injection test seam. The abstraction exists
  only for the current CPU realization and Engine lifecycle tests; it is not public backend SPI,
  a registry, or a promise of mixed-backend schedule composition.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [backend glossary](../../../../glossary.md#backend); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [registry glossary](../../../../glossary.md#term-registry))</span>
- Add exact direct `implementation(project(":modules:model"))` and
  `implementation(project(":modules:planning"))` dependencies to `modules/engine/build.gradle.kts`.
  The public advanced API names Model <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model))</span>`Tensor` and `HostTensorStorage`, and the package-private
  composition seam names Planning <span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>[`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider); Engine must not obtain those
  contracts accidentally through Compiler, Prepare, CPU, or another transitive edge.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Update the focused dependency explanation and Engine architecture test to record and enforce the
  exact direct dependency inventory without changing the authoritative architecture direction.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- Update `testing/integration-tests/build.gradle.kts` with the exact direct test compile/runtime
  dependencies required by <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>`EngineAdvancedLifecycleIntegrationTest`. Keep the existing
  `implementation(project(":modules:engine"))`, then add ordered `testImplementation` dependencies
  on Compiler, Runtime, Config, Model, and CPU. Do not make any Engine dependency `api` and do not
  rely on Engine's intentionally non-exported implementation dependencies.
- Make Engine lifecycle methods synchronous and safe for concurrent use. Implement one cold
  lifecycle gate that prevents new calls after closure begins, lets already-admitted calls finish
  or roll back, and serializes repeated/concurrent close.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous))</span>
- Add focused public-shape, ownership, ordering, failure, concurrency, and real CPU lifecycle
  tests plus one integration test and focused architecture enforcement.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [concurrency glossary](../../../../glossary.md#term-concurrency); [integration test glossary](../../../../glossary.md#term-integration-test))</span>
- Require a separate clean documentation-focused pass after implementation to finalize Javadocs,
  public API/runtime explanations, glossary impact, and planning evidence in the same overall
  change.

## Out of [scope](../../../../glossary.md#term-scope)

- Standard built-in construction for ordinary users; that is task 0002.
- More than one registered lifecycle adapter, successful mixed-backend or multi-partition schedule
  assembly, or combining several complete-schedule assemblers.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [partition glossary](../../../../glossary.md#partition))</span>
- [Metal](../../../../glossary.md#term-metal) or [CUDA](../../../../glossary.md#term-cuda) [lifecycle](../../../../glossary.md#lifecycle) adapters. Their current [modules](../../../../glossary.md#term-module) contain placeholder marker types only.
- Successful zero-node/pass-through preparation. The CPU adapter must continue to reject it
  because current Prepare has no backend-declared buffer assignment for the requested value.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [CPU glossary](../../../../glossary.md#term-cpu); [Prepare glossary](../../../../glossary.md#prepare); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- A public generic backend interface, custom-backend adapter API, backend builder, borrowed
  backend ownership mode, plugin system, or backend discovery.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [backend ownership glossary](../../../../glossary.md#backend-ownership); [plugin glossary](../../../../glossary.md#term-plugin))</span>
- `CompileConfig`, `PrepareConfig`, `RunOptions`, defaults, or replacement aggregate
  configuration. The six current standalone compile inputs remain explicit.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- Typed logical input binding or validation against expected Tensor type, Shape, layout span,
  capacity, or access role; that is task 0003.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- Published-result value or representation access, typed publication mapping, host
  materialization, or persistence support; those are tasks 0003 and 0004.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- One-shot forward/backward convenience or tuning composition; those are tasks 0005-0007.
- Any `Tensor.execute`, `Tensor.backward`, gradient field, Runtime dependency, hidden compilation
  scope, or lifecycle state on Model types.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [Runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [Model glossary](../../../../glossary.md#term-model))</span>
- Engine exception translation for the future ordinary-user facade. Existing Compiler, Prepare,
  Runtime, and CPU unchecked failures preserve their identity in this advanced seam except for
  explicit Engine owner/state/argument validation.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [facade glossary](../../../../glossary.md#term-facade); [Compiler glossary](../../../../glossary.md#term-compiler); [identity glossary](../../../../glossary.md#term-identity-meanings); [owner glossary](../../../../glossary.md#term-owner); [argument glossary](../../../../glossary.md#term-argument))</span>
- Changes to Compiler, Prepare, Runtime, Config, Trace, Model, Backend Contract, CPU, Metal, CUDA,
  OpenBLAS provider, NN, Training, tuning, generated code, operation semantics, kernel logic, or
  legacy source.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [operation glossary](../../../../glossary.md#operation); [kernel glossary](../../../../glossary.md#kernel))</span>
- Any build or dependency change other than the two exact Engine-to-Model/Planning lines and the
  exact integration-test dependency closure required by this task.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Reflection, [classpath](../../../../glossary.md#term-classpath) or annotation scanning, `ServiceLoader`, a service locator, registry
  singleton, mutable process-global state, or Runtime backend selection.<span data-glossary-navigation="true"> ([service locator glossary](../../../../glossary.md#term-service-locator); [registry glossary](../../../../glossary.md#term-registry); [singleton glossary](../../../../glossary.md#term-singleton); [mutable glossary](../../../../glossary.md#term-mutable))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core lifecycle, Core invariants,
  Engine, Runtime, Prepare, concrete backends, dependency rules, and compile/prepare/run lifecycle.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [backends glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Engine master plan](../master-plan.md)
- [Compiler 0006B3 complete integration port](../../compiler/tasks/0006b3-public-constant-free-complete-compile-entry.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [integration port glossary](../../../../glossary.md#term-integration-port))</span>
- [Prepare 0003 complete orchestration](../../prepare/tasks/0003-prepare-orchestration-and-validation.md)
- [Prepare 0004 opaque tuning handoff](../../prepare/tasks/0004-opaque-backend-candidate-batch-and-selected-decision-handoff.md)<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- [Runtime 0010 prepared runner](../../runtime/tasks/0010-prepared-runner-and-dynamic-execution.md)
- [CPU 0010F supported lifecycle adapter](../../../backends/cpu/tasks/0010f-supported-cpu-lifecycle-integration-adapter.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture constraints

- Engine is the explicit outer composition root. It may depend on public Compiler, Prepare,
  Runtime, Config, Trace, and supported concrete-backend APIs; no inward module may depend on
  Engine.<span data-glossary-navigation="true"> ([composition root glossary](../../../../glossary.md#term-composition-root); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [backend glossary](../../../../glossary.md#backend); [module glossary](../../../../glossary.md#term-module))</span>
- Because Engine source directly names Model and Planning public types, its build must directly
  depend on <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning))</span>`modules:model` and `modules:planning`. These inward edges realize the existing outer
  composition-root direction and do not authorize Model or Planning to depend on Engine.
- [Engine](../../../../glossary.md#term-engine)'s dependencies remain non-exported `implementation` dependencies. The integration-test
  module must therefore declare every project whose types its test source imports or must resolve
  from the public `AdvancedEngine` method [signatures](../../../../glossary.md#term-signature); [widening](../../../../glossary.md#term-widening) Engine dependencies to `api` would
  expose implementation composition transitively and is not authorized.
- Engine must not import any [concrete backend](../../../../glossary.md#concrete-backend) `.internal` package. The public
  `CpuBackendIntegration` is the only current concrete [lifecycle](../../../../glossary.md#lifecycle) adapter.
- [Compiler](../../../../glossary.md#term-compiler) owns [graph](../../../../glossary.md#graph) [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) and [immutable](../../../../glossary.md#term-immutable) `CompileArtifacts`; Prepare owns complete staged
  preparation and validation; CPU owns lowering, representations, finalization, and its complete
  one-partition schedule assembler; Runtime owns isolated per-run state and execution. Engine
  coordinates those owners without reproducing their logic.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [partition glossary](../../../../glossary.md#partition); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- Runtime performs no backend discovery, selection, lookup, lowering, or route choice. All
  capability and availability inputs are supplied during Engine-owned compilation, and the sole
  schedule assembler is fixed before Runtime execution.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- The current public seams do not define how several backend-owned complete-schedule assemblers
  contribute to one heterogeneous schedule. Task 0001 therefore owns exactly one complete CPU
  composition and must reject every artifact shape rejected by <span data-glossary-navigation="true"> ([heterogeneous glossary](../../../../glossary.md#term-heterogeneous); [artifact glossary](../../../../glossary.md#term-artifact); [shape glossary](../../../../glossary.md#shape))</span>`CpuBackendIntegration`, including
  mixed/multi-partition and zero-partition artifacts.
- The package-private Engine composition abstraction must have exactly the current CPU realization
  and the fake test realization. It must not expose a public hypothetical backend hierarchy,
  backend-specific parameter bag, or string-based dispatch.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Opaque compiled and prepared handles retain their exact Engine owner and inward delegate. Only
  their owner may consume them, and no public accessor may expose <span data-glossary-navigation="true"> ([Opaque glossary](../../../../glossary.md#term-opaque); [owner glossary](../../../../glossary.md#term-owner))</span>`CompileArtifacts` or
  [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution).
- A prepared handle may be shared by concurrent Engine runs. Each call still creates exactly one
  isolated Runtime [`RunState`](../../../../glossary.md#run-state--runstate); no mutable run state is retained in Engine, the compiled handle, or
  the prepared handle.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>
- `BufferRepresentation` appears only on the explicit advanced borrow/run surface. It is not
  described as the future ordinary input or result API.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- If implementation needs a shared schedule-contribution contract, another concrete adapter,
  changes to an inward public seam, a new module dependency, an architecture rule, or public
  exposure of an inward delegate, stop and report the exact prerequisite instead of widening the
  task.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing package used and changed:

- `io.github.pho001.synaptik.engine` becomes the intentional advanced lifecycle surface and owns
  the opaque lifecycle handles plus package-private composition/lifecycle machinery.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>

Existing public packages consumed unchanged:

- `io.github.pho001.synaptik.compiler` supplies `GraphCompilationPort`,
  `FunctionalGradientRequest`, and `CompileArtifacts` internally to [Engine](../../../../glossary.md#term-engine).
- `io.github.pho001.synaptik.prepare` supplies complete graph preparation and CPU-supplied
  positional/schedule collaborations.<span data-glossary-navigation="true"> ([graph preparation glossary](../../../../glossary.md#graph-preparation--graphpreparation))</span>
- `io.github.pho001.synaptik.runtime` packages supply the prepared runner, nominal
  representation input, prepared delegate, and result delegate internally to Engine.<span data-glossary-navigation="true"> ([nominal glossary](../../../../glossary.md#term-nominal))</span>
- `io.github.pho001.synaptik.backend.cpu` supplies the sole supported owned [lifecycle](../../../../glossary.md#lifecycle) adapter.
- Current [Model](../../../../glossary.md#term-model) and Config packages supply public [compile](../../../../glossary.md#compile) and representation-borrow inputs.

Build impact:

- `modules/engine/build.gradle.kts` gains exactly the direct Model and Planning project
  dependencies required by named source contracts. Its exact ordered project-dependency inventory
  becomes:

```kotlin
implementation(project(":modules:model"))
implementation(project(":modules:planning"))
implementation(project(":modules:compiler"))
implementation(project(":modules:runtime"))
implementation(project(":modules:prepare"))
implementation(project(":modules:config"))
implementation(project(":modules:trace"))
implementation(project(":backends:cpu"))
implementation(project(":backends:metal"))
implementation(project(":backends:cuda"))
```

The two new inward contracts are inserted before the unchanged existing [dependency](../../../../glossary.md#term-dependency) sequence.

The integration-test [module](../../../../glossary.md#term-module) has a separate test-only dependency closure. Preserve its existing
Engine module dependency and use this exact ordered inventory:

```kotlin
implementation(project(":modules:engine"))
testImplementation(project(":modules:compiler"))
testImplementation(project(":modules:runtime"))
testImplementation(project(":modules:config"))
testImplementation(project(":modules:model"))
testImplementation(project(":backends:cpu"))
```

- Engine is the module under [integration test](../../../../glossary.md#term-integration-test).
- [Compiler](../../../../glossary.md#term-compiler) is required because javac must resolve public `FunctionalGradientRequest` in
  `AdvancedEngine.compile(...)`.
- [Runtime](../../../../glossary.md#term-runtime) is required because javac must resolve public `BufferRepresentation` in
  `AdvancedEngine.borrow(...)` and `run(...)`.
- Config is required by the test's direct compile-request imports.
- Model is required by the test's direct Tensor, descriptor, layout, shape, data-type, and host-
  storage imports.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [host glossary](../../../../glossary.md#term-host))</span>
- [CPU](../../../../glossary.md#term-cpu) is required by the test's direct `CpuBackendIntegration` construction.

[Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Trace](../../../../glossary.md#trace), [Backend](../../../../glossary.md#backend) Contract, [Metal](../../../../glossary.md#term-metal), [CUDA](../../../../glossary.md#term-cuda), and [OpenBLAS](../../../../glossary.md#term-openblas) provider are not named or
resolved by this integration test and must not be added. Test-only dependencies are used for the
five new entries because only `src/test` needs them; the existing main-[scope](../../../../glossary.md#term-scope) Engine dependency is
retained unchanged.

Packages added or changed:

- No Java package is added. The existing Engine root package changes from a placeholder to a
  deliberate public advanced facade plus colocated package-private machinery.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>

Type placement:

- `io.github.pho001.synaptik.engine.AdvancedEngine` - public advanced composition root and sole
  lifecycle/lifetime owner.<span data-glossary-navigation="true"> ([composition root glossary](../../../../glossary.md#term-composition-root); [lifetime glossary](../../../../glossary.md#term-lifetime); [owner glossary](../../../../glossary.md#term-owner))</span>
- `io.github.pho001.synaptik.engine.AdvancedCompiledGraph` - public opaque owner-bound compiled
  handle; Engine owns its package-private construction and delegate access.
- `io.github.pho001.synaptik.engine.AdvancedPreparedExecution` - public opaque owner-bound
  prepared handle; Engine owns its package-private construction and delegate access.
- `io.github.pho001.synaptik.engine.AdvancedRunResult` - public lifecycle-only result wrapper that
  coordinates exactly-once Runtime-result cleanup with Engine closure.
- `io.github.pho001.synaptik.engine.EngineBackendComposition` - package-private exact
  collaboration needed by the current Engine orchestration and fake lifecycle tests.
- `io.github.pho001.synaptik.engine.CpuEngineBackendComposition` - package-private adapter over the
  supported public `CpuBackendIntegration`; it contains no CPU [lowering](../../../../glossary.md#lowering) or representation logic.

Tests needing package-private access mirror `io.github.pho001.synaptik.engine`. The public-[shape](../../../../glossary.md#shape)
test uses `io.github.pho001.synaptik.engine.api`, and the end-to-end test remains in the existing
integration-test package.

## Exact public and package-private [API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape)

`AdvancedEngine` is public, final, and implements `AutoCloseable`. It has no public constructor
and exactly these public declarations:

```java
public static AdvancedEngine takeOwnership(CpuBackendIntegration cpuIntegration);

public AdvancedCompiledGraph compile(
        CompileMode mode,
        List<Tensor> forwardOutputs,
        Optional<FunctionalGradientRequest> functionalGradientRequest,
        GraphOptimizationConfig optimizationConfig,
        BackendIntent backendIntent,
        PartitionScoringConfig partitionScoringConfig);

public AdvancedPreparedExecution prepare(AdvancedCompiledGraph compiledGraph);

public BufferRepresentation borrow(HostTensorStorage storage);

public AdvancedRunResult run(
        AdvancedPreparedExecution preparedExecution,
        List<BufferRepresentation> callerInputs);

public boolean isClosed();

@Override
public void close();
```

`AdvancedCompiledGraph` and `AdvancedPreparedExecution` are public final [opaque](../../../../glossary.md#term-opaque) classes. They have
no public or protected constructors, no public delegate/[owner](../../../../glossary.md#term-owner) accessor, and no public method. They
retain final exact owner and delegate references for package-private [Engine](../../../../glossary.md#term-engine) [validation](../../../../glossary.md#term-validation).

`AdvancedRunResult` is public, final, implements `AutoCloseable`, has no public or protected
constructor, and exposes exactly:

```java
public int resultCount();
public boolean isClosed();
@Override public void close();
```

`EngineBackendComposition` and `CpuEngineBackendComposition` are package-private. Their exact
collaboration shape is:

```java
interface EngineBackendComposition extends AutoCloseable {
    List<BackendCapabilityProvider> capabilityProviders();
    List<BackendAvailabilitySnapshot> availabilitySnapshots();
    PreparedExecution prepare(CompileArtifacts artifacts);
    BufferRepresentation borrow(HostTensorStorage storage);
    @Override void close();
}

final class CpuEngineBackendComposition implements EngineBackendComposition {
    CpuEngineBackendComposition(CpuBackendIntegration integration);
    // exactly the five interface implementations
}
```

Both ordered lists are [immutable](../../../../glossary.md#term-immutable) singletons for the retained [CPU](../../../../glossary.md#term-cpu) adapter. The CPU realization's
[`prepare`](../../../../glossary.md#prepare) method alone obtains `preparations(artifacts)` and `scheduleAssembler()` from that exact
adapter and calls `GraphPreparation.prepare(...)`; this keeps complete-schedule [ownership](../../../../glossary.md#term-ownership-meanings) in one
place. `AdvancedEngine` has one package-private constructor accepting an
`EngineBackendComposition` for the real factory and same-package [lifecycle](../../../../glossary.md#lifecycle) tests. The opaque
compiled/prepared handles each have one package-private constructor plus package-private exact
owner/delegate accessors. `AdvancedRunResult` has one package-private owner/delegate constructor
and no delegate accessor. Any additional package-private lifecycle-gate operations remain private
methods or nested implementation details of `AdvancedEngine`; no additional top-level type is
authorized. Neither composition type may be named from a public or protected [signature](../../../../glossary.md#term-signature) or
[Javadoc](../../../../glossary.md#term-javadoc) link.

No public method returns `CompileArtifacts`, [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [`RunResult`](../../../../glossary.md#run-result--runresult), a Prepare
collaboration, [`RunState`](../../../../glossary.md#run-state--runstate), a CPU internal type, or a collection of [backend](../../../../glossary.md#backend) services.

## [Lifecycle](../../../../glossary.md#lifecycle), ordering, and failure semantics

Construction and [ownership](../../../../glossary.md#term-ownership-meanings):

1. `takeOwnership(null)` throws `NullPointerException("cpuIntegration")` and transfers nothing.
2. Successful construction transfers sole cleanup responsibility for the exact supplied CPU
   integration to the Engine. The caller must not close or use it independently afterward.
3. Task 0001 offers no borrowed adapter, no multiple-registration overload, and no implicit CPU
   opening. Task 0002 owns ordinary built-in construction.

Each [compile](../../../../glossary.md#compile) call:

1. validates the [Engine](../../../../glossary.md#term-engine) is open;
2. requires every top-level argument and the optional/list containers to be non-null, leaving
   nested validation and precise compile-list element rules to the existing Compiler contract;<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
3. delegates once with singleton CPU capability/availability lists and the six exact caller
   compile inputs;<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>
4. preserves the inward pipeline's [argument](../../../../glossary.md#term-argument) order and unchecked failure [identity](../../../../glossary.md#term-identity-meanings); and
5. on success returns an [owner](../../../../glossary.md#term-owner)-bound [opaque](../../../../glossary.md#term-opaque) handle only if closure has not begun.

Each [prepare](../../../../glossary.md#prepare) call:

1. validates Engine openness before non-nullity and exact handle-owner identity, so any call begun
   after closure reports the stable closed-Engine failure before argument or owner validation;
2. obtains preparations and the assembler from the same owned composition;
3. delegates once to `GraphPreparation.prepare`; and
4. preserves CPU/Prepare unchecked failure identity and returns an owner-bound handle only if
   closure has not begun.

Each borrow call validates Engine openness and delegates once. The returned representation is
[borrowed](../../../../glossary.md#ownership-borrowing-and-lease) input state: Engine does not track or close it, and closing it must retain [CPU](../../../../glossary.md#term-cpu) 0010F's
non-owning semantics. The caller must keep the underlying storage live and accessible through all
runs that use it.

Each [run](../../../../glossary.md#term-run) call:

1. validates Engine openness before the prepared handle, caller-input list, exact handle-owner
   identity, and list elements, then retains the existing Runtime validation order before action
   execution;
2. invokes the retained stateless runner synchronously;
3. registers the successful wrapped result before ending its admitted Engine [operation](../../../../glossary.md#operation); and
4. if closure began during the call, closes the newly created Runtime result, attaches any
   distinct rollback failure as suppressed to a stable <span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>`IllegalStateException`, and returns no
   live result.

Engine closure marks closure begun before waiting, rejects every newly attempted compile,
prepare, borrow, or run with `IllegalStateException("advanced engine is closed")`, waits
uninterruptibly for already-admitted [synchronous](../../../../glossary.md#term-synchronous) operations to leave the gate while restoring the
caller's interrupt flag, closes registered open results once in reverse successful-run order, and
then closes the owned [backend composition](../../../../glossary.md#backend-composition) once. It attempts all cleanup, preserves the first
unchecked exception or error, suppresses later distinct failures, and records completion before
returning or throwing. Repeated and concurrent `close()` calls wait for the first close attempt
and then either return or rethrow the same retained cleanup failure instance. `isClosed()` becomes
true when closure begins.

`AdvancedRunResult.close()` is safe against concurrent Engine/result close, delegates to the
exact [Runtime](../../../../glossary.md#term-runtime) result at most once, and unregisters itself even when cleanup fails. It preserves
the Runtime cleanup failure identity. Its `resultCount()` is the [immutable](../../../../glossary.md#term-immutable) count captured at
construction and remains available after closure; `isClosed()` becomes true before delegate
cleanup begins. A result never exposes its delegate.

Compiled and prepared handles are immutable and [thread-safe](../../../../glossary.md#term-thread-safe). Engine and result state uses no
process-global [mutable](../../../../glossary.md#term-mutable) state. The lifecycle gate is cold orchestration; it must not enter Runtime
execution loops or add per-[element](../../../../glossary.md#term-element)/per-[node](../../../../glossary.md#node) work.

## Affected files

Expected production paths:

- remove `modules/engine/src/main/java/io/github/pho001/synaptik/engine/EngineModule.java`
- add `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java`
- add `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedCompiledGraph.java`
- add `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedPreparedExecution.java`
- add `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedRunResult.java`
- add `modules/engine/src/main/java/io/github/pho001/synaptik/engine/EngineBackendComposition.java`
- add `modules/engine/src/main/java/io/github/pho001/synaptik/engine/CpuEngineBackendComposition.java`
- add `modules/engine/src/main/java/io/github/pho001/synaptik/engine/package-info.java`

Expected focused test paths:

- add `modules/engine/src/test/java/io/github/pho001/synaptik/engine/AdvancedEngineLifecycleTest.java`
- add `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/AdvancedEnginePublicShapeTest.java`
- add `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineAdvancedLifecycleIntegrationTest.java`
- add `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/EngineCompositionContractTest.java`

Expected build paths:

- `modules/engine/build.gradle.kts`, adding exactly
  `implementation(project(":modules:model"))` and
  `implementation(project(":modules:planning"))`
- `testing/integration-tests/build.gradle.kts`, retaining the existing [Engine](../../../../glossary.md#term-engine) `implementation`
  line and adding exactly the five ordered `testImplementation` lines specified under Build
  impact

Expected documentation and [planning](../../../../glossary.md#term-planning) paths finalized in the same overall implementation change:

- `docs/api/public-api.md`
- `docs/api/compile-api.md`
- `docs/api/runtime-api.md`
- `docs/architecture/dependency-rules.md`, limited to explaining the two direct Engine dependency
  realizations and their unchanged one-way direction<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- `docs/glossary.md`, only if the documentation review finds a reusable Engine term or stale
  lifecycle status that existing entries do not cover<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `docs/planning/modules/engine/master-plan.md`
- `docs/planning/modules/engine/tasks/0001-advanced-composition-and-representation-level-lifecycle-foundation.md`
- `docs/planning/roadmap.md`

No other production, test, build, architecture-contract, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), or planning path is expected to
change. The documentation pass must record reasoned
no-change conclusions for architecture explanations, [backend](../../../../glossary.md#backend) guides, Training [API](../../../../glossary.md#term-application-programming-interface-api), Config,
[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[CPU](../../../../glossary.md#term-cpu) Javadocs, [Metal](../../../../glossary.md#term-metal)/[CUDA](../../../../glossary.md#term-cuda) placeholders, backend [conformance](../../../../glossary.md#term-conformance), generated
code, and performance evidence unless a concrete contradiction is found.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create, modify, or remove at most 22 paths:

- eight [Engine](../../../../glossary.md#term-engine) production paths, including removal of the placeholder;
- two Engine test paths;
- one integration-test path;
- one architecture-test path;
- two build paths: one Engine production build and one integration-test build;
- up to five explanatory/glossary/architecture documentation paths; and
- exactly three Engine task/master/roadmap [planning](../../../../glossary.md#term-planning) paths.

If another build/[dependency](../../../../glossary.md#term-dependency) change, an inward-[module](../../../../glossary.md#term-module) source change, a [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) change,
another production type, or a twenty-third path is required, stop and propose the smallest
follow-up or prerequisite task.

## Acceptance criteria

- `AdvancedEngine`, its three opaque lifecycle values, and the exact public methods above match
  the specified modifiers, signatures, visibility, nullability, and Javadocs.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [lifecycle glossary](../../../../glossary.md#lifecycle); [signatures glossary](../../../../glossary.md#term-signature); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Successful construction transfers ownership of exactly one current CPU adapter; no borrowed,
  implicit, multiple, Metal, CUDA, generic custom-backend, or discovery path exists.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [backend glossary](../../../../glossary.md#backend))</span>
- [Compile](../../../../glossary.md#compile) delegates once through `GraphCompilationPort` with exact singleton CPU provider and
  availability order and preserves current compile semantics and failures.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>
- Prepare accepts only a non-null compiled handle from the same open Engine, uses only the same
  owned CPU adapter's preparations and complete-schedule assembler, and preserves CPU's explicit
  zero-node and mixed/multi-partition rejection.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [node glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition))</span>
- Run accepts only a non-null prepared handle from the same open Engine and representation-level
  caller inputs, delegates once to <span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>[`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner), and returns only the lifecycle-
  level result surface.
- Handles from another Engine and every operation after closure fail before inward backend,
  Prepare, or Runtime work.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Concurrent compile/prepare/run calls use immutable delegates and isolated Runtime state.
  Concurrent/repeated Engine and result close is deterministic, invokes each owned cleanup at
  most once, respects result-before-backend reverse ordering, and preserves the specified primary
  and suppressed failure tree.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Calls admitted before closure either complete registration while still open or roll back and
  return no live handle/result after closure begins.
- Borrowed inputs and caller storage are never closed by Engine; run-owned Runtime resources are
  closed by result closure, Engine closure, or existing failure rollback exactly as owned.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- The real CPU integration test compiles, prepares, and concurrently runs one supported non-empty
  CPU graph with separate caller representations and closes every result/Engine cleanly.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test); [graph glossary](../../../../glossary.md#graph))</span>
- Focused negative coverage proves zero-node prepare rejection and prevents any Metal/CUDA
  placeholder or second fake complete-schedule adapter from being represented as a supported
  successful composition.
- Source/architecture checks prove [Engine](../../../../glossary.md#term-engine) imports no `.internal` package and uses no reflection,
  classpath scanning, <span data-glossary-navigation="true"> ([classpath glossary](../../../../glossary.md#term-classpath))</span>`ServiceLoader`, service locator, registry singleton, process-global mutable
  state, Runtime backend selection, or raw/unchecked generic bridge.<span data-glossary-navigation="true"> ([service locator glossary](../../../../glossary.md#term-service-locator); [registry glossary](../../../../glossary.md#term-registry); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- `modules/engine/build.gradle.kts` contains the exact ordered direct dependency inventory,
  including Model and Planning exactly once and preserving the existing Compiler, Runtime,
  Prepare, Config, Trace, CPU, Metal, and CUDA dependencies. The focused architecture test rejects
  a missing, duplicate, transitive-only, reordered, extra, or Engine-reversing dependency.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [Trace glossary](../../../../glossary.md#trace); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- `testing/integration-tests/build.gradle.kts` retains the existing Engine `implementation` and
  adds exactly the ordered Compiler, Runtime, Config, Model, and CPU `testImplementation`
  dependencies above. Focused architecture coverage rejects omission, reordering, duplication,
  `api` substitution, or any unrelated Planning, Prepare, Trace, Backend Contract, Metal, CUDA,
  OpenBLAS-provider, or other project dependency.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- No [`Tensor`](../../../../glossary.md#tensor) API, inward public contract, backend implementation, kernel, generated-code path,
  operation semantic, architecture contract, or ADR changes; the only build changes are the two
  required Engine dependency lines and the exact integration-test closure.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [kernel glossary](../../../../glossary.md#kernel); [architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Every changed or added Java type and method has meaningful Javadoc covering ownership,
  lifecycle, threading, nullability, inputs, results, and failures without claiming ordinary-user
  typed binding or result access.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [binding glossary](../../../../glossary.md#term-binding))</span>
- A separate documentation-focused clean context finalizes affected Javadocs, API documentation,
  glossary impact, and planning evidence before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused checks:

```bash
./gradlew :modules:engine:test --tests '*AdvancedEngine*'
./gradlew :testing:integration-tests:test --tests '*EngineAdvancedLifecycleIntegrationTest*'
./gradlew :testing:architecture-tests:test --tests '*EngineCompositionContractTest*'
./gradlew :modules:engine:test
```

The [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle) test must use the package-private composition seam to prove exact delegation,
[owner](../../../../glossary.md#term-owner) mismatch, validation-before-[callback](../../../../glossary.md#term-callback) ordering, admitted-call/close coordination,
result-before-[backend](../../../../glossary.md#backend) cleanup, exactly-once concurrent close, and primary/suppressed
`RuntimeException` and `Error` behavior. The [integration test](../../../../glossary.md#term-integration-test) must cross the real public [Compiler](../../../../glossary.md#term-compiler),
[CPU](../../../../glossary.md#term-cpu), [Prepare](../../../../glossary.md#prepare), and [Runtime](../../../../glossary.md#term-runtime) seams without importing CPU internals.

Documentation-focused pass:

```bash
./gradlew :modules:engine:javadoc
git diff --check
```

Manual and structural validation:

- compile a small distinct-package Java fixture using only the public advanced Engine, CPU,
  Config, Model storage, and Runtime representation types;<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [fixture glossary](../../../../glossary.md#term-fixture); [Model glossary](../../../../glossary.md#term-model))</span>
- inspect `javap -public` for the four public Engine types and confirm the exact surface, opaque
  handle constructors, and absence of inward delegates;<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- scan Engine production and test sources for forbidden `.internal` imports, reflective/discovery
  mechanisms, service/registry/global state, raw types, unchecked casts, and Tensor execution or
  backward additions;<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [Tensor glossary](../../../../glossary.md#tensor))</span>
- verify `modules/engine/build.gradle.kts` changed only by the exact direct Model and Planning
  lines, and that its full ordered dependency inventory remains one-way and permitted;<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- verify `testing/integration-tests/build.gradle.kts` has exactly the six ordered lines specified
  above, uses `testImplementation` for all five newly required projects, and adds no unrelated
  integration dependency;
- validate changed Markdown headings, local links and anchors, fences, terminology, trailing
  whitespace, and final newlines;
- verify the exact changed-path allowlist and 22-path ceiling;
- confirm task 0001 is `Complete`, Engine 0002 is the next Draft frontier but not [`Ready`](../../../../glossary.md#term-ready), Engine
  0002-0008 remain `Draft`, and no later Engine task specification exists; and
- [run](../../../../glossary.md#term-run) `git diff --check` on the final combined change.

After executable code and the clean documentation pass stabilize, run one repository-wide
[dependency](../../../../glossary.md#term-dependency) [checkpoint](../../../../glossary.md#term-checkpoint):

```bash
./gradlew test :testing:architecture-tests:test
```

The repository-wide run is required because task 0001 changes [module](../../../../glossary.md#term-module) dependencies and creates the
first end-to-end Engine lifecycle, even though it changes no authoritative architecture rule or
shared build logic. Do not repeat a successful module or repository suite in the documentation
context unless executable code changes or a concrete risk requires it. Backend [conformance](../../../../glossary.md#term-conformance) is
unchanged because CPU behavior and its accepted [artifact](../../../../glossary.md#term-artifact) domain do not change. No benchmark is
required because this is cold orchestration with [constant](../../../../glossary.md#term-constant)-time lifecycle bookkeeping outside per-
[element](../../../../glossary.md#term-element), per-[node](../../../../glossary.md#node), and Runtime hot loops.

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) 0006B3 public [constant](../../../../glossary.md#term-constant)-free complete [compile](../../../../glossary.md#compile) [integration port](../../../../glossary.md#term-integration-port) - Complete.
- [Prepare](../../../../glossary.md#prepare) 0003 complete orchestration and [validation](../../../../glossary.md#term-validation) - Complete.
- Prepare 0004 [opaque](../../../../glossary.md#term-opaque) tuning transport - Complete and unused by this untuned task.
- [Runtime](../../../../glossary.md#term-runtime) 0010 [prepared execution runner](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner), [`RunResult`](../../../../glossary.md#run-result--runresult), representation contracts, and later Runtime
  closure hardening through 0014 - Complete.
- [CPU](../../../../glossary.md#term-cpu) 0010F supported [lifecycle](../../../../glossary.md#lifecycle) integration adapter - Complete.
- Existing Engine dependency declarations for Compiler, Prepare, Runtime, Config, Trace, CPU,
  Metal, and CUDA - present; direct Model and Planning declarations are the proven missing task
  prerequisites and are added by this task. Only CPU has a lifecycle adapter. Placeholder
  Metal/CUDA dependencies authorize no composition claim.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency); [Trace glossary](../../../../glossary.md#trace); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning))</span>
- The integration-test module's existing Engine dependency is present. Direct test-only Compiler,
  Runtime, Config, Model, and CPU dependencies are the proven missing compilation prerequisites
  and are added by this task; no other integration dependency is required.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>

## Follow-up tasks

- Engine 0002: standard deterministic built-in construction. Its initial supported inventory must
  be truthful about available adapters; it may start with CPU only. It must not add mixed-backend
  success until a complete schedule-composition contract and concrete adapter justify it.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
- Engine 0003: typed logical input binding and published-result access without Runtime
  representation leakage.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Engine](../../../../glossary.md#term-engine) 0004: explicit [host](../../../../glossary.md#term-host) [materialization](../../../../glossary.md#materialization).
- Engine 0005-0006: Engine-owned one-shot forward and [scalar](../../../../glossary.md#term-scalar)-[objective](../../../../glossary.md#term-objective) backward convenience.
- Engine 0007: optional Config/tuning composition.
- Engine 0008: [lifecycle](../../../../glossary.md#lifecycle) capability [checkpoint](../../../../glossary.md#term-checkpoint).
- A future mixed-backend schedule-contribution prerequisite must be planned in the owning
  Engine/Prepare boundary only when a second concrete lifecycle adapter exists. It is not hidden
  inside task 0001 or pre-created as a speculative detailed task.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>

## Architecture impact

Expected architecture-decision impact: None. [Dependency](../../../../glossary.md#term-dependency) realization: two direct [Engine](../../../../glossary.md#term-engine) edges.

This task realizes the existing Engine composition-root rule with the only current supported
complete [lifecycle](../../../../glossary.md#lifecycle) adapter. Direct Engine-to-[Model](../../../../glossary.md#term-model) and Engine-to-[Planning](../../../../glossary.md#term-planning) dependencies are
required because Engine source names those public contracts; they point inward and preserve every
authoritative forbidden edge. The focused explanatory dependency update and [architecture test](../../../../glossary.md#term-architecture-test)
make that existing direction truthful and enforceable. `ARCHITECTURE.md` and ADRs remain
unchanged. If implementation requires another edge, a [heterogeneous](../../../../glossary.md#term-heterogeneous) schedule contract, or another
architecture decision, stop and report the conflict rather than updating architecture from this
task.

The integration-test dependencies are [validation](../../../../glossary.md#term-validation)-[fixture](../../../../glossary.md#term-fixture) dependencies, not production
architecture edges. They preserve Engine's non-exported dependency policy and require no further
change to `ARCHITECTURE.md`, an [ADR](../../../../glossary.md#term-architecture-decision-record-adr), or the explanatory production [dependency direction](../../../../glossary.md#java-platform-and-build-tools).

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in /Users/phujka/IdeaProjects/Synaptik on Engine task 0001.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/modules/engine/master-plan.md, and
docs/planning/modules/engine/tasks/0001-advanced-composition-and-representation-level-lifecycle-foundation.md
in full. Read the focused lifecycle/module/dependency architecture pages and inspect the current
public Compiler, Prepare, Runtime, CPU, Metal, CUDA, and Engine surfaces named by the task.

Implement task 0001 exactly as specified. Build only the advanced owner-bound CPU composition and
representation-level compile -> prepare -> run seam. Do not invent mixed-backend schedule
composition, Config aggregates, typed logical binding/results, host materialization, standard or
one-shot convenience, tuning, discovery, shared-contract changes, or backend internals. Stop and
report if the exact task cannot fit its architecture or 22-path ceiling. Do not commit or push.

Add only the two exact direct Engine project dependencies on modules/model and modules/planning;
do not rely on transitive exposure and do not add another build edge. Enforce the complete Engine
dependency inventory in the focused architecture test.

Retain testing/integration-tests' existing Engine implementation dependency and add only the five
ordered testImplementation dependencies on Compiler, Runtime, Config, Model, and CPU specified by
the task. Extend the focused architecture test to enforce that exact integration inventory. Do
not change Engine dependencies to api and do not add transitive convenience dependencies.

Run the focused Engine, integration, architecture, module, public-shape, forbidden-mechanism,
scope, and whitespace validation specified by the task. After executable Java stabilizes, hand
the actual diff and exact test evidence to a separate clean documentation-focused agent/thread in
the same overall change. That pass must follow docs/developer-guide/documentation-rules.md,
independently finalize affected Javadocs, API documentation, glossary impact, and planning
evidence, and reuse successful Java tests unless executable behavior changes.

Update this task with local decisions, known limitations, implementation notes, exact validation
evidence, documentation-context identity/result, completion summary, and final status. Mark
Complete only after every acceptance criterion and the documentation pass succeed.
```

## Local decisions

- Task 0001 composes exactly one owned CPU adapter. A single-entry composition is the smallest
  honest realization of the current architecture; an ordered heterogeneous set would be fiction
  until another adapter and a schedule-contribution contract exist.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [heterogeneous glossary](../../../../glossary.md#term-heterogeneous))</span>
- Opaque Engine handles keep inward Compiler and Runtime recipes behind the advanced facade and
  make exact owner validation possible. They are preferable to returning raw artifacts that can
  bypass the Engine lifetime.<span data-glossary-navigation="true"> ([Opaque glossary](../../../../glossary.md#term-opaque); [Engine glossary](../../../../glossary.md#term-engine); [Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [facade glossary](../../../../glossary.md#term-facade); [owner glossary](../../../../glossary.md#term-owner); [validation glossary](../../../../glossary.md#term-validation); [artifacts glossary](../../../../glossary.md#term-artifact); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Caller-supplied CPU ownership is transfer-only on successful Engine construction. A borrowed
  adapter mode is excluded because the Engine could not guarantee that its recipes and runs
  remain inside the adapter lifetime.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- The package-private composition seam exists for the real CPU wrapper and deterministic lifecycle
  tests only. It is not a public abstraction for hypothetical backends.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [backends glossary](../../../../glossary.md#backend))</span>
- Direct Engine dependencies on Model and Planning are required by the source types task 0001
  names. They are existing inward composition dependencies, not new ownership or architecture
  rules.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning))</span>
- `BufferRepresentation` is accepted only because this is explicitly the temporary advanced
  representation-level run seam. Task 0003 replaces that experience with typed logical binding.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding))</span>
- The Engine closes open results before its CPU adapter so run-owned resources and any admitted
  backend work cannot outlive the concrete lifecycle owner.

## Known limitations

- Only one non-empty maximal CPU partition can prepare and run. Mixed/multi-partition artifacts
  and every Metal/CUDA lifecycle remain unsupported.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [artifacts glossary](../../../../glossary.md#term-artifact); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A zero-node/pass-through compiled graph can be represented by Compiler but cannot be prepared
  through the current CPU/Prepare boundary.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Inputs must already be [CPU](../../../../glossary.md#term-cpu)-compatible `BufferRepresentation` values. Borrowing host storage
  validates only intrinsic storage facts, not its expected logical Tensor binding.<span data-glossary-navigation="true"> ([Borrowing glossary](../../../../glossary.md#term-borrowing); [host storage glossary](../../../../glossary.md#host-storage); [Tensor glossary](../../../../glossary.md#tensor); [binding glossary](../../../../glossary.md#term-binding))</span>
- Results expose publication count and cleanup only. They expose no value, representation, Tensor,
  host payload, or publication-role mapping.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [host glossary](../../../../glossary.md#term-host); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- No ordinary standard factory, one-shot convenience, exception translation, tuning, or aggregate
  configuration exists in this task.

## [Validation](../../../../glossary.md#term-validation) evidence

- Planning evidence only: a clean implementation attempt created the specified Engine Java and
  test surface, then `./gradlew :modules:engine:test --tests '*AdvancedEngine*'` failed during
  `:modules:engine:compileJava` with 13 missing-package/type errors. The public API directly needed
  Model <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model))</span>[`Tensor`](../../../../glossary.md#tensor) and `HostTensorStorage`, and the package-private composition seam directly needed
  Planning [`BackendCapabilityProvider`](../../../../glossary.md#backend-capability-provider--backendcapabilityprovider), while `modules/engine/build.gradle.kts` declared neither
  project. Avoiding those types would have violated this task's exact API. The implementation
  context obeyed the stop condition and removed all Java/test changes, leaving only the planning
  diff. This evidence corrects task scope; it completes no implementation acceptance criterion.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- [Planning](../../../../glossary.md#term-planning) evidence only from the corrected implementation attempt: [Engine](../../../../glossary.md#term-engine) [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) passed;
  `./gradlew :modules:engine:test --tests '*AdvancedEngine*'` passed 6 tests; the corrected
  `./gradlew :testing:architecture-tests:test --tests '*EngineCompositionContractTest*'` passed 1
  test after a test-only false-positive correction; and `git diff --check` passed. The required
  `./gradlew :testing:integration-tests:test --tests '*EngineAdvancedLifecycleIntegrationTest*'`
  then failed in integration-test compilation with 48 missing or inaccessible type errors because
  that module declared only Engine while its test directly imports CPU, Config, Engine, and Model
  and javac must also resolve Compiler <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Compiler glossary](../../../../glossary.md#term-compiler))</span>`FunctionalGradientRequest` and [Runtime](../../../../glossary.md#term-runtime)
  `BufferRepresentation` from public Engine signatures. The implementation correctly stopped
  without editing the excluded integration build file. These partial results justify this second
  scope correction. They are historical planning evidence, not final failures.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- Final implementation evidence from context `01a09fba-9026-7c30-8a41-5e6147d05120`: focused
  Engine validation passed 7/7 tests, the real CPU integration passed 2/2 tests, and the focused
  Engine composition architecture check passed 1/1 test. The repository dependency checkpoint
  passed 3,067 tests with zero failures or errors and 28 skipped. The final Engine project
  dependency closure is Model, Planning, Compiler, Runtime, Prepare, Config, Trace, CPU, Metal,
  and CUDA in that order. The integration-test closure is Engine <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda))</span>`implementation` followed by
  Compiler, Runtime, Config, Model, and CPU `testImplementation` dependencies. Its test JVM
  enables <span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>`jdk.incubator.vector` because real [CPU](../../../../glossary.md#term-cpu) preparation loads the [Vector API](../../../../glossary.md#term-vector-api).
- Post-review executable correction: `prepare(...)` and `run(...)` now enter the lifecycle gate
  before their top-level null checks. Calls begun after closure therefore preserve<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  `IllegalStateException("advanced engine is closed")` ahead of argument, owner, and inward
  validation. Focused regressions cover closed <span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [owner glossary](../../../../glossary.md#term-owner))</span>`prepare(null)`, closed `run(null, null)`, and
  closed `run(validPrepared, null)`. The coordinator-reported fresh validation remained 7/7
  focused Engine tests, 2/2 integration tests, and 1/1 focused architecture test, with<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
  `git diff --check` passing.
- Final documentation evidence from context `01a09fd3-35d6-7e60-92d9-776603857790`:
  `./gradlew :modules:engine:javadoc` passed; generated pages for the package and all four public
  types contain the finalized lifecycle contracts; a distinct-package public API fixture compiled
  with only the documented public Engine, CPU, Config, Model, and Runtime types; and<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>
  `javap -public` showed only the seven specified `AdvancedEngine` methods, no public methods on
  either opaque handle, and the three specified <span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>`AdvancedRunResult` methods. A lexical
  comment-stripped comparison proved unchanged executable tokens in all seven new Engine Java
  paths, including <span data-glossary-navigation="true"> ([tokens glossary](../../../../glossary.md#term-token))</span>`package-info.java`. Targeted validation passed for local Markdown links and
  anchors, heading hierarchy, fences, whitespace, and final newlines in all eight changed
  Markdown files.
  Forbidden-source, exact dependency, no-later-task-specification, exact 22-path allowlist, and
  `git diff --check` checks also passed.

## Implementation notes

- `AdvancedEngine` owns the exact CPU adapter after successful construction, snapshots caller
  lists, gates synchronous operations against closure, and keeps Compiler and Runtime delegates
  behind opaque owner-bound handles.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [synchronous glossary](../../../../glossary.md#term-synchronous); [Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [opaque glossary](../../../../glossary.md#term-opaque); [owner glossary](../../../../glossary.md#term-owner))</span>
- Prepare and run enter that gate before top-level argument or owner validation, preserving the
  stable closed-Engine exception for every call begun after closure.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [argument glossary](../../../../glossary.md#term-argument); [validation glossary](../../../../glossary.md#term-validation); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Each run creates isolated Runtime state. Successful results transfer that state into an
  idempotently closeable Engine-tracked wrapper; Engine close waits for admitted work, closes open
  results in reverse run order, and closes the adapter last while preserving the specified
  failure tree.
- CPU preparation remains deliberately limited to exactly one non-empty maximal CPU partition.
  No zero-node, mixed-backend, multi-partition, typed binding/result, host materialization,
  standard composition, one-shot, backward convenience, or tuning claim was introduced.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [node glossary](../../../../glossary.md#node); [backend glossary](../../../../glossary.md#backend); [binding glossary](../../../../glossary.md#term-binding); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization))</span>
- The dependency corrections implement existing boundaries: Engine directly names Model and
  Planning contracts, while the integration fixture directly names or must resolve its five
  test-only dependencies. No authoritative architecture decision changed.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [fixture glossary](../../../../glossary.md#term-fixture))</span>

## Completion summary

- Completed changes: replaced the Engine placeholder with the advanced CPU-only composition and
  owner-bound compile, prepare, borrow, run, result, concurrency, and cleanup lifecycle; added the
  exact dependency closures and focused Engine, integration, public-shape, and architecture tests;
  finalized all affected Engine/package Javadocs, public/compile/runtime API explanations,
  dependency explanation, glossary status, Engine plan, task, and roadmap.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [concurrency glossary](../../../../glossary.md#term-concurrency); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency); [shape glossary](../../../../glossary.md#shape); [runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Files changed or created: exactly the 22 allowlisted paths recorded by this task, including eight
  Engine production-path changes, two Engine test files, one integration test, one architecture
  test, two build files, five explanatory/API documentation files, and three planning files.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- Validation: 7/7 focused Engine, 2/2 integration, 1/1 focused architecture, and repository
  checkpoint 3,067/3,067 with zero failures or errors and 28 skipped; Engine Javadoc, generated
  documentation inspection, distinct-package fixture, exact public <span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Javadoc glossary](../../../../glossary.md#term-javadoc); [fixture glossary](../../../../glossary.md#term-fixture))</span>`javap` shape, comment-stripped
  token proof, forbidden-source/dependency/path checks, eight-file Markdown validation, status
  consistency, absence of an Engine 0002 task specification, and <span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token))</span>`git diff --check` all passed.
  The final focused Engine evidence includes the post-review closed-call precedence regressions
  for null prepared handles and caller-input lists.
- Documentation no-change conclusions: `ARCHITECTURE.md` and ADRs remain unchanged because this is
  an existing composition-root dependency realization. Other architecture explanations, backend
  guides, Training and Config API guides, inward Compiler/Prepare/Runtime/CPU Javadocs,
  Metal/CUDA placeholders, backend conformance, generated code, and performance evidence need no
  update because their contracts or behavior did not change. The glossary changed only to remove
  its stale statement that all Engine composition remained planned; no new reusable term was
  introduced.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Compiler glossary](../../../../glossary.md#term-compiler); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Unresolved issues: none for task 0001. Follow-up remains Engine 0002, the next Draft frontier;
  it is not Ready and has no task specification.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Contexts: [planning](../../../../glossary.md#term-planning) `01a09f9f-b304-7943-97cf-ba7d322bb1f8`; implementation
  `01a09fba-9026-7c30-8a41-5e6147d05120`; documentation
  `01a09fd3-35d6-7e60-92d9-776603857790`.

Status: Complete
