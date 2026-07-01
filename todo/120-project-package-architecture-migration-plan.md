# 120. Project Package Architecture Migration Plan

## Stav

Status: `COMPLETE`

Implementacni faze migrace jsou dokoncene. Finalni architektura pouziva jedinou
partition lifecycle:

```text
PlannedPartition
  -> ExecutablePartitionPlan(PlannedPartition, PartitionExecutionPlan)
  -> LoweredPartition
  -> backend prepare
  -> PreparedExecution
```

Dokonceni znamena, ze produkcni zdroje, testy, package hranice a aktivni
dokumentace pouzivaji partition model bez paralelniho druheho lifecycle. Plny
`./gradlew test` byl spusten dvakrat, ale oba behy selhaly az pri Gradle finalizaci
vysledku kvuli chybejicimu
`build/test-results/test/binary/in-progress-results-*.bin`; nebylo zaznamenano
zadne assertion selhani. Tento infrastrukturalni vysledek neni prezentovan jako
uspesny plny test.

Legenda:

- `[x]` implementovano a overeno dostupnou validaci
- `[!]` prikaz probehl, ale jeho Gradle finalizace selhala mimo test assertions

Stav fazi:

- [x] Faze 0: baseline, inventory a vynutitelne package hranice
- [x] Faze 1: dependency leaves, compiled graph model a lifecycle facade
- [x] Faze 2: top-level trace DTO a partition diagnostika
- [x] Faze 3: kompletni top-level planning a partition execution planning
- [x] Faze 4: partition memory, runtime residency a accelerator buffer kontrakty
- [x] Faze 5: prepared execution kontrakt a odstraneni runtime-to-backend dispatch
- [x] Faze 6: presun prepared execution do top-level runtime
- [x] Faze 7: partition prepare context, validace a orchestrace
- [x] Faze 8: backend consumer a BLAS ownership migrace vcetne cpu1
- [x] Faze 9: partition testy, dokumentace a source-tree hygiene
- [x] Faze 10: finalni verifikace a uzavreni migrace

## Finalni Cil A Ownership

Projekt je rozdelen podle lifecycle vypoctu a smeru zavislosti:

```text
tensor/operations
  uzivatelska semantika a data

graph.model
  immutable compiled graph snapshot bez planning/runtime zavislosti

graph.optimizer
  backend-neutral semanticke graph rewrites

planning.partition
  vlastnictvi partition, target, kind, source a backend plan

planning.partition.execution
  execution units, value flow, materialized outputs a planning trace

planning.partition.specialization
  backend-neutral detekce a payload specializaci

planning.partition.execution.lowering
  backend-neutral klasifikace operacni semantiky pro execution planning

planning.memory
  partition value flow, bindings, handoff, lifetime a materialization plan

graph.compile
  composition compile faze a immutable CompileArtifacts

backend.lowering.partition
  backend-owned partition execution kontrakty a payloady

prepare.context + prepare.validation
  sdilene immutable prepare vstupy, indexy a partition boundary validace

backend.<name>.prepare
  backend compiler konkretniho kernelu, storage, launch a executable artifactu

prepare.orchestration
  backend selection, lowering invocation a sestaveni PreparedExecution

runtime
  provedeni pripraveneho planu, run state, residency, publication a resources

trace
  immutable diagnosticke DTO bez zpetne zavislosti na producentech
```

Public lifecycle zustava:

```java
CompiledGraph graph = CompiledGraph.compile(output, compileConfig);
PreparedExecution execution = graph.prepare(runtimeConfig);
execution.execute(ExecutionMode.FORWARD);
```

Nevznika old/new API, compatibility alias, forwarding facade ani paralelni plan.

## Jediny Partition Lifecycle

### `PlannedPartition`: ownership output

`planning.partition.PlannedPartition` je vystup backend ownership planningu.
Obsahuje prijatou `Partition`, backend `PartitionPlan` a mnozinu kompatibilnich
backendu. Vlastnici identitu a zdrojove vlastnosti partition pres `Partition`:

- `partitionId`, anchor a usporadane node ids,
- `PartitionTarget` a `PartitionKind`,
- source/planner metadata,
- partition inputs, outputs a pozadavky na materializaci,
- backend plan a kompatibilni backendy.

Execution planning tato vlastnicka data nekopiruje do druheho zaznamu.

### `ExecutablePartitionPlan`: immutable enrichment

`planning.partition.ExecutablePartitionPlan` je immutable enrichment vytvoreny
az po execution planningu:

```java
public record ExecutablePartitionPlan(
        PlannedPartition plannedPartition,
        PartitionExecutionPlan executionPlan
) { }
```

Neni to dalsi ownership rozhodnuti. Drzi puvodni `PlannedPartition` a pripojuje k
nemu presne jeden hotovy `PartitionExecutionPlan`. `CompiledProgram`,
`CompileArtifacts`, memory planning a backend lowering predavaji tento celek,
takze nemusi synchronizovat paralelni seznamy partitions a execution plans.

### `PartitionExecutionPlan`: pouze execution details

`planning.partition.execution.PartitionExecutionPlan` uklada pouze:

```text
executionUnits
executionValues/value flow
materializedOutputs
PartitionExecutionTrace
```

Zamerne neduplikuje partition id, target, kind ani source metadata. Tyto hodnoty
se vzdy ctou z `ExecutablePartitionPlan.plannedPartition().partition()`.

### `LoweredPartition`: backend lowering output

`backend.lowering.LoweredPartition` obsahuje zdrojovy `ExecutablePartitionPlan`
a immutable seznam `LoweredExecutionUnit`. Backend-specific jednotky nesou
`backend.lowering.partition.BackendPartitionExecutionPlan` nebo jiny konkretni
lowered artifact. Lowering tim zachovava vazbu na jediny ownership output a
nevytvari alternativni partition identitu.

Finalni call path:

```text
BackendPlanningService
  -> List<PlannedPartition>
CompileSession + PartitionExecutionPlanner
  -> List<ExecutablePartitionPlan>
MemoryPlanner(MemoryPlanningInput.executablePartitions)
  -> MemoryPlan
CompileArtifacts
  -> prepare.orchestration.PreparedExecutionBuilder
LoweringPipeline
  -> List<LoweredPartition>
BackendPrepareContext.publishLoweredPartitions(...)
  -> backend.<name>.prepare
  -> PreparedStepExecutable
  -> runtime.runner.PreparedExecutionRunner
```

Static shape, dtype, layout, storage, specialization, execution-unit selection,
materialization, provider route a launch policy zustavaji v compile/prepare.
Runtime pouze aplikuje pripravene rozhodnuti.

## Cilovy Package Strom

```text
src/main/java/
  tensor/
  operations/

  backend/
    contract/
    provider/
      blas/openblas/
    partition/
    lowering/
      LoweredPartition.java
      LoweredExecutionUnit.java
      LoweringInput.java
      LoweringPipeline.java
      partition/
        BackendPartitionExecutionPlan.java
        PartitionBackendPayload.java
        PartitionExecutionGroup.java
        PartitionExecutionKind.java
        PartitionNodePlan.java
        PartitionStorageContract.java
        CpuFusedPartitionPayload.java
        CpuSpecializedPrimitivePayload.java
        MetalPartitionPayload.java
        CudaPartitionPayload.java
    cpu/
      prepare/
      execution/
      kernels/
    cpu1/
      prepare/
      exec/
      kernels/
      launch/
      offset/
      provider/
      storage/
      trace/
    metal/
    cuda/
    opencl/

  graph/
    CompiledGraph.java
    model/
    optimizer/
    compile/
      CompileArtifacts.java
      CompiledProgram.java
      GraphCompiler.java
      session/

  planning/
    backend/
    descriptor/
    intent/
    memory/
      PartitionBindingAllocator.java
      PartitionHandoffPlanner.java
      PartitionMemoryPlan.java
      PartitionValueFlowPlan.java
      PartitionValueFlowPlanner.java
      PartitionValueLifetime.java
    partition/
      PlannedPartition.java
      ExecutablePartitionPlan.java
      Partition.java
      PartitionPlan.java
      PartitionKind.java
      PartitionTarget.java
      cost/
      execution/
        PartitionExecutionPlan.java
        PartitionExecutionPlanner.java
        PartitionExecutionPlanningContext.java
        PartitionExecutionTrace.java
        PartitionExecutionValue.java
        ExecutionUnit.java
        StructuralPartitionExecutionUnitPlanner.java
        lowering/
          OperationSemanticClassifier.java
          OperationSemanticLevel.java
      specialization/
        PartitionSpecializationPlanner.java
        PartitionSpecializationCandidate.java
        PartitionSpecializationKind.java
        PartitionSpecializationPayload.java
    value/

  prepare/
    context/
      BackendPrepareContext.java
      LoweredPartitionIndex.java
      PartitionRoleIndex.java
    validation/
      BackendPartitionExecutionPlanValidator.java
    orchestration/
      PreparedExecutionBuilder.java
      BackendPrepareDispatcher.java
      BackendPrepareTraceContributors.java

  runtime/
    contract/
    execution/
    state/
    residency/
    memory/
    device/buffer/
    publication/
    runner/

  trace/
    compile/
      PartitionCompileTrace.java
      PartitionDecisionTrace.java
    prepare/
      GpuLoweredPartitionTrace.java
      BackendSelectionDecisionTrace.java
    execution/
    backend/
```

## Dependency Direction

```text
operations/tensor + backend.contract
                -> graph.model
                -> graph.optimizer/graph.compile
                -> planning.partition
                -> planning.partition.execution

planning.partition.execution -> planning.partition.specialization
planning.partition.execution -> planning.partition.execution.lowering
ExecutablePartitionPlan -> planning.memory
graph.compile -> ExecutablePartitionPlan + MemoryPlan

prepare.orchestration -> backend.lowering.LoweringPipeline
backend lowering -> backend.lowering.partition
backend lowering -> ExecutablePartitionPlan + MemoryPlan

prepare.context/validation -> LoweredPartition + backend.lowering.partition
prepare.orchestration -> concrete backend preparers
backend.<name>.prepare -> prepared runtime contracts + backend execution/kernels
runtime.runner -> PreparedStepExecutable

producers -> trace DTOs
concrete backend adapters -> backend.provider
```

Zakazane smery:

- `graph.model` do planning, prepare, runtime nebo konkretniho backendu.
- `planning` do `graph.CompiledGraph`, prepare, runtime nebo konkretniho backendu.
- `planning.partition.execution` do backend-specific loweringu.
- `prepare.context` nebo `prepare.validation` do orchestrace nebo concrete prepareru.
- backend preparery do `prepare.orchestration`.
- runtime do CPU, cpu1, Metal, CUDA nebo OpenCL implementace.
- trace DTO do graph/planning/prepare/runtime/backend producentu.
- shared provider do vyssich lifecycle vrstev.

## Aktivni Mapping Podle Lifecycle

### Planning A Memory

| Odpovednost | Finalni vlastnik |
|---|---|
| Backend ownership output | `planning.partition.PlannedPartition` |
| Immutable execution enrichment | `planning.partition.ExecutablePartitionPlan` |
| Units/value flow/materialized outputs/trace | `planning.partition.execution.PartitionExecutionPlan` |
| Execution planner a context | `planning.partition.execution.PartitionExecutionPlanner`, `PartitionExecutionPlanningContext` |
| Specialization detection/payload | `planning.partition.specialization` |
| Semantic classification pro execution planning | `planning.partition.execution.lowering` |
| Partition bindings a handoff | `planning.memory.PartitionBinding*`, `PartitionHandoff*` |
| Partition value flow a lifetime | `planning.memory.PartitionValueFlow*`, `PartitionValueLifetime` |
| Finalized memory plan input | `MemoryPlanningInput(List<CompiledNode>, List<ExecutablePartitionPlan>, ...)` |

`PartitionValueFlowPlanner` iteruje `ExecutablePartitionPlan`, cte pouze jeho
`executionPlan()` pro units/value flow a vlastnicka metadata bere z
`plannedPartition().partition()`.

### Lowering

| Odpovednost | Finalni vlastnik |
|---|---|
| Backend-neutral semantic helpers | `planning.partition.execution.lowering` |
| Lowering input | `backend.lowering.LoweringInput(List<ExecutablePartitionPlan>, MemoryPlan)` |
| Pipeline output | `backend.lowering.LoweredPartition` |
| Backend execution contract | `backend.lowering.partition.BackendPartitionExecutionPlan` |
| Backend payloady | `backend.lowering.partition.*PartitionPayload` |
| GPU manifest | `backend.accelerator.lowering.GpuLoweredPartitionManifest` |

`LoweringPipeline` vytvari `LoweringRequest` z `ExecutablePartitionPlan`, finalniho
`MemoryPlan`, capabilities a lowering contextu. Kazdy uspesny backend lowerer
vraci `LoweredPartition(source, units)`.

### Prepare

| Odpovednost | Finalni vlastnik |
|---|---|
| Sdilene prepare vstupy a indexy | `prepare.context` |
| Lowered partition lookup | `prepare.context.LoweredPartitionIndex` |
| Anchor/interior role | `prepare.context.PartitionExecutionRole`, `PartitionRoleIndex` |
| Boundary validation | `prepare.validation.BackendPartitionExecutionPlanValidator` |
| Backend selection/lowering composition | `prepare.orchestration.PreparedExecutionBuilder` |
| Backend dispatch pri prepare | `prepare.orchestration.BackendPrepareDispatcher` |
| Kernel/storage/launch compilation | `backend.<name>.prepare` |

Orchestrace publikuje vybrane backend plans, spusti lowering nad vybranymi
`ExecutablePartitionPlan`, publikuje `LoweredPartition` do contextu a teprve potom
vola konkretni backend preparery. Context a validator konkretni backend nevybiraji.

### Trace

| Faze | Finalni DTO |
|---|---|
| Compile partition decisions | `trace.compile.PartitionCompileTrace`, `PartitionDecisionTrace` |
| Materialization cost | `trace.compile.MaterializationCostTrace` |
| Prepare backend selection | `trace.prepare.BackendSelectionTrace`, `BackendSelectionDecisionTrace` |
| Lowered GPU partition snapshot | `trace.prepare.GpuLoweredPartitionTrace` |
| Execution step/runtime state | `trace.execution.*` |
| Backend execution contribution | `trace.backend.*TraceMetadata`, `StepTraceContribution` |

Trace snapshotuje primitive/string/record hodnoty. DTO neimportuji live
`PlannedPartition`, `ExecutablePartitionPlan`, `PartitionExecutionPlan`,
`LoweredPartition` ani backend manifest.

## Cpu1 Impact

Cpu1 zustava backend-owned. Migrace meni jeho vstupni kontrakty a package imports,
ne kernel matematiku ani hot-path policy:

```text
planning.partition.execution
  -> execution units a value transport pred loweringem

planning.partition.specialization
  -> SDPA/MSE a dalsi backend-neutral specialization payloady

planning.partition.execution.lowering
  -> semantic classification pouzita pri execution planningu

backend.lowering.partition
  -> BackendPartitionExecutionPlan a CpuSpecializedPrimitivePayload

prepare.context
  -> BackendPrepareContext, partition role a lowered partition lookup

runtime.execution
  -> ExecutionContext, PreparedStepMetadata, PreparedStepExecutable
```

`Cpu1MseLossPreparer` a `Cpu1AttentionBackwardPreparer` konzumuji finalni
specialization kontrakty. Specialized route testy pouzivaji
`backend.lowering.partition.BackendPartitionExecutionPlan`. Cpu1 executable units
ctou pouze pripraveny storage, layout, kernel, launch a provider route; neprovadeji
novy partition planning, specialization ani backend selection.

BLAS route je jednou vyhodnocena v cpu1 prepare z `BlasConfig`, tuning profilu a
low-level capability dotazu. Exec aplikuje prepared thread count a prime provider
call; trace cte immutable prepared snapshot.

## Historicka Old-To-New Mapa (Pouze Migracni Evidence)

Nasledujici tabulky jsou jedine misto, kde jsou zamerne uvedeny odstranene nazvy.
Nejsou aktivnim architektonickym tvrzenim ani podporovanym API.

| Historicky package | Finalni package |
|---|---|
| `planning.region` | `planning.partition.execution` |
| `planning.region.specialization` | `planning.partition.specialization` |
| `planning.region.lowering` | `planning.partition.execution.lowering` |
| `backend.lowering.region` | `backend.lowering.partition` |

| Historicky symbol | Finalni symbol |
|---|---|
| `PlannedRegion` | `PartitionExecutionPlan` |
| `DefaultRegionPlanner` | `PartitionExecutionPlanner` |
| `RegionPlanningContext` | `PartitionExecutionPlanningContext` |
| `RegionPlanningTrace` | `PartitionExecutionTrace` |
| `CpuRegionPlanningPolicy` | `CpuPartitionExecutionPlanningPolicy` |
| `RegionValue` | `PartitionExecutionValue` |
| `StructuralRegionUnitPlanner` | `StructuralPartitionExecutionUnitPlanner` |
| `RegionPlanValidator` | `BackendPartitionExecutionPlanValidator` |
| `LoweredRegionIndex` | `LoweredPartitionIndex` |
| `GpuLoweredRegionTrace` | `GpuLoweredPartitionTrace` |
| `GpuLoweredRegionManifest` | `GpuLoweredPartitionManifest` |
| `CpuRegionLowerer` | `CpuPartitionLowerer` |
| `NativeCpuRegionSelectionTest` | `NativeCpuPartitionSelectionTest` |

| Historicky memory symbol | Finalni symbol |
|---|---|
| `RegionBindingAllocator` | `PartitionBindingAllocator` |
| `RegionBindingAssignment` | `PartitionBindingAssignment` |
| `RegionHandoffPlanner` | `PartitionHandoffPlanner` |
| `RegionHandoffRequirement` | `PartitionHandoffRequirement` |
| `RegionMemoryBinding` | `PartitionMemoryBinding` |
| `RegionMemoryBindingKind` | `PartitionMemoryBindingKind` |
| `RegionMemoryPlan` | `PartitionMemoryPlan` |
| `RegionValueFlowPlan` | `PartitionValueFlowPlan` |
| `RegionValueFlowPlanner` | `PartitionValueFlowPlanner` |
| `RegionValueLifetime` | `PartitionValueLifetime` |

Odstranene historicke packages nemaji produkcni ani testovaci source tree.

## Implementacni Faze

### Faze 0: Baseline A Package Hranice

Status: `[x]`

- Zachycen source inventory a dirty worktree.
- `PackageOwnershipBoundaryTest` a `SourceTreeHygieneTest` vynucuji smer zavislosti.
- Boundary checks nejsou vypnute ani podminene existenci ciloveho adresare.

### Faze 1: Graph Model A Dependency Leaves

Status: `[x]`

- `graph.CompiledGraph` je jedina lifecycle facade v graph vrstve.
- `graph.model` je immutable dependency leaf.
- `CompiledNodeSnapshotter` a `CompiledProgram` jsou v `graph.compile`.
- `backend.contract.ComputeBackend` je backend-neutral leaf.

### Faze 2: Top-Level Trace A Partition Diagnostika

Status: `[x]`

- Trace DTO byly oddeleny od producentu.
- Compile trace pouziva partition rozhodnuti.
- Prepare trace snapshotuje lowered GPU partition jako
  `trace.prepare.GpuLoweredPartitionTrace`.
- Dtype se snapshotuje jako lossless `DataType.name()` string.

### Faze 3: Planning A Partition Execution

Status: `[x]`

- Backend ownership vraci `PlannedPartition`.
- Execution planning vytvari `PartitionExecutionPlan` v
  `planning.partition.execution`.
- Immutable dvojice je `ExecutablePartitionPlan`.
- Specializace jsou v `planning.partition.specialization`.
- Semantic helpers jsou v `planning.partition.execution.lowering`.
- Stary paralelni lifecycle byl odstranen atomicky bez aliasu.

### Faze 4: Partition Memory A Runtime Device Kontrakty

Status: `[x]`

- `planning.memory` planuje nad `List<ExecutablePartitionPlan>`.
- Binding, handoff, memory plan a value flow pouzivaji partition terminologii.
- Runtime vlastni residency, transfer, native CPU memory a device buffers.
- Materializace je explicitni planning/prepared output, ne kernel fallback.

### Faze 5: Prepared Execution Kontrakt

Status: `[x]`

- `PreparedStepMetadata` vyzaduje non-null `PreparedStepExecutable`.
- Runtime runner vola executable artifact primo.
- Centralni concrete-backend dispatch byl odstranen.
- Backend prepare resolve kernel/storage/launch pred hot path.

### Faze 6: Kompletni Runtime

Status: `[x]`

- Prepared execution, run state, residency, publication a runner jsou pod runtime.
- Runtime neimportuje konkretni backend implementation package.
- Public Tensor lifecycle pouziva finalni runtime kontrakty.

### Faze 7: Partition Prepare Context, Validace A Orchestrace

Status: `[x]`

- `prepare.context.LoweredPartitionIndex` indexuje `LoweredPartition`.
- `prepare.validation.BackendPartitionExecutionPlanValidator` kontroluje boundary
  coverage nad `backend.lowering.partition.BackendPartitionExecutionPlan`.
- Orchestrace jako jedina vybira backend, spousti lowering a vola preparery.
- Concrete preparery neimportuji orchestration.

### Faze 8: Backend Consumers, Cpu1 A BLAS

Status: `[x]`

- CPU, cpu1, Metal, CUDA a OpenCL pouzivaji finalni partition kontrakty.
- Cpu1 specialized prepare pouziva `planning.partition.specialization` a
  `backend.lowering.partition`.
- Shared OpenBLAS provider je JDK/FFM-only leaf v
  `backend.provider.blas.openblas`.
- Provider route, threshold, capability, debug a thread policy jsou resolve-nuty v
  prepare a hot path je znovu nevyhodnocuje.

### Faze 9: Testy, Hygiene A Dokumentace

Status: `[x]`

- Test packages a nazvy odpovidaji partition lifecycle.
- Aktivni dokumentace ukazuje pouze finalni package strom.
- Hygiene kontroluje nepritomnost odstraneneho lifecycle a legacy backend stromu.
- Dokumentace OpenBLAS popisuje `openblas.lib` -> `OPENBLAS_LIB` -> `openblas`.

### Faze 10: Finalni Verifikace

Status: `[x]`; implementacni a architektonicke brany jsou uzavrene. Vysledek
plneho Gradle test tasku je presne zaznamenan jako finalization failure, nikoliv
jako uspesny test.

Evidence k 2026-07-01:

- `[x]` `./gradlew classes` -- BUILD SUCCESSFUL.
- `[x]` Targeted suite -- 400 testu, 0 failures, 0 errors, 27 skipped.
- `[x]` `./gradlew metalTest` -- 307 testu, 0 failures, 0 errors, 1 skipped.
- `[!]` `./gradlew test` byl spusten dvakrat. V obou behach Gradle selhal pri
  finalizaci vysledku, protoze chybel
  `build/test-results/test/binary/in-progress-results-*.bin`. Nebylo hlaseno zadne
  assertion failure, ale ani jeden beh neni oznacen jako BUILD SUCCESSFUL.
- `[x]` Package/path existence audity prosly.
- `[x]` Legacy symbol a dependency audity vratily ocekavany prazdny vystup.
- `[x]` Zadny commit ani push nebyl v ramci finalni dokumentacni opravy proveden.

## Verifikacni Prikazy

### Compile A Targeted Suite

```bash
./gradlew classes

./gradlew test \
  --tests SourceTreeHygieneTest \
  --tests PackageOwnershipBoundaryTest \
  --tests PreparedExecutionBuildTest \
  --tests CompiledGraphTraceTest \
  --tests ComputeModeTraceTest \
  --tests 'planning.partition.*' \
  --tests 'planning.partition.execution.*' \
  --tests 'planning.partition.specialization.*' \
  --tests 'planning.memory.*' \
  --tests 'backend.lowering.*' \
  --tests 'backend.lowering.partition.*' \
  --tests prepare.validation.BackendPartitionExecutionPlanValidatorTest \
  --tests backend.cpu.nativecpu.NativeCpuPartitionSelectionTest \
  --tests 'backend.cpu1.*ExecutionContractTest' \
  --tests backend.cpu1.Cpu1FusedGeneratedExecutionTest \
  --tests backend.cuda.CudaAcceleratorExecutionPathTest \
  --tests backend.metal.MetalLayoutAwareDeviceFlowTest \
  --tests backend.metal.MetalBufferTraceSmokeTest
```

Zaznamenany targeted vysledek: 400 testu, 0 failures, 0 errors, 27 skipped.

### Metal Suite

```bash
./gradlew metalTest
```

Zaznamenany vysledek: 307 testu, 0 failures, 0 errors, 1 skipped.

### Plny Test Task

```bash
./gradlew test
./gradlew test
```

Oba pokusy dosly k Gradle result finalization failure kvuli chybejicimu
`build/test-results/test/binary/in-progress-results-*.bin`. Z teto evidence se
nesmi odvodit ani deklarovat uspesny plny test task.

### Finalni Source A Package Audit

```bash
test -d src/main/java/planning/partition/execution
test -d src/main/java/planning/partition/specialization
test -d src/main/java/planning/partition/execution/lowering
test -d src/main/java/backend/lowering/partition

test -f src/main/java/planning/partition/PlannedPartition.java
test -f src/main/java/planning/partition/ExecutablePartitionPlan.java
test -f src/main/java/planning/partition/execution/PartitionExecutionPlan.java
test -f src/main/java/backend/lowering/LoweredPartition.java
test -f src/main/java/prepare/context/LoweredPartitionIndex.java
test -f src/main/java/prepare/validation/BackendPartitionExecutionPlanValidator.java

rg -n 'import planning\.partition\.(execution|specialization)|import backend\.lowering\.partition' \
  src/main/java src/test/java
```

`rg` je pozitivni audit finalnich consumeru a musi vratit nalezy. Samostatny
stale-symbol audit je soucasti finalni verifikace migrace a musi byt prazdny.

## Rizika A Mitigace

| Riziko | Mitigace |
|---|---|
| Ownership a execution metadata se rozdeli do paralelnich seznamu | `ExecutablePartitionPlan` drzi non-null `PlannedPartition` i `PartitionExecutionPlan`. |
| Execution plan zduplikuje partition identitu | `PartitionExecutionPlan` ma jen units, value flow, materialized outputs a trace. |
| Memory planning ztrati vazbu na vlastnika | `MemoryPlanningInput` a `PartitionValueFlowPlanner` konzumuji `ExecutablePartitionPlan`. |
| Backend lowering prepise ownership | `LoweredPartition.source()` zachovava puvodni `ExecutablePartitionPlan`. |
| Prepare bude znovu planovat | Context pouze indexuje lowered output; orchestrace sklada a preparery kompiluji. |
| Cpu1 hot path znovu vybere specializaci | Specializace se uzavre v planning/lowering/prepare; exec pouzije prepared unit. |
| Trace vytvori zavislost zpet do produceru | Trace obsahuje snapshoty, ne live partition/planning/backend objekty. |
| Plny test bude omylem oznacen za uspesny | Dva finalization failures jsou explicitne oddeleny od assertion vysledku. |

## Definition Of Done

- [x] Celkovy status je `COMPLETE`, protoze vsechny implementacni faze jsou hotove.
- [x] Existuje jediny lifecycle
  `PlannedPartition -> ExecutablePartitionPlan -> LoweredPartition`.
- [x] `PlannedPartition` je jediny ownership output pro partition identitu,
  target, kind, source, backend plan a kompatibilitu.
- [x] `ExecutablePartitionPlan` je immutable enrichment obsahujici non-null
  `PlannedPartition` a non-null `PartitionExecutionPlan`.
- [x] `PartitionExecutionPlan` obsahuje jen units, value flow, materialized outputs
  a trace; neduplikuje partition id, target, kind ani source.
- [x] Execution planning je v `planning.partition.execution`.
- [x] Specializace jsou v `planning.partition.specialization`.
- [x] Semantic execution-planning helpers jsou v
  `planning.partition.execution.lowering`.
- [x] Backend partition execution kontrakty jsou v `backend.lowering.partition`.
- [x] `CompiledProgram`, `CompileArtifacts`, memory planning a lowering pouzivaji
  `ExecutablePartitionPlan` jako jednotny celek.
- [x] `planning.memory` pouziva partition binding, handoff, memory a value-flow
  model.
- [x] `LoweredPartition.source()` zachovava `ExecutablePartitionPlan`.
- [x] Prepare context pouziva `LoweredPartitionIndex` a validace pouziva
  `BackendPartitionExecutionPlanValidator`.
- [x] Cpu1 prepare a testy pouzivaji finalni specialization a backend lowering
  partition kontrakty; kernel hot path nebyl rozsiren o planning.
- [x] Compile/prepare/execution trace pouzivaji partition DTO a snapshoty.
- [x] Odstranene historicke nazvy jsou v tomto dokumentu pouze v jasne oznacene
  historical old-to-new mape.
- [x] `graph.CompiledGraph` je jedina graph lifecycle facade.
- [x] Runtime neimportuje konkretni backend implementation package.
- [x] Backend preparery neimportuji `prepare.orchestration`.
- [x] Trace DTO neimportuji sve producenty ani live planning/backend objekty.
- [x] `./gradlew classes` prosel.
- [x] Targeted suite ma 400 testu, 0 failures/errors a 27 skipped.
- [x] `./gradlew metalTest` ma 307 testu, 0 failures/errors a 1 skipped.
- [x] Dva behy `./gradlew test` jsou presne zdokumentovane jako Gradle result
  finalization failure kvuli chybejicimu `in-progress-results-*.bin`, bez assertion
  failure a bez tvrzeni o uspesnem plnem testu.
- [x] Source/path a stale-symbol audity jsou prazdne tam, kde maji byt prazdne.
- [x] Dokumentace neobsahuje aktivni tvrzeni o odstranene architekture.
- [x] Nebyl vytvoren compatibility alias, docasna facade, duplicitni registry ani
  dalsi implementacni faze.

## Commit Stav

Dokument nepozaduje ani neprovadi finalni commit. V ramci teto opravy se necommituje
a nepushuje.
