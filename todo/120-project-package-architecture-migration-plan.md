# 120. Project Package Architecture Migration Plan

## Stav

Status: `IN_PROGRESS`

Tento dokument je definitivni migracni plan. Obsahuje vsechny presuny, prejmenovani,
zmeny zavislosti, zmeny testu a verifikacni brany potrebne k uzavreni migrace.
Po dokonceni posledni faze nezustane zadne nerozhodnute architektonicke tema z rozsahu
tohoto dokumentu.

Legenda:

- `[ ]` neimplementovano
- `[~]` rozpracovano
- `[x]` implementovano a overeno
- `[blocked: duvod]` nelze pokracovat bez odstraneni uvedene prekazky

Stav fazi:

- [x] Faze 0: baseline, inventory a vynutitelne package hranice
- [x] Faze 1: dependency leaves, compiled graph model a lifecycle facade hranice
- [x] Faze 2: top-level `trace` bez zpetnych zavislosti
- [x] Faze 3: kompletni top-level `planning` a definitivni region naming
- [x] Faze 4: runtime memory, residency a accelerator buffer kontrakty
- [x] Faze 5: prepared execution kontrakt a odstraneni runtime-to-backend dispatch
- [x] Faze 6: presun celeho `graph.execution` do `runtime`
- [x] Faze 7: rozdeleni prepare contextu, validace a orchestrace bez cyklu
- [x] Faze 8: BLAS ownership migrace a prime prepojeni vsech backendu, vcetne kompletniho cpu1 auditu
- [x] Faze 9: testy, dokumentace a source-tree hygiene
- [ ] Faze 10: celkova verifikace a uzavreni migrace

## Cil

Projekt bude rozdelen podle lifecycle vypoctu a podle smeru zavislosti:

```text
tensor/operations
  uzivatelska semantika a data

graph.model
  immutable compiled graph snapshot bez planning/runtime zavislosti

graph.optimizer
  semanticke graph rewrites nad tensor/graph modelem

planning
  backend-neutral compile-time descriptors, partition, region, value a memory plan

graph.compile
  composition compile faze: optimizer + planning + lowering inputs

prepare.context + prepare.validation
  sdilene prepare-time kontrakty konzumovane backend preparery

backend.<name>.prepare
  backend compiler: konkretni kernel, storage, launch a executable artifact

prepare.orchestration
  cross-backend sestaveni PreparedExecution

runtime
  spusteni hotoveho planu, per-run state, residency, publication a resources

backend.<name>.exec/kernels
  konkretni hot path

trace
  immutable diagnosticke DTO bez zavislosti na producentech
```

Migrace je behavior-preserving s jedinou vnitrni vykonnostne neutralni zmenou:
runtime prestane volat centralni `backend.ComputeEngine` a zavola executable artifact,
ktery byl vybran behem prepare. Verejne volani zustane:

```java
CompiledGraph graph = CompiledGraph.compile(output, compileConfig);
PreparedExecution execution = graph.prepare(runtimeConfig);
execution.execute(ExecutionMode.FORWARD);
```

Nevznikne old/new API, alias ani adapter zachovavajici puvodni package.

## Pevna Architektonicka Rozhodnuti

### 1. `CompiledGraph` je jedina lifecycle facade v `graph`

`graph.CompiledGraph` zustane verejnym vstupem pro `compile()` a `prepare()`.
Je to composition facade, nikoliv compiled model. Smí importovat:

```text
graph.compile.*
planning.intent.*
prepare.orchestration.PreparedExecutionBuilder
runtime.execution.PreparedExecution
runtime.contract.ExecutionMode
trace.compile.CompileTrace
```

Zadna jina trida v root package `graph` ani v `graph.model`, `graph.optimizer`
nebo `graph.compile` nesmi importovat `prepare.*` nebo `runtime.*`.

Zpetny smer je zakazan:

```text
prepare.*  -X-> graph.CompiledGraph
runtime.*  -X-> graph.CompiledGraph
planning.* -X-> graph.CompiledGraph
backend.*  -X-> graph.CompiledGraph
```

Prepare dostane `graph.compile.CompileArtifacts`, ne `CompiledGraph`. Tohle zachova
verejne API a nevytvori konkretni package cyklus.

### 2. `graph.model` je dependency leaf compiled grafu

Model nesmi zaviset na planning outputu. Proto se snapshot builder oddeli od hodnot:

```text
graph.model
  CompiledNode
  CompiledGradientBinding
  CompiledTensorDataSnapshot
  ConstantGradientValue
  AliasViewPolicy

graph.compile
  CompiledNodeSnapshotter
  CompiledProgram
```

`CompiledProgram` patri do compile outputu, protoze agreguje `PartitionPlan`,
`PlannedRegion` a `MemoryPlan`. Neni dependency-leaf model.

### 3. Trace je samostatny top-level diagnostics model

Trace DTO nepatri do runtime. Producenti zustanou ve svych vrstvach:

```text
graph.compile/planning -> vytvari trace.compile DTO
prepare.orchestration -> vytvari trace.prepare DTO
runtime.runner         -> vytvari trace.execution DTO
backend.*.trace        -> vytvari trace.backend DTO
```

`trace` nesmi importovat `graph`, `planning`, `prepare` ani konkretni backend.
Hodnoty z techto vrstev se snapshotuji do stringu, cisel a trace-owned recordu.

### 4. Prepare je rozdelen podle role, ne podle backendu

```text
prepare.context
  sdileny vstup a indexy; nesmi znat konkretni preparery

prepare.validation
  validace region kontraktu; nesmi znat konkretni preparery

prepare.orchestration
  composition root; smi volat konkretni backend preparery

backend.cpu1.prepare
  cpu1 compiler; smi importovat context a validation, nikdy orchestration
```

Tim je smer konkretniho grafu zavislosti:

```text
prepare.orchestration -> backend.cpu1.prepare -> prepare.context
prepare.orchestration -> backend.metal.prepare -> prepare.validation
```

Neexistuje hrana z backend prepareru zpet do `prepare.orchestration`.

### 5. Runtime nevola konkretni backend

`backend.ComputeEngine` se odstrani. Prepare pripoji ke kazdemu vykonnemu kroku
`PreparedStepExecutable`. Runtime runner vola pouze tento kontrakt:

```java
step.metadata().executable().execute(step.compiledNode(), step.metadata(), context);
```

Backendy implementuji kontrakt a zavisi na runtime. Runtime neimportuje
`backend.cpu`, `backend.cpu1`, `backend.metal`, `backend.cuda` ani `backend.opencl`.

### 6. Backend-neutral identita backendu je dependency leaf

`backend.ComputeBackend` se presune do `backend.contract.ComputeBackend`.
`backend.contract` nesmi importovat graph, planning, prepare, runtime, trace ani
konkretni backend. Runtime smi importovat pouze `backend.contract`, ne konkretni
backend packages. Trace backend identitu snapshotuje jako `String`.

### 7. Region naming je definitivne planning naming

V teto migraci se provedou vsechny nasledujici zmeny:

```text
OptimizedRegion              -> PlannedRegion
DefaultRegionOptimizer       -> DefaultRegionPlanner
RegionOptimizationContext    -> RegionPlanningContext
RegionOptimizationTrace      -> RegionPlanningTrace
CpuRegionOptimizationPolicy  -> CpuRegionPlanningPolicy
optimizedRegions             -> plannedRegions
```

Po fazi 3 nesmi ve zdrojich zustat zadny z levy-strannych symbolu.

### 8. Public Tensor lifecycle je uzka povolena integracni hranice

Public `Tensor` API dnes nabizi `prepare()`/`compute()` a proto musi znat verejny
prepared execution typ. Tuto zavislost nerusime ani neschovavame za reflection nebo
adapter. Povolene runtime importy v tensor stromu jsou presne:

```text
tensor/Tensor.java
  -> runtime.contract.ExecutionMode
  -> runtime.execution.PreparedExecution

tensor/internal/TensorExecution.java
  -> runtime.contract.ExecutionMode
  -> runtime.execution.PreparedExecution

tensor/storage/NativeMemoryAllocation.java
  -> runtime.memory.ExecutionResource
```

Runtime soucasne pracuje s `Tensor` a `TensorStorage`, takze jde o vedomou lifecycle
hranici soucasneho public API. Boundary test ji omezi na tyto tri soubory; zadny
jiny tensor soubor nesmi importovat runtime implementation packages. Trace se do
teto hranice nezapoji a nebude importovat ani `tensor.DataType`.

### 9. External compute provider je backend-neutral low-level leaf

Externi compute knihovna sdilena vice backendy patri pod `backend.provider`, ne pod
konkretni backend ani do obecneho `backend.blas`. OpenBLAS ma jedine cilove vlastnictvi:

```text
backend.provider.blas.openblas
  OpenBlasRuntime
  OpenBlasSymbols
  OpenBlasGemmLayout
  OpenBlasArrayGemm
  OpenBlasSegmentGemm
```

Tento package je low-level external compute provider sdileny `backend.cpu` a
`backend.cpu1`. Smi importovat pouze JDK API vcetne FFM. Nesmi importovat `config`,
`graph`, `planning`, `prepare`, `runtime`, `trace`, `tensor` ani zadny konkretni
`backend.cpu`, `backend.cpu1`, `backend.metal`, `backend.cuda` nebo `backend.opencl`.
Poskytuje pouze capability query, OpenBLAS thread control a prime GEMM entrypointy.
Nezna route, threshold, debug policy, tuning profil, fallback ani prepared artifact.

`BlasProvider` je data-only enum v `config.runtime.BlasProvider`. Nema parser ani
system-property logiku. `BlasRuntime` se bez nahrady odstrani. Vsechny route,
threshold, shape gate, debug a thread volby pochazeji z `BlasConfig` nebo tuning
profilu a jsou definitivne vyhodnoceny v prepare. Hot path ani provider je znovu
nectou. CPU adapter `backend.cpu.provider.linalg.matmul.blas.MatMulBlasBackend`
zustava CPU-owned a importuje sdileny OpenBLAS provider.

## Cilovy Strom

Scope nezahrnuje reorganizaci `tensor` ani vytvareni obecneho `backend.api`.
Tyto aspirativni adresare nejsou soucasti ciloveho stromu.

```text
src/main/java/
  tensor/                         # beze zmeny struktury
  operations/                     # beze zmeny struktury

  backend/
    contract/
      ComputeBackend.java
    select/
    partition/
    lowering/
    accelerator/
    provider/
      blas/
        openblas/
          OpenBlasRuntime.java
          OpenBlasSymbols.java
          OpenBlasGemmLayout.java
          OpenBlasArrayGemm.java
          OpenBlasSegmentGemm.java
    cpu/
      prepare/
      execution/
      kernels/
      nativecpu/                  # pouze CPU kernel/policy casti
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
    CompiledGraph.java            # jedina lifecycle facade
    model/
      AliasViewPolicy.java
      CompiledGradientBinding.java
      CompiledNode.java
      CompiledTensorDataSnapshot.java
      ConstantGradientValue.java
    compile/
      CompileArtifacts.java
      CompiledNodeSnapshotter.java
      CompiledProgram.java
      GraphCompiler.java
      GraphStructureContract.java
      canonical/
        SemanticForwardCanonicalizer.java
      publication/
      session/
    optimizer/

  planning/
    backend/
    descriptor/
    intent/
    memory/
    partition/
      cost/
    region/
      lowering/
      specialization/
    value/

  prepare/
    context/
    validation/
    orchestration/

  runtime/
    contract/
    execution/
    state/
    residency/
    memory/
      nativecpu/
      transfer/
    device/
      buffer/
    publication/
    runner/

  trace/
    ExecutionTrace.java
    compile/
    prepare/
    execution/
    backend/

  config/
    runtime/
      BlasProvider.java
  tuning/
  training/
  numerics/
  onnx/
  utils/
```

Po migraci nesmi existovat:

```text
src/main/java/graph/execution
src/main/java/graph/compile/descriptor
src/main/java/graph/compile/intent
src/main/java/graph/compile/planning
src/main/java/backend/prepare
src/main/java/backend/runtime
src/main/java/backend/memory
src/main/java/backend/blas
src/main/java/backend/ComputeEngine.java
```

## Cilovy Dependency Graph A Lifecycle Hranice

Sipka znamena `A smi importovat B`:

```text
backend.contract -> JDK
backend.provider.blas.openblas -> JDK/FFM
runtime.contract -> JDK
config.runtime.BlasProvider -> JDK

graph.model -> backend.contract, operations, tensor
trace -> runtime.contract

planning -> graph.model, operations, tensor, config, trace.compile
graph.optimizer -> graph.model, operations, tensor, trace.compile
backend.partition/lowering -> planning, graph.model, backend.contract
graph.compile -> graph.model, graph.optimizer, planning,
                 backend.partition/lowering, trace.compile

runtime.memory/device -> runtime.contract, tensor, backend.contract
runtime.state/residency -> runtime.memory/device, graph.model, planning.memory,
                           planning.descriptor, trace.execution
runtime.execution -> runtime.contract, runtime.state/residency, graph.model,
                     planning, trace
runtime.runner/publication -> runtime.execution/state, trace

prepare.context/validation -> graph.model, planning, backend.lowering,
                              runtime.execution, config
backend.<name>.prepare -> prepare.context/validation, planning,
                          runtime.execution, backend.<name>.kernels,
                          config.runtime, backend.provider.blas.openblas
prepare.orchestration -> graph.compile, planning, prepare.context/validation,
                         backend.<name>.prepare, runtime.execution, trace.prepare

graph.CompiledGraph -> graph.compile, planning.intent,
                       prepare.orchestration, runtime.execution, trace.compile

tensor.Tensor/tensor.internal.TensorExecution -> graph.CompiledGraph,
                                                 runtime.contract,
                                                 runtime.execution
tensor.storage.NativeMemoryAllocation -> runtime.memory.ExecutionResource
runtime -> tensor/tensor.storage
```

Posledni tri radky jsou uzce povolena public lifecycle/storage hranice. Neplati pro
zbytek tensor stromu a nesmi se rozsirit na runtime state, residency, publication,
runner ani backend implementace.

Zakazane prime importy:

```text
graph.model       -X-> planning, prepare, runtime, concrete backend
graph.optimizer   -X-> prepare, runtime, concrete backend
planning          -X-> graph.compile, graph.CompiledGraph, prepare, runtime,
                       concrete backend
trace             -X-> graph, planning, prepare, concrete backend
prepare.context   -X-> prepare.orchestration, concrete backend preparers
prepare.validation-X-> prepare.orchestration, concrete backend preparers
backend.*.prepare -X-> prepare.orchestration
runtime           -X-> backend.cpu/cpu1/metal/cuda/opencl
backend.provider.blas.openblas -X-> config, graph, planning, prepare, runtime,
                                    trace, tensor, backend.cpu/cpu1/metal/cuda/opencl
backend kernels   -X-> graph.optimizer, prepare.orchestration
tensor mimo tri povolene soubory -X-> runtime
```

## Lifecycle Po Migraci

```text
Tensor expression
  -> graph.CompiledGraph.compile(...)
  -> graph.compile.GraphCompiler
  -> graph.compile.CompiledNodeSnapshotter
  -> graph.optimizer
  -> planning.backend/partition/region/memory
  -> graph.compile.CompileArtifacts
  -> graph.CompiledGraph.prepare(...)
  -> prepare.orchestration.PreparedExecutionBuilder
  -> prepare.orchestration.BackendPrepareDispatcher
  -> backend.<name>.prepare
  -> runtime.execution.PreparedStepMetadata + PreparedStepExecutable
  -> runtime.execution.PreparedExecution
  -> runtime.runner.PreparedExecutionRunner
  -> PreparedStepExecutable.execute(...)
  -> backend hot path
```

Cpu1 priklad:

```text
planning:
  PlannedRegion + CompiledTensorDescriptor + BackendIntentPlan

prepare.orchestration:
  vybere CPU route a zavola Cpu1NodePreparer

backend.cpu1.prepare:
  Cpu1StorageAccessPlan
  Cpu1StorageKind
  Cpu1LayoutKind
  Cpu1KernelId
  Cpu1LaunchConfig
  Cpu1PreparedArtifact implements PreparedStepExecutable

BLAS prepare hranice:
  BlasConfig + tuning profile
  -> CPU/cpu1 prepare precte provider, thresholdy, shape gates, debug a threads
  -> prepare jednou dotaze backend.provider.blas.openblas.OpenBlasRuntime na capability
  -> route a capability/debug/thread/trace snapshot se ulozi do prepared CPU/cpu1 state
  -> runtime/hot path pouzije pouze prepared hodnoty
  -> CPU adapter nebo cpu1 kernel vola prime OpenBlasArrayGemm/OpenBlasSegmentGemm
  -> provider necte config ani system properties a nerozhoduje fallback

runtime.runner:
  artifact.execute(context)

backend.cpu1.exec/kernels:
  bez dtype/layout/storage rozhodovani, ktere uz probehlo v prepare
```

## Faze 0: Baseline A Hranice

Status: `[x]`

### Evidence A Aktualni Stav Faze 0

Stav k 2026-06-30:

- `[x]` Inventory prikazy byly spusteny a `./gradlew classes` prosel.
- `[x]` `PackageOwnershipBoundaryTest` je implementovany; vsechny 3 boundary testy prosly.
- `[x]` Baseline targeted testy `SourceTreeHygieneTest` a `PreparedExecutionBuildTest` prosly.
- `[x]` Commit `2add3e88` s presnou zpravou `test: lock package ownership migration boundaries`
  uspel a obsahuje pouze `PackageOwnershipBoundaryTest.java`, `PreparedExecutionBuildTest.java`
  a `SourceTreeHygieneTest.java`.

Faze 0 je `[x]`; implementace, targeted testy, boundary audit i commit jsou dokonceny.

### Task 0.1: Zachytit aktualni inventory

Status: `[x]` inventory dokonceno; baseline targeted testy prosly.

Pred editaci ulozit vystupy pouze do terminal logu, ne do repozitare:

```bash
rg --files src/main/java/graph src/main/java/backend/prepare \
  src/main/java/backend/runtime src/main/java/backend/memory
rg -n "graph\.execution|graph\.compile\.planning|graph\.compile\.descriptor|graph\.compile\.intent|backend\.prepare|backend\.runtime|backend\.memory" \
  src/main/java src/test/java
./gradlew classes
./gradlew test --tests SourceTreeHygieneTest
./gradlew test --tests PreparedExecutionBuildTest
```

### Task 0.2: Pridat `PackageOwnershipBoundaryTest`

Status: `[x]` implementovano a overeno 3 prochazejicimi boundary testy.

Vytvorit `src/test/java/PackageOwnershipBoundaryTest.java`. Ve fazi 0 trida
obsahuje helpery a pouze pravidla, ktera uz plati nad soucasnym stromem:

```text
operations nesmi importovat graph/runtime/backend internals
graph.optimizer nesmi importovat konkretni backend kernels
tensor nesmi importovat runtime residency implementaci
```

Kazda dalsi faze prida sve cilove aserce ve stejnem commitu jako move. Test nikdy
nesmi preskakovat kontrolu podle existence adresare a nesmi obsahovat vypnute
aserce. Definitivni podoba po fazi 9:

```java
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PackageOwnershipBoundaryTest {
    private static final Path MAIN = Path.of("src/main/java");

    @Test
    void compiledGraphIsTheOnlyGraphLifecycleFacade() throws IOException {
        List<String> all = importsUnder(MAIN.resolve("graph"), Set.of("prepare.", "runtime."));
        List<String> disallowed = all.stream()
                .filter(line -> !line.startsWith("src/main/java/graph/CompiledGraph.java:"))
                .toList();
        assertTrue(disallowed.isEmpty(), () -> "graph lifecycle imports outside CompiledGraph: " + disallowed);
    }

    @Test
    void graphModelIsAPlanningAndRuntimeLeaf() throws IOException {
        assertNoImports("graph/model", Set.of("planning.", "prepare.", "runtime.",
                "backend.cpu.", "backend.cpu1.", "backend.metal.", "backend.cuda.", "backend.opencl."));
    }

    @Test
    void planningDoesNotDependOnCompileRuntimeOrConcreteBackends() throws IOException {
        assertNoImports("planning", Set.of("graph.CompiledGraph", "graph.compile.", "prepare.", "runtime.",
                "backend.cpu.", "backend.cpu1.", "backend.metal.", "backend.cuda.", "backend.opencl."));
    }

    @Test
    void traceDtosDoNotDependOnTheirProducers() throws IOException {
        assertNoImports("trace", Set.of("tensor.", "graph.", "planning.", "prepare.",
                "backend."));
    }

    @Test
    void tensorRuntimeImportsAreLimitedToLifecycleAndStorageContracts() throws IOException {
        Map<String, Set<String>> allowedByFile = Map.of(
                "src/main/java/tensor/Tensor.java", Set.of(
                        "runtime.contract.ExecutionMode",
                        "runtime.execution.PreparedExecution"
                ),
                "src/main/java/tensor/internal/TensorExecution.java", Set.of(
                        "runtime.contract.ExecutionMode",
                        "runtime.execution.PreparedExecution"
                ),
                "src/main/java/tensor/storage/NativeMemoryAllocation.java", Set.of(
                        "runtime.memory.ExecutionResource"
                )
        );
        List<String> offenders = importsUnder(MAIN.resolve("tensor"), Set.of("runtime.")).stream()
                .filter(line -> allowedByFile.entrySet().stream().noneMatch(entry ->
                        line.startsWith(entry.getKey() + ":")
                                && entry.getValue().stream().anyMatch(line::contains)))
                .toList();
        assertTrue(offenders.isEmpty(), () -> "tensor runtime imports outside lifecycle boundary: " + offenders);
    }

    @Test
    void externalOpenBlasProviderIsLowLevelAndBackendNeutral() throws IOException {
        assertNoImports("backend/provider/blas/openblas", Set.of(
                "config.", "graph.", "planning.", "prepare.", "runtime.", "trace.", "tensor.",
                "backend.cpu.", "backend.cpu1.", "backend.metal.", "backend.cuda.", "backend.opencl."
        ));
        Path root = MAIN.resolve("backend/provider/blas/openblas");
        try (Stream<Path> paths = Files.walk(root)) {
            List<String> nonJdkImports = paths.filter(path -> path.toString().endsWith(".java"))
                    .flatMap(path -> {
                        try {
                            return Files.readAllLines(path).stream()
                                    .map(String::trim)
                                    .filter(line -> line.startsWith("import "))
                                    .filter(line -> !line.startsWith("import java.")
                                            && !line.startsWith("import static java."))
                                    .map(line -> path + ": " + line);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .toList();
            assertTrue(nonJdkImports.isEmpty(),
                    () -> "OpenBLAS provider has non-JDK imports: " + nonJdkImports);
        }
    }

    @Test
    void backendPreparersDoNotDependOnPrepareOrchestration() throws IOException {
        for (String backend : List.of("cpu", "cpu1", "metal", "cuda")) {
            assertNoImports("backend/" + backend + "/prepare", Set.of("prepare.orchestration."));
        }
    }

    @Test
    void runtimeDoesNotDispatchToConcreteBackends() throws IOException {
        assertNoImports("runtime", Set.of("backend.cpu.", "backend.cpu1.", "backend.metal.",
                "backend.cuda.", "backend.opencl.", "backend.ComputeEngine"));
    }

    @Test
    void removedPackageTreesStayRemoved() {
        List<Path> removed = List.of(
                MAIN.resolve("graph/execution"),
                MAIN.resolve("graph/compile/descriptor"),
                MAIN.resolve("graph/compile/intent"),
                MAIN.resolve("graph/compile/planning"),
                MAIN.resolve("backend/prepare"),
                MAIN.resolve("backend/runtime"),
                MAIN.resolve("backend/memory"),
                MAIN.resolve("backend/blas"),
                MAIN.resolve("backend/ComputeEngine.java")
        );
        assertEquals(List.of(), removed.stream().filter(Files::exists).map(Path::toString).toList());
    }

    private static void assertNoImports(String relativeRoot, Set<String> forbidden) throws IOException {
        List<String> offenders = importsUnder(MAIN.resolve(relativeRoot), forbidden);
        assertTrue(offenders.isEmpty(), () -> relativeRoot + " has forbidden imports: " + offenders);
    }

    private static List<String> importsUnder(Path root, Set<String> forbidden) throws IOException {
        if (!Files.exists(root)) {
            return List.of();
        }
        List<String> offenders = new ArrayList<>();
        try (Stream<Path> paths = Files.walk(root)) {
            for (Path path : paths.filter(Files::isRegularFile)
                    .filter(file -> file.toString().endsWith(".java")).toList()) {
                for (String line : Files.readAllLines(path)) {
                    String trimmed = line.trim();
                    if (!trimmed.startsWith("import ")) {
                        continue;
                    }
                    if (forbidden.stream().anyMatch(trimmed::contains)) {
                        offenders.add(path + ": " + trimmed);
                    }
                }
            }
        }
        return offenders;
    }
}
```

### Task 0.3: Baseline commit

Status: `[x]` commit `2add3e88` byl vytvoren s presnou zpravou
`test: lock package ownership migration boundaries`.

Commit scope:

```text
test: lock package ownership migration boundaries
```

## Faze 1: Dependency Leaves A Graph Model

Status: `[x]` implementace, targeted testy, boundary audit a commit jsou dokonceny.

### Evidence A Aktualni Stav Faze 1

Stav k 2026-06-30:

- `[x]` Presuny dependency leaf a compiled model typu, `CompiledNodeSnapshotter`, alias consumer rewrite
  a odstraneni `ExecutionMode` z `OptimizerState` jsou implementovany.
- `[x]` `./gradlew classes`, `CompiledGraphTraceTest`, `graph.optimizer.*`,
  `PackageOwnershipBoundaryTest`, `SourceTreeHygieneTest` a targeted snapshot/descriptor/runtime alias testy prosly.
- `[x]` Boundary a source audit ma 0 starych importu/referenci, 0 volani `CompiledNode.snapshot`,
  0 starych source souboru a 0 zakazanych importu v `backend.contract` a `graph.model`.
- `[x]` Commit `c385237f473b922201d1fb8b00e740b389cc703f` se zpravou
  `refactor: isolate compiled graph model dependencies` byl vytvoren a overen proti phase-only diffu.

### Task 1.1: Presun backend identity

```text
src/main/java/backend/ComputeBackend.java
  -> src/main/java/backend/contract/ComputeBackend.java
```

Zmenit package deklaraci a vsechny importy z `backend.ComputeBackend` na
`backend.contract.ComputeBackend`. Enum telo se nemeni.

`backend/ApproxMode.java` zustane na miste; neni soucasti graph/runtime cyklu.

### Task 1.2: Oddelit pure compiled model

Move map:

```text
graph/CompiledGradientBinding.java     -> graph/model/CompiledGradientBinding.java
graph/CompiledNode.java                -> graph/model/CompiledNode.java
graph/CompiledTensorDataSnapshot.java  -> graph/model/CompiledTensorDataSnapshot.java
graph/ConstantGradientValue.java       -> graph/model/ConstantGradientValue.java
graph/AliasViewPolicy.java             -> graph/model/AliasViewPolicy.java
graph/CompiledProgram.java             -> graph/compile/CompiledProgram.java
graph/SemanticForwardCanonicalizer.java
  -> graph/compile/canonical/SemanticForwardCanonicalizer.java
```

`CompiledNode.snapshot(...)` se presune do nove package-private tridy
`graph.compile.CompiledNodeSnapshotter`. Tim `graph.model.CompiledNode` prestane
importovat `planning.intent.BackendIntentPlan` a zustane hodnotovym modelem.

Cilovy vstup snapshotteru:

```java
package graph.compile;

final class CompiledNodeSnapshotter {
    private CompiledNodeSnapshotter() {
    }

    static List<CompiledNode> snapshot(
            List<Tensor> orderedGraph,
            BackendIntentPlan backendIntentPlan
    ) {
        // Presne puvodni telo CompiledNode.snapshot(...).
        // Konstrukce CompiledNode se vola pres package-visible static factory CompiledNode.create(...).
    }
}
```

Do `CompiledNode` pridat pouze package-public factory nelze, protoze je v jinem
package. Pouzit verejnou statickou factory s dokumentovanym compile-only ucelem:

```java
public static CompiledNode compiledSnapshot(
        int id,
        Operation operation,
        ComputeBackend backend,
        List<Integer> inputIds,
        int storageOwnerId,
        int[] shape,
        int[] strides,
        int storageOffset,
        DataType dataType,
        boolean backwardNode,
        boolean leaf,
        boolean requiresGrad,
        boolean trainableParameter,
        boolean contiguous,
        boolean hasStorageOffset,
        boolean gradientTarget,
        int flatDataSize,
        String label,
        CompiledTensorDataSnapshot staticDataSnapshot
) {
    return new CompiledNode(/* stejne argumenty */);
}
```

Nepridavat builder ani mutable model.

`AliasViewPolicy` po presunu ponecha jen overload pracujici s `Tensor` a
`Operation.OpType`. Overload s descriptor indexem se odstrani. Runtime pouzije
jiz vypocitany `CompiledNode.storageOwnerId()`:

```java
boolean aliasesInput0 = node.storageOwnerId() != node.id();
```

To odstrani zavislost compiled modelu na descriptor package a zachova jedno
compile-time rozhodnuti o aliasingu.

### Task 1.3: Upravit `CompiledGraph`

`CompiledGraph` zustane v `graph`. Importy modelu a compile outputu budou:

```java
import graph.compile.CompiledProgram;
import graph.compile.canonical.SemanticForwardCanonicalizer;
import graph.model.CompiledNode;
```

V teto fazi se jeste nemeni prepare/runtime package importy; ty se prepnou ve
fazich 6 a 7 ve stejnem commitu jako cilove tridy.

### Task 1.4: Odstranit runtime mode z optimizer state

`graph.optimizer.state.OptimizerState` dnes drzi `ExecutionMode` i
`supportsBackward`, ale optimizer nepouziva mode pro zadne rozhodnuti. Odstranit:

```text
record field executionMode
constructor normalization ExecutionMode.FORWARD
withExecutionMetadata(ExecutionMode, boolean, int)
import backend.runtime.ExecutionMode
```

Nahradit metodou obsahujici pouze pouzivana metadata:

```java
public OptimizerState withCompileMetadata(
        boolean supportsBackward,
        int forwardBoundaryNodeId
) {
    return new OptimizerState(
            graph,
            forwardOutput,
            supportsBackward,
            forwardBoundaryNodeId,
            rewriteMap,
            trace
    );
}
```

`CompileSession` prepnout z `withExecutionMetadata(...)` na
`withCompileMetadata(...)`. Tohle odstrani `graph.optimizer -> runtime` bez noveho
enumu nebo bridge kontraktu.

### Task 1.5: Overeni

```bash
./gradlew classes
./gradlew test --tests CompiledGraphTraceTest
./gradlew test --tests graph.optimizer.*
./gradlew test --tests PackageOwnershipBoundaryTest
rg -n "import graph\.(CompiledNode|CompiledProgram|CompiledGradientBinding|CompiledTensorDataSnapshot|ConstantGradientValue|AliasViewPolicy|SemanticForwardCanonicalizer)" src/main/java src/test/java
```

Commit: `[x]` `c385237f473b922201d1fb8b00e740b389cc703f`

```text
refactor: isolate compiled graph model dependencies
```

## Faze 2: Top-Level Trace

Status: `[x]` implementace Faze 2 je dokoncena; vsechny phase-owned validace prosly.

### Evidence A Aktualni Stav Faze 2

Stav k 2026-06-30:

- `[x]` Task 2.1: kompletni trace move map, runtime kontrakty a `StepExecutionTracer`
  byly fyzicky presunuty; stare source tridy byly odstraneny bez aliasu a fasad.
- `[x]` Task 2.2: dtype v execution/transfer/optimizer trace je lossless `String`
  snapshot pres `DataType.name()` a `src/main/java/trace` nema zadny `tensor.*` import.
- `[x]` Task 2.3: optimizer cost explanation, partition cost a GPU lowered manifest
  jsou trace-owned immutable snapshoty; optimizer/planning/backend lowering typy se
  konvertuji v producerech a reporty pouzivaji `TraceCostExplanationAdapter`,
  `TraceCostScoreAdapter` a `GpuLoweredRegionTraceRenderer`.
- `[x]` Task 2.4: vsechny main/test/report importy a konzumenti byly prepsany;
  boundary vyhledavani vratila 0 starych importu, 0 starych source souboru a 0
  `tensor.*`, planning nebo backend lowering zavislosti v novych trace DTO.
- `[x]` Task 2.5: `./gradlew classes`, `CompiledGraphTraceTest`,
  `ComputeModeTraceTest`, `BenchmarkSessionTest`, `GpuCoverageSummaryTest`,
  `CrossBackendRouterEvidenceTest`, `NativeCpuNonBlasBenchmarkGateTest`,
  `backend.cpu.nativecpu.NativeCpuElementwiseChainTest`, `TensorMutationGuardsTest`,
  `backend.metal.MetalBufferTraceSmokeTest`, `PackageOwnershipBoundaryTest` a
  `SourceTreeHygieneTest` prosly. `TrainingOptimizerTest` byl spusten, ale 4 testy
  (`requireNativeF32SgdUsesNativeRoute`, `requireNativeF32AdamUsesNativeRoute`,
  `requireNativeSgdRejectsUnsupportedDTypes`, `requireNativeAdamRejectsUnsupportedDTypes`)
  selhavaji pred optimizer trace v existujicim native CPU `MUL` route: konfigurace
  `REQUIRE_NATIVE` narazi na array fallback pro broadcast vstup `shape=[2], strides=[0],
  storageOffset=0`. Izolovany prikaz
  `./gradlew test --tests 'TrainingOptimizerTest.requireNativeF32SgdUsesNativeRoute'`
  reprodukuje stejny pre-trace stack v `StorageAwareBinaryElementwiseKernel` /
  `ElementwiseNativeSupport`; Faze 2 meni v tomto call path pouze package import
  runtime kontraktu a test meni jen dtype assertion z enumu na lossless String.
  Kvuli existujicim dirty native CPU/runtime zmenam nebyl tento nesouvisejici route
  opravovan ani revertovan.
- `[x]` `git diff --check` prosel. Zadny commit ani push nebyl proveden.

### Task 2.1: Kompletni trace move map

Status: `[x]`

```text
graph/execution/trace/CompileTrace.java
  -> trace/compile/CompileTrace.java
graph/execution/trace/PartitionCompileTrace.java
  -> trace/compile/PartitionCompileTrace.java
graph/execution/trace/PartitionDecisionTrace.java
  -> trace/compile/PartitionDecisionTrace.java
graph/optimizer/state/OptimizerTrace.java
  -> trace/compile/OptimizerTrace.java

graph/execution/trace/PrepareTrace.java
  -> trace/prepare/PrepareTrace.java
graph/execution/trace/BackendPrepareDiagnosticTrace.java
  -> trace/prepare/BackendPrepareDiagnosticTrace.java
graph/execution/trace/BackendSelectionTrace.java
  -> trace/prepare/BackendSelectionTrace.java
graph/execution/trace/BackendSelectionDecisionTrace.java
  -> trace/prepare/BackendSelectionDecisionTrace.java

graph/execution/trace/RunTrace.java
  -> trace/execution/RunTrace.java
graph/execution/trace/ExecutionStepTrace.java
  -> trace/execution/ExecutionStepTrace.java
graph/execution/trace/StepExecutionMetadata.java
  -> trace/execution/StepExecutionMetadata.java
graph/execution/trace/CpuMaterializationTrace.java
  -> trace/execution/CpuMaterializationTrace.java
graph/execution/trace/HostDeviceTransferTrace.java
  -> trace/execution/HostDeviceTransferTrace.java
graph/execution/trace/NativeCpuMemoryTrace.java
  -> trace/execution/NativeCpuMemoryTrace.java
graph/execution/trace/NativeOptimizerTrace.java
  -> trace/execution/NativeOptimizerTrace.java

graph/execution/trace/ComputeTraceMetadata.java
  -> trace/backend/ComputeTraceMetadata.java
graph/execution/trace/ConvTraceMetadata.java
  -> trace/backend/ConvTraceMetadata.java
graph/execution/trace/DispatchTraceMetadata.java
  -> trace/backend/DispatchTraceMetadata.java
graph/execution/trace/FusedTraceMetadata.java
  -> trace/backend/FusedTraceMetadata.java
graph/execution/trace/LayoutTraceMetadata.java
  -> trace/backend/LayoutTraceMetadata.java
graph/execution/trace/MatMulTraceMetadata.java
  -> trace/backend/MatMulTraceMetadata.java
graph/execution/trace/ReductionTraceMetadata.java
  -> trace/backend/ReductionTraceMetadata.java
graph/execution/trace/StepTraceContribution.java
  -> trace/backend/StepTraceContribution.java

graph/execution/trace/ExecutionTrace.java
  -> trace/ExecutionTrace.java

backend/runtime/ExecutionMode.java
  -> runtime/contract/ExecutionMode.java
backend/memory/CpuMaterializationReason.java
  -> runtime/contract/CpuMaterializationReason.java
backend/memory/StorageResidency.java
  -> runtime/contract/StorageResidency.java
graph/execution/trace/HostDeviceTransferKind.java
  -> runtime/contract/HostDeviceTransferKind.java

graph/execution/trace/contrib/StepExecutionTracer.java
  -> runtime/runner/StepExecutionTracer.java
```

Ctverice runtime enumu se presouva soucasne s trace, protoze trace DTO je potrebuji
a nesmi kvuli nim importovat puvodni backend/runtime package. `StepExecutionTracer`
se presouva soucasne, protoze je producer, nikoliv DTO. Do faze 6 bude importovat
stare `graph.execution` typy; faze 6 tyto importy prepise spolu s jejich presunem.

### Task 2.2: Snapshotovat dtype bez zavislosti na tensor

Status: `[x]`

Tri trace DTO dnes importuji `tensor.DataType`. Vsechna dtype pole se zmeni na
stabilni `String`, jehoz hodnotou je `DataType.name()`. Jde o lossless diagnosticky
snapshot; report stale dostane hodnoty `FLOAT32`, `FLOAT64`, `BFLOAT16`, `INT32`,
`INT64` nebo `BOOL`.

Konkretni zmeny signatures:

```java
// trace.execution.ExecutionStepTrace
public record ExecutionStepTrace(
        int index,
        String label,
        String opType,
        List<Integer> shape,
        String dataType,
        String backend,
        String kernel,
        long durationNs,
        StepExecutionMetadata metadata
) { }

// trace.execution.HostDeviceTransferTrace
public record HostDeviceTransferTrace(
        int nodeId,
        String backend,
        String dataType,
        StorageResidency sourceResidency,
        StorageResidency targetResidency,
        HostDeviceTransferKind transferKind,
        long bytes,
        long javaArrayBytes,
        long nativeBytes,
        long deviceBytes,
        long durationNs,
        boolean syncOnly,
        boolean directTransferSupported,
        boolean success,
        String fallbackReason,
        String detail
) { }

// trace.execution.NativeOptimizerTrace
public record NativeOptimizerTrace(
        String optimizer,
        String route,
        String dataType,
        int parameterNodeId,
        int gradientNodeId,
        int elementCount,
        String fallbackReason,
        String publicationPolicy,
        String gradientPublication,
        String optimizerStateStorage,
        String bf16TrainingPolicy,
        String nativeCpuFailurePolicy,
        String parameterResidencyBefore,
        String parameterResidencyAfter,
        String gradientResidencyBefore,
        String gradientResidencyAfter,
        String publicationSkippedReason
) { }
```

Canonical constructors normalizuji `dataType == null ? "" : dataType`.

Producer rewrites:

```text
runtime.runner.StepExecutionTracer
  node.dataType() -> node.dataType().name()

runtime.state.RuntimeMaterializationService
  target.getDataType() -> target.getDataType().name()

backend.metal.buffer.MetalAcceleratorBufferBinder
  descriptor/buffer dtype -> dtype.name()

backend.cuda.buffer.CudaAcceleratorBufferBinder
  descriptor/buffer dtype -> dtype.name()

training.optimizer.AbstractTrainableOptimizer
  parameter.getDataType() -> parameter.getDataType().name()
```

Consumer rewrites:

```text
tuning/benchmark/report/JsonBenchmarkReportRenderer
  odstranit .name() a serializovat jiz snapshotovany String

tuning/benchmark/report/TextBenchmarkReportRenderer
  zachovat prime append(String)

tuning/benchmark/report/Bf16PerformanceBenchmarkGate
tuning/benchmark/report/Bf16PerformanceSummary
  dataType() != DataType.BFLOAT16 -> !"BFLOAT16".equals(dataType())

TrainingOptimizerTest
  assertEquals(DataType.X, trace.dataType()) -> assertEquals("X", trace.dataType())

backend/cpu/nativecpu/NativeCpuElementwiseChainTest
  step.dataType() == DataType.FLOAT32 -> "FLOAT32".equals(step.dataType())

CrossBackendRouterEvidenceTest, GpuCoverageSummaryTest,
NativeCpuNonBlasBenchmarkGateTest
  synthetic trace constructors dostanou DataType.X.name()
```

Po zmene zadny soubor pod `src/main/java/trace` nesmi importovat `tensor.*`.

### Task 2.3: Odstranit trace-to-producer zavislosti

Status: `[x]`

`PartitionCompileTrace` a `PartitionDecisionTrace` nesmi importovat planning enumy
ani `AcceleratorPartitionScoreModel`. Pole `strategy` a `target` budou `String`.
Pridat trace-owned record:

```java
package trace.compile;

public record MaterializationCostTrace(
        String preset,
        int boundaryCount,
        long estimatedTransferBytes,
        long layoutFallbackBytes,
        long estimatedComputeWork,
        long avoidedIntermediateBytes,
        double dispatchCost,
        double finalScore,
        String reasonCode,
        String fallbackMode,
        String layoutClass
) {
    public MaterializationCostTrace {
        preset = preset == null ? "" : preset;
        boundaryCount = Math.max(0, boundaryCount);
        estimatedTransferBytes = Math.max(0L, estimatedTransferBytes);
        layoutFallbackBytes = Math.max(0L, layoutFallbackBytes);
        estimatedComputeWork = Math.max(0L, estimatedComputeWork);
        avoidedIntermediateBytes = Math.max(0L, avoidedIntermediateBytes);
        dispatchCost = Math.max(0.0d, dispatchCost);
        reasonCode = reasonCode == null ? "" : reasonCode;
        fallbackMode = fallbackMode == null ? "" : fallbackMode;
        layoutClass = layoutClass == null ? "" : layoutClass;
    }
}
```

Planning producer provede explicitni snapshot:

```java
private static MaterializationCostTrace traceCost(MaterializationCostSummary source) {
    if (source == null) {
        return null;
    }
    return new MaterializationCostTrace(
            source.preset(), source.boundaryCount(), source.estimatedTransferBytes(),
            source.layoutFallbackBytes(), source.estimatedComputeWork(),
            source.avoidedIntermediateBytes(), source.dispatchCost(), source.finalScore(),
            source.reasonCode(), source.fallbackMode(), source.layoutClass()
    );
}
```

Metody `toCostScore()` se z trace DTO odstrani. Report adapter vytvori `CostScore`
z `MaterializationCostTrace`; diagnosticky model nebude importovat optimizer.

Pridat package-private
`tuning.benchmark.report.TraceCostScoreAdapter` se dvema konkretnimi overloady:

```java
final class TraceCostScoreAdapter {
    private TraceCostScoreAdapter() {
    }

    static CostScore toCostScore(MaterializationCostTrace trace) {
        Objects.requireNonNull(trace, "trace cannot be null");
        return CostScore.of(
                "AcceleratorPartitionCostModel",
                "accelerator-partition-materialization",
                List.of(
                        CostComponent.higherIsBetter("finalScore", trace.finalScore(),
                                "materialization-aware accelerator partition score"),
                        CostComponent.higherIsBetter("estimatedComputeWork", trace.estimatedComputeWork(),
                                "larger accelerator work can amortize dispatch and transfer cost"),
                        CostComponent.higherIsBetter("avoidedIntermediateBytes", trace.avoidedIntermediateBytes(),
                                "intermediate bytes retained inside the accelerator region"),
                        CostComponent.lowerIsBetter("boundaryCount", trace.boundaryCount(),
                                "CPU/accelerator boundaries introduce handoff cost"),
                        CostComponent.lowerIsBetter("estimatedTransferBytes", trace.estimatedTransferBytes(),
                                "estimated bytes copied across accelerator boundaries"),
                        CostComponent.lowerIsBetter("layoutFallbackBytes", trace.layoutFallbackBytes(),
                                "bytes affected by layout fallback or dense materialization"),
                        CostComponent.lowerIsBetter("dispatchCost", trace.dispatchCost(),
                                "fixed accelerator dispatch cost applied by the preset"),
                        CostComponent.informational("preset", 0.0d, trace.preset()),
                        CostComponent.informational("fallbackMode", 0.0d, trace.fallbackMode()),
                        CostComponent.informational("layoutClass", 0.0d, trace.layoutClass())
                )
        );
    }

    static CostScore toCostScore(PartitionDecisionTrace.CandidateCostTrace trace) {
        Objects.requireNonNull(trace, "trace cannot be null");
        return CostScore.of(
                "AcceleratorPartitionCostModel",
                "accelerator-partition-finalist",
                List.of(
                        CostComponent.higherIsBetter("finalScore", trace.finalScore(),
                                "materialization-aware accelerator partition finalist score"),
                        CostComponent.higherIsBetter("estimatedComputeWork", trace.estimatedComputeWork(),
                                "larger accelerator work can amortize dispatch and transfer cost"),
                        CostComponent.lowerIsBetter("boundaryCount", trace.boundaryCount(),
                                "CPU/accelerator boundaries introduce handoff cost"),
                        CostComponent.lowerIsBetter("estimatedTransferBytes", trace.estimatedTransferBytes(),
                                "estimated bytes copied across accelerator boundaries"),
                        CostComponent.lowerIsBetter("layoutFallbackBytes", trace.layoutFallbackBytes(),
                                "bytes affected by layout fallback or dense materialization"),
                        CostComponent.informational("preset", 0.0d, trace.preset())
                )
        );
    }
}
```

Text a JSON renderer pouziji tento adapter. Planning-owned
`AcceleratorPartitionScoreModel.MaterializationCostSummary.toCostScore()` zustane,
protoze neni soucast trace DTO a pouzivaji ho planning cost testy.

`BackendSelectionDecisionTrace` nesmi drzet
`backend.accelerator.lowering.GpuLoweredRegionManifest`. Pridat lossless snapshot;
zadna informace pouzivana tuning reporty se nesmi zahodit:

```java
package trace.prepare;

public record GpuLoweredRegionTrace(
        String regionId,
        String backend,
        int anchorNodeId,
        List<Integer> orderedNodeIds,
        List<Integer> externalInputNodeIds,
        List<Integer> outputNodeIds,
        int selectedRegionLength,
        List<OriginalOperation> originalOperations,
        List<LoweredPrimitive> loweredPrimitives,
        List<ValueAssumption> inputAssumptions,
        List<ValueAssumption> outputAssumptions,
        CompoundSummary compoundSummary,
        List<FusedSubpattern> fusedSubpatterns,
        List<Rejection> rejections,
        CandidateSpan candidateSpan,
        Map<String, String> backendExtensions
) {
    public GpuLoweredRegionTrace {
        regionId = regionId == null ? "" : regionId;
        backend = backend == null ? "" : backend;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        selectedRegionLength = Math.max(0, selectedRegionLength);
        originalOperations = List.copyOf(originalOperations == null ? List.of() : originalOperations);
        loweredPrimitives = List.copyOf(loweredPrimitives == null ? List.of() : loweredPrimitives);
        inputAssumptions = List.copyOf(inputAssumptions == null ? List.of() : inputAssumptions);
        outputAssumptions = List.copyOf(outputAssumptions == null ? List.of() : outputAssumptions);
        compoundSummary = compoundSummary == null ? CompoundSummary.none() : compoundSummary;
        fusedSubpatterns = List.copyOf(fusedSubpatterns == null ? List.of() : fusedSubpatterns);
        rejections = List.copyOf(rejections == null ? List.of() : rejections);
        candidateSpan = candidateSpan == null ? CandidateSpan.empty() : candidateSpan;
        backendExtensions = Map.copyOf(backendExtensions == null ? Map.of() : backendExtensions);
    }

    public record OriginalOperation(
            int nodeId, String opType, List<Integer> inputNodeIds,
            List<Integer> outputNodeIds, String dataType, List<Integer> shape,
            List<String> loweredPrimitiveIds, List<String> reasons
    ) {
        public OriginalOperation {
            opType = opType == null ? "UNKNOWN" : opType;
            inputNodeIds = List.copyOf(inputNodeIds == null ? List.of() : inputNodeIds);
            outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
            dataType = dataType == null ? "" : dataType;
            shape = List.copyOf(shape == null ? List.of() : shape);
            loweredPrimitiveIds = List.copyOf(loweredPrimitiveIds == null ? List.of() : loweredPrimitiveIds);
            reasons = List.copyOf(reasons == null ? List.of() : reasons);
        }
    }

    public record LoweredPrimitive(
            String primitiveId, String primitiveType, List<Integer> sourceOriginalNodeIds,
            List<String> inputRefs, String outputRef, String dataType,
            List<Integer> shape, List<String> reasons
    ) {
        public LoweredPrimitive {
            primitiveId = primitiveId == null ? "" : primitiveId;
            primitiveType = primitiveType == null ? "UNKNOWN" : primitiveType;
            sourceOriginalNodeIds = List.copyOf(sourceOriginalNodeIds == null ? List.of() : sourceOriginalNodeIds);
            inputRefs = List.copyOf(inputRefs == null ? List.of() : inputRefs);
            outputRef = outputRef == null ? "" : outputRef;
            dataType = dataType == null ? "" : dataType;
            shape = List.copyOf(shape == null ? List.of() : shape);
            reasons = List.copyOf(reasons == null ? List.of() : reasons);
        }
    }

    public record ValueAssumption(
            int nodeId, String role, String dataType, int rank, List<Integer> shape,
            String layout, boolean contiguous, boolean hasStorageOffset, long storageOffset
    ) {
        public ValueAssumption {
            role = role == null ? "UNKNOWN" : role;
            dataType = dataType == null ? "" : dataType;
            rank = Math.max(0, rank);
            shape = List.copyOf(shape == null ? List.of() : shape);
            layout = layout == null ? "UNKNOWN" : layout;
            storageOffset = Math.max(0L, storageOffset);
        }
    }

    public record CompoundSummary(
            String backend, String patternType, boolean supported, String reason,
            List<Integer> orderedNodeIds, List<Integer> externalInputNodeIds,
            List<Integer> outputNodeIds, List<String> dagNodeTypes,
            List<String> postOps, String detail
    ) {
        public CompoundSummary {
            backend = backend == null ? "" : backend;
            patternType = patternType == null ? "NONE" : patternType;
            reason = reason == null ? "" : reason;
            orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
            externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
            outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
            dagNodeTypes = List.copyOf(dagNodeTypes == null ? List.of() : dagNodeTypes);
            postOps = List.copyOf(postOps == null ? List.of() : postOps);
            detail = detail == null ? "" : detail;
        }

        public static CompoundSummary none() {
            return new CompoundSummary("", "NONE", false, "", List.of(), List.of(),
                    List.of(), List.of(), List.of(), "");
        }
    }

    public record FusedSubpattern(
            String patternType, boolean supported, List<Integer> originalOperationNodeIds,
            List<String> loweredPrimitiveIds, int loweredPrimitiveCount,
            String reason, String detail
    ) {
        public FusedSubpattern {
            patternType = patternType == null ? "NONE" : patternType;
            originalOperationNodeIds = List.copyOf(
                    originalOperationNodeIds == null ? List.of() : originalOperationNodeIds
            );
            loweredPrimitiveIds = List.copyOf(loweredPrimitiveIds == null ? List.of() : loweredPrimitiveIds);
            loweredPrimitiveCount = Math.max(0, loweredPrimitiveCount);
            reason = reason == null ? "" : reason;
            detail = detail == null ? "" : detail;
        }
    }

    public record Rejection(
            String level, int originalNodeId, String primitiveId,
            String fusedPatternType, String reason, String detail
    ) {
        public Rejection {
            level = level == null ? "UNKNOWN" : level;
            primitiveId = primitiveId == null ? "" : primitiveId;
            fusedPatternType = fusedPatternType == null ? "" : fusedPatternType;
            reason = reason == null ? "" : reason;
            detail = detail == null ? "" : detail;
        }
    }

    public record CandidateSpan(
            List<Integer> originalCandidateNodeIds, List<Integer> acceptedNodeIds,
            int rejectedOriginalNodeId, String rejectedPrimitiveId, String reason
    ) {
        public CandidateSpan {
            originalCandidateNodeIds = List.copyOf(
                    originalCandidateNodeIds == null ? List.of() : originalCandidateNodeIds
            );
            acceptedNodeIds = List.copyOf(acceptedNodeIds == null ? List.of() : acceptedNodeIds);
            rejectedPrimitiveId = rejectedPrimitiveId == null ? "" : rejectedPrimitiveId;
            reason = reason == null ? "" : reason;
        }

        public static CandidateSpan empty() {
            return new CandidateSpan(List.of(), List.of(), -1, "", "");
        }
    }
}
```

Canonical konstruktory vsech nested recordu doplni stejne null normalization jako
puvodni manifest recordy. Konverze z manifestu bude private metoda v
`backend.select.DefaultBackendSelectionPolicy` a zkopiruje vsechna pole 1:1,
vcetne dtype/reason enumu pres `.name()`. Tuning reporty a testy se prepnou na
snapshot pole; coverage, dtype, layout, fusion a rejection reporty musi zustat
obsahove shodne.

Definitivni `BackendSelectionDecisionTrace` nebude importovat planning ani backend
lowering:

```java
package trace.prepare;

public record BackendSelectionDecisionTrace(
        int anchorNodeId,
        List<Integer> nodeIds,
        List<String> compatibleBackends,
        boolean selected,
        String selectedBackend,
        String reason,
        long estimatedWork,
        MaterializationCostTrace costSummary,
        List<PartitionDecisionTrace.CandidateCostTrace> finalists,
        GpuLoweredRegionTrace gpuLoweredRegionManifest
) {
    public BackendSelectionDecisionTrace {
        nodeIds = List.copyOf(nodeIds == null ? List.of() : nodeIds);
        compatibleBackends = List.copyOf(compatibleBackends == null ? List.of() : compatibleBackends);
        selectedBackend = selectedBackend == null ? "" : selectedBackend;
        reason = reason == null ? "" : reason;
        estimatedWork = Math.max(0L, estimatedWork);
        finalists = List.copyOf(finalists == null ? List.of() : finalists).stream()
                .limit(3)
                .toList();
        if (!selected) {
            gpuLoweredRegionManifest = null;
        }
    }
}
```

Trace snapshot consumers, ktere se ve stejnem tasku prepnou z backend manifestu:

```text
tuning/benchmark/report/GpuCoverageSummary.java
tuning/benchmark/report/JsonBenchmarkReportRenderer.java
tuning/benchmark/report/TextBenchmarkReportRenderer.java
CompiledGraphTraceTest
PreparedExecutionBuildTest
CrossBackendRouterEvidenceTest
GpuCoverageRegressionGateTest
GpuCoverageSummaryTest
BenchmarkSessionTest
debug/TransformerMetalPartitionAnalysisTest
```

Pro text report pridat `tuning.benchmark.report.GpuLoweredRegionTraceRenderer`.
Puvodni backend renderer zustane pouze pro backend lowering manifest a nebude mit
overload pro trace snapshot. Tohle udrzi backend lowering model a report DTO oddelene.

### Task 2.4: Import rewrite

Status: `[x]`

```text
graph.execution.trace.CompileTrace                 -> trace.compile.CompileTrace
graph.execution.trace.PartitionCompileTrace        -> trace.compile.PartitionCompileTrace
graph.execution.trace.PartitionDecisionTrace       -> trace.compile.PartitionDecisionTrace
graph.optimizer.state.OptimizerTrace               -> trace.compile.OptimizerTrace
graph.execution.trace.PrepareTrace                 -> trace.prepare.PrepareTrace
graph.execution.trace.Backend*Trace                -> trace.prepare.Backend*Trace
graph.execution.trace.RunTrace                     -> trace.execution.RunTrace
graph.execution.trace.ExecutionStepTrace           -> trace.execution.ExecutionStepTrace
graph.execution.trace.StepExecutionMetadata        -> trace.execution.StepExecutionMetadata
graph.execution.trace.*Materialization/Transfer*   -> trace.execution.*
graph.execution.trace.Native*Trace                 -> trace.execution.*
graph.execution.trace.*TraceMetadata               -> trace.backend.*TraceMetadata
graph.execution.trace.StepTraceContribution        -> trace.backend.StepTraceContribution
graph.execution.trace.ExecutionTrace               -> trace.ExecutionTrace
backend.runtime.ExecutionMode                      -> runtime.contract.ExecutionMode
backend.memory.CpuMaterializationReason            -> runtime.contract.CpuMaterializationReason
backend.memory.StorageResidency                    -> runtime.contract.StorageResidency
graph.execution.trace.HostDeviceTransferKind       -> runtime.contract.HostDeviceTransferKind
```

Protoze se `ExecutionMode` fyzicky presouva v teto fazi, soucasne prepsat jeho
public Tensor lifecycle konzumenty:

```text
src/main/java/tensor/Tensor.java
  backend.runtime.ExecutionMode -> runtime.contract.ExecutionMode
src/main/java/tensor/internal/TensorExecution.java
  backend.runtime.ExecutionMode -> runtime.contract.ExecutionMode
```

`PreparedExecution` import v techto dvou souborech se prepne ve fazi 6, kdy se
fyzicky presune jeho trida.

### Task 2.5: Overeni

Status: `[x]` vsechny phase-owned kontroly prosly; nesouvisejici dirty-worktree
native CPU blocker je zaznamenan v evidence vyse.

```bash
./gradlew classes
./gradlew test --tests CompiledGraphTraceTest
./gradlew test --tests ComputeModeTraceTest
./gradlew test --tests BenchmarkSessionTest
./gradlew test --tests TrainingOptimizerTest
./gradlew test --tests GpuCoverageSummaryTest
./gradlew test --tests CrossBackendRouterEvidenceTest
./gradlew test --tests NativeCpuNonBlasBenchmarkGateTest
./gradlew test --tests backend.cpu.nativecpu.NativeCpuElementwiseChainTest
./gradlew test --tests TensorMutationGuardsTest
./gradlew test --tests backend.metal.MetalBufferTraceSmokeTest
./gradlew test --tests PackageOwnershipBoundaryTest
rg -n "graph\.execution\.trace|graph\.optimizer\.state\.OptimizerTrace|backend\.runtime\.ExecutionMode|backend\.memory\.(CpuMaterializationReason|StorageResidency)" \
  src/main/java src/test/java \
  --glob '!SourceTreeHygieneTest.java' --glob '!PackageOwnershipBoundaryTest.java'
rg -n '^import tensor\.' src/main/java/trace
```

Commit:

```text
refactor: extract lifecycle trace model
```

## Faze 3: Kompletni Planning

Status: `[x]`

### Task 3.1: Descriptor a intent

```text
graph/compile/descriptor/CompiledTensorDescriptor.java        -> planning/descriptor/CompiledTensorDescriptor.java
graph/compile/descriptor/CompiledTensorDescriptorBuilder.java -> planning/descriptor/CompiledTensorDescriptorBuilder.java
graph/compile/descriptor/CompiledTensorDescriptorIndex.java   -> planning/descriptor/CompiledTensorDescriptorIndex.java
graph/compile/descriptor/LayoutClass.java                     -> planning/descriptor/LayoutClass.java

graph/compile/intent/BackendIntentPlan.java       -> planning/intent/BackendIntentPlan.java
graph/compile/intent/BackendIntentPropagator.java -> planning/intent/BackendIntentPropagator.java
```

### Task 3.2: Root planning package

```text
graph/compile/planning/BackendPlanningJob.java                  -> planning/backend/BackendPlanningJob.java
graph/compile/planning/BackendPlanningJobResolver.java          -> planning/backend/BackendPlanningJobResolver.java
graph/compile/planning/BackendPlanningRequest.java              -> planning/backend/BackendPlanningRequest.java
graph/compile/planning/BackendPlanningRequirementValidator.java -> planning/backend/BackendPlanningRequirementValidator.java
graph/compile/planning/BackendPlanningResult.java               -> planning/backend/BackendPlanningResult.java
graph/compile/planning/BackendPlanningService.java              -> planning/backend/BackendPlanningService.java
graph/compile/planning/ExplicitBackendIntent.java               -> planning/backend/ExplicitBackendIntent.java
```

Planning dnes importuje konkretni `BackendPartitionDescriptorRegistry`, ktery
zpet importuje planning typy. Zavest planning-owned kontrakt:

```java
package planning.backend;

public interface BackendPartitionCapabilityRegistry {
    BackendPartitionCapability partitionCapabilityFor(PartitionTarget target);
}
```

`backend.partition.BackendPartitionDescriptorRegistry` ho implementuje.
`BackendPlanningRequest` drzi interface, ne konkretni registry. `GraphCompiler`
vytvori defaults registry a injectne ji do requestu. Lowerer registry zustane
v `backend.partition` a pouzije se az v compile/lowering composition vrstve.

`planning.partition.PartitionPlan` dnes vraci backend-owned
`GpuLoweredRegionManifest`, cimz vytvari opacnou hranu z planning do backend
loweringu. Metodu `gpuLoweredRegionManifest()` z `PartitionPlan` odstranit.
Pridat backend-owned rozsireni:

```java
package backend.accelerator.lowering;

public interface AcceleratorPartitionPlan extends PartitionPlan {
    GpuLoweredRegionManifest gpuLoweredRegionManifest();
}
```

`MetalPartitionPlan` a `CudaGpuPartitionPlan` implementuji
`AcceleratorPartitionPlan`. `DefaultBackendSelectionPolicy` muze jako composition
code pouzit `instanceof AcceleratorPartitionPlan`; planning package tento backend
typ neimportuje.

`planning.memory.MemoryPlanningInput` dnes drzi zaroven `ExecutionMode` a
`supportsBackward`. Jde o duplicitni informaci a planning kvuli ni importuje
runtime enum. Odstranit field `executionMode`; memory planning pouzije pouze
`supportsBackward`. Z `graph.compile.session.CompileSession` odstranit konstrukci
a import runtime `ExecutionMode` pro memory planning. Tim planning neimportuje
runtime.

### Task 3.3: Memory package inventory

Vsechny soubory se presunou mechanicky do `planning.memory`:

```text
MaterializationPlanEntry, MemoryPlan, MemoryPlanSummary,
MemoryPlanSummaryBuilder, MemoryPlanner, MemoryPlannerPolicy,
MemoryPlanningInput, MemoryRole, NodeLifetime,
RegionBindingAllocator, RegionBindingAssignment,
RegionHandoffPlanner, RegionHandoffRequirement,
RegionMemoryBinding, RegionMemoryBindingKind, RegionMemoryPlan,
RegionValueFlowPlan, RegionValueFlowPlanner, RegionValueLifetime,
ReusableInterval, ReusableIntervalBuilder, ReusableSlotAllocator,
ReusableSlotAssignment, RuntimeBindingPlan, RuntimeMemoryBindingPolicy,
RuntimeMemoryBindingPolicyPlanner, StructuralMemoryView,
StructuralValueFlow, TensorLifetimePlan, TensorLifetimePlanner,
TensorMemoryPlan
```

Zdroj je vzdy `graph.compile.planning.memory`, cil `planning.memory`.

### Task 3.4: Partition package inventory

Do `planning.partition` presunout:

```text
BackendPartitionCapability, CpuNaturalExecutionRegionPlanner,
ExecutionRegionKind, MaxRegionPartitionPlanner, Partition,
PartitionAssembly, PartitionBoundaryReason, PartitionCandidate,
PartitionEdge, PartitionPlan, PartitionPlanner, PartitionPlannerStrategy,
PartitionPlanningContext, PartitionPlanningRequest, PartitionPlanningResult,
PartitionSourcePolicy, PartitionTarget, PartitionValue, PlannedPartition,
RegionExpansionPolicy, ScoredCandidatePartitionPlanner,
UnsupportedBackendPartitionCapability
```

Do `planning.partition.cost` presunout:

```text
AcceleratorPartitionScoreModel
```

### Task 3.5: Region package inventory a definitivni rename

Move + rename:

```text
CpuRegionOptimizationPolicy.java -> planning/region/CpuRegionPlanningPolicy.java
DefaultRegionOptimizer.java      -> planning/region/DefaultRegionPlanner.java
OptimizedRegion.java             -> planning/region/PlannedRegion.java
RegionOptimizationContext.java   -> planning/region/RegionPlanningContext.java
RegionOptimizationTrace.java     -> planning/region/RegionPlanningTrace.java
```

Prime move bez rename:

```text
ElementwiseFusionPlanner, ExecutionUnit, ExecutionUnitFactory,
ExecutionUnitKind, MaterializationDecision, RegionValue,
StructuralRegionUnitPlanner, ValueTransportKind, ValueTypeContract
```

Do `planning.region.lowering`:

```text
OperationSemanticClassifier, OperationSemanticLevel
```

Do `planning.region.specialization`:

```text
DefaultRegionSpecializationCapability,
EmptyRegionSpecializationPayload,
MatmulBiasReluSpecializationDetector,
MatmulBiasSpecializationDetector,
MatmulReluSpecializationDetector,
MseLossSpecializationDetector,
RegionSpecializationCandidate,
RegionSpecializationCapability,
RegionSpecializationDecision,
RegionSpecializationKind,
RegionSpecializationPayload,
RegionSpecializationPlanner,
RegionSpecializationResult,
SdpaBackwardOutputKind,
SdpaBackwardSpecializationDetector,
SdpaBackwardSpecializationPayload
```

Do `planning.value`:

```text
GraphValueKind, GraphValueRef
```

### Task 3.6: Method a field rename

Prepsat vsechny declarations/call sites:

```text
optimizedRegions()       -> plannedRegions()
optimizedRegion          -> plannedRegion
optimizeRegion(...)      -> planRegion(...)
optimizationContext      -> planningContext
optimizationTrace        -> planningTrace
```

Nazvy tykajici se skutecne graph optimizer faze se nemeni.

### Task 3.7: Test move map

```text
test/graph/compile/descriptor/CompiledTensorDescriptorIndexTest.java
  -> test/planning/descriptor/CompiledTensorDescriptorIndexTest.java
test/graph/compile/planning/BackendPlanningRequirementValidatorTest.java
  -> test/planning/backend/BackendPlanningRequirementValidatorTest.java
test/graph/compile/planning/memory/MemoryPlannerRegionViewTest.java
  -> test/planning/memory/MemoryPlannerRegionViewTest.java
test/graph/compile/planning/partition/CpuNaturalExecutionRegionPlannerTest.java
  -> test/planning/partition/CpuNaturalExecutionRegionPlannerTest.java
test/graph/compile/planning/partition/MaxRegionPartitionPlannerTest.java
  -> test/planning/partition/MaxRegionPartitionPlannerTest.java
test/graph/compile/planning/partition/cost/AcceleratorPartitionScoreModelTest.java
  -> test/planning/partition/cost/AcceleratorPartitionScoreModelTest.java
test/graph/compile/planning/region/DefaultRegionOptimizerTest.java
  -> test/planning/region/DefaultRegionPlannerTest.java
test/graph/compile/planning/region/DefaultRegionOptimizerServiceTest.java
  -> test/planning/region/DefaultRegionPlannerServiceTest.java
```

Ve stejnem tasku aktualizovat `SourceTreeHygieneTest` metody, ktere ctou
`graph/compile/planning/partition` nebo porovnavaji stare planning import stringy.
Jinak by targeted hygiene test po fyzickem move cetl neexistujici adresar.

### Task 3.8: Overeni

```bash
./gradlew classes
./gradlew test --tests planning.descriptor.*
./gradlew test --tests planning.backend.*
./gradlew test --tests planning.memory.*
./gradlew test --tests planning.partition.*
./gradlew test --tests planning.region.*
./gradlew test --tests CompiledGraphTraceTest
./gradlew test --tests PackageOwnershipBoundaryTest
rg -n "graph\.compile\.(descriptor|intent|planning)|OptimizedRegion|DefaultRegionOptimizer|RegionOptimizationContext|RegionOptimizationTrace|CpuRegionOptimizationPolicy|optimizedRegions" \
  src/main/java src/test/java \
  --glob '!SourceTreeHygieneTest.java' --glob '!PackageOwnershipBoundaryTest.java'
```

Validation evidence (2026-06-30):

- `./gradlew classes` -- BUILD SUCCESSFUL.
- `./gradlew test --tests planning.descriptor.*` -- BUILD SUCCESSFUL.
- `./gradlew test --tests planning.backend.*` -- BUILD SUCCESSFUL.
- `./gradlew test --tests planning.memory.*` -- BUILD SUCCESSFUL.
- `./gradlew test --tests planning.partition.*` -- BUILD SUCCESSFUL.
- `./gradlew test --tests planning.region.*` -- BUILD SUCCESSFUL (re-run after final naming cleanup).
- `./gradlew test --tests CompiledGraphTraceTest` -- BUILD SUCCESSFUL.
- `./gradlew test --tests PackageOwnershipBoundaryTest` -- BUILD SUCCESSFUL.
- `./gradlew test --tests SourceTreeHygieneTest` -- BUILD SUCCESSFUL.
- Phase 3 legacy symbol/import audit above -- no matches.
- Planning dependency audit for concrete backend, backend composition/lowering, runtime,
  prepare, and `graph.compile` imports -- no matches.
- `git diff --check` -- clean.

Commit:

```text
refactor: extract complete compile planning model
```

## Faze 4: Runtime Memory A Device Kontrakty

Status: `[x]` implementace Faze 4 je dokoncena; vsechny phase-owned validace prosly.

### Evidence A Aktualni Stav Faze 4

Stav k 2026-06-30:

- `[x]` Task 4.1: vsech osm zbyvajicich trid z `backend.memory` bylo presunuto do
  `runtime.memory`, `runtime.device.buffer`, `runtime.residency` a
  `runtime.memory.transfer`; enumy presunute ve Fazi 2 zustaly jedine kopie.
  `tensor.storage.NativeMemoryAllocation` nyni extends
  `runtime.memory.ExecutionResource` a stary `backend.memory` strom byl odstranen.
- `[x]` Task 4.2: vsech dvacet backend-neutral accelerator buffer trid a tri jejich
  package-local testy bylo atomicky presunuto do `runtime.device.buffer`; vsichni
  Metal, CUDA, graph execution, prepare, training a test konzumenti pouzivaji novy
  finalni package bez aliasu nebo forwarding wrapperu.
- `[x]` Task 4.3: allocator, allocation, materializer, pool, statistiky, storage
  factory a trace state byly presunuty do `runtime.memory.nativecpu`; backend package
  obsahuje jen compute/policy tridy a `layout/**`. `NativeCpuStorageTest` byl presunut
  do runtime package a ostatni backend native CPU testy zustaly na puvodnich mistech
  s explicitnimi runtime memory importy.
- `[x]` Task 4.4: transfer matrix i support enum jsou v
  `runtime.memory.transfer`, pouzivaji leaf enumy z `runtime.contract` a runtime
  memory nema trace import.
- `[x]` Task 4.5: residency a transfer testy byly presunuty do finalnich runtime
  packages. `PackageOwnershipBoundaryTest` nyni vynucuje runtime-to-concrete-backend
  zakaz, presny tensor runtime whitelist a neexistenci legacy memory/buffer stromu.
- `[x]` Task 4.6: `./gradlew classes` prosel. Jednim Gradle během prosly vsechny
  explicitni targeted filtry teto tasky, vcetne platnych wildcard filtru
  `backend.metal.buffer.*` a `backend.cuda.buffer.*`; samostatny spolecny beh
  `PackageOwnershipBoundaryTest` a `SourceTreeHygieneTest` take prosel.
  Legacy symbol audit vratil 0 vysledku, runtime concrete-backend import audit vratil
  0 vysledku, stare source adresare neexistuji a `git diff --check` je cisty.
  Zadny commit ani push nebyl proveden.

### Task 4.1: Uplny audit `backend.memory`

Status: `[x]`

Kazdy aktualni soubor ma jeden konkretni cil:

| Zdroj | Cil |
|---|---|
| `CpuMaterializationReason.java` | `runtime.contract.CpuMaterializationReason` |
| `CpuMaterializationResult.java` | `runtime.memory.CpuMaterializationResult` |
| `DeviceBufferBinding.java` | `runtime.device.buffer.DeviceBufferBinding` |
| `DeviceToCpuMaterializer.java` | `runtime.memory.DeviceToCpuMaterializer` |
| `DeviceToNativeMaterializer.java` | `runtime.memory.DeviceToNativeMaterializer` |
| `ExecutionResource.java` | `runtime.memory.ExecutionResource` |
| `StorageResidency.java` | `runtime.contract.StorageResidency` |
| `TensorResidencyState.java` | `runtime.residency.TensorResidencyState` |
| `transfer/DeviceTransferMatrix.java` | `runtime.memory.transfer.DeviceTransferMatrix` |
| `transfer/DeviceTransferSupport.java` | `runtime.memory.transfer.DeviceTransferSupport` |

Po tomto tasku se `src/main/java/backend/memory` odstrani.

Enumy `CpuMaterializationReason` a `StorageResidency` byly presunuty ve fazi 2.
Tabulka je uvadi kvuli uplnosti auditu a nesmi z nich zustat druha kopie.

Soucasne prepnout storage contract consumer:

```text
src/main/java/tensor/storage/NativeMemoryAllocation.java
  backend.memory.ExecutionResource -> runtime.memory.ExecutionResource
```

`NativeMemoryAllocation` zustane v `tensor.storage`; meni se pouze jeho parent
interface. Runtime memory muze nadale zavirat tensor-owned native allocation pres
jednotny `ExecutionResource` kontrakt.

### Task 4.2: Accelerator buffer model

Status: `[x]`

Vsechny backend-neutral soubory z `backend.accelerator.buffer` se presunou do
`runtime.device.buffer`, protoze je konzumuje runtime layout/residency logika:

```text
AcceleratorBufferAccessMode, AcceleratorBufferBindings,
AcceleratorBufferDecision, AcceleratorBufferExecutionPath,
AcceleratorBufferInputDecision, AcceleratorBufferLayout,
AcceleratorBufferLayoutClass, AcceleratorBufferLayoutClassifier,
AcceleratorBufferOutputDecision, AcceleratorBufferReasonCode,
AcceleratorBufferRequest, AcceleratorLayoutAbiV2Descriptor,
AcceleratorLayoutAbiV2ReasonCodes, AcceleratorLayoutAbiV2Status,
AcceleratorLayoutAbiV2StatusCode, AcceleratorLayoutAbiV2Support,
AcceleratorLayoutTransformDecision, AcceleratorLayoutTransformKind,
AcceleratorLayoutTransformPlanner, AcceleratorLayoutTransformRequest
```

`DeviceBufferBinding` bude ve stejnem package, takze nevznikne kruh
`runtime.memory <-> runtime.device.buffer`.

Presunout vsechny aktualni testy tohoto modelu a zmenit jejich package declarations
z `backend.accelerator.buffer` na `runtime.device.buffer`:

```text
src/test/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifierTest.java
  -> src/test/java/runtime/device/buffer/AcceleratorBufferLayoutClassifierTest.java
src/test/java/backend/accelerator/buffer/AcceleratorLayoutAbiV2DescriptorTest.java
  -> src/test/java/runtime/device/buffer/AcceleratorLayoutAbiV2DescriptorTest.java
src/test/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlannerTest.java
  -> src/test/java/runtime/device/buffer/AcceleratorLayoutTransformPlannerTest.java
```

### Task 4.3: Native CPU runtime memory

Status: `[x]`

Runtime state dnes importuje allocator/pool z konkretniho CPU backendu. Presunout:

```text
backend/cpu/nativecpu/NativeCpuAllocation.java    -> runtime/memory/nativecpu/NativeCpuAllocation.java
backend/cpu/nativecpu/NativeCpuAllocator.java     -> runtime/memory/nativecpu/NativeCpuAllocator.java
backend/cpu/nativecpu/NativeCpuMaterializer.java  -> runtime/memory/nativecpu/NativeCpuMaterializer.java
backend/cpu/nativecpu/NativeCpuMemoryPool.java    -> runtime/memory/nativecpu/NativeCpuMemoryPool.java
backend/cpu/nativecpu/NativeCpuMemoryStats.java   -> runtime/memory/nativecpu/NativeCpuMemoryStats.java
backend/cpu/nativecpu/NativeCpuStorageFactory.java-> runtime/memory/nativecpu/NativeCpuStorageFactory.java
backend/cpu/nativecpu/NativeCpuTraceState.java    -> runtime/memory/nativecpu/NativeCpuTraceState.java
```

V `backend.cpu.nativecpu` zustavaji pouze backend compute/policy soubory:

```text
CpuNativeStorageSupport
CpuNativeTraceSupport
NativeBFloat16Kernels
NativeCpuRuntimePolicy
layout/**
```

Nativecpu test inventory ma definitivni rozdeleni.

Presunout test runtime allocatoru, poolu, storage factory a materializeru:

```text
src/test/java/backend/cpu/nativecpu/NativeCpuStorageTest.java
  -> src/test/java/runtime/memory/nativecpu/NativeCpuStorageTest.java
```

Zmenit package declaration na `runtime.memory.nativecpu`. Test zustane package-local
k runtime native memory implementaci.

Nasledujici testy zustavaji na miste, protoze testuji backend kernel, layout,
OpenBLAS route, planning policy nebo end-to-end CPU native execution:

```text
src/test/java/backend/cpu/nativecpu/NativeBFloat16KernelsTest.java
src/test/java/backend/cpu/nativecpu/NativeCpuElementwiseChainTest.java
src/test/java/backend/cpu/nativecpu/NativeCpuRegionSelectionTest.java
src/test/java/backend/cpu/nativecpu/NativeOpenBlasMatMulExecutableTest.java
src/test/java/backend/cpu/nativecpu/NativeOpenBlasPlannerTest.java
src/test/java/backend/cpu/nativecpu/layout/NativeSegmentStridedKernelsTest.java
src/test/java/backend/cpu/nativecpu/layout/TensorPhysicalViewTest.java
```

Jejich package declarations se nemeni. Importy presunutych
`NativeCpuStorageFactory`, `NativeCpuMaterializer`, runtime enums, prepared execution
a trace DTO se prepnou na finalni packages podle prislusnych fazi.

### Task 4.4: Transfer matrix

Status: `[x]`

`DeviceTransferMatrix` bude importovat `runtime.contract.HostDeviceTransferKind`
a `runtime.contract.StorageResidency`. Trace DTO budou importovat stejne leaf enumy.
Memory package nesmi importovat trace.

### Task 4.5: Test move map

Status: `[x]`

```text
test/backend/memory/TensorResidencyStateTest.java
  -> test/runtime/residency/TensorResidencyStateTest.java
test/backend/memory/transfer/DeviceTransferMatrixTest.java
  -> test/runtime/memory/transfer/DeviceTransferMatrixTest.java
```

### Task 4.6: Overeni

Status: `[x]`

```bash
./gradlew classes
./gradlew test --tests runtime.residency.TensorResidencyStateTest
./gradlew test --tests runtime.memory.transfer.DeviceTransferMatrixTest
./gradlew test --tests runtime.device.buffer.AcceleratorBufferLayoutClassifierTest
./gradlew test --tests runtime.device.buffer.AcceleratorLayoutAbiV2DescriptorTest
./gradlew test --tests runtime.device.buffer.AcceleratorLayoutTransformPlannerTest
./gradlew test --tests runtime.memory.nativecpu.NativeCpuStorageTest
./gradlew test --tests backend.cpu.nativecpu.NativeBFloat16KernelsTest
./gradlew test --tests backend.cpu.nativecpu.NativeCpuElementwiseChainTest
./gradlew test --tests backend.cpu.nativecpu.NativeCpuRegionSelectionTest
./gradlew test --tests backend.cpu.nativecpu.NativeOpenBlasMatMulExecutableTest
./gradlew test --tests backend.cpu.nativecpu.NativeOpenBlasPlannerTest
./gradlew test --tests backend.cpu.nativecpu.layout.NativeSegmentStridedKernelsTest
./gradlew test --tests backend.cpu.nativecpu.layout.TensorPhysicalViewTest
./gradlew test --tests TensorStorageDataTypeTest
./gradlew test --tests backend.metal.buffer.*
./gradlew test --tests backend.cuda.buffer.*
./gradlew test --tests PackageOwnershipBoundaryTest
rg -n "backend\.memory|backend\.runtime\.ExecutionMode|backend\.accelerator\.buffer" \
  src/main/java src/test/java \
  --glob '!SourceTreeHygieneTest.java' --glob '!PackageOwnershipBoundaryTest.java'
```

Commit:

```text
refactor: move shared runtime memory and device contracts
```

## Faze 5: Prepared Execution Kontrakt

Status: `[x]` implementace Faze 5 je dokoncena; vsechny phase-owned validace prosly.

### Evidence A Aktualni Stav Faze 5

Stav k 2026-06-30:

- `[x]` Task 5.1: prepared metadata, executable kontrakt, allocator, residency
  kontrakty, execution context, execution state a workspace store byly presunuty do
  `runtime.execution`; stare produkcni cesty a symboly byly odstraneny bez aliasu.
- `[x]` Task 5.2: `PreparedStepMetadata` ma povinny non-null
  `PreparedStepExecutable` a explicitni input/output residency. `ExecutionContext`
  vystavuje jediny backend-neutral `requireWorkspace(nodeId, Class<T>)`; runtime
  neimportuje konkretni backend workspace typy.
- `[x]` Task 5.3: CPU, cpu1 a accelerator artifacty vykonavaji pripravenou praci
  primo. Direct CUDA/OpenCL artifacty resolve-nou registry kernel pri prepare a hot
  path uz registry necte. Interior partition preparery fail-fast; builder vytvari
  pouze nepokryte boundary/standalone kroky.
- `[x]` Task 5.4: `ComputeEngine`, `MetalBackend`, `CudaGpuBackend`, `CudaBackend` a
  `OpenClBackend` byly odstraneny. Runner vola non-null executable bez switche podle
  konkretniho backendu; `CpuBackend` zustava backend-internal implementace volana z
  CPU artifactu.
- `[x]` Task 5.5: `./gradlew classes` a `testClasses` prosly. Presna Phase 5 sada
  (`PreparedExecutionBuildTest`, `Cpu1ExecutionContractTest`,
  `CudaAcceleratorExecutionPathTest`, `MetalLayoutAwareDeviceFlowTest`,
  `PackageOwnershipBoundaryTest`, `SourceTreeHygieneTest`) prosla v jednom behu
  spolu s direct CUDA/OpenCL a metadata testy. Sirsi CPU/CPU1/Metal/CUDA/OpenCL sada
  prosla: 495 testu, 3 standardni platformni skips. Legacy symbol audit, runtime
  concrete-backend import audit, odstranene-source audit a `git diff --check` jsou
  ciste. Audit `artifact()` nadale nachazi pouze nesouvisejici lowering artifacty a
  lokalni benchmark fixture accessory; prepared metadata pouziva vyhradne
  `executable()`.
- Zadny commit ani push nebyl proveden.

### Task 5.1: Prime move a rename

Status: `[x]`

Tightly coupled prepared metadata a execution context budou ve stejnem package.
Toto je zamerne: rozdeleni do `runtime.plan` a `runtime.context` by vytvorilo
obousmernou zavislost, protoze executable potrebuje context a context poskytuje
metadata lookup.

```text
graph/execution/plan/CompiledNodeExecutionMetadata.java
  -> runtime/execution/PreparedStepMetadata.java
graph/execution/plan/PreparedExecutionArtifact.java
  -> runtime/execution/PreparedStepExecutable.java
graph/execution/plan/PreparedRuntimeStateAllocator.java
  -> runtime/execution/PreparedRuntimeStateAllocator.java
graph/execution/plan/InputResidencyRequirement.java
  -> runtime/execution/InputResidencyRequirement.java
graph/execution/plan/OutputResidencyEffect.java
  -> runtime/execution/OutputResidencyEffect.java
backend/runtime/ExecutionContext.java
  -> runtime/execution/ExecutionContext.java
graph/execution/state/ExecutionState.java
  -> runtime/execution/ExecutionState.java
graph/execution/state/RuntimeWorkspaceStore.java
  -> runtime/execution/RuntimeWorkspaceStore.java
```

`ExecutionState` a `RuntimeWorkspaceStore` jsou soucast prepared-run modelu.
`RuntimeWorkspaceStore` implementuje `PreparedRuntimeStateAllocator` a iteruje
`PreparedStepMetadata`; jejich rozdeleni mezi `runtime.state` a
`runtime.execution` by vytvorilo obousmerny package import. Ostatni state sluzby
zustanou v `runtime.state` a neimportuji `runtime.execution`.

### Task 5.2: Executable kontrakt

Status: `[x]`

Definitivni interface:

```java
package runtime.execution;

import graph.model.CompiledNode;
import trace.backend.StepTraceContribution;

public interface PreparedStepExecutable {
    void execute(
            CompiledNode node,
            PreparedStepMetadata metadata,
            ExecutionContext context
    );

    default void allocateRuntimeState(int nodeId, PreparedRuntimeStateAllocator allocator) {
    }

    default StepTraceContribution traceContribution(
            CompiledNode node,
            PreparedStepMetadata metadata,
            ExecutionContext context
    ) {
        return StepTraceContribution.empty();
    }
}
```

Metadata ma non-null executable:

```java
package runtime.execution;

public record PreparedStepMetadata(
        ComputeBackend backend,
        Operation executionOperation,
        List<Integer> executionInputNodeIds,
        PreparedStepExecutable executable,
        InputResidencyRequirement inputResidencyRequirement,
        OutputResidencyEffect outputResidencyEffect
) {
    public PreparedStepMetadata {
        Objects.requireNonNull(backend, "backend cannot be null");
        Objects.requireNonNull(executable, "executable cannot be null");
        executionInputNodeIds = List.copyOf(
                executionInputNodeIds == null ? List.of() : executionInputNodeIds
        );
        inputResidencyRequirement = Objects.requireNonNull(
                inputResidencyRequirement, "inputResidencyRequirement cannot be null"
        );
        outputResidencyEffect = Objects.requireNonNull(
                outputResidencyEffect, "outputResidencyEffect cannot be null"
        );
    }
}
```

Default residency odvozena pouze z backend enumu se odstrani z constructoru.
Kazdy backend preparer musi kontrakt nastavit explicitne.

Z `ExecutionContext` odstranit backend-specific metody:

```text
cpuWorkspaceForNodeId(...)
cpu1ScratchBufferForNodeId(...)
```

Nahradit je jednim backend-neutral typove kontrolovanym accessorem:

```java
public <T> T requireWorkspace(int nodeId, Class<T> workspaceType) {
    Objects.requireNonNull(workspaceType, "workspaceType cannot be null");
    Object workspace = workspaceForNodeId(nodeId);
    if (workspace == null) {
        throw new IllegalStateException("Missing runtime workspace for nodeId=" + nodeId);
    }
    if (!workspaceType.isInstance(workspace)) {
        throw new IllegalStateException("Runtime workspace for nodeId=" + nodeId
                + " is not " + workspaceType.getName()
                + ": " + workspace.getClass().getName());
    }
    return workspaceType.cast(workspace);
}
```

CPU call sites pouziji `requireWorkspace(nodeId, CpuNodeWorkspace.class)`, cpu1
call sites `requireWorkspace(nodeId, Cpu1ScratchBuffer.class)`. Runtime tim
neimportuje konkretni backend workspace typ.

### Task 5.3: Presunout dispatch do artifactu

Status: `[x]`

`Cpu1PreparedArtifact` uz ma `execute(ExecutionContext)`. Upravit signaturu:

```java
@Override
public void execute(
        CompiledNode node,
        PreparedStepMetadata metadata,
        ExecutionContext context
) {
    executableUnit.run(context);
}
```

`CpuNodeExecutionArtifact` a `CpuFusedExecutionArtifact` implementuji execute
delegaci na `CpuBackend.execute(...)`. `CpuBackend` zustava, protoze ho vedle
artifactu pouziva `PreparedAcceleratorExecutionSupport` pro CPU fallback a jeho
`buildExecutionPlan(...)` pouziva `CpuNodePreparer`. Runtime ho uz neimportuje.
Kernel, plan a workspace zustanou prepare-time fields.

Z `CpuBackend.execute(...)` odstranit specialni vetev
`metadata.executable() instanceof Cpu1PreparedArtifact`. Cpu1 artifact se po teto
zmene spousti sam pres `PreparedStepExecutable`; stary CPU backend uz nesmi znat
cpu1 artifact.

`AcceleratorExecutionArtifact` implementuje:

```java
@Override
public void execute(CompiledNode node, PreparedStepMetadata metadata, ExecutionContext context) {
    if (executable == null) {
        throw new IllegalStateException("Missing prepared accelerator executable for node " + node.id());
    }
    executable.execute(context);
}
```

Pro direct legacy cesty pridat konkretni artifacty:

```text
backend/cuda/exec/CudaDirectPreparedExecutable.java
backend/opencl/exec/OpenClDirectPreparedExecutable.java
```

Ty pri prepare resolve-nou kernel z registry a pri execute pouze sestavi runtime
input views a zavolaji kernel. Neprovadeji novy registry lookup v hot path.

Interior partition node se v `PreparedExecutionBuilder` nevytvari jako step.
Preparery pro interior node prestanou vracet metadata s `null` artifactem a misto
toho vyhodi `IllegalStateException`, pokud je orchestrace zavola navzdory coverage.

### Task 5.4: Odstranit centralni runtime dispatch

Status: `[x]`

Odstranit:

```text
src/main/java/backend/ComputeEngine.java
```

`runtime.runner.PreparedExecutionRunner` po fazi 6 pouzije:

```java
PreparedStepExecutable executable = step.metadata().executable();
executable.execute(step.compiledNode(), step.metadata(), context);
```

Odstranit konkretne tyto mrtve dispatch fasady, jejichz jediny production caller
je dnes `ComputeEngine`:

```text
src/main/java/backend/metal/MetalBackend.java
src/main/java/backend/cuda/CudaGpuBackend.java
src/main/java/backend/cuda/CudaBackend.java
src/main/java/backend/opencl/OpenClBackend.java
```

Jejich logika bude pokryta takto:

- Metal/CUDA partition route vykonava `AcceleratorExecutionArtifact`.
- Direct CUDA route vykonava `CudaDirectPreparedExecutable`.
- Direct OpenCL route vykonava `OpenClDirectPreparedExecutable`.
- CPU route zustava v `CpuBackend`, ale je volana z CPU artifactu, ne z runtime.

Test `CudaAcceleratorExecutionPathTest` se prepise na pripraveny executable, ne na
`ComputeEngine.compute`. `CpuKernelFamilyArchitectureTest` se upravi tak, aby
kontroloval `CpuBackend` jako backend-internal executor a ne centralni runtime
dispatch.

### Task 5.5: Overeni

Status: `[x]`

```bash
./gradlew classes
./gradlew test --tests PreparedExecutionBuildTest
./gradlew test --tests backend.cpu1.Cpu1ExecutionContractTest
./gradlew test --tests backend.cuda.CudaAcceleratorExecutionPathTest
./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest
./gradlew test --tests PackageOwnershipBoundaryTest
rg -n "ComputeEngine|CompiledNodeExecutionMetadata|PreparedExecutionArtifact|backend\.runtime\.ExecutionContext|artifact\(\)" \
  src/main/java src/test/java \
  --glob '!SourceTreeHygieneTest.java' --glob '!PackageOwnershipBoundaryTest.java'
```

Posledni audit muze najit `artifact()` pouze v textu dokumentace. V Java zdrojich
ma byt accessor `executable()`.

Commit:

```text
refactor: execute prepared artifacts without backend dispatch
```

## Faze 6: Kompletni Runtime

Status: `[x]`

### Evidence A Aktualni Stav Faze 6

Stav k 2026-06-30:

- `[x]` Task 6.1: prepared execution facade, step, run a publication policy byly
  presunuty do `runtime.execution`; `CompiledGraph`, builder, Tensor lifecycle,
  tuning, numerics, training a vsichni produkcni/test konzumenti pouzivaji finalni
  typy bez aliasu nebo wrapperu.
- `[x]` Task 6.2: run-scoped stores, registry a materialization state jsou v
  `runtime.state`. Alias binding v `RuntimeTensorStore` a `RuntimeMemoryBinder`
  pouziva immutable `CompiledNode.storageOwnerId()` misto runtime volani
  `AliasViewPolicy`.
- `[x]` Task 6.3: residency, publication, device helpers a runner jsou ve finalnich
  `runtime.*` baliccich. `StepExecutionTracer` i backend adaptery pouzivaji nove
  execution/device kontrakty.
- `[x]` Task 6.4: tri zbyvajici `graph.execution` testy byly fyzicky presunuty do
  `runtime.device`, `runtime.execution` a `runtime.residency`; metadata test uz byl
  ve finalnim `runtime.execution` balicku. Hygiene cesty byly aktualizovany.
- `[x]` Task 6.5: `classes`, `testClasses`, cela explicitni Phase 6 sada, rozsirena
  lifecycle/publication/runner/device/residency sada, `PackageOwnershipBoundaryTest`
  a `SourceTreeHygieneTest` prosly. Legacy import/path audit, Tensor lifecycle audit,
  concrete-backend runtime dependency audit a `git diff --check` jsou ciste.
- `[x]` Produkcni i testovaci `graph.execution` strom byl fyzicky odstranen;
  `test ! -e src/main/java/graph/execution` a
  `test ! -e src/test/java/graph/execution` prosly. Zaverecny legacy import audit
  nenasel skutecne Java konzumenty `graph.execution`. Zadny commit ani push nebyl
  proveden.

### Task 6.1: Execution facade

Status: `[x]`

```text
graph/execution/PreparedExecution.java     -> runtime/execution/PreparedExecution.java
graph/execution/PreparedExecutionStep.java -> runtime/execution/PreparedExecutionStep.java
graph/execution/ExecutionRun.java          -> runtime/execution/ExecutionRun.java
graph/execution/PublicationPolicy.java     -> runtime/execution/PublicationPolicy.java
```

`PreparedExecution` pouzije `runtime.memory.nativecpu.NativeCpuMemoryPool` a
`runtime.contract.ExecutionMode`.

V tomtez tasku prepnout verejne Tensor lifecycle konzumenty presouvaneho typu:

```text
src/main/java/tensor/Tensor.java
  graph.execution.PreparedExecution -> runtime.execution.PreparedExecution

src/main/java/tensor/internal/TensorExecution.java
  graph.execution.PreparedExecution -> runtime.execution.PreparedExecution
```

Signatury `Tensor.prepare(...)`, `TensorExecution.prepare(...)` a
`TensorExecution.compute(PreparedExecution, ExecutionMode)` se matematicky ani
behavioralne nemeni; meni se pouze package finalniho prepared execution kontraktu.

### Task 6.2: State

Status: `[x]`

Presunout do `runtime.state`:

```text
RuntimeDeviceMemoryState
RuntimeMaterializationService
RuntimeNativeCpuMemoryState
RuntimeResourceRegistry
RuntimeStorageKind
RuntimeStorageSlotCache
RuntimeStorageSlotKey
RuntimeStorageSlotScope
RuntimeTensorStore
```

`ExecutionState` a `RuntimeWorkspaceStore` byly presunuty do `runtime.execution`
ve fazi 5 kvuli odstraneni package cyklu.

`RuntimeTensorStore` a `RuntimeMemoryBinder` nahradi descriptor-based volani
`AliasViewPolicy` podminkou:

```java
node.storageOwnerId() != node.id()
```

### Task 6.3: Residency, publication, device a runner

Status: `[x]`

```text
graph/execution/residency/DeviceBindingRegistry.java
  -> runtime/residency/DeviceBindingRegistry.java
graph/execution/residency/NativeCpuStorageRegistry.java
  -> runtime/residency/NativeCpuStorageRegistry.java
graph/execution/residency/RuntimeMemoryBinder.java
  -> runtime/residency/RuntimeMemoryBinder.java
graph/execution/residency/RuntimeResidencyStore.java
  -> runtime/residency/RuntimeResidencyStore.java

graph/execution/publication/ExecutionPublisher.java
  -> runtime/publication/ExecutionPublisher.java

graph/execution/device/DeviceLayoutMaterializer.java
  -> runtime/device/DeviceLayoutMaterializer.java
graph/execution/device/DeviceLayoutViewPropagator.java
  -> runtime/device/DeviceLayoutViewPropagator.java

graph/execution/runner/PreparedExecutionRunner.java
  -> runtime/runner/PreparedExecutionRunner.java
```

`StepExecutionTracer` byl presunut do `runtime.runner` ve fazi 2; zde se pouze
prepnou jeho importy z `graph.execution.*` na finalni `runtime.execution.*`.

### Task 6.4: Test move map

Status: `[x]`

```text
test/graph/execution/CompiledNodeExecutionMetadataTest.java
  -> test/runtime/execution/PreparedStepMetadataTest.java
test/graph/execution/DeviceLayoutViewPropagationTest.java
  -> test/runtime/device/DeviceLayoutViewPropagationTest.java
test/graph/execution/ExecutionStateResidencyTest.java
  -> test/runtime/execution/ExecutionStateResidencyTest.java
test/graph/execution/RuntimeMemoryBinderTest.java
  -> test/runtime/residency/RuntimeMemoryBinderTest.java
```

Ve stejnem tasku prepnout cestu `RuntimeMemoryBinder` v
`SourceTreeHygieneTest` na `src/main/java/runtime/residency/RuntimeMemoryBinder.java`.

### Task 6.5: Overeni

Status: `[x]`

```bash
./gradlew classes
./gradlew test --tests runtime.execution.PreparedStepMetadataTest
./gradlew test --tests runtime.device.DeviceLayoutViewPropagationTest
./gradlew test --tests runtime.execution.ExecutionStateResidencyTest
./gradlew test --tests runtime.residency.RuntimeMemoryBinderTest
./gradlew test --tests PreparedExecutionBuildTest
./gradlew test --tests TensorMutationGuardsTest
./gradlew test --tests TensorStorageDataTypeTest
./gradlew test --tests PackageOwnershipBoundaryTest
rg -n "graph\.execution" src/main/java src/test/java \
  --glob '!SourceTreeHygieneTest.java' --glob '!PackageOwnershipBoundaryTest.java'
rg -n "graph\.execution\.PreparedExecution|backend\.runtime\.ExecutionMode" \
  src/main/java/tensor/Tensor.java src/main/java/tensor/internal/TensorExecution.java
```

Commit:

```text
refactor: move prepared execution runtime out of graph
```

## Faze 7: Prepare Context, Validace A Orchestrace

Status: `[x]`

### Evidence A Aktualni Stav Faze 7

Stav k 2026-06-30:

- `[x]` Task 7.1: `BackendPrepareContext`, immutable inputs, package-private indexy
  a `PartitionExecutionRole` jsou ve `prepare.context`; konkretni CPU, CPU1,
  Metal, CUDA a shared accelerator prepare konzumenti pouzivaji finalni context
  importy.
- `[x]` Task 7.2: `RegionPlanValidator` i jeho test byly fyzicky presunuty do
  `prepare.validation`; validator zavisi pouze na shared contextu a
  backend-neutral lowering modelu.
- `[x]` Task 7.3: builder, dispatcher a prepare trace contributors jsou v
  `prepare.orchestration`, README je v root `prepare` a puvodni tracked
  `backend.prepare` strom neobsahuje zadne zbyvajici soubory. Existujici CPU1
  direct-route zmeny v mixed dispatcheru byly pri presunu zachovany.
- `[x]` Task 7.4: `CompiledGraph.prepare(RuntimeConfig)` vola
  `prepare.orchestration.PreparedExecutionBuilder.prepare(artifacts,
  effectiveConfig)` naprimo; nebyla pridana zadna facade, compatibility alias ani
  wrapper.
- `[x]` Task 7.5: `classes`, `testClasses`, cela explicitni Phase 7 test sada,
  rozsirene CPU/CPU1 prepare testy, `PackageOwnershipBoundaryTest` a
  `SourceTreeHygieneTest` prosly. Legacy `backend.prepare` a puvodni partition-role
  import audity jsou ciste; context/validation neimportuji konkretni backend ani
  orchestration a backend preparery neimportuji orchestration.
- `[x]` Souvisejici source-tree a projektova dokumentace ukazuje na finalni
  `prepare.context`, `prepare.validation` a `prepare.orchestration` cesty. Zadny
  commit ani push nebyl proveden.

### Task 7.1: Context package

Status: `[x]`

```text
backend/prepare/BackendPrepareContext.java -> prepare/context/BackendPrepareContext.java
backend/prepare/PrepareInputs.java          -> prepare/context/PrepareInputs.java
backend/prepare/PreparedMetadataIndex.java -> prepare/context/PreparedMetadataIndex.java
backend/prepare/BackendPlanIndex.java       -> prepare/context/BackendPlanIndex.java
backend/prepare/PartitionRoleIndex.java     -> prepare/context/PartitionRoleIndex.java
backend/prepare/LoweredRegionIndex.java     -> prepare/context/LoweredRegionIndex.java
backend/accelerator/exec/PartitionExecutionRole.java
  -> prepare/context/PartitionExecutionRole.java
```

Package-private indexy zustanou package-private. `BackendPrepareContext` je jedina
verejna fasada nad nimi. Backend preparery importuji jen context.

`PartitionExecutionRole` je prepare-time role (`NONE`, `ANCHOR`, `INTERIOR`), ne
accelerator executable. Presun zabrani tomu, aby `prepare.context` importoval
`backend.accelerator.exec`. `CpuNodePreparer`, `MetalNodePreparer`,
`CudaGpuNodePreparer`, dispatcher a `SourceTreeHygieneTest` se prepnou na
`prepare.context.PartitionExecutionRole`.

### Task 7.2: Validation package

Status: `[x]`

```text
backend/prepare/RegionPlanValidator.java
  -> prepare/validation/RegionPlanValidator.java
```

Validator importuje `prepare.context.BackendPrepareContext`, ale zadny konkretni
backend. Metal a CUDA preparery importuji validator naprimo.

Test:

```text
src/test/java/backend/prepare/RegionPlanValidatorTest.java
  -> src/test/java/prepare/validation/RegionPlanValidatorTest.java
```

Zmenit package declaration na `prepare.validation` a import contextu na novy cil.

### Task 7.3: Orchestration package

Status: `[x]`

```text
backend/prepare/PreparedExecutionBuilder.java
  -> prepare/orchestration/PreparedExecutionBuilder.java
backend/prepare/BackendPrepareDispatcher.java
  -> prepare/orchestration/BackendPrepareDispatcher.java
backend/prepare/BackendPrepareTraceContributors.java
  -> prepare/orchestration/BackendPrepareTraceContributors.java
backend/prepare/README.md
  -> prepare/README.md
```

`PreparedExecutionBuilder` je jedina production entry point orchestrace:

```java
public static PreparedExecution prepare(
        CompileArtifacts artifacts,
        RuntimeConfig runtimeConfig
) {
    // puvodni orchestrace, nove importy context/validation/runtime/planning/trace
}
```

`BackendPrepareDispatcher` muze importovat konkretni preparery. Zadny konkretni
preparer nesmi importovat dispatcher ani builder.

Ve stejnem tasku prepnout `backendPrepareDoesNotRebuildOptimizerArtifacts()` v
`SourceTreeHygieneTest` na `src/main/java/prepare/orchestration` a na finalni
planning symboly. Test musi projit v tomto commitu, ne az ve fazi 9.

### Task 7.4: CompiledGraph lifecycle facade

Status: `[x]`

Aktualizovat importy a Javadoc v `graph.CompiledGraph`:

```java
import prepare.orchestration.PreparedExecutionBuilder;
import runtime.contract.ExecutionMode;
import runtime.execution.PreparedExecution;
import trace.compile.CompileTrace;
```

Metoda zustane prima:

```java
public PreparedExecution prepare(RuntimeConfig runtimeConfig) {
    RuntimeConfig effectiveConfig = runtimeConfig == null
            ? (supportsBackward()
                ? RuntimeConfig.trainingDefaults(rootTensor.getDataType())
                : RuntimeConfig.inferenceDefaults(rootTensor.getDataType()))
            : runtimeConfig;
    return PreparedExecutionBuilder.prepare(artifacts, effectiveConfig);
}
```

Nevytvari se jina facade a verejny call chain se nemeni.

### Task 7.5: Overeni

Status: `[x]`

Pouzit pouze existujici route testy:

```bash
./gradlew classes
./gradlew test --tests PreparedExecutionBuildTest
./gradlew test --tests prepare.validation.RegionPlanValidatorTest
./gradlew test --tests backend.cpu1.BackendPrepareDispatcherCpu1DirectRouteTest
./gradlew test --tests backend.cpu1.BackendPrepareDispatcherCpu1FusedRouteTest
./gradlew test --tests backend.cpu1.BackendPrepareDispatcherCpu1SpecializedRouteTest
./gradlew test --tests backend.cuda.CudaAcceleratorExecutionPathTest
./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest
./gradlew test --tests PackageOwnershipBoundaryTest
rg -n "backend\.prepare" src/main/java src/test/java \
  --glob '!SourceTreeHygieneTest.java' --glob '!PackageOwnershipBoundaryTest.java'
```

Commit:

```text
refactor: separate prepare context from orchestration
```

## Faze 8: Backend Consumer Migrace

Status: `[x]`

Tato faze nejdrive dokonci BLAS ownership migraci a pak prepise shared imports ve
vsech backend rodinach. Je to uplny audit; po fazi nezustane puvodni BLAS package,
konfiguracni runtime accessor ani prehlednuty puvodni symbol.

### Task 8.1: Definitivni BLAS ownership migrace

Status: `[x]`

#### Presny move/delete map sedmi produkcnich souboru

Aktualni pocet je presne sedm Java souboru: sest se presune a jeden se smaze.

| Zdroj | Cil / akce | Definitivni zmena |
|---|---|---|
| `src/main/java/backend/blas/BlasProvider.java` | `src/main/java/config/runtime/BlasProvider.java` | Zmenit package na `config.runtime`; ponechat jen hodnoty `NONE`, `OPENBLAS_FFM`; odstranit `fromProperty` a import `Locale`. |
| `src/main/java/backend/blas/BlasRuntime.java` | smazat | Neexistuje nahradni facade; konstanty, property parsery a accessors se odstrani. |
| `src/main/java/backend/blas/OpenBlasRuntime.java` | `src/main/java/backend/provider/blas/openblas/OpenBlasRuntime.java` | Pouze capability query a thread control. |
| `src/main/java/backend/blas/OpenBlasSymbols.java` | `src/main/java/backend/provider/blas/openblas/OpenBlasSymbols.java` | FFM symbol table; odstranit JavaCPP `Loader`, `openblas` import a `BUNDLED_JAVACPP` lookup. Lookup zustane JDK/FFM pres explicitni knihovnu, environment nebo system library. |
| `src/main/java/backend/blas/OpenBlasGemmLayout.java` | `src/main/java/backend/provider/blas/openblas/OpenBlasGemmLayout.java` | Package-only move beze zmeny chovani range kontrol. |
| `src/main/java/backend/blas/OpenBlasArrayGemm.java` | `src/main/java/backend/provider/blas/openblas/OpenBlasArrayGemm.java` | Package-only move; prime array GEMM API. |
| `src/main/java/backend/blas/OpenBlasSegmentGemm.java` | `src/main/java/backend/provider/blas/openblas/OpenBlasSegmentGemm.java` | Package-only move; prime FFM segment GEMM API. |

Cilovy stav je sest produkcnich souboru rozdelenych `1 + 5`: jeden enum v
`config.runtime`, pet OpenBLAS trid v `backend.provider.blas.openblas`, nula souboru
v puvodnim adresari. `backend.blas` po migraci neexistuje.

`build.gradle` soucasne odstrani jediny
`implementation 'org.bytedeco:openblas-platform:0.3.31-1.5.13'` dependency radek.
Po odstraneni JavaCPP lookupu jej zadny zdroj nepouziva a provider tak ma skutecne
jen JDK/FFM compile-time zavislosti.

`OpenBlasSymbols.resolveLookup` bude mit presne tuto FFM-only strukturu; zadna
dalsi lookup vetev ani classpath/bundled loader se nepridava:

```java
private static LookupResolution resolveLookup(Arena arena) {
    String explicit = System.getProperty("openblas.lib");
    if (explicit != null && !explicit.isBlank()) {
        return new LookupResolution(
                SymbolLookup.libraryLookup(explicit.trim(), arena),
                LookupSource.EXPLICIT_PROPERTY
        );
    }

    String environment = System.getenv("OPENBLAS_LIB");
    if (environment != null && !environment.isBlank()) {
        return new LookupResolution(
                SymbolLookup.libraryLookup(environment.trim(), arena),
                LookupSource.ENVIRONMENT
        );
    }

    return new LookupResolution(
            SymbolLookup.libraryLookup("openblas", arena),
            LookupSource.SYSTEM_LIBRARY
    );
}

enum LookupSource {
    EXPLICIT_PROPERTY,
    ENVIRONMENT,
    SYSTEM_LIBRARY
}
```

Poradi je definitivne `openblas.lib` -> `OPENBLAS_LIB` -> system library name
`openblas`. Nastavena, ale neplatna explicitni property nebo environment cesta
selze v dane prioritni vetvi; neni potichu nahrazena nizsi prioritou.
`openblas.lib` a `OPENBLAS_LIB` jsou jedine hodnoty, ktere provider smi cist, a
znamenaji pouze umisteni nativni knihovny. Provider z nich nikdy neodvozuje route,
threshold, debug ani thread policy. `LookupSource.BUNDLED_JAVACPP` se odstrani.

#### Config, profile IO a tuning consumers

Vsechny nasledujici produkcni soubory prepnou import na
`config.runtime.BlasProvider`:

```text
config/runtime/BlasConfig.java
config/runtime/Conv2dConfig.java
config/profile/Conv2dPlatformProfile.java
config/profile/MatmulPlatformProfile.java
config/profile/PlatformRuntimeProfile.java
config/profile/ExecutionProfileIO.java
config/profile/PlatformRuntimeProfileIO.java
tuning/benchmark/Bf16PerformanceBenchmark.java
tuning/calibration/PlatformCalibrationDefaults.java
tuning/calibration/runtime/PlatformRuntimeProfileMutators.java
tuning/candidate/explicit/ExplicitProfileMutators.java
tuning/etalon/FrameworkEtalon.java
```

`BlasProvider` je data-only; parsing vlastni profile IO. Dva soucasne
`BlasProvider.fromProperty(...)` call sites v `ExecutionProfileIO` se prepisou na
existujici generic parser, bez compatibility metody v enumu:

```java
BlasProvider provider = findEnum(
        json,
        "provider",
        defaultProfile.runtime().blas().provider(),
        BlasProvider.class
);
BlasProvider conv2dProvider = findEnum(
        json,
        "conv2dProvider",
        defaultProfile.runtime().conv2d().provider(),
        BlasProvider.class
);
```

`PlatformRuntimeProfileIO` uz pouziva svuj generic
`findEnum(json, key, fallback, BlasProvider.class)` pro `blasProvider` a
`conv2dBlasProvider`; meni se jen import. Neplatna nebo chybejici hodnota tedy
zustane profile-IO fallbackem, ne chovanim enumu. Tuning a kalibrace tvori
`BlasConfig`/profile kandidaty a nikdy nevoli provider pres system property.

#### Uplne odstraneni `BlasRuntime`

Audit vsech accessorů v `BlasRuntime` je uzavren takto:

| Accessor / state | Soucasne produkcni usage | Presna odstranovaci cesta |
|---|---:|---|
| `provider()` a `isOpenBlasFfmEnabled()` | 0 call sites | Smazat; prepare cte `BlasConfig.provider()`. |
| `matMulMinWork()` | 0 call sites | Smazat; CPU a cpu1 prepare cte `BlasConfig.matmulMinWork()`. |
| `f32RequireMgeK()` a `f32MaxNOverK()` | 0 call sites | Smazat; `MatMulPlanner` cte normal/wide shape gates z `BlasConfig`. |
| `debug()` | 9 call sites, vsechny v `MatMulBlasBackend` | Pridat explicitni `boolean debug` do CPU adapter volani; hodnotu snapshotuje prepare z `BlasConfig.debug()`. |
| `PROP_PROVIDER`, `PROP_MATMUL_MIN_WORK`, `PROP_DEBUG`, `PROP_F32_REQUIRE_M_GE_K`, `PROP_F32_MAX_N_OVER_K` | pouze uvnitr `BlasRuntime` | Smazat; zadny produkcni system-property zdroj nezustane. |
| vsechny `DEFAULT_*` a private `parseLongProperty`, `parseBooleanProperty`, `parseDoubleProperty` | pouze uvnitr `BlasRuntime` | Smazat; defaults a validace vlastni `BlasConfig` a profile IO. |

`BlasRuntime.java` se smaze ve stejnem commitu jako posledni call-site rewrite.
Nikde se nezavadi nahradni wrapper. Provider, adapter ani hot path nesmi znovu
precist provider, threshold, shape gate, debug nebo thread setting.

Presunuty `OpenBlasRuntime` se soucasne zredukuje na povolenou low-level API:

| `OpenBlasRuntime` API | Cil |
|---|---|
| `isAvailable`, `unavailableReason`, dtype/symbol availability a `lookupSource` | Ponechat jako capability query; produkcne je vola jen CPU/cpu1 prepare. |
| `getNumThreads`, `setNumThreads` | Ponechat jako thread control; execute dostava pozadovany pocet z prepared state. |
| `getParallelMode` | Ponechat jako capability query pro provider test/benchmark diagnostics. |
| `threadPolicy()` a `threadPolicy(int)` | Smazat; CPU/cpu1 prepare vytvori `AUTO_UNCONTROLLED` nebo `SET_NUM_THREADS(n)` z explicitniho prepared thread countu. `Cpu1MatmulBenchmarkTest`, `Cpu1MlpBenchmarkTest` a `NativeOpenBlasSegmentGemmBenchmarkTest` pouziji stejny lokalni test helper. |
| `parallelModeDescription()` | Smazat; `Cpu1MlpBenchmarkTest` formatuje hodnotu vracenou `getParallelMode()` ve svem test helperu. |

Tim jsou vsechny soucasne `OpenBlasRuntime` accessors bud capability/thread API,
nebo maji explicitni odstranovaci cestu. Production trace, adapter a kernel je
nevolaji pro rozhodovani.

#### CPU prepare, lowering, provider a trace

Konkretni produkcni rewrite mapa:

- `backend/cpu/lowering/CpuRegionLowerer.java` importuje
  `config.runtime.BlasProvider`; lowering pouze prenasi policy do prepare inputu.
- `backend/cpu/prepare/linalg/matmul/MatMulPlanner.java` importuje data enum a
  `backend.provider.blas.openblas.OpenBlasRuntime`. Pouze zde se pri prepare spoji
  `BlasConfig` route/threshold/shape policy s jednorazovym capability dotazem.
  `ResolvedMatMulHints` dostane explicitni `blasDebug`, route-specific
  `openBlasThreads`, ctyri capability booleany, lookup source a hotovy
  thread-policy string.
- `backend/cpu/provider/linalg/matmul/blas/MatMulBlasBackend.java` zustane CPU
  adapterem. Importuje `OpenBlasArrayGemm` a `OpenBlasRuntime`, prijima prepared
  `boolean debug` a `int openBlasThreads`, aplikuje thread override kolem primeho
  GEMM a predchozi hodnotu obnovi v `finally`. Odstrani importy i volani
  `BlasRuntime` a capability dotazy z adapteru; failure logging/fallback pouzije
  pouze predany debug snapshot.
- `F32BlasMatMulExecutable`, `F32BatchedBlasMatMulExecutable`,
  `F64BlasMatMulExecutable`, `F64BatchedBlasMatMulExecutable`,
  `BF16BlasMatMulExecutable` a `BF16BatchedBlasMatMulExecutable` jsou vsechny
  prime call sites `MatMulBlasBackend`; predaji mu `blasDebug` a
  `openBlasThreads` z prepared `ResolvedMatMulHints`/kernel contextu. Zadny z nich
  necte `RuntimeConfig`.
- `F32NativeBlasMatMulExecutable`, `F64NativeBlasMatMulExecutable` a
  `BF16NativeBlasMatMulExecutable` importuji `OpenBlasSegmentGemm`. Capability
  guardy se odstrani z hot path, protoze nedostupny symbol nemuze byt vybran
  `MatMulPlanner`; execute vola pripraveny route primo a aplikuje prepared native
  segment thread count pres `OpenBlasRuntime` se stejnym `finally` restore.
- `backend/cpu/CpuStepTraceContributor.java` neimportuje OpenBLAS provider ani
  config. Vsechny capability, lookup-source, debug/thread policy a fallback
  hodnoty cte z `ResolvedMatMulHints`/prepared step trace snapshotu. Trace tedy
  neprovadi capability query behem execute.

#### Cpu1 prepare, exec, kernels a trace

- `backend/cpu1/prepare/Cpu1MatmulPreparer.java` importuje
  `config.runtime.BlasProvider` a `OpenBlasRuntime`, vyhodnoti provider,
  threshold, storage route a dtype capability jednou a ulozi route, efektivni
  thread count, ctyri capability booleany, lookup source a thread-policy string
  do `Cpu1PreparedMatmulUnit`.
- `backend/cpu1/exec/Cpu1MatmulExecutableUnit.java` smi importovat
  `OpenBlasRuntime` pouze pro `getNumThreads`/`setNumThreads`. Thread count bere
  vyhradne z `Cpu1PreparedMatmulUnit.openBlasThreads()`; provider/config znovu
  necte a puvodni thread count obnovi v `finally`.
- `Cpu1OpenBlasArrayMatmulLoops` importuje jen `OpenBlasArrayGemm` a
  `Cpu1OpenBlasNativeSegmentMatmulLoops` jen `OpenBlasSegmentGemm`. Jejich
  capability checks a unavailable-reason dotazy se odstrani; pripraveny kernel
  dela jen prime GEMM.
- `backend/cpu1/trace/Cpu1TraceContributor.java` nema provider import. Vsechny
  availability, lookup source a thread-policy atributy cte z immutable
  `Cpu1PreparedMatmulUnit`, takze trace ani hot path neprovadi novy dotaz.

Provider package po tomto rewritu obsahuje pouze JDK/FFM imports. CPU a cpu1
prepare jsou jedina produkcni mista, ktera kombinuji capability s dispatch policy;
thread-control calls v CPU adapteru/native executable a cpu1 exec pouze aplikuji
uz pripravenou hodnotu.

#### Test move a rewrite mapa

Jedinym presouvaným testem je provider-owned GEMM test:

```text
src/test/java/OpenBlasGemmTest.java
  -> src/test/java/backend/provider/blas/openblas/OpenBlasGemmTest.java
package backend.provider.blas.openblas;
```

Jeho source-path assertions se prepnou na
`src/main/java/backend/provider/blas/openblas/OpenBlasSegmentGemm.java`; assertion
odstraneneho bridge kontroluje novy provider adresar. Tri metody, jejichz nazvy
dnes tvrdi bundled lookup, se prejmenuji podle finalni semantiky:

```text
bundledOrConfiguredOpenBlasProvidesRequiredGemmSymbols
  -> explicitEnvironmentOrSystemOpenBlasProvidesRequiredGemmSymbols
bundledOrConfiguredOpenBlasProvidesBFloat16ToFloatGemmWhenAdvertised
  -> explicitEnvironmentOrSystemOpenBlasProvidesBFloat16ToFloatGemmWhenAdvertised
bundledOrConfiguredOpenBlasProvidesBFloat16OutputGemmWhenAdvertised
  -> explicitEnvironmentOrSystemOpenBlasProvidesBFloat16OutputGemmWhenAdvertised
```

Nasledujicich presne 18 testu zustane na stejne ceste a dostane pouze import nebo
fully-qualified-name rewrite na `config.runtime.BlasProvider` a/nebo
`backend.provider.blas.openblas.*`:

```text
AutotuneDefaultStrategySelectorTest
BFloat16BlasDispatchTest
BenchmarkSessionTest
ComputeModeTraceTest
ExecutionProfileIoTest
LinearExecutionTest
MatMulTest
PlatformRuntimeProfileMutatorsTest
PlatformRuntimeProfileResolverTest
PreparedExecutionBuildTest
ProfileGridCandidateSpaceTest
TuningStoreTest
backend.cpu.lowering.CpuRegionLowererTest
backend.cpu.nativecpu.NativeCpuElementwiseChainTest
backend.cpu.nativecpu.NativeOpenBlasMatMulExecutableTest
backend.cpu.nativecpu.NativeOpenBlasPlannerTest
backend.cpu1.Cpu1LinearExecutionContractTest
backend.cpu1.Cpu1MatmulExecutionContractTest
```

Tri benchmark testy zustavaji na svych cestach, ale nejsou import-only:

```text
backend.cpu1.Cpu1MatmulBenchmarkTest
backend.cpu1.Cpu1MlpBenchmarkTest
debug.NativeOpenBlasSegmentGemmBenchmarkTest
```

Vsechny tri dostanou lokalni `threadPolicy(int requestedThreads)` formatter pro
`AUTO_UNCONTROLLED`/`SET_NUM_THREADS(n)`, protoze provider helper se smaze.
`Cpu1MlpBenchmarkTest` navic dostane lokalni `parallelModeDescription(OptionalInt)`
formatter. Soucasne prepnou import `OpenBlasRuntime` na novy provider package.
Nadale mohou primo pouzit provider capability/thread API pro test setup a obnoveni
thread stavu; produkcni route rozhodovani tim nevznikne.

`SourceTreeHygieneTest` zustava na sve ceste, ale neni import-only: rozsiri se o
zakaz stareho adresare a o JDK/FFM-only import kontrolu noveho provider package
popsanou v Task 9.1 a prepise sve stare BLAS source/package literaly.

Uplna aritmetika 23 existujicich `backend.blas` test consumeru je definitivne:

```text
1 moved test with package/source-path/method-name rewrites
18 unchanged-path import/FQN-only rewrites
3 unchanged-path benchmark helper rewrites
1 unchanged-path hygiene extension
= 23
```

#### Boundary, targeted verification a commit

`PackageOwnershipBoundaryTest.externalOpenBlasProviderIsLowLevelAndBackendNeutral`
zakaze provideru importy `config`, `graph`, `planning`, `prepare`, `runtime`,
`trace`, `tensor` a vsech konkretnich backendu. `removedPackageTreesStayRemoved`
a `SourceTreeHygieneTest.legacyArchitecturePackagesAreRemoved` oba obsahuji
`src/main/java/backend/blas`. Oba testy kontroluji, ze cesta neobsahuje zadny
Java source. Nekontroluji samotnou existenci prazdneho lokalniho adresare,
protoze Git prazdne adresare neeviduje a takovy stav neni soucasti repository.

```bash
./gradlew classes
./gradlew test --tests backend.provider.blas.openblas.OpenBlasGemmTest
./gradlew test --tests BFloat16BlasDispatchTest
./gradlew test --tests backend.cpu.nativecpu.NativeOpenBlasMatMulExecutableTest
./gradlew test --tests backend.cpu.nativecpu.NativeOpenBlasPlannerTest
./gradlew test --tests backend.cpu1.Cpu1MatmulExecutionContractTest
./gradlew test --tests PackageOwnershipBoundaryTest
./gradlew test --tests SourceTreeHygieneTest

test -z "$(find src/main/java/backend/blas -type f -name '*.java' -print 2>/dev/null)"
rg -n "backend\\.blas|\\bBlasRuntime\\b" src/main/java src/test/java \
  --glob '!SourceTreeHygieneTest.java' --glob '!PackageOwnershipBoundaryTest.java'
rg -n '^import (config|graph|planning|prepare|runtime|trace|tensor|backend\\.(cpu|cpu1|metal|cuda|opencl))\\.' \
  src/main/java/backend/provider/blas/openblas
rg -n --pcre2 '^import (?!java\\.|static java\\.)' \
  src/main/java/backend/provider/blas/openblas
rg -n 'org\\.bytedeco|openblas-platform' build.gradle src/main/java
```

Vsechny ctyri `rg` vystupy musi byt prazdne. Commit:

```text
refactor: assign blas configuration and provider ownership
```

### Task 8.2: Cpu1 import a ownership mapa

Status: `[x]`

Evidence (2026-07-01): audit covered all 273 production and 41 test Java files
under `backend.cpu1`, including all 17 named preparers/components, all 66 production
and 24 test `ExecutionContext` consumers, and all 37 production and 3 test
`CpuMaterializationReason` consumers. The explicit legacy-prefix audit for
`backend.runtime`, `backend.memory`, `graph.execution.plan`,
`graph.execution.trace`, `graph.compile.descriptor`, `graph.compile.intent`,
`graph.compile.planning.region.specialization`, `graph.compile.planning.value`, and
`backend.prepare` returned no matches. All consumers already use the final shared
contracts, so no cpu1 kernel body, dispatch policy, launch policy, threshold, or hot
path required an additional Task 8.2 edit.

`backend.cpu1.prepare`, `backend.cpu1.exec`, `backend.cpu1.kernels`, storage,
launch, provider a trace zustavaji backend-owned. Meni se pouze konzumovane
shared contracts:

```text
backend.runtime.ExecutionContext
  -> runtime.execution.ExecutionContext
backend.runtime.ExecutionMode
  -> runtime.contract.ExecutionMode
backend.memory.CpuMaterializationReason
  -> runtime.contract.CpuMaterializationReason
backend.memory.CpuMaterializationResult
  -> runtime.memory.CpuMaterializationResult
backend.memory.StorageResidency
  -> runtime.contract.StorageResidency
backend.memory.DeviceBufferBinding
  -> runtime.device.buffer.DeviceBufferBinding
graph.execution.plan.CompiledNodeExecutionMetadata
  -> runtime.execution.PreparedStepMetadata
graph.execution.plan.PreparedExecutionArtifact
  -> runtime.execution.PreparedStepExecutable
graph.execution.plan.PreparedRuntimeStateAllocator
  -> runtime.execution.PreparedRuntimeStateAllocator
graph.execution.trace.StepTraceContribution
  -> trace.backend.StepTraceContribution
graph.execution.trace.*TraceMetadata
  -> trace.backend.*TraceMetadata
graph.compile.descriptor.*
  -> planning.descriptor.*
graph.compile.intent.*
  -> planning.intent.*
graph.compile.planning.region.specialization.*
  -> planning.region.specialization.*
graph.compile.planning.value.*
  -> planning.value.*
backend.prepare.BackendPrepareContext
  -> prepare.context.BackendPrepareContext
```

Konkretni preparery, ktere audit musi pokryt:

```text
Cpu1NodePreparer, Cpu1LayoutPreparer, Cpu1LossPreparer,
Cpu1IndexPreparer, Cpu1Pool2dPreparer, Cpu1Conv2dPreparer,
Cpu1NormalizationPreparer, Cpu1MatmulPreparer, Cpu1DTypePreparer,
Cpu1ReductionPreparer, Cpu1AttentionPreparer,
Cpu1AttentionBackwardPreparer, Cpu1MseLossPreparer,
Cpu1FusedElementwisePreparer, Cpu1PreparedArtifact,
Cpu1StorageAccessPlan, Cpu1TraceContributor
```

Vsechny cpu1 executable units a kernely s `ExecutionContext` nebo
`CpuMaterializationReason` se prepnou mechanicky. Kernel body, dispatch policy,
launch policy, thresholdy a hot path se touto mechanickou migraci nemeni, s jedinou
explicitni vyjimkou: BLAS capability guardy, prepared thread hodnoty, trace snapshoty
a lokalni benchmark formatters se meni presne podle Task 8.1. Task 8.2 nepridava
zadnou dalsi cpu1 kernel ani hot-path zmenu nad ramec Task 8.1.

### Task 8.3: Cpu1 targeted validation

Status: `[x]`

Evidence (2026-07-01): all nine commands below completed with `BUILD SUCCESSFUL`.

```bash
./gradlew test --tests backend.cpu1.Cpu1ExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1LayoutExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1ReductionExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1MatmulExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1AttentionExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1AttentionBackwardExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1FusedElementwisePreparerTest
./gradlew test --tests backend.cpu1.Cpu1FusedGeneratedExecutionTest
./gradlew test --tests backend.cpu1.Cpu1StorageAccessPlanTest
```

### Task 8.4: CPU, Metal, CUDA, OpenCL a accelerator audit

Status: `[x]`

Evidence (2026-07-01): exhaustive audit covered all 477 production and 78 test
Java files under the seven named backend trees: CPU 304/40, Metal 52/16, CUDA
34/13, OpenCL 4/1, accelerator 50/6, partition 2/0, and lowering 31/2. The
corrected legacy shared-contract audit, including the exact standalone
`\bBlasRuntime\b` symbol alternative, returned no matches. All consumers use the
final contracts, so no backend source or test migration edit was required. Direct
source inspection confirmed the Metal/CUDA buffer bindings, all present Metal/CUDA
device materializers, all four named prepared executable artifacts, backend trace
contributions, and backend preparer context/validation imports. No production
backend preparer imports `prepare.orchestration`. `./gradlew classes` and the
focused cross-family test command covering CPU, Metal, CUDA, OpenCL, accelerator,
shared lowering, `PackageOwnershipBoundaryTest`, and `SourceTreeHygieneTest` both
completed with `BUILD SUCCESSFUL`.

Prepsat stejne shared imports v:

```text
backend.cpu/**
backend.metal/**
backend.cuda/**
backend.opencl/**
backend.accelerator/**
backend.partition/**
backend.lowering/**
```

Specialni kontroly:

- Metal/CUDA buffer binding implementuje `runtime.device.buffer.DeviceBufferBinding`.
- Metal/CUDA materializery implementuji `runtime.memory.DeviceTo*Materializer`.
- `AcceleratorExecutionArtifact` implementuje `PreparedStepExecutable`.
- `CpuNodeExecutionArtifact`, `CpuFusedExecutionArtifact` a
  `Cpu1PreparedArtifact` implementuji `PreparedStepExecutable`.
- Backend trace contributors vraci `trace.backend.StepTraceContribution`.
- Backend preparery importuji `prepare.context`/`prepare.validation`, ne orchestration.

### Task 8.5: Global symbol audit

Status: `[x]`

Evidence (2026-07-01): the full command below ran over `src/main/java` and
`src/test/java` with only the two documented hygiene-test exclusions and returned
no matches (`rg` exit code 1). Replacing the broad `BlasRuntime` alternative with
the exact symbol regex `\bBlasRuntime\b` is not an audit weakening: it still catches
the removed standalone legacy class/symbol while correctly excluding the valid
Task 8.1 low-level provider contract `OpenBlasRuntime` and helper identifiers such
as `bfloat16BlasRuntime`.

```bash
rg -n "graph\.execution|graph\.compile\.(descriptor|intent|planning)|backend\.prepare|backend\.runtime|backend\.memory|backend\.blas|\bBlasRuntime\b|backend\.accelerator\.buffer|backend\.ComputeBackend|CompiledNodeExecutionMetadata|PreparedExecutionArtifact|OptimizedRegion|DefaultRegionOptimizer|RegionOptimizationContext|RegionOptimizationTrace|CpuRegionOptimizationPolicy" \
  src/main/java src/test/java \
  --glob '!SourceTreeHygieneTest.java' --glob '!PackageOwnershipBoundaryTest.java'
```

Vystup musi byt prazdny.

Commit:

```text
refactor: migrate backend consumers to final contracts
```

## Faze 9: Testy, Hygiene A Dokumentace

Status: `[x]`

### Evidence A Aktualni Stav Faze 9

Stav k 2026-07-01:

- `[x]` Existujici `SourceTreeHygieneTest` kontroluje vsechny finalni legacy cesty
  podle pritomnosti Java zdroju a OpenBLAS provider omezuje na `java.*`/`static java.*`.
- `[x]` `PackageOwnershipBoundaryTest` pokryva finalni lifecycle, planning, trace,
  tensor, runtime, prepare a provider hranice nad skutecnymi cilovymi cestami.
- `[x]` Globalni test audit nenasel stare package importy ani stare prejmenovane
  produkcni symboly; tri prejmenovane test suites prosly.
- `[x]` Povinne README a BLAS dokumenty popisuji finalni ownership; petisouborovy
  audit `backend/blas|org.bytedeco|BUNDLED_JAVACPP|bundled OpenBLAS` ma 0 nalezu
  a vsech pet `test -f` kontrol proslo.
- `[x]` `./gradlew classes`, `SourceTreeHygieneTest`, `PackageOwnershipBoundaryTest`,
  `CompiledGraphTraceTest` a prejmenovane targeted testy prosly; `git diff --check`
  je cisty. `PreparedExecutionBuildTest` prosel: 131 testu celkem, 104 uspesnych,
  27 preskocenych a 0 selhani/chyb. Pri nedostupnem OpenBLAS byl capability-dependent
  `float64MatmulPrepareBuildsBlasExecutableWhenEligible` korektne preskocen.

### Task 9.1: Finalni audit `SourceTreeHygieneTest`

Status: `[x]`

Existujici test byl prubezne prepisovan ve fazich 3, 6 a 7. Zde provest finalni
kontrolu vsech cest; nepisat druhou hygiene implementaci.

Zmeny cest:

```text
src/main/java/backend/prepare
  -> src/main/java/prepare/orchestration
src/main/java/graph/execution/residency/RuntimeMemoryBinder.java
  -> src/main/java/runtime/residency/RuntimeMemoryBinder.java
src/main/java/graph/compile/planning/partition
  -> src/main/java/planning/partition
src/main/java/backend/blas
  -> odstraneno; sest zdrojovych souboru ma cil v Task 8.1 a `BlasRuntime` je smazan
graph.compile.planning.memory.MemoryPlanner
  -> planning.memory.MemoryPlanner
graph.compile.planning.region.DefaultRegionOptimizer
  -> planning.region.DefaultRegionPlanner
graph.compile.planning.region.RegionOptimizationContext
  -> planning.region.RegionPlanningContext
```

Pridat dve definitivni metody:

```java
@Test
void legacyArchitecturePackagesAreRemoved() throws IOException {
    List<Path> legacy = List.of(
            Path.of("src/main/java/graph/execution"),
            Path.of("src/main/java/graph/compile/descriptor"),
            Path.of("src/main/java/graph/compile/intent"),
            Path.of("src/main/java/graph/compile/planning"),
            Path.of("src/main/java/backend/prepare"),
            Path.of("src/main/java/backend/runtime"),
            Path.of("src/main/java/backend/memory"),
            Path.of("src/main/java/backend/blas"),
            Path.of("src/main/java/backend/ComputeEngine.java"),
            Path.of("src/test/java/backend/accelerator/buffer"),
            Path.of("src/test/java/backend/cpu/nativecpu/NativeCpuStorageTest.java")
    );
    List<String> offenders = legacy.stream()
            .flatMap(path -> {
                try {
                    return javaFilesUnder(path).stream();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            })
            .sorted()
            .toList();
    assertTrue(offenders.isEmpty(), () -> "Legacy architecture Java sources remain: " + offenders);
}

@Test
void openBlasProviderSourcesDoNotImportHigherLayersOrConcreteBackends() throws IOException {
    Path root = Path.of("src/main/java/backend/provider/blas/openblas");
    List<String> projectForbidden = List.of(
            "import config.", "import graph.", "import planning.", "import prepare.",
            "import runtime.", "import trace.", "import tensor.",
            "import backend.cpu.", "import backend.cpu1.", "import backend.metal.",
            "import backend.cuda.", "import backend.opencl."
    );
    try (Stream<Path> files = Files.walk(root)) {
        List<String> offenders = files.filter(path -> path.toString().endsWith(".java"))
                .flatMap(path -> {
                    try {
                        return Files.readAllLines(path).stream()
                                .map(String::trim)
                                .filter(line -> line.startsWith("import "))
                                .filter(line -> projectForbidden.stream().anyMatch(line::startsWith)
                                        || (!line.startsWith("import java.")
                                        && !line.startsWith("import static java.")))
                                .map(line -> path + ": " + line);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })
                .toList();
        assertTrue(offenders.isEmpty(), () -> "OpenBLAS provider ownership violations: " + offenders);
    }
}
```

Oba ownership testy tedy povoluji pouze `java.*` a `static java.*`; tim odmitnou
`org.bytedeco` i jakykoli jiny externi nebo projektovy import, nejen vyjmenovane
vyssi vrstvy a konkretni backendy. `javax.*` se ted nepovoluje, protoze jej zadny
z peti provider zdroju nepotrebuje; finalni allowlist proto nema zadny `javax`
prefix ani obecny external-import otvor.

### Task 9.2: Test package declarations a imports

Status: `[x]`

Krome explicitnich test moves z fazi 3, 4, 6 a 7 provest globalni import rewrite
ve vsech tests. Test class names se meni jen tam, kde se zmenil produkcni symbol:

```text
CompiledNodeExecutionMetadataTest -> PreparedStepMetadataTest
DefaultRegionOptimizerTest -> DefaultRegionPlannerTest
DefaultRegionOptimizerServiceTest -> DefaultRegionPlannerServiceTest
```

Neodkazovat na neexistujici Metal/CUDA dispatcher testy. Accelerator overeni je
pokryto existujicimi `CudaAcceleratorExecutionPathTest`,
`MetalLayoutAwareDeviceFlowTest`, `MetalBufferTraceSmokeTest` a buffer tests.

### Task 9.3: Dokumentace

Status: `[x]`

Aktualizovat:

```text
src/main/java/graph/README.md
src/main/java/graph/optimizer/README.md
src/main/java/backend/README.md
src/main/java/prepare/README.md
README.md po povinnem `rg` auditu puvodnich package nazvu
docs/development.md
docs/configuration.md
docs/testing.md
docs/native-bridges-and-blas.md
docs/troubleshooting.md
```

Dokumentace musi popsat:

- `CompiledGraph` jako jedinou lifecycle facade,
- planning jako compile-time model,
- prepare context vs orchestration,
- backend-specific prepare jako backend compiler,
- executable artifact dispatch bez `ComputeEngine`,
- top-level trace DTO a producer ownership,
- runtime memory/residency hranici.
- `config.runtime.BlasProvider` jako data-only volbu a
  `backend.provider.blas.openblas` jako JDK/FFM-only external compute provider.

Pet aktivnich BLAS dokumentu ma tento presny rewrite scope:

- `docs/development.md`: odstranit tvrzeni o bundled JavaCPP fallbacku a uvadet
  pouze `openblas.lib` -> `OPENBLAS_LIB` -> system library name `openblas`.
- `docs/configuration.md`: odstranit tabulku `cg.cpu.blas.*` system properties,
  protoze provider/threshold/debug/shape/thread policy vlastni
  `RuntimeConfig`/`BlasConfig` a profile IO; popsat `openblas.lib` a
  `OPENBLAS_LIB` vyhradne jako library-location vstupy se stejnym trojclennym
  lookup poradim.
- `docs/testing.md`: odstranit predpoklad bundled OpenBLAS dostupnosti; native
  test setup dokumentuje explicitni property, environment cestu nebo system
  library a zachova capability-gated testy.
- `docs/native-bridges-and-blas.md`: nahradit lookup ukazku presnym FFM-only
  `resolveLookup` z Task 8.1, odstranit JavaCPP preset/dependency a
  `BUNDLED_JAVACPP` source, upravit lookup-source/thread-policy popis a vsechny
  source links na finalni ownership.
- `docs/troubleshooting.md`: odstranit doporuceni spolehat na bundled runtime,
  dokumentovat presne poradi `openblas.lib` -> `OPENBLAS_LIB` -> `openblas` a
  diagnostiku chybne explicitni/environment cesty bez ticheho fallbacku.

Ve vsech peti souborech se kazdy stary OpenBLAS source odkaz
`src/main/java/backend/blas/{OpenBlasSymbols,OpenBlasRuntime,OpenBlasGemmLayout,OpenBlasArrayGemm,OpenBlasSegmentGemm}.java`
prepise na `src/main/java/backend/provider/blas/openblas/...`. Odkaz na
`backend/blas/BlasProvider.java` se prepise na
`src/main/java/config/runtime/BlasProvider.java`; odkaz na `BlasRuntime.java` se
odstrani a jeho policy obsah se prepise na `BlasConfig`/profile IO. Zadny bundled
JavaCPP claim, `org.bytedeco` odkaz ani stara BLAS source cesta v techto dokumentech
po migraci nezustane.

### Task 9.4: Boundary test finalni kontrola

Status: `[x]`

Overit, ze `PackageOwnershipBoundaryTest` obsahuje vsechny cilove metody uvedene
ve fazi 0, kontroluje skutecne cilove cesty a nema zadnou podminenou nebo vypnutou
aserci.

### Task 9.5: Overeni

Status: `[x]`; vsechny Phase 9-owned kontroly vcetne `PreparedExecutionBuildTest`
prosly; capability-dependent OpenBLAS pripady byly pri nedostupne knihovne preskoceny.

```bash
./gradlew classes
./gradlew test --tests SourceTreeHygieneTest
./gradlew test --tests PackageOwnershipBoundaryTest
./gradlew test --tests PreparedExecutionBuildTest
./gradlew test --tests CompiledGraphTraceTest

test -f docs/development.md
test -f docs/configuration.md
test -f docs/testing.md
test -f docs/native-bridges-and-blas.md
test -f docs/troubleshooting.md
rg -n 'backend/blas|org\.bytedeco|BUNDLED_JAVACPP|bundled (JavaCPP )?OpenBLAS' \
  docs/development.md docs/configuration.md docs/testing.md \
  docs/native-bridges-and-blas.md docs/troubleshooting.md
```

Vsechny `test -f` prikazy musi uspet a dokumentacni `rg` vystup musi byt prazdny.

Commit:

```text
test: enforce final package architecture
```

## Faze 10: Finalni Verifikace

Status: `[ ]`

### Task 10.1: Compile a targeted suites

```bash
./gradlew classes
./gradlew test --tests SourceTreeHygieneTest
./gradlew test --tests PackageOwnershipBoundaryTest
./gradlew test --tests PreparedExecutionBuildTest
./gradlew test --tests CompiledGraphTraceTest
./gradlew test --tests ComputeModeTraceTest
./gradlew test --tests runtime.*
./gradlew test --tests planning.*
./gradlew test --tests runtime.device.buffer.*
./gradlew test --tests runtime.memory.nativecpu.NativeCpuStorageTest
./gradlew test --tests backend.cpu.nativecpu.NativeBFloat16KernelsTest
./gradlew test --tests backend.cpu.nativecpu.NativeCpuElementwiseChainTest
./gradlew test --tests backend.cpu.nativecpu.NativeCpuRegionSelectionTest
./gradlew test --tests backend.cpu.nativecpu.NativeOpenBlasMatMulExecutableTest
./gradlew test --tests backend.cpu.nativecpu.NativeOpenBlasPlannerTest
./gradlew test --tests backend.provider.blas.openblas.OpenBlasGemmTest
./gradlew test --tests BFloat16BlasDispatchTest
./gradlew test --tests backend.cpu.nativecpu.layout.*
./gradlew test --tests backend.cpu1.*ExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1FusedGeneratedExecutionTest
./gradlew test --tests backend.cuda.CudaAcceleratorExecutionPathTest
./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest
./gradlew test --tests backend.metal.MetalBufferTraceSmokeTest
```

### Task 10.2: Plna test suite

```bash
./gradlew test
./gradlew metalTest
```

Pokud `metalTest` nema dostupny native shim, zaznamenat presny environment blocker
v commit/PR popisu; nezaskrtnout Metal bod Definition of Done, dokud neni test
proveden v prostredi se shimem.

### Task 10.3: Finalni source audit

```bash
test ! -e src/main/java/graph/execution
test ! -e src/main/java/graph/compile/descriptor
test ! -e src/main/java/graph/compile/intent
test ! -e src/main/java/graph/compile/planning
test ! -e src/main/java/backend/prepare
test ! -e src/main/java/backend/runtime
test ! -e src/main/java/backend/memory
test -z "$(find src/main/java/backend/blas -type f -name '*.java' -print 2>/dev/null)"
test ! -e src/main/java/backend/ComputeEngine.java
test ! -e src/test/java/backend/accelerator/buffer
test ! -e src/test/java/backend/cpu/nativecpu/NativeCpuStorageTest.java

rg -n "graph\.execution|graph\.compile\.(descriptor|intent|planning)|backend\.prepare|backend\.runtime|backend\.memory|backend\.blas|\bBlasRuntime\b|backend\.accelerator\.buffer|backend\.ComputeBackend|CompiledNodeExecutionMetadata|PreparedExecutionArtifact|OptimizedRegion|DefaultRegionOptimizer|RegionOptimizationContext|RegionOptimizationTrace|CpuRegionOptimizationPolicy" \
  src/main/java src/test/java \
  --glob '!SourceTreeHygieneTest.java' --glob '!PackageOwnershipBoundaryTest.java'

test -f src/main/java/config/runtime/BlasProvider.java
test -f src/main/java/backend/provider/blas/openblas/OpenBlasRuntime.java
test -f src/main/java/backend/provider/blas/openblas/OpenBlasSymbols.java
test -f src/main/java/backend/provider/blas/openblas/OpenBlasGemmLayout.java
test -f src/main/java/backend/provider/blas/openblas/OpenBlasArrayGemm.java
test -f src/main/java/backend/provider/blas/openblas/OpenBlasSegmentGemm.java
test -f src/test/java/backend/provider/blas/openblas/OpenBlasGemmTest.java

rg -n '^import (config|graph|planning|prepare|runtime|trace|tensor|backend\.(cpu|cpu1|metal|cuda|opencl))\.' \
  src/main/java/backend/provider/blas/openblas

rg -n --pcre2 '^import (?!java\.|static java\.)' \
  src/main/java/backend/provider/blas/openblas

rg -n 'org\.bytedeco|openblas-platform' build.gradle src/main/java

rg -n '^import tensor\.' src/main/java/trace

rg -n "graph\.execution\.PreparedExecution|backend\.runtime\.ExecutionMode" \
  src/main/java/tensor/Tensor.java src/main/java/tensor/internal/TensorExecution.java

rg -n "backend\.memory\.ExecutionResource" \
  src/main/java/tensor/storage/NativeMemoryAllocation.java
```

Vsechny `test ! -e` prikazy a kontrola prazdneho legacy BLAS source tree musi
vratit exit code 0 a vsechny `rg` vystupy musi byt prazdne.

### Task 10.4: Finalni commit

```text
docs: finalize package architecture migration
```

## Risky A Mitigace

| Riziko | Konkretni mitigace |
|---|---|
| Verejne `CompiledGraph.prepare()` se rozbije | Facade zustava; meni se jen import builderu a return type package. Vsechny call sites se prepnou v jednom commitu. |
| Runtime prestane dispatchovat spravny backend | Kazdy prepare route test asertuje konkretni `PreparedStepExecutable`; runner nema fallback. |
| Interior partition node dostane executable step | Builder coverage test a explicitni exception v prepareru. |
| Trace presun vytvori kruh zpet do planning/backend | Trace records drzi snapshot hodnoty; boundary test zakazuje producer imports. |
| Planning presun vytvori kruh s backend registry | Planning-owned `BackendPartitionCapabilityRegistry`; concrete registry se injectuje z GraphCompiler. |
| Native memory pool zmeni lifetime | Implementace se presune beze zmeny tela; ExecutionState/PreparedExecution tests kontroluji close/reuse. |
| Alias view se vyhodnoti jinak | Rozhodnuti se pouzije z `storageOwnerId` vypocteneho snapshotterem; view propagation tests zustanou. |
| Cpu1 hot path se zmeni package migraci | Cpu1 kernel body se nemeni; targeted contract a generated fusion tests jsou povinne. |
| BLAS provider znovu prevezme config nebo backend policy | Dva boundary testy zakazuji vyssi vrstvy a konkretni backend imports; route/debug/thread snapshot se asertuje v prepare tests. |
| Odstraneni hot-path capability checku zavola chybejici symbol | CPU a cpu1 prepare query capability pred vyberem route; native executable a kernel lze vytvorit jen pro pripraveny dostupny symbol. |
| OpenBLAS thread override unikne do dalsiho kroku | CPU adapter/native executable i cpu1 executable aplikuji prepared thread count a vzdy obnovi predchozi hodnotu v `finally`; contract a benchmark tests kontroluji obnoveni. |
| Global import rewrite zasahne dokumentaci nebo string fixtures | Prepisovat imports pres symbol-aware IDE nebo pres presne package prefixy; po kazde fazi compile + rg. |
| Stare package zustane skryte v testu | SourceTreeHygieneTest a PackageOwnershipBoundaryTest kontroluji nepritomnost Java sources pod legacy cestou; finalni rg kontroluje package symboly. Samotny prazdny adresar se netestuje, protoze jej Git neeviduje. |

## Definition Of Done

- [ ] `graph.CompiledGraph` je jediny graph soubor importujici prepare/runtime.
- [ ] `graph.model` neimportuje planning, prepare, runtime ani konkretni backend.
- [ ] `CompiledNodeSnapshotter` vlastni Tensor-to-CompiledNode snapshot logiku.
- [ ] `CompiledProgram` je v `graph.compile`.
- [ ] `backend.contract.ComputeBackend` je dependency leaf.
- [ ] Sedm puvodnich `backend/blas` Java souboru ma presne sest move cilu a jeden delete cil podle Task 8.1.
- [ ] `config.runtime.BlasProvider` je data-only enum bez `fromProperty` a profile IO pouziva generic `findEnum`.
- [ ] `BlasRuntime` neexistuje a `src/main/java/backend/blas` neobsahuje zadny Java source.
- [ ] Presne pet OpenBLAS implementacnich trid je v `backend.provider.blas.openblas` a importuje jen JDK/FFM.
- [ ] JavaCPP OpenBLAS imports a `openblas-platform` Gradle dependency jsou odstraneny.
- [ ] CPU a cpu1 prepare jednou rozresi route, thresholdy, capability, debug a thread hodnoty z `BlasConfig`/tuning state.
- [ ] Provider ani produkcni hot path znovu nectou provider, threshold, shape gate, debug nebo thread konfiguraci.
- [ ] `MatMulBlasBackend` zustava CPU adapterem a importuje sdileny OpenBLAS provider.
- [ ] Test consumer arithmetic je presne 23: `OpenBlasGemmTest` je jeden move se source-path/method-name rewritem, 18 testu je import/FQN-only, tri benchmark testy maji lokalni formatting helper rewrite a `SourceTreeHygieneTest` ma hygiene extension.
- [ ] PackageOwnershipBoundaryTest i SourceTreeHygieneTest zakazuji vyssi vrstvy a konkretni backend imports v OpenBLAS provideru.
- [ ] Vsechny trace DTO jsou pod top-level `trace`.
- [ ] Trace DTO neimportuji sve producenty.
- [ ] Trace DTO neimportuji `tensor`; dtype je lossless `DataType.name()` string.
- [ ] Step, transfer a native optimizer trace producenti snapshotuji dtype explicitne.
- [ ] Vsechny descriptor, intent a planning packages jsou pod `planning`.
- [ ] `planning.partition.cost`, `planning.region.lowering` a
  `planning.region.specialization` jsou kompletne presunuty.
- [ ] Pouzivaji se pouze `PlannedRegion`, `DefaultRegionPlanner`,
  `RegionPlanningContext`, `RegionPlanningTrace`, `CpuRegionPlanningPolicy`.
- [ ] Vsech deset souboru puvodniho `backend.memory` ma cil podle audit tabulky.
- [ ] `backend.memory` adresar neexistuje.
- [ ] `backend.runtime` adresar neexistuje a `ExecutionMode` je v runtime contractu.
- [ ] Backend-neutral accelerator buffer model je v `runtime.device.buffer`.
- [ ] Vsechny tri accelerator buffer model testy jsou v `runtime.device.buffer`.
- [ ] Runtime native allocator/pool je v `runtime.memory.nativecpu`.
- [ ] `NativeCpuStorageTest` je v `runtime.memory.nativecpu`.
- [ ] Sedm backend nativecpu kernel/policy/layout testu zustava v puvodnich backend packages.
- [ ] `PreparedStepMetadata` vyzaduje non-null `PreparedStepExecutable`.
- [ ] Runtime runner vola executable artifact naprimo.
- [ ] `backend.ComputeEngine` a mrtve backend dispatch fasady jsou odstraneny.
- [ ] Runtime neimportuje konkretni backend implementation package.
- [ ] Prepare context/validation neimportuji concrete preparery ani orchestration.
- [ ] Backend preparery neimportuji `prepare.orchestration`.
- [ ] `backend.prepare` adresar neexistuje.
- [ ] `RegionPlanValidatorTest` je v `prepare.validation`.
- [ ] Cpu1 prepare zustava backend-owned a pouziva finalni shared contracts.
- [ ] `Tensor` a `TensorExecution` importuji finalni `ExecutionMode` a `PreparedExecution`.
- [ ] `NativeMemoryAllocation` extends finalni `runtime.memory.ExecutionResource`.
- [ ] Zadny dalsi tensor soubor neimportuje runtime implementation package.
- [ ] SourceTreeHygieneTest pouziva vsechny finalni cesty.
- [ ] PackageOwnershipBoundaryTest nema vypnute ani podminene aserce.
- [ ] Global legacy symbol audit je prazdny.
- [ ] `./gradlew classes` prochazi.
- [ ] Vsechny targeted suites z faze 10 prochazeji.
- [ ] `./gradlew test` prochazi.
- [ ] `./gradlew metalTest` prochazi v prostredi s native shimem.
- [ ] Dokumentace popisuje pouze finalni strukturu.
- [ ] Vsech pet aktivnich BLAS dokumentu popisuje lookup pouze jako `openblas.lib` -> `OPENBLAS_LIB` -> `openblas`, nema JavaCPP/bundled claim a odkazuje na finalni provider/config cesty.
- [ ] Neexistuje compatibility alias, docasna facade, duplicitni registry ani mrtva cesta.

## Commit Strategie

Kazdy commit musi byt samostatne kompilovatelny a reviewovatelny:

```text
1. test: lock package ownership migration boundaries
2. refactor: isolate compiled graph model dependencies
3. refactor: extract lifecycle trace model
4. refactor: extract complete compile planning model
5. refactor: move shared runtime memory and device contracts
6. refactor: execute prepared artifacts without backend dispatch
7. refactor: move prepared execution runtime out of graph
8. refactor: separate prepare context from orchestration
9. refactor: assign blas configuration and provider ownership
10. refactor: migrate backend consumers to final contracts
11. test: enforce final package architecture
12. docs: finalize package architecture migration
```

Faze 8 ma dva tematicke commity: prvni obsahuje presne BLAS `6 move + 1 delete`,
prepared-value rewrites a test scope `1 move + 18 import/FQN-only + 3 helper
rewrites + 1 hygiene extension = 23`; druhy obsahuje zbyvajici backend consumer
migraci. Tim souhlasi source/test pocty v Task 8.1, Definition of Done i commit
scope.

Do commitu nepatri lokalni benchmark/calibration artefakty, build output ani
temporary verification soubory.

## Tracking Tabulka

| Faze | Implementace | Targeted testy | Boundary audit | Commit |
|---|---|---|---|---|
| 0 | [x] | [x] | [x] | [x] |
| 1 | [x] | [x] | [x] | [x] |
| 2 | [ ] | [ ] | [ ] | [ ] |
| 3 | [ ] | [ ] | [ ] | [ ] |
| 4 | [ ] | [ ] | [ ] | [ ] |
| 5 | [ ] | [ ] | [ ] | [ ] |
| 6 | [ ] | [ ] | [ ] | [ ] |
| 7 | [ ] | [ ] | [ ] | [ ] |
| 8 | [ ] | [ ] | [ ] | [ ] |
| 9 | [ ] | [ ] | [ ] | [ ] |
| 10 | [ ] | [ ] | [ ] | [ ] |
