# Graph Package Clean Architecture Refactor

## Stav Rozpracovani

Status: `IMPLEMENTED`

Implementovane slices:

- `PreparedNodeExecution` byl prejmenovan na `PreparedExecutionStep` ve zdrojich, testech a aktivni dokumentaci.
- `PreparedExecutionRunner` uz nerozhoduje CPU input materialization/output residency podle konkretnich CPU/native CPU artefaktu; pouziva neutralni `InputResidencyRequirement` a `OutputResidencyEffect` v prepared metadatech.
- CUDA graph lowering family byla sjednocena na `CUDA_GRAPH_REGION`; fused elementwise zustava pattern/metadata, ne samostatna CUDA lowering family.
- Backend-neutral `MATMUL_EPILOGUE` execution unit byla odstranena ze structural region planningu; matmul + bias/activation se nerozhoduje jako obecna graph-level unit pred backend/provider volbou.
- CPU fused execution zachovava offset/strided external input metadata a nenecha obecnou prepared-input materializaci rozbit fused plan storage offsets.
- `StepExecutionTracer` uz nesklada CPU/OpenBLAS/native CPU/accelerator detaily sam; bere backend-owned `StepTraceContribution` z prepared artifactu a pridava jen obecne storage/fallback summary atributy.
- Stara `graph.execution.trace.contrib` contributor/support vrstva byla odstranena; CPU trace evidence je v `backend.cpu.CpuStepTraceContributor`, accelerator trace evidence v `AcceleratorExecutionArtifact`.
- `RuntimeWorkspaceStore` uz neimportuje CPU/accelerator artifacty; run-scoped workspace a prepared inputs alokuje pres backend-neutral `PreparedRuntimeStateAllocator` volany z prepared artifactu.
- `PreparedExecution.backward()` uz nepise na stdout; unsupported backward convenience call explicitne vyhodi `IllegalStateException`.
- `SourceTreeHygieneTest` hlida trace import hranici, opaque runtime workspace store a zakaz stdout/stderr v `src/main/java/graph`.
- CPU fused execution uz nevznika pres `ANCHOR`/`INTERIOR` role nad puvodnimi nody. Prepare emituje jeden multi-node `PreparedExecutionStep` pro fused unit, s explicitnim `orderedNodeIds` a `boundaryOutputNodeIds`, a covered nody uz nedostanou samostatne runtime stepy.
- CPU native region a Metal/CUDA graph regiony uz take vznikaji jako explicitni multi-node `PreparedExecutionStep` podle startu regionu, ne jako runtime anchor plus skip interior nodu.
- `CompiledNodeExecutionMetadata`, `ComputeEngine`, `PreparedExecutionBuilder` a test metadata factory uz neobsahuji `PartitionExecutionRole`; role zustava pouze backend-prepare-local index pro legacy lookup.
- Region prepared stepy podporuji vice boundary outputs. Reprezentativni boundary node nese backend artifact, ale `PreparedExecutionRunner` po provedeni oznaci residency pro vsechny `boundaryOutputNodeIds`.
- `OptimizerGraphSupport` byl nahrazen domenovym pracovnim modelem `OptimizerGraph`.
- `RegionOptimizationUnitSupport` byl rozdelen na `ExecutionUnitFactory` a `ElementwiseFusionPlanner`.
- `MemoryPlan` byl rozdelen na `TensorMemoryPlan`, `RegionMemoryPlan` a `RuntimeBindingPlan`; hlavni plan je agregat explicitnich casti.
- `GreedyMaxRegionPartitionPlanner` a `AnchorBasedPartitionPlanner` byly slouceny do `MaxRegionPartitionPlanner` s uzkym `SeedOrdering`.
- `CompileSession` uz neni script nad statickymi stage tridami; compile use-case vlastni primo pres pojmenovane workflow metody.
- `ExecutionState` zustava public per-run entrypoint, ale native CPU memory a device memory invarianty jsou oddelene do `RuntimeNativeCpuMemoryState` a `RuntimeDeviceMemoryState`.
- Graph source hygiene hlida odstraneni `*Support`/`*Helper`/`*Adapter` junk-drawer trid v `graph`, sjednoceny max-region planner a odstranene staticke compile stage tridy.
- `.planning/codebase`, graph README, optimizer README a aktivni docs byly aktualizovane na aktualni package ownership.

Zustava rozpracovane:

- Nic z rozsahu tohoto todo. Dalsi prace by uz mela byt nova oblast nebo follow-up vykonove ladeni mimo tento architektonicky refaktor.

Navazuje na:

- `todo/38-graph-architecture-cleanup-and-reorganization.md`
- `todo/88-graph-compile-partition-lowering-model-cleanup.md`
- `todo/89-graph-runtime-boundary-cleanup.md`
- `todo/90-memory-planner-responsibility-split.md`
- `todo/94-graph-publication-contract-and-compiled-program-cleanup.md`

## Kontext

Balicek `graph` uz ma spravny smer: public `Tensor` zustava logicky, compile vytvari immutable program snapshot, prepare vytvari runtime metadata a kazdy execution run ma vlastni `ExecutionState`.

Po poslednim review ale zustavaji architektonicke dluhy, ktere snizuji citelnost a rozsiritenost:

- `graph.execution` porad zna konkretni CPU/native CPU a OpenBLAS detaily pres `PreparedExecutionRunner` a `StepExecutionTracer`.
- `ExecutionState` je prilis siroky runtime objekt: runtime tensors, workspaces, residency, device bindings, native storage, resources a materialization service jsou vystavene pres jednu tridu.
- `CompileSession` je sice rozdeleny do stage trid, ale vysledek je porad script poskladany ze statickych pseudo-helperu.
- `MemoryPlan` je jeden pretezene velky kontejner pro tensor-level i region-level memory planning.
- `GreedyMaxRegionPartitionPlanner` a `AnchorBasedPartitionPlanner` sdili velkou cast algoritmu, ale maji ji zkopirovanou.
- Loop fusion neni v `graph.optimizer`, ale dokumentace a nazvy kolem `FUSE` to porad mohou plest. Skutecny owner je compile region planning plus backend CPU fused prepare/codegen.
- Partitioning je dnes pro fusion prilis silna hranice: pokud vznikne uzka partition, loop fusion uz ji nemuze prekrocit, i kdyz jde o stejny backend a zadnou skutecnou materializacni barieru.
- `ExecutionUnit` uz dnes existuje jako region-level artefakt: CPU lowerer podle nej loweruje jednotlive kroky, zatimco Metal/CUDA lowerery typicky loweruji cely region a units pouzivaji jen jako summary/hint.
- Runtime dnes fusion maskuje pres `ANCHOR`/`INTERIOR` role nad puvodnimi graph nody. To dela z fused execution unity special case v node runneru misto primarni jednotky prepared execution planu.
- Graph layer obsahuje `*Support` technicke junk-drawery, ktere nejsou domenove modely.
- `graph.execution` ma stale user-facing side effect pres `System.out.println(...)` v `PreparedExecution.backward()`.
- `.planning/codebase` a cast docs stale popisuji starsi optimizer package ownership.

Cilem tohoto dokumentu je sjednotit tyto jednotlive cleanupy do jednoho ciloveho refaktoringu balicku `graph`.

## Cile

- Udelat `graph` citelny podle lifecycle hranic: compile, optimize, planning, prepare, execution, publication, trace.
- Udrzet public `Tensor` API logicke a bez backend residency.
- Odstranit backend-specific znalosti z graph execution vrstvy.
- Zmensit runtime state surface tak, aby backendy pracovaly pres uzke, zamerne runtime kontrakty.
- Nahradit staticke stage/helper tridy primymi domenovymi workflow objekty.
- Zpresnit memory planning model tak, aby nepouzival prazdne mapy jako signal rezimu.
- Sloucit duplicitni max-region partition planning algoritmus bez abstraktniho planner frameworku.
- Zafixovat loop fusion ownership mimo `graph.optimizer`: optimizer dela rewrite, backend-specific execution planning voli granularitu, CPU backend dela fused executable.
- Zachovat `ExecutionUnit` jako spolecny region-level planning artefakt, ale ne jako povinny runtime rozpad pro vsechny backendy.
- Oddelit backend/region partitioning od loop fusion: partition je coarse backend/materialization boundary, CPU fusion je execution-unit planning uvnitr CPU regionu, GPU backend muze region lowerovat vcelku.
- Nahradit `ANCHOR`/`INTERIOR` skipovani prepared execution planem, ktery primo obsahuje spustitelne kroky.
- Odstranit stdout/logging side effects z graph runtime API.
- Aktualizovat dokumentaci a hygiene pravidla tak, aby odpovidala realnemu balicku.

## Cile Mimo Scope

- Nemenit public `Tensor` API.
- Nemenit semantiku compile/prepare/execute.
- Nezavadet `GraphV2`, compatibility aliasy, docasne adaptery, prechodove wrappery ani paralelni old/new cesty.
- Nezavadet obecne helpery, support tridy, manager tridy, visitor frameworky nebo registry jen kvuli organizaci.
- Nezavadet nove abstrakce, pokud neodstranuji realnou neurcitost nebo realnou duplicitu.
- Nepresouvat backend implementation code do `graph`.
- Neresit kalibraci vykonovych heuristik partitioningu, provider selectu nebo memory reuse mimo nutne behavior-preserving upravy. Tento plan smi zpresnit architektonicke misto, kde se provider/family rozhodnuti dela, ne ladit finalni thresholdy.

## Cilovy Stav

### Package Ownership

```text
graph
  public compile facade and immutable compile model

graph.compile
  compiler workflow, compile artifacts, graph contract, descriptors

graph.compile.planning
  backend ownership planning, partitioning, region planning, memory planning

graph.optimizer
  backend-neutral graph rewrites only

graph.execution
  prepared execution model, execution run orchestration, publication, trace records

graph.execution.state
  run-scoped state stores and narrow runtime services
```

`graph.execution` nesmi importovat konkretni CPU/Metal/CUDA/OpenBLAS execution payloady jen kvuli runtime input policy, residency finalization nebo trace atributum.

### Terminologie

Tento plan pouziva tyto pojmy presne:

- `optimizer` znamena semanticke/canonical graph rewrites. Neni to misto pro execution planning.
- `region planning` znamena compile-time rozhodnuti o backend/materialization hranicich.
- `ExecutionUnit` znamena region-level planning artefakt popisujici navrzenou fyzickou granularitu a hodnotove hranice.
- `lowering family` znamena finalni backend fyzickou family po loweringu, napriklad `BLAS`, `FUSED_NATIVE`, `METAL_GRAPH_REGION`, `CUDA_GRAPH_REGION`.
- `PreparedExecutionStep` znamena runtime-spustitelny prepared krok. Neni to wrapper/adaptacni vrstva nad puvodnim node runnerem.

Slovo "optimizer" se nema pouzivat pro region/execution-unit planning tridy. Pokud dnes existuje `DefaultRegionOptimizer`, cilovy nazev ma byt planning-oriented, napriklad `DefaultRegionPlanner` nebo konkretnejsi domenove jmeno podle odpovednosti.

### Compile Workflow

`CompileSession` ma byt vlastnik compile use-case, ne jen orchestrator statickych stage helperu.

Cilovy tok:

```text
captureForwardGraph()
compileBackwardClosure()
optimizeGraph()
snapshotProgram()
planBackendOwnership()
planRegionsAndMemory()
buildPublicationPlan()
assembleArtifacts()
```

Tyto kroky maji byt instance metody `CompileSession`, pripadne male domenove objekty s vlastnim stavem a jasnou odpovednosti. `ForwardGraphCapture`, `OptimizerSnapshotStage`, `CompiledProgramSnapshotStage`, `BackendOwnershipPlanningStage`, `RegionAndMemoryPlanningStage` a `PublicationPlanBuilder` nemaji zustat jako staticke helper/stage tridy, pokud pouze rozsekavaji jeden script.

### Optimizer

`graph.optimizer` zustava uzky:

- algebraic/canonical rewrites,
- constant folding,
- CSE,
- DCE,
- semantic-only lowering, pokud zachovava backend-neutral graph semantiku.

Neobsahuje:

- loop fusion,
- CPU fused executable planning,
- ASM/vector fused backend selection,
- backend-specific fusion capability truth,
- memory reuse planning,
- backend ownership planning.

`backend-neutral lowering` nesmi znamenat vyber backend family, provideru, storage layoutu, fused executable ani device graphu. Pokud transformace zavisi na backend capability nebo cost modelu, patri az za backend/region planning.

`OptimizerGraphSupport` se ma nahradit domenovym modelem pracovnio optimizer graphu, napriklad `OptimizerGraph`, ktery vlastni:

- input rewiring,
- replacement chain resolution,
- observable roots,
- topological closure rebuild.

Toto neni pomocny helper. Je to explicitni model mutable optimizer working graphu.

### Loop Fusion Ownership

Loop fusion ma zustat mimo `graph.optimizer`.

Cilovy tok:

```text
graph.optimizer
  -> canonical graph rewrites only

graph.compile.planning.partition
  -> ownership regions

graph.compile.planning.region
  -> backend-aware region planning; CPU may split into execution units, Metal/CUDA may keep whole graph regions

backend.lowering / backend.cpu.prepare
  -> backend/cost selected physical execution, including optional matmul epilogues and prepared fused executables

backend.cpu.fused
  -> interpreted/generated/vector/ASM execution implementation
```

Pravidla:

- `graph.optimizer/FUSE.md` bud aktualizovat jako historicky/conceptual document, nebo presunout obsah pod compile region planning docs.
- `Operation.OpType.FUSED` zustava CPU/backend execution artifact, ne obecny optimizer rewrite target.
- Region optimizer muze rikat "tahle subchain je fused execution unit"; nesmi rozhodovat, jaky CPU fused backend ji vykona.
- Matmul epilogue fusion neni obecna graph-level execution unit. Je to backend/cost rozhodnuti: napriklad pri OpenBLAS muze byt lepsi ponechat `MATMUL` jako provider call a bias/aktivaci spustit samostatne v Java/CPU fused ceste.
- `ExecutionUnit` neni prikaz backendu. Je to planovaci navrh fyzicke granularity a hodnotovych hranic. Backend lowerer ji muze prijmout, pregrupovat, rozpadnout, nebo ignorovat, pokud ma lepsi whole-region lowering.
- CPU pouziva execution units aktivne kvuli provider callum, Java kernelum, MemorySegment/native storage cestam a loop fusion. Metal/CUDA graph backend muze idealne dostat celou region/partition jako jeden lowered graph a units ponechat jen jako hint/trace summary.
- CUDA graph lowering family ma byt zarovnana s Metalem: `CUDA_GRAPH_REGION` je fyzicka family pro graph/DAG region. `FUSED_ELEMENTWISE` je pattern metadata/hint, ne samostatna CUDA lowering family typu `CUDA_FUSED_ELEMENTWISE_GRAPH`.
- CPU fused backend nesmi prosakovat zpet do `graph.optimizer`.

### Region Vs Execution Unit Planning

Partitioning nesmi byt primarni mechanismus fusion.

Cilovy obecny tok:

```text
Compiled graph
  -> BackendRegion planning
  -> Backend-specific execution planning
  -> Backend lowering
  -> PreparedExecution steps
```

`BackendRegion` znamena hrubou fyzickou oblast:

- stejny backend target,
- zadna nutna materializacni bariera uvnitr,
- stabilni storage/residency podminky,
- hranice pro accelerator launch nebo CPU natural execution region.

`BackendRegion` neznamena "jeden kernel" ani "jedna fused loop".

Backend-specific execution planning muze mit ruznou granularitu podle targetu. Po zvoleni region targetu muze CPU planner rozhodnout jinak nez Metal/CUDA planner, protoze fyzicke families a cost model jsou jine. Stejny logicky podgraf muze byt:

- na CPU/OpenBLAS: `PROVIDER_CALL MATMUL` + samostatny bias/activation step,
- na CPU/java kernelu: pripadne fused matmul epilogue, pokud to konkretni kernel umi a vyplati se,
- na Metal/CUDA graph backendu: jeden lowered graph region, pokud backend umi cely pattern efektivne.

Execution units jsou tedy spolecny region-level planning artefakt, ale jejich zavaznost je backend-specific. CPU/JVM je pouziva jako jemnejsi fyzicke kroky uvnitr CPU regionu. Pro Metal/CUDA neni cilem rozpadat region na `UNIT_KERNEL`/`FUSED_ELEMENTWISE` units, pokud backend umi prijmout a optimalizovat cely graph region.

### ExecutionUnit Contract

`ExecutionUnit` ma popisovat:

- ktere compiled nody patri k planovane fyzicke granularite,
- vstupni a vystupni `GraphValueRef`,
- ktere hodnoty jsou materializovane,
- ktere hodnoty jsou virtualni mezivysledky,
- odhad prace a trace duvod, proc byla granularita navrzena.

`ExecutionUnit` nema popisovat:

- konkretni provider typu OpenBLAS, Accelerate nebo MemorySegment executor,
- finalni CPU dispatch family,
- Metal/CUDA graph executable,
- backend-specific generated code.

Invarianty:

- Units uvnitr jednoho regionu maji pokryvat region nodes deterministicky v topologickem poradi.
- Jeden node nesmi byt soucasti vice execution units ve stejnem regionu.
- `inputRefs` jsou hodnoty ctene z vnejsku unity nebo z drivejsich units; nesmi obsahovat virtualni hodnotu produkovanou pozdejsi unitou.
- `outputRefs` jsou hodnoty publikovane z unity dalsim units, region boundary nebo publication planu.
- `virtualRefs` jsou pouzitelne pouze uvnitr unity; memory/runtime plan je nesmi vyzadovat jako samostatne runtime bindingy.
- `materializedRefs` jsou hodnoty, ktere musi mit runtime storage kvuli region boundary, fanoutu, publication, gradientu, fallbacku nebo backend handoffu.
- Backend lowerer smi unit ignorovat jako execution granularitu, ale nesmi ignorovat jeji value-boundary fakta bez ekvivalentni nahrady v lowered region planu.

Tyto volby patri do lowering/prepare. CPU lowerer ma podle `ExecutionUnit` a runtime/compile policy vybrat napriklad:

```text
CPU region:
  matmul -> add_bias -> relu

OpenBLAS/Accelerate path:
  ExecutionUnit [matmul]
    -> ProviderCallStep(MATMUL)
  ExecutionUnit [add_bias, relu]
    -> FusedElementwiseStep

Java/native CPU path:
  ExecutionUnit [matmul, add_bias, relu]
    -> optional backend-selected native matmul epilogue only if supported and cheaper
```

Pro elementwise chain:

```text
ExecutionUnit FUSED_ELEMENTWISE [add, mul, tanh]
  -> Java loop / Vector API / ASM / MemorySegment prepared step podle backend policy
```

Pro Metal/CUDA:

```text
BackendRegion [1, 2, 3, 4, 5]
  executionUnits = optional structural hints/summary
  lowering -> one Metal/CUDA graph region when supported
```

CUDA/Metal lowering family naming:

```text
METAL_GRAPH_REGION
CUDA_GRAPH_REGION
```

Elementwise/fused informace zustava mimo `LoweringFamily`, napriklad v:

- `GpuCompoundRegionSummary.patternType`,
- `GpuRegionLoweredUnitSummary`,
- region trace/plan metadata,
- execution-unit summary v backend payloadu.

Nepouzivat `CUDA_FUSED_ELEMENTWISE_GRAPH`, protoze micha fyzicky execution family (`CUDA_GRAPH_REGION`) se strukturou regionu (`FUSED_ELEMENTWISE`).

Matmul epilogue specializace:

- `MATMUL_EPILOGUE` nema vznikat jako obecna graph-level unit v backend-neutral/structural region planningu.
- Soucasnou `MATMUL_EPILOGUE` emisi v `StructuralRegionUnitPlanner` je nutne zreviewovat a odstranit nebo presunout do CPU/backend-selected planningu.
- Pokud backend skutecne umi a chce matmul epilogue, ma to byt lowering/prepared-step family zvolena az po provider/cost rozhodnuti.
- OpenBLAS/Accelerate/provider matmul nesmi byt nucene slouceny s bias/activation jen proto, ze pattern existuje v graphu.

Cilovy CPU priklad:

```text
CPU BackendRegion:
  orderedNodeIds = [1, 2, 3, 4, 5, 6]

Execution units:
  FUSED_ELEMENTWISE [1, 2, 3]
  UNIT_KERNEL       [4]
  FUSED_ELEMENTWISE [5, 6]
```

Tri sloucene CPU nody nejsou oznacene v puvodnim graphu. Jsou popsane pouze execution unitou:

```java
public record ExecutionUnit(
        String id,
        ExecutionUnitKind kind,
        List<Integer> orderedNodeIds,
        List<GraphValueRef> inputRefs,
        List<GraphValueRef> outputRefs,
        List<GraphValueRef> virtualRefs,
        List<GraphValueRef> materializedRefs
) {}
```

Pro loop fusion:

```text
kind = FUSED_ELEMENTWISE
orderedNodeIds = [12, 13, 14]
outputRefs = [node(14)]
virtualRefs = [node(12), node(13)]
```

Pokud CPU backend potrebuje anchor, odvodi si ho z execution unity jako posledni publikovany output. Anchor nesmi byt graph-level koncept ani duvod, proc runtime prochazi vsechny puvodni nody a nektere skipuje. Pokud Metal/CUDA backend potrebuje region anchor pro prepared region artifact, ma to byt lokalni lowering/prepare detail toho backendu.

### Prepared Execution Steps

Runtime nema prochazet puvodni compiled nody a skipovat `INTERIOR`.

Cilovy model:

```text
PreparedExecution:
  step 0: CpuFusedStep for [12, 13, 14]
  step 1: CpuKernelStep for [15]
  step 2: CpuFusedStep for [16, 17]
```

Runner:

```java
for (PreparedExecutionStep step : prepared.steps()) {
    step.execute(context);
}
```

`PreparedExecutionStep` je spustitelna jednotka prepare-time planu. Muze reprezentovat single node, fused CPU loop, native CPU region, Metal/CUDA region, provider call nebo backend-selected epilogue. Neni to adapter nad puvodnim node seznamem; je to cilovy runtime model.

`PartitionRoleIndex`, `PartitionExecutionRole.ANCHOR` a `PartitionExecutionRole.INTERIOR` maji byt po refaktoru odstranene z CPU fused cesty. Pokud akcelerator backend docasne potrebuje podobny pojem pro region anchor, ma byt lokalni soucasti lowered/prepared region planu, ne obecny graph execution contract.

`PreparedExecutionStep` contract:

- nese stabilni step id a ordered node ids kvuli trace/debugu,
- deklaruje runtime input node ids / value refs,
- deklaruje boundary outputs, ktere musi byt dostupne pro dalsi steps nebo publication,
- deklaruje execution phase/scope, pokud prepared plan rozlisuje forward, backward nebo combined execution,
- deklaruje input residency requirement a output residency effect bez toho, aby graph runner znal CPU/Metal/CUDA artefakty,
- vlastni backend prepared artifact nebo region executable,
- umi dodat trace attributes pres backend-owned trace contributor,
- explicitne nese fallback route/evidence, pokud prepared step muze spadnout na CPU nebo jinou family.

Graph runner nesmi z prepared stepu vyvozovat backend detaily podle `instanceof Cpu...` nebo podle `LoweringFamily`. Ma jen respektovat neutralni step contract.

### Runtime Execution Boundary

Graph runner ma delat pouze:

- projit `PreparedExecutionStep` seznam,
- vyzadat CPU-readable vstupy jen pres backend-neutral policy v prepared kroku,
- zavolat backend dispatch,
- zaznamenat run trace,
- spustit publication.

Backend-specific input-read policy a backend-specific trace atributy patri do backend-owned prepared artifacts nebo backend trace contributors, ne do centralniho graph runneru.

Single-node execution muze cilove zustat jen jednim typem prepared stepu. Pokud zustane samostatny value object, muze nest:

```java
public record PreparedNodeExecution(
        CompiledNode compiledNode,
        CompiledNodeExecutionMetadata metadata,
        InputResidencyRequirement inputResidencyRequirement,
        OutputResidencyEffect outputResidencyEffect
) {}
```

Pokud se tyto dve nove hodnoty ukazou jako prilis siroke, maji se zuzit podle realnych call sites. Nesmime ale nechat graph runner castecky castovat CPU plany jen proto, aby vedel, jestli ma volat `requireCpuReadable`.

Fused, region a provider-call execution nesmi byt modelovane jako "node metadata na anchor nodu plus skip interior nodu". Maji mit vlastni prepared step s jasnymi vstupy, vystupy, materializacnimi efekty a trace evidence.

Upresneni rozsahu:

- CPU fused path presel na prepared steps jako prvni, protoze nejvice zneuzival `ANCHOR`/`INTERIOR` jako execution mechanismus.
- CPU native region a Metal/CUDA regiony maji nasledovat stejny runtime model: prepared plan emituje jeden explicitni multi-node step pro region start; backend-local anchor muze zustat jen pro plan/executable lookup.
- Cilem neni rozbit accelerator region prepare; cilem je odstranit obecny graph runner model, kde se spusteni regionu maskuje jako jeden compiled node a ostatni nody se skipuji.

Graph runtime API nesmi tisknout na stdout. Stav typu "backward neni podporovany" ma byt:

- no-op pouze pokud je takove chovani explicitne soucasti API contractu,
- nebo vyjimka,
- nebo trace/return value.

Ne centralni `System.out.println(...)` v graph vrstve.

### Execution State

`ExecutionState` zustane jediny public per-run entry pro backendy, ale vnitrne se ma rozdelit na uzke domenove sluzby:

- runtime tensor lookup,
- workspace lookup,
- residency transitions,
- device binding access,
- native CPU storage access,
- resource lifecycle,
- materialization.

Backendy nemaji dostat sirsi API, nez realne potrebuji. Cilem neni zabalit vse do dalsi vrstvy, ale pojmenovat stabilni runtime odpovednosti a omezit nahodne krizove mutace.

### Memory Planning

`MemoryPlan` nema byt 17parametrovy kontejner s prazdnymi mapami jako rezimovym signalem.

Cilovy model:

```java
public record MemoryPlan(
        TensorMemoryPlan tensor,
        RegionMemoryPlan region,
        RuntimeBindingPlan runtimeBinding,
        MemoryPlannerPolicy policy,
        MemoryPlanSummary summary
) {}
```

Pravidla:

- `TensorMemoryPlan` vlastni tensor lifetimes, reusable intervals, tensor slot assignment.
- `RegionMemoryPlan` vlastni structural view, region value lifetimes, materialization, region bindings a handoffs.
- `RuntimeBindingPlan` vlastni per-node/per-tensor runtime binding policy.
- Empty stav je explicitni empty value objekt, ne sada prazdnych map v hlavnim konstruktoru.

### Partition Planning

`GreedyMaxRegionPartitionPlanner` a `AnchorBasedPartitionPlanner` maji cilove zmizet jako dve kopie stejneho algoritmu.

Cilovy model:

```java
public final class MaxRegionPartitionPlanner implements PartitionPlanner {
    public MaxRegionPartitionPlanner(SeedOrdering seedOrdering) { ... }
}
```

`SeedOrdering` neni abstraktni framework. Je to uzka hodnota nebo enum policy pro rozdil mezi node-order greedy planningem a anchor-first planningem.

`ScoredCandidatePartitionPlanner` a `CpuNaturalExecutionRegionPlanner` zustanou oddelene, protoze maji jinou algoritmickou povahu.

### Trace Ownership

Graph trace model muze zustat centralni, ale sestavovani backend-specific atributu patri backend contributorum.

`StepExecutionTracer` cilove:

- sestavi obecna pole step trace,
- zavola contributor pipeline,
- neprimo pres backend-neutral metadata prida common layout/compute metadata.

Nesmime v nem mit OpenBLAS symbol checks, CPU fused internals, native CPU parity klasifikaci ani route-specific matmul branchovani.

Contributor pipeline nesmi byt obecny plugin/adapter framework. Ma jit o explicitni backend-owned trace value objekty nebo male contributor tridy napojene z prepared artifactu. Pokud backend nema specialni atributy, nepridava nic.

Trace schema musi rozlisit:

- `loweringFamily`: fyzicka family, napr. `CUDA_GRAPH_REGION`,
- `regionPattern` nebo ekvivalent: strukturalni pattern/hint, napr. `FUSED_ELEMENTWISE`,
- `executionKind`: graph executable, provider call, direct kernel, fused kernel,
- `fallbackRoute`: zadna, CPU fallback, provider fallback nebo interpretovana fused cesta.

### Documentation Truth

Po refaktoru musi byt aktualni:

- `src/main/java/graph/README.md`
- `src/main/java/graph/optimizer/README.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/STRUCTURE.md`
- `.planning/codebase/CONCERNS.md`
- relevantni docs pod `docs/`

Dokumentace nesmi odkazovat na legacy `graph.optimizer.partition`, `graph.optimizer.region`, `graph.optimizer.memory`, `graph.optimizer.cse`, pokud ty packages realne neexistuji.

## Implementacni Plan

### 1. Dokumentace a Source Truth Audit

Udelat kratky cleanup docs pred kodem.

Kroky:

1. Aktualizovat stale odkazy v `.planning/codebase/*`, `README.md` a graph readme.
2. Popsat aktualni package ownership podle realneho stromu.
3. Pridat hygiene check, ktery brani navratu legacy optimizer planning packages a backend-specific graph execution imports.

Acceptance:

- `rg "graph.optimizer.(partition|region|memory|cse)" .planning/codebase docs README.md` nevraci stale guidance.
- Source hygiene test jasne chrani hranici `graph.execution` vs konkretni backend internals.

### 2. Odstranit Backend-Specific Znalosti Z Graph Runneru

Kroky:

1. Zmapovat importy v `graph/execution/**` na `backend.cpu`, `backend.blas`, `backend.metal`, `backend.cuda`.
2. Definovat `PreparedExecutionStep` contract jako cilovy runtime model, ne jako docasny wrapper kolem `PreparedNodeExecution`.
3. Presunout CPU input-read policy z `PreparedExecutionRunner` do prepared metadata nebo CPU artifactu.
4. Presunout backend-specific trace atributy ze `StepExecutionTracer` do backend-owned trace value/contributoru.
5. Zachovat publication order a output availability podle compiled program/publication planu.
6. Zmenit graph runner tak, aby jeho primarni vstup byl seznam `PreparedExecutionStep`, ne implicitni iterace pres vsechny compiled nody.
7. Nechat graph runner jen jako lifecycle runner.

Acceptance:

- `PreparedExecutionRunner` neimportuje CPU/native CPU artefakty.
- `StepExecutionTracer` neimportuje OpenBLAS, CPU fused, CPU matmul executable ani backend-specific route enumy.
- CPU fused execution se nespousti pres anchor node metadata a skipovani interior nodu.
- CPU native a Metal/CUDA region execution se nespousti pres graph-runtime `PartitionExecutionRole`, ale pres explicitni multi-node `PreparedExecutionStep`.
- Prepared execution order je odvozeny z prepared steps a zachovava vsechny publication outputs.
- Existing execution trace tests zustanou zelene nebo jsou aktualizovane na stejny evidence contract pres contributory.

### 3. Zuzit ExecutionState Surface

Kroky:

1. Rozdelit `ExecutionState` public metody podle realnych konzumentu.
2. Ponechat state jako per-run root, ale vnitrni odpovednosti vyjadrit domenovymi objekty.
3. Zamezit tomu, aby device binding, native storage a CPU materialization vzajemne obchazely svoje invarianty.
4. Pridat focused tests na residency transitions a resource cleanup.

Acceptance:

- Residency transition API je mensi a pojmenovane podle domeny.
- Backendy nemohou omylem oznacit device/current stav bez validni binding kontroly.
- `ExecutionState` zustane citelny bez velkeho switch/cross-domain driftu.

### 4. Prepsat CompileSession Na Cilovy Workflow

Kroky:

1. Sloucit staticke session stage tridy zpet do `CompileSession` nebo do domenovych workflow objektu.
2. Zachovat jediny compile path pro inference/training s explicitnimi rozdily.
3. Odstranit `*Stage` tridy, ktere pouze preposilaji parametry.
4. Nezavadet `CompilePipeline`, `CompileManager`, ani obecny workflow framework.

Acceptance:

- `CompileSession.compile()` je citelna sekvence pojmenovanych metod.
- Stage/helper tridy bez stabilni domenove odpovednosti jsou odstranene.
- Compile trace a publication plan zustanou behavior-compatible.

### 5. Rozdelit MemoryPlan Na Explicitni Casti

Kroky:

1. Zavest `TensorMemoryPlan`, `RegionMemoryPlan`, `RuntimeBindingPlan`.
2. Upravit `MemoryPlanner.plan(...)`, aby sklada explicitni casti.
3. Upravit `RuntimeMemoryBinder` a lowering call sites na novy model.
4. Odstranit prazdne mapy jako implicitni disabled state.

Acceptance:

- `MemoryPlan` nema dlouhy konstruktor s nesouvisejicimi mapami.
- Tensor-level a region-level memory plan se daji cist samostatne.
- Runtime binding tests pokryvaji F64, F32, BF16, INT32, INT64 a BOOL slot reuse.

### 6. Sloucit Max-Region Partition Planning

Kroky:

1. Porovnat `GreedyMaxRegionPartitionPlanner` a `AnchorBasedPartitionPlanner`.
2. Vytahnout spolecny algoritmus do jednoho konkretniho `MaxRegionPartitionPlanner`.
3. Vyjadrit rozdil pouze pres seed ordering policy.
4. Odstranit puvodni duplicitni tridy, neudrzovat je jako fasady.

Acceptance:

- Neexistuji dve kopie expansion/rejection/trace algoritmu.
- Greedy a anchor-first chovani zustane testovane samostatne.
- Planner API zustane jednoduche: `PartitionPlanner.plan(request)`.

### 7. Odstranit Graph Support/Helper Junk-Drawery

Kroky:

1. `OptimizerGraphSupport` nahradit domenovym `OptimizerGraph`.
2. `RegionOptimizationUnitSupport` rozbit podle odpovednosti: `ExecutionUnit` value construction zustane u unit modelu/planneru, elementwise chain rules u elementwise planneru, matmul epilogue pravidla se presunou do backend-selected planningu nebo zmizi.
3. `BackendTraceSupport` rozdelit mezi backend trace contributory nebo pojmenovane trace value builders.

Acceptance:

- `find src/main/java/graph -name '*Support.java' -o -name '*Helper.java' -o -name '*Adapter.java'` nevraci nove graph junk-drawery.
- Pokud nejaka trida zustane, ma jasne domenove jmeno a duvod existence.
- Zbyla trida nesmi existovat jen proto, aby sdilela staticke metody bez vlastni domenove odpovednosti.

### 8. Redesign Loop Fusion Jako Backend-Specific Execution Planning

Kroky:

1. Aktualizovat `graph.optimizer/FUSE.md` a `graph/optimizer/README.md`, aby netvrdily, ze optimizer vlastni loop fusion.
2. Popsat `ExecutionUnit` contract v compile planning docs: spolecny region-level artefakt, ne prikaz backendu ani povinny runtime rozpad.
3. Popsat `FUSED_ELEMENTWISE` jako execution-unit pattern, ktery CPU aktivne loweruje a GPU muze pouzit jako hint/summary.
4. Rozdelit model na `BackendRegion` a backend-specific execution planning: region je coarse backend/materialization boundary; CPU muze region rozdelit na execution units, Metal/CUDA muze region predat whole-graph loweringu.
5. Presunout pravidla z `RegionOptimizationUnitSupport` do konkretniho execution-unit planneru, napriklad `ElementwiseFusionPlanner` nebo `CpuExecutionUnitPlanner`, bez nove helper vrstvy.
6. Zrusit CPU fusion zavislost na tom, ze partition je presne fused chain. CPU planner ma umet vytvorit vice execution units uvnitr jednoho CPU regionu.
7. Drzet matmul epilogue jako backend/cost-selected fyzicky pattern, ne jako obecnou graph-level unit. CPU planner/lowering musi respektovat, jestli matmul pujde pres OpenBLAS/Accelerate/provider call, Java kernel, MemorySegment/native storage cestu nebo jinou family.
8. Auditovat a odstranit graph-level `MATMUL_EPILOGUE` emisi ze `StructuralRegionUnitPlanner`, pokud neni primo backend-selected.
9. Vyjadrit fused chain pouze pres `ExecutionUnitKind.FUSED_ELEMENTWISE`, `orderedNodeIds`, `inputRefs`, `outputRefs`, `virtualRefs` a `materializedRefs`.
10. Pro Metal/CUDA nevyzadovat `ExecutionUnitKind` rozpad, pokud backend umi lowerovat cely region jako graph/DAG; units mohou zustat jako metadata/hint.
11. Sjednotit CUDA graph family nazvoslovi s Metalem: odstranit `LoweringFamily.CUDA_FUSED_ELEMENTWISE_GRAPH`, `CudaRegionLowerer` ma pro graph/DAG regiony vracet `CUDA_GRAPH_REGION` i kdyz je region ciste `FUSED_ELEMENTWISE`.
12. Presunout elementwise CUDA pattern informaci do metadata/trace: `GpuCompoundRegionSummary`, `GpuRegionLoweredUnitSummary`, `RegionExecutionPlan` payload nebo explicitni trace atribut, ne do `LoweringFamily`.
13. Aktualizovat CUDA tests, trace assertions a docs, ktere ocekavaji `CUDA_FUSED_ELEMENTWISE_GRAPH`, na `CUDA_GRAPH_REGION` plus pattern metadata.
14. Odstranit runtime cestu pres `PartitionRoleIndex`/`ANCHOR`/`INTERIOR`; prepare ma vytvorit `PreparedExecutionStep` pro fused unit, CPU native region i Metal/CUDA region.
15. Overit, ze `graph.optimizer` neimportuje CPU fused/backend fused tridy.
16. Pridat hygiene nebo focused test, ktery brani navratu CPU fused implementation imports do `graph.optimizer`.

Acceptance:

- Loop fusion ownership je dokumentovany jako backend-specific execution planning plus `backend.cpu.fused`.
- `graph.optimizer` zustava graph-rewrite-only.
- CPU fused executable planning zustava v backend prepare/codegen vrstve.
- `ExecutionUnit` je dokumentovana jako navrh/hint fyzicke granularity; backend lowerer muze jednotku prijmout, pregrupovat, rozpadnout nebo ignorovat.
- Partitioning neni pouzity jako CPU fused-loop identity. Jeden CPU backend region muze obsahovat vice execution units.
- Metal/CUDA region planning muze predat cely accepted region do graph loweringu bez umeleho `UNIT_KERNEL`/`FUSED_ELEMENTWISE` rozkladu, ale zachovat units jako trace/summary metadata.
- CUDA lowering family je zarovnana s Metalem: CUDA graph regiony pouzivaji `CUDA_GRAPH_REGION`; neexistuje `CUDA_FUSED_ELEMENTWISE_GRAPH` jako fyzicka lowering family.
- Fused/elementwise CUDA pattern je stale dohledatelny v metadata/trace, aby se neztratila observabilita a testovatelnost.
- `MATMUL_EPILOGUE` nevznika v backend-neutral/structural planneru; pokud existuje, je to backend-selected lowering/prepared-step family.
- Matmul epilogue fusion neni vybrana pred backend/provider rozhodnutim; OpenBLAS/provider path muze zustat samostatny matmul step.
- Runtime nespousti fused loop ani CPU/GPU regiony pres anchor node a skip interior nody.

### 9. Odstranit Graph Runtime Stdout Side Effects

Kroky:

1. Najit vsechny `System.out`/`System.err` v `src/main/java/graph`.
2. Odstranit tisk z `PreparedExecution.backward()`.
3. Zvolit cilovy contract pro unsupported backward convenience call: no-op bez tisku, nebo vyjimka. Preferovat konzistenci s explicitnim `execute(FORWARD_BACKWARD)`.
4. Pridat source hygiene check pro stdout v `graph`, pokud neexistuje jasna vyjimka.

Acceptance:

- `src/main/java/graph` nema user-facing stdout side effects.
- Unsupported backward behavior je testovane a dokumentovane.

### 10. Finalni Package Documentation A Gates

Kroky:

1. Aktualizovat graph a optimizer README.
2. Aktualizovat codebase maps.
3. Pridat nebo upravit source hygiene tests pro nove hranice.
4. Pridat hygiene gate pro aktivni source/docs: zadne `CUDA_FUSED_ELEMENTWISE_GRAPH` mimo historicke planning artefakty.
5. Pridat hygiene gate, ze `graph.optimizer` neimportuje backend fused/lowering/prepare tridy.
6. Pridat hygiene gate, ze `graph.execution` neimportuje konkretni CPU/Metal/CUDA/OpenBLAS execution payloady.
7. Pridat hygiene gate nebo focused test, ze `MATMUL_EPILOGUE` nevznika v backend-neutral/structural graph planningu.
8. Zkontrolovat public API javadocs pro `CompiledGraph`, `CompiledProgram`, `PreparedExecution`, `ExecutionState`.

Acceptance:

- Dokumentace odpovida realnemu stromu.
- Source hygiene brani navratu legacy packages, graph backend import driftu a helper/adaptor junk files.
- Aktivni kod a docs nepouzivaji `CUDA_FUSED_ELEMENTWISE_GRAPH`; CUDA graph lowering je `CUDA_GRAPH_REGION`.
- Graph-level planner negeneruje backend-neutral `MATMUL_EPILOGUE`.
- `./gradlew classes` a focused graph tests projdou.

## Navrhovane Poradi Commitu

1. Docs/source truth cleanup.
2. Prepared execution steps and graph runner backend import cleanup.
3. Step trace contributor ownership cleanup.
4. ExecutionState surface cleanup.
5. CompileSession workflow cleanup.
6. MemoryPlan explicit split.
7. Max-region planner unification.
8. ExecutionUnit contract and backend-specific planning split.
9. CUDA graph family naming cleanup.
10. CPU low-level family lowering and fused prepared steps.
11. Loop fusion ownership docs/gates.
12. Graph stdout side-effect cleanup.
13. Support/helper removal and hygiene gates.

Kazdy commit ma byt topic-sized a bez compatibility vrstvy. Pokud zmena vyzaduje prejmenovani modelu, prejmenovat call sites rovnou.

## Verifikace

Minimalni lokalni sada po kazde vlne:

```bash
./gradlew classes
./gradlew test --tests SourceTreeHygieneTest
./gradlew test --tests graph.execution.*
./gradlew test --tests graph.compile.*
./gradlew test --tests graph.optimizer.*
```

Podle dotcene oblasti pridat:

```bash
./gradlew test --tests graph.compile.planning.*
./gradlew test --tests graph.compile.planning.memory.*
./gradlew test --tests backend.prepare.*
./gradlew test --tests backend.ComputeBackendTest
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest
./gradlew test --tests CompiledGraphTraceTest
```

Source truth spot checks:

```bash
rg "CUDA_FUSED_ELEMENTWISE_GRAPH" src/main/java src/test/java docs README.md
rg "MATMUL_EPILOGUE" src/main/java/graph src/test/java
rg "backend\\.(cpu|metal|cuda|blas)" src/main/java/graph/execution
```

Tyto prikazy maji byt interpretovane podle ciloveho kontraktu: prvni nema vratit aktivni odkazy, druhy nema ukazat backend-neutral graph-level epilogue planning a treti nema ukazat konkretni backend execution payload importy.

Pro accelerator/runtime boundary zmeny podle potreby:

```bash
./gradlew metalTest
```

## Rizika

- Trace cleanup muze zmenit report field ownership. Mit predem testy na obsah trace/report evidence, ne jen na to, ze execution projde.
- MemoryPlan split muze mit velky blast radius v lowering a runtime binding. Delat ho po runner/trace cleanupu.
- CompileSession cleanup nesmi znovu zavest live semantic graph mutation mimo compile-local snapshot.
- Partition planner unification nesmi zhorsit anchor-first training/accelerator coverage. Zachovat focused planner tests a trace assertions.
- Loop fusion cleanup nesmi presunout executable CPU fusion zpet do optimizeru. Optimizer smi maximalne produkovat backend-neutral graph formy.
- Execution-unit planning split nesmi zhorsit fusion coverage tim, ze region hranice zustanou zbytecne uzke. Testovat mixed region s vice fused subchainy.
- `ExecutionUnit` contract nesmi byt preinterpretovany jako prikaz pro GPU lowering. Metal/CUDA musi moct lowerovat cely region a units ponechat jako summary.
- Odstraneni `CUDA_FUSED_ELEMENTWISE_GRAPH` nesmi ztratit observabilitu fused/elementwise CUDA regionu. Pattern musi zustat v payloadu, summary nebo trace.
- Testy a benchmark/report consumers, ktere cte `loweringFamily`, musi byt upravene tak, aby fyzickou family cetly jako `CUDA_GRAPH_REGION` a pattern cetly oddelene.
- CPU family selection nesmi predcasne spojit `MATMUL` epilogue tam, kde OpenBLAS/Accelerate/provider call plus samostatny fused elementwise step vychazi lepe.
- Odstraneni `ANCHOR`/`INTERIOR` CPU fused cesty nesmi zmenit output publication ani trace evidence pro fused execution.
- Odstraneni `PreparedExecution.backward()` stdout nesmi potichu zmenit explicitni `execute(FORWARD_BACKWARD)` error contract.

## Hotovo Kdyz

- `graph` package se da vysvetlit podle lifecycle vrstvy bez vyjimek.
- `graph.execution` nema konkretni CPU/Metal/CUDA/OpenBLAS znalosti mimo backend-neutral contracts.
- Compile workflow je cten jako cilovy compiler flow, ne jako sklad statickych helper stage trid.
- Memory planning model ma explicitni casti a zadne rezimove prazdne mapy v hlavnim kontraktu.
- Max-region partitioning nema duplicitni algoritmus.
- Partitioning je coarse backend/materialization boundary, ne fused-loop identity.
- Loop fusion je jasne vlastnena backend-specific execution planningem a CPU backendem, ne optimizerem.
- `ExecutionUnit` je zachovany jako region-level planning artefakt a hint; CPU ho aktivne loweruje na provider/kernel/fused steps, GPU ho muze ignorovat pri whole-region graph loweringu.
- CUDA a Metal graph regiony maji sjednocene lowering-family nazvoslovi: `METAL_GRAPH_REGION` a `CUDA_GRAPH_REGION`; fused/elementwise zustava pattern metadata, ne lowering family.
- Prepared execution spousti explicitni stepy; CPU fused, CPU native a Metal/CUDA region cesty uz nestoji na anchor/interior skipovani puvodnich nodu.
- `*Support`, `*Helper`, `*Adapter` tridy v `graph` jsou odstranene nebo prejmenovane na skutecne domenove modely.
- `graph` runtime nema stdout/stderr side effects.
- Dokumentace a source hygiene testy chrani novy stav.
