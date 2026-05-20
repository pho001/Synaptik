# Graph Publication Contract And Compiled Program Cleanup

## Stav Rozpracovani

Status: `IMPLEMENTED`

Implementation note:

- `GraphStructureContract` now guards prepared execution against topology, layout, dtype, backend intent, grad, and trainable-parameter drift while still allowing input value changes.
- Publication moved to `PublicationPlan`; `CompiledNode` no longer owns publication tensors.
- Executable compile data moved to `CompiledProgram`; diagnostics remain outside the program.
- Runtime per-call orchestration moved behind `ExecutionRun`.
- Backend partition planning uses `BackendPartitionCapability`; `PartitionPlanningRequest` no longer owns `RegionLegalityAdapter` or Metal-specific transfer fields.
- `CompileSession` is now a thin stage orchestrator.
- `CompiledGraph` is narrowed to compile/prepare plus read-only `program()` and `publication()` inspection; one-shot execute/traced/executePrepared conveniences and root tensor exposure are removed.
- Generic compile/profile cost configuration uses backend-neutral `TransferCostPreset`; old profile JSON key `partitionMetalTransferModel` is read only as a migration fallback.

Navazuje na:

- `todo/28-graph-execution-isolation-and-immutability.md`
- `todo/38-graph-architecture-cleanup-and-reorganization.md`
- `todo/88-graph-compile-partition-lowering-model-cleanup.md`
- `todo/89-graph-runtime-boundary-cleanup.md`
- `todo/91-live-tensor-boundary-cleanup.md`

## Problem

`graph` uz ma spravny smer: public `Tensor` zustava logicky, compile vytvari snapshot, prepare vytvari runtime metadata a kazdy run ma vlastni `ExecutionState`.

Slaba mista jsou dnes hlavne v hranici mezi immutable compile programem a user-visible publikaci:

- `GraphStructureContract` nehlida layout-relevant metadata, ktera prepared/backend plan realne pouziva.
- `CompiledNode` porad nese publication tensor vazbu, takze executable node snapshot neni ciste oddeleny od mutable public tensor sveta.
- `CompileArtifacts` micha executable program, publication vazby, mutable live `Tensor` reference a compile-planning vystupy.
- `CompiledGraph` je prilis siroka fasada: umi recompile, prepare, execute, traced execute, optimizer step, zeroGrad a vraci mutable optimized tensor graph.
- `CompileSession` zustava dlouhy mutable tok, ktery v jedne tride dela forward capture, backward graph, optimizer snapshot, node snapshot, backend planning, region optimization, memory planning a publication mapping.
- Partition planning pouziva `RegionLegalityAdapter` a backend-specific `MetalTransferModel` v obecnem graph requestu.

Cilem tohoto todo neni kosmeticky rename. Cilem je udelat compile/runtime hranice cistejsi, citelnejsi a lepe rozsiritelne bez helperu, adapteru, transition vrstev a compatibility fasad.

## Architektonicka Pravidla

- Public `Tensor` zustava logicky. Backend residency patri do compile/prepare/execute runtime state.
- `CompiledNode` je executable node snapshot, ne publication binding.
- Publication je samostatny domenovy plan s explicitnimi node id vazbami na user-visible tensors.
- Compile artifact nesmi vystavovat mutable optimized `Tensor` graph jako beznou public hranici.
- Zadne `V2`, alias tridy, adaptery, wrappery okolo stareho API ani dlouha paralelni kompatibilita.
- Zadne obecne `Helper`, `Support`, `Adapter`, `Bridge`, `Manager` junk-drawery.
- Nova abstrakce smi vzniknout jen pokud ma stabilni domenovou odpovednost a odstrani realnou neurcitost.
- Refaktor jde rovnou na cilovy navrh, ale po malych verifikovatelnych krocich.

## Cilovy Stav

### Compile Program

Executable compile vystup je `CompiledProgram`.

Vlastni:

- `List<CompiledNode>` executable node snapshots,
- `CompiledTensorDescriptorIndex`,
- forward output node id,
- forward boundary node id,
- backward support flag,
- planned partitions,
- optimized regions,
- memory plan.

Nevlastni:

- user-visible root tensor,
- publication tensors,
- gradient publication targets,
- mutable optimized `Tensor` graph,
- compile diagnostics nebo partition planning trace.

### Publication Plan

Publication je `PublicationPlan`.

Vlastni:

- user-visible root tensor,
- graph structure contract,
- runtime input / leaf source bindings,
- root output publication binding,
- optional all-forward-value publication bindings,
- alias publication repair plan,
- gradient publication bindings,
- forward seed gradient binding,
- trainable parameter bindings.

Publication binding pouziva node id, ne hledani pres `CompiledNode.publicationTensor()`.

`PublicationPlan` je jedina zamerna hranice, ktera smi drzet user-visible `Tensor` reference pro runtime seeding a publication. `CompiledProgram` a `CompiledNode` je drzet nesmi.

Cilovy binding model nema byt mapa jako domenovy model. Pouzit explicitni recordy:

```java
public record RuntimeInputBinding(
        int nodeId,
        Tensor sourceTensor,
        RuntimeInputBindingKind kind
) {}

public enum RuntimeInputBindingKind {
    FORWARD_LEAF_ALIAS,
    BACKWARD_LEAF_COPY,
    STATIC_LEAF_COPY
}

public record ForwardPublicationBinding(
        Tensor targetTensor,
        int sourceNodeId,
        PublicationKind kind,
        List<AliasRepairStep> aliasRepairChain
) {}

public enum PublicationKind {
    ROOT_OUTPUT,
    FORWARD_VALUE,
    ACTUAL_FORWARD_ROOT_FOR_ALIAS
}

public record AliasRepairStep(
        Tensor aliasTensor,
        Tensor sourceTensor
) {}

public record GradientPublicationBinding(
        Tensor targetTensor,
        CompiledGradientBinding binding
) {}

public record TrainableParameterBinding(
        Tensor parameterTensor,
        int parameterNodeId,
        CompiledGradientBinding gradientBinding
) {}
```

Presna enum hodnota muze byt jednodussi podle implementace, ale pole musi byt pojmenovana podle domeny. Nepouzivat `Map<Tensor, Integer>` jako final model publication vazeb.

`TrainableParameterBinding` je povinna soucast training publication modelu. Nestaci `List<Tensor> trainableParameters`, protoze optimizer runtime potrebuje user-visible parameter tensor pro explicitni parameter filtering, parameter node id pro residency/runtime storage a gradient binding pro gradient node nebo constant gradient.

Constant gradient binding nesmi drzet mutable `Tensor` template jako immutable compile/publication boundary. Cilovy model:

```java
public record ConstantGradientValue(
        DataType dataType,
        int[] shape,
        int[] strides,
        int storageOffset,
        CompiledTensorDataSnapshot data
) {}
```

Presny storage snapshot muze byt jednodussi podle dostupnych snapshot trid, ale accessor musi vracet value/defensive copy, ne mutable `Tensor` sdileny z compile artifactu. `CompiledGradientBinding.ConstantBinding` ma cilove drzet `ConstantGradientValue`, ne `Tensor template`.

### Compile Result

`CompileArtifacts` bud zmizi, nebo se zmeni na tenky immutable aggregate:

```java
public record CompileArtifacts(
        CompiledProgram program,
        PublicationPlan publication,
        CompileDiagnostics diagnostics
) {}
```

`CompileDiagnostics` nebo `CompileTrace` vlastni partition planning trace, optimizer trace a dalsi diagnostiku. `CompiledProgram` ji nevlastni, protoze trace neni executable program.

Pokud `CompileTrace` zustane mimo artifacts kvuli stavajicimu API, `CompileArtifacts` stale nesmi michat programova data, publication data a diagnostiku ve stejnem recordu bez jasnych pojmenovanych poli.

### CompiledGraph

`CompiledGraph` je public immutable compile result facade.

Smi:

- drzet `CompiledProgram`,
- drzet `PublicationPlan`,
- drzet `CompileTrace`,
- pripravit `PreparedExecution` pro `RuntimeConfig`,
- vystavit uzke read-only inspect metody.

Nesmi:

- mit public `compile()` mutujici existujici instanci,
- vracet mutable optimized `Tensor` graph jako public model,
- vlastnit `zeroGrad`,
- vlastnit optimizer-step orchestration,
- duplikovat execute overloady, ktere jen vytvari temporary prepared execution.

### Prepared Execution

`PreparedExecution` dostane explicitne:

- `CompiledProgram`,
- `PublicationPlan`,
- `RuntimeConfig`,
- prepared step plan,
- prepare trace.

Runtime run orchestrace zustane prima a bez frameworku. Pokud se `PreparedExecution` zmensuje, cilovy domenovy objekt je `ExecutionRun`, ne obecny helper. `ExecutionRun` vlastni jeden run: state, context, trace, publication a cleanup.

### Backend Partition Capability

`RegionLegalityAdapter` se zrusi.

Backend partition descriptor vystavi primo capability kontrakt, napr.:

```java
public interface BackendPartitionCapability {
    PartitionTarget target();
    boolean canExecute(CompiledNode node, PartitionPlanningContext context);
    boolean canSeed(CompiledNode node, PartitionPlanningContext context);
    boolean canUseExternalInput(CompiledNode producer, CompiledNode consumer, Set<Integer> selectedNodeIds, PartitionPlanningContext context);
    PartitionCandidate createCandidate(Set<Integer> selectedNodeIds, PartitionPlanningContext context, Set<GraphValueRef> requiredMaterializedValueRefs);
    PartitionPlan createPlan(PartitionCandidate candidate, PartitionPlanningContext context);
}
```

Presny nazev muze byt kratsi, ale nesmi to byt adapter. Je to backend capability, tedy cilovy domenovy kontrakt.

Backend-specific cost fakta, vcetne Metal transfer cost modelu, patri do backend/cost capability nebo do backend target planning policy, ne do obecneho `PartitionPlanningRequest`.

## Plan

### 1. Failing test pro stale layout contract

Pridat regression test, ktery dnes spadne.

Scenar:

1. Vytvorit graph, kde prepared/backend rozhodnuti zavisi na layout metadata.
2. Zavolat `compile(...)` a `prepare(...)`.
3. Po prepare zmenit layout-relevant metadata user-visible tensor graphu.
4. Zavolat `execute(...)`.
5. Ocekavat `IllegalStateException` se stale contract zpravou.

Test ma byt co nejmensi a deterministicky. Nemusi vyzadovat Metal/CUDA.

Kandidati:

- view op: reshape/permute/slice/expand alias path,
- CPU path s non-contiguous view,
- graph output alias publication.

Acceptance:

- Test pred opravou selze, protoze `GraphStructureContract` layout metadata nekontroluje.
- Test po oprave projde.
- Test neporovnava storage contents; hodnoty vstupu se smi menit mezi prepare a execute.
- Test nesmi pouzit `getCompiledGraphAsList()` jako legitimni protected boundary. Pokud potrebuje diagnosticky pristup k optimized graphu, musi byt test pojmenovany jako diagnosticky a nesmi simulovat supported mutation path.

### 2. Rozsirit GraphStructureContract

`GraphStructureContract` musi kontrolovat vsechna metadata, na kterych prepared program zavisi:

- tensor identity v topologickem poradi,
- operation descriptor identity a op type,
- parent topology,
- shape,
- strides,
- storage offset,
- dtype,
- backend intent,
- contiguous flag,
- has storage offset flag,
- requiresGrad,
- trainableParameter.

Storage contents zustanou mimo contract.

Poznamka k operation descriptors:

- Pokud operation descriptors zustavaji immutable, staci identity + op type.
- Pokud existuje mutable descriptor nebo descriptor s mutable option objektem, pridat semantic fingerprint pozdeji jako samostatny presny krok. Nedavat obecny reflection fingerprint.

Acceptance:

- Prepared execution odmita stale layout, grad a trainable-parameter contract drift.
- Prepared execution stale dovoluje nove input hodnoty pri stejne graph structure.
- Focused tests pro stale shape/dtype/topology/layout prochazeji.
- Pred oznacenim contract opravy za hotovou musi byt exposed optimized graph dira zavrena: `getCompiledGraphAsList()` se odstrani, nebo se presune za explicitni diagnostic-only API, ktere neni povazovane za runtime/public mutation boundary.

### 3. Zavest PublicationPlan jako autoritativni publication model

Vytvorit `graph.execution.publication.PublicationPlan` nebo `graph.compile.publication.PublicationPlan`; final umisteni zvolit podle toho, zda plan vznikne v compile package a pouziva ho execution package.

Plan ma obsahovat explicitni node-id based vazby:

```java
public record PublicationPlan(
        Tensor rootTensor,
        GraphStructureContract graphContract,
        List<RuntimeInputBinding> runtimeInputBindings,
        ForwardPublicationBinding rootOutput,
        List<ForwardPublicationBinding> forwardValuePublications,
        List<GradientPublicationBinding> gradientPublications,
        CompiledGradientBinding forwardSeedGradient,
        List<TrainableParameterBinding> trainableParameters
) {}
```

Presna pole doladit podle call sites. Dulezite je, aby runtime input seeding ani runtime publication nemusely prohledavat `List<CompiledNode>` podle live tensor reference.

`runtimeInputBindings` jsou P1 soucast modelu. Dnes `RuntimeTensorStore` seeduje leaf runtime tensors z `node.publicationTensor()`. Po odstraneni `CompiledNode.publicationTensor()` musi `RuntimeTensorStore` dostat autoritativni `nodeId -> source Tensor` vazbu z `PublicationPlan`.

`ForwardPublicationBinding.aliasRepairChain` je P1 soucast modelu. Dnes `ExecutionPublisher` resi alias publication pres `resolvePublicationTarget(...)`, `repairPublicationAliasChain(...)` a fallback na actual forward runtime root. Po odstraneni node-side publication tensoru musi byt tato alias oprava predpocitana v publication planu, ne rekonstruovana skenem pres compiled nodes.

`trainableParameters` jsou P1 soucast modelu. Dnes `OptimizerStepContext.trainableParameters()` sklada `TrainableParameterRef` ze vsech compiled nodes a hleda gradient binding pres `node.publicationTensor()`. Po odstraneni node-side publication tensoru musi optimizer dostat explicitni `TrainableParameterBinding(parameterTensor, parameterNodeId, gradientBinding)` z `PublicationPlan`.

Constant gradient hodnoty musi byt snapshotovane jako value. Dnes constant binding vraci mutable tensor template; to je stejny typ boundary leaku jako live tensor v compile artifactu, jen mensi. Publication refaktor ho musi zavrit.

Prepnout execution/publication na `PublicationPlan`:

- runtime leaf/input seeding,
- root sync,
- all-forward publication,
- alias-chain repair po publication,
- gradient publication,
- clear gradients,
- trainable parameter list,
- optimizer parameter selection,
- compatibility check.

Pravidla:

- Nevytvaret prechodovy wrapper, ktery jen deleguje na `CompiledNode.publicationTensor()`.
- `PublicationPlan` je cilovy model a musi se pouzivat jako source of truth.
- Pokud je nutny jeden mezikrok kvuli velikosti diffu, musi byt v jednom topic branch/commitu a hned nasledovan odstranenim stare vazby.

Acceptance:

- `RuntimeTensorStore` seeduje leaf/runtime input tensors z `PublicationPlan.runtimeInputBindings`, ne z `CompiledNode.publicationTensor()`.
- `ExecutionPublisher` publikuje podle `PublicationPlan`.
- Alias publication repair je reprezentovany explicitnim bindingem nebo alias publication planem; neni ztracen pri odstraneni `CompiledNode.publicationTensor()`.
- `PreparedExecution.requireCompatibleGraph(...)` kontroluje contract z `PublicationPlan`.
- `OptimizerStepContext` a `AbstractTrainableOptimizer.selectedParameters(...)` pouzivaji `PublicationPlan.trainableParameterBindings`, ne stream pres nodes ani `CompiledNode.publicationTensor()`.
- `CompiledGraph.trainableParameters()` je odvozene z `TrainableParameterBinding.parameterTensor()`, ne stream pres nodes.
- `CompiledGradientBinding.ConstantBinding` nedrzi mutable `Tensor` template, nebo vraci defensive copy a ma nasledny cilovy krok na value snapshot.

### 4. Odstranit publication tensor z CompiledNode

Po kroku 3 odstranit z `CompiledNode`:

- `publicationTensor` field,
- `publicationTensor()` accessor,
- publication remapping z `CompiledNode.snapshot(...)`.

`CompiledNode.snapshot(...)` dostane jen ordered graph a vytvori executable node snapshots.

Publication mapping se sklada mimo `CompiledNode`, idealne v compile session fazi, ktera zna vztah mezi optimized graph tensors a user-visible tensors.

Acceptance:

- `rg "publicationTensor\\(" src/main/java` nenajde produkcni call sites.
- `CompiledNode` dokumentace netvrdi, ze drzi user-visible publication tensor reference.
- Node snapshot je citelne executable metadata, ne mix executable metadata a publication vazeb.

### 5. Rozdelit CompileArtifacts na CompiledProgram + PublicationPlan

Zavest `CompiledProgram`.

Presunout programova pole:

- final executable nodes,
- descriptor index,
- memory plan,
- optimized regions,
- planned partitions,
- supportsBackward,
- forward boundary,
- forward output node id.

Presunout publication pole do `PublicationPlan`:

- root tensor,
- graph contract,
- runtime input / leaf source bindings,
- forward publication bindings,
- alias repair bindings,
- gradient publication bindings,
- forward seed gradient,
- trainable parameter bindings,
- constant gradient value snapshots.

Presunout diagnostiku mimo `CompiledProgram`:

- partition planning trace,
- optimizer trace,
- compile timing,
- rejected planning candidates,
- backend planning diagnostics.

Diagnostika patri do `CompileTrace` nebo `CompileDiagnostics`. `CompiledProgram` zustava executable model.

Zrusit public vystaveni mutable optimized `Tensor` graphu jako standardni compile artifact. Pokud nejaky test potrebuje inspect optimizer graph, pridat explicitni diagnostic-only API s jasnym nazvem a bez pouziti v prepare/runtime.

Acceptance:

- `backend.prepare.PreparedExecutionBuilder` prijima `CompiledProgram` + `PublicationPlan`, nebo jeden aggregate, ktery je pouze nese.
- `CompileArtifacts` uz nema `List<Tensor> finalGraph`.
- `CompileArtifacts` uz nema paralelni mix executable programu a publication vazeb.
- `CompiledProgram` nema partition planning trace ani jinou diagnostiku.
- Constant gradient publication nepouziva sdileny mutable `Tensor` template.
- Compile diagnostics zustanou dostupne pres `CompiledGraph.compileTrace()` nebo pojmenovany diagnostics accessor.
- Existing prepare/execution tests zustanou zelene.

### 6. Zuzit CompiledGraph

Po predchozich krocich zmensit `CompiledGraph` na immutable compile facade.

Odstranit nebo presunout:

- public `compile()` mutating method,
- `getCompiledGraphAsList()`,
- `zeroGrad()`,
- optimizer-step convenience orchestration,
- execute overloady, ktere jen delaji prepare + execute a zvetsuji public surface.

Cilove API:

```java
public final class CompiledGraph {
    public static CompiledGraph compile(Tensor rootTensor, CompileConfig config, CompileMode mode);
    public PreparedExecution prepare(RuntimeConfig runtimeConfig);
    public CompileTrace compileTrace();
    public boolean supportsBackward();
    public List<Tensor> trainableParameters();
}
```

Pokud je kvuli ergonomii nutne ponechat convenience execute metody, musi byt minimalni a nesmi obsahovat vlastni training/optimizer logiku. Preferovane je, aby optimizer step zustal na training optimizer API nebo `PreparedExecution`, ne na compile result.

Acceptance:

- `CompiledGraph` nema mutable `artifacts` lifecycle.
- Recompile znamena vytvorit novy `CompiledGraph`, ne mutovat stary.
- `CompiledGraph` neobsahuje dtype-specific gradient zeroing switch.

### 7. Zmensit PreparedExecution pres ExecutionRun

`PreparedExecution` je prepared artifact. Jeden execution call muze pouzit `ExecutionRun`.

`ExecutionRun` vlastni:

- execution mode,
- publication policy,
- optional training optimizer,
- execution state,
- execution context,
- run trace accumulation,
- resource cleanup.

Nejde o helper. Je to domenovy objekt pro jeden run.

Pravidla:

- Zadne obecne lifecycle frameworky.
- Zadne callback manager objekty.
- Zadne adaptery pro stare `PreparedExecution` konstruktory.
- `ExecutionRun` nesmi prezit jeden execute call.

Acceptance:

- `PreparedExecution.executeInternal(...)` se zmensi na validaci + vytvoreni `ExecutionRun`.
- Resource cleanup a suppressed close failure zustanou zachovane.
- `RunTrace` zustane stejne nebo citelneji slozeny.

### 8. Rozbit CompileSession na prime domenove faze

Az bude program/publication model cisty, rozbit dlouhy mutable compile tok.

Cilove faze:

1. `ForwardGraphCapture` - vybere semantic forward output a optional canonicalized forward graph.
2. `BackwardGraphCompiler` - rozhodne training support, sestavi backward graph a gradient targets.
3. `OptimizerSnapshot` - vytvori detached optimizer graph snapshot a original mapping.
4. `CompiledProgramSnapshot` - vytvori compiled nodes, descriptors a forward boundary.
5. `PublicationPlanCompiler` - sestavi publication plan z optimized graph mappingu.
6. `BackendOwnershipPlanner` - spusti backend planning a vrati planned partitions.
7. `RegionPlanCompiler` - sestavi optimized regions.
8. `MemoryPlanCompiler` - sestavi memory plan.

Kazda faze musi mit konkretni domenu. Pokud by vznikla trida typu `CompileHelper`, plan je spatne.

Acceptance:

- `CompileSession` bud zmizi, nebo zustane jen kratky orchestrator bez dlouheho seznamu mutable fields.
- Shared state mezi fazemi je explicitni record, ne mnoho mutovanych private fields.
- Compile trace se sklada z vysledku fazi, ne z vedlejsich efektu session objektu.

### 9. Nahradit RegionLegalityAdapter backend partition capability kontraktem

Odstranit `RegionLegalityAdapter`.

Backend partition descriptor ma primo vratit capability objekt pro target.

Zmenit naming a API tak, aby bylo jasne:

- planner vlastni search strategii,
- backend capability vlastni backend fakta,
- backend capability umi vytvorit structural candidate a backend plan,
- graph planning request nenese adapter ani backend-specific transfer model.

`PartitionPlanningRequest` ma cilove obsahovat:

- strategy,
- target,
- planning context,
- search/scoring policy,
- backend capability,
- source policy,
- required materialized value refs,
- target cost policy.

`target cost policy` musi byt backend-neutral interface/value. Metal-specific transfer facts nesmi byt top-level pole obecneho requestu.

Acceptance:

- `rg "Adapter" src/main/java/graph/compile/planning` nenajde `RegionLegalityAdapter`.
- `PartitionPlanningRequest` nema `MetalTransferModel`.
- Backend-specific cost data nevytvari compile-layer import na Metal-only config.

## Implementacni Poradi

1. Failing stale-layout test.
2. Rozsireni `GraphStructureContract`.
3. `PublicationPlan` jako source of truth pro publication.
4. Odstraneni `CompiledNode.publicationTensor()`.
5. `CompiledProgram` + `PublicationPlan` split.
6. Zuzit `CompiledGraph`.
7. `ExecutionRun` pro jeden runtime call.
8. Compile pipeline faze misto dlouhe `CompileSession`.
9. Backend partition capability misto `RegionLegalityAdapter`.

Toto poradi je zamerne. Nejdrive se opravi konkretni correctness gap, potom se vycisti publication boundary, potom compile artifact model, a az nakonec nejvetsi compile/planning reorganizace.

## Test Plan

Po kazdem kroku:

```bash
./gradlew classes
```

Focused testy podle oblasti:

```bash
./gradlew test --tests PreparedExecutionBuildTest
./gradlew test --tests CompiledGraphTraceTest
./gradlew test --tests graph.execution.*
./gradlew test --tests graph.compile.*
./gradlew test --tests backend.prepare.*
```

Po partition capability kroku:

```bash
./gradlew test --tests SourceTreeHygieneTest
./gradlew test --tests '*Partition*Test'
./gradlew test --tests '*Lowering*Test'
```

Metal/CUDA focused tests spoustet jen pro dotcene backend routes:

```bash
./gradlew test --tests backend.metal.*
./gradlew test --tests backend.cuda.*
./gradlew metalTest
```

## Current Closure Status

- [x] Stale prepared execution failne pri layout/grad/trainable contract driftu.
- [x] Prepared execution stale dovoluje menit hodnoty vstupnich tensoru bez recompile.
- [x] `CompiledNode` nedrzi publication tensor.
- [x] Runtime input/leaf seeding ma explicitni `PublicationPlan` bindingy.
- [x] Alias output publication a alias-chain repair jsou explicitne reprezentovane v `PublicationPlan`.
- [x] Optimizer step pouziva explicitni `TrainableParameterBinding` s tensor, parameter node id a gradient binding.
- [x] Publication a gradient publication bezi pres `PublicationPlan`.
- [x] Constant gradient binding je value snapshot nebo ma defensive-copy boundary, ne sdileny mutable `Tensor` template.
- [x] Compile executable data jsou v `CompiledProgram`.
- [x] Partition planning trace a dalsi diagnostika nejsou soucasti `CompiledProgram`.
- [x] Mutable optimized graph neni vystaveny jako supported public/runtime boundary.
- [x] Public compile facade je immutable.
- [~] Public compile facade je uzka: mutujici/training convenience je pryc; siroce pouzivane execute/trace/inspection convenience metody zustavaji pragmaticky zachovane.
- [x] `CompileSession` neni dlouhy mutable god object.
- [x] Partition planning nepouziva adapter naming ani Metal-specific pole v obecnem requestu.
- [x] Architektonicke guard testy chrani nove hranice.

## Explicitne Nedelat

- Nedelat `CompiledGraphV2`, `PublicationPlanAdapter`, `LegacyCompileArtifacts`, deprecated aliasy ani compatibility overloady.
- Nedelat obecny helper pro graph publication.
- Nedelat obecny helper pro compile session.
- Nedelat base planner framework.
- Nedelat public device tensor API.
- Nekontrolovat storage contents v graph structure contractu.
- Nemichat tento cleanup s memory planner split z `todo/90`.
- Nemichat tento cleanup s tensor package cleanup z `todo/93`.
