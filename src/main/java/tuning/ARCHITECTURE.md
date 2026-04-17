# Tuning Architecture

Tuning vrstva je postavená na jednom tvrdém pravidle:

- tuning nesmí vytvořit paralelní execution model vedle runtime

Každý benchmarkovaný nebo autotunovaný kandidát musí být reálně spustitelný jako:

- `ExecutionProfile`
- `CompiledGraph`
- `PreparedExecution`

Tuning tedy nebenchmarkuje abstraktní "knob set". Benchmarkuje a vyhodnocuje skutečně spustitelný profil.

## Reading Guide

Tento dokument popisuje:

- jak se skládá finální executable profil
- jak se liší benchmark, autotune a platform calibration
- které vrstvy vlastní runtime knoby a které graph policy
- jaké calibration families dnes existují
- jak orchestrace kombinuje workload, measurement, validation, search a persistence

## Core Artifacts

Je potřeba rozlišovat čtyři artefakty:

### 1. `PlatformRuntimeProfile`

Machine-specific runtime defaults.

Obsahuje runtime rodiny:

- matmul
- fused
- elementwise dispatch
- reduction
- scheduler
- materialization
- numerics

Neobsahuje:

- optimizer stage order
- rewrite policy
- workload-specific graph winners

### 2. `GraphExecutionPolicy`

Graph-level policy.

Dnes je to prakticky wrapper kolem:

- `OptimizerConfig`

Tedy:

- stage order
- rewrite config
- CSE config
- fuse config
- memory config

### 3. `ExecutionProfile`

Runnnable artifact, který se opravdu měří.

Vzniká složením:

- graph policy
- runtime profile
- dtype
- execution mode
- optional workload metadata

### 4. Persistence / Explain Artifacts

Sem patří:

- best profile records
- tuning history
- platform calibration reports
- benchmark/autotune reports

Nejsou to přímé execute contracts.

## Assembly Boundary

Jediné správné místo, kde se skládá finální runnable profil, je:

- [ExecutionProfileAssembler.java](../config/profile/ExecutionProfileAssembler.java)

Assembler bere:

- `PlatformRuntimeProfile`
- `GraphExecutionPolicy`
- dtype
- execution mode

a vrací:

- `ExecutionProfile`

To je zásadní boundary:

- platform calibration mutuje `PlatformRuntimeProfile`
- graph autotune mutuje nebo vybírá `ExecutionProfile`
- runtime vždy nakonec dostane `ExecutionProfile`

## Workflow Split

Tuning balík dnes obsahuje tři oddělené workflow.

### Benchmark

Role:

- porovnat explicitně dané kandidáty
- ukázat trace a hotspoty
- spočítat speedup vůči baseline

Nedělá:

- search
- candidate refinement
- mutaci runtime defaults

Entry:

- [BenchmarkSession.java](./session/BenchmarkSession.java)

### Graph Autotune

Role:

- vybrat nejlepší `ExecutionProfile` pro konkrétní workload
- použít search strategii
- perzistovat best profile a history

Entry:

- [AutotuneSession.java](./session/AutotuneSession.java)

### Platform Calibration

Role:

- naladit platform runtime defaults po family krocích
- uložit výsledný `PlatformRuntimeProfile`

Entry:

- [PlatformCalibrationSession.java](./session/PlatformCalibrationSession.java)

## Session Responsibilities

`session` vrstva je orchestrátor. Kombinuje:

- candidate generation
- validation
- measurement
- search
- progress reporting
- persistence hooks

Neřeší:

- kernel execution detail
- optimizer internals
- workload implementation detail

## Module Split

### `workload`

Definuje:

- `WorkloadSpec`
- `WorkloadInstance`
- workload metadata
- standard and calibration workload catalogs

### `candidate`

Definuje:

- `Candidate`
- `CandidateSpace`
- `RefinableCandidateSpace`
- `ExecutionProfileMutator`

### `measure`

Definuje:

- `MeasurementPolicy`
- `MeasurementEngine`
- `MeasurementResult`

Aktuální default measurement engine:

- compile graph
- prepare execution
- optional traced run
- warmup
- steady-state repeats

### `validate`

Definuje:

- workload correctness checks
- baseline/reference validation

### `search`

Řeší:

- ordering kandidátů
- refinement
- tree search
- history-aware preference/pruning

### `report`

Řeší:

- text a JSON explain artifacts
- suite summaries
- candidate summaries
- calibration/tuning result renderers

### `store`

Řeší:

- platform profile store
- best profile store
- history store
- hardware/workload fingerprinting
- path helpers

## Benchmark Flow

`BenchmarkSession` dnes reálně dělá:

1. pro každý `BenchmarkEntry` instanciuje fresh workload
2. spustí validation
3. pokud validation projde, změří kandidáta
4. vrátí `BenchmarkReport`

Tedy:

- benchmark nepracuje s jedním sdíleným graph instance napříč kandidáty
- každý kandidát dostává fresh workload instance

To je správně, protože compiled/prepared runtime může měnit graph strukturu i cache stav.

## Autotune Flow

`DefaultAutotuneSession` dnes dělá:

1. vytvoří `SearchContext`
2. nechá search strategii vybrat počáteční batch
3. kandidáty validuje a měří
4. pokud strategie podporuje refinement, iteruje dál
5. seřadí úspěšné kandidáty podle steady-state median
6. vybere finalisty
7. perzistuje history a best profile

Důležité:

- search vybírá kandidáty
- session je teprve měří
- "best" znamená dnes nejnižší steady-state median

## Platform Calibration Flow

`DefaultPlatformCalibrationSession` jde family po family:

1. vezme seed `PlatformRuntimeProfile`
2. vytvoří candidate space pro první family krok
3. z kandidátů složí runnable `ExecutionProfile`
4. spustí benchmark suite pro zadané calibration workloads
5. score policy vybere vítěze
6. vítězný runtime profile se stane seedem další family
7. po posledním kroku uloží finální `PlatformRuntimeProfile`

To je zásadní rozdíl oproti graph autotune:

- calibration nehledá workload-specific winner
- calibration hledá reusable platform defaults

## Current Calibration Families

Aktuální enum je:

- `MATMUL`
- `ATTENTION_MATMUL`
- `FUSED_THRESHOLDS`
- `FUSED_CHEAP_CONTIGUOUS`
- `FUSED_CHEAP_STRIDED`
- `FUSED_NON_CHEAP_CONTIGUOUS`
- `FUSED_NON_CHEAP_STRIDED`
- `FUSED_ARITHMETIC`
- `ELEMENTWISE_DISPATCH`
- `REDUCTION`
- `ATTENTION_THRESHOLDS`
- `SCHEDULER`
- `MATERIALIZATION`
- `CONV2D`
- `NUMERICS`

Ale ne všechny rodiny jsou dnes používané ve standardních presets.

### Standard Training/Inference Presets Today

`PlatformCalibrationDefaults.standardTrainingSteps(...)` a `standardInferenceSteps(...)` dnes skládají hlavně:

- `MATMUL`
- fused threshold + ASM width families
- `ELEMENTWISE_DISPATCH`
- volitelně `REDUCTION`
- volitelně `ATTENTION_THRESHOLDS`
- volitelně `ATTENTION_MATMUL`
- volitelně `SCHEDULER`
- volitelně `MATERIALIZATION`
- volitelně `NUMERICS`

Momentální důležitá realita:

- `FUSED_ARITHMETIC` v enumu dnes standardní presets nepoužívají
- `CONV2D` family zatím není součást standardních preset kroků

Dokumentace musí tohle říkat explicitně, jinak budí dojem, že se kalibruje víc, než se ve skutečnosti kalibruje.

## Family Ownership

Každý runtime knob musí mít jasného vlastníka.

### `MATMUL`

Sem dnes patří například:

- BLAS min work
- BLAS threads
- `f32RequireMgeK`
- `f32MaxNOverK`
- `cpu.matMulParallelMinSize`
- microkernel volba
- tile volba
- attention matmul tile/microkernel volba

### `FUSED_THRESHOLDS`

Sem patří:

- `cpu.fusedCheapVectorMinSize`
- `cpu.fusedTranscendentalVectorMinSize`
- `cpu.fusedCheapParallelMinSize`
- `cpu.fusedTranscendentalParallelMinSize`

### Fused ASM Width Families

Sem patří width knoby pro konkrétní dispatch families:

- cheap contiguous
- cheap strided
- non-cheap contiguous
- non-cheap strided

To už dnes nejsou jen "interní experimentální proměnné". Jsou součástí platform calibration surface.

### `ELEMENTWISE_DISPATCH`

Sem patří non-fused elementwise thresholds:

- cheap vector
- transcendental vector
- cheap parallel
- transcendental parallel

### `REDUCTION`

Sem patří:

- reduction vector threshold
- reduction parallel threshold
- attention vector threshold
- attention parallel threshold
- `sumAccuracyMode`

### `SCHEDULER`

Sem patří:

- target chunks per worker
- minimum chunk sizes
- common pool threshold

### `MATERIALIZATION`

Sem patří:

- `cpu.contiguousMaterializeThreshold`

### `NUMERICS`

Sem patří:

- `approxMode`
- `forceExactTranscendentals`

### `GRAPH_POLICY`

Sem patří:

- optimizer stage order
- rewrite configs
- conv2d lowering mode

Tohle ale není `PlatformRuntimeProfile`. To je `GraphExecutionPolicy`.

## Search And Calibration Are Different

Tohle je jeden z nejčastějších zdrojů zmatení:

- calibration candidate space generuje `PlatformRuntimeProfile` mutace
- autotune candidate space generuje `ExecutionProfile` varianty

První je reuse across workloads.
Druhý je workload-specific search.

## Score Policy

Platform calibration používá explicitní score policy per step.

Dnes jsou běžné:

- `averageMedianMs()`
- `weightedGeometricMeanWithWorstBucketPenalty(alpha)`

To je důležité třeba pro attention families, kde jedna workload bucket nemá úplně dominovat a přesto se penalizuje slabý worst case.

## Tracing Boundary

Trace data generuje execution layer.

Tuning je jen konzumuje přes:

- compile trace
- prepare trace
- run trace
- step trace metadata

To znamená:

- tuning report může říct, že kandidát běžel s `vectorWidth=4`
- ale tuning to nepočítá sám, jen čte trace z runtime

## Persistence Boundary

Persistence se také dělí podle workflow:

- platform calibration ukládá `PlatformRuntimeProfile`
- autotune ukládá best `ExecutionProfile`
- history ukládá candidate-level evidence
- reporty ukládají explain artifacts

Více v:

- [PERSISTENCE.md](./PERSISTENCE.md)

## Example: Assemble Executable From Platform Defaults

```java
ExecutionProfile profile = ExecutionProfileAssembler.assemble(
        "abc-f64",
        "abc-f64",
        DataType.FLOAT64,
        ExecutionMode.FORWARD_BACKWARD,
        platformRuntimeProfile,
        GraphExecutionPolicy.trainingDefaults()
);
```

To je finální artifact, který jde do benchmarku nebo běhu aplikace.

## Example: Platform Calibration Vs Autotune

Platform calibration:

- vezme training/inference seed
- mutuje runtime defaults
- vrátí `PlatformRuntimeProfile`

Autotune:

- vezme workload
- vezme seed profile
- searchuje candidate `ExecutionProfile` varianty
- vrátí best executable profile pro daný workload

## Common Mistakes

- míchat runtime knoby a optimizer policy do jednoho profilu bez jasné ownership
- považovat platform calibration winner za graph-specific best profile
- benchmarkovat kandidáty, které nejsou skutečně spustitelné `ExecutionProfile`
- ukládat explain artifacts jako execute source of truth

## Related Docs

- overview: [README.md](./README.md)
- workloads: [WORKLOADS.md](./WORKLOADS.md)
- knobs: [KNOBS.md](./KNOBS.md)
- persistence: [PERSISTENCE.md](./PERSISTENCE.md)
- search: [SEARCH.md](./SEARCH.md)
- reporting: [REPORTING.md](./REPORTING.md)
