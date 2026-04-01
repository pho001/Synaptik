# Tuning Package Rewrite

## Cil

Zahodit dnesni benchmark-first strukturu a nahradit ji novou nadrazenou vrstvou:

- `tuning/`

která bude mit jasne oddelene tri odpovednosti:

- `benchmark/` = performance mereni
- `numerics/` = numericka validace
- `autotune/` = hledani a persistence optimalnich profilu

Tato vrstva se ma prizpusobit:

- `Tensor`
- `CompiledGraph`
- backendum
- compileru

nikoliv naopak.

## Proc prepis

Soucasny `benchmark` balik je historicky smes:

- benchmark scenaru
- candidate generatoru
- optimizer builderu
- autotune workflow
- persistence
- casti numerics post-check

To vede k problemum:

- tuning knobs ziji v benchmark-centric modelu
- execution API a profiling jsou tlaceny do formy, kterou chce benchmark
- per-graph autotune se navrhuje obtizne
- numerics je konceptualne bokem, ale logicky patri do stejné vyssi evaluate/tuning vrstvy

## Cilova struktura

Navrzena struktura:

```text
src/main/java/tuning/
  benchmark/
  numerics/
  autotune/
  profile/
```

### `tuning/benchmark`

Zodpovednost:

- performance mereni
- priprava benchmark scenaru
- measurement policies
- scoring

Sem patri:

- `MeasurementExecutor`
- `MeasurementPolicy`
- `TieredMeasurementPolicy`
- benchmark scenario sources
- benchmark scenario recipes

Sem nepatri:

- perzistence defaultnich execution profilu
- optimizer profile loading
- numerics verdict logika

### `tuning/numerics`

Zodpovednost:

- A/B porovnani vystupu
- ULP / abs / rel metriky
- safety/borderline/unsafe verdict
- numerics reporty

Sem patri aktualni `numerics` modul po presunu:

- `NumericsHarness`
- `NumericsMetrics`
- `NumericsPolicy`
- `NumericsReport`
- `NumericsCli`

Je spravne, aby numerics zustaly oddelene od performance benchmarku, protoze:

- nemeri vykon
- meri korektnost a stabilitu

### `tuning/autotune`

Zodpovednost:

- candidate search
- candidate evaluation
- finalist selection
- profile persistence
- navazani na numerics post-check

Sem patri nove:

- `GraphAutotuner`
- `AutotuneSpec`
- `AutotuneResult`
- `AutotuneCandidate`
- `AutotuneEvaluator`
- `AutotuneProfileStore`

Sem se maji postupne presunout / prepsat casti dnesniho:

- `benchmark.autotune.*`

ale uz ne v benchmark-first tvaru.

### `tuning/profile`

Zodpovednost:

- resolve implicitnich default profilu
- nacitani persisted defaultu
- hardware bucket profile
- profile source priority

Sem patri:

- `ExecutionProfileResolver`
- `DefaultExecutionProfileResolver`
- `ExecutionProfileStore`

`config.profile.ExecutionProfile` muze zustat v `config.profile`, ale resolver/persistence workflow patri do `tuning.profile`.

## Doporucene rozdeleni odpovednosti

### `config.*`

- drzi pouze konfiguracni objekty a DTO

### `tuning.*`

- drzi workflow a rozhodovaci logiku

### `graph` / `backend`

- drzi compile/prepare/execute artefakty

To znamena:

- `ExecutionProfile` je config DTO
- `ExecutionProfileResolver` je tuning/profile service
- `CompiledGraph.autotune(...)` je facade nad `tuning.autotune.GraphAutotuner`

## Jak ma vypadat implicitni execution po prepisu

```java
ExecutionProfile profile = tuning.profile.DefaultExecutionProfileResolver.resolve(root, mode);
CompiledGraph.compile(root, profile.optimizer())
        .prepare(profile.runtime())
        .execute(profile.mode());
```

## Jak ma vypadat explicitni autotune

```java
CompiledGraph graph = CompiledGraph.compile(root, optimizerConfig);
AutotuneResult result = graph.autotune(AutotuneSpec.quickInference());
graph.prepare(result.profile().runtime()).execute(result.profile().mode());
```

## Co zachovat z dnesni vrstvy

Zachovat dává smysl:

- tuning knobs
- cast measurement utilit
- persistence formatu profilu
- cast numerics harnessu

Zahodit nebo prepsat:

- candidate generation navazanou na benchmark-specific stage combinatorics
- benchmark-first orchestration
- workflow, kde benchmark rozhoduje o architekture ostatnich vrstev

## Navrh balicku po prepisu

```text
tuning/
  benchmark/
    scenario/
    measure/
    score/
  numerics/
  autotune/
    candidate/
    search/
    persistence/
  profile/
```

## Doporucene poradi

1. Vytvorit novy balicek `tuning/`
2. Presunout `numerics` pod `tuning/numerics`
3. Vytahnout `ExecutionProfileResolver` a profile persistence do `tuning/profile`
4. Vytvorit novy `tuning/autotune` nad `CompiledGraph`
5. Teprve potom postupne zahodit stary `benchmark.autotune`

## Minimalni pravidlo po prepisu

- performance benchmark nesmi byt source of truth pro execution architekturu
- numerics musi zustat samostatna validacni vrstva
- autotune musi byt graph-first a profile-first

To je cilovy tvar.
