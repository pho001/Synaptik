# Numerics Harness

`numerics` je malý samostatný A/B harness pro porovnání numerického driftu mezi dvěma `ExecutionProfile` variantami. Není to benchmark subsystem a není to autotuner.

Jeho úkol je prostý:

- vezme dvě executable profile varianty
- spustí je nad stejnými deterministickými vstupy
- porovná výstupy a vybrané gradienty
- vrátí lidsky čitelný verdict

## Reading Guide

Použij tento modul, pokud chceš:

- rychle zkontrolovat, že nová optimizer stage order neporušila numeriku
- porovnat baseline vs agresivnější rewrite/fusion variantu
- zkontrolovat drift mezi `FLOAT32` a `FLOAT64` policy variantami
- ověřit, že approximation policy nezpůsobila nepřijatelný rozptyl

Nepoužívej ho jako:

- performance benchmark
- náhradu unit testů
- důkaz úplné numerické správnosti celého frameworku

## Main Components

- CLI
  - [NumericsCli.java](./NumericsCli.java)
- harness orchestrace
  - [NumericsHarness.java](./NumericsHarness.java)
- scénáře / graph recipe
  - [NumericsGraphFactory.java](./NumericsGraphFactory.java)
- metriky
  - [NumericsMetrics.java](./NumericsMetrics.java)
- tolerance policy
  - [NumericsPolicy.java](./NumericsPolicy.java)
- report
  - [NumericsReport.java](./NumericsReport.java)

## What It Measures

Harness dnes porovnává pět signálů:

- `out`
  - forward výstup benchmark-like graphu
- `gradA`
- `gradB`
- `gradC`
  - gradienty tří leaf vstupů benchmark-like graphu
- `broadcast`
  - forward výstup samostatného broadcast-heavy graphu

Pro každý signál počítá:

- `maxAbs`
- `avgAbs`
- `maxRel`
- `maxUlp`
- `p50Ulp`
- `p95Ulp`
- `invalidCount`

Pak z nich složí aggregate metriky a na ně aplikuje `NumericsPolicy`.

## What It Actually Runs

`NumericsHarness` nespouští jeden jediný graph. Spouští dva scénáře:

### 1. Optimizer-like training graph

V [NumericsGraphFactory.java](./NumericsGraphFactory.java) je složený z:

- opakovaných elementwise bloků nad `A`, `B`, `C`
- několika `linear(...)` vrstev
- finální scalar reduction

Tenhle graph se spouští v režimu:

- `FORWARD_BACKWARD`

a harness z něj sbírá:

- forward output
- gradienty `A`, `B`, `C`

### 2. Broadcast-heavy forward graph

Samostatný graph:

- `a.add(b).mul(c).add(a).sigmoid()`

nad broadcast-compatible shapes.

Ten se spouští v režimu:

- `FORWARD`

a slouží k odhalení driftu v broadcast/elementwise path, který by optimizer-like graph sám nemusel zachytit.

## Why Two Graphs

Tohle není náhoda.

Jeden graph sám typicky nezachytí obě věci:

- numeriku v širším training-style graphu
- numeriku v broadcast-heavy shape/layout situacích

Proto harness kombinuje:

- jeden "hlubší" training-like scénář
- jeden "plošší" broadcast scénář

## Determinism And Input Policy

Vstupy jsou deterministické:

- seed je řízený `numerics.seed`
- input arrays se generují jednou
- obě candidate profily dostávají stejná data

To je naprosto zásadní. Bez toho by výsledky neříkaly nic o numerickém driftu mezi kandidáty, ale jen o různých datech.

## Candidate Model

CLI dnes skládá dva kandidáty přes:

- dtype
- jméno varianty
- stage order

`NumericsHarness.profile(...)` vytváří `ExecutionProfile` takto:

- `OptimizerConfig.trainingDefaults().withStageOrder(...)`
- `RuntimeConfig.trainingDefaults()`

To znamená:

- harness dnes primárně porovnává graph-policy varianty
- runtime policy zůstává fixní

Pokud chceš porovnat i runtime policy, musíš si kandidáty složit programově mimo základní CLI.

## CLI Usage

Main class:

- `numerics.NumericsCli`

Příklad:

```bash
java --add-modules jdk.incubator.vector \
  -Dnumerics.dtype=FLOAT32 \
  -Dnumerics.stageA=NONE \
  -Dnumerics.stageB=AR,CSE,FUSE \
  -Dnumerics.nameA=baseline \
  -Dnumerics.nameB=optimized \
  -Dnumerics.size=200000 \
  -Dnumerics.graphBlocks=6 \
  -Dnumerics.broadcastB0=128 \
  -Dnumerics.broadcastB1=8 \
  -Dnumerics.broadcastF=128 \
  -Dnumerics.seed=42 \
  -cp build/classes/java/main \
  numerics.NumericsCli
```

## Properties

- `numerics.dtype`
  - `FLOAT32` nebo `FLOAT64`
  - default `FLOAT32`
- `numerics.stageA`
  - comma nebo `+` separated stage list
  - `NONE` znamená prázdný stage order
- `numerics.stageB`
  - stejné jako `stageA`
- `numerics.nameA`
  - label kandidáta A
- `numerics.nameB`
  - label kandidáta B
- `numerics.size`
  - velikost flat training input arrays
- `numerics.graphBlocks`
  - kolik opakovaných optimizer-like bloků graph obsahuje
- `numerics.broadcastB0`
- `numerics.broadcastB1`
- `numerics.broadcastF`
  - shape parametry broadcast scénáře
- `numerics.seed`
  - RNG seed

## Stage Syntax

`NumericsHarness.parseStages(...)` akceptuje:

- `NONE`
- `AR`
- `AR,CSE`
- `AR+CSE+FUSE`

To je užitečné hlavně při rychlém A/B:

- baseline bez optimalizace vs rewrite-only
- rewrite-only vs rewrite+CSE
- inference-like stage order vs training-like stage order

## Tolerance Policy

Default policy se volí podle dtype:

- `FLOAT64`
  - `absTol = 1e-12`
  - `relTol = 1e-12`
  - `maxUlpTol = 16`
- `FLOAT32`
  - `absTol = 1e-5`
  - `relTol = 1e-5`
  - `maxUlpTol = 128`

Verdict může být:

- `SAFE`
- `BORDERLINE`
- `UNSAFE`

### Meaning Of Verdicts

- `SAFE`
  - vše se vešlo do abs/rel i ULP tolerance
- `BORDERLINE`
  - část metrik je mimo hlavní toleranci, ale stále v přijatelném ULP pásmu
  - nebo naopak ULP drift přesáhl limit při malém absolutním rozdílu
- `UNSAFE`
  - invalid hodnoty
  - nebo výrazné překročení tolerance bez rozumného vysvětlení

To není matematický důkaz korektnosti. Je to pragmatický guardrail pro rychlou regresní kontrolu.

## Example Output

Výstup reportu vypadá zhruba takto:

```text
Numerics Report
scenario=benchmark-like, A=baseline, B=optimized
out: maxAbs=1.234e-06, avgAbs=2.100e-08, maxRel=7.000e-07, maxUlp=5, p50Ulp=0, p95Ulp=1, invalid=0
gradA: ...
gradB: ...
gradC: ...
broadcast: ...
aggregate: maxAbs=1.234e-06, maxRel=7.000e-07, maxUlp=5, invalid=0
verdict=SAFE (within abs/rel and ulp tolerance)
```

## Real Usage Patterns

### 1. Ověření nové optimizer stage kombinace

Použij:

- `stageA=AR,CSE`
- `stageB=AR,CSE,FUSE`

Smysl:

- rychle zjistíš, jestli nově zapnutá fusion nezhoršila numeriku nad rozumnou mez

### 2. Kontrola rewrite regrese

Použij:

- `stageA=NONE`
- `stageB=AR`

Smysl:

- validuješ, že rewrite family nepoškodila forward ani gradienty

### 3. Broadcast audit

Zvyš:

- `numerics.broadcastB0`
- `numerics.broadcastB1`
- `numerics.broadcastF`

Smysl:

- vynutíš vyšší váhu broadcast-heavy scénáře

## What The Harness Does Not Guarantee

Nezaručuje:

- pokrytí všech operation families
- pokrytí všech dtype/layout corner cases
- odhalení výkonových regresí
- odhalení všech long-tail NaN/Inf problémů v hlubokých sítích

Je to rychlý smoke/regression harness, ne formální numerics certification layer.

## Extending The Harness

Když chceš přidat nový scénář:

1. přidej deterministický graph recipe do `NumericsGraphFactory`
2. rozhodni, jaké signály z něj chceš sbírat
3. rozšiř `OutputSet`
4. doplň metriky a report
5. drž scénáře malé a reprodukovatelné

Nedělej z toho:

- benchmark suite
- workload zoo s dvaceti různými konfiguracemi
- druhý autotune framework

## Common Mistakes

- interpretovat `SAFE` jako důkaz absolutní správnosti
- porovnávat kandidáty s různými vstupy
- míchat numerics harness s performance benchmarkem
- přidávat syntetické scénáře bez reálného diagnostického přínosu

## Related Modules

- graph: [../graph/README.md](../graph/README.md)
- tuning: [../tuning/README.md](../tuning/README.md)
