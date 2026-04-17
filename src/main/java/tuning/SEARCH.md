# Tuning Search

Search vrstva rozhoduje, které kandidáty stojí za to změřit a v jakém pořadí. Sama nic nespouští.

Její kontrakt je:

- dostane `SearchContext`
- vrátí `SearchResult`

Případně může podporovat refinement nad už změřenými kandidáty.

## Reading Guide

Tento dokument vysvětluje:

- jaký je minimální kontrakt search strategie
- jak fungují exhaustive a tree strategie
- jak funguje history-aware ordering
- jak se volí default strategie podle candidate space

## Core Contracts

### `SearchStrategy`

Interface:

```java
interface SearchStrategy {
    SearchResult search(SearchContext context);
}
```

Volitelně:

```java
boolean supportsRefinement();
SearchResult refine(
        SearchContext context,
        List<BenchmarkCandidateReport> evaluatedSoFar,
        int round,
        Set<String> seenFingerprints
);
```

To znamená:

- search může být jednokolový
- nebo iterativní

### `SearchContext`

Obsahuje:

- `AutotuneRequest`
- `CandidateSpace`

### `SearchResult`

Obsahuje:

- `selectedCandidates`
- `preferredCandidate`

`preferredCandidate` je hint, ne execute contract.

### `SearchPolicy`

Nese budget:

- `maxCandidates`
- `beamWidth`
- `maxRounds`
- `allowPruning`

## Candidate Spaces

Search nepracuje přímo s raw knobs. Pracuje s candidate spaces.

Základní typy:

- `CandidateSpace`
- `RefinableCandidateSpace`

`CandidateSpace`:

- umí vygenerovat počáteční kandidáty

`RefinableCandidateSpace`:

- umí generovat sousedy kolem již známého kandidáta

To je to, co umožňuje tree search bez paralelního execute modelu.

## Search Lifecycle In Autotune

`DefaultAutotuneSession` dnes dělá:

1. `search(context)` pro initial batch
2. kandidáty validuje a měří
3. pokud strategie umí refinement:
   - volá `refine(...)`
   - znovu validuje a měří nový batch
4. vybere best finalist podle median

Tedy:

- search vrstva nikdy nevolá measurement přímo
- session ji používá jako policy vrstvu nad evaluation loop

## Simple Strategies

### `ExhaustiveSearchStrategy`

Použij, když:

- candidate grid je malý
- chceš úplné pokrytí

Výhody:

- jednoduchost
- žádná heuristická chyba

Nevýhoda:

- neškáluje

### `FirstKSearchStrategy`

Použij, když:

- chceš seed batch
- potřebuješ budget guard

Sama o sobě většinou není finální strategie. Často slouží jako seed pro tree strategie.

### `CompositeSearchStrategy`

Použij, když:

- chceš spojit více ordering heuristik
- chceš deduplikovaný seznam seed kandidátů

## Tree Strategies

Tree search dává smysl jen pokud candidate space umí refinement nebo sousednost.

### `TreeBeamSearchStrategy`

Myšlenka:

1. vyber seed kandidáty
2. změř je
3. nech si nejlepší frontier podle `beamWidth`
4. expanduj jejich neighborhood
5. opakuj

Použij, když:

- chceš rozumný kompromis mezi šířkou a cenou
- candidate space je refinable

### `BestFirstTreeSearchStrategy`

Myšlenka:

- v každém kroku expanduj jen nejperspektivnější frontier node

Použij, když:

- score modelu věříš
- chceš agresivní focus místo breadth

### `BranchAndBoundSearchStrategy`

Myšlenka:

1. měj current best measured score
2. pro frontier node spočítej optimistic bound
3. pokud bound je horší než best, větev zahodíš
4. expanduješ jen zbývající větve

Použij, když:

- workload family má rozumný bound model
- candidate space je větší

## Score And Bound Models

### Score Model

Relevantní třídy:

- [CandidateScoreModel.java](./search/CandidateScoreModel.java)
- [MedianSteadyStateScoreModel.java](./search/MedianSteadyStateScoreModel.java)

Aktuální default skóre je:

- nižší steady-state median = lepší

### Bound Models

Relevantní třídy:

- [CandidateBoundModel.java](./search/CandidateBoundModel.java)
- [ZeroBoundModel.java](./search/ZeroBoundModel.java)
- [ParentScoreBoundModel.java](./search/ParentScoreBoundModel.java)
- [WorkloadAwareBoundModel.java](./search/WorkloadAwareBoundModel.java)

`WorkloadAwareBoundModel` dnes dispatchuje podle `WorkloadKind`:

- `CONV2D`
- `MATMUL`
- `TRANSFORMER_HOT_PATH`
- jinak generic fallback

To znamená:

- search heuristiky mohou být workload-aware
- ale pořád vracejí jen ordering/pruning hint, ne execute semantics

## History-Aware Search

`HistoryAwareSearchStrategy` je wrapper nad jinou strategií.

Dělá:

1. načte persisted best profile pro aktuální hardware + workload
2. pokud sedí fingerprint, posune ho dopředu
3. načte history entries pro stejný kontext
4. preferuje historicky dobré kandidáty
5. může přeskočit historicky invalid kandidáty, pokud je pruning povolený

Důležitá realita:

- neprovádí vlastní scoring
- jen reorderuje candidate space před delegováním na vnitřní strategii

## Default Strategy Selection

Výběr default strategie řeší:

- [AutotuneDefaultStrategySelector.java](./session/AutotuneDefaultStrategySelector.java)

Aktuální logika:

- non-refinable space
  - `Exhaustive`
- refinable space a dost velký candidate count
  - `BranchAndBound`
- refinable space střední velikosti
  - `TreeBeam`
- pokud je persistence zapnutá
  - obalí se do `HistoryAwareSearchStrategy`

Tedy:

- default selection není natvrdo uvnitř strategií
- je to policy vrstva

## Example: Small Stage-Order Space

Pokud ladíš malý stage-order grid:

- candidate space je malý
- refinement typicky nedává smysl

Použij:

- `ExhaustiveSearchStrategy`

## Example: Matmul Runtime Search

Pokud máš větší refinable matmul candidate space:

- tiles
- microkernels
- thresholds

rozumný default je:

- `BranchAndBoundSearchStrategy`

protože:

- prostor je větší
- `WorkloadAwareBoundModel` umí matmul hinty

## Example: Repeated Tuning On Same Machine

Pokud máš už uložené:

- best profile
- history JSONL

obal strategii přes:

- `HistoryAwareSearchStrategy`

Smysl:

- znovu otestuješ pravděpodobně dobré kandidáty dřív
- můžeš vynechat historicky invalid varianty

## Search Does Not Own Persistence

Search může persistence číst jako prior, ale nevlastní její lifecycle.

Persistence lifecycle řeší session/store vrstvy.

To je důležité, protože:

- historie je pomocná evidence
- search ji nesmí proměnit v execute source of truth

## Common Mistakes

- chtít po search strategii, aby sama měřila kandidáty
- používat branch-and-bound bez rozumného bound modelu
- zapomenout na candidate deduplikaci přes fingerprint
- považovat history-aware reordering za důkaz, že uložený winner je stále správný

## Related Docs

- architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)
- persistence: [PERSISTENCE.md](./PERSISTENCE.md)
- reporting: [REPORTING.md](./REPORTING.md)
