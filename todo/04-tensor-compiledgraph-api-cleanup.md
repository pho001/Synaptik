# Tensor and CompiledGraph API Cleanup

## Stav dokumentu

Tento dokument nahrazuje starsi kompromisni navrh. Cilem uz neni jen "uklidit overloady", ale udelat ciste oddeleni vrstev:

- `Tensor` = graph node + data + uzivatelske expression API
- `CompiledGraph` = explicitni compile artifact
- `PreparedExecution` = explicitni runtime artifact
- `ExecutionProfile` = jediny high-level execution policy objekt

Testy a benchmark se v tomto kroku zamerne ignoruji. Tento dokument popisuje cilovou architekturu, ne minimalni kompatibilni mezistav.

---

## 1. Problem dnesniho API

Aktualni `Tensor` stale smichava nekolik ruznych odpovednosti:

- graph/data vrstvu
- compile lifecycle
- cast runtime lifecycle
- implicitni optimizer selection
- convenience API, ktere skryva, kdy se graph kompiluje a kdy jen spousti
- compile cache (`compiledGraph`, `compiledWithOptimizer`)

To vytvari nekolik systemovych problemu:

- neni jasne, co je source of truth pro compile artifact
- `Tensor` ma skryty mutabilni execution stav
- explicitni reuse compiled graphu je nejasny a obtizne kontrolovatelny
- autotune nema prirozeny anchor point
- performance-sensitive call sites nemaji ciste oddelenou compile a runtime fazi
- API overloady zamlzuji, co je compile-time concern a co runtime concern

Jinymi slovy: `Tensor` je dnes soucasne AST node, value buffer, compiler facade a castecne i execution owner. To je spatne.

---

## 2. Cilovy architektonicky model

### `Tensor`

`Tensor` ma reprezentovat:

- data / storage
- metadata (`shape`, `strides`, `dtype`, `label`)
- graph vazby (`operation`, `prevTensors`)
- gradient a backward hooks
- expression-building API (`add`, `mul`, `matmul`, `reshape`, ...)

`Tensor` nema reprezentovat:

- compiled graph
- optimizer choice
- compile cache
- explicitni runtime plan
- autotune state

### `CompiledGraph`

`CompiledGraph` ma reprezentovat:

- vysledek kompilace konkretniho root graphu
- optimized / ordered graph
- compile-time rozhodnuti
- introspekcni a debug view na compiled strukturu

Je to jediny spravny owner pro:

- `OptimizerConfig`
- graph fingerprint / compile signature
- compile-time metadata
- explicitni reuse zkompilovane varianty graphu

### `PreparedExecution`

`PreparedExecution` ma reprezentovat:

- runtime-bound variantu konkretniho `CompiledGraph`
- prepared node kroky
- backend dispatch metadata
- runtime config
- forward/backward execution API

### `ExecutionProfile`

`ExecutionProfile` ma byt jediny high-level policy objekt, ktery urcuje:

- compile-time politiku (`OptimizerConfig`)
- runtime politiku (`RuntimeConfig`)
- execution mode (`FORWARD_BACKWARD` / `FORWARD`)

Autotune ma vracet `ExecutionProfile`, ne zavadet dalsi pseudo-mode.

---

## 3. Finalni odpovednosti po tridach

## `Tensor`

Povinne odpovednosti:

- drzet data a metadata
- drzet graph links
- drzet gradient state
- poskytovat expression API
- poskytovat tenkou convenience facade nad profile-driven execution

Nepovolene odpovednosti:

- vlastnit `CompiledGraph`
- pamatovat si compile cache
- rozhodovat optimizer podle runtime overloadu
- poskytovat explicitni compile API
- exponovat compiled artifact navenek

## `CompiledGraph`

Povinne odpovednosti:

- zkompilovat graph z root tensoru a `OptimizerConfig`
- drzet final ordered graph
- drzet compile-time metadata
- pripravit runtime-specific `PreparedExecution`
- vykonavat explicitni execution nad runtime/profile
- byt anchor point pro autotune

## `PreparedExecution`

Povinne odpovednosti:

- drzet runtime config
- drzet prepared steps
- umet `execute(mode)`
- umet `backward()`
- byt terminal execution artifact

---

## 4. Finalni verejne API

Tohle je cilovy kontrakt. Ne kompromis, ale cilovy stav.

## `Tensor`

Ponechat pouze:

```java
public PreparedExecution prepare(ExecutionProfile profile);
public void compute(ExecutionProfile profile);
public void compute(PreparedExecution execution, ExecutionMode mode);
```

Volitelna ergonomicka vrstva:

```java
public ExecutionProfile autotune(AutotuneRequest request);
```

Tato metoda ma byt pouze facade:

```java
return GraphAutotuner.autotune(this, request).bestProfile();
```

### Co na `Tensor` nezustane

Smazat:

```java
public CompiledGraph compile(...);
public PreparedExecution prepare(RuntimeConfig runtimeConfig);
public PreparedExecution prepare(GraphOptimizer optimizer, RuntimeConfig runtimeConfig);
public void compute();
public void compute(RuntimeConfig runtimeConfig, ExecutionMode mode);
public void compute(GraphOptimizer optimizer);
public void compute(GraphOptimizer optimizer, RuntimeConfig runtimeConfig);
public void compute(GraphOptimizer optimizer, RuntimeConfig runtimeConfig, ExecutionMode mode);
public CompiledGraph getCompiledGraph();
public void resetCompiledGraph();
public void prepareCompiledGraph(...);
```

Toto API je compile/runtime leakage do `Tensor`.

## `CompiledGraph`

Ponechat / zavest:

```java
public static CompiledGraph compile(Tensor root, OptimizerConfig optimizerConfig);
public PreparedExecution prepare(RuntimeConfig runtimeConfig);
public PreparedExecution prepare(ExecutionProfile profile);
public void execute(RuntimeConfig runtimeConfig, ExecutionMode mode);
public void execute(ExecutionProfile profile);
public void executePrepared(PreparedExecution execution, ExecutionMode mode);
public void zeroGrad();
public List<Tensor> getCompiledGraphAsList();
```

Volitelne do dalsi faze:

```java
public GraphFingerprint fingerprint();
public ExecutionProfile autotune(AutotuneRequest request);
```

## `PreparedExecution`

Ponechat:

```java
public void execute(ExecutionMode mode);
public void backward();
public RuntimeConfig runtimeConfig();
public boolean supportsBackward();
public List<PreparedNodeExecution> forwardSteps();
public List<PreparedNodeExecution> backwardSteps();
```

---

## 5. Proc `Tensor.compute(AUTOTUNE)` neni spravne

`AUTOTUNE` neni execution mode. Je to search workflow nad profilem.

Tyto veci jsou ruzne:

- `FORWARD_BACKWARD`
- `FORWARD`
- `AUTOTUNE`

Prvni dve rikaji, co se ma vykonat.
Posledni rika, jak hledat nejlepsi policy pred vykonanim.

Spravny model je:

```java
ExecutionProfile tuned = tensor.autotune(request);
tensor.compute(tuned);
```

ne:

```java
tensor.compute(AUTOTUNE);
```

`Tensor` muze mit ergonomickou facade, ale autotune se ma opirat o:

```java
CompiledGraph.compile(...)
PreparedExecution.prepare(...)
execute(...)
```

---

## 6. Jak bude fungovat autotune nad novym API

Autotuner nema menit `Tensor`.
Autotuner meni policy nad graph-em.

Jedna tuning iterace ma vypadat takto:

```java
CompiledGraph graph = CompiledGraph.compile(root, candidate.optimizer());
PreparedExecution execution = graph.prepare(candidate.runtime());
execution.execute(candidate.mode());
```

Autotuner tedy hleda nejlepsi:

- `OptimizerConfig`
- `RuntimeConfig`

pro dany graph fingerprint.

To znamena:

- `Tensor` je specifikace vypoctu
- `CompiledGraph` je jedna compile varianta
- `PreparedExecution` je jedna runtime materializace
- autotuner vybira nejlepsi `ExecutionProfile`

---

## 7. Co odstranit z interniho stavu `Tensor`

Nasledujici pole nemaji v cilovem `Tensor` existovat:

- `compiledGraph`
- `compiledWithOptimizer`
- jakakoli value-based compile cache v `Tensor`
- `lastPreparedExecution`, pokud se rozhodneme odstranit i `Tensor.backward()`

Pokud ponechame `Tensor.backward()` jen jako ergonomicky sugar, pak muze kratkodobe existovat:

- `lastPreparedExecution`

ale pouze jako backward facade, ne jako compile/runtime source of truth.

### Doporuceni

Nejcistsi varianta je odstranit i `Tensor.backward()` a pouzivat pouze:

```java
PreparedExecution execution = tensor.prepare(profile);
execution.execute(FORWARD_BACKWARD);
```

nebo:

```java
execution.backward();
```

Pokud ale chceme ergonomii, je prijatelne nechat:

```java
public void backward();
```

jako tenkou delegaci na posledni `PreparedExecution`.

---

## 8. Migrační plan refaktoru

## Faze A - zafixovat cilovy kontrakt

- potvrdit finalni public API z tohoto dokumentu
- nezkouset mezikroky typu dalsi overloady v `Tensor`
- benchmark a testy zamerne ted neresit

## Faze B - posilit `CompiledGraph` jako owner lifecycle

- `CompiledGraph.compile(root, OptimizerConfig)` bude jedina explicitni compile cesta
- vsechny explicitni compile/prepare/execute flow budou prepsany sem
- `ExecutionProfile` vstup bude brat `CompiledGraph`, ne `Tensor` internals

## Faze C - osekat `Tensor`

- smazat explicitni compile API
- smazat optimizer-centric compute/prepare overloady
- smazat compiled graph cache pole
- smazat `getCompiledGraph()`

## Faze D - doplnit autotune entry point

- zavest `AutotuneRequest`
- zavest `GraphAutotuner`
- volitelne pridat `Tensor.autotune(request)` jako facade
- vracet `ExecutionProfile`, ne side-effect

## Faze E - az potom prepsat call sites

- testy
- numerics
- benchmark
- ostatni helpery

Call sites se maji prizpusobit cile, ne naopak.

---

## 9. Doporucene usage patterns po refaktoru

## High-level user path

```java
ExecutionProfile profile = ...;
tensor.compute(profile);
```

## Explicitni performance path

```java
CompiledGraph graph = CompiledGraph.compile(tensor, profile.optimizer());
PreparedExecution execution = graph.prepare(profile.runtime());
execution.execute(profile.mode());
```

## Reuse compiled graphu

```java
CompiledGraph graph = CompiledGraph.compile(tensor, optimizerConfig);

PreparedExecution fastExec = graph.prepare(runtimeFast);
fastExec.execute(FORWARD);

PreparedExecution accurateExec = graph.prepare(runtimeAccurate);
accurateExec.execute(FORWARD);
```

## Autotune path

```java
ExecutionProfile tuned = tensor.autotune(request);
tensor.compute(tuned);
```

nebo explicitne:

```java
ExecutionProfile tuned = GraphAutotuner.autotune(tensor, request).bestProfile();
CompiledGraph graph = CompiledGraph.compile(tensor, tuned.optimizer());
PreparedExecution execution = graph.prepare(tuned.runtime());
execution.execute(tuned.mode());
```

---

## 10. Finalni rozhodnuti

Tento dokument zavadi nasledujici architektonicke rozhodnuti:

- `Tensor` bude deklarativni a tenky
- `CompiledGraph` bude jediny explicitni compile owner
- `PreparedExecution` bude jediny explicitni runtime owner
- `ExecutionProfile` bude jediny high-level execution policy objekt
- autotune bude vracet `ExecutionProfile`
- `AUTOTUNE` nebude novy `ExecutionMode`

To je nejcistsi smer pro:

- vykon
- citelnost API
- reuse compiled artifactu
- autotune nad graph-em
- oddeleni compile a runtime vrstev
