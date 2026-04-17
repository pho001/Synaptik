# Graph

`graph` vrstva převádí tensor expression DAG na explicitní runnable artifact. To je její jediný hlavní úkol. Není to backend a není to veřejná tensor surface.

Dnešní kontrakt je:

- `Tensor` skládá semantický DAG
- `CompiledGraph` z něj vytvoří optimalizovaný execution graph
- `PreparedExecution` k němu přidá runtime-specific metadata
- backend pak spouští prepared node steps

## Reading Guide

Sem jdi, pokud potřebuješ pochopit:

- co přesně dělá `compile(...)`
- co přesně dělá `prepare(...)`
- kde vzniká hranice forward/backward
- jak se připravují fused executables
- jak se hot path trace vrací zpět do benchmarků a debug tooling

Související dokumentace:

- tensor/public API: [../tensor/README.md](../tensor/README.md)
- operation descriptors: [../operations/README.md](../operations/README.md)
- optimizer pipeline: [../graph/optimizer/README.md](../graph/optimizer/README.md)
- backend execution: [../backend/README.md](../backend/README.md)

## Main Components

- compile artifact
  - [CompiledGraph.java](../graph/CompiledGraph.java)
- prepared runtime artifact
  - [PreparedExecution.java](../graph/execution/PreparedExecution.java)
  - [PreparedNodeExecution.java](../graph/execution/PreparedNodeExecution.java)
  - [CompiledNodeExecutionMetadata.java](../graph/execution/CompiledNodeExecutionMetadata.java)
- tracing
  - [CompileTrace.java](../graph/execution/trace/CompileTrace.java)
  - [PrepareTrace.java](../graph/execution/trace/PrepareTrace.java)
  - [RunTrace.java](../graph/execution/trace/RunTrace.java)
- fused preparation
  - [FusedExecutionPlan.java](../graph/fused/FusedExecutionPlan.java)
  - [FusedExecutionBackendResolver.java](../graph/fused/FusedExecutionBackendResolver.java)
  - [PreparedFusedExecutable.java](../graph/fused/PreparedFusedExecutable.java)

## Lifecycle

Nejdůležitější je držet v hlavě tři rozdílné artefakty:

### 1. `Tensor` graph

Semantický graph složený z veřejných tensor operací.

Obsahuje:

- operation descriptors
- input dependencies
- gradient references
- metadata a runtime data storage

Neobsahuje:

- prepared backend metadata
- runtime dispatch hints
- prepared fused executable

### 2. `CompiledGraph`

Compile-time artifact.

Obsahuje:

- final topological node order
- oddělení forward/backward části
- optimizer output
- compile trace

### 3. `PreparedExecution`

Runtime-bound artifact.

Obsahuje:

- ordered prepared forward steps
- ordered prepared backward steps
- per-node prepared metadata
- runtime config, se kterou byl graph připraven
- prepare trace

`PreparedExecution` je to, co máš držet pro opakované hot execution nad stejným grafem.

## Compile Pipeline

`CompiledGraph.compile(root, optimizerConfig)` dnes dělá tento flow:

1. vezme `rootTensor.forwardOutput()`
2. udělá topological sort forward closure
3. pokud graf nemá trainable leaf inputs:
   - optimalizuje jen forward graph
   - uloží boundary na forward output
4. pokud graf podporuje backward:
   - seedne root gradient
   - zavolá `buildBackwardGraph()` od konce forward order
   - sesbírá backward targets
   - vytvoří dočasný `noop` super-root pro sjednocení sinků
   - optimalizuje celý combined graph
5. uloží index konce forward části

To znamená, že optimizer běží nad jedním velkým grafem, který může obsahovat forward i backward sekci.

## Forward/Backward Boundary

Boundary není odvozená až za běhu. Je explicitně uložená už v `CompiledGraph`.

To má několik důsledků:

- optimizer pravidla musí respektovat forward/backward phase boundary
- tracing může oddělit forward a backward kroky
- `PreparedExecution` nemusí znovu hádat, co je která sekce

Backward existence se dnes pozná přes:

- `CompiledGraph.supportsBackward()`
- `PreparedExecution.supportsBackward()`

## Prepare Pipeline

`CompiledGraph.prepare(runtimeConfig)` je runtime-specific krok. Není to jen "copy finalGraph do jiného objektu".

Reálně dělá:

1. zvolí effective runtime config
   - explicitní vstup
   - nebo inference/training defaults podle podpory backward
2. vytvoří `CpuExecutionPlanner`
3. projde `finalGraph`
4. pro každý executable node připraví `CompiledNodeExecutionMetadata`
5. rozdělí prepared steps na forward/backward
6. vrátí `PreparedExecution`

Připravená metadata obsahují podle typu node:

- resolved backend
- `CpuKernel`
- `CpuNodeExecutionPlan`
- `PreparedFusedExecutable`
- `CpuNodeWorkspace`

## What Prepare Resolves

Tohle je klíčové: `prepare(...)` řeší rozhodnutí, která nechceme dělat v hot inner loop.

Typicky:

- input materialization / prepared inputs
- broadcast plan
- `where` broadcast plan
- compute contract
- dispatch hints
- reduction hints
- matmul hints
- fused executable generation
- workspace allocation
- některé BF16 continuation policies

## Execution Flow

`PreparedExecution.execute(mode)` dělá:

1. vytvoří `ExecutionContext`
2. spustí prepared forward steps v topological order
3. synchronizuje data z optimized forward output do původního root tensoru
4. pokud je režim `FORWARD_BACKWARD`:
   - vynuluje gradienty
   - seedne root gradient jedničkami
   - spustí prepared backward steps

To je důležité pro korektní benchmark:

- compile overhead do steady-state nepatří
- prepare overhead do steady-state nepatří
- opakované execution má běžet nad jedním `PreparedExecution`

## Traced Execution

`PreparedExecution.executeTraced(...)` vrací `RunTrace`.

Každý step trace nese:

- label node
- `Operation.OpType`
- shape
- dtype
- backend
- kernel class
- step duration
- structured metadata

Structured metadata obsahují například:

- compute metadata
- layout metadata
- dispatch metadata
- reduction metadata
- matmul metadata
- fused metadata

Praktický význam:

- ověříš, že benchmark opravdu běžel na očekávané path
- zjistíš `vectorWidth`, worker count, tile sizes, BLAS use
- najdeš scalar fallbacky nebo neočekávaný strided path

## Fused Preparation

Fused execution má dvě odlišné vrstvy:

### Graph descriptor layer

- `FusedOperation`
- `FusedExpressionPlan`
- `FusedExternalInputPlan`
- další codegen/fusion pomocné deskriptory

To je stále graph-level reprezentace.

### Prepared runtime layer

- `FusedExecutionPlan`
- `PreparedFusedExecutable`
- `FusedExecutionBackendResolver`

Tohle už je runtime-specific executable vrstva.

## Current Fused Reality

Tady byl historicky největší drift mezi dokumentací a kódem, takže explicitně:

- optimizer může vytvořit fused node
- `prepare(...)` pro něj spočítá fused execution plan
- `FusedExecutionBackendResolver` dnes používá ASM fused backend
- pokud plán ASM backend nepodporuje, prepare skončí chybou

Tedy:

- fused path dnes není direct/vector hybrid backend s fallbackem
- prepared fused executable je dnes ASM-generated executable
- runtime scheduling nad ním pořád může být scalar/vector/parallel podle prepared dispatch hints

`PreparedFusedExecutable` má kontrakt:

- `applyRangeScalar(...)`
- `applyRangeVector(...)`

Default vector implementace na interfacu fallbackuje do scalar. Skutečná vektorizace tedy závisí na konkrétní připravené implementaci.

## Fused Access Model

Fused compiler nerozlišuje jen "jaká operace se počítá", ale i "jak se sahá do vstupních tensorů".

To je důvod, proč se rozlišuje:

- compute algebra
- access algebra

### Compute algebra

Sem patří fused per-element výpočet:

- unary numeric ops
- binary numeric ops
- compare ops
- logical ops
- `where`

### Access algebra

Sem patří view/layout transformace na vstupu:

- `SELECT`
- `PERMUTE`
- `EXPAND`
- `RESHAPE`
- `EXPAND_DIMS`
- `SQUEEZE`

Ty se neberou jako fused compute nodes. Jsou absorbované do `FusedExternalInputPlan`.

To umožní:

- jednu output-space loop
- bez mezitensorů
- ale se správným stride/offset/broadcast mappingem na backing storage

## Example: Fused Arithmetic Chain

```java
Tensor out = a.add(b).relu().exp();
```

Nefused runtime model konceptuálně dělá:

1. `tmp0 = a + b`
2. materialize `tmp0`
3. `tmp1 = relu(tmp0)`
4. materialize `tmp1`
5. `out = exp(tmp1)`

Fused runtime model dělá:

```java
for (int i = 0; i < out.numel(); i++) {
    double v0 = a[i] + b[i];
    double v1 = Math.max(v0, 0.0);
    out[i] = Math.exp(v1);
}
```

Smysl fused path je přesně tenhle:

- odstranit intermediate materialization
- zredukovat dispatch overhead
- držet výpočet v jedné output-space loop

## Example: Access Chain Absorption

```java
Tensor base = ...;
Tensor view = base.select(0, 1).permute(1, 0);
Tensor out = view.relu().exp();
```

Co se děje:

- `relu` a `exp` jsou fused compute nodes
- `select` a `permute` nejsou fused compute nodes
- fused node dostane jako external input backing tensor `base`
- access metadata popíší offset/strides/logical mapping

To je důležité i architektonicky:

- graph dál nese semantiku view operací
- runtime fused executable nedostává celý graph
- dostává jen už rozložený prepared access contract

## Barriers

Fused cluster nemůže spolknout cokoli. Dnes typicky fungují jako bariéra:

- indexing
  - `GATHER`
  - `TAKE_ALONG_AXIS`
  - `SCATTER_ADD`
- reductions
  - `SUM`
  - `MEAN`
  - `REDUCE_MIN`
  - `REDUCE_MAX`
  - `REDUCE_ALL`
  - `REDUCE_ANY`
  - `SOFTMAX`
  - `LOG_SOFTMAX`
- linear algebra
  - `MATMUL`
- losses a special structured kernels
- special gradient kernels

To je záměr. Tyhle families mají vlastní traversal/kernel logiku a nejsou jen "lokální per-element algebra".

## Relationship To Optimizer Rewrites

Graph vrstva sama nic nepřepisuje. Jen aplikuje optimizer pipeline. Ale je důležité vědět, že po optimizeru už graf může obsahovat specializovaná primitiva místo rozpadlých patternů.

Například:

- `matmul + bias` může být přepsané na `LINEAR`
- attention pattern může být přepsaný na `SCALED_DOT_PRODUCT_ATTENTION`
- backward softmax pattern může být přepsaný na `SOFTMAX_GRAD`
- cross-entropy-from-indices pattern může být přepsaný na `CROSS_ENTROPY_LOSS_INDICES`

Graph vrstva to pak bere jako hotovou compile-time realitu a připravuje metadata pro daný descriptor.

## Workspaces

Některé nodes potřebují extra prepared workspace. `CompiledGraph` je přiděluje už v prepare fázi.

Příklady:

- max-pool argmax buffer
- BF16 float workspace pro `MATMUL`
- packed weights workspace pro `LINEAR`
- float workspace pro vybrané continuation paths

Smysl:

- workspace se nevytváří ad hoc uvnitř každého hot kernel callu
- prepared metadata přesně říkají, který node workspace má

## Example: Explicit Compile / Prepare Reuse

```java
Tensor out = logits.logSoftmax(-1).sum();

CompiledGraph graph = CompiledGraph.compile(out, OptimizerConfig.trainingDefaults());
PreparedExecution prepared = graph.prepare(RuntimeConfig.trainingDefaults());

prepared.execute(ExecutionMode.FORWARD_BACKWARD);
prepared.execute(ExecutionMode.FORWARD_BACKWARD);
```

Použij to pro:

- výkonová měření
- trace collection
- opakované inference/training běhy

## Example: Trace Audit

Typický výkonový audit vypadá takto:

1. sestav graf přes `Tensor` API
2. `CompiledGraph.compile(...)`
3. `PreparedExecution prepared = graph.prepare(...)`
4. `RunTrace trace = prepared.executeTraced(...)`
5. analyzuj step metadata

Sleduj hlavně:

- kernel class
- `compute.backend`
- dispatch mode
- `vectorWidth`
- `plannedWorkers`
- matmul `useBlas`
- fused executable class

## Public Entry Points

Veřejně relevantní entrypointy:

- `CompiledGraph.compile(Tensor root, OptimizerConfig optimizerConfig)`
- `CompiledGraph.prepare(RuntimeConfig runtimeConfig)`
- `CompiledGraph.execute(...)`
- `CompiledGraph.executeTraced(...)`
- `PreparedExecution.execute(...)`
- `PreparedExecution.executeTraced(...)`

Lower-level `GraphOptimizer` injection stále existuje, ale není to preferovaný public compile contract.

## Common Mistakes

- benchmarkovat `Tensor.compute(profile)` místo reuse `PreparedExecution`
- považovat `Operation` descriptor za hot executable
- myslet si, že `prepare(...)` je jen levný wrapper bez runtime rozhodnutí
- brát fused node jako "hotovou compiled ASM class" už v optimizeru
- míchat graph policy a runtime policy do jedné vrstvy

## Related Modules

- tensor: [../tensor/README.md](../tensor/README.md)
- operations: [../operations/README.md](../operations/README.md)
- optimizer: [../graph/optimizer/README.md](../graph/optimizer/README.md)
- backend: [../backend/README.md](../backend/README.md)
- numerics: [../numerics/README.md](../numerics/README.md)
