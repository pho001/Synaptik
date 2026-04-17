# Backend

Backend vrstva provádí skutečný výpočet nad připraveným grafem. Neřeší stavbu grafu ani optimizer transformace. Její kontrakt je:

- dostane `Tensor` node
- dostane prepared metadata z `CompiledGraph.prepare(...)`
- spustí správný kernel family entrypoint pro zvolený backend

Dnes je plně implementovaný pouze CPU backend.

## Reading Guide

Tento dokument je určený hlavně pro:

- implementaci nového CPU kernelu
- audit runtime dispatch flow
- pochopení prepared metadata a compute contractu
- výkonové ladění backend path

Pokud hledáš:

- jak vzniká graf: [../tensor/README.md](../tensor/README.md)
- co je operation descriptor: [../operations/README.md](../operations/README.md)
- jak funguje compile/prepare lifecycle: [../graph/README.md](../graph/README.md)
- jak se backend knoby kalibrují: [../tuning/README.md](../tuning/README.md)

## Main Components

- dispatch facade
  - [ComputeEngine.java](../backend/ComputeEngine.java)
  - [ComputeBackend.java](../backend/ComputeBackend.java)
- concrete backends
  - [CPUBackend.java](../backend/CPUBackend.java)
  - [CudaBackend.java](../backend/CudaBackend.java)
  - [OpenClBackend.java](../backend/OpenClBackend.java)
- CPU kernel resolver
  - [CpuKernelResolver.java](../backend/registry/CpuKernelResolver.java)
- prepared runtime metadata
  - [CompiledNodeExecutionMetadata.java](../graph/execution/CompiledNodeExecutionMetadata.java)
  - [CpuNodeExecutionPlan.java](../backend/kernels/cpu/CpuNodeExecutionPlan.java)
  - [CpuNodeWorkspace.java](../backend/kernels/cpu/CpuNodeWorkspace.java)

## End-To-End Execution Flow

Skutečný runtime flow vypadá takto:

1. `CompiledGraph.prepare(runtimeConfig)` vytvoří `PreparedExecution`
2. pro každý runtime node připraví `CompiledNodeExecutionMetadata`
3. `PreparedExecution.execute(...)` iteruje prepared forward/backward steps
4. každý step volá `ComputeEngine.compute(node, metadata, context)`
5. `ComputeEngine` přepne na konkrétní backend
6. `CPUBackend.execute(...)` vezme prepared plan a spustí odpovídající `CpuKernel`

To je důležitý boundary:

- backend nesmí znovu vymýšlet optimizer policy
- runtime hot path nesmí znovu rozhodovat to, co už šlo spočítat v prepare fázi

## Prepared CPU Metadata

`CPUBackend.buildExecutionPlan(...)` připravuje per-node metadata pro CPU path. Typicky obsahují:

- layout plan
  - prepared/materiálizované inputy
  - broadcast plan
  - `where` broadcast plan
  - informace o strided path
- compute contract
  - storage dtype
  - compute dtype
  - accumulate dtype
  - resolved backend kind
- dispatch hints
  - scalar/vector/parallel mode
  - vector width
  - chunk sizes
  - worker count
- reduction hints
  - reduction mode
  - vector width
  - chunking
  - accuracy mode
- matmul hints
  - use BLAS vs Java
  - tile sizes
  - parallelism
  - selected microkernel
- optional workspace
  - float continuation
  - packed linear weights
  - max-pool argmax buffers

Výsledkem není obecná abstraktní "execution descriptor language". Je to konkrétní runtime recipe pro jeden backend node.

## CPU Package Structure

Root `backend.kernels.cpu` schválně obsahuje jen shared contracts a planner vrstvy:

- planner
- context
- dispatch hints
- dtype helpers
- workspace
- prepared plan records
- thread pool

Samotné family entrypointy jsou rozdělené tematicky:

- `elementwise/`
- `reduction/`
- `linalg/`
- `nn/`
- `index/`
- `layout/`
- `fused/`
- `grad/`

To odpovídá tomu, jak dnes CPU backend reálně vypadá v kódu.

## CPU Kernel Design Rules

Aktuální design není "jeden obří kernel class na všechno". Opakující se pattern je:

- `Cpu*Kernel`
  - tenký runtime entrypoint
  - implementuje `CpuKernel`
  - přebírá prepared metadata a zavolá family executor
- `*Executor`
  - family orchestrace
  - validace family-specific invariants
  - dispatch na konkrétní low-level path
- `*Loops` nebo `*Backend`
  - hot inner loops nebo specializovaná compute implementace

Ne všechny families mají přesně stejnou trojici názvů, ale princip je stejný:

- runtime entrypoint zůstává tenký
- orchestrace se drží mimo hot loops
- low-level compute zůstává lokálně pohromadě

## Elementwise Families

Elementwise batch je rozdělený na:

- `elementwise/binary`
- `elementwise/unary`
- `elementwise/compare`
- `elementwise/logical`
- `elementwise/where`

Uvnitř `binary` a `unary` jsou dtype-specialized leaf implementace:

- `f64`
- `f32`
- `bf16`

Architektonicky se dnes odděluje:

- lokální algebra operace
- shared loop execution
- broadcasting / stride walking
- vector vs scalar dispatch
- parallel chunk scheduling

To znamená:

- `CpuAddKernel` neobsahuje celý runtime orchestration příběh
- loop struktura je sdílená uvnitř family executorů
- vlastní per-element algebra je držena v užších leaf implementacích

### Non-Contiguous Routing

CPU používá hybridní strategii:

- malé non-contiguous tensory
  - běží přes strided path
  - [CpuStridedElementWise.java](../backend/kernels/cpu/CpuStridedElementWise.java)
- větší non-contiguous tensory
  - vstupy se materiálizují do contiguous temporary storage
  - pak běží běžný fast path

Hraniční knob:

- `cpu.contiguousMaterializeThreshold`

Tahle volba patří do runtime family tuning, ne do optimizer stage policy.

## Reduction Families

CPU reduction path není jedna homogenní větev. Má několik family executorů podle struktury operace.

### Sum-Like

Sdílená family pro:

- `SUM`
- `MEAN`

Hlavní části:

- [SumLikeReduction.java](../backend/kernels/cpu/reduction/SumLikeReduction.java)
- [SumLikeReductionExecutor.java](../backend/kernels/cpu/reduction/SumLikeReductionExecutor.java)
- [SumLoops.java](../backend/kernels/cpu/reduction/SumLoops.java)

Rozdíl mezi `SUM` a `MEAN` je primárně ve finalizaci, ne v traversal engine.

### Generic Axis Reductions

Sdílený traversal pro:

- `REDUCE_MIN`
- `REDUCE_MAX`
- `REDUCE_ALL`
- `REDUCE_ANY`

Hlavní části:

- [ReductionTraversal.java](../backend/kernels/cpu/reduction/ReductionTraversal.java)

Traversal vrstva řeší:

- mapování output group -> base storage offset
- reduction-axis walking
- optional parallel chunking

Per-op algebra pak zůstává v leaf reduction implementation.

### Softmax-Like

Sdílená strukturovaná family pro:

- `SOFTMAX`
- `LOG_SOFTMAX`

a jejich gradient families:

- `SOFTMAX_GRAD`
- `LOG_SOFTMAX_GRAD`

Hlavní části:

- [SoftmaxLikeReduction.java](../backend/kernels/cpu/reduction/SoftmaxLikeReduction.java)
- [SoftmaxLikeTraversal.java](../backend/kernels/cpu/reduction/SoftmaxLikeTraversal.java)
- [SoftmaxLikeExecutor.java](../backend/kernels/cpu/reduction/SoftmaxLikeExecutor.java)

Tohle není skládání přes obecné elementwise kernels. Je to specializovaný structured kernel family se svojí traversal logikou.

### Loss Reductions

Sdílená family pro:

- `NLL_LOSS`
- `CROSS_ENTROPY_LOSS`
- `CROSS_ENTROPY_LOSS_INDICES`
- gradient variants tam, kde dává smysl

Hlavní části:

- [LossReduction.java](../backend/kernels/cpu/reduction/LossReduction.java)
- [LossReductionTraversal.java](../backend/kernels/cpu/reduction/LossReductionTraversal.java)
- [LossReductionExecutor.java](../backend/kernels/cpu/reduction/LossReductionExecutor.java)

## Linear Algebra Families

### MatMul

Matmul path používá tuto vrstvu:

- [CpuMatMulKernel.java](../backend/kernels/cpu/linalg/CpuMatMulKernel.java)
- [MatMulExecutor.java](../backend/kernels/cpu/linalg/MatMulExecutor.java)
- [MatMulJavaBackend.java](../backend/kernels/cpu/linalg/MatMulJavaBackend.java)
- [MatMulBlasBackend.java](../backend/kernels/cpu/linalg/MatMulBlasBackend.java)

`MatMulExecutor` rozhoduje:

- BLAS vs Java backend
- BF16 continuation policy
- batched vs non-batched flow
- použití workspace

`ResolvedMatMulHints` nese už spočítaný runtime recept:

- tiles
- parallelism
- microkernel
- BLAS enablement

### Linear

`LINEAR` není implementovaný jako úplně samostatný GEMM systém. Reuseuje matmul family a přidává:

- bias epilog
- packed weight workspace
- BF16 continuation policy

Relevantní třídy:

- [CpuLinearKernel.java](../backend/kernels/cpu/linalg/CpuLinearKernel.java)
- [LinearExecutor.java](../backend/kernels/cpu/linalg/LinearExecutor.java)

## NN Spatial Families

### Conv2d

Existují dvě hlavní forward cesty:

- direct convolution
- GEMM lowered convolution

Hlavní části:

- [Conv2dExecutor.java](../backend/kernels/cpu/nn/Conv2dExecutor.java)
- [Conv2dDirectBackend.java](../backend/kernels/cpu/nn/Conv2dDirectBackend.java)
- [Conv2dGemmExecutor.java](../backend/kernels/cpu/nn/Conv2dGemmExecutor.java)
- [Conv2dGemmBackend.java](../backend/kernels/cpu/nn/Conv2dGemmBackend.java)

Volba `CONV2D` vs `CONV2D_GEMM` ale není backend runtime decision. To je compile-time rewrite/lowering decision v optimizeru.

### Pool2d

Pool family:

- [Pool2dExecutor.java](../backend/kernels/cpu/nn/Pool2dExecutor.java)
- [Pool2dDirectBackend.java](../backend/kernels/cpu/nn/Pool2dDirectBackend.java)

Max-pool backward navíc reuseuje prepared int workspace pro argmax indexy.

### Attention

Attention dnes není provozovaná jako rozpadlá `matmul + softmax + where` runtime cesta, pokud rewrite najde pattern. Může se přepsat na specializovaná primitiva:

- `SCALED_DOT_PRODUCT_ATTENTION`
- `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD`
- `SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS`

CPU resolver na ně má explicitní kernel entrypointy:

- [CpuScaledDotProductAttentionKernel.java](../backend/kernels/cpu/linalg/CpuScaledDotProductAttentionKernel.java)
- [CpuScaledDotProductAttentionBackwardKernel.java](../backend/kernels/cpu/linalg/CpuScaledDotProductAttentionBackwardKernel.java)
- [CpuScaledDotProductAttentionWeightsKernel.java](../backend/kernels/cpu/linalg/CpuScaledDotProductAttentionWeightsKernel.java)

## Index And Layout Families

### Index

Indexing family zahrnuje:

- `GATHER`
- `GATHER_GRAD`
- `TAKE_ALONG_AXIS`
- `TAKE_ALONG_AXIS_GRAD`
- `SCATTER_ADD`

Hlavní části:

- [IndexExecutor.java](../backend/kernels/cpu/index/IndexExecutor.java)
- [IndexReadWriteBackend.java](../backend/kernels/cpu/index/IndexReadWriteBackend.java)

Tohle jsou hard barriers pro fused compute. Nejsou to jen "elementwise access variants".

### Layout

Layout family řeší:

- `RESHAPE`
- `EXPAND`
- `SELECT`
- `PERMUTE`
- `EXPAND_DIMS`
- `SQUEEZE`
- `CONTIGUOUS`

Hlavní část:

- [LayoutExecutor.java](../backend/kernels/cpu/layout/LayoutExecutor.java)

Důležitý kontrakt:

- některé layout ops jsou alias/view-only
- `CONTIGUOUS` je explicitní materialization node
- `RESHAPE` může být alias nebo materialization podle layout reality

## Fused Family

Fused runtime je dnes postavený takto:

- optimizer vytvoří `FusedOperation` descriptor
- `CompiledGraph.prepare(...)` z něj vytvoří `PreparedFusedExecutable`
- `CpuFusedKernel` je tenký entrypoint
- `FusedExecutor` řeší runtime scheduling

Relevantní třídy:

- [CpuFusedKernel.java](../backend/kernels/cpu/fused/CpuFusedKernel.java)
- [FusedExecutor.java](../backend/kernels/cpu/fused/FusedExecutor.java)
- [PreparedFusedExecutable.java](../graph/fused/PreparedFusedExecutable.java)
- [FusedExecutionBackendResolver.java](../graph/fused/FusedExecutionBackendResolver.java)

### Důležitá realita

CPU fused backend dnes nepoužívá starý direct fused backend.

`FusedExecutionBackendResolver` aktuálně:

- zkusí ASM fused backend
- pokud plán nepodporuje, vyhodí chybu

To znamená:

- prepared fused executable je dnes generované ASM path
- `applyRangeVector(...)` existuje jako kontrakt na interfacu
- default implementace na interfacu fallbackuje do scalar
- ale skutečný výkon dnes stojí na tom, co vygeneruje ASM backend

### Co řídí runtime fused dispatch

`FusedExecutor` používá prepared `ResolvedDispatchHints`:

- `SCALAR`
- `VECTOR`
- `PARALLEL`
- `PARALLEL_VECTOR`

a podle nich volá:

- `applyRangeScalar(...)`
- `applyRangeVector(...)`

Hot fused node tedy stále běží jako:

- jeden prepared executable
- jedna output-space loop structure
- bez mezimaterializace intermediate tensorů

## Compute Contract

CPU runtime rozlišuje:

- storage dtype
- compute dtype
- accumulate dtype
- resolved backend kind

To je zásadní hlavně pro `BFLOAT16`.

Příklady:

- `FLOAT64` storage -> compute `FLOAT64`
- `FLOAT32` storage -> compute `FLOAT32`
- `BFLOAT16` storage -> compute `FLOAT32`

Tenhle kontrakt se řeší v prepare fázi a ukládá se do `ResolvedCpuComputeContract`.

## BF16 Continuations And Workspace

Některé kernels umí držet mezivýsledky ve `float[]` workspace déle než jen do okamžiku public tensor materialization.

To se týká hlavně:

- `MATMUL`
- `LINEAR`
- `CONV2D_GEMM`
- některých reduction/layout kombinací, kde je explicitní float workspace

Je potřeba rozlišovat dvě věci:

- intra-kernel continuation
  - mezivýsledek zůstává ve workspace uvnitř jedné kernel family
- cross-node continuation
  - producer publikuje float continuation pro konkrétního podporovaného consumera

Cross-node continuation je záměrně úzká a explicitně plánovaná.

## Runtime Dispatch Knobs

CPU planner čte runtime knoby z `CpuKernelConfig` a příbuzných config objektů.

Typické knob families:

- elementwise dispatch
- fused dispatch
- reduction dispatch
- scheduler chunking
- materialization threshold
- matmul heuristics
- numerics approximation policy

Tuning surface je popsaná detailněji v:

- [../tuning/KNOBS.md](../tuning/KNOBS.md)

## BLAS Path

CPU matmul může volitelně přepnout na OpenBLAS přes FFM.

Relevantní runtime properties:

- `cg.cpu.blas.provider=NONE|OPENBLAS_FFM`
- `cg.cpu.blas.matmulMinWork=<long>`
- `cg.cpu.blas.f32RequireMgeK=true|false`
- `cg.cpu.blas.f32MaxNOverK=<double>`
- `cg.cpu.blas.threads=<int>`
- `cg.cpu.blas.debug=true|false`
- `openblas.lib=<absolute-path>`

V praxi:

- BLAS se používá jen pro vhodné contiguous workloady
- heuristika pro `F32` je přísnější než pro `F64`
- fallback zpět na Java backend je automatický

## Trace And Debug Metadata

`PreparedExecution.executeTraced(...)` umí vrátit step-level trace. Backend metadata se do něj promítají přes:

- compute metadata
- layout metadata
- dispatch metadata
- reduction metadata
- matmul metadata
- fused metadata

To je důležitý nástroj pro:

- ověření, že benchmark opravdu běžel na očekávané path
- audit `vectorWidth`, worker count, tiles, BLAS use
- hledání neočekávaných scalar fallbacků

## Adding A New CPU Kernel

Doporučený postup:

1. přidej operation descriptor nebo reuseuj existující `Operation.OpType`
2. vytvoř `Cpu*Kernel` entrypoint v odpovídající family
3. pokud jde o širší family, vytvoř nebo reuseuj `*Executor`
4. drž hot loops v leaf implementation, ne v resolveru
5. zaregistruj op v [CpuKernelResolver.java](../backend/registry/CpuKernelResolver.java)
6. doplň prepare-time hints nebo workspace, pokud je kernel potřebuje
7. přidej execution a regression testy
8. pokud vznikl nový runtime knob, propíchni ho do tuning surface a dokumentace

## Adding A New Backend

1. přidej enum value do `ComputeBackend`
2. implementuj concrete backend class
3. vytvoř registry/resolver pro nový backend
4. rozšiř `ComputeEngine.compute(...)`
5. doplň runtime config a tuning persistence, pokud backend přináší vlastní knoby
6. neporuš boundary:
   - graph stále skládá `Operation` deskriptory
   - backend stále jen exekuuje prepared graph

## Current Limitations

- CPU je jediný plně implementovaný backend
- CUDA/OpenCL jsou scaffold
- fused execution na CPU je dnes ASM-only prepare path
- některé performance optimalizace jsou úzce vázané na prepared metadata a nejsou obecný tensor runtime kontrakt
