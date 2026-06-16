# 117. cpu1 -> cpu Functional Parity Plan

## Stav Implementace

Status: `PLANNED`

Tento dokument je zivy implementacni checklist pro postupne dovedeni `backend.cpu1`
na funkcni paritu se starym `backend.cpu`, pri zachovani cistejsi cpu1 architektury.

Legenda:

- `[ ]` neimplementovano
- `[~]` rozpracovano
- `[x]` hotovo a overeno
- `[!]` zamerne neprebirat 1:1, vyzaduje jiny cpu1 design

Aktualni stav fazi:

- [x] Faze 0: parity inventory a ochrana pracovnich hranic
- [x] Faze 1: reduction runtime infrastruktura
- [~] Faze 2: reduction storage/layout/native parita
- [ ] Faze 3: loss vetev pro NLL a CrossEntropy
- [ ] Faze 4: index/gather/scatter operace
- [ ] Faze 5: NN a normalization kernely
- [ ] Faze 6: linear algebra a attention parita
- [ ] Faze 7: layout/view residualy a materializacni politika
- [ ] Faze 8: native storage, BF16 a mixed residency policy
- [ ] Faze 9: fused/codegen parity a benchmark evidence
- [ ] Faze 10: trace, tuning, coverage gate a default route readiness

## Cil

Cil neni prepsat `backend.cpu` do `backend.cpu1` 1:1. Cil je mit cpu1 jako novy
runtime, ktery umi spustit stejne uzivatelske vypocetni grafy jako stary CPU
backend, ale s hranici:

```text
graph/lowering:
  rozhoduje CO se bude pocitat a jake execution units vzniknou

cpu1 prepare:
  rozhoduje JAK presne to pobezi na CPU
  vybira storage, layout, kernel variantu, launch policy, scratch buffer

cpu1 execute:
  neplanuje
  nebrouzda v Tensor/autograd semantice
  pouze binduje runtime storage a vola prepared executable unit
```

Koncovy stav:

```text
backend.cpu1
  exec/
    Cpu1...ExecutableUnit
    Cpu1TensorView
    Cpu1ScratchBuffer
  prepare/
    Cpu1...Preparer
    Cpu1Prepared...Unit
    dispatch/
  kernels/
    elementwise/
    fused/
    dtype/
    layout/
    reduction/
    loss/
    index/
    matmul/
    linalg/
    nn/
  launch/
  storage/
  trace/
```

## Non-Goals

- Nezavadet compatibility vrstvu, ktera by z cpu1 volala stare `backend.cpu`
  kernely.
- Nevracet `Tensor`/autograd metadata do cpu1 hot path.
- Nezavadet docasne fallbacky bez viditelne trace informace.
- Neprebirat historicke balickove zarazeni ze stareho CPU, pokud je vec
  architektonicky jinde. Priklad: `CrossEntropyLoss` patri v cpu1 do `loss`,
  ne do obecne `reduction`.
- Necistit stary `backend.cpu` jako soucast tohoto planu.
- Nekomitovat lokalni benchmark/profilove artefakty.
- Nezapinat cpu1 jako default CPU backend, dokud neni hotovy coverage gate,
  parity test matrix a benchmark evidence.

## Dulezita Korekce K "Parite"

Stary `backend.cpu` nema prime runtime kernely pro legacy backward op typy:

- `MIN_GRAD`
- `MAX_GRAD`
- `REDUCE_MIN_GRAD`
- `REDUCE_MAX_GRAD`
- `SOFTMAX_GRAD`
- `LOG_SOFTMAX_GRAD`
- `CROSS_ENTROPY_LOSS_INDICES_GRAD`
- gather/scatter grad opy
- attention backward op

`CpuKernelRegistry` je explicitne odmita:

```java
case MIN_GRAD, MAX_GRAD, REDUCE_MIN_GRAD, REDUCE_MAX_GRAD,
        SOFTMAX_GRAD, LOG_SOFTMAX_GRAD,
        GATHER_GRAD, GATHER_AXIS_GRAD, GATHER_ND_GRAD, TAKE_ALONG_AXIS_GRAD,
        CROSS_ENTROPY_LOSS_INDICES_GRAD,
        SCALED_DOT_PRODUCT_ATTENTION_BACKWARD ->
        throw new IllegalStateException("CPU has no direct kernel for legacy backward op type " + type);
```

Proto tyto opy nejsou "chybejici cpu1 reduction kernely" v uzkem smyslu.
Jsou to graph/backward/lowering specializace a maji byt reseny oddelene.

## Soucasna Architektura: cpu vs cpu1

### Stary backend.cpu

Stary CPU backend je obecny storage-aware kernel runtime:

```text
CpuBackend
  -> CpuNodeExecutionArtifact / CpuFusedExecutionArtifact
  -> CpuKernelExecutor
  -> CpuKernelContext
  -> CpuKernelRegistry.resolve(opType)
  -> CpuKernel.execute(CpuKernelCall)
```

Silne stranky:

- siroka op coverage
- `CpuStorageView` umi array/MemorySegment/storage offset/strides
- native CPU materialization a memory pool existuji
- reductions/lossy maji strided traversal helpery
- index/layout/NN/linalg operace jsou implementacne dal
- fused ASM runtime je zralejsi z pohledu historicke parity

Slabiny:

- hot path casto taha pres obecny `CpuKernelContext`
- kernel si casto sam resi runtime rozhodovani
- mnoho storage/layout rozhodnuti je rozptyleno mezi prepare a kernel runtime
- balickove zarazeni je misty historicke, napriklad loss kernely jsou v
  `kernels/reduction`
- tezsi hranice mezi graph/lowering a backend execution

### Novy backend.cpu1

cpu1 je prepare-time runtime s rodinami prepared units:

```text
Cpu1NodePreparer
  -> Cpu1PreparedArtifact
     -> Cpu1PreparedElementwiseUnit
     -> Cpu1PreparedFusedElementwiseUnit
     -> Cpu1PreparedLayoutUnit
     -> Cpu1PreparedDTypeUnit
     -> Cpu1PreparedReductionUnit
     -> Cpu1PreparedMatmulUnit
     -> Cpu1PreparedMseLossUnit
  -> Cpu1...ExecutableUnit
  -> concrete kernel loop
```

Silne stranky:

- cistejsi hranice prepare/execute
- storage kind, layout kind, vectorization a launch config jsou prepare-time
  rozhodnuti
- elementwise ma generated kernel-id matrix
- fused cpu1 ma vlastni ASM/codegen-first smer
- `Cpu1ScratchBuffer` a `Cpu1ScratchBufferSpec` existuji pro prepared runtime
  scratch
- `Cpu1RangeLauncher` a `Cpu1LaunchPolicy` uz existuji
- dtype `CAST` uz je samostatna dtype vetev, ne layout hack

Slabiny:

- coverage je uzsi nez stary CPU
- reductions jsou funkcne siroke, ale implementacne zatim scalar/dense
- index/loss/NN casti nejsou dorovnane
- native `MemorySegment` parita neni systematicka ve vsech rodinach
- trace/coverage gate jeste neni centralni zdroj pravdy pro "cpu1 umi tento op"

## Parity Matrix Podle Rodin

### 1. Core Prepare/Execute Runtime

| Oblast | backend.cpu | backend.cpu1 | Stav cpu1 | Plan |
|---|---|---|---|---|
| Central registry | `CpuKernelRegistry.resolve(OpType)` | family-specific registries/preparers | lepsi architektura, ne kompletni coverage | zachovat cpu1 family dispatch |
| Runtime context | `CpuKernelContext` | `ExecutionContext` + prepared unit + tensor views | lepsi hranice | doplnit missing prepared units |
| Scratch | `CpuNodeWorkspace` | `Cpu1ScratchBuffer` | existuje, pouziva se ne vsude | sjednotit pro reductions/loss/NN |
| Threading | `CpuThreadPool`, hints | `Cpu1LaunchPolicy`, `Cpu1RangeLauncher` | existuje, ne vsude zapojeno | doplnit do reductions/loss/index |
| Trace | `CpuStepTraceContributor` | `Cpu1TraceContributor` | partial | rozsiruj pri kazde nove rodine |

### 2. Elementwise

Stary CPU podporuje:

- binary: `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, `POW_TENSOR`
- compare: `GT`, `GE`, `LT`, `LE`, `EQ`, `NE`
- logical: `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`
- unary: `NEG`, `INV`, `LOG`, `EXP`, `FAST_EXP`, `ERF`, `TANH`,
  `FAST_TANH`, `POW`, `SQRT`, `ABS`, `FLOOR`, `CEIL`, `SIGN`,
  `MUL_SCALAR`, `RELU`, `CLAMP_MIN`, `CLAMP_MAX`, `SIGMOID`
- ternary/select-like: `WHERE`

cpu1 stav:

- elementwise family existuje
- generated `Cpu1ElementwiseKernelId` pokryva `JAVA_ARRAY` i `MEMORY_SEGMENT`
  pro F32/F64/BF16 a broadcast/strided scalar + contiguous/broadcast vector
- `Cpu1DispatchPolicy` rozhoduje vector/scalar/threading z configu a tuning
- fused ASM cesta existuje a je cilovy smer pro delsi chainy

Zbyvajici parity prace:

- overit generated coverage testem proti `Operation` traits
- overit, ze vsechny elementwise ops maji stejna dtype pravidla jako stary CPU
- doplnit benchmark parity pro:
  - array scalar/vector/parallel
  - segment scalar/vector/parallel
  - broadcast inner
  - strided rank2/rank3/rank4/generic

Verdikt:

```text
Functional op coverage: skoro dorovnano
Runtime width: dobre, ale stale potrebuje coverage gate a benchmark matrix
```

### 3. Fused Elementwise

Stary CPU:

- ma fused IR
- ma ASM generator
- ma interpreted fallback
- umi vector/scalar variants
- pouziva stary `CpuKernelContext`

cpu1:

- ma vlastni fused IR: `backend.cpu1.fused.ir`
- ma vlastni codegen-first ASM route
- ma generated kernel cache/signature
- nema chteny interpreted fallback
- drzi se cpu1 prepared unit/executable unit modelu

Zbyvajici parity prace:

- doplnit benchmarky pro segment vector/parallel
- doplnit operation coverage matrix proti `Operation.isFusable()`
- zkontrolovat, ze vsechny scalar parametry jdou pres source `Operation`
  objekty, ne pres duplicitni trait/helper tabulky
- dokoncil-li se ASM vector path, udelat regression test na native segment
  end-to-end bez prevodu

Verdikt:

```text
Architektura cpu1 je cilova.
Neportovat stary fused runtime 1:1.
Dorovnavat pouze coverage, benchmarky a edge-case support.
```

### 4. DType / CAST

Stary CPU:

- ma `CpuCastKernel`
- historicky je v layout baliku
- umi storage-aware cast pres `CpuCastStorageLoops`
- native fast path je specializovana hlavne pro F32 <-> BF16

cpu1:

- ma samostatnou dtype vetev:
  - `Cpu1DTypePreparer`
  - `Cpu1PreparedDTypeUnit`
  - `Cpu1DTypeExecutableUnit`
  - `Cpu1DTypeKernelDispatch`
  - `Cpu1CastLoops`
- podporuje `FLOAT64/FLOAT32/BFLOAT16/INT32/INT64/BOOL`
- podporuje `JAVA_ARRAY` i `MEMORY_SEGMENT`
- podporuje contiguous i strided logical order
- pouziva `launchPolicy`, tedy scalar loops umi bezet paralelne po chunkach

Zbyvajici parity prace:

- doplnit explicitni parallel contract test pro `CAST`
- benchmarknout, jestli se vyplati vector specializace pro velke contiguous
  F32/F64 casty; bez benchmarku nezavadet

Verdikt:

```text
cpu1 je architektonicky cistsi nez stary CPU.
Funkcne je CAST v dobrem stavu.
```

### 5. Layout/View

Stary CPU podporuje:

- `NOOP`
- `CONTIGUOUS`
- `RESHAPE`
- `EXPAND`
- `SELECT`
- `SLICE`
- `PERMUTE`
- `EXPAND_DIMS`
- `SQUEEZE`
- `SLICE_BACKWARD`
- `CONCAT`
- `PAD`
- `TILE`
- `UNFOLD_AXIS`
- `UNFOLD2D`
- `FOLD2D`
- `CAST` historicky v layout baliku

cpu1 stav:

- alias ops existuji:
  - `NOOP_ALIAS`
  - `RESHAPE_ALIAS`
  - `EXPAND_ALIAS`
  - `SELECT_ALIAS`
  - `SLICE_ALIAS`
  - `PERMUTE_ALIAS`
  - `EXPAND_DIMS_ALIAS`
  - `SQUEEZE_ALIAS`
- materializacni/layout copy ops existuji:
  - contiguous scalar/vector
  - concat scalar/vector block specializace
  - pad scalar/vector block specializace
  - tile scalar/vector specializace
  - unfold axis/unfold2d
  - fold2d non-overlap/direct/copy
  - slice backward scalar
- `CAST` je spravne vyvedeny do dtype vetve

Zbyvajici parity prace:

- overit plnou shape/stride semantiku proti starym layout testum
- native segment parity pro layout copy ops
- explicitni materialization policy: kdy alias, kdy copy, kdy block-copy
- vector/parallel coverage gate pro vsechny layout kernels, kde to dava smysl
- `SLICE_BACKWARD` sjednotit napric CPU/cpu1/graph namingem

Verdikt:

```text
Op coverage je vysoka.
Nejvetsi risk je semantika edge cases a native segment/materialization policy.
```

### 6. Reductions

Stary CPU podporuje forward reduction ops:

- `SUM`
- `MEAN`
- `REDUCE_MIN`
- `REDUCE_MAX`
- `REDUCE_PROD`
- `REDUCE_ALL`
- `REDUCE_ANY`
- `ARGMAX`
- `CUMSUM`
- `SOFTMAX`
- `LOG_SOFTMAX`

cpu1 podporuje stejne forward op typy.

Rozdil neni primarne v seznamu opu, ale v sirce runtime implementace.

Stary CPU ma:

- `CpuStorageView`
- strided/storage-offset aware traversal
- array/segment/mixed support v helper vrstvach
- reduction hints pro paralelizaci ve vybranych cestach
- loss traversal helpery pro specialni loss kernels

cpu1 ma:

- `Cpu1PreparedReductionUnit`
- dense contiguous scalar loops
- omezeny segment support: dnes hlavne `SUM/MEAN F32/F64`
- zadny `launchConfig/launchPolicy` v prepared reduction unit
- `Cpu1ScratchBufferSpec.none()` ve vsech reduction prepare cestach

Zbyvajici parity prace:

- pridat launch config/policy do reduction prepared unit
- pridat scratch partial buffers
- parallel output-element reductions
- partial axis reductions pro scalar/low-output-count pripady
- native segment parity pro min/max/prod/all/any/argmax/cumsum/softmax
- strided/view reduction input support nebo explicitni materialization policy
- softmax/logSoftmax group parallel path

Verdikt:

```text
Op coverage je dorovnana.
Runtime width a vykon nejsou dorovnane.
```

### 7. Loss

Stary CPU ma forward loss kernels:

- `NLL_LOSS`
- `CROSS_ENTROPY_LOSS`
- `CROSS_ENTROPY_LOSS_INDICES`

Stary CPU je ma v `kernels/reduction`, ale architektonicky jsou to loss opy.

cpu1 ma:

- `Cpu1MseLossPreparer`
- `Cpu1PreparedMseLossUnit`
- `Cpu1MseLossExecutableUnit`
- `backend.cpu1.kernels.loss.mse`

cpu1 nema:

- `NLL_LOSS`
- `CROSS_ENTROPY_LOSS`
- `CROSS_ENTROPY_LOSS_INDICES`

Zbyvajici parity prace:

- vytvorit loss family pro NLL/CrossEntropy
- nepretezovat `Cpu1ReductionPreparer`
- podporovat distribution targets a index targets
- podporovat `LossReduction.NONE/SUM/MEAN`
- podporovat `ignoreIndex`
- doplnit native segment route az po array correctness

Verdikt:

```text
Nejvetsi chybejici forward op coverage po reductions.
Implementovat jako loss family, ne reduction family.
```

### 8. Matmul / Linear Algebra

Stary CPU:

- `MATMUL`
- `LINEAR`
- `SCALED_DOT_PRODUCT_ATTENTION`
- `SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS`
- Java loops
- OpenBLAS/native routes
- attention runtime cache

cpu1:

- `MATMUL` ma samostatnou prepared family
- Java scalar/vector routes existuji
- OpenBLAS array route existuje
- OpenBLAS native segment route existuje
- matmul post-op/epilogue zacal vznikat
- `LINEAR` neni dorovnany jako samostatny op
- attention ops nejsou dorovnane

Zbyvajici parity prace:

- rozhodnout, zda `LINEAR` lowerovat na `MATMUL + ADD/bias`, nebo mit
  specializovanou cpu1 linear family
- attention prenest az po stabilnim workspace/native/matmul provider layeru
- doplnit batched matmul edge cases proti staremu CPU
- sjednotit OpenBLAS thread knobs s tuning/runtime configem

Verdikt:

```text
MATMUL je v cpu1 velmi dulezity a relativne daleko.
LINEAR a attention jsou chybejici parity oblasti.
```

### 9. Index / Gather / Scatter

Stary CPU podporuje:

- `GATHER`
- `GATHER_AXIS`
- `GATHER_ND`
- `TAKE_ALONG_AXIS`
- `SCATTER_ADD`
- `SCATTER_AXIS_ADD`
- `SCATTER_ELEMENTS`
- `SCATTER_ND`

cpu1:

- nema samostatnou index family
- cast backward/layout scatter semantiky se resila pres `SLICE_BACKWARD`
- index gradients nejsou primy CPU kernel ani ve starem CPU registry

Zbyvajici parity prace:

- vytvorit `backend.cpu1.kernels.index`
- vytvorit `Cpu1IndexPreparer`
- podporovat first wave:
  - `GATHER`
  - `GATHER_AXIS`
  - `TAKE_ALONG_AXIS`
- second wave:
  - `GATHER_ND`
  - `SCATTER_ADD`
  - `SCATTER_AXIS_ADD`
- third wave:
  - `SCATTER_ELEMENTS`
  - `SCATTER_ND`
- explicitne resit duplicate index semantiku a determinismus

Verdikt:

```text
Velka missing funkcni oblast.
Implementovat jako samostatnou index family.
```

### 10. NN / Normalization

Stary CPU podporuje:

- `CONV2D`
- `MAX_POOL2D`
- `AVG_POOL2D`
- `LAYER_NORM`
- `RMS_NORM`

cpu1:

- zatim nema samostatnou `nn` family
- nektere normalized/loss-like patterny by mohly byt casem fused/specialized
- zadna prima conv/pool parity

Zbyvajici parity prace:

- nejdriv `LAYER_NORM` a `RMS_NORM`, protoze jsou blizko reduction +
  elementwise + workspace modelu
- potom pool2d
- conv2d az po jasnem rozhodnuti, jestli pouzit direct Java loops, im2col +
  matmul, nebo provider abstraction

Verdikt:

```text
Missing op coverage.
Implementovat po reductions/loss/index, protoze potrebuje stejny scratch,
layout a native policy zaklad.
```

### 11. Native CPU Storage

Stary CPU:

- ma `NativeCpuStorageFactory`
- memory pool/allocator
- materializer
- native layout helpers
- mixed array/segment fallback policy

cpu1:

- umi `Cpu1StorageKind.MEMORY_SEGMENT`
- elementwise a matmul segment cesty existuji
- dtype cast segment cesta existuje
- reductions segment cesta je omezena
- layout segment coverage neni systematicky dorovnana

Zbyvajici parity prace:

- centralizovat cpu1 native input/output binding policy
- rozhodnout, kdy native output storage reuse probiha pres runtime memory
  binder a kdy pres output allocation
- zamezit per-execute alokacim v hot path
- pridat trace atributy pro native reuse/copy-in/copy-out

Verdikt:

```text
cpu1 ma zaklad, ale potrebuje jednotny storage policy layer.
```

## Navazujici Implementacni Faze

Kazda faze musi byt samostatny commit nebo sada tematickych commitu.
Po kazde fazi se aktualizuje tento dokument.

---

## Faze 0: Parity Inventory A Coverage Gate

Status: `[x]`

### Proc

Nez zacneme doplnovat dalsi kernely, potrebujeme mit automaticky gate, ktery
rekne:

```text
Operation.OpType X:
  old CPU direct kernel: ano/ne
  cpu1 prepare route: ano/ne
  cpu1 tested: ano/ne
  cpu1 default eligible: ano/ne
```

Jinak budeme parity stav drzet v hlave a znovu se k nemu vracet.

### Tasky

- [x] Vytvorit test `Cpu1CpuParityInventoryTest`
- [x] Udelat explicitni mapu direct old CPU opu
- [x] Udelat explicitni mapu cpu1 family routes
- [x] Oddelit `LEGACY_BACKWARD_NO_DIRECT_CPU_KERNEL`
- [x] Oddelit `INTENTIONALLY_GRAPH_LOWERED`
- [x] Test nesmi vyzadovat, aby cpu1 hned vse umel; ma produkovat presny
  seznam missing parity oblasti

### Cilovy Kod

```java
package backend.cpu1;

import operations.Operation;
import org.junit.jupiter.api.Test;

import java.util.EnumSet;

import static org.junit.jupiter.api.Assertions.assertTrue;

final class Cpu1CpuParityInventoryTest {
    private static final EnumSet<Operation.OpType> OLD_CPU_DIRECT_KERNELS = EnumSet.of(
            Operation.OpType.ADD,
            Operation.OpType.SUB,
            Operation.OpType.MUL,
            Operation.OpType.DIV,
            Operation.OpType.MIN,
            Operation.OpType.MAX,
            Operation.OpType.GT,
            Operation.OpType.GE,
            Operation.OpType.LT,
            Operation.OpType.LE,
            Operation.OpType.EQ,
            Operation.OpType.NE,
            Operation.OpType.LOGICAL_AND,
            Operation.OpType.LOGICAL_OR,
            Operation.OpType.LOGICAL_NOT,
            Operation.OpType.GATHER,
            Operation.OpType.GATHER_AXIS,
            Operation.OpType.GATHER_ND,
            Operation.OpType.TAKE_ALONG_AXIS,
            Operation.OpType.SCATTER_ADD,
            Operation.OpType.SCATTER_AXIS_ADD,
            Operation.OpType.SCATTER_ELEMENTS,
            Operation.OpType.SCATTER_ND,
            Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION,
            Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS,
            Operation.OpType.LINEAR,
            Operation.OpType.CONV2D,
            Operation.OpType.MAX_POOL2D,
            Operation.OpType.AVG_POOL2D,
            Operation.OpType.LAYER_NORM,
            Operation.OpType.RMS_NORM,
            Operation.OpType.SUM,
            Operation.OpType.MEAN,
            Operation.OpType.REDUCE_MIN,
            Operation.OpType.REDUCE_MAX,
            Operation.OpType.REDUCE_PROD,
            Operation.OpType.CUMSUM,
            Operation.OpType.ARGMAX,
            Operation.OpType.REDUCE_ALL,
            Operation.OpType.REDUCE_ANY,
            Operation.OpType.SOFTMAX,
            Operation.OpType.LOG_SOFTMAX,
            Operation.OpType.NLL_LOSS,
            Operation.OpType.CROSS_ENTROPY_LOSS,
            Operation.OpType.CROSS_ENTROPY_LOSS_INDICES,
            Operation.OpType.MATMUL,
            Operation.OpType.NEG,
            Operation.OpType.INV,
            Operation.OpType.LOG,
            Operation.OpType.EXP,
            Operation.OpType.FAST_EXP,
            Operation.OpType.ERF,
            Operation.OpType.TANH,
            Operation.OpType.FAST_TANH,
            Operation.OpType.POW,
            Operation.OpType.POW_TENSOR,
            Operation.OpType.SQRT,
            Operation.OpType.ABS,
            Operation.OpType.FLOOR,
            Operation.OpType.CEIL,
            Operation.OpType.SIGN,
            Operation.OpType.MUL_SCALAR,
            Operation.OpType.RELU,
            Operation.OpType.CLAMP_MIN,
            Operation.OpType.CLAMP_MAX,
            Operation.OpType.SIGMOID,
            Operation.OpType.WHERE,
            Operation.OpType.CONTIGUOUS,
            Operation.OpType.RESHAPE,
            Operation.OpType.EXPAND,
            Operation.OpType.SELECT,
            Operation.OpType.SLICE,
            Operation.OpType.PERMUTE,
            Operation.OpType.EXPAND_DIMS,
            Operation.OpType.SQUEEZE,
            Operation.OpType.SLICE_BACKWARD,
            Operation.OpType.CONCAT,
            Operation.OpType.PAD,
            Operation.OpType.TILE,
            Operation.OpType.UNFOLD_AXIS,
            Operation.OpType.UNFOLD2D,
            Operation.OpType.FOLD2D,
            Operation.OpType.CAST,
            Operation.OpType.NOOP,
            Operation.OpType.FUSED
    );

    private static final EnumSet<Operation.OpType> CPU1_INTENTIONALLY_NOT_DIRECT = EnumSet.of(
            Operation.OpType.FUSED
    );

    @Test
    void inventoryIsExplicit() {
        assertTrue(OLD_CPU_DIRECT_KERNELS.contains(Operation.OpType.CROSS_ENTROPY_LOSS_INDICES));
        assertTrue(CPU1_INTENTIONALLY_NOT_DIRECT.contains(Operation.OpType.FUSED));
    }
}
```

### Overeni

```bash
./gradlew test --tests backend.cpu1.Cpu1CpuParityInventoryTest
```

---

## Faze 1: Reduction Runtime Infrastruktura

Status: `[x]`

### Proc

cpu1 reduction op typy uz existuji, ale runtime neni dorovnany:

- prepared unit nema `launchConfig`
- prepared unit nema `launchPolicy`
- scratch spec je vzdy `none`
- kernels bezi single-thread loops

Bez teto faze nelze ciste pridat parallel reductions, partial sums ani segment
paritu.

### Dnesni Tok

```java
Cpu1PreparedReductionUnit unit = new Cpu1PreparedReductionUnit(
        node.id(),
        node.inputIds().getFirst(),
        opType,
        node.dataType(),
        config.storageKind(),
        kernelId(...),
        axis,
        axisSize,
        innerSize,
        outerSize,
        node.flatDataSize(),
        keepDims,
        argMaxLastIndexWins(...),
        cumSumExclusive(...),
        cumSumReverse(...),
        Cpu1ScratchBufferSpec.none()
);
```

Kernel pak dela:

```java
for (int outer = 0; outer < unit.outerSize(); outer++) {
    for (int inner = 0; inner < unit.innerSize(); inner++) {
        float sum = 0.0f;
        for (int index = 0; index < unit.axisSize(); index++) {
            sum += inputArray[inputOuterBase + index * unit.innerSize() + inner];
        }
        outputArray[outputOuterBase + inner] = sum;
    }
}
```

To je jednoduche, ale single-thread.

### Cilovy Model

```java
public final class Cpu1PreparedReductionUnit {
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final Cpu1ScratchBufferSpec scratchBufferSpec;

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    public Cpu1LaunchPolicy launchPolicy() {
        return launchPolicy;
    }
}
```

Preparer nesmi brat thread count z `Runtime.getRuntime()`. Zdroj pravdy je:

```java
config.launchConfig()
```

### Tasky

- [x] Rozsirit `Cpu1PreparedReductionUnit` o `launchConfig`
- [x] Rozsirit `Cpu1PreparedReductionUnit` o `launchPolicy`
- [x] Upravit konstruktor a vsechny test fixtures
- [x] Do `Cpu1ReductionPreparer` pridat helper:

```java
private static Cpu1LaunchPolicy launchPolicy(Cpu1LaunchConfig launchConfig) {
    if (launchConfig.workerCount() == 1) {
        return new Cpu1SingleThreadLaunch(launchConfig);
    }
    return new Cpu1ParallelLaunch(launchConfig);
}
```

- [x] Trace doplnit o:
  - `cpu1ReductionLaunchWorkers`
  - `cpu1ReductionLaunchChunkSize`
  - `cpu1ReductionScratchF32`
  - `cpu1ReductionScratchF64`
  - `cpu1ReductionScratchI32`
- [x] Pridat test, ze parallel config se propsal do prepared unit

### Cilovy Test

```java
@Test
void reductionPrepareCarriesLaunchConfig() {
    Tensor x = new Tensor(new float[] {1, 2, 3, 4}, new int[] {2, 2}, null, "x", DataType.FLOAT32);
    Tensor y = x.sum(1);
    Fixture fixture = fixture(y);

    Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
            fixture.node(),
            fixture.descriptorIndex(),
            Cpu1PrepareConfig.vectorParallel(4)
    );

    Cpu1PreparedReductionUnit unit = artifact.preparedReductionUnit();
    assertEquals(4, unit.launchConfig().workerCount());
    assertInstanceOf(Cpu1ParallelLaunch.class, unit.launchPolicy());
}
```

### Overeni

```bash
./gradlew test --tests backend.cpu1.Cpu1ReductionExecutionContractTest
```

---

## Faze 2: Parallel SUM/MEAN A Partial Workspace

Status: `[~]`

### Proc

`SUM/MEAN` jsou nejjednodussi reductions a zaroven caste v loss/normalization.
Na nich se ma zavest obecny pattern:

1. parallel pres output work items, kdyz je vystupnich prvku hodne
2. partial axis reduction, kdyz je vystupnich prvku malo
3. scratch buffer bez per-execute alokaci

### Dva Typy Paralelizace

Priklad 1: reduce `[1024, 1000]` pres axis 1.

```text
outerSize = 1024
axisSize = 1000
innerSize = 1
outputWorkItems = 1024
```

Tady paralelizujeme vystupni prvky:

```text
thread 0 -> samples 0..255
thread 1 -> samples 256..511
thread 2 -> samples 512..767
thread 3 -> samples 768..1023
```

Priklad 2: reduce `[10_000_000]` do scalaru.

```text
outerSize = 1
axisSize = 10_000_000
innerSize = 1
outputWorkItems = 1
```

Tady vystupni paralelizace nepomuze. Musime delat partial sums:

```text
thread 0 -> input 0..2_499_999 -> partial[0]
thread 1 -> input 2_500_000..4_999_999 -> partial[1]
thread 2 -> input 5_000_000..7_499_999 -> partial[2]
thread 3 -> input 7_500_000..9_999_999 -> partial[3]
main     -> sum(partial)
```

### Cilovy Scratch Spec

Prepare spocte pocet slotu:

```java
private static Cpu1ScratchBufferSpec scratchSpec(
        Operation.OpType opType,
        DataType dataType,
        Cpu1LaunchConfig launchConfig,
        int outputWorkItems,
        int axisSize
) {
    if ((opType == Operation.OpType.SUM || opType == Operation.OpType.MEAN)
            && launchConfig.workerCount() > 1
            && outputWorkItems <= 1
            && axisSize >= launchConfig.workerCount()) {
        int slots = Cpu1RangeLauncher.slotCount(axisSize, launchConfig);
        return Cpu1ScratchBufferSpec.arrays(0, slots, 0);
    }
    return Cpu1ScratchBufferSpec.none();
}
```

Proc `double[]` i pro F32/BF16:

- sum akumulace je numericky stabilnejsi
- BF16 se stejne pocita pres F32/F64 akumulator a az na konci materializuje
  do BF16

### Cilovy Kernel Skeleton

```java
private static void reduceF32(Cpu1PreparedReductionUnit unit, ExecutionContext context, boolean mean) {
    Cpu1TensorView input = inputArrayView(unit, context);
    Cpu1TensorView output = outputArrayView(unit, context);
    float[] in = input.float32Array();
    float[] out = output.float32Array();

    int outputWorkItems = Math.multiplyExact(unit.outerSize(), unit.innerSize());
    if (unit.launchConfig().workerCount() > 1 && outputWorkItems > 1) {
        reduceF32OutputParallel(unit, in, out, input.storageOffset(), output.storageOffset(), mean);
    } else if (unit.launchConfig().workerCount() > 1 && outputWorkItems == 1) {
        reduceF32AxisPartial(unit, context, in, out, input.storageOffset(), output.storageOffset(), mean);
    } else {
        reduceF32SingleThread(unit, in, out, input.storageOffset(), output.storageOffset(), mean);
    }

    markOutputWritten(unit, output, context);
}
```

Output-parallel path:

```java
private static void reduceF32OutputParallel(
        Cpu1PreparedReductionUnit unit,
        float[] input,
        float[] output,
        int inputBase,
        int outputBase,
        boolean mean
) {
    int outputWorkItems = unit.outerSize() * unit.innerSize();
    unit.launchPolicy().launch(outputWorkItems, (start, end) -> {
        for (int work = start; work < end; work++) {
            int outer = work / unit.innerSize();
            int inner = work - outer * unit.innerSize();
            int inputOuterBase = inputBase + outer * unit.axisSize() * unit.innerSize();
            int outputOffset = outputBase + outer * unit.innerSize() + inner;

            double sum = 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                sum += input[inputOuterBase + index * unit.innerSize() + inner];
            }
            output[outputOffset] = (float) (mean ? sum / unit.axisSize() : sum);
        }
    });
}
```

Partial axis path:

```java
private static void reduceF32AxisPartial(
        Cpu1PreparedReductionUnit unit,
        ExecutionContext context,
        float[] input,
        float[] output,
        int inputBase,
        int outputBase,
        boolean mean
) {
    Cpu1ScratchBuffer scratch = context.requireWorkspace(unit.nodeId(), Cpu1ScratchBuffer.class);
    double[] partial = scratch.requireF64Array(Cpu1RangeLauncher.slotCount(unit.axisSize(), unit.launchConfig()));

    Cpu1RangeLauncher.launchIndexed(unit.axisSize(), unit.launchConfig(), (slot, start, end) -> {
        double sum = 0.0d;
        for (int index = start; index < end; index++) {
            sum += input[inputBase + index];
        }
        partial[slot] = sum;
    });

    double total = 0.0d;
    int slots = Cpu1RangeLauncher.slotCount(unit.axisSize(), unit.launchConfig());
    for (int slot = 0; slot < slots; slot++) {
        total += partial[slot];
    }
    output[outputBase] = (float) (mean ? total / unit.axisSize() : total);
}
```

Poznamka: presny API pro `context.requireWorkspace(...)` je nutne overit podle
existujicich `Cpu1ScratchBuffer` consumeru. Pokud dnes chybi type-safe accessor,
pridat minimalni helper v existujicim stylu, ne obecnou novou fasadu.

### Tasky

- [x] Pridat scratch spec pro partial SUM/MEAN
- [x] Implementovat F32 array output-parallel SUM/MEAN
- [x] Implementovat F64 array output-parallel SUM/MEAN
- [x] Implementovat BF16 array output-parallel SUM/MEAN s F32/F64 accumulator
- [x] Implementovat F32/F64/BF16 partial scalar-output path
- [x] Test: large outputWorkItems pouzije parallel launch
- [x] Test: scalar output pouzije scratch partials
- [ ] Benchmark: scalar large reduction single vs parallel

### Overeni

```bash
./gradlew test --tests backend.cpu1.Cpu1ReductionExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1ScratchBufferTest
```

---

## Faze 3: Reduction Native Segment A Strided/View Input Policy

Status: `[~]`

### Proc

Stary CPU umi diky `CpuStorageView` mnohem vice kombinaci:

- array
- MemorySegment
- storage offset
- strides
- mixed input/output storage

cpu1 reductions jsou dnes dense contiguous only a segment cesta je omezena.

### Rozhodnuti: Sdileny Access Plan, Potom Kernels

Musime si rict, co se stane, kdyz reduction input neni dense contiguous.

Moznosti:

1. Materializovat contiguous pred reduction
2. Pouzit strided reduction kernel
3. Odmitnout route s jasnou trace/reason

Prvni krok je sdileny prepare-time policy typ v `backend.cpu1.storage`:

```java
Cpu1StorageAccessPlan.fromDescriptor(input)
Cpu1StorageAccessPlan.fromNode(output)
```

`Cpu1StorageAccessPlan` klasifikuje jen compile metadata:

- `DENSE_CONTIGUOUS`
- `DENSE_WITH_OFFSET`
- `STRIDED`
- `BROADCAST`
- `UNSUPPORTED`

Plan uklada kind, shape, strides, storageOffset, elementCount a volitelny
rejectionReason. Neobsahuje `Tensor`, runtime data array, `MemorySegment`,
typed read/write gettery ani obecny runtime access framework.

Proc to nekopiruje stary `CpuStorageView` 1:1:

- stary `CpuStorageView` micha storage handle, layout a runtime pristup do
  kernelu
- cpu1 chce mit storage/layout rozhodnuti v prepare vrstve a runtime view
  ponechat jen na bindovane buffery
- tento krok pouze sjednocuje klasifikaci; nedela materializaci, obecne
  read/write helpery ani strided reduction kernels

Aktualni reduction preparer smi dal odmitat vse mimo `DENSE_CONTIGUOUS`, ale
chyba musi uvadet access kind/reason. Strided/broadcast/offset kernels jsou
nasledujici krok.

Update: `Cpu1StorageAccessPlan` je sdileny prepare-time klasifikator pro
reduction contract checky, elementwise layout/input planning a fused external
input planning. Broadcastovana logical-shape klasifikace zustava metadata-only:
nepridava `Tensor`, storage handle, typed accessors ani obecne runtime
read/write helpery.

### Segment Kernel Skeleton

```java
private static void reduceF32SegmentOutputParallel(
        Cpu1PreparedReductionUnit unit,
        MemorySegment input,
        MemorySegment output,
        int inputBase,
        int outputBase,
        boolean mean
) {
    int outputWorkItems = unit.outerSize() * unit.innerSize();
    unit.launchPolicy().launch(outputWorkItems, (start, end) -> {
        for (int work = start; work < end; work++) {
            int outer = work / unit.innerSize();
            int inner = work - outer * unit.innerSize();
            int inputOuterBase = inputBase + outer * unit.axisSize() * unit.innerSize();
            int outputOffset = outputBase + outer * unit.innerSize() + inner;

            double sum = 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                int inputOffset = inputOuterBase + index * unit.innerSize() + inner;
                sum += input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
            }
            output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES,
                    (float) (mean ? sum / unit.axisSize() : sum));
        }
    });
}
```

### Tasky

- [x] Pridat `Cpu1StorageAccessKind`
- [x] Pridat `Cpu1StorageAccessPlan` pro descriptor/node metadata
- [x] Pokryt dense, offset, strided, broadcast a defensive-copy pripady testy
- [x] Zapojit access plan do `Cpu1ReductionPreparer` contract checku
- [x] Pouzit common access plan pro elementwise a fused input planning
- [~] Navrhnout dalsi segment/strided policy testy podle noveho access planu
- [ ] Rozsirit segment support pro `SUM/MEAN BF16`
- [ ] Rozsirit segment support pro `MIN/MAX/PROD F32/F64/BF16`
- [ ] Rozsirit segment support pro `ALL/ANY BOOL`
- [ ] Rozsirit segment support pro `ARGMAX F32/F64/BF16/I32/I64 -> I64`
- [ ] Rozsirit segment support pro `CUMSUM F32/F64/BF16/I32/I64`
- [ ] Pridat dense segment contract tests
- [ ] Pridat prvni strided direct kernel jen pro `SUM/MEAN F32/F64`

### Overeni

```bash
./gradlew test --tests backend.cpu1.Cpu1StorageAccessPlanTest
./gradlew test --tests backend.cpu1.Cpu1ReductionExecutionContractTest
```

---

## Faze 4: Softmax / LogSoftmax Runtime Width

Status: `[ ]`

### Proc

`SOFTMAX` a `LOG_SOFTMAX` jsou jednovstupove, ale nejsou klasicke scalar
reductions. Vystup ma stejny shape jako vstup.

Dnes cpu1 dela pro kazdou skupinu:

1. najdi max
2. secti exp(x - max)
3. zapis output

To je spravne, ale scalar single-thread.

### Spravna Paralelizace

Paralelizovat pres nezavisle skupiny:

```text
groupCount = outerSize * innerSize
```

Priklad `[batch, classes]`, axis je `classes`:

```text
groupCount = batch
```

Threading:

```java
unit.launchPolicy().launch(groupCount, (startGroup, endGroup) -> {
    for (int group = startGroup; group < endGroup; group++) {
        computeOneSoftmaxGroup(group);
    }
});
```

### Cilovy Skeleton

```java
private static void computeF32Parallel(
        float[] input,
        float[] output,
        int inputBase,
        int outputBase,
        Cpu1PreparedReductionUnit unit,
        boolean log
) {
    int groupCount = unit.outerSize() * unit.innerSize();
    unit.launchPolicy().launch(groupCount, (start, end) -> {
        for (int group = start; group < end; group++) {
            int outer = group / unit.innerSize();
            int inner = group - outer * unit.innerSize();
            int inputOuterBase = inputBase + outer * unit.axisSize() * unit.innerSize();
            int outputOuterBase = outputBase + outer * unit.axisSize() * unit.innerSize();

            double max = Double.NEGATIVE_INFINITY;
            for (int index = 0; index < unit.axisSize(); index++) {
                max = Math.max(max, input[inputOuterBase + index * unit.innerSize() + inner]);
            }

            double sum = 0.0d;
            for (int index = 0; index < unit.axisSize(); index++) {
                sum += Math.exp(input[inputOuterBase + index * unit.innerSize() + inner] - max);
            }

            double logDenominator = Math.log(sum);
            for (int index = 0; index < unit.axisSize(); index++) {
                int inOffset = inputOuterBase + index * unit.innerSize() + inner;
                int outOffset = outputOuterBase + index * unit.innerSize() + inner;
                double shifted = input[inOffset] - max;
                output[outOffset] = log
                        ? (float) (shifted - logDenominator)
                        : (float) (Math.exp(shifted) / sum);
            }
        }
    });
}
```

### Tasky

- [ ] F32/F64/BF16 array parallel group path
- [ ] F32/F64/BF16 segment group path
- [ ] Threshold z config/tuningu pro kdy parallel zapnout
- [ ] Benchmark `batch x classes`: 1k, 10k, 100k groups
- [ ] Parity test proti starym `SoftmaxExecutionTest` a `LogSoftmaxExecutionTest`

---

## Faze 5: Loss Family - NLL A CrossEntropy

Status: `[ ]`

### Proc

`NLL_LOSS`, `CROSS_ENTROPY_LOSS` a `CROSS_ENTROPY_LOSS_INDICES` nejsou obecne
reductions, i kdyz uvnitr redukuji pres class axis. Jsou to loss operace s
vlastni semantikou vstupu, shape pravidly a reduction modem.

Proto je cil:

```text
backend.cpu1.prepare.Cpu1LossPreparer
backend.cpu1.prepare.Cpu1PreparedLossUnit
backend.cpu1.exec.Cpu1LossExecutableUnit
backend.cpu1.kernels.loss.crossentropy
backend.cpu1.kernels.loss.nll
```

Nemame pretezovat `Cpu1ReductionPreparer`.

### Prvni Wave

Nejprve `CROSS_ENTROPY_LOSS_INDICES`, protoze je prakticky nejdulezitejsi:

```java
Tensor logits = model.forward(input);       // [batch, classes]
Tensor targets = ...;                       // [batch] INT32/INT64
Tensor loss = logits.crossEntropy(targets); // scalar nebo per-sample
```

Semantika:

```text
loss(sample) = -log_softmax(logits[sample])[targetClass]
```

Stabilni vypocet:

```text
max = max(logits)
logSumExp = log(sum(exp(logits - max))) + max
loss = logSumExp - logits[targetClass]
```

### Prepared Unit

```java
public final class Cpu1PreparedCrossEntropyLossUnit {
    private final int nodeId;
    private final int logitsNodeId;
    private final int targetsNodeId;
    private final Operation.OpType opType;
    private final DataType logitsDataType;
    private final DataType targetDataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1CrossEntropyKernelId kernelId;
    private final Cpu1CrossEntropyKernel kernel;
    private final int classAxis;
    private final int axisSize;
    private final int groupCount;
    private final int[] logitsShape;
    private final int[] targetShape;
    private final tensor.loss.LossReduction reduction;
    private final Integer ignoreIndex;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final Cpu1ScratchBufferSpec scratchBufferSpec;
}
```

### Preparer Route

`Cpu1NodePreparer`:

```java
if (Cpu1LossPreparer.isLossOp(opType)) {
    return new Cpu1LossPreparer().prepare(node, descriptorIndex, config);
}
```

`Cpu1LossPreparer`:

```java
public static boolean isLossOp(Operation.OpType opType) {
    return opType == Operation.OpType.NLL_LOSS
            || opType == Operation.OpType.CROSS_ENTROPY_LOSS
            || opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES;
}
```

### Kernel Skeleton: Index Targets F32 Array

```java
private static void crossEntropyIndicesF32Array(
        Cpu1PreparedCrossEntropyLossUnit unit,
        Cpu1TensorView logits,
        Cpu1TensorView targets,
        Cpu1TensorView output,
        ExecutionContext context
) {
    float[] logitsArray = logits.float32Array();
    int[] targetArray = targets.int32Array();
    float[] outputArray = output.float32Array();

    if (unit.reduction() == LossReduction.NONE) {
        unit.launchPolicy().launch(unit.groupCount(), (start, end) -> {
            for (int group = start; group < end; group++) {
                SampleLoss sample = computeSampleF32(
                        logitsArray,
                        logits.storageOffset(),
                        targetArray[targets.storageOffset() + group],
                        group,
                        unit
                );
                outputArray[output.storageOffset() + group] = sample.valid()
                        ? (float) sample.loss()
                        : 0.0f;
            }
        });
        return;
    }

    // SUM/MEAN uses partial workspace.
}
```

Partial reduction:

```java
double[] partialLoss = scratch.requireF64Array(slotCount);
int[] partialValid = scratch.requireI32Array(slotCount);

Cpu1RangeLauncher.launchIndexed(unit.groupCount(), unit.launchConfig(), (slot, start, end) -> {
    double localLoss = 0.0d;
    int localValid = 0;
    for (int group = start; group < end; group++) {
        SampleLoss sample = computeSampleF32(...);
        if (sample.valid()) {
            localLoss += sample.loss();
            localValid++;
        }
    }
    partialLoss[slot] = localLoss;
    partialValid[slot] = localValid;
});

double total = 0.0d;
int valid = 0;
for (int slot = 0; slot < slotCount; slot++) {
    total += partialLoss[slot];
    valid += partialValid[slot];
}
output[0] = reduction == SUM ? total : total / valid;
```

### Tasky

- [ ] Vytvorit `Cpu1LossPreparer`
- [ ] Vytvorit `Cpu1PreparedCrossEntropyLossUnit`
- [ ] Vytvorit `Cpu1LossExecutableUnit`
- [ ] Vytvorit `Cpu1CrossEntropyKernelId`
- [ ] Vytvorit `Cpu1CrossEntropyKernelDispatch`
- [ ] Implementovat `CROSS_ENTROPY_LOSS_INDICES F32 ARRAY`
- [ ] Implementovat `CROSS_ENTROPY_LOSS_INDICES F64 ARRAY`
- [ ] Implementovat `CROSS_ENTROPY_LOSS_INDICES BF16 ARRAY`
- [ ] Podporovat INT32 a INT64 target indices
- [ ] Podporovat `LossReduction.NONE`
- [ ] Podporovat `LossReduction.SUM`
- [ ] Podporovat `LossReduction.MEAN`
- [ ] Podporovat `ignoreIndex`
- [ ] Pridat trace metadata
- [ ] Pridat tests podle `IndexTargetCrossEntropyLossExecutionTest`

### Druha Wave

- [ ] `NLL_LOSS` dense target distribution
- [ ] `CROSS_ENTROPY_LOSS` dense target distribution
- [ ] native segment cesty
- [ ] strided logits/targets policy

### Overeni

```bash
./gradlew test --tests IndexTargetCrossEntropyLossExecutionTest
./gradlew test --tests IgnoreIndexLossExecutionTest
./gradlew test --tests backend.cpu1.Cpu1CrossEntropyLossExecutionContractTest
```

---

## Faze 6: Index / Gather / Scatter Family

Status: `[ ]`

### Proc

Index operace jsou velka funkcni mezera. Nepatri do elementwise ani layout,
protoze cteni/zapis je urcovan index tensorem a casto ma akumulacni semantiku.

### Cilova Struktura

```text
backend.cpu1.prepare
  Cpu1IndexPreparer
  Cpu1PreparedIndexUnit

backend.cpu1.exec
  Cpu1IndexExecutableUnit

backend.cpu1.kernels.index
  Cpu1IndexKernel
  Cpu1IndexKernelId
  Cpu1IndexKernelDispatch
  gather/
    Cpu1GatherLoops
  gatheraxis/
    Cpu1GatherAxisLoops
  gathernd/
    Cpu1GatherNdLoops
  scatter/
    Cpu1ScatterLoops
```

### Prepared Unit

```java
public final class Cpu1PreparedIndexUnit {
    private final int nodeId;
    private final List<Integer> inputNodeIds;
    private final Operation.OpType opType;
    private final DataType outputDataType;
    private final DataType indexDataType;
    private final Cpu1StorageKind storageKind;
    private final Cpu1IndexKernelId kernelId;
    private final Cpu1IndexKernel kernel;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final int[] outputShape;
    private final int[] sourceShape;
    private final int axis;
}
```

### Wave 1

- [ ] `GATHER`
- [ ] `GATHER_AXIS`
- [ ] `TAKE_ALONG_AXIS`

Proc prvni:

- jsou read-only z pohledu source
- nemaji duplicate write race
- dobre se paralelizuji pres output elements

### Wave 2

- [ ] `GATHER_ND`
- [ ] `SCATTER_ADD`
- [ ] `SCATTER_AXIS_ADD`

### Wave 3

- [ ] `SCATTER_ELEMENTS`
- [ ] `SCATTER_ND`

### Race Semantics

Scatter operace mohou mit duplicate indices. Musime explicitne rozhodnout:

```text
SCATTER_ADD:
  duplicate index -> suma vsech prispevku
  parallel path musi byt deterministicky nebo trace musi rict, ze je non-deterministic

SCATTER_ELEMENTS/SCATTER_ND:
  duplicate index -> podle existujici stare CPU semantiky
```

Pro zacatek:

```text
scatter ops single-thread scalar first
gather ops parallel output range allowed
```

### Overeni

```bash
./gradlew test --tests ScatterAddExecutionTest
./gradlew test --tests Int32IndexDtypeTest
```

---

## Faze 7: Linear, NN A Normalization

Status: `[ ]`

### Proc

Po reductions/loss/index budou chybet hlavne higher-level numeric kernels:

- `LINEAR`
- `CONV2D`
- `MAX_POOL2D`
- `AVG_POOL2D`
- `LAYER_NORM`
- `RMS_NORM`

### Poradi

1. `LAYER_NORM`
2. `RMS_NORM`
3. `LINEAR`
4. `MAX_POOL2D`
5. `AVG_POOL2D`
6. `CONV2D`

Proc takto:

- norm opy vyuziji reduction + elementwise + scratch infrastrukturu
- linear muze byt lowering na matmul + bias, nebo specializovany provider
- pool2d je layout/index heavy, ale jednodussi nez conv2d
- conv2d ma nejvic route rozhodnuti

### LayerNorm Prepared Unit Skeleton

```java
public final class Cpu1PreparedLayerNormUnit {
    private final int nodeId;
    private final int inputNodeId;
    private final int gammaNodeId;
    private final int betaNodeId;
    private final DataType dataType;
    private final int normalizedSize;
    private final int groupCount;
    private final float epsilon;
    private final Cpu1StorageKind storageKind;
    private final Cpu1LaunchConfig launchConfig;
    private final Cpu1LaunchPolicy launchPolicy;
    private final Cpu1ScratchBufferSpec scratchBufferSpec;
}
```

### LayerNorm Loop Skeleton

```java
unit.launchPolicy().launch(unit.groupCount(), (start, end) -> {
    for (int group = start; group < end; group++) {
        int base = inputBase + group * unit.normalizedSize();

        double sum = 0.0d;
        for (int i = 0; i < unit.normalizedSize(); i++) {
            sum += input[base + i];
        }
        double mean = sum / unit.normalizedSize();

        double varianceSum = 0.0d;
        for (int i = 0; i < unit.normalizedSize(); i++) {
            double d = input[base + i] - mean;
            varianceSum += d * d;
        }
        double invStd = 1.0d / Math.sqrt(varianceSum / unit.normalizedSize() + unit.epsilon());

        for (int i = 0; i < unit.normalizedSize(); i++) {
            output[base + i] = (float) ((input[base + i] - mean) * invStd * gamma[i] + beta[i]);
        }
    }
});
```

### Conv2D Rozhodnuti

Moznosti:

1. direct Java loops jako stary CPU
2. im2col + matmul
3. provider abstraction pro native knihovny

Prvni implementace:

```text
direct Java scalar correctness first
parallel over output batches/channels/spatial tiles
native/provider route later
```

### Overeni

```bash
./gradlew test --tests '*LayerNorm*'
./gradlew test --tests '*RmsNorm*'
./gradlew test --tests '*Pool2d*'
./gradlew test --tests '*Conv2d*'
```

---

## Faze 8: Linear Algebra A Attention

Status: `[ ]`

### Proc

`MATMUL` je cpu1 uz daleko, ale stary CPU ma jeste:

- `LINEAR`
- `SCALED_DOT_PRODUCT_ATTENTION`
- `SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS`

### LINEAR

Preferovane reseni:

```text
graph/lowering:
  LINEAR(input, weight, bias)
    -> MATMUL(input, weight.T)
    -> ADD bias epilogue nebo broadcast add
```

Pokud lowering neni vzdy mozne, udelat cpu1 `linear` family jako tenkou
specializaci nad matmul providerem.

Skeleton:

```java
public final class Cpu1LinearPreparer {
    public Cpu1PreparedArtifact prepare(CompiledNode node, CompiledTensorDescriptorIndex descriptors, Cpu1PrepareConfig config) {
        // Validate input/weight/bias.
        // Reuse Cpu1MatmulPreparer route selection.
        // Select bias post-op if supported.
        // Otherwise require graph lowering to materialize ADD.
    }
}
```

### Attention

Attention je slozitejsi:

```text
scores = Q @ K^T / sqrt(d)
weights = softmax(scores + mask)
out = weights @ V
```

cpu1 by nemelo hned delat obrovsky monolit, dokud neni stabilni:

- matmul provider
- softmax group parallel
- workspace/reuse
- native storage route

Prvni cpu1 attention plan:

1. lowering na existujici primitives
2. potom specializovany attention weights kernel
3. potom full SDPA kernel s workspace

### Tasky

- [ ] Zmapovat, kdy graph ponechava `LINEAR` jako op
- [ ] Rozhodnout lowering vs direct `Cpu1LinearPreparer`
- [ ] Pridat LINEAR contract tests
- [ ] Zmapovat SDPA current CPU semantics
- [ ] Navrhnout cpu1 SDPA workspace layout
- [ ] Implementovat attention az po softmax parallel group path

---

## Faze 9: Native Storage A BF16 Policy

Status: `[ ]`

### Proc

cpu1 ma podporu pro `Cpu1StorageKind.MEMORY_SEGMENT`, ale neni vsude
systematicka. Pri dorovnani stareho CPU je dulezite, aby native cesta nebyla
jen "funguje", ale aby neztracela na per-execute alokacich nebo zbytecnych
prevodech.

### Cile

- native input/output binding pravidla na jednom miste
- output storage reuse pres runtime memory plan
- zadne skryte array materialization bez trace
- BF16 compute policy:
  - akumulace ve F32/F64 podle op
  - vystup do BF16 az na konci
  - zadna snaha predstirat nativni BF16 arithmetic v Java hot path

### Navrzeny Helper

Pokud se opakuje bindovani native vstupu/vystupu, zavadet jen maly cpu1-local
helper, ne obecnou compatibility vrstvu:

```java
final class Cpu1NativeBindingSupport {
    private Cpu1NativeBindingSupport() {
    }

    static Cpu1TensorView requireNativeInput(
            ExecutionContext context,
            int nodeId,
            Tensor tensor,
            CpuMaterializationReason reason
    ) {
        NativeTensorStorage storage = context.requireNativeReadable(nodeId, reason);
        return Cpu1TensorView.fromNativeStorage(tensor, storage);
    }

    static NativeTensorStorage requireNativeOutput(
            ExecutionContext context,
            int nodeId,
            DataType dataType,
            int elementCount,
            String label
    ) {
        return context.requireNativeOutputStorage(nodeId, dataType, elementCount, label);
    }
}
```

Zavadet az ve chvili, kdy mame aspon tri rodiny se stejnym opakovanim.
Jinak preferovat explicitni kod v executable unit.

### Tasky

- [ ] Audit per-execute native output allocation v cpu1
- [ ] Trace atributy: native input reused/copy-in/output reused/copy-out
- [ ] BF16 policy testy pro reduction/loss/matmul
- [ ] Native segment parity benchmark matrix

---

## Faze 10: Trace, Tuning A Coverage Gate

Status: `[ ]`

### Proc

Jakmile cpu1 dorovnava stary CPU, potrebujeme videt:

- proc byl vybran cpu1 nebo stary cpu
- proc byl vybran scalar/vector/parallel
- proc doslo k materializaci
- proc native cesta spadla na array
- jaky kernel id bezelo

### Trace Checklist

Kazda cpu1 prepared family musi traceovat:

- `cpu1KernelId`
- family-specific kernel id
- storage kind
- input/output dtype
- layout kind/access model
- vectorization kind
- launch workers
- launch chunk size
- scratch spec
- fallback/materialization reason, pokud existuje

### Coverage Gate

Cilovy test:

```java
@Test
void cpu1CoverageGateListsAllOldCpuDirectOps() {
    Cpu1CoverageReport report = Cpu1CoverageReport.current();
    assertThat(report.missingRequiredOps()).containsExactlyInAnyOrder(
            Operation.OpType.GATHER,
            Operation.OpType.GATHER_AXIS,
            Operation.OpType.GATHER_ND,
            Operation.OpType.TAKE_ALONG_AXIS,
            Operation.OpType.SCATTER_ADD,
            Operation.OpType.SCATTER_AXIS_ADD,
            Operation.OpType.SCATTER_ELEMENTS,
            Operation.OpType.SCATTER_ND,
            Operation.OpType.NLL_LOSS,
            Operation.OpType.CROSS_ENTROPY_LOSS,
            Operation.OpType.CROSS_ENTROPY_LOSS_INDICES,
            Operation.OpType.LINEAR,
            Operation.OpType.CONV2D,
            Operation.OpType.MAX_POOL2D,
            Operation.OpType.AVG_POOL2D,
            Operation.OpType.LAYER_NORM,
            Operation.OpType.RMS_NORM,
            Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION,
            Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS
    );
}
```

Tento test se bude aktualizovat pri kazde fazi a bude branit tomu, aby stav
parity zustal nejasny.

### Benchmark Matrix

Minimalni benchmark srovnani:

```text
elementwise:
  old cpu array
  cpu1 array scalar/vector/parallel
  cpu1 segment scalar/vector/parallel

reduction:
  old cpu dense
  cpu1 dense single
  cpu1 dense parallel
  cpu1 segment single/parallel

matmul:
  old cpu java
  old cpu openblas
  cpu1 java scalar/vector/parallel
  cpu1 openblas array
  cpu1 openblas native segment

loss:
  old cpu cross entropy indices
  cpu1 cross entropy indices array
  cpu1 cross entropy indices segment
```

---

## Doporucene Poradi Prace

Realisticke poradi, ktere minimalizuje prekopavani:

1. Faze 0: coverage gate
2. Faze 1: reduction prepared unit launch/scratch infrastructure
3. Faze 2: SUM/MEAN parallel + scratch
4. Faze 3: reduction segment/strided policy
5. Faze 4: softmax/logSoftmax parallel group path
6. Faze 5: loss family, nejdrive `CROSS_ENTROPY_LOSS_INDICES`
7. Faze 6: index/gather/scatter family
8. Faze 7: layer/rms norm, potom pool/conv
9. Faze 8: linear/attention
10. Faze 9-10: native/tuning/trace hardening a default-readiness

## Aktualni Known Gaps

Tento seznam se ma menit pri implementaci:

- [ ] cpu1 nema centralni parity coverage report
- [ ] reductions nemaji launch policy v prepared unit
- [ ] reductions nepouzivaji scratch partial buffers
- [ ] reductions nemaji systematickou segment paritu
- [ ] reductions nemaji strided/view input policy
- [ ] softmax/logSoftmax nejsou group-parallel
- [ ] NLL/CrossEntropy loss family chybi
- [ ] index/gather/scatter family chybi
- [ ] LINEAR direct/lowering policy neni uzavrena pro cpu1
- [ ] SDPA/attention cpu1 parita chybi
- [ ] Conv/pool/norm cpu1 parita chybi
- [ ] native storage policy neni sjednocena napric rodinami
- [ ] benchmark matrix neni kompletni

## Definition Of Done Pro Cely Plan

Plan je hotovy az kdyz:

- [ ] vsechny old CPU direct forward op typy maji cpu1 route nebo explicitni
  graph-lowering route
- [ ] legacy backward op typy jsou klasifikovane mimo direct CPU kernel parity
- [ ] cpu1 coverage gate nehlasi nezdokumentovane missing ops
- [ ] targeted parity tests pro kazdou rodinu prochazi
- [ ] benchmark report ukazuje, kde je cpu1 rychlejsi/pomalejsi a proc
- [ ] trace u kazde cpu1 route ukazuje kernel/storage/layout/threading
- [ ] native/array residency je explicitni a bez skrytych per-execute alokaci
- [ ] dokument je aktualizovan na `IMPLEMENTED_AND_VERIFIED`
