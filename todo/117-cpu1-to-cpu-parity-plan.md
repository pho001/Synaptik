# 117. cpu1 -> cpu Functional Parity Plan

## Stav Implementace

Status: `PLANNED`

Tento dokument je zivy implementacni checklist pro postupne dovedeni `backend.cpu1`
na funkcni paritu se starym `backend.cpu`, pri zachovani cistejsi cpu1 architektury.

Legenda:

- `[ ]` neimplementovano
- `[~]` rozpracovano
- `[x]` hotovo a overeno
- `[deferred]` zamerne odlozeno do samostatneho planu
- `[!]` zamerne neprebirat 1:1, vyzaduje jiny cpu1 design

Aktualni stav fazi:

- [x] Faze 0: parity inventory a ochrana pracovnich hranic
- [x] Faze 1: reduction runtime infrastruktura
- [x] Faze 2: parallel SUM/MEAN a partial scratch buffer
- [x] Faze 3: shared storage access plan, reduction native segment a strided/view policy
- [x] Faze 4: softmax/logSoftmax runtime width
- [x] Faze 5: loss vetev pro NLL a CrossEntropy; dense scope complete, strided/view deferred
- [x] Faze 6: index/gather/scatter operace; dense direct scope complete, strided/view a parallel scatter deferred
- [x] Faze 7: NN a normalization kernely; LAYER_NORM/RMS_NORM dense JAVA_ARRAY/MEMORY_SEGMENT slices, LINEAR dense matmul-backed subset, MAX_POOL2D/AVG_POOL2D dense direct routes and CONV2D dense direct correctness/fallback route complete; generic MATMUL_EPILOGUE IR deferred to follow-up plan 119 and optimized CONV2D -> UNFOLD2D -> MATMUL route deferred
- [~] Faze 8: linear algebra a attention parita
- [x] Faze 9: cpu1 storage mode contract; execute boundary binding sjednocen,
  ad-hoc residency rozhodovani v cpu1 hot paths odstraneno
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
- `LINEAR` cpu1 route existuje jako semantic MATMUL(input, weight)
  plus optional last-dim bias epilogue; region specialization maps
  standalone LINEAR+bias and exact MATMUL+ADD bias through the current concrete
  matmul-bias specialization route
- dense direct attention ops jsou dorovnane pro `JAVA_ARRAY` a
  `MEMORY_SEGMENT`

Zbyvajici parity prace:

- obecne backend-neutral MATMUL epilogue IR neni soucast tohoto planu; je
  presunute do [todo/119-general-matmul-epilogue-ir-plan.md](119-general-matmul-epilogue-ir-plan.md)
- rozsireni `LINEAR` zbyva pro strided/view inputs a pro prechod z konkretni
  matmul-bias specialization route na obecny `MATMUL_EPILOGUE` payload;
  aktualne dense contiguous no-offset Java-array route pokryva bias/no-bias a
  native MemorySegment route pokryva explicitni podporovane matmul-backed route
- attention view/native/vector backward parity a broader optimized-attention
  navazat na budouci materialization a optimized-attention faze
- doplnit batched matmul edge cases proti staremu CPU
- sjednotit OpenBLAS thread knobs s tuning/runtime configem

Verdikt:

```text
MATMUL je v cpu1 velmi dulezity a relativne daleko.
LINEAR direct dense subset je pokryty matmul-backed routou; attention ma dense
direct cpu1 route vcetne F32/F64 Vector API cesty a graph-lowered SDPA
backward specialized primitive route pro dense `JAVA_ARRAY` i
`MEMORY_SEGMENT` F32/F64 scalar/vector dQ/dK/dV. Strided/view, BF16 backward a
broader-optimized oblasti zustavaji chybejici parity oblast.
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

- ma samostatnou index family pro dense contiguous/no-offset read-only gather
  slice (`GATHER`, `GATHER_AXIS`, `GATHER_ND`, `TAKE_ALONG_AXIS`)
- cast backward/layout scatter semantiky se resila pres `SLICE_BACKWARD`
- index gradients nejsou primy CPU kernel ani ve starem CPU registry

Zbyvajici parity prace:

- rozsirit existujici `backend.cpu1.kernels.index` mimo dense read-only gather
- second wave zbyva:
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

- `LAYER_NORM` a `RMS_NORM` maji samostatnou `nn.normalization` family
- `MAX_POOL2D` ma samostatnou `nn.pool.maxpool` dense direct family
- `AVG_POOL2D` ma samostatnou `nn.pool.avgpool` dense direct family
- `CONV2D` ma samostatnou `nn.conv.conv2d` dense direct family
- nektere normalized/loss-like patterny by mohly byt casem fused/specialized

Zbyvajici parity prace:

- zadna zbyvajici dense NN/normalization correctness prace ve fazi 7
- strided/view/native-storage policy doplnovani podle planu 118
- preferovana budouci optimalizovana `CONV2D -> UNFOLD2D -> MATMUL` route je
  odlozena mimo fazi 7; aktualni `CONV2D` route je direct correctness/fallback

Verdikt:

```text
Dense NN/normalization op coverage pro fazi 7 je hotova.
Zbyvaji odlozene materialization/optimization prace mimo fazi 7.
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

Status: `[x]`

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
- [x] Benchmark: scalar large reduction single vs parallel

Benchmark harness:

- `src/test/java/backend/cpu1/Cpu1ReductionBenchmarkTest.java`
- JUnit tagged `@Tag("benchmark")`
- Covers scalar-output large vector reductions for `SUM` and `MEAN` on F32/F64
- Compares `Cpu1PrepareConfig.scalarSingleThread()` against explicit 4-worker scalar
  parallel launch with partial F64 scratch

Local benchmark evidence from one in-JVM run on 2026-06-16:

```text
elements=5_000_000, warmup=4, measure=12, parallelWorkers=4
F32 SUM:  single 3.2317 ms, parallel 1.0763 ms, speedup 3.00x
F32 MEAN: single 3.2159 ms, parallel 1.0653 ms, speedup 3.02x
F64 SUM:  single 3.3009 ms, parallel 1.0505 ms, speedup 3.14x
F64 MEAN: single 3.3468 ms, parallel 1.0463 ms, speedup 3.20x
```

Run command:

```bash
./gradlew test --tests backend.cpu1.Cpu1ReductionBenchmarkTest
```

### Overeni

```bash
./gradlew test --tests backend.cpu1.Cpu1ReductionExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1ScratchBufferTest
./gradlew test --tests backend.cpu1.Cpu1ReductionBenchmarkTest
```

---

## Faze 3: Reduction Native Segment A Strided/View Input Policy

Status: `[x]`

### Proc

Stary CPU umi diky `CpuStorageView` mnohem vice kombinaci:

- array
- MemorySegment
- storage offset
- strides
- mixed input/output storage

cpu1 reductions jsou dnes dense contiguous only a segment cesta je omezena.

### Rozhodnuti: Sdileny Access Plan, Potom Reduction Kernels

Musime si rict, co se stane, kdyz reduction input neni dense contiguous.

Moznosti:

1. Materializovat contiguous pred reduction
2. Pouzit strided reduction kernel
3. Odmitnout route s jasnou trace/reason

Prvni krok je hotovy: sdileny prepare-time policy typ v
`backend.cpu1.storage`:

```java
Cpu1StorageAccessPlan.fromDescriptor(input)
Cpu1StorageAccessPlan.fromNode(output)
Cpu1StorageAccessPlan.forBroadcastedLogicalShape(input, logicalShape)
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

`fromDescriptor(...)` popisuje tensor/view tak, jak existuje v compiled
descriptoru.

`fromNode(...)` popisuje output node.

`forBroadcastedLogicalShape(...)` popisuje, jak se input cte v logickem loopu
nad cilovym shape. Typicky:

```java
input.shape = [1, 3]
input.strides = [3, 1]
logicalShape = [2, 3]

// vysledek
kind = BROADCAST
shape = [2, 3]
strides = [0, 1]
elementCount = 6
```

Tato metoda je obecna. Neni elementwise-specific. Pouziva ji elementwise
prepare a fused input planning, pozdeji ji muze pouzit cast/layout
materialization policy.

Proc to nekopiruje stary `CpuStorageView` 1:1:

- stary `CpuStorageView` micha storage handle, layout a runtime pristup do
  kernelu
- cpu1 chce mit storage/layout rozhodnuti v prepare vrstve a runtime view
  ponechat jen na bindovane buffery
- tento krok pouze sjednocuje klasifikaci; nedela materializaci, obecne
  read/write helpery ani strided reduction kernels

Aktualni stav po prepisu:

- reduction contract checky pouzivaji `Cpu1StorageAccessPlan`
- elementwise prepare si vytvari output/input access plany a z nich vybira
  `Cpu1LayoutKind`
- `Cpu1PreparedElementwiseUnit` drzi input/output access plany jako prepare
  metadata pro testy/trace/policy
- fused IR drzi base access plan i logical access plan pro kazdy external
  input
- `Cpu1FusedAccessKind` zustava fused/codegen-specific, ale odvozuje se z
  common logical access planu
- zadny kernel inner loop necita pres `Cpu1StorageAccessPlan`

Aktualni stav po dokonceni Faze 3:

- reduction prepared unit drzi input/output `Cpu1StorageAccessPlan`
- dense input umi `DENSE_CONTIGUOUS` i `DENSE_WITH_OFFSET`
- dense `MEMORY_SEGMENT` cesta pokryva `SUM/MEAN`, `MIN/MAX/PROD`,
  `ALL/ANY`, `ARGMAX` a `CUMSUM` pro dtype kombinace uvedene v task listu
- prvni direct strided cesta je zamerne uzka: `SUM/MEAN FLOAT32/FLOAT64`
- `BROADCAST`, `UNSUPPORTED` a strided op/dtype mimo tento uzky vyrez se dal
  odmita v prepare s viditelnym access kind/reason
- zadny reduction inner loop necita pres `Cpu1StorageAccessPlan` ani obecny
  storage accessor; plany jsou prepare/trace metadata a strided kernely si z
  nich jednorazove berou shape/stride fakta

Zamerne non-goals mimo Fazi 3:

- BF16 strided `SUM/MEAN`
- strided `MIN/MAX/PROD`, `ALL/ANY`, `ARGMAX`, `CUMSUM`
- broadcast materialization policy pro reductions
- softmax/logSoftmax native/strided runtime width; to patri do Faze 4

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
- [x] Pridat `forBroadcastedLogicalShape(...)` pro logical loop access
- [x] Pokryt dense, offset, strided, broadcast a defensive-copy pripady testy
- [x] Zapojit access plan do `Cpu1ReductionPreparer` contract checku
- [x] Pouzit common access plan pro elementwise layout/input planning
- [x] Ulozit elementwise input/output access plany v prepared unit
- [x] Pouzit common access plan pro fused external input planning
- [x] Zachovat `Cpu1FusedAccessKind` jako codegen-specific derived metadata
- [x] Overit, ze access plan zustava mimo hot path a bez read/write helperu
- [x] Navrhnout dense segment policy tests podle access planu
- [x] Rozsirit segment support pro `SUM/MEAN BF16`
- [x] Rozsirit segment support pro `MIN/MAX/PROD F32/F64/BF16`
- [x] Rozsirit segment support pro `ALL/ANY BOOL`
- [x] Rozsirit segment support pro `ARGMAX F32/F64/BF16/I32/I64 -> I64`
- [x] Rozsirit segment support pro `CUMSUM F32/F64/BF16/I32/I64`
- [x] Pridat dense segment contract tests
- [x] Navrhnout strided/view reduction policy tests podle access planu
- [x] Pridat prvni strided direct kernel jen pro `SUM/MEAN F32/F64`

### Overeni

```bash
./gradlew test --tests backend.cpu1.Cpu1StorageAccessPlanTest
./gradlew test --tests backend.cpu1.Cpu1ReductionExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1ExecutionContractTest
./gradlew test --tests backend.cpu1.fused.Cpu1FusedIrBuilderTest
./gradlew classes
git diff --check
```

---

## Faze 4: Softmax / LogSoftmax Runtime Width

Status: `[x]`

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

- [x] F32/F64/BF16 array parallel group path
- [x] F32/F64/BF16 segment group path
- [x] Threshold z config/tuningu pro kdy parallel zapnout
- [x] Benchmark `batch x classes`: 1k, 10k, 100k groups
- [x] Parity test proti starym `SoftmaxExecutionTest` a `LogSoftmaxExecutionTest`

Implementace:

- `Cpu1SoftmaxReductionLoops` pouziva prepared `Cpu1LaunchConfig` pro range
  launch pres `outerSize * innerSize` nezavislych skupin.
- Stejne concrete kernel idy obsluhuji `JAVA_ARRAY` i `MEMORY_SEGMENT` podle
  prepared `storageKind`; execute nevybira novy kernel ani nevola stary CPU
  fallback.
- Segment cesta cte/zapisuje `MemorySegment` pres `ValueLayout` pro
  `FLOAT32`, `FLOAT64` a `BFLOAT16`; BF16 akumuluje pres F32/double a zapisuje
  BF16 bits.
- Automatic launch pro softmax/logSoftmax pouziva
  `CpuKernelConfig.reductionParallelMinSize()`,
  `highCostTargetChunksPerWorker()` a `minReductionChunkSize()` uz v prepare.

Benchmark harness:

- `src/test/java/backend/cpu1/Cpu1ReductionSoftmaxBenchmarkTest.java`
- JUnit tagged `@Tag("benchmark")`
- Covers `SOFTMAX` and `LOG_SOFTMAX`, `batch x classes`, groups
  `1_024`, `10_000`, `100_000`, array and native segment, single-thread vs
  4-worker group-parallel launch.

### Overeni

```bash
./gradlew test --tests backend.cpu1.Cpu1ReductionExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1ReductionSoftmaxBenchmarkTest
./gradlew test --tests backend.cpu1.Cpu1StorageAccessPlanTest
./gradlew test --tests backend.cpu1.Cpu1ExecutionContractTest
./gradlew test --tests backend.cpu1.fused.Cpu1FusedIrBuilderTest
./gradlew classes
git diff --check
```

---

## Faze 5: Loss Family - NLL A CrossEntropy

Status: `[x]` dense scope complete; strided/view policy `[deferred]` do
[todo/118-cpu1-graph-input-materialization-plan.md](118-cpu1-graph-input-materialization-plan.md)

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

- [x] Vytvorit `Cpu1LossPreparer`
- [x] Vytvorit `Cpu1PreparedCrossEntropyLossUnit`
- [x] Vytvorit `Cpu1LossExecutableUnit`
- [x] Vytvorit `Cpu1CrossEntropyKernelId`
- [x] Vytvorit `Cpu1CrossEntropyKernelDispatch`
- [x] Implementovat `CROSS_ENTROPY_LOSS_INDICES F32 ARRAY`
- [x] Implementovat `CROSS_ENTROPY_LOSS_INDICES F64 ARRAY`
- [x] Implementovat `CROSS_ENTROPY_LOSS_INDICES BF16 ARRAY`
- [x] Implementovat `CROSS_ENTROPY_LOSS_INDICES F32/F64/BF16 MEMORY_SEGMENT`
- [x] Podporovat INT32 a INT64 target indices
- [x] Podporovat `LossReduction.NONE`
- [x] Podporovat `LossReduction.SUM`
- [x] Podporovat `LossReduction.MEAN`
- [x] Podporovat `ignoreIndex`
- [x] Pridat trace metadata
- [x] Pridat tests podle `IndexTargetCrossEntropyLossExecutionTest`

### Druha Wave

- [x] `NLL_LOSS` dense target distribution
- [x] `CROSS_ENTROPY_LOSS` dense target distribution
- [x] dense contiguous native segment cesty pro `NLL_LOSS` a dense `CROSS_ENTROPY_LOSS`
- [deferred] strided logits/targets policy; resi samostatny plan
  [todo/118-cpu1-graph-input-materialization-plan.md](118-cpu1-graph-input-materialization-plan.md)

Aktualni omezeni:

- loss family podporuje `JAVA_ARRAY` a dense contiguous `MEMORY_SEGMENT`
  pro FLOAT32/FLOAT64/BFLOAT16; BF16 segment je explicitni `JAVA_SHORT`
  raw-bit cesta, ne fallback pres obecny accessor
- `CROSS_ENTROPY_LOSS_INDICES` native cesta podporuje INT32/INT64 target segmenty
- strided/view loss vstupy zustavaji zamerne odmítnute v prepare pres dense
  contiguous kontrakt; cpu1 prepare/kernel nesmi skryte materializovat vstupy.
  Graph/lowering-driven materializace je odlozena do
  [todo/118-cpu1-graph-input-materialization-plan.md](118-cpu1-graph-input-materialization-plan.md)

### Overeni

```bash
./gradlew test --tests backend.cpu1.Cpu1CrossEntropyLossExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1NllLossExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1DenseCrossEntropyLossExecutionContractTest
```

---

## Faze 6: Index / Gather / Scatter Family

Status: `[x]` dense direct scope complete; strided/view/offset materialization `[deferred]` do
[todo/118-cpu1-graph-input-materialization-plan.md](118-cpu1-graph-input-materialization-plan.md);
parallel scatter `[deferred]`

Poznamka: Wave 1 slice je hotovy pro `GATHER`, `GATHER_AXIS` a
`TAKE_ALONG_AXIS` dense contiguous/no-offset `JAVA_ARRAY` s hodnotovymi dtype
`FLOAT32`, `FLOAT64`, `BFLOAT16`, `INT32`, `INT64`, `BOOL` a index dtype
`INT32`/`INT64`.

Dense contiguous/no-offset `MEMORY_SEGMENT` podpora pro tyto read-only index
opy je samostatne naplanovana jako Wave 1B. Neni to soucast strided/view
materialization policy: ma jit o prime segment kernely bez obecneho runtime
storage accessoru a bez fallbacku do stareho `backend.cpu`.

`GATHER_ND` bylo doplneno ve Wave 2A se stejnym dense contiguous/no-offset
kontraktem. `SCATTER_ADD` a `SCATTER_AXIS_ADD` jsou doplnene ve Wave 2B jako
dense direct single-thread scatter-add slice. Wave 3 doplnila `SCATTER_ELEMENTS`
a `SCATTER_ND`, takze dense direct index/scatter scope je hotovy.

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
  takealongaxis/
    Cpu1TakeAlongAxisLoops
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

- [x] `GATHER`
- [x] `GATHER_AXIS`
- [x] `TAKE_ALONG_AXIS`

Proc prvni:

- jsou read-only z pohledu source
- nemaji duplicate write race
- dobre se paralelizuji pres output elements

### Wave 1B

Dense contiguous/no-offset `MEMORY_SEGMENT` parita pro hotove read-only index
opy:

- [x] `GATHER MEMORY_SEGMENT`
- [x] `GATHER_AXIS MEMORY_SEGMENT`
- [x] `TAKE_ALONG_AXIS MEMORY_SEGMENT`

Proc samostatne:

- algoritmus a tvarova semantika uz jsou overene na `JAVA_ARRAY`
- segment cesta musi pridat explicitni `MemorySegment.get/set` hot path pro
  `FLOAT32`, `FLOAT64`, `BFLOAT16`, `INT32`, `INT64`, `BOOL`
- index segmenty musi podporovat `INT32` a `INT64`
- BF16 musi kopirovat raw `short` bity, ne jit pres obecny accessor
- nechceme zavest univerzalni segment reader/writer do hot path jen kvuli
  rychlemu pokryti

Rozsah:

- stale pouze dense contiguous/no-offset input, indices a output
- zadna strided/view/offset cesta
- zadna skryta materializace v cpu1 prepare/kernel
- zadny fallback do stareho `backend.cpu`
- prepare vybere konkretni segment kernel id a execute jen spusti pripraveny
  kernel

Overeni:

```bash
./gradlew test --tests backend.cpu1.Cpu1GatherExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1CpuParityInventoryTest
./gradlew classes
git diff --check
```

### Wave 2

- [x] `GATHER_ND`
- [x] `SCATTER_ADD`
- [x] `SCATTER_AXIS_ADD`

#### Wave 2A: `GATHER_ND`

Hotovo:

- dense contiguous/no-offset `JAVA_ARRAY`
- dense contiguous/no-offset `MEMORY_SEGMENT`
- value dtype:
  - `FLOAT32`
  - `FLOAT64`
  - `BFLOAT16`
  - `INT32`
  - `INT64`
  - `BOOL`
- index dtype:
  - `INT32`
  - `INT64`
- tuple indexed elementy
- tuple indexed slices
- `batchDims`
- project scalar shape `[1]`
- negativni indexy normalizovane podle velikosti indexovane dimenze
- BF16 raw `short` bit copy

Implementacni hranice:

- zadny fallback do stareho `backend.cpu`
- zadny generic hot-path storage accessor
- zadna skryta materializace v cpu1 prepare/kernel
- strided/view/offset vstupy, indexy a vystupy jsou stale odmítnute pres
  dense contiguous/no-offset kontrakt
- `GATHER_ND_GRAD` neni novy primy cpu1 kernel; stejne jako ve verejne
  semantice zustava backward skladany pres scatter lowering

Overeni:

```bash
./gradlew test --tests backend.cpu1.Cpu1GatherExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1CpuParityInventoryTest
./gradlew classes
git diff --check
```

#### Wave 2B: `SCATTER_ADD` A `SCATTER_AXIS_ADD`

Hotovo:

- `SCATTER_ADD` dense contiguous/no-offset `JAVA_ARRAY`
- `SCATTER_ADD` dense contiguous/no-offset `MEMORY_SEGMENT`
- `SCATTER_AXIS_ADD` dense contiguous/no-offset `JAVA_ARRAY`
- `SCATTER_AXIS_ADD` dense contiguous/no-offset `MEMORY_SEGMENT`
- value dtype:
  - `FLOAT32`
  - `FLOAT64`
  - `BFLOAT16`
- index dtype:
  - `INT32`
  - `INT64`
- output zacina jako kopie base/data tensoru
- duplicate indices akumuluji deterministicky v logical order
- `SCATTER_ADD` zachovava stare CPU chovani: negativni index je out-of-bounds
- `SCATTER_AXIS_ADD` zachovava gather-axis chovani: negativni index se
  normalizuje podle velikosti osy
- BF16 pocita akumulaci pres `F32` a zapisuje `BF16` raw bity
- prepare vynuti deterministic single-thread launch (`workerCount=1`) i pokud
  runtime config pozaduje paralelizaci
- trace ukazuje kernel id, storage kind, op type, update element count a
  single-thread launch

Implementacni hranice:

- zadny fallback do stareho `backend.cpu`
- zadny generic hot-path storage accessor
- zadna skryta materializace v cpu1 prepare/kernel
- strided/view/offset base/data, indices, updates a output jsou odmítnute pres
  dense contiguous/no-offset kontrakt
- paralelni scatter neni implementovany kvuli duplicate-index write race
- `SCATTER_ELEMENTS` a `SCATTER_ND` zustavaji ve Wave 3

Overeni:

```bash
./gradlew test --tests backend.cpu1.Cpu1GatherExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1CpuParityInventoryTest
./gradlew classes
git diff --check
```

### Wave 3

- [x] `SCATTER_ELEMENTS`
- [x] `SCATTER_ND`

Hotovo:

- `SCATTER_ELEMENTS` dense contiguous/no-offset `JAVA_ARRAY`
- `SCATTER_ELEMENTS` dense contiguous/no-offset `MEMORY_SEGMENT`
- `SCATTER_ND` dense contiguous/no-offset `JAVA_ARRAY`
- `SCATTER_ND` dense contiguous/no-offset `MEMORY_SEGMENT`
- value dtype:
  - `FLOAT32`
  - `FLOAT64`
  - `BFLOAT16`
  - `INT32`
  - `INT64`
  - `BOOL`
- index dtype:
  - `INT32`
  - `INT64`
- reduction:
  - `NONE`
  - `ADD`
  - `MUL`
  - `MAX`
  - `MIN`
- `BOOL` podporuje jen `NONE` a ostatni reduction mody jsou odmítnute v prepare
- duplicate targety pri `NONE` jsou odmítnute pres explicitni seen pole
- duplicate targety pri redukcich se zpracovavaji deterministicky v logical order
- negativni indexy jsou normalizovane podle cilove dimenze
- `SCATTER_ND` podporuje `batchDims` a tuple-indexed element/slice updates
- prepare vynuti deterministic single-thread launch (`workerCount=1`)
- trace ukazuje kernel id, storage kind, op type, reduction, update element count
  a single-thread launch

Implementacni hranice:

- zadny fallback do stareho `backend.cpu`
- zadny generic hot-path storage accessor
- prvni implementace je dense contiguous/no-offset only
- paralelni scatter neni implementovany kvuli duplicate-index write race

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

## Faze 7: NN A Normalization

Status: `[x]`

### Proc

Po reductions/loss/index zbyvaly higher-level numeric kernels. Dense
correctness scope faze 7 je hotovy pro:

- `LINEAR`
- `LAYER_NORM`
- `RMS_NORM`
- `CONV2D`
- `MAX_POOL2D`
- `AVG_POOL2D`

### Poradi

1. `LAYER_NORM` - dense contiguous no-offset `JAVA_ARRAY` and `MEMORY_SEGMENT` slice implemented for FLOAT32/FLOAT64/BFLOAT16
2. `RMS_NORM` - dense contiguous no-offset `JAVA_ARRAY` and `MEMORY_SEGMENT` slice implemented for FLOAT32/FLOAT64/BFLOAT16
3. `LINEAR` - matmul-backed dense contiguous no-offset epilogue route implemented;
   `JAVA_ARRAY` supports no-bias and bias for FLOAT32/FLOAT64/BFLOAT16, explicit
   `OPENBLAS_NATIVE_SEGMENT` supports no-bias FLOAT32/FLOAT64
4. `MAX_POOL2D` - dense contiguous no-offset `JAVA_ARRAY` and `MEMORY_SEGMENT`
   direct forward route implemented for FLOAT32/FLOAT64/BFLOAT16; scalar loop
   body, parallelized over output element ranges through existing cpu1 launch
   policy/config
5. `AVG_POOL2D` - dense contiguous no-offset `JAVA_ARRAY` and `MEMORY_SEGMENT`
   direct forward route implemented for FLOAT32/FLOAT64/BFLOAT16; scalar loop
   body, parallelized over output element ranges through existing cpu1 launch
   policy/config; countIncludePad and ceilMode semantics covered
6. `CONV2D` - dense contiguous no-offset `JAVA_ARRAY` and `MEMORY_SEGMENT`
   direct Java correctness/fallback route implemented for
   FLOAT32/FLOAT64/BFLOAT16; supports optional bias, stride, padding, dilation,
   groups/depthwise-style grouped convolution, and range parallelization over
   output elements

Proc takto:

- norm opy vyuziji reduction + elementwise + scratch infrastrukturu
- linear je aktualne matmul-backed epilogue route v `Cpu1MatmulPreparer` bez
  extra provider vrstvy; zobecneni matmul epilogue IR je mimo fazi 7 a patri do
  planu [119](119-general-matmul-epilogue-ir-plan.md)
- pool2d je layout/index heavy, ale jednodussi nez conv2d
- conv2d ma nejvic route rozhodnuti; v teto fazi je zamerne vyreseny jen
  direct correctness/fallback route bez `UNFOLD2D`, OpenBLAS nebo obecneho
  matmul epilogue IR

### Scope Korekce Po Plan 119

Zobecneni matmul epilogue specializaci uz neni soucast faze 7 tohoto parity
planu. Faze 7 smi jen udrzovat aktualni `LINEAR` dense subset funkcni a
testovany. Obecna nahrada konkretni specialization enum logiky:

```text
MATMUL_RELU
MATMUL_ADD_BIAS
MATMUL_ADD_BIAS_RELU
```

za backend-neutral:

```text
RegionSpecializationKind.MATMUL_EPILOGUE
  payload: MatmulEpiloguePlan
```

je samostatny follow-up v
[todo/119-general-matmul-epilogue-ir-plan.md](119-general-matmul-epilogue-ir-plan.md).

To znamena:

- faze 7 nepokracuje dalsim refaktorem matmul epilogue IR
- faze 7 nepokracuje rozsirovanim `LINEAR` pres nove graph-level epilogue kindy
- faze 7 muze pridat jen missing NN parity opy, ktere jsou nezavisle na obecnem
  matmul epilogue IR
- preferovana budouci optimalizovana conv route je
  `CONV2D -> UNFOLD2D -> MATMUL`; prvni implementace v tomto planu zustava
  direct Java correctness/fallback route a optimalizace patri az do samostatne
  navazne prace po stabilizaci planu 118/119 a faze 8/9

Prakticky stav faze 7 po CONV2D:

```text
Dense NN/normalization scope hotovy: LAYER_NORM, RMS_NORM, LINEAR dense subset,
MAX_POOL2D, AVG_POOL2D a CONV2D direct correctness/fallback.
```

`MAX_POOL2D` bylo prvni, protoze nevyzaduje akumulacni deleni jako AVG pool a
nevyzaduje kernel/filter semantiku jako CONV2D. Overilo indexaci NCHW,
padding, stride, ceil-mode shape inference, dense storage kontrakt a paralelni
output-range launch bez zasahu do matmul/epilogue architektury.

### CONV2D Stav

Status: `[x]`

Implementovane soubory:

- `src/main/java/backend/cpu1/prepare/Cpu1Conv2dPreparer.java`
- `src/main/java/backend/cpu1/prepare/Cpu1PreparedConv2dUnit.java`
- `src/main/java/backend/cpu1/exec/Cpu1Conv2dExecutableUnit.java`
- `src/main/java/backend/cpu1/kernels/nn/conv/conv2d/Cpu1Conv2dKernelId.java`
- `src/main/java/backend/cpu1/kernels/nn/conv/conv2d/Cpu1Conv2dKernel.java`
- `src/main/java/backend/cpu1/kernels/nn/conv/conv2d/Cpu1Conv2dKernelDispatch.java`
- `src/main/java/backend/cpu1/kernels/nn/conv/conv2d/Cpu1Conv2dLoops.java`
- `src/test/java/backend/cpu1/Cpu1Conv2dExecutionContractTest.java`

Pokryti:

- Operation route: `Cpu1NodePreparer -> Cpu1Conv2dPreparer` pro
  `Operation.OpType.CONV2D`
- dtype: `FLOAT32`, `FLOAT64`, `BFLOAT16`
- storage: `JAVA_ARRAY`, `MEMORY_SEGMENT`
- layout: dense contiguous no-offset input/weight/bias/output
- semantika: rank-4 NCHW input/output, OIHW weight,
  stride/padding/dilation/groups podle `Conv2dOptions`
- optional bias: treti input se cte jako `[outChannels]`
- grouped/depthwise-style conv: podporovano pres obecny `groups` kontrakt,
  `weightShape[1] * groups == inputChannels`
- BF16: vstup/weight/bias se cte jako BF16, akumulace bezi ve F32 a vysledek
  se az na konci materializuje zpet do BF16
- launch: scalar direct kernel body, range parallelization over output elements
  pres existujici `Cpu1LaunchPolicy`; automatic mode pouziva
  `CpuKernelConfig.matMulParallelMinSize()` nad odhadem prace
  `outputElements * channelsPerGroup * kernelH * kernelW` a
  `minScalarChunkSize()`
- trace: `cpu1Conv2dKernelId`, dtype, storage, bias, groups, stride, padding,
  dilation, access kinds a launch workers/chunk

Zamerne nepokryto v teto casti:

- strided/view inputy nebo vystupy; materializaci ma rozhodnout graph/lowering
  podle samostatneho planu 118
- preferovana budouci optimalizovana `CONV2D -> UNFOLD2D -> MATMUL`, OpenBLAS
  nebo provider route; direct route z faze 7 zustava correctness/fallback cesta
- conv backward direct specializace; soucasny autograd sklada gradient pres
  layout/matmul/reduction primitiva

### MAX_POOL2D Stav

Status: `[x]`

Implementovane soubory:

- `src/main/java/backend/cpu1/prepare/Cpu1Pool2dPreparer.java`
- `src/main/java/backend/cpu1/prepare/Cpu1PreparedMaxPool2dUnit.java`
- `src/main/java/backend/cpu1/exec/Cpu1MaxPool2dExecutableUnit.java`
- `src/main/java/backend/cpu1/kernels/nn/pool/maxpool/Cpu1MaxPool2dKernelId.java`
- `src/main/java/backend/cpu1/kernels/nn/pool/maxpool/Cpu1MaxPool2dKernel.java`
- `src/main/java/backend/cpu1/kernels/nn/pool/maxpool/Cpu1MaxPool2dKernelDispatch.java`
- `src/main/java/backend/cpu1/kernels/nn/pool/maxpool/Cpu1MaxPool2dLoops.java`
- `src/test/java/backend/cpu1/Cpu1Pool2dExecutionContractTest.java`

Pokryti:

- Operation route: `Cpu1NodePreparer -> Cpu1Pool2dPreparer` pro
  `Operation.OpType.MAX_POOL2D`
- dtype: `FLOAT32`, `FLOAT64`, `BFLOAT16`
- storage: `JAVA_ARRAY`, `MEMORY_SEGMENT`
- layout: dense contiguous no-offset input/output
- semantika: rank-4 NCHW, kernel/stride/padding/ceilMode podle
  `Pool2dOptions`
- padding: padding se nikdy nebere jako hodnota 0, maximum se inicializuje az
  prvnim validnim vstupem; zaporne vstupy tedy zustanou korektni
- launch: scalar kernel body, range parallelization over output elements pres
  existujici `Cpu1LaunchPolicy`; automatic mode pouziva `CpuKernelConfig`
  thresholdy a chunk tuning, ne hardcoded thread count

Zamerne nepokryto v teto casti:

- strided/view inputy nebo vystupy; materializaci ma rozhodnout graph/lowering
  podle samostatneho planu 118
- backward-special workspace/argmax cache; soucasny graph backward rozklada
  pool gradient pres layout/index/reduction primitiva

### AVG_POOL2D Stav

Status: `[x]`

Implementovane soubory:

- `src/main/java/backend/cpu1/prepare/Cpu1Pool2dPreparer.java`
- `src/main/java/backend/cpu1/prepare/Cpu1PreparedAvgPool2dUnit.java`
- `src/main/java/backend/cpu1/exec/Cpu1AvgPool2dExecutableUnit.java`
- `src/main/java/backend/cpu1/kernels/nn/pool/avgpool/Cpu1AvgPool2dKernelId.java`
- `src/main/java/backend/cpu1/kernels/nn/pool/avgpool/Cpu1AvgPool2dKernel.java`
- `src/main/java/backend/cpu1/kernels/nn/pool/avgpool/Cpu1AvgPool2dKernelDispatch.java`
- `src/main/java/backend/cpu1/kernels/nn/pool/avgpool/Cpu1AvgPool2dLoops.java`
- `src/test/java/backend/cpu1/Cpu1Pool2dExecutionContractTest.java`

Pokryti:

- Operation route: `Cpu1NodePreparer -> Cpu1Pool2dPreparer` pro
  `Operation.OpType.AVG_POOL2D`
- dtype: `FLOAT32`, `FLOAT64`, `BFLOAT16`
- storage: `JAVA_ARRAY`, `MEMORY_SEGMENT`
- layout: dense contiguous no-offset input/output
- semantika: rank-4 NCHW, kernel/stride/padding/ceilMode podle
  `Pool2dOptions`
- average divisor:
  - `countIncludePad=false`: deli se poctem validnich vstupnich prvku v okne
  - `countIncludePad=true`: deli se plnou velikosti `kernelH * kernelW`
- BF16: vstup se cte jako BF16, akumulace bezi ve F32 a vysledek se az na konci
  materializuje zpet do BF16
- launch: scalar kernel body, range parallelization over output elements pres
  existujici `Cpu1LaunchPolicy`; automatic mode pouziva `CpuKernelConfig`
  thresholdy a chunk tuning, ne hardcoded thread count

Zamerne nepokryto v teto casti:

- strided/view inputy nebo vystupy; materializaci ma rozhodnout graph/lowering
  podle samostatneho planu 118
- pool backward specializace; soucasny graph backward rozklada pool gradient pres
  layout/index/reduction primitiva

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
2. `CONV2D -> UNFOLD2D -> MATMUL`
3. provider abstraction pro native knihovny

Rozhodnuti pro fazi 7:

```text
direct Java scalar correctness/fallback route
parallel over output element ranges
preferred optimized CONV2D -> UNFOLD2D -> MATMUL route deferred
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

Status: `[~]` dense direct `SCALED_DOT_PRODUCT_ATTENTION` /
`SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS` implemented; optimized/broader
attention parity remains scoped below.

### Proc

`MATMUL` je cpu1 uz daleko, ale stary CPU ma jeste:

- `LINEAR`
- `SCALED_DOT_PRODUCT_ATTENTION`
- `SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS`

### LINEAR

Status: `[x]` for dense contiguous no-offset direct forward subset.

Generic backend-neutral `MATMUL_EPILOGUE` IR is intentionally deferred to
[todo/119-general-matmul-epilogue-ir-plan.md](119-general-matmul-epilogue-ir-plan.md).
This section tracks only the current dense cpu1 parity subset.

Preferovane reseni:

```text
cpu1 prepare:
  LINEAR(input, weight, bias)
    -> validate LinearSpec contract:
       input [..., inFeatures], weight [inFeatures, outFeatures],
       optional bias [outFeatures] or [1, outFeatures]
    -> prepared matmul route over MATMUL(input, weight)
    -> optional ADD_BIAS epilogue for Java scalar/vector routes
```

Implementovano jako tenka matmul-epilogue specializace nad existujicim
`Cpu1MatmulPreparer`, `Cpu1PreparedMatmulUnit` a matmul provider selection.
Nepridava samostatnou linear provider vrstvu.

Aktualni pokryti:

- dense contiguous no-offset inputs/output/bias only
- `JAVA_ARRAY`: FLOAT32/FLOAT64/BFLOAT16 no-bias and bias
- `OPENBLAS_NATIVE_SEGMENT`: FLOAT32/FLOAT64 no-bias and bias epilogues,
  explicit route, native-current dense contiguous no-offset inputs/bias required
- unsupported dtype/route/post-op combinations fail in prepare
- region specialization maps standalone `LINEAR(input, weight, bias)` and exact
  `MATMUL(input, weight).add(bias)` through the current concrete
  matmul-bias specialization route until plan 119 replaces it with
  `MATMUL_EPILOGUE`
- existing `LINEAR -> RELU` specialization remains on the current concrete
  bias+relu epilogue route until plan 119 replaces it with `MATMUL_EPILOGUE`
- LINEAR stays aligned with current cpu1 MATMUL support: dense contiguous
  inputs/output with batch broadcast offsets only; broad strided LINEAR is not
  introduced until MATMUL has the same execution support.

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

Aktualni cpu1 attention stav:

- dense contiguous no-offset direct forward route implemented for
  `SCALED_DOT_PRODUCT_ATTENTION`
- dense contiguous no-offset direct cached-weights publication implemented for
  `SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS`
- `JAVA_ARRAY`: FLOAT32/FLOAT64/BFLOAT16 direct execution
- `MEMORY_SEGMENT`: FLOAT32/FLOAT64/BFLOAT16 direct execution when all runtime
  inputs already have current native CPU storage
- FLOAT32/FLOAT64 dense direct route supports scalar and Vector API execution
  for both `JAVA_ARRAY` and `MEMORY_SEGMENT`; prepare selects vector only when
  runtime config requests it
- FLOAT32/FLOAT64 SDPA backward specialized route supports dense Java Vector
  API execution for `JAVA_ARRAY` and `MEMORY_SEGMENT`; native segment inputs
  remain native CPU segments and are not materialized into array storage
- BFLOAT16 remains scalar with FLOAT32 accumulation; it intentionally does not
  pretend to have a vector kernel while Java has no BF16 primitive/vector lane
- mask handling matches old CPU direct semantics, including all-masked rows
  becoming a uniform average over value rows
- weights publication requires the input compiled descriptor to be an actual
  `SCALED_DOT_PRODUCT_ATTENTION` node and requires the attention output to have
  `requiresGrad=true`
- no old `backend.cpu` attention implementation is imported by cpu1

### Tasky

- [x] Zmapovat, kdy graph ponechava `LINEAR` jako op
- [x] Rozhodnout lowering vs direct route: matmul-backed epilogue route in
  `Cpu1MatmulPreparer`
- [x] Pridat LINEAR contract tests
- [x] Zmapovat SDPA current CPU semantics
- [x] Implementovat dense direct SDPA forward route
- [x] Implementovat dense direct SDPA weights publication route
- [x] Pridat targeted cpu1 attention contract tests
- [deferred] Strided/view attention inputs and outputs; current route
  intentionally rejects STRIDED and DENSE_WITH_OFFSET descriptors instead of
  materializing. Input materialization policy belongs to
  [todo/118-cpu1-graph-input-materialization-plan.md](118-cpu1-graph-input-materialization-plan.md).
- [x] Vectorized dense direct attention forward kernels for FLOAT32/FLOAT64
  `JAVA_ARRAY` and `MEMORY_SEGMENT`; dot products and weighted value
  accumulation use Vector API, while softmax row normalization remains scalar
- [deferred] Broader blocked/tiled optimized attention kernels; this is a
  separate performance phase after materialization policy and benchmark coverage
  are stable
- [deferred] Broader attention materialization policy; current route requires
  runtime storage to already match the prepared JAVA_ARRAY/MEMORY_SEGMENT route.
  Unsupported views must be made explicit by graph/lowering through plan 118,
  not silently copied by `Cpu1AttentionPreparer` or the attention kernel.
- [x] Graph-lowered SDPA backward specialized primitive route for dQ/dK/dV:
  region specialization recognizes the canonical primitive backward DAG and
  cpu1 prepares dense `JAVA_ARRAY` and dense `MEMORY_SEGMENT` FLOAT32/FLOAT64
  scalar kernels; `outGrad` may be dense or explicit no-offset broadcast,
  which covers the common `attention.sum()` gradient path without hidden
  materialization
- [x] Vectorized dense direct SDPA backward kernels for FLOAT32/FLOAT64
  `JAVA_ARRAY` and `MEMORY_SEGMENT`; dQ/dK accumulate over depth with Vector
  API, dV accumulates over valueDim for dense outGrad, and dScore row dot
  products use Vector API where the prepared access is contiguous. BF16,
  strided/view inputs, BLAS routing, and blocked/tiled multi-output backward
  remain outside this scope.
- [deferred] Legacy direct `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD` op ownership,
  BF16 backward kernels, strided/view backward inputs, and multi-output
  blocked/tiled optimized backward attention kernels

---

## Faze 8.1: SDPA Backward Performance

Status: `[x]`

### Proc

Faze 8 zavedla dense scalar cpu1 SDPA backward specializaci pro dQ/dK/dV.
Funkcne je to spravna prvni route, ale benchmark ukazal, ze vykonnostni
problem neni primarne `MEMORY_SEGMENT`. Hlavni bottleneck je dK algoritmus a
spatne planovani paralelizace.

Do teto faze patri performance opravy pro existujici dense F32/F64
`JAVA_ARRAY` a `MEMORY_SEGMENT` route a Java Vector API backward cesta pro
dense F32/F64 `JAVA_ARRAY` i `MEMORY_SEGMENT`. Nepatri sem strided/view
materializace, BF16, BLAS route ani obecny blocked/tiled attention provider.

### Benchmark Evidence

Docasny runner mimo repo:

```text
/private/tmp/SdpaBackwardCpu1Benchmark.java
```

Runner meril samotny prepared cpu1 SDPA backward step pres:

```java
new Cpu1Backend().execute(step.compiledNode(), step.metadata(), context);
```

Tedy ne compile, ne prepare a ne cely forward/backward graf.

Default runtime:

```text
shape = large-b4-h8-t64-d32
warmup = 3
iters = 7
workers = 16
```

Namereno:

```text
large b4 h8 t64 d32, F32
dQ array     11.631 ms
dQ segment   10.378 ms
dK array    486.422 ms
dK segment  531.499 ms
dV array      8.670 ms
dV segment    8.817 ms

large b4 h8 t64 d32, F64
dQ array      9.554 ms
dQ segment   10.107 ms
dK array    492.545 ms
dK segment  530.938 ms
dV array      8.653 ms
dV segment    8.949 ms
```

Forced chunk experiment:

```text
shape = large-b4-h8-t64-d32
dtype = F32
grad = dK
forceAttentionChunk = 64
warmup = 1
iters = 3
```

Namereno:

```text
dK array    52.597 ms
dK segment  52.152 ms
```

Zaver:

```text
1. dK je o rad az dva rady pomalejsi nez dQ/dV.
2. Default launch config ukazuje workers=16, ale chunk=16384 pri rows=2048,
   takze se vytvori jen jeden task a realne se neparalelizuje.
3. Po forced chunk=64 spadne dK z ~486-531 ms na ~52 ms.
4. MEMORY_SEGMENT neni hlavni bottleneck; po spravnem chunkingu je segment dK
   prakticky stejne rychly jako array dK.
```

### Cilovy Stav

Po fazi 8.1 ma cpu1 SDPA backward:

- rozhodovat paralelizaci podle odhadovane prace, ne podle samotneho
  `rowCount`
- pro dK nepocitat stejne `dScores` opakovane pro kazdy key
- pouzivat scratch velikost podle `outputKind`
- zachovat explicitni dense/no-offset kontrakt
- zachovat explicitni `JAVA_ARRAY` vs `MEMORY_SEGMENT` runtime storage route
- nezavadet skrytou materializaci
- nezavadet fallback do stareho `backend.cpu`

### Target Files

```text
src/main/java/backend/cpu1/prepare/Cpu1AttentionBackwardPreparer.java
src/main/java/backend/cpu1/prepare/Cpu1PreparedAttentionBackwardUnit.java
src/main/java/backend/cpu1/kernels/linalg/attention/backward/Cpu1AttentionBackwardLoops.java
src/test/java/backend/cpu1/Cpu1AttentionBackwardExecutionContractTest.java
src/test/java/backend/cpu1/Cpu1AttentionBackwardBenchmarkTest.java
todo/117-cpu1-to-cpu-parity-plan.md
```

`Cpu1AttentionBackwardBenchmarkTest` bude canonical benchmark test
`@Tag("benchmark")`. Docasny `/private/tmp` runner se do repa necommituje.

### Oprava 1: Launch Policy Podle Prace

Problem dnes:

```java
private static Cpu1LaunchConfig launchConfig(int rowCount, Cpu1PrepareConfig config) {
    CpuKernelConfig cpuKernelConfig = config.cpuKernelConfig();
    int maxWorkers = config.launchConfig().workerCount();
    if (maxWorkers <= 1 || rowCount <= 1 || rowCount < cpuKernelConfig.attentionParallelMinSize()) {
        return Cpu1LaunchConfig.singleThread();
    }
    int plannedWorkers = Math.min(maxWorkers, rowCount);
    int targets = Math.max(1, plannedWorkers * cpuKernelConfig.highCostTargetChunksPerWorker());
    int candidate = (Math.max(1, rowCount) + targets - 1) / targets;
    return Cpu1LaunchConfig.parallel(
            plannedWorkers,
            Math.max(cpuKernelConfig.minReductionChunkSize(), candidate)
    );
}
```

Proc je to spatne:

```text
rowCount = 2048
minReductionChunkSize = 16384
chunk = 16384
taskCount = ceil(2048 / 16384) = 1
```

Takze `workers=16` v prepare metadata neznamena skutecnou paralelizaci.

Cilovy kod:

```java
private static Cpu1LaunchConfig launchConfig(
        SdpaBackwardOutputKind outputKind,
        ShapeContract shape,
        Cpu1PrepareConfig config
) {
    if (!config.automaticLaunch()) {
        return config.launchConfig();
    }
    CpuKernelConfig cpuKernelConfig = config.cpuKernelConfig();
    if (cpuKernelConfig == null) {
        throw new IllegalArgumentException("Automatic cpu1 SDPA_BACKWARD dispatch requires CpuKernelConfig.");
    }

    int itemCount = launchItemCount(outputKind, shape);
    long workPerItem = estimatedWorkPerItem(outputKind, shape);
    long totalWork = Math.multiplyExact((long) itemCount, workPerItem);
    int maxWorkers = config.launchConfig().workerCount();

    if (maxWorkers <= 1 || itemCount <= 1 || totalWork < cpuKernelConfig.attentionParallelMinSize()) {
        return Cpu1LaunchConfig.singleThread();
    }

    int plannedWorkers = Math.min(maxWorkers, itemCount);
    int targetTasks = Math.max(1, plannedWorkers * cpuKernelConfig.highCostTargetChunksPerWorker());
    int candidateItemsPerChunk = (itemCount + targetTasks - 1) / targetTasks;
    int minItemsPerChunk = Math.max(
            1,
            (int) Math.max(1L, cpuKernelConfig.minReductionChunkSize() / Math.max(1L, workPerItem))
    );
    return Cpu1LaunchConfig.parallel(
            plannedWorkers,
            Math.max(minItemsPerChunk, candidateItemsPerChunk)
    );
}

private static int launchItemCount(SdpaBackwardOutputKind outputKind, ShapeContract shape) {
    return switch (outputKind) {
        case QUERY -> Math.multiplyExact(shape.batchCount(), shape.queryLen());
        case KEY -> Math.multiplyExact(shape.batchCount(), shape.keyLen());
        case VALUE -> Math.multiplyExact(shape.batchCount(), shape.keyLen());
    };
}

private static long estimatedWorkPerItem(SdpaBackwardOutputKind outputKind, ShapeContract shape) {
    return switch (outputKind) {
        case QUERY -> Math.addExact(
                Math.multiplyExact((long) shape.keyLen(), shape.valueDim()),
                Math.multiplyExact((long) shape.keyLen(), shape.depth())
        );
        case KEY -> Math.addExact(
                Math.multiplyExact((long) shape.queryLen(), Math.multiplyExact(shape.keyLen(), shape.valueDim())),
                Math.multiplyExact((long) shape.queryLen(), shape.depth())
        );
        case VALUE -> Math.multiplyExact((long) shape.queryLen(), shape.valueDim());
    };
}
```

Poznamka:

`launchItemCount(KEY)` zustava `batchCount * keyLen`, aby prvni oprava menila
jen chunking a zachovala existujici paralelni rozdeleni podle key row.
Algoritmicka oprava dK nize muze internim loopem pouzit jinou praci na chunk,
ale verejny launch contract zustane kompatibilni.

Tasky:

- [x] Nahradit stary `launchConfig(int rowCount, ...)`
- [x] Volat novou metodu jako
  `launchConfig(outputKind, shape, config)`
- [x] Pridat prepare test, ktery pro velky dK overi `chunkSize < rowCount`
- [x] Pridat prepare test, ktery pro maly dQ/dV zustane single-thread podle
  thresholdu

### Oprava 2: Scratch Sizing Podle Output Kind

Problem:

dQ a dK dnes pouzivaji scratch:

```text
dWeights[keyLen]
dScores[keyLen]
```

To staci pro jeden `computeScoreRow(...)`, ale optimalizovany dK potrebuje
ulozit `dScores` pro vice query rows, aby je nepocital znovu pro kazdy key.

Cilovy kod v:

```text
src/main/java/backend/cpu1/prepare/Cpu1PreparedAttentionBackwardUnit.java
```

```java
public int scratchElementsPerSlot() {
    return switch (outputKind) {
        case VALUE -> 0;
        case QUERY -> Math.multiplyExact(keyLen, 2);
        case KEY -> Math.addExact(
                Math.multiplyExact(keyLen, 2),
                Math.multiplyExact(queryLen, keyLen)
        );
    };
}

public int dScoresScratchOffset(int slotIndex) {
    if (outputKind != SdpaBackwardOutputKind.KEY) {
        throw new IllegalStateException("dScores matrix scratch is used only by SDPA dK.");
    }
    return Math.addExact(
            Math.multiplyExact(slotIndex, scratchElementsPerSlot()),
            Math.multiplyExact(keyLen, 2)
    );
}

public Cpu1ScratchBufferSpec scratchBufferSpec() {
    if (outputKind == SdpaBackwardOutputKind.VALUE) {
        return Cpu1ScratchBufferSpec.none();
    }
    int elements = Math.multiplyExact(scratchSlotCount, scratchElementsPerSlot());
    return dataType == DataType.FLOAT64
            ? Cpu1ScratchBufferSpec.arrays(0, elements, 0)
            : Cpu1ScratchBufferSpec.arrays(elements, 0, 0);
}
```

Proc:

```text
KEY scratch =
  temporary row dWeights[keyLen]
  temporary row dScores[keyLen]
  reusable dScoresMatrix[queryLen * keyLen]
```

Tahle pamet je per slot, tedy per parallel task. Pro typicky shape
`queryLen=64`, `keyLen=64` je to:

```text
2 * 64 + 64 * 64 = 4224 float/double prvku na slot
```

To je levne proti opakovanemu prepocitavani cele softmax-backward row pro kazdy
key.

Tasky:

- [x] Pridat `scratchElementsPerSlot()`
- [x] Pridat `dScoresScratchOffset(int slotIndex)`
- [x] Prepsat `scratchBufferSpec()`
- [x] Pridat test pro dQ scratch size
- [x] Pridat test pro dK scratch size
- [x] Pridat test pro dV no scratch

### Oprava 3: Prepsat dK Algoritmus

Problem dnes:

Aktualni dK loop dela zjednodusene:

```java
for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
        computeF32ScoreRow(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
        float dScore = scratch[dScoresBase + keyIndex];
        output[outputBase + depthIndex] += dScore * query[queryRowBase + depthIndex];
    }
}
```

`computeScoreRow(query)` vypocita `dScores` pro vsechny keys. Kdyz ho volame
uvnitr smycky pres keys, pocitame stejnou query row znovu a znovu.

Spravne:

```text
for each batch/head:
  for each query:
    compute dScores(query, all keys)
    uloz do dScoresMatrix[query, key]

  for each key:
    for each depth:
      dK[key, depth] = sum_query dScoresMatrix[query, key] * Q[query, depth]
```

#### F32 Array Cilovy Kod

Soubor:

```text
src/main/java/backend/cpu1/kernels/linalg/attention/backward/Cpu1AttentionBackwardLoops.java
```

```java
private static void runF32Key(
        Cpu1PreparedAttentionBackwardUnit unit,
        F32Inputs inputs,
        float[] output,
        float[] scratch
) {
    Cpu1RangeLauncher.launchIndexed(unit.batchCount(), unit.launchConfig(), (slot, start, end) -> {
        int slotBase = slot * unit.scratchElementsPerSlot();
        int dWeightsBase = slotBase;
        int dScoresBase = slotBase + unit.keyLen();
        int dScoresMatrixBase = unit.dScoresScratchOffset(slot);
        for (int batch = start; batch < end; batch++) {
            for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                computeF32ScoreRow(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                int matrixBase = dScoresMatrixBase + queryIndex * unit.keyLen();
                System.arraycopy(scratch, dScoresBase, scratch, matrixBase, unit.keyLen());
            }

            int outputBatchBase = batch * unit.keyLen() * unit.depth();
            int queryBatchBase = batch * unit.queryLen() * unit.depth();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                int outputBase = outputBatchBase + keyIndex * unit.depth();
                for (int depthIndex = 0; depthIndex < unit.depth(); depthIndex++) {
                    float sum = 0.0f;
                    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                        float dScore = scratch[dScoresMatrixBase + queryIndex * unit.keyLen() + keyIndex];
                        sum += dScore * inputs.query[queryBatchBase + queryIndex * unit.depth() + depthIndex];
                    }
                    output[outputBase + depthIndex] = sum;
                }
            }
        }
    });
}
```

#### F64 Array Cilovy Kod

```java
private static void runF64Key(
        Cpu1PreparedAttentionBackwardUnit unit,
        F64Inputs inputs,
        double[] output,
        double[] scratch
) {
    Cpu1RangeLauncher.launchIndexed(unit.batchCount(), unit.launchConfig(), (slot, start, end) -> {
        int slotBase = slot * unit.scratchElementsPerSlot();
        int dWeightsBase = slotBase;
        int dScoresBase = slotBase + unit.keyLen();
        int dScoresMatrixBase = unit.dScoresScratchOffset(slot);
        for (int batch = start; batch < end; batch++) {
            for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                computeF64ScoreRow(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                int matrixBase = dScoresMatrixBase + queryIndex * unit.keyLen();
                System.arraycopy(scratch, dScoresBase, scratch, matrixBase, unit.keyLen());
            }

            int outputBatchBase = batch * unit.keyLen() * unit.depth();
            int queryBatchBase = batch * unit.queryLen() * unit.depth();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                int outputBase = outputBatchBase + keyIndex * unit.depth();
                for (int depthIndex = 0; depthIndex < unit.depth(); depthIndex++) {
                    double sum = 0.0d;
                    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                        double dScore = scratch[dScoresMatrixBase + queryIndex * unit.keyLen() + keyIndex];
                        sum += dScore * inputs.query[queryBatchBase + queryIndex * unit.depth() + depthIndex];
                    }
                    output[outputBase + depthIndex] = sum;
                }
            }
        }
    });
}
```

#### F32 Segment Cilovy Kod

```java
private static void runF32KeySegment(
        Cpu1PreparedAttentionBackwardUnit unit,
        F32SegmentInputs inputs,
        MemorySegment output,
        float[] scratch
) {
    Cpu1RangeLauncher.launchIndexed(unit.batchCount(), unit.launchConfig(), (slot, start, end) -> {
        int slotBase = slot * unit.scratchElementsPerSlot();
        int dWeightsBase = slotBase;
        int dScoresBase = slotBase + unit.keyLen();
        int dScoresMatrixBase = unit.dScoresScratchOffset(slot);
        for (int batch = start; batch < end; batch++) {
            for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                computeF32ScoreRowSegment(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                int matrixBase = dScoresMatrixBase + queryIndex * unit.keyLen();
                System.arraycopy(scratch, dScoresBase, scratch, matrixBase, unit.keyLen());
            }

            int outputBatchBase = batch * unit.keyLen() * unit.depth();
            int queryBatchBase = batch * unit.queryLen() * unit.depth();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                int outputBase = outputBatchBase + keyIndex * unit.depth();
                for (int depthIndex = 0; depthIndex < unit.depth(); depthIndex++) {
                    float sum = 0.0f;
                    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                        float dScore = scratch[dScoresMatrixBase + queryIndex * unit.keyLen() + keyIndex];
                        sum += dScore * f32(inputs.query, queryBatchBase + queryIndex * unit.depth() + depthIndex);
                    }
                    setF32(output, outputBase + depthIndex, sum);
                }
            }
        }
    });
}
```

#### F64 Segment Cilovy Kod

```java
private static void runF64KeySegment(
        Cpu1PreparedAttentionBackwardUnit unit,
        F64SegmentInputs inputs,
        MemorySegment output,
        double[] scratch
) {
    Cpu1RangeLauncher.launchIndexed(unit.batchCount(), unit.launchConfig(), (slot, start, end) -> {
        int slotBase = slot * unit.scratchElementsPerSlot();
        int dWeightsBase = slotBase;
        int dScoresBase = slotBase + unit.keyLen();
        int dScoresMatrixBase = unit.dScoresScratchOffset(slot);
        for (int batch = start; batch < end; batch++) {
            for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                computeF64ScoreRowSegment(unit, inputs, batch, queryIndex, scratch, dWeightsBase, dScoresBase);
                int matrixBase = dScoresMatrixBase + queryIndex * unit.keyLen();
                System.arraycopy(scratch, dScoresBase, scratch, matrixBase, unit.keyLen());
            }

            int outputBatchBase = batch * unit.keyLen() * unit.depth();
            int queryBatchBase = batch * unit.queryLen() * unit.depth();
            for (int keyIndex = 0; keyIndex < unit.keyLen(); keyIndex++) {
                int outputBase = outputBatchBase + keyIndex * unit.depth();
                for (int depthIndex = 0; depthIndex < unit.depth(); depthIndex++) {
                    double sum = 0.0d;
                    for (int queryIndex = 0; queryIndex < unit.queryLen(); queryIndex++) {
                        double dScore = scratch[dScoresMatrixBase + queryIndex * unit.keyLen() + keyIndex];
                        sum += dScore * f64(inputs.query, queryBatchBase + queryIndex * unit.depth() + depthIndex);
                    }
                    setF64(output, outputBase + depthIndex, sum);
                }
            }
        }
    });
}
```

Proc je to lepsi:

```text
Predtim:
  pro kazdy key znovu pocitej dScores pro kazdy query

Potom:
  pro kazdy query pocitej dScores jednou
  potom jen dScoresMatrix^T @ Q
```

Segment specific win:

Stara segment dK cesta delala read-modify-write:

```java
setF32(output, outputIndex, f32(output, outputIndex) + delta);
```

Nova cesta drzi `sum` v lokalni promene a do segmentu zapise az finalni
hodnotu:

```java
setF32(output, outputBase + depthIndex, sum);
```

To je pro `MemorySegment` vyrazne vhodnejsi hot-path chovani.

Tasky:

- [x] Prepsat `runF32Key`
- [x] Prepsat `runF64Key`
- [x] Prepsat `runF32KeySegment`
- [x] Prepsat `runF64KeySegment`
- [x] Zachovat `runF32Query`, `runF64Query`, `runF32Value`, `runF64Value`
  beze zmen mimo pripadneho `slotBase` vypoctu
- [x] Pridat numerickou paritu proti baseline pro dK F32/F64 array
- [x] Pridat numerickou paritu proti baseline pro dK F32/F64 segment

### Oprava 4: Canonical Benchmark Test

Do repa pridat benchmark test:

```text
src/test/java/backend/cpu1/Cpu1AttentionBackwardBenchmarkTest.java
```

Skeleton:

```java
package backend.cpu1;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("benchmark")
final class Cpu1AttentionBackwardBenchmarkTest {
    private static final int WARMUP_ITERATIONS = 3;
    private static final int MEASURE_ITERATIONS = 7;

    @Test
    void benchmarkDenseSdpaBackwardArrayVsSegment() {
        // cases:
        // - medium b2 h4 t64 d32
        // - large b4 h8 t64 d32
        // - FLOAT32/FLOAT64
        // - QUERY/KEY/VALUE
        // - JAVA_ARRAY/MEMORY_SEGMENT
        //
        // Measure only Cpu1Backend.execute(step.compiledNode(), step.metadata(), context).
        // Print median/p90/min/max and prepared launch workers/chunk.
    }
}
```

Report musi obsahovat:

```text
shape
dtype
outputKind
storageKind
kernelId
workers
chunkSize
rowCount
medianMs
p90Ms
segment/array ratio
```

Benchmark test se nepousti v defaultnim overeni, jen rucne:

```bash
./gradlew test --tests backend.cpu1.Cpu1AttentionBackwardBenchmarkTest
```

Tasky:

- [x] Pridat benchmark test
- [x] V reportu tisknout launch config
- [x] V reportu tisknout array vs segment ratio
- [x] Pridat do teto faze nove benchmark evidence po implementaci

Historical scalar-only local benchmark evidence from
`./gradlew test --tests backend.cpu1.Cpu1AttentionBackwardBenchmarkTest` on
2026-06-22:

```text
shape,dtype,outputKind,storageKind,kernelId,workers,chunkSize,rowCount,itemCount,medianMs,p90Ms,minMs,maxMs,segmentArrayRatio
medium-b2-h4-t64-d32,FLOAT32,QUERY,JAVA_ARRAY,SDPA_BACKWARD_DQ_F32_ARRAY_DENSE_SCALAR,16,32,512,512,0.424458,6.122384,0.374667,14.600584,1.0000
medium-b2-h4-t64-d32,FLOAT32,QUERY,MEMORY_SEGMENT,SDPA_BACKWARD_DQ_F32_SEGMENT_DENSE_SCALAR,16,32,512,512,0.435709,0.477867,0.416917,0.481292,1.0265
medium-b2-h4-t64-d32,FLOAT32,KEY,JAVA_ARRAY,SDPA_BACKWARD_DK_F32_ARRAY_DENSE_SCALAR,16,32,512,512,2.179541,3.319758,0.492250,3.657708,1.0000
medium-b2-h4-t64-d32,FLOAT32,KEY,MEMORY_SEGMENT,SDPA_BACKWARD_DK_F32_SEGMENT_DENSE_SCALAR,16,32,512,512,0.523375,2.351458,0.501167,4.930083,0.2401
medium-b2-h4-t64-d32,FLOAT32,VALUE,JAVA_ARRAY,SDPA_BACKWARD_DV_F32_ARRAY_DENSE_SCALAR,16,32,512,512,0.347542,0.367350,0.325250,0.377625,1.0000
medium-b2-h4-t64-d32,FLOAT32,VALUE,MEMORY_SEGMENT,SDPA_BACKWARD_DV_F32_SEGMENT_DENSE_SCALAR,16,32,512,512,0.321209,0.365217,0.298000,0.369292,0.9242
medium-b2-h4-t64-d32,FLOAT64,QUERY,JAVA_ARRAY,SDPA_BACKWARD_DQ_F64_ARRAY_DENSE_SCALAR,16,32,512,512,2.074041,2.907991,1.478875,3.147416,1.0000
medium-b2-h4-t64-d32,FLOAT64,QUERY,MEMORY_SEGMENT,SDPA_BACKWARD_DQ_F64_SEGMENT_DENSE_SCALAR,16,32,512,512,0.400000,0.447325,0.387875,0.493375,0.1929
medium-b2-h4-t64-d32,FLOAT64,KEY,JAVA_ARRAY,SDPA_BACKWARD_DK_F64_ARRAY_DENSE_SCALAR,16,32,512,512,2.259708,2.667325,0.529959,3.025625,1.0000
medium-b2-h4-t64-d32,FLOAT64,KEY,MEMORY_SEGMENT,SDPA_BACKWARD_DK_F64_SEGMENT_DENSE_SCALAR,16,32,512,512,2.591041,3.611642,0.583708,3.690917,1.1466
medium-b2-h4-t64-d32,FLOAT64,VALUE,JAVA_ARRAY,SDPA_BACKWARD_DV_F64_ARRAY_DENSE_SCALAR,16,32,512,512,0.318791,0.356025,0.313791,0.358125,1.0000
medium-b2-h4-t64-d32,FLOAT64,VALUE,MEMORY_SEGMENT,SDPA_BACKWARD_DV_F64_SEGMENT_DENSE_SCALAR,16,32,512,512,0.349458,0.376267,0.315042,0.393292,1.0962
large-b4-h8-t64-d32,FLOAT32,QUERY,JAVA_ARRAY,SDPA_BACKWARD_DQ_F32_ARRAY_DENSE_SCALAR,16,128,2048,2048,1.077916,1.120308,1.048667,1.125709,1.0000
large-b4-h8-t64-d32,FLOAT32,QUERY,MEMORY_SEGMENT,SDPA_BACKWARD_DQ_F32_SEGMENT_DENSE_SCALAR,16,128,2048,2048,1.156125,1.168941,1.136166,1.176667,1.0726
large-b4-h8-t64-d32,FLOAT32,KEY,JAVA_ARRAY,SDPA_BACKWARD_DK_F32_ARRAY_DENSE_SCALAR,16,128,2048,2048,1.024500,1.104341,0.981625,1.105291,1.0000
large-b4-h8-t64-d32,FLOAT32,KEY,MEMORY_SEGMENT,SDPA_BACKWARD_DK_F32_SEGMENT_DENSE_SCALAR,16,128,2048,2048,1.165667,1.244300,1.107584,1.273500,1.1378
large-b4-h8-t64-d32,FLOAT32,VALUE,JAVA_ARRAY,SDPA_BACKWARD_DV_F32_ARRAY_DENSE_SCALAR,16,128,2048,2048,0.909375,0.962075,0.878083,0.985875,1.0000
large-b4-h8-t64-d32,FLOAT32,VALUE,MEMORY_SEGMENT,SDPA_BACKWARD_DV_F32_SEGMENT_DENSE_SCALAR,16,128,2048,2048,0.907750,0.930459,0.853125,0.949459,0.9982
large-b4-h8-t64-d32,FLOAT64,QUERY,JAVA_ARRAY,SDPA_BACKWARD_DQ_F64_ARRAY_DENSE_SCALAR,16,128,2048,2048,1.024375,1.090100,0.948583,1.129125,1.0000
large-b4-h8-t64-d32,FLOAT64,QUERY,MEMORY_SEGMENT,SDPA_BACKWARD_DQ_F64_SEGMENT_DENSE_SCALAR,16,128,2048,2048,1.154584,1.197500,1.044583,1.224500,1.1271
large-b4-h8-t64-d32,FLOAT64,KEY,JAVA_ARRAY,SDPA_BACKWARD_DK_F64_ARRAY_DENSE_SCALAR,16,128,2048,2048,1.118583,1.128450,1.103584,1.134750,1.0000
large-b4-h8-t64-d32,FLOAT64,KEY,MEMORY_SEGMENT,SDPA_BACKWARD_DK_F64_SEGMENT_DENSE_SCALAR,16,128,2048,2048,1.125667,1.216583,1.063500,1.217333,1.0063
large-b4-h8-t64-d32,FLOAT64,VALUE,JAVA_ARRAY,SDPA_BACKWARD_DV_F64_ARRAY_DENSE_SCALAR,16,128,2048,2048,0.934500,0.954267,0.852708,0.969167,1.0000
large-b4-h8-t64-d32,FLOAT64,VALUE,MEMORY_SEGMENT,SDPA_BACKWARD_DV_F64_SEGMENT_DENSE_SCALAR,16,128,2048,2048,0.936292,0.958067,0.921917,0.961542,1.0019
```

Current benchmark coverage now prints separate Java-array scalar,
Java-array Vector API, MemorySegment scalar, and MemorySegment Vector API rows
via
`storageKind,vectorizationKind,kernelId,...,medianVsArrayScalar,medianVsSegmentScalar`,
so future runs can compare SDPA backward vector support directly against the
scalar array and scalar segment baselines.

### Deferred Follow-Up: SDPA Backward Segment-Specific Vector Calibration

Status: `[deferred]`

Soucasny stav:

- cpu1 ma samostatne native elementwise thresholdy:
  - `cpu.nativeF32CheapVectorMinSize`
  - `cpu.nativeF64CheapVectorMinSize`
- attention/SDPA ma jen hrube obecne thresholdy:
  - `cpu.attentionVectorMinSize`
  - `cpu.attentionParallelMinSize`
- SDPA backward prepare dnes vybira Vector API binarne podle
  `Cpu1PrepareConfig.vectorizationKind()`, `storageKind` a `dtype`; nepouziva
  route-specific kalibrovany threshold pro `JAVA_ARRAY` vs `MEMORY_SEGMENT`
  ani pro `QUERY`/`KEY`/`VALUE`.

Proc to nechavame jako follow-up:

Benchmark po zavedeni `MEMORY_SEGMENT` Vector API ukazal, ze prinos neni
homogenni. Nektere kombinace se zrychli, jine jsou neutralni a nektere se muzou
zpomalit. Proto nechceme hardcodovat pravidla typu "segment dK vector vzdy
zapni/vypni" primo do kernel prepareru.

Cilovy follow-up:

```text
storageKind: JAVA_ARRAY | MEMORY_SEGMENT
dtype:       FLOAT32 | FLOAT64
outputKind:  QUERY | KEY | VALUE
mode:        SCALAR | VECTOR
```

Kalibrace ma pro kazdou route zmerit scalar/vector a nastavit threshold podle
odhadovane prace, ne jen podle poctu vystupnich prvku.

Priklad ciloveho prepare pravidla:

```java
private static Cpu1VectorizationKind attentionBackwardVectorizationKind(
        SdpaBackwardOutputKind outputKind,
        ShapeContract shape,
        DataType dataType,
        Cpu1StorageKind storageKind,
        Cpu1PrepareConfig config
) {
    if (!config.automaticVectorization()) {
        return config.vectorizationKind();
    }
    if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64) {
        return Cpu1VectorizationKind.SCALAR;
    }
    long work = estimatedVectorWork(outputKind, shape);
    long threshold = config.cpuKernelConfig()
            .attentionBackwardVectorMinWork(storageKind, dataType, outputKind);
    return work >= threshold ? Cpu1VectorizationKind.VECTOR : Cpu1VectorizationKind.SCALAR;
}
```

Poznamka:

Metoda `attentionBackwardVectorMinWork(...)` zatim neexistuje. Follow-up musi
nejdriv navrhnout profilovy/konfiguracni tvar tak, aby nezanesl dalsi
hardcoded vyjimky do cpu1. Pokud route vychazi opakovane pomaleji ve vector
rezimu, kalibrace muze nastavit velmi vysoky threshold a tim vector pro danou
kombinaci prakticky vypnout.

Tasky follow-upu:

- [ ] Navrhnout konfiguracni tvar pro SDPA backward vector thresholdy
      rozliseny podle `storageKind`, `dtype` a `outputKind`
- [ ] Rozsirit runtime/profile IO o nove thresholdy bez rozbiti starych
      profilu
- [ ] Prepsat `Cpu1AttentionBackwardPreparer` tak, aby pri
      `automaticVectorization()` pouzival kalibrovany threshold
- [ ] Rozsirit attention calibration step o SDPA backward route matrix
- [ ] Pridat prepare tests: pod threshold scalar, nad threshold vector pro
      `JAVA_ARRAY` i `MEMORY_SEGMENT`
- [ ] Nechat benchmark reportovat scalar/vector ratio tak, aby bylo videt,
      ktere route kalibrace vypnula

### Non-Goals Pro Fazi 8.1

Do teto faze nepatri:

- BF16 SDPA backward
- strided/view SDPA backward inputs
- hidden contiguous materialization
- OpenBLAS/BLAS route pro attention backward
- obecny blocked/tiled multi-output attention backward provider
- zmena graph lowering ownership mimo nutne testy

Tyto oblasti zustavaji samostatne:

```text
strided/view materialization -> todo/118-cpu1-graph-input-materialization-plan.md
general matmul epilogue      -> todo/119-general-matmul-epilogue-ir-plan.md
blocked/tiled attention      -> future optimized attention provider plan
```

### Overeni

Po implementaci faze 8.1 spustit:

```bash
./gradlew classes
./gradlew test --tests backend.cpu1.Cpu1AttentionBackwardExecutionContractTest
./gradlew test --tests graph.compile.planning.region.DefaultRegionOptimizerTest
./gradlew test --tests backend.cpu1.Cpu1AttentionBackwardBenchmarkTest
git diff --check
```

Ak `./gradlew test --tests ...` selze pred spustenim cilenych testu kvuli
globalnimu `compileTestJava` problemu, zaznamenat to ve finalnim reportu a
minimalne zachovat:

```bash
./gradlew classes
```

### Definition Of Done

Faze 8.1 je hotova az kdyz:

- [x] dK default prepared launch config vytvari vice nez jeden task pro velke
  attention shapes
- [x] dK F32/F64 array numericky sedi proti baseline
- [x] dK F32/F64 segment numericky sedi proti baseline
- [x] dK large F32 default benchmark uz neni radove pomalejsi kvuli
  `chunkSize > rowCount`
- [x] segment dK zustava blizko array dK po opravach
- [x] dQ/dV benchmark neukaze regresi proti stavu pred fazi
- [x] zadna nova hidden materialization/fallback cesta nebyla pridana

---

## Faze 9: cpu1 Storage Mode Contract

Status: `[x]`

### Proc

Faze 9 se prepisuje z puvodniho per-op native binding navrhu na jednotny
`cpu1 Storage Mode Contract`. `Tensor` nema byt druhy zdroj pravdy pro
execution storage mode a nezavadime `Tensor.toNative()`.

Execution storage mode je runtime/prepare contract:

```text
RuntimeConfig / CpuStorageProfile / Cpu1PrepareConfig
  CPU_ARRAY  -> Cpu1StorageKind.JAVA_ARRAY
  CPU_NATIVE -> Cpu1StorageKind.MEMORY_SEGMENT
```

Prepared cpu1 run nema nahodne michat array/native storage. Materializace je
boundary/runtime vec, musi byt traceovana a v strict/no-materialization rezimu
se ridi runtime/config policy, ne ad-hoc kontrolami uvnitr family kernelu.

### Cile

- sjednotit cpu1 execute binding podle `preparedUnit.storageKind()`
- `JAVA_ARRAY`: `requireCpuReadable(...)` a `Cpu1TensorView.fromTensor(...)`
- `MEMORY_SEGMENT`: `requireNativeReadable(...)` a
  `Cpu1TensorView.fromNativeStorage(...)`
- vystupy pres existujici runtime boundary:
  - array vystup: `Cpu1TensorView.fromTensor(...)` + `markCpuCurrent(...)`
  - native vystup: `requireNativeOutputStorage(...)` +
    `attachNativeStorage(...)`
- zadne skryte array/native michani uvnitr prepared runu
- materializace pouze pres runtime/boundary API a s trace informaci
- strict/no-materialization jako runtime/config policy
- BF16 compute policy:
  - akumulace ve F32/F64 podle op
  - vystup do BF16 az na konci
  - zadna snaha predstirat nativni BF16 arithmetic v Java hot path

### Non-Goals

- zadny `Tensor.toNative()`
- zadny cpu1 memory planner
- zadne per-op skryte michani `JAVA_ARRAY`/`MEMORY_SEGMENT`
- zadna residency logika v kernel hot path
- zadna prima zavislost `backend.cpu1` na `graph.compile.planning.memory`

### Audit Aktualniho cpu1 Storage Bindingu

Audit scope:

```bash
rg -n "requireCpuReadable|requireNativeReadable|requireNativeOutputStorage|attachNativeStorage|residencyForNodeId|nativeStorageForNodeId|requireNativeCurrent|graph\\.compile\\.planning\\.memory" src/main/java/backend/cpu1
```

Zisteny stav:

- Primy binding podle `preparedUnit.storageKind()` uz maji zejmena:
  - `exec/Cpu1DTypeExecutableUnit.java`: `MEMORY_SEGMENT` pouziva
    `requireNativeReadable` + `requireNativeOutputStorage`, `JAVA_ARRAY`
    pouziva `requireCpuReadable`.
  - `exec/Cpu1IndexExecutableUnit.java`: vetvi `MEMORY_SEGMENT` a
    `JAVA_ARRAY` explicitne a vytvari odpovidajici `Cpu1TensorView`.
  - `exec/Cpu1Conv2dExecutableUnit.java`,
    `exec/Cpu1MaxPool2dExecutableUnit.java`,
    `exec/Cpu1AvgPool2dExecutableUnit.java`: vstupy/vystupy binduji podle
    storage kindu a validuji dense/no-offset runtime view.
  - `exec/Cpu1ElementwiseExecutableUnit.java` a
    `exec/Cpu1FusedElementwiseExecutableUnit.java`: native vetev pouziva
    `requireNativeReadable` + `requireNativeOutputStorage`, array vetev
    zatim jde rovnou pres `Cpu1TensorView.fromTensor(...)` bez explicitniho
    `requireCpuReadable(...)`.
- Kernel-level input binding podle storage kindu je stale rozptyleny v
  rodinach:
  - reductions: `Cpu1SumMeanReductionLoops`, `Cpu1CumSumReductionLoops`,
    `Cpu1SoftmaxReductionLoops`, `Cpu1ArgMaxReductionLoops`,
    `Cpu1BoolReductionLoops`, `Cpu1MinMaxProdReductionLoops`
  - loss: `Cpu1MseLossLoops`, `Cpu1NllLossLoops`,
    `Cpu1DenseCrossEntropyLossLoops`, `Cpu1CrossEntropyLossIndicesLoops`
  - matmul array cesty: `Cpu1JavaScalarMatmulLoops`,
    `Cpu1JavaVectorMatmulLoops`, `Cpu1OpenBlasArrayMatmulLoops`
  - layout materialization: `Cpu1LayoutKernelSupport`
- Native output allocation/publish se uz deje pres
  `requireNativeOutputStorage(...)` a `attachNativeStorage(...)` v:
  - executable wrappers: elementwise, fused elementwise, dtype, index,
    conv2d, maxpool2d, avgpool2d, layernorm, rmsnorm
  - kernel families: reductions, loss, layout, OpenBLAS native matmul,
    attention forward a attention backward
- Ad-hoc residency/native-current kontroly zustavaji v:
  - `exec/Cpu1LayerNormExecutableUnit.java`: `requireCpuArrayCurrent`,
    `requireNativeCurrent`, `residencyForNodeId`, `nativeStorageForNodeId`
  - `exec/Cpu1RmsNormExecutableUnit.java`: stejne ad-hoc helpery
  - `kernels/matmul/Cpu1OpenBlasNativeSegmentMatmulLoops.java`:
    `requireNativeCurrent`
  - `kernels/linalg/attention/Cpu1AttentionLoops.java`: array/native current
    helpery
  - `kernels/linalg/attention/backward/Cpu1AttentionBackwardLoops.java`:
    array/native current helpery
  - `kernels/layout/Cpu1LayoutKernelSupport.java`: `canReadNative` /
    `canReadAllNative` sahaji primo na `residencyForNodeId` a
    `nativeStorageForNodeId`
- Prima zavislost z `src/main/java/backend/cpu1` na
  `graph.compile.planning.memory` nebyla nalezena (`rg` bez matchu).

Audit zaver:

```text
cpu1 uz ma cast pozadovaneho contractu v executable wrappers a ve
family helper metodach. Nedokonceny kus je sjednoceni boundary bindingu
a odstraneni ad-hoc residency rozhodovani z family/kernel hot path.
```

Implementacni vysledek:

- `Cpu1OpenBlasNativeSegmentMatmulLoops`, `Cpu1AttentionLoops`,
  `Cpu1AttentionBackwardLoops` a `Cpu1LayoutKernelSupport` uz nebrouzdaji
  primo v `residencyForNodeId` / `nativeStorageForNodeId` ani v lokalnich
  `requireNativeCurrent` / `requireCpuArrayCurrent` helperech.
- Kontrolni grep:

  ```bash
  rg -n "residencyForNodeId|nativeStorageForNodeId|requireNativeCurrent|requireCpuArrayCurrent|canReadNative|canReadAllNative|graph\\.compile\\.planning\\.memory" src/main/java/backend/cpu1
  ```

  nema zadny match.
- Materializace vstupu pro `JAVA_ARRAY` / `MEMORY_SEGMENT` jde pres
  `requireCpuReadable(...)` / `requireNativeReadable(...)`; actual
  materialization zustava v `CpuMaterializationTrace`.
- Native output publish zustava pres `requireNativeOutputStorage(...)` a
  `attachNativeStorage(...)`.

### Tasky

- [x] 1. Aktualizovat fazi 9 v planu 117 na `cpu1 Storage Mode Contract`.
- [x] 2. Auditovat aktualni cpu1 storage binding pres `rg` nad
  `src/main/java/backend/cpu1` a zapsat vysledek sem.
- [x] 3. Sjednotit prepare-time mapping `CPU_ARRAY`/`CPU_NATIVE` ->
  `Cpu1StorageKind.JAVA_ARRAY`/`Cpu1StorageKind.MEMORY_SEGMENT` jako jediny
  zdroj pravdy pro cpu1 execution storage mode.
- [x] 4. Sjednotit execute boundary binding tak, aby prepared run pro
  `JAVA_ARRAY` volal `requireCpuReadable(...)` a pro `MEMORY_SEGMENT` volal
  `requireNativeReadable(...)` pred vytvorenim `Cpu1TensorView`.
- [x] 5. Odstranit ad-hoc `requireNativeCurrent` /
  `requireCpuArrayCurrent` / `residencyForNodeId` rozhodovani z family kernelu
  a nahradit ho runtime/config policy pro materializaci nebo strict fail.
- [x] 6. Zachovat native output publish pres `requireNativeOutputStorage(...)`
  a `attachNativeStorage(...)`; staticke trace atributy pro materializacni
  pocty nebyly doplneny, protoze skutecne copy-in/copy-out/alloc/reuse stavy
  patri do runtime trace (`CpuMaterializationTrace`) a Phase 10 tracingu.
- [x] 7. Overit BF16 policy v dotcenych rodinach: compute/accumulate ve
  F32/F64, final BF16 store.
- [x] 8. Doplnit native segment parity benchmark/coverage evidence az po
  implementaci contractu.

### Phase 9 Follow-Up

- Runtime array -> native materialization MVP podporuje jen dense contiguous
  CPU-array tensors bez `storageOffset`. Broadcast/strided CPU views pro
  `MEMORY_SEGMENT` prepared inputy ted failuji na runtime boundary s explicitni
  chybou misto skryteho per-kernel fallbacku. Rozsireni materializeru pro
  broadcast/strided view materializaci patri do samostatneho runtime/storage
  planu.
- Detailni staticke trace atributy pro allocation/reuse/copy-out nebyly
  pridany, protoze aktualni skutecne udalosti materializace se meri runtime
  tracem a Phase 10 ma samostatny trace/tuning gate.

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
            Operation.OpType.CONV2D,
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
8. Faze 7: layer/rms norm, LINEAR dense subset, pool/conv direct correctness
9. Faze 8: SDPA/attention a sirsi linear algebra
10. Faze 9-10: native/tuning/trace hardening a default-readiness

## Aktualni Known Gaps

Tento seznam se ma menit pri implementaci:

- [ ] cpu1 nema centralni parity coverage report
- [ ] reductions maji jen uzkou strided direct podporu pro SUM/MEAN F32/F64
- [deferred] strided/view loss input materialization policy je samostatny planning
  item v [todo/118-cpu1-graph-input-materialization-plan.md](118-cpu1-graph-input-materialization-plan.md);
  Faze 5 dense NLL/CrossEntropy scope je hotovy pro `JAVA_ARRAY` a dense
  contiguous `MEMORY_SEGMENT`
- [deferred] index family dense direct scope je hotovy pro `GATHER`,
  `GATHER_AXIS`, `TAKE_ALONG_AXIS`, `GATHER_ND`, `SCATTER_ADD`,
  `SCATTER_AXIS_ADD`, `SCATTER_ELEMENTS` a `SCATTER_ND`; otevrene zustavaji
  strided/view/offset index paths v planu 118 a pripadna deterministic parallel
  scatter cesta
- [deferred] LINEAR dense epilogue subset je hotovy; zobecneni concrete
  matmul-bias/relu specialization route na backend-neutral `MATMUL_EPILOGUE`
  payload je samostatny follow-up v
  [todo/119-general-matmul-epilogue-ir-plan.md](119-general-matmul-epilogue-ir-plan.md);
  strided/view LINEAR support zustava navazany na budouci matmul/view policy
- [~] SDPA/attention dense direct cpu1 parita je hotova vcetne F32/F64 Vector
  API forward routes pro `JAVA_ARRAY` i `MEMORY_SEGMENT` a F32/F64 SDPA
  backward specialized routes pro dense `JAVA_ARRAY`/`MEMORY_SEGMENT`,
  vcetne Java Vector API backward cesty pro `JAVA_ARRAY` i `MEMORY_SEGMENT`;
  strided/view, broader blocked/tiled optimization, BF16 backward casti a
  route-specific SDPA backward vector calibration pro `MEMORY_SEGMENT`
  zustavaji otevrene ve fazi 8/follow-up
- [x] `CONV2D`, `LAYER_NORM`, `RMS_NORM`, `MAX_POOL2D` a `AVG_POOL2D` dense
  contiguous no-offset `JAVA_ARRAY`/`MEMORY_SEGMENT` slices jsou hotove;
  `CONV2D` je direct correctness/fallback route, preferovana budouci
  `CONV2D -> UNFOLD2D -> MATMUL` optimalizace je odlozena
- [ ] native storage policy neni dokoncena napric rodinami
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
