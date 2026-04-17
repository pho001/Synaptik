# Tuning Knobs

Tento dokument popisuje skutečný tuning surface, ne hypotetický budoucí seznam.

Je potřeba rozlišovat dvě velké skupiny:

- platform runtime knobs
  - ukládají se do `PlatformRuntimeProfile`
  - ladí se v platform calibration
- graph policy knobs
  - žijí v `ExecutionProfile.optimizer()`
  - typicky se searchují v graph autotune

## Reading Guide

Pokud řešíš:

- co se dnes opravdu kalibruje per hardware
- co je workload-specific candidate mutace
- které knoby jsou už veřejné a které jen rezervované
- jaké candidate ranges dnes používají standardní calibration presets

pak jsi správně tady.

## Runtime Vs Graph Policy

### Platform Runtime Knobs

Patří sem:

- CPU thresholds
- tiles
- microkernels
- scheduler policy
- materialization thresholds
- numerics policy

Tyto knoby se mají sdílet napříč workloady na daném stroji.

### Graph Policy Knobs

Patří sem:

- `optimizer.stageOrder`
- `optimizer.rewrite.*`
- `optimizer.fuse.*`
- `optimizer.memory.*`

Tyto knoby jsou workload-sensitive. Nejsou součástí `PlatformRuntimeProfile`.

## Public Runtime Families

### `MATMUL`

Sem dnes patří:

- `runtime.blas.provider`
  - prakticky dnes `NONE` nebo `OPENBLAS_FFM`
- `runtime.blas.matmulMinWork`
- `runtime.blas.threads`
- `runtime.blas.f32RequireMgeK`
- `runtime.blas.f32MaxNOverK`
- `cpu.matMulParallelMinSize`
- `cpu.matMulTileM`
- `cpu.matMulTileN`
- `cpu.matMulTileK`
- `cpu.matMulMicroKernel`
- `cpu.attentionMatMulTileM`
- `cpu.attentionMatMulTileN`
- `cpu.attentionMatMulTileK`
- `cpu.attentionMatMulMicroKernel`

Knob, který je v runtime profilu uložený, ale dnešní standardní calibration presets ho nesweepují:

- `cpu.loopUnrollFactor`

### `FUSED_THRESHOLDS`

- `cpu.fusedCheapVectorMinSize`
- `cpu.fusedTranscendentalVectorMinSize`
- `cpu.fusedCheapParallelMinSize`
- `cpu.fusedTranscendentalParallelMinSize`

Tohle řídí scheduler decision pro fused node, ne volbu backendu.

### Fused ASM Width Knobs

Dnes jsou reálně součást tuning surface:

- `cpu.fusedCheapContiguousAsmVectorWidth`
- `cpu.fusedCheapStridedAsmVectorWidth`
- `cpu.fusedNonCheapContiguousAsmVectorWidth`
- `cpu.fusedNonCheapStridedAsmVectorWidth`

Tyhle width knoby jsou kalibrované po dispatch family, ne jedním globálním číslem.

To je důležitá realita:

- nejsou už jen interní experimentální nastavení
- standardní platform calibration je umí hledat

### `ELEMENTWISE_DISPATCH`

- `cpu.cheapVectorMinSize`
- `cpu.transcendentalVectorMinSize`
- `cpu.cheapParallelMinSize`
- `cpu.transcendentalParallelMinSize`

Týká se non-fused elementwise kernel families.

### `REDUCTION`

- `cpu.reductionVectorMinSize`
- `cpu.reductionParallelMinSize`
- `cpu.attentionVectorMinSize`
- `cpu.attentionParallelMinSize`
- `cpu.sumAccuracyMode`

`attention*` thresholds jsou uloženy v reduction profile family, protože patří do strukturovaných reduction-like kernels, ne do generického fused/elementwise dispatch.

### `SCHEDULER`

- `cpu.lowCostTargetChunksPerWorker`
- `cpu.mediumCostTargetChunksPerWorker`
- `cpu.highCostTargetChunksPerWorker`
- `cpu.minScalarChunkSize`
- `cpu.minVectorChunkSize`
- `cpu.minReductionChunkSize`
- `cpu.commonPoolLowCostMaxWorkPerWorker`

Tyhle knoby mají význam až po tom, co už je zvoleno, že se poběží paralelně.

### `MATERIALIZATION`

- `cpu.contiguousMaterializeThreshold`

Rozhoduje, od jaké velikosti je výhodnější non-contiguous input materiálizovat do contiguous temporary storage.

### `NUMERICS`

- `runtime.approximation.approxMode`
- `runtime.approximation.forceExactTranscendentals`

To jsou veřejné runtime policy knoby, ne jen lokální benchmark hack.

## Current Calibration Ranges

Následující rozsahy popisují to, co dnes používají standardní `PlatformCalibrationDefaults`, ne obecně všechny myslitelné hodnoty.

### Matmul

`blasThreads`

- `0`
- `1`
- `2`
- `4`

`matMulParallelMinSize`

- `100_000`
- `500_000`
- `2_000_000`

`f32ShapeHeuristics`

- `f32RequireMgeK`
  - `true`
  - `false`
- `f32MaxNOverK`
  - `1.5`
  - `2.0`
  - `3.0`
  - `4.0`
  - `6.0`

`matMulMicroKernel`

- `FLOAT64`
  - `F64_2X1`
  - `F64_4X1`
  - `F64_2X2`
- `FLOAT32`
  - `F32_2X4`
  - `F32_2X8`
  - `F32_4X2`
  - `F32_4X4`

`matMulTiles`

- `FLOAT64`
  - `16x64x32`
  - `32x64x32`
  - `32x64x64`
  - `32x128x64`
- `FLOAT32`
  - `32x64x64`
  - `32x128x64`
  - `64x128x64`
  - `64x128x128`
  - `64x256x128`

`attentionMatMulTiles`

- `FLOAT64`
  - `16x64x32`
  - `32x64x32`
  - `32x128x64`
- `FLOAT32`
  - `32x64x64`
  - `32x128x64`
  - `64x128x64`
  - `64x128x128`
  - `64x256x128`

### Fused Thresholds

`cheapVector`

- `64`
- `128`
- `256`
- `512`
- `1024`

`transcendentalVector`

- `16`
- `32`
- `64`
- `128`
- `256`

`cheapParallel`

- `4096`
- `8192`
- `16384`
- `32768`

`transcendentalParallel`

- `1024`
- `2048`
- `4096`
- `8192`

### Fused ASM Widths

Candidate widths se odvozují podle dtype a dostupné preferred vector species:

- vždy `1`
- pokud HW dovolí, pak i `2`
- pokud HW dovolí, pak i `4`
- v některých `F32/BF16 cheap contiguous` případech i `8`

To znamená:

- width space je family-specific
- není to jedno univerzální číslo pro všechny fused workloads

### Elementwise Dispatch

`cheapVector`

- `128`
- `256`
- `512`
- `1024`
- `2048`

`transcendentalVector`

- `32`
- `64`
- `128`
- `256`
- `512`

`cheapParallel`

- `8192`
- `16384`
- `32768`
- `65536`

`transcendentalParallel`

- `2048`
- `4096`
- `8192`
- `16384`

### Reduction

`reductionVector`

- `512`
- `2048`
- `8192`
- `16384`

`reductionParallel`

- `8192`
- `16384`
- `32768`
- `65536`

`attentionThresholds`

- `attentionVector`
  - `512`
  - `2048`
  - `8192`
  - `16384`
- `attentionParallel`
  - `2048`
  - `8192`
  - `16384`
  - `32768`

### Scheduler

Scheduler calibration dnes spíš lokálně refineuje okolí seed winner hodnot:

- target chunks per worker
- min chunk sizes
- common pool threshold

To je záměr. Scheduler knobs bývají silně závislé na tom, co už vyhrálo v ostatních families.

### Materialization

Candidate set vzniká kolem aktuálního thresholdu a doplňuje explicitní anchor body:

- `262_144`
- `524_288`
- `1_048_576`

### Numerics

`approxMode`

- `OFF`
- `TRAINING_ONLY`
- `ALWAYS`

`forceExactTranscendentals`

- `true`
- `false`

## Graph Policy Knobs

Tyhle knoby nejsou součást platform runtime profile, ale jsou součást `ExecutionProfile.optimizer()`.

### `optimizer.stageOrder`

Platné stage prvky:

- `AR`
- `CSE`
- `FUSE`
- `MEM`

Candidate spaces běžně používají:

- explicitní seznam stage orders
- constrained stage order space
- exhaustive permutation/subset space

### Rewrite Policy

Aktuálně dává smysl ladit hlavně:

- `optimizer.rewrite.conv2dLowering.mode`
  - `OFF`
  - `HEURISTIC`
  - `ALWAYS`

Piecewise lowering config také existuje, ale v běžném autotune workflow dnes nebývá hlavní knob surface.

## Knobs That Exist But Are Not Very Useful Today

### `runtime.fused.primaryBackend`

Technicky existuje ve fused execution policy.

Praktická realita na CPU dnes:

- meaningful backend je `ASM`

Takže to není zvlášť zajímavý knob pro standardní tuning.

### `runtime.fused.allowBackendFallback`

Technicky existuje, ale protože CPU fused prepare dnes stojí na ASM backendu, není to hlavní performance lever.

### `cpu.loopUnrollFactor`

Je uložený v runtime profilu, ale standardní platform calibration ho dnes nesweepuje.

## Example: Runtime Profile Candidate

Platform calibration kandidát může například měnit:

- `cpu.matMulTileM/N/K`
- `cpu.matMulMicroKernel`
- `cpu.matMulParallelMinSize`

ale stále jde o jeden konkrétní `PlatformRuntimeProfile`, ne o oddělený knob map.

## Example: Graph Candidate

Graph autotune kandidát může změnit:

- `optimizer.stageOrder`

nebo pro `CONV2D` workload:

- `optimizer.rewrite.conv2dLowering.mode`

Výsledkem je zase normální `ExecutionProfile`.

## Common Mistakes

- považovat runtime knoby za workload-specific graph policy
- myslet si, že fused ASM widths nejsou součást veřejného tuning surface
- chtít kalibrovat `runtime.fused.primaryBackend`, i když na CPU dnes dává smysl jen `ASM`
- ignorovat, že některé uložené runtime fields se dnes ve standardních presets nesweepují

## Related Docs

- architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)
- search: [SEARCH.md](./SEARCH.md)
- persistence: [PERSISTENCE.md](./PERSISTENCE.md)
