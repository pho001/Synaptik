# Tuning Knobs

## Contents

- [Purpose](#purpose)
- [Family Split](#family-split)
- [MATMUL](#matmul)
- [FUSED](#fused)
- [ELEMENTWISE_DISPATCH](#elementwise_dispatch)
- [REDUCTION](#reduction)
- [SCHEDULER](#scheduler)
- [MATERIALIZATION](#materialization)
- [NUMERICS](#numerics)
- [GRAPH_POLICY](#graph_policy)
- [Non-Public Knobs](#non-public-knobs)

## Purpose

This document is the public knob reference for the tuning package.

It explains:

- what each knob means
- which workflow uses it
- what candidate values are currently recommended

## Family Split

Families:

- `MATMUL`
- `FUSED`
- `ELEMENTWISE_DISPATCH`
- `REDUCTION`
- `SCHEDULER`
- `MATERIALIZATION`
- `NUMERICS`
- `GRAPH_POLICY`

## MATMUL

- `runtime.blas.matmulMinWork`
  - role:
    - platform calibration
    - benchmark option
  - values:
    - `32k, 64k, 128k, 256k, 512k, 1M, 2M, 4M, 8M`

- `runtime.blas.threads`
  - role:
    - platform calibration
    - benchmark option
  - values:
    - `0, 1, 2, 4, 8`

- `runtime.blas.f32RequireMgeK`
  - role:
    - platform calibration
    - benchmark option
  - values:
    - `true, false`

- `runtime.blas.f32MaxNOverK`
  - role:
    - platform calibration
    - benchmark option
  - values:
    - `1.5, 2.0, 3.0, 4.0, 6.0, 8.0`

- `cpu.loopUnrollFactor`
  - values:
    - `1, 2, 4, 8`

- `cpu.matMulTileM`
  - values:
    - `16, 32, 64`

- `cpu.matMulTileN`
  - values:
    - `16, 32, 64, 128`

- `cpu.matMulTileK`
  - values:
    - `16, 32, 64, 128`

- `cpu.matMulParallelMinSize`
  - values:
    - `64k, 128k, 256k, 512k, 1M, 2M, 4M`

## FUSED

- `cpu.fusedCheapVectorMinSize`
  - values:
    - `64, 128, 256, 512, 1024, 2048`

- `cpu.fusedTranscendentalVectorMinSize`
  - values:
    - `16, 32, 64, 128, 256, 512`

- `cpu.fusedCheapParallelMinSize`
  - values:
    - `4k, 8k, 16k, 32k, 64k, 128k`

- `cpu.fusedTranscendentalParallelMinSize`
  - values:
    - `1k, 2k, 4k, 8k, 16k, 32k`

## ELEMENTWISE_DISPATCH

- `cpu.cheapVectorMinSize`
  - values:
    - `128, 256, 512, 1024, 2048, 4096`

- `cpu.transcendentalVectorMinSize`
  - values:
    - `32, 64, 128, 256, 512, 1024`

- `cpu.cheapParallelMinSize`
  - values:
    - `8k, 16k, 32k, 64k, 128k, 256k`

- `cpu.transcendentalParallelMinSize`
  - values:
    - `2k, 4k, 8k, 16k, 32k, 64k`

## REDUCTION

- `cpu.reductionVectorMinSize`
  - values:
    - `128, 256, 512, 1024, 2048, 4096`

- `cpu.reductionParallelMinSize`
  - values:
    - `8k, 16k, 32k, 64k, 128k, 256k`

- `cpu.sumAccuracyMode`
  - values:
    - `FAST, KAHAN, NEUMAIER`

## SCHEDULER

These are refinement knobs over already selected parallel execution paths.

- `cpu.lowCostTargetChunksPerWorker`
  - values:
    - local refinement around current winner

- `cpu.mediumCostTargetChunksPerWorker`
  - values:
    - local refinement around current winner

- `cpu.highCostTargetChunksPerWorker`
  - values:
    - local refinement around current winner

- `cpu.minScalarChunkSize`
  - values:
    - local refinement around current winner

- `cpu.minVectorChunkSize`
  - values:
    - local refinement around current winner

- `cpu.minReductionChunkSize`
  - values:
    - local refinement around current winner

- `cpu.commonPoolLowCostMaxWorkPerWorker`
  - values:
    - local refinement around current winner

## MATERIALIZATION

- `cpu.contiguousMaterializeThreshold`
  - role:
    - platform calibration
    - benchmark option
  - meaning:
    - threshold from which non-contiguous tensors should be materialized before fast execution
  - values:
    - local refinement around current winner
    - typical practical range:
      - `4k` up to `1M+`

## NUMERICS

- `runtime.approximation.approxMode`
  - values:
    - `OFF, TRAINING_ONLY, ALWAYS`

- `runtime.approximation.forceExactTranscendentals`
  - values:
    - `true, false`

- `cpu.sumAccuracyMode`
  - values:
    - `FAST, KAHAN, NEUMAIER`

These are public tuning and benchmark knobs, but they are not mandatory in every default platform-calibration preset.

## GRAPH_POLICY

- `optimizer.stageOrder`
  - values:
    - valid subsets and permutations of:
      - `AR`
      - `CSE`
      - `FUSE`
      - `MEM`

- `optimizer.rewrite.conv2dLowering.mode`
  - values:
    - `OFF, HEURISTIC, ALWAYS`

## Non-Public Knobs

These may still exist internally, but they are not part of the target public tuning surface:

- `runtime.fused.primaryBackend`
- `runtime.fused.allowBackendFallback`
- `cpu.fusedAsmVectorWidth`
