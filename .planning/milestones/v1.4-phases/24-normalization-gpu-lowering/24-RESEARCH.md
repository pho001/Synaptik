# Research 24: Normalization GPU Lowering

**Status:** Complete
**Date:** 2026-05-01

## What Matters For Planning

`LAYER_NORM` and `RMS_NORM` are currently matrix-visible fallback rows with stable `DEFERRED_FUSED_REGION` reasons. Phase 23 closed forward reductions, so Phase 24 can build normalization as a GPU sub-DAG instead of treating it as an opaque operation.

The existing normalization descriptors carry only `normalizedRank` and `epsilon`. Gamma and beta arrive as normal graph inputs. The CPU kernels compute groups by splitting the input into leading groups and trailing normalized elements; gamma/beta are contiguous tail parameters whose flat size equals the normalized trailing size.

## Required DAG Pieces

Existing shared DAG primitives already cover:

- `MEAN` with axis and keep-dims metadata from Phase 23.
- `SUB`, `MUL`, `ADD`, `SQRT`, `INV`.
- External inputs for gamma and beta.

Missing or incomplete pieces:

- Scalar epsilon addition is not represented as a shared DAG primitive. Add an internal accelerator DAG op such as `ADD_SCALAR`.
- CUDA native buffer execution currently needs broader dense elementwise support than `RELU`, `ADD`, and reductions. Normalization requires `SUB`, `MUL`, `SQRT`, `INV`, scalar add, and suffix broadcast for gamma/beta and keep-dims reduction outputs.
- CUDA compile metadata currently tracks input0 shape for reductions; normalization binary kernels need input1 shape metadata too.

## Lowering Shape

Layer norm lowering:

1. `mean = repeated MEAN(input, axis=last..tailStart, keepDims=true)`
2. `centered = SUB(input, mean)`
3. `square = MUL(centered, centered)`
4. `variance = repeated MEAN(square, axis=last..tailStart, keepDims=true)`
5. `varianceEps = ADD_SCALAR(variance, epsilon)`
6. `invStd = INV(SQRT(varianceEps))`
7. `normalized = MUL(centered, invStd)`
8. `scaled = MUL(normalized, gamma)`
9. `output = ADD(scaled, beta)`

RMS norm lowering:

1. `square = MUL(input, input)`
2. `meanSquares = repeated MEAN(square, axis=last..tailStart, keepDims=true)`
3. `meanSquaresEps = ADD_SCALAR(meanSquares, epsilon)`
4. `invRms = INV(SQRT(meanSquaresEps))`
5. `normalized = MUL(input, invRms)`
6. `output = MUL(normalized, gamma)`

## Validation Architecture

Plans must prove:

- Lowered DAG primitives exist for both normalization operations and encode epsilon and normalized trailing axes.
- Metal explicit shim can execute a simple normalization DAG or a representative subset with scalar epsilon addition and broadcasted gamma/beta.
- CUDA native execution has dense rank 1-4 elementwise kernels with suffix broadcast, or rejects unsupported cases before claiming support.
- Prepared execution selects GPU regions for legal `FLOAT32` layer/rms norm and no longer reports normalization as `DEFERRED_FUSED_REGION`.
- CPU parity tests compare output values with tolerance that accounts for reduction accumulation order.

## Risks

- Marking coverage supported before CUDA native execution supports all required primitives would recreate a hidden fallback problem.
- Gamma/beta broadcasting is the main shape-risk; suffix-only broadcast is enough for this phase and should be explicit.
- Multi-axis normalization must reduce trailing axes with `keepDims=true`; reducing without keep-dims will break subsequent broadcast.
