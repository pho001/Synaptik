# Summary 24-02: Native Normalization Primitive Execution

**Status:** Complete
**Date:** 2026-05-01

## Completed

- Added Metal MPSGraph `ADD_SCALAR` execution using a Float32 scalar constant and graph addition.
- Added an explicit Metal native buffer test for a LayerNorm-like lowered sub-DAG.
- Extended CUDA native node metadata to track input1 rank, dimensions, and element count.
- Reworked CUDA buffer execution to allocate/reuse internal DAG intermediate buffers instead of requiring one caller output buffer per node.
- Added CUDA dense/broadcast execution support for normalization-required unary and binary primitives: `SUB`, `MUL`, `DIV`, `SQRT`, `INV`, `ADD_SCALAR`, and broadcast-capable `ADD`.
- Added CUDA suffix-broadcast validation with stable `unsupported broadcast` failure text before kernel launch.

## Verification

Passed:

```bash
./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest
```

Native Metal gate passed:

```bash
./gradlew metalTest
```

CUDA native compile was not run locally because `nvcc` is not available in this environment.

## Deviations from Plan

None. CUDA native validation remains capability/toolchain-gated.
