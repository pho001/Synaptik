# 30-01 Summary: BF16 Storage, ABI, And Materialization Path

## Completed

- Added lowered DAG output dtype metadata and preserved it in Metal executable descriptors.
- Added output dtype to accelerator executable signatures so BF16 and FLOAT32 DAGs cannot share native cache entries.
- Extended Metal buffer allocation, upload, readback, and strided materialization for raw BF16 `short[]` storage.
- Extended Metal buffer validation to accept BF16 outputs only when executable metadata expects BF16.
- Added an additive native `synaptik_apple_mps_compile_partition_dtype_v3` compile symbol carrying node output dtype metadata.
- Kept legacy `synaptik_apple_mps_compile_partition_f32` as a wrapper that preserves FLOAT32 behavior.
- Updated native tensor-array and buffer execution byte sizing/type mapping to recognize BF16 descriptor code where the dtype-v3 path uses it.

## Verification

- `./gradlew classes`
- `./gradlew test --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest`
- `./gradlew metalTest`
- `nm -gU build/native/apple/libsynaptik_apple_mps.dylib | rg "compile_partition"`
- `git diff --check`

## Outcome

`METALBF16-01` and the storage side of `METALBF16-02` now have the transport metadata needed for scoped BF16 operation admission. BF16 is still not broadly admitted as native Metal compute until the Phase 30-02 primitive legality and parity work lands.
