<!-- generated-by: gsd-doc-writer -->
# CPU BF16 Runtime

Navigation: [Index](index.md#recommended-reading-paths) | [Configuration](configuration.md#runtimeconfig) | [Backend Planning](backend-planning-and-regions.md#cpu-natural-regions) | [Native Bridges & BLAS](native-bridges-and-blas.md#bf16-and-batched-blas) | [Troubleshooting](troubleshooting.md#performance-regressions)

This document explains the current CPU `BFLOAT16` path and why BF16 is not automatically faster than `FLOAT32` on CPU.

## Term Map

| Term | Meaning |
|---|---|
| BF16 / `BFLOAT16` | 16-bit floating-point storage format with an 8-bit exponent and 7 explicit mantissa bits. It has roughly the exponent range of `FLOAT32`, but much less precision. |
| F16 / half | Different 16-bit floating-point format with a 5-bit exponent and 10 explicit mantissa bits. A Java `short` is not automatically F16. |
| Storage dtype | How tensor values are stored in memory, such as `short[]` for BF16 storage. |
| Compute dtype | The precision used while executing an operation. Current CPU BF16 elementwise operations generally compute in `float`. |
| Accumulation dtype | Precision used for reductions or dot-product accumulation. Some paths accumulate wider than storage dtype. |
| Packing | Converting a wider compute value, usually `float`, into BF16 bits. |
| Unpacking | Converting BF16 bits into a wider compute value, usually `float`. |
| Native BF16 arithmetic | Hardware or native-library operations that directly operate on BF16 data without Java-level per-element promote/round loops. |

## BF16 Is Not Java `short`

BF16 storage uses a Java `short[]` because BF16 is 16 bits wide. That does not mean Java arithmetic on `short` performs BF16 math.

Conceptually:

```text
FLOAT32 bits:
  sign: 1 bit
  exponent: 8 bits
  mantissa: 23 bits

BF16 bits:
  sign: 1 bit
  exponent: 8 bits
  mantissa: 7 bits
```

BF16 can often represent very large and very small magnitudes similarly to F32 because the exponent width is the same. It cannot represent values with the same fine precision because most mantissa bits are dropped.

Example:

```text
F32 value:
  1.234567

BF16 storage:
  approximately 1.234375
```

The exact rounded value depends on the conversion routine. The important point is that a BF16 value stored in `short[]` must be unpacked to a numeric type before ordinary Java arithmetic can use it.

## Current CPU Contract

Current CPU BF16 behavior is best described as:

```text
storage:      BF16 bits in short[]
elementwise:  unpack BF16 -> float, compute in float, pack float -> BF16
matmul:       BF16-aware paths exist; native acceleration depends on provider and shape gates
reductions:   may accumulate wider than BF16, often F32 or F64 depending on operation contract
publication:  controlled separately by PublicationPolicy
```

Relevant source areas:

- `src/main/java/tensor/BFloat16Storage.java`
- `src/main/java/backend/cpu/kernels/CpuDTypeOps.java`
- `src/main/java/backend/cpu/kernels/plan/CpuComputeContractResolver.java`
- `src/main/java/backend/cpu/kernels/elementwise/binary/bf16/*`
- `src/main/java/backend/cpu/kernels/elementwise/unary/bf16/*`
- `src/main/java/backend/cpu/kernels/linalg/matmul/bf16/*`
- `src/main/java/backend/cpu/kernels/linalg/matmul/blas/MatMulBlasBackend.java`

## Elementwise Example

For BF16 addition:

```java
Tensor y = a.add(b);
```

The CPU BF16 loop is conceptually:

```text
for each element i:
  float av = bf16_to_f32(a_short[i])
  float bv = bf16_to_f32(b_short[i])
  float cv = av + bv
  out_short[i] = f32_to_bf16(cv)
```

This can be slower than F32:

```text
F32 add:
  load float
  load float
  add float
  store float

BF16 add today:
  load short
  unpack to float
  load short
  unpack to float
  add float
  round/pack to BF16
  store short
```

BF16 moves fewer bytes, but it does extra conversion work. If the operation is compute-light, conversion can dominate.

## Chain Example

Consider:

```java
Tensor y = x.add(b).mul(scale).tanh();
```

The ideal BF16 CPU chain would keep values in a wider register or native BF16 vector lane across the whole chain:

```text
load once -> compute add/mul/tanh -> store once
```

The current architecture may split the chain into execution units. When that happens, each boundary can store BF16 and the next unit can unpack it again:

```text
unit 1 add:
  unpack x and b -> compute F32 -> pack BF16 temporary

unit 2 mul:
  unpack BF16 temporary and scale -> compute F32 -> pack BF16 temporary

unit 3 tanh:
  unpack BF16 temporary -> compute F32 tanh -> pack BF16 output
```

CPU region planning can group a large natural region, but the region optimizer may still split it into unit kernels and fused units. A "large CPU partition" therefore does not guarantee a single conversion-free BF16 chain.

## What Fusion Helps

CPU fusion can reduce repeated materialization and repeated pack/unpack inside elementwise chains.

Example:

```text
unfused:
  add stores BF16 temporary
  mul loads/unpacks that temporary
  tanh loads/unpacks another temporary

fused:
  one loop computes add, then mul, then tanh before storing output
```

Even in a fused loop, the current Java CPU path usually computes in F32. Fusion helps by reducing intermediate storage traffic and conversion boundaries. It does not magically create native BF16 arithmetic.

## `publishFloatContinuation`

Some CPU code can carry an opportunistic float continuation for a produced value. In plain language, that means a node may make its F32 result available to the next compatible consumer so the next step does not immediately unpack the freshly packed BF16 value.

This is useful, but it is not a full region-wide BF16 live-range optimizer.

It helps most when:

- the next consumer is compatible
- the execution order preserves the continuation
- there is a single-consumer chain
- the boundary does not require materialization for layout, publication, or region handoff

It helps less when:

- there are multiple consumers
- a reduction or matmul boundary interrupts the chain
- region optimization splits the chain
- a public tensor must be published
- a backend handoff requires materialized storage

## Reductions And Accumulation

Reductions are operations such as `sum`, `mean`, `min`, and `max` over one or more dimensions.

For BF16, reductions often should not accumulate in BF16 storage precision. A sum of many BF16 values accumulated in BF16 would lose precision quickly. Current CPU paths may promote accumulation to F32 or F64 depending on the operation contract and accuracy policy.

Example:

```text
sum of 4096 BF16 values:
  read BF16 storage
  unpack each value
  accumulate wider value
  store final result as target dtype
```

This is more numerically stable than pure BF16 accumulation, but it also means BF16 reductions are not necessarily faster than F32 reductions.

## Matmul And BLAS

Matmul is where BF16 can become useful on CPU, but only if the runtime path uses an implementation that is actually optimized for BF16.

Current layers:

- Java BF16 matmul executables exist.
- BLAS routing can be enabled through `BlasProvider.OPENBLAS_FFM`.
- Optional native BF16 GEMM support depends on the OpenBLAS build and exposed symbols.
- Shape gates still decide whether BLAS is eligible.

Important consequence:

```text
BF16 matmul with BLAS disabled or unavailable:
  may be slower than expected

BF16 matmul with native BF16 GEMM available and shape gates met:
  can become meaningfully faster
```

Check the calibrated runtime profile for BF16 matmul-heavy workloads. If the BF16 profile records `blasProvider` as `NONE`, matmul-heavy BF16 benchmarks are not exercising an optimized native BF16 GEMM route.

## Why ByteBuffer Or MemorySegment Does Not Solve BF16

`ByteBuffer` and `MemorySegment` are memory access mechanisms. They can improve interop and buffer ownership, but they do not add BF16 arithmetic to Java.

They can help with:

- passing BF16 buffers to native libraries
- avoiding extra Java array copies at FFM boundaries
- representing off-heap or device-adjacent storage
- creating a cleaner ABI for native kernels

They do not help with:

- making `short + short` mean BF16 addition
- making the Java Vector API expose BF16 vector arithmetic
- eliminating conversion when the CPU kernel still computes in `float`

Useful mental model:

```text
MemorySegment solves "where is the memory and how do I pass it?"
It does not solve "which instruction performs BF16 add/mul/tanh?"
```

## When BF16 Can Be Slower Than F32

BF16 can be slower on CPU when the workload is dominated by:

- elementwise operations with little arithmetic per element
- repeated execution-unit boundaries
- reductions that accumulate in wider precision
- missing native BF16 BLAS for matmul-heavy shapes
- publication or materialization that forces CPU-visible BF16 storage after each run

Example:

```text
workload:
  add -> gelu-like chain -> reduction -> backward graph

F32:
  contiguous float loops, vectorizable operations, fewer conversions

BF16:
  repeated BF16 unpack/pack, F32 transcendental compute, wider reduction accumulation
```

The BF16 path saves storage bandwidth, but the saved bytes may not compensate for conversion and less mature dispatch.

## How To Read Traces

For a BF16 performance investigation, separate these questions:

| Question | Evidence |
|---|---|
| Did compile make large CPU regions? | Compile partition planning trace. |
| Did region optimization fuse elementwise chains? | Optimized region / execution-unit trace. |
| Did matmul use BLAS? | Matmul trace metadata and runtime profile. |
| Did execution materialize intermediate values? | Run trace materialization metadata. |
| Were public tensors copied back after execution? | `PublicationPolicy` and run trace CPU materialization counts. |
| Did reductions promote accumulation? | Reduction trace metadata and CPU compute contract. |

Do not diagnose BF16 by looking only at region count. Region count is a compile ownership signal, not a direct measure of conversion count.

## Practical Guidance

For benchmark comparisons:

1. Keep compile policy fixed when comparing dtype behavior.
2. Use calibrated runtime profiles for the dtype and execution mode.
3. Report whether BLAS is enabled and selected.
4. Use `PublicationPolicy.OUTPUT_ONLY` or `NONE` for benchmark-only paths when public tensor synchronization is not being measured.
5. Inspect hot steps: elementwise-heavy BF16 may be conversion-bound, while matmul-heavy BF16 may be BLAS-bound.

For implementation work:

1. Improve fused BF16 elementwise chains before expecting broad CPU BF16 speedups.
2. Treat native BF16 GEMM as the most promising matmul path.
3. Use `MemorySegment` for native ABI clarity, not as a substitute for BF16 arithmetic.
4. Keep public `Tensor` API logical; storage residency and native buffer ownership belong in runtime/prepare layers.

## See Also

- [Backend Planning And Regions](backend-planning-and-regions.md#backend-planning-and-regions)
- [Native Bridges & BLAS](native-bridges-and-blas.md#term-map-at-a-glance)
- [Configuration](configuration.md#runtimeconfig)
- [Compute Flow](compute-flow.md#compile)
