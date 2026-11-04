# Task 0002: [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [Row-Major](../../../../glossary.md#term-row-major) [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) Invocation

## Reader context

The caller supplies native `MemorySegment` objects containing A/B/C values, not Tensor descriptors. `OpenBlasLibrary.sgemm/dgemm` checks open lifetime, delegates validation to `OpenBlasGemmInvocation` and borrows segments for one synchronous CBLAS downcall. A/B are read inputs, C writable output; allocation and close remain the caller's responsibility.

CPU normalizes layout and chooses the route before this API. Empty output skips native work only after validation; empty contraction with nonempty C still invokes GEMM so beta can act on C. Historical fake-handle tests prove forwarding/rejection, not arbitrary OpenBLAS numerical conformance.

- [Current GEMM ownership and zero behavior](../../../../backend-guide/cpu-backend.md#current-low-level-openblas-foundation) — Connects matrix values to low-level borrowing.
- [Current CPU normalization and selection](../../../../backend-guide/cpu-backend.md#current-narrow-openblas-matmul-route) — Shows why graph/layout policy stays outside the provider.

## Status

Complete

## Goal

Add the smallest usable [general matrix multiplication](../../../../glossary.md#general-matrix-multiplication--gemm) ([GEMM](../../../../glossary.md#general-matrix-multiplication--gemm)) invocation surface to the existing
caller-owned [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary): validated [FLOAT32](../../../../glossary.md#term-float32) and [FLOAT64](../../../../glossary.md#term-float64) calls for dense [row-major](../../../../glossary.md#term-row-major),
non-transposed `A[m,k]`, `B[k,n]`, and `C[m,n]` [native memory](../../../../glossary.md#aliasing-and-arena) segments.

The provider forwards the exact caller-supplied [scalars](../../../../glossary.md#term-scalar) to the already-bound [OpenBLAS](../../../../glossary.md#term-openblas) [CBLAS](../../../../glossary.md#term-cblas)
symbols and mutates only the logical `C` output. It does not select a [CPU](../../../../glossary.md#term-cpu) route, allocate or copy
[matrix](../../../../glossary.md#term-matrix-meanings) storage, interpret [Tensor](../../../../glossary.md#tensor) [metadata](../../../../glossary.md#term-metadata), or promise numerical results.

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

```text
caller-owned open OpenBlasLibrary
  -> validate m, n, k and caller-owned native A/B/C segments
  -> derive fixed row-major/no-transpose CBLAS arguments
  -> invoke the already-bound cblas_sgemm or cblas_dgemm handle exactly
  -> caller still owns A, B, C, and the library lifetime

future CPU prepare/execution
  -> owns batching, broadcasting, transposition, layout conversion, packing, and loops
  -> calls this provider only for one already-normalized dense GEMM
```

Dense [row-major](../../../../glossary.md#term-row-major) no-[transpose](../../../../glossary.md#term-transpose) [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) is the concrete first native building block for current [MATMUL](../../../../glossary.md#matrix-multiplication--matmul),
[linear](../../../../glossary.md#term-linear-meanings), [scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention), and [convolution](../../../../glossary.md#term-channels-first-convolution-layers) [lowering](../../../../glossary.md#lowering). A broader provider surface would
move [CPU](../../../../glossary.md#term-cpu)-owned [normalization](../../../../glossary.md#term-normalization) and route policy into the [leaf](../../../../glossary.md#term-leaf). A separate public invocation
abstraction has no current consumer and would separate calls from the [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) [lifetime](../../../../glossary.md#term-lifetime)
they require, so the two methods belong directly on that public final [owner](../../../../glossary.md#term-owner).

## [Scope](../../../../glossary.md#term-scope)

- Add exactly two public `void` methods to [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary), one for FLOAT32 SGEMM and one for
  FLOAT64 DGEMM.<span data-glossary-navigation="true"> ([SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- Support only dense [row-major](../../../../glossary.md#term-row-major), non-transposed `A[m,k]`, `B[k,n]`, and `C[m,n]` [matrices](../../../../glossary.md#term-matrix-meanings).
- Use the standard 32-bit-`blasint` C [ABI](../../../../glossary.md#term-application-binary-interface-abi) already selected and bound by task 0001.
- Fix `CblasRowMajor = 101` and `CblasNoTrans = 111` for both operands.
- Derive `lda = max(1, k)`, `ldb = max(1, n)`, and `ldc = max(1, n)` internally.
- Validate library lifetime, dimensions, segment references and properties, alignment, required
  spans, and output/input overlap in the exact order and with the stable messages below.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [dimensions glossary](../../../../glossary.md#dimension); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Preserve every raw [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) `alpha` and `beta` value unchanged, including NaN payloads
  where the Java carrier retains them, infinities, and both signed zeros.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [carrier glossary](../../../../glossary.md#term-storage-carrier); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Invoke the existing exact typed `OpenBlasNativeBindings.sgemm()` or `dgemm()` handle with
  `invokeExact` through one field-free package-private helper.
- Translate every non-`Error` throwable from the native-handle invocation into one stable
  `IllegalStateException` while retaining the original cause; rethrow every `Error` unchanged.
- Add deterministic tests using exact typed fake method handles; ordinary tests must not require
  or load an installed OpenBLAS library.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Finalize affected Javadocs, package documentation, CPU guide, glossary, and planning records in
  the required separate clean documentation pass.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- offsets, strides, caller-supplied leading dimensions, transpose flags, column-major input, or
  non-dense matrix views<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [dimensions glossary](../../../../glossary.md#dimension); [transpose glossary](../../../../glossary.md#term-transpose); [column-major glossary](../../../../glossary.md#term-column-major); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- batched GEMM, broadcasting, vector/matrix rank adaptation, grouping, tiling, packing, layout
  conversion, transpose materialization, or provider-owned loops<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [vector glossary](../../../../glossary.md#term-vector); [rank glossary](../../../../glossary.md#term-rank); [tiling glossary](../../../../glossary.md#tiling); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization))</span>
- [FLOAT16](../../../../glossary.md#term-float16), [BFLOAT16](../../../../glossary.md#term-bfloat16), complex, integral, sparse, quantized, or mixed-[precision](../../../../glossary.md#term-precision) [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm)
- a result object, allocation, copy, storage [ownership](../../../../glossary.md#term-ownership-meanings), segment closure, or retained segment [view](../../../../glossary.md#term-view-meanings)
- accepting [heap](../../../../glossary.md#term-heap) segments or automatically copying them to [native memory](../../../../glossary.md#aliasing-and-arena)
- CPU capability reporting, lowering, route choice, fallback, workload classification, tuning,
  cache behavior, thread-count choice, or thread-control invocation<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [cache glossary](../../../../glossary.md#term-cache))</span>
- Tensor, Model, Compiler, Planning, Prepare, Runtime, Engine, Trace, Backend Contract, Config, or
  concrete CPU implementation changes<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [Trace glossary](../../../../glossary.md#trace); [Backend glossary](../../../../glossary.md#backend))</span>
- numerical tolerance, accumulation, reassociation, fused-operation, NaN propagation, signed-zero,
  determinism, or cross-platform result guarantees for OpenBLAS<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance); [reassociation glossary](../../../../glossary.md#term-reassociation); [fused glossary](../../../../glossary.md#specialization-and-fusion); [operation glossary](../../../../glossary.md#operation); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- a new public failure type, public invocation abstraction, callback, registry, manager, service,
  generic native seam, or exposed <span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [registry glossary](../../../../glossary.md#term-registry))</span>[`MethodHandle`](../../../../glossary.md#native-numerical-library-and-instruction-architecture)
- synchronization, active-call accounting, reference counting, close/call serialization, or a
  successful-call guarantee when <span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [serialization glossary](../../../../glossary.md#term-serialization))</span>`close()` races an invocation
- OpenBLAS thread control, compatible installed-library validation, native access launch policy,
  or the provider capability checkpoint retained by task 0003<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Gradle, native packaging, dependency, architecture-contract, ADR, architecture-test,
  backend-conformance, integration-test, or CPU task changes<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- a detailed task-0003 specification

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - [OpenBLAS](../../../../glossary.md#term-openblas) provider
  - [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) routes
  - [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module)
  - [Dependency](../../../../glossary.md#term-dependency) rules
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare))</span>
- [CPU kernel strategy](../../../../design/notes/cpu-kernel-strategy.md)<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>
- [OpenBLAS provider master plan](../master-plan.md)
- [Task 0001: Library loading and required symbol binding](0001-library-loading-and-required-symbol-binding.md)<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [binding glossary](../../../../glossary.md#term-binding))</span>

Primary external contracts:

- [JDK 26 `MemorySegment`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/MemorySegment.html)<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
  defines native versus heap segments, physical native addresses, byte sizes, scopes,
  current-thread accessibility, read-only state, and actual backing-region overlap.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
- [JDK 26 `Linker`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/Linker.html)
  defines exact downcall method handles and <span data-glossary-navigation="true"> ([downcall glossary](../../../../glossary.md#term-downcall))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment) address [arguments](../../../../glossary.md#term-argument).
- [JDK 26 `MethodHandle`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/invoke/MethodHandle.html)<span data-glossary-navigation="true"> ([MethodHandle glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture))</span>
  requires the `invokeExact` call-site descriptor to match the handle's method type exactly and
  specifies that any throwable from the underlying operation propagates through the handle call.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation))</span>
- [OpenBLAS `cblas.h` at the reviewed revision](https://github.com/OpenMathLib/OpenBLAS/blob/c3db185d6c779e5ea222c3ff550bf0082d7e0c09/cblas.h)
  defines `CblasRowMajor = 101`, `CblasNoTrans = 111`, and the ordered SGEMM/DGEMM C signatures.
  The implementation remains constrained to task 0001's ordinary 32-bit-<span data-glossary-navigation="true"> ([SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [signatures glossary](../../../../glossary.md#term-signature))</span>`blasint` ABI; the pinned
  source is durable ABI evidence rather than a minimum-version promise.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- The official [OpenBLAS user manual's CBLAS interface example](https://www.openmathlib.org/OpenBLAS/docs/user_manual/#call-cblas-interface)<span data-glossary-navigation="true"> ([CBLAS glossary](../../../../glossary.md#term-cblas))</span>
  confirms that C callers include `cblas.h` and invoke `cblas_dgemm`. It is usage evidence only
  and does not establish a minimum compatible OpenBLAS version or supersede the pinned header.

## Architecture constraints

- `backends/openblas-provider` remains the low-level native-interoperability leaf assigned to
  library loading, symbol binding, GEMM calls, and thread control.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [symbol glossary](../../../../glossary.md#term-symbol); [binding glossary](../../../../glossary.md#term-binding); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- The [dependency direction](../../../../glossary.md#java-platform-and-build-tools) remains `backends/cpu -> backends/openblas-provider`; the provider
  gains no project dependency and imports no CPU or lifecycle-layer type.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [CPU glossary](../../../../glossary.md#term-cpu); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Planning selects CPU ownership. Future CPU prepare owns route selection and every normalization
  step needed to present one dense row-major no-transpose invocation to this provider.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepare glossary](../../../../glossary.md#prepare); [normalization glossary](../../../../glossary.md#term-normalization); [row-major glossary](../../../../glossary.md#term-row-major); [transpose glossary](../../../../glossary.md#term-transpose))</span>
- [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) remains the caller-owned native lookup lifetime. GEMM methods check and consume
  its package-private bindings; no handle or arena crosses the public boundary.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [lifetime glossary](../../../../glossary.md#term-lifetime); [arena glossary](../../../../glossary.md#term-arena))</span>
- Caller segments remain caller-owned. The provider retains, allocates, copies, reinterprets, and
  closes no caller memory, and it mutates only the logical `C` output through the native call.
- Ordinary unit tests use the existing package-private `OpenBlasNativeAccess` and
  `OpenBlasNativeBindings` path with exact typed fake method handles. Do not add another seam.
- The implementation introduces no synchronization or mutable call state on the invocation hot
  path.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [mutable glossary](../../../../glossary.md#term-mutable); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- `legacy/pre-rewrite` is capability evidence only. Do not copy its API, source structure,
  lifecycle coupling, fallback, discovery, or execution policy.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- If implementation needs a new module edge, CPU behavior, an independent public abstraction,
  another native symbol, or lifecycle coordination beyond the recorded close-race boundary, stop
  and request a planning or architecture decision.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [race glossary](../../../../glossary.md#term-race))</span>

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.backend.provider.openblas` — extends the deliberate public lifetime
  owner with two inseparable low-level calls and adds one package-private stateless invocation
  helper beside the existing bindings.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [owner glossary](../../../../glossary.md#term-owner))</span>

No package is added.

Type placement:

- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary` — remains the public final
  owner because every invocation requires and is bounded by this caller-owned open lifetime.
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasGemmInvocation` — package-private
  final, field-free helper that owns validation, derived CBLAS arguments, exact invocation, and
  invocation-failure translation without becoming a public abstraction or test seam.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [CBLAS glossary](../../../../glossary.md#term-cblas); [arguments glossary](../../../../glossary.md#term-argument))</span>
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasNativeBindings` — unchanged; its
  existing package-private exact SGEMM/DGEMM accessors already supply the required handles.<span data-glossary-navigation="true"> ([SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>

Tests remain in the production package because they must inject exact handles through the existing
package-private loading/[binding](../../../../glossary.md#term-binding) boundary and lock the helper's visibility and stateless [shape](../../../../glossary.md#shape).

## Exact public and package-private [API](../../../../glossary.md#term-application-programming-interface-api) intent

Add exactly these public methods to the existing public final [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary):

```java
public void sgemm(
        int m,
        int n,
        int k,
        float alpha,
        MemorySegment a,
        MemorySegment b,
        float beta,
        MemorySegment c);

public void dgemm(
        int m,
        int n,
        int k,
        double alpha,
        MemorySegment a,
        MemorySegment b,
        double beta,
        MemorySegment c);
```

Both methods first call the existing package-private `bindings()` check and then delegate to
`OpenBlasGemmInvocation`. They return normally with no result object. Do not add overloads,
offsets, [transpose](../../../../glossary.md#term-transpose)/[layout](../../../../glossary.md#layout) flags, leading-[dimension](../../../../glossary.md#dimension) [parameters](../../../../glossary.md#term-parameter), generic numeric [dispatch](../../../../glossary.md#term-dispatch), or a new
public exception.

`OpenBlasGemmInvocation` is one package-private final type with no instance fields or externally
[mutable](../../../../glossary.md#term-mutable) state. Static methods may be used if they keep construction and surface minimal. It must
not expose or accept a native-access abstraction; it accepts the already-checked exact [bindings](../../../../glossary.md#term-binding)
and invocation inputs.

## [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) semantics and fixed [CBLAS](../../../../glossary.md#term-cblas) mapping

For both precisions, the logical [operation](../../../../glossary.md#operation) is:

```text
C[m,n] <- alpha * (A[m,k] x B[k,n]) + beta * C[m,n]
```

All [matrices](../../../../glossary.md#term-matrix-meanings) are dense [row-major](../../../../glossary.md#term-row-major) and non-transposed from byte offset zero of their supplied
segments. The native [argument](../../../../glossary.md#term-argument) order is exactly:

| Position | CBLAS argument | Value |
|---:|---|---|
| 1 | `Order` | `CblasRowMajor` (`101`) |
| 2 | `TransA` | `CblasNoTrans` (`111`) |
| 3 | `TransB` | `CblasNoTrans` (`111`) |
| 4 | `M` | caller `m` |
| 5 | `N` | caller `n` |
| 6 | `K` | caller `k` |
| 7 | `alpha` | caller raw [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) value unchanged |
| 8 | `A` | caller segment `a` |
| 9 | `lda` | `max(1, k)` |
| 10 | `B` | caller segment `b` |
| 11 | `ldb` | `max(1, n)` |
| 12 | `beta` | caller raw FLOAT32/FLOAT64 value unchanged |
| 13 | `C` | caller segment `c` |
| 14 | `ldc` | `max(1, n)` |

The helper uses exact typed `invokeExact`; it must not use `invoke`, `invokeWithArguments`,
reflection, arrays, boxing, or string [dispatch](../../../../glossary.md#term-dispatch) for the [downcall](../../../../glossary.md#term-downcall).

## [Validation](../../../../glossary.md#term-validation) order, types, and stable messages

Validation is observable and must occur in this exact order for both methods:

1. [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) acquires [bindings](../../../../glossary.md#term-binding) through `bindings()`. A closed owner fails with the
   existing <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>`IllegalStateException("OpenBLAS library is closed")` before any [argument](../../../../glossary.md#term-argument) validation.
2. Validate `m`, `n`, and `k` in that order. Each is the standard [LP64](../../../../glossary.md#term-lp64) 32-bit `blasint` carrier
   and must be non-negative. Failure is <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`IllegalArgumentException` with exactly
   `<name> must be non-negative: <value>`.
3. Apply `Objects.requireNonNull` to `a`, `b`, and `c` in that order with exact parameter-name
   messages <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>`a`, `b`, and `c`.
4. Require native `a`, `b`, and `c` in that order using `MemorySegment.isNative()`. Failure is
   `IllegalArgumentException("<role> must be a native memory segment")`.
5. Require live scopes for `a`, `b`, and `c` in that order using `scope().isAlive()`. Failure is
   `IllegalStateException("<role> scope is not alive")`.
6. Require current-thread accessibility for `a`, `b`, and `c` in that order using
   `isAccessibleBy(Thread.currentThread())`. Failure is
   `IllegalStateException("<role> is not accessible by the current thread")`.
7. Require writable `c` using `!c.isReadOnly()`. Failure is
   `IllegalArgumentException("c must be writable")`. `a` and `b` may be read-only.
8. Require each base physical address in `a`, `b`, and `c` order to be aligned modulo the element
   size: four bytes for SGEMM and eight bytes for DGEMM. Failure is<span data-glossary-navigation="true"> ([SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
   `IllegalArgumentException("<role> address must be aligned to <bytes> bytes")`.
9. Derive `lda`, `ldb`, and `ldc`, then compute checked required [element](../../../../glossary.md#term-element) and byte spans in
   `A`, `B`, and `C` order using `long`. Any `Math.addExact` or `Math.multiplyExact` failure is
   translated to `IllegalArgumentException("<role> required byte span overflows long", cause)`,
   where the cause is the original `ArithmeticException`; no incidental `Math` message escapes
   as the provider's message.
10. Compare `byteSize()` coverage in `A`, `B`, and `C` order. Failure is
    `IllegalArgumentException("<role> requires at least <required> bytes, but segment has <actual>")`.
11. Reject logical output overlap in `C`-with-`A`, then `C`-with-`B` order with exact messages
    `c must not overlap a` and `c must not overlap b`. Use `MemorySegment.asOverlappingSlice` on
    the required ranges so actual native backing-region overlap is detected for independently
    supplied segments and slices. A zero required span never overlaps. `A` and `B` may overlap
    because both are read-only inputs to the operation.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

Only after every validation succeeds may zero-output handling or native invocation occur.
Validation failures occur outside the native-invocation `try` block and retain their exact types
and messages.

### Required-span formulas

For [row-major](../../../../glossary.md#term-row-major) storage with logical `rows`, `columns`, and derived `leadingDimension`:

```text
requiredElements = 0                                      if rows == 0 or columns == 0
requiredElements = (rows - 1) * leadingDimension + columns otherwise
requiredBytes    = requiredElements * elementBytes
```

Apply the formula as follows:

| Role | Rows | Columns | [Leading dimension](../../../../glossary.md#term-leading-dimension) | [Element](../../../../glossary.md#term-element) bytes |
|---|---:|---:|---:|---:|
| [SGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) `A` | `m` | `k` | `max(1, k)` | 4 |
| SGEMM `B` | `k` | `n` | `max(1, n)` | 4 |
| SGEMM `C` | `m` | `n` | `max(1, n)` | 4 |
| [DGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) `A` | `m` | `k` | `max(1, k)` | 8 |
| DGEMM `B` | `k` | `n` | `max(1, n)` | 8 |
| DGEMM `C` | `m` | `n` | `max(1, n)` | 8 |

The implementation must use checked `long` arithmetic even though public [dimensions](../../../../glossary.md#dimension) are
non-negative `int` values. This locks the provider contract and avoids relying on incidental
limits of the current dimension [carrier](../../../../glossary.md#term-storage-carrier).

## Zero-[dimension](../../../../glossary.md#dimension) and [scalar](../../../../glossary.md#term-scalar) behavior

- After complete [validation](../../../../glossary.md#term-validation), if `m == 0 || n == 0`, return without invoking the native handle
  because there are no output elements. The contract still rejects read-only `c` and every other
  invalid argument so no-op calls have the same stable preconditions.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- If `k == 0` while `m > 0 && n > 0`, invoke native [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm). Required `A` and `B` spans are zero,
  required `C` is `m * n` [elements](../../../../glossary.md#term-element), `lda` is one, and [OpenBLAS](../../../../glossary.md#term-openblas) receives the exact `alpha` and
  `beta`; this permits the [CBLAS](../../../../glossary.md#term-cblas) [operation](../../../../glossary.md#operation) `C <- beta * C` for the empty product term.
- Do not claim whether [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture) dereferences a zero-span `A` or `B`. The provider validates
  their references, native kind, scopes, accessibility, alignment, and zero-byte coverage before
  invocation.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- Accept every raw `alpha` and `beta` carrier value without finiteness checks, canonicalization,
  or special-value policy. Fake-handle tests verify raw-bit forwarding for ordinary finite values,
  infinities, both signed zeros, and NaN payloads where feasible without asserting OpenBLAS
  numerical behavior.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [finite glossary](../../../../glossary.md#term-finite); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>

## Segment [ownership](../../../../glossary.md#term-ownership-meanings) and overlap contract

- `a`, `b`, and `c` are non-null native [`MemorySegment`](../../../../glossary.md#term-memorysegment) values whose scopes are alive and whose
  memory is accessible by the current calling thread.
- `a` and `b` may be read-only. `c` must be writable, including for calls that later return as an
  output-empty no-op.
- Base addresses must be aligned to the physical element width. Byte size must cover the exact
  checked logical row-major range beginning at offset zero.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- `c` must not overlap the required `a` or `b` range. `a` and `b` may overlap each other.
- Temporary slices returned by JDK overlap inspection are non-owning validation views only; the
  provider retains no slice and allocates, copies, reinterprets, or closes no caller memory.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [validation glossary](../../../../glossary.md#term-validation); [views glossary](../../../../glossary.md#term-view-meanings); [slice glossary](../../../../glossary.md#slice))</span>
- The caller owns all segment lifetimes and must keep them valid for the complete native call. The
  provider mutates only the logical `c` range through OpenBLAS and makes no promise about bytes
  outside the required ranges.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>

## Invocation failure and [concurrency](../../../../glossary.md#term-concurrency) semantics

- `OpenBlasLibrary.sgemm` and `dgemm` perform the existing open-owner check before argument
  validation and delegation.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [argument glossary](../../../../glossary.md#term-argument))</span>
- Around only the exact `invokeExact` call, catch `Error` first and rethrow it unchanged. This
  preserves every VM-fatal, linkage, assertion, and other `Error` without swallowing or wrapping
  it.
- Then catch `Throwable` and wrap it in `IllegalStateException` with exactly
  `OpenBLAS sgemm invocation failed` or `OpenBLAS dgemm invocation failed`, retaining the original
  cause. This includes checked throwables, `RuntimeException`, and `WrongMethodTypeException`
  thrown by invocation.
- Provider [validation](../../../../glossary.md#term-validation) failures occur before that `try` block and therefore retain their specified
  type, message, and overflow cause.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- No new public exception type is added.
- Concurrent calls while the owner remains open are permitted, subject to caller-managed
  nonconflicting segment access and the behavior of the supplied OpenBLAS library. The provider
  adds no call synchronization.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- A caller must not [race](../../../../glossary.md#term-race) `close()` with invocation. There is no lock, active-call counter,
  reference count, or successful-call guarantee. A race may fail at the stable owner-open check
  or later through a JDK/native lifetime failure.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- This task does not invoke or [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [OpenBLAS](../../../../glossary.md#term-openblas) thread-control functions. Different
  [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) owners may still refer to one process-global native thread state.

## Affected files

Production:

- modify `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibrary.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasGemmInvocation.java`
- modify `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/package-info.java`

Tests:

- modify `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibraryPublicShapeTest.java`
- add `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasGemmInvocationTest.java`
- modify `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasAbiContractTest.java`

Explanatory documentation:

- modify `docs/backend-guide/cpu-backend.md`
- modify `docs/glossary.md`

[Planning](../../../../glossary.md#term-planning):

- this task
- modify `docs/planning/backends/openblas-provider/master-plan.md`
- modify `docs/planning/roadmap.md`

`OpenBlasNativeBindings`, `OpenBlasNativeAccess`, and `FfmOpenBlasNativeAccess` are review-only:
the required exact handles, accessors, [descriptors](../../../../glossary.md#term-descriptor), and deterministic injection path already
exist. No source or test elsewhere is authorized.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 11 paths:

- 3 provider production paths;
- 3 provider test paths;
- 2 explanatory documentation paths; and
- 3 [planning](../../../../glossary.md#term-planning) paths.

No task-0001 byte may change. No [Gradle](../../../../glossary.md#term-gradle), root/settings, [architecture contract](../../../../glossary.md#architecture-contract)/explanation, [ADR](../../../../glossary.md#term-architecture-decision-record-adr),
[architecture test](../../../../glossary.md#term-architecture-test), [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) test, [integration test](../../../../glossary.md#term-integration-test), [CPU](../../../../glossary.md#term-cpu) source/test, provider [binding](../../../../glossary.md#term-binding)/
access source, generated output, or other [module path](../../../../glossary.md#term-module-path) may change. If implementation requires
another public type or method, another seam or [symbol](../../../../glossary.md#term-symbol), another file, or more than 11 paths, stop
and propose a bounded planning correction or follow-up.

## Acceptance criteria

- [x] [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) exposes exactly the two specified additional public `void` methods with
      no overload, offset, transpose, leading-dimension, result, or new public-failure surface.<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [dimension glossary](../../../../glossary.md#dimension))</span>
- [x] Each method performs the existing bindings/open check before any argument validation and
      delegates to one field-free package-private <span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [argument glossary](../../../../glossary.md#term-argument))</span>`OpenBlasGemmInvocation` helper.
- [x] SGEMM and DGEMM validate every condition in the exact recorded order with the exact stable
      messages and exception types, including stable overflow translation with cause.<span data-glossary-navigation="true"> ([SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- [x] The fixed constants, derived leading dimensions, exact fourteen-argument order, and
      precision-specific typed <span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [precision glossary](../../../../glossary.md#term-precision))</span>`invokeExact` calls match the existing task-0001 [ABI](../../../../glossary.md#term-application-binary-interface-abi) [descriptors](../../../../glossary.md#term-descriptor).
- [x] Required byte spans use the checked row-major formula and validate A, B, and C coverage in
      order for ordinary, padded-by-leading-dimension, zero, and overflow cases.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major))</span>
- [x] Actual required native regions enforce C-with-A then C-with-B non-overlap; A/B overlap is
      accepted, disjoint slices are accepted, overlapping slices are rejected, and zero spans do
      not overlap.
- [x] `m == 0 || n == 0` returns only after complete [validation](../../../../glossary.md#term-validation) and makes no native call; `k == 0`
      with positive output geometry invokes native GEMM with zero A/B spans and exact scalars.<span data-glossary-navigation="true"> ([GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [scalars glossary](../../../../glossary.md#term-scalar))</span>
- [x] Raw FLOAT32/FLOAT64 scalar bits are forwarded unchanged for finite values, infinities,
      signed zeros, and NaN payloads where feasible; production imposes no result promise.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [finite glossary](../../../../glossary.md#term-finite); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>
- [x] Invocation rethrows every `Error` unchanged and wraps every non-`Error` throwable from
      `invokeExact` in the exact [operation](../../../../glossary.md#operation)-specific `IllegalStateException` with cause.
- [x] Caller lifetime, ownership, read-only input, writable output, alignment, close-race, and
      concurrent-call boundaries are fully documented in Javadoc and package documentation.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [alignment glossary](../../../../glossary.md#term-alignment); [race glossary](../../../../glossary.md#term-race); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [x] Deterministic fake exact typed handles record all fourteen arguments and optionally perform
      small in-Java arithmetic only to prove Java-side call occurrence and forwarding, never
      OpenBLAS numerical correctness.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- [x] Ordinary tests never load OpenBLAS, skip on missing OpenBLAS, or require native-access JVM
      enablement.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- [x] Tests lock exact public/package-private shape, imports, zero project dependencies, excluded
      mechanisms, and the narrow update replacing the prior source assertion that no<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
      `invokeExact` exists.
- [x] No allocation, copy, retained slice, caller-memory closure, call synchronization,
      thread-control invocation, config, fallback, route, Tensor, Prepare, Runtime, or CPU
      behavior is added.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [synchronization glossary](../../../../glossary.md#term-synchronization); [Tensor glossary](../../../../glossary.md#tensor); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- [x] `OpenBlasNativeBindings`, native-access/binder sources, Gradle files, architecture files and
      tests, conformance/integration projects, CPU source, and task 0001 remain unchanged.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [x] A separate clean documentation-focused agent finalizes Javadocs, the CPU guide, glossary,
      this task, master plan, and roadmap in the same overall change without duplicating a
      successful final Java test run.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- [x] Task 0002 remains [`Ready`](../../../../glossary.md#term-ready) until all implementation, documentation, and validation criteria
      pass. Task 0003 remains a Draft row and has no detailed task file; no CPU task is Ready.
- [x] Exact scope, status/order/dependency, generated-Javadoc, Markdown link/anchor/fence,
      whitespace, and final diff checks pass.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development tests may [run](../../../../glossary.md#term-run) while executable Java stabilizes. Tests must cover:

- both [precision](../../../../glossary.md#term-precision)-specific public methods and exact public return/[parameter](../../../../glossary.md#term-parameter) [shape](../../../../glossary.md#shape);
- the package-private helper being final and field-free with no public surface;
- [constants](../../../../glossary.md#term-constant) `101`, `111`, all fourteen ordered [arguments](../../../../glossary.md#term-argument), derived `lda`/`ldb`/`ldc`, and exact
  SGEMM/DGEMM method types;<span data-glossary-navigation="true"> ([SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- the complete validation order, exception classes, exact messages, and [overflow](../../../../glossary.md#term-overflow) causes;
- native/heap, live/closed, current-thread/inaccessible, read-only/writable, aligned/misaligned,
  undersized/exact/oversized, and overlap/disjoint segment cases;<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
- ordinary spans and all `m == 0`, `n == 0`, combined-zero, and positive-output `k == 0` cases;
- raw [scalar](../../../../glossary.md#term-scalar)-bit forwarding and exact call/no-call counts;
- invocation checked throwable, [runtime](../../../../glossary.md#term-runtime) throwable, wrong method type, and `Error` behavior;
- owner close-before-call and the documented unsupported close race, without a flaky race outcome
  assertion;<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [race glossary](../../../../glossary.md#term-race))</span>
- caller ownership: no allocation, copy, retained slice, reinterpretation, segment close, or
  output replacement; and<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [slice glossary](../../../../glossary.md#slice))</span>
- source/import/build exclusions, exact changed surface, and no installed-library access.

After executable Java stabilizes, the implementation context runs exactly one final affected-
[module](../../../../glossary.md#term-module) command:

```bash
./gradlew :backends:openblas-provider:test
```

The clean documentation-focused pass receives and reuses that exact evidence unless it changes
executable Java or records a concrete stale-evidence risk. After finalizing Javadocs and Markdown,
it runs:

```bash
./gradlew :backends:openblas-provider:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/backend-guide/cpu-backend.md \
  docs/glossary.md \
  docs/planning/backends/openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md \
  docs/planning/backends/openblas-provider/master-plan.md \
  docs/planning/roadmap.md
git diff --check
```

It must inspect generated Javadocs and manually or through stable automated assertions verify the
exact public/package-private surface, argument order/types, package placement, zero project
dependencies, exact 11-path maximum, unchanged task 0001, synchronized task/master/roadmap
statuses, task 0003 remaining Draft without a file, no [CPU](../../../../glossary.md#term-cpu) task, relative links, anchors, fences,
final newlines, and trailing whitespace.

Repository-wide tests, architecture tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), integration tests, and real-native
numerical validation are deferred to task 0003's explicitly supplied installed-library provider
capability [checkpoint](../../../../glossary.md#term-checkpoint) or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). This task changes one [leaf](../../../../glossary.md#term-leaf) module without a
[dependency](../../../../glossary.md#term-dependency), architecture, shared-build, or cross-module executable change.

## Dependencies

- [OpenBLAS provider 0001](0001-library-loading-and-required-symbol-binding.md) — Complete; supplies
  caller-directed loading, the caller-owned shared lookup lifetime, exact SGEMM/DGEMM bindings,
  and the deterministic package-private injection path.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [lifetime glossary](../../../../glossary.md#term-lifetime); [SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Prepare 0003, Runtime 0014, Compiler 0006, Planning 0006, and Backend Contract 0004 — Complete;
  establish the downstream ownership boundaries without becoming provider dependencies.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Backend glossary](../../../../glossary.md#backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26 `java.lang.foreign` [API](../../../../glossary.md#term-application-programming-interface-api) and Java 26 [toolchain](../../../../glossary.md#term-toolchain) — current root build baseline.
- [OpenBLAS](../../../../glossary.md#term-openblas) [CBLAS](../../../../glossary.md#term-cblas) header contract — external [ABI](../../../../glossary.md#term-application-binary-interface-abi) evidence only, not a Java/project [dependency](../../../../glossary.md#term-dependency).

No installed OpenBLAS library is required for ordinary implementation or test [validation](../../../../glossary.md#term-validation).

## Follow-up tasks

- OpenBLAS provider 0003 remains a Draft row only. It owns low-level thread-count query/control,
  explicitly supplied installed-library numerical/ABI validation, JDK native-access enablement,
  process/global thread-state semantics, and the provider capability checkpoint.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [validation glossary](../../../../glossary.md#term-validation); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- CPU 0001 and later CPU work remain Draft. CPU will own capability truth, route/fallback policy,
  batching, broadcasting, transpose/layout conversion, packing, loops, storage, execution, and
  use of this low-level GEMM surface.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [transpose glossary](../../../../glossary.md#term-transpose); [layout glossary](../../../../glossary.md#layout); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>

Do not create a task-0003 file or a [CPU](../../../../glossary.md#term-cpu) task in this change.

## Architecture impact

Expected impact: None.

This task implements [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) calls already assigned to the [OpenBLAS](../../../../glossary.md#term-openblas) provider, preserves the provider
as a zero-project-[dependency leaf](../../../../glossary.md#term-dependency-leaf), and leaves [CPU](../../../../glossary.md#term-cpu) route selection and execution [ownership](../../../../glossary.md#term-ownership-meanings)
unchanged. If implementation needs the provider to interpret [Tensor](../../../../glossary.md#tensor)/config/[layout](../../../../glossary.md#layout)/route facts,
depend on another project [module](../../../../glossary.md#term-module), [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) active calls, or expose a native handle, stop and
report the conflict rather than editing architecture.

## Documentation impact

The separate documentation-focused pass must:

- finalize [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary), `OpenBlasGemmInvocation`, and package Javadocs for exact matrix
  geometry, scalars, segments, validation/failures, ownership, mutation, concurrency, close race,
  zero dimensions, and numerical non-guarantees;<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [scalars glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [concurrency glossary](../../../../glossary.md#term-concurrency); [race glossary](../../../../glossary.md#term-race); [dimensions glossary](../../../../glossary.md#dimension))</span>
- update the CPU backend guide to distinguish the current low-level GEMM provider call from the
  still-planned CPU route, normalization, storage, execution, and fallback work;<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- update the glossary only for the reusable GEMM/provider distinction and dense row-major
  vocabulary justified by the final change, or record a reasoned no-change conclusion;<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major))</span>
- synchronize task 0002/master/roadmap evidence and statuses only after implementation and the
  clean documentation pass finish; task 0002 remained [`Ready`](../../../../glossary.md#term-ready) until those gates passed;
- preserve architecture documents because no [ownership](../../../../glossary.md#term-ownership-meanings) or [dependency](../../../../glossary.md#term-dependency) rule changes; and
- record reasoned no-change conclusions for Tensor, Compile, and Training APIs; Model
  capabilities/master plan and the MATMUL/linear/attention/convolution semantic contracts;
  Prepare, Runtime, Backend Contract, Config, CPU implementation; architecture/tests;
  backend-conformance/integration; Gradle/root/settings; native packaging; and other modules.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [linear glossary](../../../../glossary.md#term-linear-meanings); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Implementation prompt

```text
You are a clean-context implementation agent working in the Synaptik repository. Do not commit or
push.

Read in full: AGENTS.md; ARCHITECTURE.md; docs/planning/planning-guide.md;
docs/developer-guide/documentation-rules.md; the General, API/Javadoc, Backend Guide, and Planning
documentation profiles; docs/planning/backends/openblas-provider/master-plan.md; completed task
0001; and task 0002 at
docs/planning/backends/openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md.
Read the task's directly referenced architecture and official JDK 26/OpenBLAS sources, final
provider source/tests/generated Javadocs/build inventory, CPU placeholder/build and master-plan
boundary, focused current MATMUL/linear/attention/convolution contracts, relevant architecture
tests, root/settings Java 26 build, and Prepare/Runtime/Backend Contract/Config master-plan
boundaries. Treat legacy only as read-only capability evidence, never design authority.

Implement task 0002 exactly within its eleven authorized paths. Add only the two public calls and
one package-private field-free helper over the existing bindings/seam, with the exact semantics,
validation order/messages, span/overlap rules, zero behavior, scalar forwarding, invokeExact
failure policy, ownership, and concurrency boundaries in the task. Do not modify task 0001,
bindings/access/binder sources, Gradle, architecture, CPU code, or later tasks. Stop on an
architecture conflict, scope overflow, need for another public abstraction/symbol/seam, or any
ambiguity that would change the recorded contract.

Use focused tests while developing, then run exactly one final
./gradlew :backends:openblas-provider:test after executable Java stabilizes. Hand the actual diff
and exact test evidence to a separate clean-context documentation-focused agent in the same
overall change. That pass reuses the Java evidence and independently finalizes Javadocs, package
documentation, CPU guide, glossary, task/master/roadmap, provider Javadoc, Markdown/link/surface/
scope/status checks, and git diff --check without rerunning successful Java tests unless it changes
executable behavior or records a concrete stale-evidence risk. Keep task 0002 Ready until every
criterion passes; only then record evidence/completion and mark it Complete. Leave task 0003 Draft
without a file and every CPU task Draft.
```

## Local decisions

- The public calls remain on [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) because invocation is inseparable from its
  caller-owned lookup lifetime and no independent public abstraction has a current consumer.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- The provider accepts one already-normalized dense row-major no-transpose matrix product. CPU
  owns every higher-level shape, layout, batching, packing, and loop concern.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [transpose glossary](../../../../glossary.md#term-transpose); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- One package-private field-free helper keeps validation and exact invocation cohesive without
  broadening the existing native seam or making the lifetime owner a multi-concept implementation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [owner glossary](../../../../glossary.md#term-owner))</span>
- Required spans describe only the logical row-major regions passed to native code. Overlap is
  checked on those actual backing regions with non-owning JDK views; A/B aliasing is safe at this
  boundary because both are inputs.<span data-glossary-navigation="true"> ([native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [views glossary](../../../../glossary.md#term-view-meanings); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- Complete validation precedes output-empty return, which keeps failure behavior uniform. A
  positive-output empty contraction still calls GEMM to preserve <span data-glossary-navigation="true"> ([contraction glossary](../../../../glossary.md#contraction-dimension); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>`beta * C` semantics.
- Scalars are raw ABI inputs. The provider forwards rather than interprets them and promises no
  OpenBLAS numerical policy.<span data-glossary-navigation="true"> ([Scalars glossary](../../../../glossary.md#term-scalar); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Invocation catches `Error` separately and unchanged, then wraps every other throwable. Moving
  validation outside that catch keeps provider validation failures stable.
- Close/call coordination is deliberately caller-owned, matching task 0001's lifetime contract;
  no hot-path synchronization or active-call state is justified.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Known limitations

- Only [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) dense [row-major](../../../../glossary.md#term-row-major) non-transposed [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) with 32-bit `blasint` dimensions is
  supported.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension))</span>
- The API starts each matrix at segment byte offset zero and exposes no stride, offset, transpose,
  column-major, batch, or packing control.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [stride glossary](../../../../glossary.md#term-stride); [transpose glossary](../../../../glossary.md#term-transpose); [column-major glossary](../../../../glossary.md#term-column-major); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Heap segments are rejected; callers must provide suitably aligned native memory with sufficient
  spatial, temporal, and thread-access bounds.<span data-glossary-navigation="true"> ([Heap glossary](../../../../glossary.md#term-heap); [native memory glossary](../../../../glossary.md#aliasing-and-arena); [spatial glossary](../../../../glossary.md#term-spatial-axes))</span>
- The provider does not define or verify OpenBLAS numerical accuracy, rounding, exceptional-value
  behavior, determinism, or performance.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding))</span>
- Ordinary tests prove Java validation and ABI forwarding with fake handles only. Real native ABI
  and numerical validation remains task 0003 work with an explicitly supplied compatible library.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- A caller racing `close()` with invocation has no successful-call guarantee.
- Thread-count choice and process-global [OpenBLAS](../../../../glossary.md#term-openblas) state remain outside this task.

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation context ran exactly one final
  `./gradlew :backends:openblas-provider:test` after executable Java stabilized. It passed four
  suites and 39 tests with zero skips, failures, or errors: `OpenBlasAbiContractTest` 9,
  `OpenBlasGemmInvocationTest` 17, `OpenBlasLibraryPublicShapeTest` 4, and
  `OpenBlasLibraryTest` 9. Executable Java and tests did not change afterward.
- Clean documentation context `/root/openblas_0002_docs` reused that evidence without rerunning
  Java tests because it changed only Javadoc and Markdown after executable stabilization.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The documentation context ran `./gradlew :backends:openblas-provider:javadoc`; it passed, and
  generated package and [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) pages were inspected for the finalized geometry,
  ownership, zero-dimension, concurrency, close-race, failure, and numerical boundaries.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dimension glossary](../../../../glossary.md#dimension); [concurrency glossary](../../../../glossary.md#term-concurrency); [race glossary](../../../../glossary.md#term-race))</span>
- The five-file `python3 /tmp/validate_synaptik_markdown.py ...` command passed all relative-link,
  heading-anchor, fence, final-newline, and trailing-whitespace checks.
- Stable source, reflection/bytecode, build, planning, and diff checks confirmed the exact public
  and package-private shapes and argument carriers/order, helper placement, zero project
  dependencies, exact eleven-path maximum, unchanged task 0001, synchronized Complete status,
  task 0003 Draft with no task file, every CPU task Draft, and no forbidden scope expansion.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [planning glossary](../../../../glossary.md#term-planning); [shapes glossary](../../../../glossary.md#shape); [argument glossary](../../../../glossary.md#term-argument); [CPU glossary](../../../../glossary.md#term-cpu); [scope glossary](../../../../glossary.md#term-scope))</span>
- `git diff --check` passed on the final combined change.

## Implementation notes

- Added exactly public `void sgemm(...)` and `void dgemm(...)` to [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary); both acquire
  the existing checked bindings before delegating.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding))</span>
- Added one package-private final, field-free `OpenBlasGemmInvocation`. It implements the recorded
  validation order and messages, checked row-major spans, required-region overlap checks, zero
  behavior, exact scalar/ABI forwarding, and throwable policy without another native seam.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [row-major glossary](../../../../glossary.md#term-row-major); [scalar glossary](../../../../glossary.md#term-scalar); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Deterministic fake-handle tests cover both typed call sites, exact ABI shape and values,
  validation/failure order, spans and overflow, overlap, zero dimensions, scalar families,
  lifecycle ownership, and invocation throwable translation without installed OpenBLAS.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [overflow glossary](../../../../glossary.md#term-overflow); [dimensions glossary](../../../../glossary.md#dimension); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Finalized public/helper/package Javadocs and the CPU guide. Added a reusable glossary distinction
  between low-level GEMM invocation and backend-independent MATMUL semantics.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [backend glossary](../../../../glossary.md#backend); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- No Tensor, Compile, or Training API changes are justified: the provider accepts native segments
  only and does not expose model, compiler, or extension semantics. Model capabilities/master
  planning and current MATMUL, linear, attention, and convolution contracts remain accurate
  because they describe semantic expressions, not backend lowering or execution.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [linear glossary](../../../../glossary.md#term-linear-meanings); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [lowering glossary](../../../../glossary.md#lowering))</span>
- No Prepare, Runtime, Backend Contract, Config, or CPU implementation changes are justified:
  ownership, route normalization, storage, execution, and fallback remain future CPU work. No
  architecture or architecture-test change is justified because module ownership and dependency
  direction are unchanged. Backend-conformance/integration and real-native numerical validation
  remain deferred to task 0003's provider checkpoint. Gradle/root/settings, native packaging, and
  every other module remain unchanged because the leaf keeps its JDK-only dependency surface.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [normalization glossary](../../../../glossary.md#term-normalization); [module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Gradle glossary](../../../../glossary.md#term-gradle); [leaf glossary](../../../../glossary.md#term-leaf); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

- Completed the exact two-method low-level FLOAT32/FLOAT64 row-major GEMM surface and its
  package-private validation/invocation helper.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [row-major glossary](../../../../glossary.md#term-row-major); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [validation glossary](../../../../glossary.md#term-validation))</span>
- Added focused deterministic coverage and finalized affected Javadocs, package documentation,
  CPU guide, glossary terminology, and synchronized planning evidence.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>
- Changed exactly the eleven authorized paths: three provider production files, three provider
  tests, the CPU guide, glossary, this task, provider master plan, and roadmap.
- Reused the final 39-test provider evidence; provider Javadoc, five-file Markdown validation,
  generated-page/manual surface and scope checks, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` all passed.
- Unresolved issues: none for task 0002. Follow-up: task 0003 remains Draft and owns thread
  control plus explicitly supplied installed-library ABI/numerical validation; all CPU work
  remains Draft.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>

Status: Complete
