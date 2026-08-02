# CPU backend

## Outcome and status

This guide defines the CPU integration boundary and helps contributors avoid treating CPU routes
as separate backends. The CPU backend itself remains a placeholder. The lower-level OpenBLAS
provider now implements explicit library loading, required-symbol binding, and a caller-owned
lookup lifetime plus low-level FLOAT32/FLOAT64 dense row-major general matrix multiplication
(GEMM). This does not make the placeholder CPU backend executable.

## Prerequisites and terms

Contributors need JDK 26 and the ownership rules in the [architecture
contract](../../ARCHITECTURE.md#cpu-backend-routes). Real loading also requires a compatible
OpenBLAS binary already installed or otherwise available to the operating-system loader, plus
deployment JVM permission for restricted native access.

- A **route** is one backend-local implementation choice. The glossary entry for
  [backend route](../glossary.md#backend-route) explains why OpenBLAS is not a separate backend.
- The **Foreign Function and Memory (FFM) API** is the JDK native-interoperability API used to
  load the library and bind C symbols.
- An [OpenBLAS library handle](../glossary.md#openblas-library-handle-openblaslibrary) is the
  caller-owned Java lifetime for one complete lookup and binding set.

## Ownership mental model and planned scope

The CPU backend will own capability reporting, partition lowering, specialization, fusion, scalar and optimized routes, executable units, host-side backend storage/workspaces, and typed tracing. Scalar, Vector API, ASM, specialized, fused, and OpenBLAS implementations are routes within one CPU owner.

```text
planning: owner = CPU
CPU prepare: choose scalar / Vector API / OpenBLAS / specialized / fused
runtime: invoke prepared CPU executable
```

The low-level OpenBLAS provider owns library loading, symbol binding, GEMM calls, and thread control only. Dependency direction is `backends/cpu -> backends/openblas-provider`.

## Current low-level OpenBLAS foundation

[`OpenBlasLibrary`](../glossary.md#openblas-library-handle-openblaslibrary) is the current public
lifetime boundary. A caller opens exactly one supplied operating-system library name or one
absolute path:

The first example proves only that the supplied library loaded, exported the complete required
symbol set, and remained open inside one caller-owned scope. It assumes that the loader resolves
the exact name `openblas` to a compatible 32-bit-`blasint` build.

```java
import io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary;

try (OpenBlasLibrary library = OpenBlasLibrary.open("openblas")) {
    if (!library.isOpen()) {
        throw new IllegalStateException("unexpected closed OpenBLAS handle");
    }
}
```

`open("openblas")` passes that name unchanged to the JDK lookup and returns only after all four
required symbols bind. `isOpen()` then reports this owner's local lifecycle state. Exiting the
try-with-resources block closes the lookup lifetime, including when the body fails. The final
observable state is a closed Java owner; this proves loading, binding, and cleanup only, not a
GEMM result or viable CPU route.

The input is explicit: the provider does not choose a platform filename, inspect configuration,
read an environment variable or system property, search directories, or decide fallback. The
path overload requires an absolute path and passes it unchanged to the JDK lookup.

Opening succeeds only after the provider binds this complete ordered set:

1. `cblas_sgemm`;
2. `cblas_dgemm`;
3. `openblas_set_num_threads`; and
4. `openblas_get_num_threads`.

The bindings use the ordinary OpenBLAS C interface with 32-bit `blasint`. A library using the
64-bit integer interface is outside the current contract. Missing symbols fail the whole open in
the order above, the partial lookup lifetime is closed, and the caller receives
`OpenBlasLoadException` with the original failure as its cause.

Each successful open returns a fresh Java owner backed by a shared Foreign Function and Memory
(FFM) arena. `close()` is safe for repeated or concurrent close attempts and ends that owner's
lookup lifetime. It does not promise physical unloading, because the JDK, operating system, or
another owner may retain the same process library. A later native call must not race closure.

The handle exposes no native address or Foreign Function and Memory (FFM) handle. It now exposes
`sgemm` and `dgemm` for one already-normalized product:

```text
C[m,n] = alpha * (A[m,k] x B[k,n]) + beta * C[m,n]
```

`A`, `B`, and `C` begin at byte offset zero in caller-owned native segments. They use dense
row-major, non-transposed geometry; `A` and `B` may be read-only, while `C` must be writable and
must not overlap either required input range. The provider validates dimensions, segment
lifetime/access, alignment, required spans, and overlap before it either calls OpenBLAS or returns
for an output-empty product. It forwards scalar bits unchanged and does not define OpenBLAS
numerical accuracy, exceptional-value, determinism, or performance behavior.

A positive-output call with `k == 0` still invokes OpenBLAS so `beta` can apply to `C`. Calls with
`m == 0 || n == 0` make no native call only after complete validation. Caller memory remains
borrowed for the call: the provider does not allocate, copy, retain, reinterpret, or close it.
Concurrent calls require caller-managed nonconflicting segment access, and callers must not race
`close()` with invocation.

This is an invocation boundary, not a CPU route. The future CPU backend must still decide whether
OpenBLAS is eligible, normalize MATMUL and higher-level operations into this exact geometry,
materialize transpose or layout conversions, pack when required, allocate and bind storage,
construct prepared execution, choose threads, and provide scalar or other fallback. The provider
also exposes no thread-control operation yet.

### Low-level invocation example

This example supplies one `A[2,4] x B[4,3] -> C[2,3]` geometry. It requires the same compatible
installed library and native-access permission as the loading example; it demonstrates the Java
contract and call boundary, not numerical correctness.

```java
import io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

try (Arena matrices = Arena.ofConfined();
        OpenBlasLibrary library = OpenBlasLibrary.open("openblas")) {
    MemorySegment a = matrices.allocate(2L * 4L * Float.BYTES, Float.BYTES);
    MemorySegment b = matrices.allocate(4L * 3L * Float.BYTES, Float.BYTES);
    MemorySegment c = matrices.allocate(2L * 3L * Float.BYTES, Float.BYTES);

    library.sgemm(2, 3, 4, 1.0f, a, b, 0.0f, c);
}
```

The three allocations provide 32, 48, and 24 bytes respectively: exactly the dense row-major
ranges for 8, 12, and 6 FLOAT32 elements. The call requests `alpha = 1` and `beta = 0`; the
provider checks the open owner and all matrix preconditions, derives leading dimensions 4, 3,
and 3, then invokes the bound SGEMM symbol. Normal return means that validation and native
invocation completed without a reported failure. It does not establish particular values in
`c`, because installed-library numerical validation belongs to the provider checkpoint.

## Typical mistakes

| Symptom | Cause | Correction |
|---|---|---|
| Loading fails immediately for a short name | The operating-system loader cannot resolve the exact supplied name. | Supply an installed name the loader recognizes or use the absolute-path overload. |
| Opening reports missing required symbols | The selected binary is incompatible or incomplete for the four-symbol contract. | Supply a compatible OpenBLAS C library; do not treat a partial binding as available. |
| A caller expects scalar fallback after `OpenBlasLoadException` | Fallback policy was placed mentally in the leaf provider. | Handle policy in later CPU/composition code; the provider only reports loading failure. |
| A caller passes batched, transposed, strided, offset, or tensor-shaped data directly to `sgemm`/`dgemm` | The low-level call was mistaken for CPU normalization. | Normalize and materialize the exact dense row-major product in future CPU prepare/execution code. |
| A caller expects the provider to allocate or return `C` | The borrowed in-place ABI boundary was mistaken for a storage API. | Supply a writable, sufficiently large native `C` segment and retain its ownership. |
| Two Java handles appear to have independent thread counts | Both may refer to the same process library and global OpenBLAS state. | Treat thread mutation as shared native state until the later thread-control contract defines coordination. |

## Toolchain and resources

The project baseline is JDK 26. The Vector API remains an incubator module and is not enabled
globally; a focused CPU task must configure and validate it. Real OpenBLAS loading requires a
compatible installed library and deployment JVM permission for restricted native access. The
ordinary provider unit tests do not require an installed OpenBLAS library.

## Limitations and validation

No CPU capability, normalization, route threshold, storage/execution path, fallback, thread-control
behavior, backend conformance, or performance result is implemented or promised. Provider unit
tests prove Java validation and exact ABI forwarding with fake handles, not installed-library
numerical correctness. Future work must compare optimized routes with a scalar reference through
backend-conformance tests and keep benchmarks reproducible.

See the [CPU master plan](../planning/backends/cpu/master-plan.md), [kernel routes](kernel-routes.md), and [CPU kernel strategy](../design/notes/cpu-kernel-strategy.md).
