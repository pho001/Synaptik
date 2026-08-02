package io.github.pho001.synaptik.backend.provider.openblas;

import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.Objects;

/**
 * Validates and invokes one normalized dense row-major OpenBLAS GEMM operation.
 *
 * <p>This package-private helper is stateless. It allocates, copies, retains, reinterprets, and
 * closes no caller memory and owns no route, fallback, or numerical policy. It accepts matrix
 * regions beginning at byte offset zero, derives dense row-major leading dimensions, validates
 * complete required byte spans and output/input non-overlap, and invokes one exact typed handle.
 * Output-empty calls are validated but not invoked; positive-output empty contractions are
 * invoked. Callers own memory lifetime, nonconflicting concurrent access, and coordination with
 * library closure.
 */
final class OpenBlasGemmInvocation {
    private static final int CBLAS_ROW_MAJOR = 101;
    private static final int CBLAS_NO_TRANS = 111;

    private OpenBlasGemmInvocation() {
    }

    /**
     * Validates and invokes the exact single-precision GEMM handle.
     *
     * @param bindings the already open exact native bindings; not {@code null}
     * @param m the non-negative output row count
     * @param n the non-negative output column count
     * @param k the non-negative contraction extent
     * @param alpha the raw scalar forwarded unchanged
     * @param a caller-owned native FLOAT32 input storage; not {@code null}
     * @param b caller-owned native FLOAT32 input storage; not {@code null}
     * @param beta the raw scalar forwarded unchanged
     * @param c caller-owned writable native FLOAT32 output storage; not {@code null}
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if a dimension or spatial segment precondition fails
     * @throws IllegalStateException if a temporal/thread-access precondition or invocation fails
     */
    static void sgemm(
            OpenBlasNativeBindings bindings,
            int m,
            int n,
            int k,
            float alpha,
            MemorySegment a,
            MemorySegment b,
            float beta,
            MemorySegment c) {
        validate(m, n, k, a, b, c, Float.BYTES);
        if (m == 0 || n == 0) {
            return;
        }
        int lda = Math.max(1, k);
        int ldb = Math.max(1, n);
        int ldc = Math.max(1, n);
        MethodHandle sgemm = bindings.sgemm();
        try {
            sgemm.invokeExact(
                    CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        } catch (Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new IllegalStateException("OpenBLAS sgemm invocation failed", failure);
        }
    }

    /**
     * Validates and invokes the exact double-precision GEMM handle.
     *
     * @param bindings the already open exact native bindings; not {@code null}
     * @param m the non-negative output row count
     * @param n the non-negative output column count
     * @param k the non-negative contraction extent
     * @param alpha the raw scalar forwarded unchanged
     * @param a caller-owned native FLOAT64 input storage; not {@code null}
     * @param b caller-owned native FLOAT64 input storage; not {@code null}
     * @param beta the raw scalar forwarded unchanged
     * @param c caller-owned writable native FLOAT64 output storage; not {@code null}
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if a dimension or spatial segment precondition fails
     * @throws IllegalStateException if a temporal/thread-access precondition or invocation fails
     */
    static void dgemm(
            OpenBlasNativeBindings bindings,
            int m,
            int n,
            int k,
            double alpha,
            MemorySegment a,
            MemorySegment b,
            double beta,
            MemorySegment c) {
        validate(m, n, k, a, b, c, Double.BYTES);
        if (m == 0 || n == 0) {
            return;
        }
        int lda = Math.max(1, k);
        int ldb = Math.max(1, n);
        int ldc = Math.max(1, n);
        MethodHandle dgemm = bindings.dgemm();
        try {
            dgemm.invokeExact(
                    CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        } catch (Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new IllegalStateException("OpenBLAS dgemm invocation failed", failure);
        }
    }

    private static void validate(
            int m, int n, int k, MemorySegment a, MemorySegment b, MemorySegment c, int elementBytes) {
        requireNonNegative("m", m);
        requireNonNegative("n", n);
        requireNonNegative("k", k);

        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(c, "c");

        requireNative("a", a);
        requireNative("b", b);
        requireNative("c", c);

        requireAlive("a", a);
        requireAlive("b", b);
        requireAlive("c", c);

        Thread currentThread = Thread.currentThread();
        requireAccessible("a", a, currentThread);
        requireAccessible("b", b, currentThread);
        requireAccessible("c", c, currentThread);

        if (c.isReadOnly()) {
            throw new IllegalArgumentException("c must be writable");
        }

        requireAligned("a", a, elementBytes);
        requireAligned("b", b, elementBytes);
        requireAligned("c", c, elementBytes);

        int lda = Math.max(1, k);
        int ldb = Math.max(1, n);
        int ldc = Math.max(1, n);
        long aBytes = requiredBytes("A", m, k, lda, elementBytes);
        long bBytes = requiredBytes("B", k, n, ldb, elementBytes);
        long cBytes = requiredBytes("C", m, n, ldc, elementBytes);

        requireCoverage("A", a, aBytes);
        requireCoverage("B", b, bBytes);
        requireCoverage("C", c, cBytes);

        requireNoOverlap("a", c, cBytes, a, aBytes);
        requireNoOverlap("b", c, cBytes, b, bBytes);
    }

    private static void requireNonNegative(String name, int value) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative: " + value);
        }
    }

    private static void requireNative(String role, MemorySegment segment) {
        if (!segment.isNative()) {
            throw new IllegalArgumentException(role + " must be a native memory segment");
        }
    }

    private static void requireAlive(String role, MemorySegment segment) {
        if (!segment.scope().isAlive()) {
            throw new IllegalStateException(role + " scope is not alive");
        }
    }

    private static void requireAccessible(String role, MemorySegment segment, Thread thread) {
        if (!segment.isAccessibleBy(thread)) {
            throw new IllegalStateException(role + " is not accessible by the current thread");
        }
    }

    private static void requireAligned(String role, MemorySegment segment, int bytes) {
        if (segment.address() % bytes != 0) {
            throw new IllegalArgumentException(role + " address must be aligned to " + bytes + " bytes");
        }
    }

    private static long requiredBytes(
            String role, int rows, int columns, int leadingDimension, int elementBytes) {
        if (rows == 0 || columns == 0) {
            return 0;
        }
        try {
            long precedingRows = Math.multiplyExact((long) rows - 1L, (long) leadingDimension);
            long elements = Math.addExact(precedingRows, columns);
            return Math.multiplyExact(elements, elementBytes);
        } catch (ArithmeticException failure) {
            throw new IllegalArgumentException(
                    role + " required byte span overflows long", failure);
        }
    }

    private static void requireCoverage(String role, MemorySegment segment, long requiredBytes) {
        if (segment.byteSize() < requiredBytes) {
            throw new IllegalArgumentException(
                    role + " requires at least " + requiredBytes
                            + " bytes, but segment has " + segment.byteSize());
        }
    }

    private static void requireNoOverlap(
            String inputRole,
            MemorySegment c,
            long cBytes,
            MemorySegment input,
            long inputBytes) {
        if (cBytes == 0 || inputBytes == 0) {
            return;
        }
        MemorySegment cRange = c.asSlice(0, cBytes);
        MemorySegment inputRange = input.asSlice(0, inputBytes);
        if (cRange.asOverlappingSlice(inputRange).isPresent()) {
            throw new IllegalArgumentException("c must not overlap " + inputRole);
        }
    }
}
