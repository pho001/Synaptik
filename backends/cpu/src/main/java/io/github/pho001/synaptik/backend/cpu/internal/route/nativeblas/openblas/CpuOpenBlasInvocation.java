package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import java.lang.foreign.MemorySegment;

/**
 * Minimal borrowed OpenBLAS invocation seam consumed by the prepared CPU route.
 * Implementations retain their caller-owned provider but do not load, configure, discover,
 * restore, or close it. The provider-free method signatures keep native-free backend-conformance
 * fakes independent of the OpenBLAS provider project.
 */
public interface CpuOpenBlasInvocation {
    /** Reports whether the borrowed provider owner remains open.
     * @return {@code true} while provider calls may still be attempted */
    boolean isOpen();

    /** Returns the provider's current positive thread count.
     * @return the observed positive thread count
     * @throws IllegalStateException if the provider is closed or the query fails */
    int threadCount();

    /**
     * Performs one dense row-major FLOAT32 GEMM call.
     *
     * @param m positive output row count
     * @param n positive output column count
     * @param k positive contraction count
     * @param alpha exact scalar multiplier for the matrix product
     * @param a exact caller-owned native left matrix segment
     * @param b exact caller-owned native right matrix segment
     * @param beta exact scalar multiplier for the prior output
     * @param c exact caller-owned writable native output segment
     * @throws NullPointerException if a matrix segment is {@code null}
     * @throws IllegalArgumentException if dimensions, segments, alignment, spans, or overlap
     *     violate the provider invocation contract
     * @throws IllegalStateException if the borrowed provider is closed or invocation fails
     */
    void sgemm(int m, int n, int k, float alpha, MemorySegment a, MemorySegment b,
            float beta, MemorySegment c);

    /**
     * Performs one dense row-major FLOAT64 GEMM call.
     *
     * @param m positive output row count
     * @param n positive output column count
     * @param k positive contraction count
     * @param alpha exact scalar multiplier for the matrix product
     * @param a exact caller-owned native left matrix segment
     * @param b exact caller-owned native right matrix segment
     * @param beta exact scalar multiplier for the prior output
     * @param c exact caller-owned writable native output segment
     * @throws NullPointerException if a matrix segment is {@code null}
     * @throws IllegalArgumentException if dimensions, segments, alignment, spans, or overlap
     *     violate the provider invocation contract
     * @throws IllegalStateException if the borrowed provider is closed or invocation fails
     */
    void dgemm(int m, int n, int k, double alpha, MemorySegment a, MemorySegment b,
            double beta, MemorySegment c);
}
