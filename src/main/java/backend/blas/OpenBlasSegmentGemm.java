package backend.blas;

import java.lang.foreign.MemorySegment;

/**
 * OpenBLAS row-major GEMM entrypoints for {@link MemorySegment}-backed CPU storage.
 */
public final class OpenBlasSegmentGemm {
    private OpenBlasSegmentGemm() {
    }

    public static void sgemmRowMajorNoTransSegment(
            int m,
            int n,
            int k,
            float alpha,
            MemorySegment a,
            long aByteOffset,
            int lda,
            MemorySegment b,
            long bByteOffset,
            int ldb,
            float beta,
            MemorySegment c,
            long cByteOffset,
            int ldc
    ) {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.sgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sgemm is unavailable: " + symbols.reason);
        }
        try {
            MemorySegment aSeg = OpenBlasGemmLayout.segmentSlice(a, aByteOffset, OpenBlasGemmLayout.requiredElements(m, lda), Float.BYTES, "a");
            MemorySegment bSeg = OpenBlasGemmLayout.segmentSlice(b, bByteOffset, OpenBlasGemmLayout.requiredElements(k, ldb), Float.BYTES, "b");
            MemorySegment cSeg = OpenBlasGemmLayout.segmentSlice(c, cByteOffset, OpenBlasGemmLayout.requiredElements(m, ldc), Float.BYTES, "c");
            symbols.sgemm.invokeExact(
                    OpenBlasSymbols.CBLAS_ROW_MAJOR,
                    OpenBlasSymbols.CBLAS_NO_TRANS,
                    OpenBlasSymbols.CBLAS_NO_TRANS,
                    m,
                    n,
                    k,
                    alpha,
                    aSeg,
                    lda,
                    bSeg,
                    ldb,
                    beta,
                    cSeg,
                    ldc
            );
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS FFM segment sgemm call failed", t);
        }
    }

    public static void dgemmRowMajorNoTransSegment(
            int m,
            int n,
            int k,
            double alpha,
            MemorySegment a,
            long aByteOffset,
            int lda,
            MemorySegment b,
            long bByteOffset,
            int ldb,
            double beta,
            MemorySegment c,
            long cByteOffset,
            int ldc
    ) {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.dgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM dgemm is unavailable: " + symbols.reason);
        }
        try {
            MemorySegment aSeg = OpenBlasGemmLayout.segmentSlice(a, aByteOffset, OpenBlasGemmLayout.requiredElements(m, lda), Double.BYTES, "a");
            MemorySegment bSeg = OpenBlasGemmLayout.segmentSlice(b, bByteOffset, OpenBlasGemmLayout.requiredElements(k, ldb), Double.BYTES, "b");
            MemorySegment cSeg = OpenBlasGemmLayout.segmentSlice(c, cByteOffset, OpenBlasGemmLayout.requiredElements(m, ldc), Double.BYTES, "c");
            symbols.dgemm.invokeExact(
                    OpenBlasSymbols.CBLAS_ROW_MAJOR,
                    OpenBlasSymbols.CBLAS_NO_TRANS,
                    OpenBlasSymbols.CBLAS_NO_TRANS,
                    m,
                    n,
                    k,
                    alpha,
                    aSeg,
                    lda,
                    bSeg,
                    ldb,
                    beta,
                    cSeg,
                    ldc
            );
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS FFM segment dgemm call failed", t);
        }
    }

    /**
     * Invokes OpenBLAS {@code cblas_sbgemm}: BF16 inputs and FLOAT32 output.
     */
    public static void sbgemmRowMajorNoTransSegment(
            int m,
            int n,
            int k,
            float alpha,
            MemorySegment aBf16,
            long aByteOffset,
            int lda,
            MemorySegment bBf16,
            long bByteOffset,
            int ldb,
            float beta,
            MemorySegment cF32,
            long cByteOffset,
            int ldc
    ) {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.sbgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sbgemm is unavailable: " + symbols.reason);
        }
        try {
            MemorySegment aSeg = OpenBlasGemmLayout.segmentSlice(aBf16, aByteOffset, OpenBlasGemmLayout.requiredElements(m, lda), Short.BYTES, "aBf16");
            MemorySegment bSeg = OpenBlasGemmLayout.segmentSlice(bBf16, bByteOffset, OpenBlasGemmLayout.requiredElements(k, ldb), Short.BYTES, "bBf16");
            MemorySegment cSeg = OpenBlasGemmLayout.segmentSlice(cF32, cByteOffset, OpenBlasGemmLayout.requiredElements(m, ldc), Float.BYTES, "cF32");
            symbols.sbgemm.invokeExact(
                    OpenBlasSymbols.CBLAS_ROW_MAJOR,
                    OpenBlasSymbols.CBLAS_NO_TRANS,
                    OpenBlasSymbols.CBLAS_NO_TRANS,
                    m,
                    n,
                    k,
                    alpha,
                    aSeg,
                    lda,
                    bSeg,
                    ldb,
                    beta,
                    cSeg,
                    ldc
            );
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS FFM segment sbgemm call failed", t);
        }
    }

    /**
     * Invokes OpenBLAS {@code cblas_bgemm}: BF16 inputs and BF16 output.
     */
    public static void bgemmRowMajorNoTransSegment(
            int m,
            int n,
            int k,
            short alpha,
            MemorySegment aBf16,
            long aByteOffset,
            int lda,
            MemorySegment bBf16,
            long bByteOffset,
            int ldb,
            short beta,
            MemorySegment cBf16,
            long cByteOffset,
            int ldc
    ) {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.bgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM bgemm is unavailable: " + symbols.reason);
        }
        try {
            MemorySegment aSeg = OpenBlasGemmLayout.segmentSlice(aBf16, aByteOffset, OpenBlasGemmLayout.requiredElements(m, lda), Short.BYTES, "aBf16");
            MemorySegment bSeg = OpenBlasGemmLayout.segmentSlice(bBf16, bByteOffset, OpenBlasGemmLayout.requiredElements(k, ldb), Short.BYTES, "bBf16");
            MemorySegment cSeg = OpenBlasGemmLayout.segmentSlice(cBf16, cByteOffset, OpenBlasGemmLayout.requiredElements(m, ldc), Short.BYTES, "cBf16");
            symbols.bgemm.invokeExact(
                    OpenBlasSymbols.CBLAS_ROW_MAJOR,
                    OpenBlasSymbols.CBLAS_NO_TRANS,
                    OpenBlasSymbols.CBLAS_NO_TRANS,
                    m,
                    n,
                    k,
                    alpha,
                    aSeg,
                    lda,
                    bSeg,
                    ldb,
                    beta,
                    cSeg,
                    ldc
            );
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS FFM segment bgemm call failed", t);
        }
    }
}
