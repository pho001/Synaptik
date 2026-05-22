package backend.blas;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * OpenBLAS row-major GEMM entrypoints for Java primitive-array storage.
 *
 * <p>These methods copy Java arrays into native call buffers before dispatching through FFM and copy
 * results back afterward. Use {@link OpenBlasSegmentGemm} when the caller already owns
 * {@link MemorySegment}-backed CPU storage.</p>
 */
public final class OpenBlasArrayGemm {
    private OpenBlasArrayGemm() {
    }

    public static void sgemmRowMajorNoTrans(
            int m,
            int n,
            int k,
            float alpha,
            float[] a,
            int lda,
            float[] b,
            int ldb,
            float beta,
            float[] c,
            int ldc
    ) {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.sgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sgemm is unavailable: " + symbols.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = OpenBlasGemmLayout.requiredElements(m, ldc);
            MemorySegment aSeg = nativeFloatSegment(callArena, a, 0, OpenBlasGemmLayout.requiredElements(m, lda));
            MemorySegment bSeg = nativeFloatSegment(callArena, b, 0, OpenBlasGemmLayout.requiredElements(k, ldb));
            MemorySegment cSeg = nativeFloatSegment(callArena, c, 0, cLength);
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
            copyFloatSegment(cSeg, c, 0, cLength);
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS FFM sgemm call failed", t);
        }
    }

    public static void sgemmRowMajorNoTransOffsets(
            int m,
            int n,
            int k,
            float alpha,
            float[] a,
            int aOffset,
            int lda,
            float[] b,
            int bOffset,
            int ldb,
            float beta,
            float[] c,
            int cOffset,
            int ldc
    ) {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.sgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sgemm is unavailable: " + symbols.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = OpenBlasGemmLayout.requiredElements(m, ldc);
            MemorySegment aSeg = nativeFloatSegment(callArena, a, aOffset, OpenBlasGemmLayout.requiredElements(m, lda));
            MemorySegment bSeg = nativeFloatSegment(callArena, b, bOffset, OpenBlasGemmLayout.requiredElements(k, ldb));
            MemorySegment cSeg = nativeFloatSegment(callArena, c, cOffset, cLength);
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
            copyFloatSegment(cSeg, c, cOffset, cLength);
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS FFM sgemm call failed", t);
        }
    }

    public static void dgemmRowMajorNoTrans(
            int m,
            int n,
            int k,
            double alpha,
            double[] a,
            int lda,
            double[] b,
            int ldb,
            double beta,
            double[] c,
            int ldc
    ) {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.dgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM dgemm is unavailable: " + symbols.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = OpenBlasGemmLayout.requiredElements(m, ldc);
            MemorySegment aSeg = nativeDoubleSegment(callArena, a, 0, OpenBlasGemmLayout.requiredElements(m, lda));
            MemorySegment bSeg = nativeDoubleSegment(callArena, b, 0, OpenBlasGemmLayout.requiredElements(k, ldb));
            MemorySegment cSeg = nativeDoubleSegment(callArena, c, 0, cLength);
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
            copyDoubleSegment(cSeg, c, 0, cLength);
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS FFM dgemm call failed", t);
        }
    }

    public static void dgemmRowMajorNoTransOffsets(
            int m,
            int n,
            int k,
            double alpha,
            double[] a,
            int aOffset,
            int lda,
            double[] b,
            int bOffset,
            int ldb,
            double beta,
            double[] c,
            int cOffset,
            int ldc
    ) {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.dgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM dgemm is unavailable: " + symbols.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = OpenBlasGemmLayout.requiredElements(m, ldc);
            MemorySegment aSeg = nativeDoubleSegment(callArena, a, aOffset, OpenBlasGemmLayout.requiredElements(m, lda));
            MemorySegment bSeg = nativeDoubleSegment(callArena, b, bOffset, OpenBlasGemmLayout.requiredElements(k, ldb));
            MemorySegment cSeg = nativeDoubleSegment(callArena, c, cOffset, cLength);
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
            copyDoubleSegment(cSeg, c, cOffset, cLength);
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS FFM dgemm call failed", t);
        }
    }

    public static void sbgemmRowMajorNoTrans(
            int m,
            int n,
            int k,
            float alpha,
            short[] a,
            int lda,
            short[] b,
            int ldb,
            float beta,
            float[] c,
            int ldc
    ) {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.sbgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sbgemm is unavailable: " + symbols.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = OpenBlasGemmLayout.requiredElements(m, ldc);
            MemorySegment aSeg = nativeShortSegment(callArena, a, 0, OpenBlasGemmLayout.requiredElements(m, lda));
            MemorySegment bSeg = nativeShortSegment(callArena, b, 0, OpenBlasGemmLayout.requiredElements(k, ldb));
            MemorySegment cSeg = nativeFloatSegment(callArena, c, 0, cLength);
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
            copyFloatSegment(cSeg, c, 0, cLength);
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS FFM sbgemm call failed", t);
        }
    }

    public static void sbgemmRowMajorNoTransOffsets(
            int m,
            int n,
            int k,
            float alpha,
            short[] a,
            int aOffset,
            int lda,
            short[] b,
            int bOffset,
            int ldb,
            float beta,
            float[] c,
            int cOffset,
            int ldc
    ) {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.sbgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sbgemm is unavailable: " + symbols.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = OpenBlasGemmLayout.requiredElements(m, ldc);
            MemorySegment aSeg = nativeShortSegment(callArena, a, aOffset, OpenBlasGemmLayout.requiredElements(m, lda));
            MemorySegment bSeg = nativeShortSegment(callArena, b, bOffset, OpenBlasGemmLayout.requiredElements(k, ldb));
            MemorySegment cSeg = nativeFloatSegment(callArena, c, cOffset, cLength);
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
            copyFloatSegment(cSeg, c, cOffset, cLength);
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS FFM sbgemm call failed", t);
        }
    }

    public static void bgemmRowMajorNoTrans(
            int m,
            int n,
            int k,
            short alpha,
            short[] a,
            int lda,
            short[] b,
            int ldb,
            short beta,
            short[] c,
            int ldc
    ) {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.bgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM bgemm is unavailable: " + symbols.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = OpenBlasGemmLayout.requiredElements(m, ldc);
            MemorySegment aSeg = nativeShortSegment(callArena, a, 0, OpenBlasGemmLayout.requiredElements(m, lda));
            MemorySegment bSeg = nativeShortSegment(callArena, b, 0, OpenBlasGemmLayout.requiredElements(k, ldb));
            MemorySegment cSeg = nativeShortSegment(callArena, c, 0, cLength);
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
            copyShortSegment(cSeg, c, 0, cLength);
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS FFM bgemm call failed", t);
        }
    }

    public static void bgemmRowMajorNoTransOffsets(
            int m,
            int n,
            int k,
            short alpha,
            short[] a,
            int aOffset,
            int lda,
            short[] b,
            int bOffset,
            int ldb,
            short beta,
            short[] c,
            int cOffset,
            int ldc
    ) {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.bgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM bgemm is unavailable: " + symbols.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = OpenBlasGemmLayout.requiredElements(m, ldc);
            MemorySegment aSeg = nativeShortSegment(callArena, a, aOffset, OpenBlasGemmLayout.requiredElements(m, lda));
            MemorySegment bSeg = nativeShortSegment(callArena, b, bOffset, OpenBlasGemmLayout.requiredElements(k, ldb));
            MemorySegment cSeg = nativeShortSegment(callArena, c, cOffset, cLength);
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
            copyShortSegment(cSeg, c, cOffset, cLength);
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS FFM bgemm call failed", t);
        }
    }

    private static MemorySegment nativeFloatSegment(Arena arena, float[] src, int offset, int length) {
        return arena.allocateFrom(JAVA_FLOAT, copyRange(src, offset, length));
    }

    private static MemorySegment nativeDoubleSegment(Arena arena, double[] src, int offset, int length) {
        return arena.allocateFrom(JAVA_DOUBLE, copyRange(src, offset, length));
    }

    private static MemorySegment nativeShortSegment(Arena arena, short[] src, int offset, int length) {
        return arena.allocateFrom(JAVA_SHORT, copyRange(src, offset, length));
    }

    private static void copyFloatSegment(MemorySegment src, float[] dst, int offset, int length) {
        System.arraycopy(src.toArray(JAVA_FLOAT), 0, dst, offset, length);
    }

    private static void copyDoubleSegment(MemorySegment src, double[] dst, int offset, int length) {
        System.arraycopy(src.toArray(JAVA_DOUBLE), 0, dst, offset, length);
    }

    private static void copyShortSegment(MemorySegment src, short[] dst, int offset, int length) {
        System.arraycopy(src.toArray(JAVA_SHORT), 0, dst, offset, length);
    }

    private static float[] copyRange(float[] src, int offset, int length) {
        validateRange(src.length, offset, length);
        return Arrays.copyOfRange(src, offset, offset + length);
    }

    private static double[] copyRange(double[] src, int offset, int length) {
        validateRange(src.length, offset, length);
        return Arrays.copyOfRange(src, offset, offset + length);
    }

    private static short[] copyRange(short[] src, int offset, int length) {
        validateRange(src.length, offset, length);
        return Arrays.copyOfRange(src, offset, offset + length);
    }

    private static void validateRange(int arrayLength, int offset, int length) {
        if (offset < 0 || length < 0 || offset > arrayLength || arrayLength - offset < length) {
            throw new IllegalArgumentException("OpenBLAS array slice is outside backing storage.");
        }
    }
}
