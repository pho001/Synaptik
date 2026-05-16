package backend.blas;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.openblas.global.openblas;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;
import java.util.Objects;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * OpenBLAS CBLAS bridge backed by the Java Foreign Function and Memory API.
 *
 * <p>The bridge discovers native symbols once at class initialization. Dispatch
 * methods throw when OpenBLAS or the requested GEMM symbol is unavailable, so
 * callers should check {@link #isAvailable()} before routing work here.</p>
 */
public final class OpenBlasFfmBridge {
    /**
     * CBLAS enum value for row-major input buffers.
     */
    public static final int CBLAS_ROW_MAJOR = 101;
    /**
     * CBLAS enum value for non-transposed operands.
     */
    public static final int CBLAS_NO_TRANS = 111;

    private static final State STATE = init();

    private OpenBlasFfmBridge() {}

    /**
     * Returns whether the OpenBLAS CBLAS symbols required by this bridge were found.
     */
    public static boolean isAvailable() {
        return STATE.available;
    }

    /**
     * Returns the native discovery failure reason, or an empty string when available.
     */
    public static String unavailableReason() {
        return STATE.reason;
    }

    /**
     * Returns whether any optional OpenBLAS BF16 GEMM symbol is available and enabled.
     */
    public static boolean isBFloat16GemmAvailable() {
        return isBFloat16ToFloatGemmAvailable() || isBFloat16OutputGemmAvailable();
    }

    /**
     * Returns whether OpenBLAS can multiply BF16 inputs and produce FLOAT32 output via {@code cblas_sbgemm}.
     */
    public static boolean isBFloat16ToFloatGemmAvailable() {
        return STATE.available && STATE.sbgemm != null;
    }

    /**
     * Returns whether OpenBLAS can multiply BF16 inputs and produce BF16 output via {@code cblas_bgemm}.
     */
    public static boolean isBFloat16OutputGemmAvailable() {
        return STATE.available && STATE.bgemm != null;
    }

    public static boolean isFloat32GemmAvailable() {
        return STATE.available && STATE.sgemm != null;
    }

    public static boolean isFloat64GemmAvailable() {
        return STATE.available && STATE.dgemm != null;
    }

    public static String lookupSource() {
        return STATE.source == null ? "UNAVAILABLE" : STATE.source.name();
    }

    public static String threadPolicy() {
        return "AUTO_UNCONTROLLED";
    }

    /**
     * Invokes row-major f32 GEMM with non-transposed operands starting at array offset zero.
     */
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
        if (!STATE.available || STATE.sgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sgemm is unavailable: " + STATE.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = requiredElements(m, ldc);
            MemorySegment aSeg = nativeFloatSegment(callArena, a, 0, requiredElements(m, lda));
            MemorySegment bSeg = nativeFloatSegment(callArena, b, 0, requiredElements(k, ldb));
            MemorySegment cSeg = nativeFloatSegment(callArena, c, 0, cLength);
            STATE.sgemm.invokeExact(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
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

    /**
     * Invokes row-major f32 GEMM with non-transposed operands and explicit array offsets.
     */
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
        if (!STATE.available || STATE.sgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sgemm is unavailable: " + STATE.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = requiredElements(m, ldc);
            MemorySegment aSeg = nativeFloatSegment(callArena, a, aOffset, requiredElements(m, lda));
            MemorySegment bSeg = nativeFloatSegment(callArena, b, bOffset, requiredElements(k, ldb));
            MemorySegment cSeg = nativeFloatSegment(callArena, c, cOffset, cLength);
            STATE.sgemm.invokeExact(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
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

    /**
     * Invokes row-major f32 GEMM directly over native memory segments.
     */
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
        if (!STATE.available || STATE.sgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sgemm is unavailable: " + STATE.reason);
        }
        try {
            MemorySegment aSeg = segmentSlice(a, aByteOffset, requiredElements(m, lda), Float.BYTES, "a");
            MemorySegment bSeg = segmentSlice(b, bByteOffset, requiredElements(k, ldb), Float.BYTES, "b");
            MemorySegment cSeg = segmentSlice(c, cByteOffset, requiredElements(m, ldc), Float.BYTES, "c");
            STATE.sgemm.invokeExact(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
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

    /**
     * Invokes row-major f64 GEMM with non-transposed operands starting at array offset zero.
     */
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
        if (!STATE.available || STATE.dgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM dgemm is unavailable: " + STATE.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = requiredElements(m, ldc);
            MemorySegment aSeg = nativeDoubleSegment(callArena, a, 0, requiredElements(m, lda));
            MemorySegment bSeg = nativeDoubleSegment(callArena, b, 0, requiredElements(k, ldb));
            MemorySegment cSeg = nativeDoubleSegment(callArena, c, 0, cLength);
            STATE.dgemm.invokeExact(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
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

    /**
     * Invokes row-major f64 GEMM with non-transposed operands and explicit array offsets.
     */
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
        if (!STATE.available || STATE.dgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM dgemm is unavailable: " + STATE.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = requiredElements(m, ldc);
            MemorySegment aSeg = nativeDoubleSegment(callArena, a, aOffset, requiredElements(m, lda));
            MemorySegment bSeg = nativeDoubleSegment(callArena, b, bOffset, requiredElements(k, ldb));
            MemorySegment cSeg = nativeDoubleSegment(callArena, c, cOffset, cLength);
            STATE.dgemm.invokeExact(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
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

    /**
     * Invokes row-major f64 GEMM directly over native memory segments.
     */
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
        if (!STATE.available || STATE.dgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM dgemm is unavailable: " + STATE.reason);
        }
        try {
            MemorySegment aSeg = segmentSlice(a, aByteOffset, requiredElements(m, lda), Double.BYTES, "a");
            MemorySegment bSeg = segmentSlice(b, bByteOffset, requiredElements(k, ldb), Double.BYTES, "b");
            MemorySegment cSeg = segmentSlice(c, cByteOffset, requiredElements(m, ldc), Double.BYTES, "c");
            STATE.dgemm.invokeExact(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
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
     * Invokes row-major BF16 GEMM with f32 accumulation starting at array offset zero.
     */
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
        if (!STATE.available || STATE.sbgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sbgemm is unavailable: " + STATE.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = requiredElements(m, ldc);
            MemorySegment aSeg = nativeShortSegment(callArena, a, 0, requiredElements(m, lda));
            MemorySegment bSeg = nativeShortSegment(callArena, b, 0, requiredElements(k, ldb));
            MemorySegment cSeg = nativeFloatSegment(callArena, c, 0, cLength);
            STATE.sbgemm.invokeExact(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
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

    /**
     * Invokes row-major BF16 GEMM with f32 accumulation and explicit array offsets.
     */
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
        if (!STATE.available || STATE.sbgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sbgemm is unavailable: " + STATE.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = requiredElements(m, ldc);
            MemorySegment aSeg = nativeShortSegment(callArena, a, aOffset, requiredElements(m, lda));
            MemorySegment bSeg = nativeShortSegment(callArena, b, bOffset, requiredElements(k, ldb));
            MemorySegment cSeg = nativeFloatSegment(callArena, c, cOffset, cLength);
            STATE.sbgemm.invokeExact(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
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

    /**
     * Invokes row-major BF16 GEMM that writes BF16 output starting at array offset zero.
     */
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
        if (!STATE.available || STATE.bgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM bgemm is unavailable: " + STATE.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = requiredElements(m, ldc);
            MemorySegment aSeg = nativeShortSegment(callArena, a, 0, requiredElements(m, lda));
            MemorySegment bSeg = nativeShortSegment(callArena, b, 0, requiredElements(k, ldb));
            MemorySegment cSeg = nativeShortSegment(callArena, c, 0, cLength);
            STATE.bgemm.invokeExact(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
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

    /**
     * Invokes row-major BF16 GEMM that writes BF16 output and supports explicit array offsets.
     */
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
        if (!STATE.available || STATE.bgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM bgemm is unavailable: " + STATE.reason);
        }
        try (Arena callArena = Arena.ofConfined()) {
            int cLength = requiredElements(m, ldc);
            MemorySegment aSeg = nativeShortSegment(callArena, a, aOffset, requiredElements(m, lda));
            MemorySegment bSeg = nativeShortSegment(callArena, b, bOffset, requiredElements(k, ldb));
            MemorySegment cSeg = nativeShortSegment(callArena, c, cOffset, cLength);
            STATE.bgemm.invokeExact(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
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

    /**
     * Invokes row-major BF16 GEMM directly over native input segments and writes f32 output.
     *
     * <p>This method models the OpenBLAS {@code cblas_sbgemm} contract explicitly: inputs are BF16 raw
     * bits and the output segment stores FLOAT32 accumulation results. It does not produce BF16 output
     * storage and must not be reported as a BF16-output native route.</p>
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
        if (!STATE.available || STATE.sbgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sbgemm is unavailable: " + STATE.reason);
        }
        try {
            MemorySegment aSeg = segmentSlice(aBf16, aByteOffset, requiredElements(m, lda), Short.BYTES, "aBf16");
            MemorySegment bSeg = segmentSlice(bBf16, bByteOffset, requiredElements(k, ldb), Short.BYTES, "bBf16");
            MemorySegment cSeg = segmentSlice(cF32, cByteOffset, requiredElements(m, ldc), Float.BYTES, "cF32");
            STATE.sbgemm.invokeExact(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
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
     * Invokes row-major BF16 GEMM directly over native BF16 segments and writes BF16 output.
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
        if (!STATE.available || STATE.bgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM bgemm is unavailable: " + STATE.reason);
        }
        try {
            MemorySegment aSeg = segmentSlice(aBf16, aByteOffset, requiredElements(m, lda), Short.BYTES, "aBf16");
            MemorySegment bSeg = segmentSlice(bBf16, bByteOffset, requiredElements(k, ldb), Short.BYTES, "bBf16");
            MemorySegment cSeg = segmentSlice(cBf16, cByteOffset, requiredElements(m, ldc), Short.BYTES, "cBf16");
            STATE.bgemm.invokeExact(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
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

    private static int requiredElements(int rows, int leadingDim) {
        return Math.max(0, rows) * Math.max(0, leadingDim);
    }

    private static MemorySegment segmentSlice(
            MemorySegment segment,
            long byteOffset,
            int elements,
            int elementBytes,
            String name
    ) {
        Objects.requireNonNull(segment, name + " segment cannot be null");
        long byteLength = Math.multiplyExact(Math.max(0L, elements), Math.max(1L, elementBytes));
        if (byteOffset < 0L || byteLength < 0L || byteOffset > segment.byteSize() || segment.byteSize() - byteOffset < byteLength) {
            throw new IllegalArgumentException("OpenBLAS segment slice is outside backing storage for " + name
                    + ". byteOffset=" + byteOffset
                    + ", byteLength=" + byteLength
                    + ", segmentBytes=" + segment.byteSize());
        }
        return segment.asSlice(byteOffset, byteLength);
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

    private static State init() {
        try {
            Arena arena = Arena.ofShared();
            LookupResolution lookupResolution = resolveLookup(arena);
            SymbolLookup lookup = lookupResolution.lookup();
            Linker linker = Linker.nativeLinker();

            MethodHandle sgemm = linker.downcallHandle(
                    lookup.find("cblas_sgemm").orElseThrow(),
                    FunctionDescriptor.ofVoid(
                            JAVA_INT, JAVA_INT, JAVA_INT,
                            JAVA_INT, JAVA_INT, JAVA_INT,
                            JAVA_FLOAT,
                            ADDRESS, JAVA_INT,
                            ADDRESS, JAVA_INT,
                            JAVA_FLOAT,
                            ADDRESS, JAVA_INT
                    )
            );

            MethodHandle dgemm = linker.downcallHandle(
                    lookup.find("cblas_dgemm").orElseThrow(),
                    FunctionDescriptor.ofVoid(
                            JAVA_INT, JAVA_INT, JAVA_INT,
                            JAVA_INT, JAVA_INT, JAVA_INT,
                            JAVA_DOUBLE,
                            ADDRESS, JAVA_INT,
                            ADDRESS, JAVA_INT,
                            JAVA_DOUBLE,
                            ADDRESS, JAVA_INT
                    )
            );

            MethodHandle sbgemm = null;
            try {
                MemorySegment sbgemmSym = lookup.find("cblas_sbgemm").orElse(null);
                if (sbgemmSym != null) {
                    sbgemm = linker.downcallHandle(
                            sbgemmSym,
                            FunctionDescriptor.ofVoid(
                                    JAVA_INT, JAVA_INT, JAVA_INT,
                                    JAVA_INT, JAVA_INT, JAVA_INT,
                                    JAVA_FLOAT,
                                    ADDRESS, JAVA_INT,
                                    ADDRESS, JAVA_INT,
                                    JAVA_FLOAT,
                                    ADDRESS, JAVA_INT
                            )
                    );
                }
            } catch (Throwable ignored) {
            }

            MethodHandle bgemm = null;
            try {
                MemorySegment bgemmSym = lookup.find("cblas_bgemm").orElse(null);
                if (bgemmSym != null) {
                    bgemm = linker.downcallHandle(
                            bgemmSym,
                            FunctionDescriptor.ofVoid(
                                    JAVA_INT, JAVA_INT, JAVA_INT,
                                    JAVA_INT, JAVA_INT, JAVA_INT,
                                    JAVA_SHORT,
                                    ADDRESS, JAVA_INT,
                                    ADDRESS, JAVA_INT,
                                    JAVA_SHORT,
                                    ADDRESS, JAVA_INT
                            )
                    );
                }
            } catch (Throwable ignored) {
            }

            return new State(true, null, lookupResolution.source(), arena, sgemm, dgemm, sbgemm, bgemm);
        } catch (Throwable t) {
            return new State(false, t.getClass().getSimpleName() + ": " + safeMessage(t), null, null, null, null, null, null);
        }
    }

    private static LookupResolution resolveLookup(Arena arena) {
        String explicit = System.getProperty("openblas.lib");
        if (explicit != null && !explicit.isBlank()) {
            return new LookupResolution(SymbolLookup.libraryLookup(explicit.trim(), arena), LookupSource.EXPLICIT_PROPERTY);
        }
        String envLib = System.getenv("OPENBLAS_LIB");
        if (envLib != null && !envLib.isBlank()) {
            return new LookupResolution(SymbolLookup.libraryLookup(envLib.trim(), arena), LookupSource.ENVIRONMENT);
        }
        try {
            return new LookupResolution(SymbolLookup.libraryLookup(Loader.load(openblas.class), arena), LookupSource.BUNDLED_JAVACPP);
        } catch (Throwable bundledFailure) {
            try {
                return new LookupResolution(SymbolLookup.libraryLookup("openblas", arena), LookupSource.SYSTEM_LIBRARY);
            } catch (Throwable systemFailure) {
                IllegalStateException combined = new IllegalStateException(
                        "OpenBLAS lookup failed for bundled JavaCPP preset and system library. Bundled: "
                                + bundledFailure.getClass().getSimpleName() + ": " + safeMessage(bundledFailure)
                                + "; system: " + systemFailure.getClass().getSimpleName() + ": " + safeMessage(systemFailure),
                        systemFailure
                );
                combined.addSuppressed(bundledFailure);
                throw combined;
            }
        }
    }

    private static String safeMessage(Throwable t) {
        String m = t.getMessage();
        return m == null ? "<no-message>" : m;
    }

    private enum LookupSource {
        EXPLICIT_PROPERTY,
        ENVIRONMENT,
        BUNDLED_JAVACPP,
        SYSTEM_LIBRARY
    }

    private record LookupResolution(SymbolLookup lookup, LookupSource source) {
    }

    private static final class State {
        private final boolean available;
        private final String reason;
        private final LookupSource source;
        @SuppressWarnings("unused")
        private final Arena arenaRef;
        private final MethodHandle sgemm;
        private final MethodHandle dgemm;
        private final MethodHandle sbgemm;
        private final MethodHandle bgemm;

        private State(boolean available, String reason, LookupSource source, Arena arenaRef, MethodHandle sgemm, MethodHandle dgemm, MethodHandle sbgemm, MethodHandle bgemm) {
            this.available = available;
            this.reason = reason;
            this.source = source;
            this.arenaRef = arenaRef;
            this.sgemm = sgemm;
            this.dgemm = dgemm;
            this.sbgemm = sbgemm;
            this.bgemm = bgemm;
        }
    }
}
