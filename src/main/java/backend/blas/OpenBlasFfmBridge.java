package backend.blas;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class OpenBlasFfmBridge {
    public static final int CBLAS_ROW_MAJOR = 101;
    public static final int CBLAS_NO_TRANS = 111;

    private static final State STATE = init();

    private OpenBlasFfmBridge() {}

    public static boolean isAvailable() {
        return STATE.available;
    }

    public static String unavailableReason() {
        return STATE.reason;
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
        if (!STATE.available || STATE.sgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sgemm is unavailable: " + STATE.reason);
        }
        try {
            MemorySegment aSeg = heapFloatSegment(a, 0, requiredElements(m, lda));
            MemorySegment bSeg = heapFloatSegment(b, 0, requiredElements(k, ldb));
            MemorySegment cSeg = heapFloatSegment(c, 0, requiredElements(m, ldc));
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
        if (!STATE.available || STATE.sgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sgemm is unavailable: " + STATE.reason);
        }
        try {
            MemorySegment aSeg = heapFloatSegment(a, aOffset, requiredElements(m, lda));
            MemorySegment bSeg = heapFloatSegment(b, bOffset, requiredElements(k, ldb));
            MemorySegment cSeg = heapFloatSegment(c, cOffset, requiredElements(m, ldc));
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
        if (!STATE.available || STATE.dgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM dgemm is unavailable: " + STATE.reason);
        }
        try {
            MemorySegment aSeg = heapDoubleSegment(a, 0, requiredElements(m, lda));
            MemorySegment bSeg = heapDoubleSegment(b, 0, requiredElements(k, ldb));
            MemorySegment cSeg = heapDoubleSegment(c, 0, requiredElements(m, ldc));
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
        if (!STATE.available || STATE.dgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM dgemm is unavailable: " + STATE.reason);
        }
        try {
            MemorySegment aSeg = heapDoubleSegment(a, aOffset, requiredElements(m, lda));
            MemorySegment bSeg = heapDoubleSegment(b, bOffset, requiredElements(k, ldb));
            MemorySegment cSeg = heapDoubleSegment(c, cOffset, requiredElements(m, ldc));
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
        if (!STATE.available || STATE.sbgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sbgemm is unavailable: " + STATE.reason);
        }
        try {
            MemorySegment aSeg = heapShortSegment(a, 0, requiredElements(m, lda));
            MemorySegment bSeg = heapShortSegment(b, 0, requiredElements(k, ldb));
            MemorySegment cSeg = heapFloatSegment(c, 0, requiredElements(m, ldc));
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
        if (!STATE.available || STATE.sbgemm == null) {
            throw new IllegalStateException("OpenBLAS FFM sbgemm is unavailable: " + STATE.reason);
        }
        try {
            MemorySegment aSeg = heapShortSegment(a, aOffset, requiredElements(m, lda));
            MemorySegment bSeg = heapShortSegment(b, bOffset, requiredElements(k, ldb));
            MemorySegment cSeg = heapFloatSegment(c, cOffset, requiredElements(m, ldc));
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
            throw new IllegalStateException("OpenBLAS FFM sbgemm call failed", t);
        }
    }

    private static int requiredElements(int rows, int leadingDim) {
        return Math.max(0, rows) * Math.max(0, leadingDim);
    }

    private static MemorySegment heapFloatSegment(float[] src, int offset, int length) {
        MemorySegment heap = MemorySegment.ofArray(src);
        long byteOffset = (long) offset * JAVA_FLOAT.byteSize();
        long byteLength = (long) length * JAVA_FLOAT.byteSize();
        return heap.asSlice(byteOffset, byteLength);
    }

    private static MemorySegment heapDoubleSegment(double[] src, int offset, int length) {
        MemorySegment heap = MemorySegment.ofArray(src);
        long byteOffset = (long) offset * JAVA_DOUBLE.byteSize();
        long byteLength = (long) length * JAVA_DOUBLE.byteSize();
        return heap.asSlice(byteOffset, byteLength);
    }

    private static MemorySegment heapShortSegment(short[] src, int offset, int length) {
        MemorySegment heap = MemorySegment.ofArray(src);
        long byteOffset = (long) offset * JAVA_SHORT.byteSize();
        long byteLength = (long) length * JAVA_SHORT.byteSize();
        return heap.asSlice(byteOffset, byteLength);
    }

    private static State init() {
        try {
            Arena arena = Arena.ofShared();
            SymbolLookup lookup = resolveLookup(arena);
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

            return new State(true, null, arena, sgemm, dgemm, sbgemm);
        } catch (Throwable t) {
            return new State(false, t.getClass().getSimpleName() + ": " + safeMessage(t), null, null, null, null);
        }
    }

    private static SymbolLookup resolveLookup(Arena arena) {
        String explicit = System.getProperty("openblas.lib");
        if (explicit != null && !explicit.isBlank()) {
            return SymbolLookup.libraryLookup(explicit.trim(), arena);
        }
        String envLib = System.getenv("OPENBLAS_LIB");
        if (envLib != null && !envLib.isBlank()) {
            return SymbolLookup.libraryLookup(envLib.trim(), arena);
        }
        return SymbolLookup.libraryLookup("openblas", arena);
    }

    private static String safeMessage(Throwable t) {
        String m = t.getMessage();
        return m == null ? "<no-message>" : m;
    }

    private static final class State {
        private final boolean available;
        private final String reason;
        @SuppressWarnings("unused")
        private final Arena arenaRef;
        private final MethodHandle sgemm;
        private final MethodHandle dgemm;
        private final MethodHandle sbgemm;

        private State(boolean available, String reason, Arena arenaRef, MethodHandle sgemm, MethodHandle dgemm, MethodHandle sbgemm) {
            this.available = available;
            this.reason = reason;
            this.arenaRef = arenaRef;
            this.sgemm = sgemm;
            this.dgemm = dgemm;
            this.sbgemm = sbgemm;
        }
    }
}
