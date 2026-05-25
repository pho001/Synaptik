import backend.blas.OpenBlasArrayGemm;
import backend.blas.OpenBlasRuntime;
import backend.blas.OpenBlasSegmentGemm;
import tensor.dtype.TensorDTypeOps;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

public class OpenBlasGemmTest {
    @Test
    void bundledOrConfiguredOpenBlasProvidesRequiredGemmSymbols() {
        Assumptions.assumeTrue(OpenBlasRuntime.isAvailable(), OpenBlasRuntime.unavailableReason());

        double[] a64 = {1.0d, 2.0d, 3.0d, 4.0d};
        double[] b64 = {5.0d, 6.0d, 7.0d, 8.0d};
        double[] c64 = new double[4];
        OpenBlasArrayGemm.dgemmRowMajorNoTrans(2, 2, 2, 1.0d, a64, 2, b64, 2, 0.0d, c64, 2);
        assertArrayEquals(new double[]{19.0d, 22.0d, 43.0d, 50.0d}, c64, 1e-12);

        float[] a32 = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b32 = {5.0f, 6.0f, 7.0f, 8.0f};
        float[] c32 = new float[4];
        OpenBlasArrayGemm.sgemmRowMajorNoTrans(2, 2, 2, 1.0f, a32, 2, b32, 2, 0.0f, c32, 2);
        assertArrayEquals(new float[]{19.0f, 22.0f, 43.0f, 50.0f}, c32, 1e-6f);
    }

    @Test
    void bundledOrConfiguredOpenBlasProvidesBFloat16ToFloatGemmWhenAdvertised() {
        Assumptions.assumeTrue(OpenBlasRuntime.isBFloat16ToFloatGemmAvailable(), OpenBlasRuntime.unavailableReason());

        short[] a = {
                TensorDTypeOps.toBFloat16Bits(1.0f),
                TensorDTypeOps.toBFloat16Bits(2.0f),
                TensorDTypeOps.toBFloat16Bits(3.0f),
                TensorDTypeOps.toBFloat16Bits(4.0f)
        };
        short[] b = {
                TensorDTypeOps.toBFloat16Bits(5.0f),
                TensorDTypeOps.toBFloat16Bits(6.0f),
                TensorDTypeOps.toBFloat16Bits(7.0f),
                TensorDTypeOps.toBFloat16Bits(8.0f)
        };
        float[] c = new float[4];

        OpenBlasArrayGemm.sbgemmRowMajorNoTrans(2, 2, 2, 1.0f, a, 2, b, 2, 0.0f, c, 2);

        assertArrayEquals(new float[]{19.0f, 22.0f, 43.0f, 50.0f}, c, 1e-6f);
    }

    @Test
    void bundledOrConfiguredOpenBlasProvidesBFloat16OutputGemmWhenAdvertised() {
        Assumptions.assumeTrue(OpenBlasRuntime.isBFloat16OutputGemmAvailable(), OpenBlasRuntime.unavailableReason());

        short[] a = {
                TensorDTypeOps.toBFloat16Bits(1.0f),
                TensorDTypeOps.toBFloat16Bits(2.0f),
                TensorDTypeOps.toBFloat16Bits(3.0f),
                TensorDTypeOps.toBFloat16Bits(4.0f)
        };
        short[] b = {
                TensorDTypeOps.toBFloat16Bits(5.0f),
                TensorDTypeOps.toBFloat16Bits(6.0f),
                TensorDTypeOps.toBFloat16Bits(7.0f),
                TensorDTypeOps.toBFloat16Bits(8.0f)
        };
        short[] c = new short[4];

        OpenBlasArrayGemm.bgemmRowMajorNoTrans(
                2,
                2,
                2,
                TensorDTypeOps.toBFloat16Bits(1.0f),
                a,
                2,
                b,
                2,
                TensorDTypeOps.toBFloat16Bits(0.0f),
                c,
                2
        );

        assertArrayEquals(new short[]{
                TensorDTypeOps.toBFloat16Bits(19.0f),
                TensorDTypeOps.toBFloat16Bits(22.0f),
                TensorDTypeOps.toBFloat16Bits(43.0f),
                TensorDTypeOps.toBFloat16Bits(50.0f)
        }, c);
    }

    @Test
    void float32SegmentGemmWritesDirectlyIntoProvidedNativeOutput() {
        Assumptions.assumeTrue(OpenBlasRuntime.isAvailable(), OpenBlasRuntime.unavailableReason());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment a = arena.allocate(JAVA_FLOAT, 4);
            MemorySegment b = arena.allocate(JAVA_FLOAT, 4);
            MemorySegment c = arena.allocate(JAVA_FLOAT, 4);
            fill(a, new float[]{1f, 2f, 3f, 4f});
            fill(b, new float[]{5f, 6f, 7f, 8f});

            OpenBlasSegmentGemm.sgemmRowMajorNoTransSegment(
                    2, 2, 2,
                    1.0f,
                    a, 0L, 2,
                    b, 0L, 2,
                    0.0f,
                    c, 0L, 2
            );

            assertArrayEquals(new float[]{19f, 22f, 43f, 50f}, readFloat(c, 4), 1e-6f);
        }
    }

    @Test
    void float64SegmentGemmHonorsByteOffsets() {
        Assumptions.assumeTrue(OpenBlasRuntime.isAvailable(), OpenBlasRuntime.unavailableReason());

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment a = arena.allocate(JAVA_DOUBLE, 6);
            MemorySegment b = arena.allocate(JAVA_DOUBLE, 6);
            MemorySegment c = arena.allocate(JAVA_DOUBLE, 6);
            fill(a, new double[]{-99d, 1d, 2d, 3d, 4d, -99d});
            fill(b, new double[]{-99d, 5d, 6d, 7d, 8d, -99d});
            fill(c, new double[]{-1d, -1d, -1d, -1d, -1d, -1d});

            OpenBlasSegmentGemm.dgemmRowMajorNoTransSegment(
                    2, 2, 2,
                    1.0d,
                    a, Double.BYTES, 2,
                    b, Double.BYTES, 2,
                    0.0d,
                    c, Double.BYTES, 2
            );

            assertArrayEquals(new double[]{-1d, 19d, 22d, 43d, 50d, -1d}, readDouble(c, 6), 1e-12);
        }
    }

    @Test
    void bfloat16SegmentGemmWritesFloat32OutputWhenSymbolIsAvailable() {
        Assumptions.assumeTrue(OpenBlasRuntime.isBFloat16ToFloatGemmAvailable(), "OpenBLAS SBGEMM is unavailable");

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment a = arena.allocate(JAVA_SHORT, 4);
            MemorySegment b = arena.allocate(JAVA_SHORT, 4);
            MemorySegment c = arena.allocate(JAVA_FLOAT, 4);
            fill(a, new short[]{
                    TensorDTypeOps.toBFloat16Bits(1f),
                    TensorDTypeOps.toBFloat16Bits(2f),
                    TensorDTypeOps.toBFloat16Bits(3f),
                    TensorDTypeOps.toBFloat16Bits(4f)
            });
            fill(b, new short[]{
                    TensorDTypeOps.toBFloat16Bits(5f),
                    TensorDTypeOps.toBFloat16Bits(6f),
                    TensorDTypeOps.toBFloat16Bits(7f),
                    TensorDTypeOps.toBFloat16Bits(8f)
            });

            OpenBlasSegmentGemm.sbgemmRowMajorNoTransSegment(
                    2, 2, 2,
                    1.0f,
                    a, 0L, 2,
                    b, 0L, 2,
                    0.0f,
                    c, 0L, 2
            );

            assertArrayEquals(new float[]{19f, 22f, 43f, 50f}, readFloat(c, 4), 1e-6f);
        }
    }

    @Test
    void bfloat16SegmentGemmWritesBFloat16OutputWhenSymbolIsAvailable() {
        Assumptions.assumeTrue(OpenBlasRuntime.isBFloat16OutputGemmAvailable(), "OpenBLAS BGEMM is unavailable");

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment a = arena.allocate(JAVA_SHORT, 4);
            MemorySegment b = arena.allocate(JAVA_SHORT, 4);
            MemorySegment c = arena.allocate(JAVA_SHORT, 4);
            fill(a, new short[]{
                    TensorDTypeOps.toBFloat16Bits(1f),
                    TensorDTypeOps.toBFloat16Bits(2f),
                    TensorDTypeOps.toBFloat16Bits(3f),
                    TensorDTypeOps.toBFloat16Bits(4f)
            });
            fill(b, new short[]{
                    TensorDTypeOps.toBFloat16Bits(5f),
                    TensorDTypeOps.toBFloat16Bits(6f),
                    TensorDTypeOps.toBFloat16Bits(7f),
                    TensorDTypeOps.toBFloat16Bits(8f)
            });

            OpenBlasSegmentGemm.bgemmRowMajorNoTransSegment(
                    2, 2, 2,
                    TensorDTypeOps.toBFloat16Bits(1f),
                    a, 0L, 2,
                    b, 0L, 2,
                    TensorDTypeOps.toBFloat16Bits(0f),
                    c, 0L, 2
            );

            assertArrayEquals(new short[]{
                    TensorDTypeOps.toBFloat16Bits(19f),
                    TensorDTypeOps.toBFloat16Bits(22f),
                    TensorDTypeOps.toBFloat16Bits(43f),
                    TensorDTypeOps.toBFloat16Bits(50f)
            }, readShort(c, 4));
        }
    }

    @Test
    void legacyOpenBlasBridgeClassIsRemoved() {
        assertFalse(Files.exists(Path.of("src/main/java/backend/blas/OpenBlasFfmBridge.java")));
    }

    @Test
    void segmentGemmMethodsDoNotUseArrayCopyHelpers() throws Exception {
        String source = Files.readString(Path.of("src/main/java/backend/blas/OpenBlasSegmentGemm.java"));
        assertNoCopyHelpers(source);
    }

    private static void assertNoCopyHelpers(String section) {
        assertFalse(section.contains("allocateFrom"));
        assertFalse(section.contains("toArray"));
        assertFalse(section.contains("copyRange"));
        assertFalse(section.contains("nativeFloatSegment"));
        assertFalse(section.contains("copyFloatSegment"));
        assertFalse(section.contains("nativeDoubleSegment"));
        assertFalse(section.contains("copyDoubleSegment"));
        assertFalse(section.contains("nativeShortSegment"));
        assertFalse(section.contains("copyShortSegment"));
    }

    private static void fill(MemorySegment segment, float[] values) {
        for (int i = 0; i < values.length; i++) {
            segment.setAtIndex(JAVA_FLOAT, i, values[i]);
        }
    }

    private static void fill(MemorySegment segment, double[] values) {
        for (int i = 0; i < values.length; i++) {
            segment.setAtIndex(JAVA_DOUBLE, i, values[i]);
        }
    }

    private static void fill(MemorySegment segment, short[] values) {
        for (int i = 0; i < values.length; i++) {
            segment.setAtIndex(JAVA_SHORT, i, values[i]);
        }
    }

    private static float[] readFloat(MemorySegment segment, int length) {
        float[] out = new float[length];
        for (int i = 0; i < length; i++) {
            out[i] = segment.getAtIndex(JAVA_FLOAT, i);
        }
        return out;
    }

    private static double[] readDouble(MemorySegment segment, int length) {
        double[] out = new double[length];
        for (int i = 0; i < length; i++) {
            out[i] = segment.getAtIndex(JAVA_DOUBLE, i);
        }
        return out;
    }

    private static short[] readShort(MemorySegment segment, int length) {
        short[] out = new short[length];
        for (int i = 0; i < length; i++) {
            out[i] = segment.getAtIndex(JAVA_SHORT, i);
        }
        return out;
    }
}
