package debug;

import backend.blas.OpenBlasFfmBridge;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Locale;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

final class NativeOpenBlasSegmentGemmBenchmarkTest {
    private static final Shape[] SHAPES = {
            new Shape(256, 512, 1024),
            new Shape(256, 1024, 1024),
            new Shape(256, 1024, 512),
            new Shape(256, 512, 128)
    };

    @Test
    void benchmarkJavaArrayOpenBlasArrayCopyAndNativeSegmentRoutes() {
        assumeTrue(
                benchmarkEnabled(),
                "Set -Dsynaptik.benchmark.nativeOpenBlasSegment=true or SYNAPTIK_BENCHMARK_NATIVE_OPENBLAS_SEGMENT=true to run diagnostic benchmark."
        );
        assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());
        System.out.println("NATIVE_OPENBLAS_SEGMENT_GEMM_BENCHMARK");
        System.out.println("lookupSource=" + OpenBlasFfmBridge.lookupSource()
                + " threadPolicy=" + OpenBlasFfmBridge.threadPolicy());
        for (Shape shape : SHAPES) {
            Result javaDirect = measureJava(shape);
            Result arrayCopy = measureArrayCopyOpenBlas(shape);
            Result nativeSegment = measureNativeSegmentOpenBlas(shape);
            System.out.println(shape + " "
                    + javaDirect
                    + " "
                    + arrayCopy
                    + " "
                    + nativeSegment);
        }
    }

    private static Result measureJava(Shape shape) {
        float[] a = data(shape.m * shape.k);
        float[] b = data(shape.k * shape.n);
        float[] c = new float[shape.m * shape.n];
        long best = Long.MAX_VALUE;
        for (int i = 0; i < 4; i++) {
            java.util.Arrays.fill(c, 0f);
            long t0 = System.nanoTime();
            javaMatmul(a, b, c, shape);
            long dt = System.nanoTime() - t0;
            if (i > 0) {
                best = Math.min(best, dt);
            }
        }
        return new Result("java-nonblas", best, 0L, 0L);
    }

    private static Result measureArrayCopyOpenBlas(Shape shape) {
        float[] a = data(shape.m * shape.k);
        float[] b = data(shape.k * shape.n);
        float[] c = new float[shape.m * shape.n];
        long best = Long.MAX_VALUE;
        for (int i = 0; i < 4; i++) {
            long t0 = System.nanoTime();
            OpenBlasFfmBridge.sgemmRowMajorNoTrans(
                    shape.m, shape.n, shape.k,
                    1f,
                    a, shape.k,
                    b, shape.n,
                    0f,
                    c, shape.n
            );
            long dt = System.nanoTime() - t0;
            if (i > 0) {
                best = Math.min(best, dt);
            }
        }
        long copyIn = ((long) a.length + b.length) * Float.BYTES;
        long copyOut = (long) c.length * Float.BYTES;
        return new Result("openblas-array-copy", best, copyIn, copyOut);
    }

    private static Result measureNativeSegmentOpenBlas(Shape shape) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment a = arena.allocate(JAVA_FLOAT, shape.m * shape.k);
            MemorySegment b = arena.allocate(JAVA_FLOAT, shape.k * shape.n);
            MemorySegment c = arena.allocate(JAVA_FLOAT, shape.m * shape.n);
            fill(a, shape.m * shape.k);
            fill(b, shape.k * shape.n);
            long best = Long.MAX_VALUE;
            for (int i = 0; i < 4; i++) {
                long t0 = System.nanoTime();
                OpenBlasFfmBridge.sgemmRowMajorNoTransSegment(
                        shape.m, shape.n, shape.k,
                        1f,
                        a, 0L, shape.k,
                        b, 0L, shape.n,
                        0f,
                        c, 0L, shape.n
                );
                long dt = System.nanoTime() - t0;
                if (i > 0) {
                    best = Math.min(best, dt);
                }
            }
            return new Result("openblas-native-segment", best, 0L, 0L);
        }
    }

    private static void javaMatmul(float[] a, float[] b, float[] c, Shape shape) {
        for (int row = 0; row < shape.m; row++) {
            int aBase = row * shape.k;
            int cBase = row * shape.n;
            for (int kk = 0; kk < shape.k; kk++) {
                float av = a[aBase + kk];
                int bBase = kk * shape.n;
                for (int col = 0; col < shape.n; col++) {
                    c[cBase + col] += av * b[bBase + col];
                }
            }
        }
    }

    private static float[] data(int size) {
        float[] out = new float[size];
        for (int i = 0; i < size; i++) {
            out[i] = (float) Math.sin(i * 0.013);
        }
        return out;
    }

    private static void fill(MemorySegment segment, int size) {
        for (int i = 0; i < size; i++) {
            segment.setAtIndex(JAVA_FLOAT, i, (float) Math.sin(i * 0.013));
        }
    }

    private static boolean benchmarkEnabled() {
        return Boolean.getBoolean("synaptik.benchmark.nativeOpenBlasSegment")
                || "true".equalsIgnoreCase(System.getenv("SYNAPTIK_BENCHMARK_NATIVE_OPENBLAS_SEGMENT"));
    }

    private record Shape(int m, int k, int n) {
        @Override
        public String toString() {
            return "[" + m + "," + k + "]x[" + k + "," + n + "]";
        }
    }

    private record Result(String route, long bestNs, long copyInBytes, long copyOutBytes) {
        @Override
        public String toString() {
            return route + "{bestMs=" + String.format(Locale.ROOT, "%.3f", bestNs / 1_000_000.0)
                    + ",copyInBytes=" + copyInBytes
                    + ",copyOutBytes=" + copyOutBytes
                    + "}";
        }
    }
}
