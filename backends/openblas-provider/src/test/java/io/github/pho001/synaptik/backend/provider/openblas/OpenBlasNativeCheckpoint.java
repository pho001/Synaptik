package io.github.pho001.synaptik.backend.provider.openblas;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

/** Runs the explicit real-native OpenBLAS provider capability checkpoint. */
public final class OpenBlasNativeCheckpoint {
    private OpenBlasNativeCheckpoint() {
    }

    /**
     * Validates one caller-supplied compatible absolute library and restores its original count.
     *
     * @param args exactly one absolute OpenBLAS shared-library path
     * @throws Throwable if argument validation, loading, invocation, numerical validation,
     *                   restoration, verification, or cleanup fails
     */
    public static void main(String[] args) throws Throwable {
        if (args.length != 1) {
            throw new IllegalArgumentException("expected exactly one absolute OpenBLAS library path argument");
        }
        Path path = Path.of(args[0]);
        if (!path.isAbsolute()) {
            throw new IllegalArgumentException("OpenBLAS library path must be absolute: " + path);
        }

        try (OpenBlasLibrary first = OpenBlasLibrary.open(path);
                OpenBlasLibrary second = OpenBlasLibrary.open(path)) {
            int original = first.threadCount();
            Throwable primary = null;
            try {
                first.setThreadCount(1);
                requireCount("first owner after set", 1, first.threadCount());
                requireCount("second owner after set", 1, second.threadCount());
                checkSgemm(first);
                checkDgemm(first);
            } catch (Throwable failure) {
                primary = failure;
                throw failure;
            } finally {
                try {
                    first.setThreadCount(original);
                    requireCount("first owner after restore", original, first.threadCount());
                    requireCount("second owner after restore", original, second.threadCount());
                } catch (Throwable restorationFailure) {
                    if (primary == null) {
                        throw restorationFailure;
                    }
                    suppressDistinct(primary, restorationFailure);
                }
            }
            System.out.println("OpenBLAS native checkpoint passed; restored thread count " + original);
        }
    }

    private static void checkSgemm(OpenBlasLibrary library) {
        float[] expected = {31.0f, 36.0f, 75.5f, 85.0f};
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment a = arena.allocateFrom(JAVA_FLOAT, 1, 2, 3, 4, 5, 6);
            MemorySegment b = arena.allocateFrom(JAVA_FLOAT, 7, 8, 9, 10, 11, 12);
            MemorySegment c = arena.allocateFrom(JAVA_FLOAT, 1, 2, 3, 4);
            library.sgemm(2, 2, 3, 0.5f, a, b, 2.0f, c);
            for (int index = 0; index < expected.length; index++) {
                float actual = c.getAtIndex(JAVA_FLOAT, index);
                float tolerance = 1.0e-4f + 1.0e-5f * Math.abs(expected[index]);
                if (!(Math.abs(actual - expected[index]) <= tolerance)) {
                    throw new AssertionError("SGEMM output[" + index + "] expected "
                            + expected[index] + " but was " + actual);
                }
            }
        }
    }

    private static void checkDgemm(OpenBlasLibrary library) {
        double[] expected = {31.0, 36.0, 75.5, 85.0};
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment a = arena.allocateFrom(JAVA_DOUBLE, 1, 2, 3, 4, 5, 6);
            MemorySegment b = arena.allocateFrom(JAVA_DOUBLE, 7, 8, 9, 10, 11, 12);
            MemorySegment c = arena.allocateFrom(JAVA_DOUBLE, 1, 2, 3, 4);
            library.dgemm(2, 2, 3, 0.5, a, b, 2.0, c);
            for (int index = 0; index < expected.length; index++) {
                double actual = c.getAtIndex(JAVA_DOUBLE, index);
                double tolerance = 1.0e-12 + 1.0e-12 * Math.abs(expected[index]);
                if (!(Math.abs(actual - expected[index]) <= tolerance)) {
                    throw new AssertionError("DGEMM output[" + index + "] expected "
                            + expected[index] + " but was " + actual);
                }
            }
        }
    }

    private static void requireCount(String observation, int expected, int actual) {
        if (actual != expected) {
            throw new AssertionError(observation + " expected " + expected + " but was " + actual);
        }
    }

    private static void suppressDistinct(Throwable primary, Throwable secondary) {
        if (secondary != primary) {
            primary.addSuppressed(secondary);
        }
    }
}
